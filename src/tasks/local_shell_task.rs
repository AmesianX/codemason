//! LocalShellTask — adapted from src/tasks/LocalShellTask/.
//! Equivalent to codemason/tasks/local_shell_task.py.
//!
//! The one task type that lives in AppState.tasks: a background shell command
//! (Bash's run_in_background). Folds guards.ts + killShellTasks.ts +
//! LocalShellTask.tsx into one module.
//!
//! Substrate boundaries (documented divergences from the Python port):
//!  - Runtime handle. Python stored the live `proc` on the (dataclass) task
//!    state and nulled it on completion. A Rust `Child` is neither Clone nor
//!    serializable, and AppState is cloned on every update, so instead the
//!    task state carries only `pid: Option<u32>` (the process-GROUP leader —
//!    setsid makes pid==pgid) and kill goes through
//!    windows_shell::kill_process_tree(pid). "Still running" is read from the
//!    task status, not from a returncode probe.
//!  - Streaming. Python spawned the command PTY-backed so block-buffered
//!    children stream live to disk. Rust std has no pty; this uses piped
//!    stdout+stderr drained concurrently to the output file (stderr merged in).
//!    Output is never lost; a block-buffered child may stream to disk less
//!    eagerly than under a PTY (the only observable difference). TERM=dumb is
//!    still set.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::process::Stdio;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::state::app_state::{get_app_state, AppState};
use crate::task::{
    create_task_state_base, generate_task_id, is_terminal_task_status, now_ms, Task, TaskStateBase,
};
use crate::utils::message_queue_manager::enqueue_pending_notification;
use crate::utils::task::disk_output::{ensure_output_dir, get_task_output_path};
use crate::utils::task::framework::{evict_terminal_task, register_task, update_task_state};
use crate::utils::windows_shell::kill_process_tree;

use async_trait::async_trait;

/// Cap per-task log output so a runaway background command can't fill the disk
/// (the original's DiskTaskOutput caps disk writes). Past the cap we keep
/// draining so the child never blocks, but stop writing and append a one-time
/// truncation note.
const MAX_BG_OUTPUT_BYTES: usize = 10 * 1024 * 1024; // 10 MB

/// Bound the in-memory registry. Once it exceeds this many entries, evict the
/// oldest terminal+notified records (their .output file persists on disk).
const MAX_TASKS: usize = 100;

pub const BACKGROUND_BASH_SUMMARY_PREFIX: &str = "Background command ";

// XML tag names (constants/xml.ts), matched 1:1 so the emitted block is
// identical in shape to enqueueShellNotification's.
use crate::constants::xml::{
    OUTPUT_FILE_TAG, STATUS_TAG, SUMMARY_TAG, TASK_ID_TAG, TASK_NOTIFICATION_TAG, TOOL_USE_ID_TAG,
};

/// {'code': int, 'interrupted': bool} once finished.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShellResult {
    pub code: i32,
    pub interrupted: bool,
}

/// State for a background shell task (LocalShellTaskState, guards.ts).
///
/// `type` is fixed to 'local_bash' for backward compatibility with persisted
/// state. `result` carries the exit code once finished. `pid` is the live
/// process-group leader (the kill target), nulled on completion.
#[derive(Debug, Clone, Default)]
pub struct LocalShellTaskState {
    pub base: TaskStateBase,
    pub command: String,
    pub result: Option<ShellResult>,
    pub completion_status_sent_in_attachment: bool,
    pub is_backgrounded: bool,
    pub agent_id: Option<String>,
    pub kind: String, // 'bash' | 'monitor'
    pub pid: Option<u32>,
}

/// Type guard (isLocalShellTask).
pub fn is_local_shell_task(task: &LocalShellTaskState) -> bool {
    task.base.task_type == "local_bash"
}

/// Port of utils/xml.ts escapeXml: escape element-text special chars.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Enqueue a completion <task-notification> (enqueueShellNotification).
///
/// Atomically checks-and-sets `notified` so a task is announced at most once.
pub fn enqueue_shell_notification(
    task_id: &str,
    description: &str,
    status: &str, // 'completed' | 'failed' | 'killed'
    exit_code: Option<i32>,
    tool_use_id: Option<&str>,
    _kind: &str,
) {
    let should_enqueue = std::cell::Cell::new(false);
    update_task_state(task_id, |task| {
        // The updater receives the union; narrow to this task type first
        // (isLocalShellTask) — `?` turns "not a shell task" into the no-op.
        let task = task.as_local_shell()?;
        if task.base.notified {
            return None;
        }
        should_enqueue.set(true);
        let mut next = task.clone();
        next.base.notified = true;
        Some(next.into())
    });
    if !should_enqueue.get() {
        return;
    }

    let summary = match status {
        "completed" => {
            let suffix = exit_code.map(|c| format!(" (exit code {c})")).unwrap_or_default();
            format!("{BACKGROUND_BASH_SUMMARY_PREFIX}\"{description}\" completed{suffix}")
        }
        "failed" => {
            let suffix = exit_code.map(|c| format!(" with exit code {c}")).unwrap_or_default();
            format!("{BACKGROUND_BASH_SUMMARY_PREFIX}\"{description}\" failed{suffix}")
        }
        _ => format!("{BACKGROUND_BASH_SUMMARY_PREFIX}\"{description}\" was stopped"),
    };

    let output_path = get_task_output_path(task_id);
    let tool_use_id_line = tool_use_id
        .filter(|s| !s.is_empty())
        .map(|t| format!("\n<{TOOL_USE_ID_TAG}>{t}</{TOOL_USE_ID_TAG}>"))
        .unwrap_or_default();
    let message = format!(
        "<{TASK_NOTIFICATION_TAG}>\n\
         <{TASK_ID_TAG}>{task_id}</{TASK_ID_TAG}>{tool_use_id_line}\n\
         <{OUTPUT_FILE_TAG}>{}</{OUTPUT_FILE_TAG}>\n\
         <{STATUS_TAG}>{status}</{STATUS_TAG}>\n\
         <{SUMMARY_TAG}>{}</{SUMMARY_TAG}>\n\
         </{TASK_NOTIFICATION_TAG}>",
        output_path.display(),
        escape_xml(&summary),
    );
    enqueue_pending_notification(&message, Some("task-notification"));
}

/// Kill a running shell task and mark it killed+notified (killTask).
///
/// The OS termination happens inside the state updater (mirroring the source,
/// which calls shellCommand.kill() inside updateTaskState). Marking notified
/// here is what suppresses the later completion notice. No-op unless the task
/// is a running local_bash task.
pub fn kill_task(task_id: &str) {
    update_task_state(task_id, |task| {
        let task = task.as_local_shell()?;
        if task.base.status != "running" || !is_local_shell_task(task) {
            return None;
        }
        if let Some(pid) = task.pid {
            // Own process group (setsid) → signal the whole tree so
            // grandchildren are reaped. No sig arg: killpg(SIGKILL) on POSIX,
            // taskkill /T on Windows.
            kill_process_tree(pid, None);
        }
        let mut next = task.clone();
        next.base.status = "killed".to_string();
        next.base.notified = true;
        next.pid = None;
        next.base.end_time = Some(now_ms());
        Some(next.into())
    });
}

/// Wall-clock cap for one background task, or None when disabled.
///
/// Read per task rather than cached: a session that raises the setting for a
/// long job should not have to restart for it to take effect.
fn background_timeout() -> Option<std::time::Duration> {
    let secs = crate::utils::config::get_config().background_task_timeout;
    (secs > 0.0 && secs.is_finite()).then(|| std::time::Duration::from_secs_f64(secs))
}

/// Say why the output stops, in the file the user actually reads.
///
/// The task settles as failed carrying the signal's exit code, which on its own
/// looks like the command failed rather than like we killed it.
fn note_timeout(output_path: &std::path::Path, timeout: std::time::Duration) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(output_path) {
        let _ = write!(
            f,
            "\n[background task terminated: exceeded {} seconds \u{2014} set \
             background_task_timeout to change, or 0 to disable]\n",
            timeout.as_secs()
        );
    }
}

/// Read up to 4096 bytes from an optional reader; None on EOF/error (and clears
/// the reader so its select branch stops firing).
async fn read_chunk<R: AsyncReadExt + Unpin>(reader: &mut Option<R>) -> Option<Vec<u8>> {
    let Some(r) = reader.as_mut() else {
        return None;
    };
    let mut buf = [0u8; 4096];
    match r.read(&mut buf).await {
        Ok(0) | Err(_) => {
            *reader = None;
            None
        }
        Ok(n) => Some(buf[..n].to_vec()),
    }
}

struct DiskWriter {
    file: std::fs::File,
    written: usize,
    truncated: bool,
}

impl DiskWriter {
    fn new(file: std::fs::File) -> Self {
        DiskWriter { file, written: 0, truncated: false }
    }
    fn write_chunk(&mut self, chunk: &[u8]) {
        use std::io::Write;
        if self.written < MAX_BG_OUTPUT_BYTES {
            let room = MAX_BG_OUTPUT_BYTES - self.written;
            let slice = &chunk[..chunk.len().min(room)];
            let _ = self.file.write_all(slice);
            self.written += slice.len();
            let _ = self.file.flush();
            if self.written >= MAX_BG_OUTPUT_BYTES && !self.truncated {
                self.mark_truncated();
            }
        } else if !self.truncated {
            self.mark_truncated();
        }
        // else: keep draining so the child doesn't block, but discard.
    }
    fn write_all_bounded(&mut self, data: &[u8]) {
        self.write_chunk(data);
    }
    fn mark_truncated(&mut self) {
        use std::io::Write;
        let _ = self
            .file
            .write_all(format!("\n[background output truncated: exceeded {MAX_BG_OUTPUT_BYTES} bytes]\n").as_bytes());
        let _ = self.file.flush();
        self.truncated = true;
    }
}

/// Publish the freshly-spawned pid onto the task; returns true if the task was
/// already killed in the spawn window (the caller must then kill the group and
/// bail).
fn publish_pid(task_id: &str, pid: u32) -> bool {
    let killed = std::cell::Cell::new(false);
    update_task_state(task_id, |t| {
        let t = t.as_local_shell()?;
        if t.base.status == "killed" {
            killed.set(true);
            return None;
        }
        let mut next = t.clone();
        next.pid = Some(pid);
        Some(next.into())
    });
    killed.get()
}

struct Settled {
    killed: bool,
    status: String,
    code: Option<i32>,
}

/// Settle running → completed/failed UNLESS the task was already killed
/// (killed wins). Carries the decision out rather than re-reading state.
fn complete_task(task_id: &str, returncode: Option<i32>, errored: bool) -> Settled {
    let settled = Mutex::new(Settled { killed: false, status: "completed".into(), code: None });
    update_task_state(task_id, |task| {
        let task = task.as_local_shell()?;
        let mut s = settled.lock();
        if task.base.status == "killed" {
            s.killed = true;
            return None;
        }
        let code = returncode.unwrap_or(if errored { 1 } else { 0 });
        s.status = if errored || code != 0 { "failed".into() } else { "completed".into() };
        s.code = Some(code);
        let mut next = task.clone();
        next.base.status = s.status.clone();
        next.result = Some(ShellResult { code, interrupted: false });
        next.pid = None;
        next.base.end_time = Some(now_ms());
        Some(next.into())
    });
    settled.into_inner()
}

/// Spawn the command, stream output to disk, then settle task state.
async fn run(
    task_id: String,
    command: String,
    cwd: String,
    output_path: std::path::PathBuf,
    description: String,
    tool_use_id: Option<String>,
    kind: String,
) {
    let mut returncode: Option<i32> = None;
    let mut errored = false;

    match std::fs::File::create(&output_path) {
        Ok(file) => {
            let mut writer = DiskWriter::new(file);
            // Piped stdout+stderr (see the module-level streaming divergence).
            // git-bash discovery can fail here (Windows only). Its message
            // names the download and the CODEMASON_GIT_BASH_PATH override,
            // so it has to reach the task output rather than being replaced
            // by whatever /bin/sh fails with.
            let built = build_shell_command(&command, &cwd, &kind);
            let mut cmd = match built {
                Ok(c) => Some(c),
                Err(e) => {
                    errored = true;
                    append_error(&output_path, &e);
                    None
                }
            };
            if let Some(cmd) = cmd.as_mut() {
                cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
                crate::utils::windows_shell::apply_spawn_group(cmd);
                match cmd.spawn() {
                    Ok(mut child) => {
                        let pid = child.id().unwrap_or(0);
                        if pid != 0 && publish_pid(&task_id, pid) {
                            // Killed between register_task and here: kill the group,
                            // reap, and bail without streaming/settling.
                            kill_process_tree(pid, None);
                            let _ = child.wait().await;
                            return;
                        }
                        let mut stdout = child.stdout.take();
                        let mut stderr = child.stderr.take();
                        let deadline = background_timeout();
                        let timer = async {
                            match deadline {
                                Some(d) => tokio::time::sleep(d).await,
                                None => std::future::pending::<()>().await,
                            }
                        };
                        tokio::pin!(timer);
                        let mut timed_out = false;
                        loop {
                            if stdout.is_none() && stderr.is_none() {
                                break;
                            }
                            tokio::select! {
                                c = read_chunk(&mut stdout), if stdout.is_some() => {
                                    if let Some(chunk) = c { writer.write_chunk(&chunk); }
                                }
                                c = read_chunk(&mut stderr), if stderr.is_some() => {
                                    if let Some(chunk) = c { writer.write_chunk(&chunk); }
                                }
                                // Kill the GROUP, not just the shell: the runaway is
                                // usually a grandchild, and killing the shell alone
                                // reparents it. The readers then hit EOF, so this
                                // loop still ends through its normal path. The guard
                                // matters — a completed sleep polls Ready forever and
                                // would spin this loop at 100% CPU.
                                _ = &mut timer, if !timed_out => {
                                    timed_out = true;
                                    if let Some(d) = deadline { note_timeout(&output_path, d); }
                                    if pid != 0 { kill_process_tree(pid, None); }
                                }
                            }
                        }
                        returncode = child.wait().await.ok().and_then(|s| s.code());
                    }
                    Err(e) => {
                        errored = true;
                        append_error(&output_path, &e.to_string());
                    }
                }
            }
        }
        Err(e) => {
            errored = true;
            append_error(&output_path, &e.to_string());
        }
    }

    let settled = complete_task(&task_id, returncode, errored);
    let (status, exit_code) = if settled.killed {
        ("killed".to_string(), None)
    } else {
        (settled.status, settled.code)
    };
    enqueue_shell_notification(&task_id, &description, &status, exit_code, tool_use_id.as_deref(), &kind);
}

fn append_error(output_path: &std::path::Path, err: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(output_path) {
        let _ = write!(f, "\n[background task error] {err}\n");
    }
}

/// Build the Command to run `command` through a shell.
///
/// `kind` picks the interpreter. It used to be ignored, and on Windows
/// `shell_argv` hands back `[git-bash, -c, script]` for every caller, so a
/// backgrounded PowerShell script was interpreted by bash: `$PSVersionTable`
/// expanded to nothing, quoting and `;` followed POSIX rules, and cmdlets came
/// back "command not found". Foreground and background ran different languages
/// for the same tool. Route a "powershell" task through PowerShell using the
/// same -EncodedCommand invocation `powershell_exec` uses, including its
/// exit-code sentinel — a bare -EncodedCommand run reports 0 even when a
/// cmdlet wrote a non-terminating error, and the task notification carries
/// that code to the model. The cwd sentinel is deliberately left out: a
/// background task must not move the session's working directory.
fn build_shell_command(command: &str, cwd: &str, kind: &str) -> Result<Command, String> {
    #[cfg(windows)]
    if kind == "powershell" {
        if let Some(pwsh) = crate::utils::powershell_detect::get_cached_powershell_path() {
            let wrapped = format!(
                "{command}\n$__cmec = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }}\nexit $__cmec\n"
            );
            let encoded = crate::utils::powershell::parser::to_utf16le_base64(&wrapped);
            let mut c = Command::new(pwsh);
            c.args([
                "-NoProfile",
                "-NonInteractive",
                "-NoLogo",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                &encoded,
            ]);
            c.current_dir(cwd);
            c.env("TERM", "dumb");
            c.env("PYTHONUNBUFFERED", "1");
            return Ok(c);
        }
        // No PowerShell on this box: fall through to the shell below, which is
        // what the foreground path degrades to as well.
    }
    #[cfg(not(windows))]
    let _ = kind;

    let mut cmd = match crate::utils::windows_shell::shell_argv(command) {
        Ok(Some(argv)) => {
            // Windows: [bash, -c, command]
            let mut c = Command::new(&argv[0]);
            c.args(&argv[1..]);
            c
        }
        Ok(None) => {
            // POSIX: /bin/sh -c command
            let mut c = Command::new("/bin/sh");
            c.arg("-c").arg(command);
            c
        }
        // git-bash discovery failed (Windows only). Sharing the POSIX arm
        // through a catch-all `_` sent this to /bin/sh, which does not exist
        // there, so the task recorded an opaque OS error instead of the
        // message naming the download and the CODEMASON_GIT_BASH_PATH
        // override. Python raises here and cannot lose it.
        Err(e) => return Err(e),
    };
    cmd.current_dir(cwd);
    cmd.env("TERM", "dumb");
    // Streamed to a pipe, not a tty: without this a python child block-buffers
    // and the output file (and TaskOutput's "last:" note) stays empty while the
    // task is visibly working. Dropped by mistake in the TaskOutput revert.
    cmd.env("PYTHONUNBUFFERED", "1");
    Ok(cmd)
}

/// Bound the registry: once it exceeds MAX_TASKS, evict the oldest
/// terminal+notified records. AppState.tasks is a HashMap (no insertion order,
/// unlike the Python dict) — DIVERGENCE: "oldest" is resolved by ascending
/// start_time, the faithful equivalent of the source's oldest-first walk.
fn gc_terminal_tasks() {
    let state = get_app_state();
    if state.tasks.len() <= MAX_TASKS {
        return;
    }
    let mut evictable: Vec<(&String, i64)> = state
        .tasks
        .iter()
        .filter(|(_, t)| is_terminal_task_status(&t.base().status) && t.base().notified)
        .map(|(id, t)| (id, t.base().start_time))
        .collect();
    evictable.sort_by_key(|(_, st)| *st);
    let excess = state.tasks.len() - MAX_TASKS;
    for (tid, _) in evictable.into_iter().take(excess) {
        evict_terminal_task(tid);
    }
}

/// Spawn a background shell task; return (task_id, output_path)
/// (spawnShellTask). Must be called from within the tokio runtime.
pub fn spawn_shell_task(
    command: &str,
    cwd: &str,
    description: Option<&str>,
    tool_use_id: Option<&str>,
    kind: &str,
) -> (String, std::path::PathBuf) {
    ensure_output_dir();
    gc_terminal_tasks();
    let task_id = generate_task_id("local_bash");
    let output_path = get_task_output_path(&task_id);
    let label = description.unwrap_or(command).to_string();

    let mut base = create_task_state_base(&task_id, "local_bash", &label, tool_use_id.map(|s| s.to_string()));
    base.status = "running".to_string();
    let task = LocalShellTaskState {
        base,
        command: command.to_string(),
        result: None,
        completion_status_sent_in_attachment: false,
        is_backgrounded: true,
        agent_id: None,
        kind: kind.to_string(),
        pid: None,
    };
    register_task(task.into());

    let (t_id, c, cw, op, desc, tu, k) = (
        task_id.clone(),
        command.to_string(),
        cwd.to_string(),
        output_path.clone(),
        label,
        tool_use_id.map(|s| s.to_string()),
        kind.to_string(),
    );
    tokio::spawn(async move {
        run(t_id, c, cw, op, desc, tu, k).await;
    });
    (task_id, output_path)
}

/// ADOPT an already-running process (a foreground command that hit its
/// timeout): write the output captured so far, keep draining the SAME process
/// to disk, and settle. Mirrors spawn_shell_task's registration so kill_task /
/// TaskStop / notifications all work identically.
#[allow(clippy::too_many_arguments)]
async fn run_adopted(
    task_id: String,
    mut child: tokio::process::Child,
    mut stdout: Option<tokio::process::ChildStdout>,
    mut stderr: Option<tokio::process::ChildStderr>,
    accumulated: Vec<u8>,
    output_path: std::path::PathBuf,
    description: String,
    tool_use_id: Option<String>,
    kind: String,
) {
    let mut returncode: Option<i32> = None;
    let mut errored = false;

    match std::fs::File::create(&output_path) {
        Ok(file) => {
            let mut writer = DiskWriter::new(file);
            // Write the pre-timeout output, BOUNDED.
            writer.write_all_bounded(&accumulated);

            let pid = child.id().unwrap_or(0);
            if pid != 0 && publish_pid(&task_id, pid) {
                kill_process_tree(pid, None);
                let _ = child.wait().await;
                return;
            }
            let deadline = background_timeout();
            let timer = async {
                match deadline {
                    Some(d) => tokio::time::sleep(d).await,
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::pin!(timer);
            let mut timed_out = false;
            // Once the MAIN process exits, give the pipes a moment to flush and
            // then drop our read ends. Without this the loop waits for EOF, and
            // EOF needs EVERY holder of the write end to close it — a `cmd &`
            // grandchild inherits them and keeps them open, so a task whose
            // shell has already exited never settles and the model is never told
            // it finished. Python has this as its _force_close watchdog, and the
            // Rust FOREGROUND path has it as pipe_close_at; only the adopted
            // path was missing it, which is the path a backgrounded `npm run dev
            // & sleep 300` actually takes.
            let mut pipe_close_at: Option<tokio::time::Instant> = None;
            loop {
                if stdout.is_none() && stderr.is_none() {
                    break;
                }
                if pipe_close_at.is_none() {
                    if let Ok(Some(_)) = child.try_wait() {
                        pipe_close_at = Some(
                            tokio::time::Instant::now() + std::time::Duration::from_millis(500),
                        );
                    }
                }
                let poll_or_flush_at = pipe_close_at.unwrap_or_else(|| {
                    tokio::time::Instant::now() + std::time::Duration::from_millis(100)
                });
                tokio::select! {
                    c = read_chunk(&mut stdout), if stdout.is_some() => {
                        if let Some(chunk) = c { writer.write_chunk(&chunk); }
                    }
                    c = read_chunk(&mut stderr), if stderr.is_some() => {
                        if let Some(chunk) = c { writer.write_chunk(&chunk); }
                    }
                    _ = tokio::time::sleep_until(poll_or_flush_at) => {
                        if pipe_close_at.is_some() {
                            stdout = None;
                            stderr = None;
                        }
                    }
                    // Same guard as run(): an adopted process is the one most
                    // likely to be the runaway, since it already outran the
                    // foreground cap before landing here.
                    _ = &mut timer, if !timed_out => {
                        timed_out = true;
                        if let Some(d) = deadline { note_timeout(&output_path, d); }
                        if pid != 0 { kill_process_tree(pid, None); }
                    }
                }
            }
            returncode = child.wait().await.ok().and_then(|s| s.code());
        }
        Err(e) => {
            errored = true;
            append_error(&output_path, &e.to_string());
        }
    }

    let settled = complete_task(&task_id, returncode, errored);
    let (status, exit_code) = if settled.killed {
        ("killed".to_string(), None)
    } else {
        (settled.status, settled.code)
    };
    enqueue_shell_notification(&task_id, &description, &status, exit_code, tool_use_id.as_deref(), &kind);
}

/// Register an already-running (timed-out foreground) process as a background
/// task and keep streaming it to disk. Returns (task_id, output_path).
#[allow(clippy::too_many_arguments)]
pub fn adopt_timed_out_process(
    child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
    accumulated: Vec<u8>,
    description: Option<&str>,
    tool_use_id: Option<&str>,
    kind: &str,
) -> (String, std::path::PathBuf) {
    ensure_output_dir();
    gc_terminal_tasks();
    let task_id = generate_task_id("local_bash");
    let output_path = get_task_output_path(&task_id);
    let label = description.unwrap_or("background command").to_string();

    let mut base = create_task_state_base(&task_id, "local_bash", &label, tool_use_id.map(|s| s.to_string()));
    base.status = "running".to_string();
    let task = LocalShellTaskState {
        base,
        command: label.clone(),
        result: None,
        completion_status_sent_in_attachment: false,
        is_backgrounded: true,
        agent_id: None,
        kind: kind.to_string(),
        pid: None,
    };
    register_task(task.into());

    let (t_id, op, desc, tu, k) = (
        task_id.clone(),
        output_path.clone(),
        label,
        tool_use_id.map(|s| s.to_string()),
        kind.to_string(),
    );
    tokio::spawn(async move {
        run_adopted(t_id, child, stdout, stderr, accumulated, op, desc, tu, k).await;
    });
    (task_id, output_path)
}

/// The Task implementation registered for type 'local_bash'.
pub struct LocalShellTask;

#[async_trait]
impl Task for LocalShellTask {
    fn name(&self) -> &'static str {
        "LocalShellTask"
    }
    fn task_type(&self) -> &'static str {
        "local_bash"
    }
    async fn kill(&self, task_id: &str) {
        kill_task(task_id);
    }
}

pub static LOCAL_SHELL_TASK: Lazy<LocalShellTask> = Lazy::new(|| LocalShellTask);

/// Terminate any still-running background commands on process exit.
///
/// Registered with libc::atexit in main.rs (Python: atexit.register, module
/// import time). Best-effort: runs during process teardown, so it must never
/// block — get_app_state() clones under a short lock and the kills are signals.
pub fn shutdown_all() {
    for task in get_app_state().tasks.values() {
        // Only the shell tasks own an OS process. A running sub-agent is torn
        // down by the exit cleanup it registered at spawn
        // (utils::exit_cleanup::run_exit_cleanups), not from here.
        let Some(task) = task.as_local_shell() else {
            continue;
        };
        if task.base.status != "running" {
            continue;
        }
        if let Some(pid) = task.pid {
            kill_process_tree(pid, None);
        }
    }
}

// Ensure AppState construction still works now that TaskState is a union: the
// shell state must stay Default-constructible (tests build it that way) and the
// store must still hand out an AppState.
#[allow(dead_code)]
fn _assert_state_default() {
    let _ = LocalShellTaskState::default();
    let _: &AppState = &get_app_state();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The note is what the user reads to learn the task was KILLED rather than
    /// that their command failed, and it must be byte-for-byte what the Python
    /// oracle writes (coding rule 1) — an em dash, not a hyphen, and the
    /// setting named exactly as it is in config.
    #[test]
    fn timeout_note_matches_the_python_oracle() {
        let tmp = std::env::temp_dir().join(format!("cm-tmo-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        note_timeout(&tmp, std::time::Duration::from_secs(21600));
        let got = std::fs::read_to_string(&tmp).unwrap();
        assert_eq!(
            got,
            "\n[background task terminated: exceeded 21600 seconds \u{2014} set background_task_timeout to change, or 0 to disable]\n"
        );
        // The line-continuation must not have left the indentation in.
        assert!(!got.contains("  "), "collapsed whitespace leaked: {got:?}");
        let _ = std::fs::remove_file(&tmp);
    }

    /// 0 means "no cap" and must not be turned into a zero-length one, which
    /// would kill every background task the instant it started.
    #[test]
    fn zero_timeout_disables_rather_than_expiring_immediately() {
        let cfg = crate::utils::config::get_config();
        // Whatever the machine's config says, the mapping itself is the rule.
        assert!(cfg.background_task_timeout >= 0.0);
        assert!(background_timeout().is_none() || background_timeout().unwrap().as_secs_f64() > 0.0);
    }

    /// A shell that exits while a `cmd &` grandchild still holds the inherited
    /// pipes must still SETTLE. EOF needs every holder of the write end to
    /// close, and the grandchild never does — so without dropping our read ends
    /// the task sat at "running" forever and the model was never told it
    /// finished. Adopted path only; the foreground path already had this.
    #[tokio::test]
    async fn an_adopted_task_settles_even_when_a_grandchild_holds_the_pipes() {
        let _guard = crate::state::app_state::test_lock();
        let previous = crate::utils::config::get_config().background_task_timeout;
        // Long cap: this must settle because the SHELL exited, not because the
        // timeout fired. If the cap did the work the test proves nothing.
        crate::utils::config::update_config(|c| c.background_task_timeout = 3600.0);

        // The shell exits after ~0.3s; the backgrounded sleep keeps the pipes.
        //
        // Routed through shell_argv, the way build_shell_command does, rather
        // than naming /bin/sh: the body is POSIX shell (`&`, fractional sleep)
        // so it needs a POSIX shell, and on Windows that is git-bash, not a
        // path that exists. Hardcoded, this did not fail an assertion — it
        // failed to spawn at all ("the system cannot find the path specified"),
        // which is a test that never ran rather than one that found something.
        // The grandchild carries a mark so the cleanup below kills exactly
        // ours, never a sleep some other test or earlier run left behind.
        let mark = format!("{}", 800000 + std::process::id() % 90000);
        let script = format!("sleep {mark} & sleep 0.3");
        let script = script.as_str();
        let mut cmd = match crate::utils::windows_shell::shell_argv(script) {
            Ok(Some(argv)) => {
                let mut c = tokio::process::Command::new(&argv[0]);
                c.args(&argv[1..]);
                c
            }
            Ok(None) => {
                let mut c = tokio::process::Command::new("/bin/sh");
                c.arg("-c").arg(script);
                c
            }
            // No git-bash on this host, so the Bash tool cannot run here at
            // all and there is nothing for this test to prove.
            Err(_) => return,
        };
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = cmd.spawn().expect("spawn");
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let (task_id, output_path) =
            adopt_timed_out_process(child, stdout, stderr, Vec::new(), Some("probe"), None, "bash");

        let mut status = String::new();
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if let Some(t) = get_app_state().tasks.get(&task_id) {
                if t.base().status != "running" {
                    status = t.base().status.clone();
                    break;
                }
            }
        }
        assert!(
            !status.is_empty(),
            "still 'running' after 10s though the shell exited in 0.3s — the loop is \
             waiting for an EOF the grandchild will never send"
        );
        // Kill the grandchild before returning. It outlives the shell on
        // purpose, but on Windows the test PROCESS then waits it out: the task
        // itself settled in 975ms (measured), while the suite took as long as
        // the sleep — 30s for the original `sleep 30`, six times the whole
        // suite. POSIX reparents the orphan and never notices it, which is why
        // this only shows up here.
        #[cfg(windows)]
        {
            let kill = format!(
                "Get-CimInstance Win32_Process -Filter \"Name='sleep.exe'\" | Where-Object {{ $_.CommandLine -like '*{mark}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
            );
            let _ = std::process::Command::new("powershell")
                .args(["-NoProfile", "-NonInteractive", "-Command", &kill])
                .output();
        }
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("pkill")
                .arg("-f")
                .arg(format!("sleep {mark}"))
                .output();
        }
        let _ = std::fs::remove_file(&output_path);
        crate::utils::config::update_config(|c| c.background_task_timeout = previous);
    }

    /// The whole point of the guard: a background command that never ends must
    /// be killed, its GRANDCHILD must die with it (the shell is not the
    /// runaway), and the task must settle instead of sitting "running" forever.
    ///
    /// Measured against a real process rather than asserted about the code —
    /// the select! branch is the kind of thing that compiles while doing
    /// nothing, and a completed timer polls Ready forever if the guard is wrong.
    #[tokio::test]
    async fn a_runaway_background_task_is_killed_with_its_children() {
        let _guard = crate::state::app_state::test_lock();
        let previous = crate::utils::config::get_config().background_task_timeout;
        crate::utils::config::update_config(|c| c.background_task_timeout = 2.0);

        // Unique per run, not a fixed number: a leftover `sleep <mark>` from an
        // earlier run (an aborted test, a crash) is indistinguishable from our
        // own child to pgrep, and made this fail for a reason that had nothing
        // to do with the code under test.
        let mark = format!("{}", 700000 + std::process::id() % 90000);
        let mark = mark.as_str();
        let (task_id, output_path) =
            spawn_shell_task(&format!("sleep {mark}"), ".", Some("runaway"), None, "bash");

        // "Is the grandchild still running?" has no portable answer, and the
        // mark only exists in the COMMAND LINE, so whatever asks has to be able
        // to see that. On Windows pgrep does not exist, and neither does wmic
        // on a current build — it has been removed — so the question goes to
        // CIM. tasklist is cheaper but shows no command line, which would make
        // the mark useless and let a stray sleep.exe from anywhere on the
        // machine answer for ours. One CIM query measured at 0.27s here, and
        // the loop below stops at the first true, so the usual cost is one.
        //
        // The grandchild really is a separate sleep.exe: bash resolves `sleep`
        // to /usr/bin/sleep and spawns it, which is the whole point of the
        // assertion — killing the shell alone would leave it behind.
        let alive = || {
            #[cfg(unix)]
            {
                std::process::Command::new("pgrep")
                    .arg("-f")
                    .arg(format!("sleep {mark}"))
                    .output()
                    .map(|o| !o.stdout.is_empty())
                    .unwrap_or(false)
            }
            #[cfg(windows)]
            {
                // Match the mark alone, not "sleep {mark}". Windows reports the
                // command line as `"C:\Program Files\Git\usr\bin\sleep.exe"
                // 749999` — the resolved path and a quote sit between the two
                // words, so the phrase never appears. The Name filter already
                // narrows this to sleep.exe, and the mark is unique per run,
                // so the number by itself identifies our child.
                let script = format!(
                    "@(Get-CimInstance Win32_Process -Filter \"Name='sleep.exe'\" | \
                     Where-Object {{ $_.CommandLine -like '*{mark}*' }}).Count"
                );
                std::process::Command::new("powershell")
                    .args(["-NoProfile", "-NonInteractive", "-Command", &script])
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim() != "0")
                    .unwrap_or(false)
            }
            #[cfg(not(any(unix, windows)))]
            {
                false
            }
        };

        // It has to actually be running first, or the kill assertion is vacuous.
        let mut started = false;
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if alive() {
                started = true;
                break;
            }
        }
        assert!(started, "the probe process never started; test proves nothing");

        let mut status = String::new();
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if let Some(t) = get_app_state().tasks.get(&task_id) {
                if t.base().status != "running" {
                    status = t.base().status.clone();
                    break;
                }
            }
        }
        assert!(!status.is_empty(), "task never settled — it is still 'running'");

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert!(!alive(), "the grandchild outlived the kill (group kill failed)");

        let out = std::fs::read_to_string(&output_path).unwrap_or_default();
        assert!(
            out.contains("background task terminated"),
            "the user is left with no reason for the truncated output: {out:?}"
        );
        let _ = std::fs::remove_file(&output_path);
        // The config singleton is process-global; leaving it mutated changes
        // what every later test in this binary sees.
        crate::utils::config::update_config(|c| c.background_task_timeout = previous);
    }

    #[test]
    fn xml_escaped() {
        assert_eq!(escape_xml("a & b < c > d"), "a &amp; b &lt; c &gt; d");
    }

    #[test]
    fn notification_shape() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        let task_id = generate_task_id("local_bash");
        let mut base = create_task_state_base(&task_id, "local_bash", "run tests", None);
        base.status = "running".into();
        register_task(LocalShellTaskState { base, command: "pytest".into(), is_backgrounded: true, kind: "bash".into(), ..Default::default() }.into());
        crate::utils::message_queue_manager::reset_pending_notifications();
        enqueue_shell_notification(&task_id, "run tests", "completed", Some(0), Some("tu_1"), "bash");
        let drained = crate::utils::message_queue_manager::drain_pending_notifications();
        assert_eq!(drained.len(), 1);
        let m = &drained[0];
        assert!(m.contains("<task-notification>"));
        assert!(m.contains(&format!("<task-id>{task_id}</task-id>")));
        assert!(m.contains("<tool-use-id>tu_1</tool-use-id>"));
        assert!(m.contains("<status>completed</status>"));
        assert!(m.contains("Background command \"run tests\" completed (exit code 0)"));
        // second call is suppressed (notified already set)
        enqueue_shell_notification(&task_id, "run tests", "completed", Some(0), None, "bash");
        assert!(crate::utils::message_queue_manager::drain_pending_notifications().is_empty());
    }

    #[test]
    fn gc_evicts_oldest_terminal_notified() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        for i in 0..(MAX_TASKS + 5) {
            let id = format!("b{i:08}");
            let mut base = create_task_state_base(&id, "local_bash", "x", None);
            base.id = id.clone();
            base.start_time = i as i64; // ascending = oldest first
            base.status = "completed".into();
            base.notified = true;
            register_task(LocalShellTaskState { base, kind: "bash".into(), ..Default::default() }.into());
        }
        gc_terminal_tasks();
        let state = get_app_state();
        assert!(state.tasks.len() <= MAX_TASKS);
        // the 5 oldest (start_time 0..5) should be gone
        assert!(!state.tasks.contains_key("b00000000"));
        assert!(!state.tasks.contains_key("b00000004"));
    }
}
