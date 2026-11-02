//! Persistent session goal, with optional completion gates.
//! Equivalent to codemason/utils/goal.py.
//!
//! A goal is a free-form user objective that stays active across turns: once
//! set with `/goal <text>`, it is injected into the system prompt on every
//! turn (as an `<active_goal>` context block) so the agent keeps the
//! overarching objective in view instead of drifting after the first sub-task.
//!
//! A goal can also carry **completion gates** — deterministic shell commands
//! that must all pass (exit 0) before the goal is genuinely done (e.g. the
//! test suite, the linter). The gates are listed to the agent inside the goal
//! block as the completion bar, and `/goal check` runs them and reports
//! pass/fail. Gates make "done" verifiable instead of a matter of opinion.
//!
//! State persists as goal.txt + goal-gates.txt **inside the project directory**
//! (`<CONFIG_DIR>/projects/<sanitized-cwd>/`), so it survives a restart until
//! explicitly cleared.
//!
//! The project directory, and not the config dir, because a goal is a property
//! of the work — and its gates are literally that project's test command. One
//! file per home cannot answer a question two instances answer differently: with
//! a `cm2` in one repo and a `codemason` in another, a config-dir goal.txt meant
//! whoever typed `/goal` last silently retargeted every other session, and
//! `/goal clear` deleted a goal the other window was still working to. That is
//! the same reasoning — and the same key — as session_storage's per-directory
//! settings and the transcripts that already live there. (No migration: the old
//! config-dir files were never released with anything that wrote them.)

use std::path::PathBuf;

use super::session_storage::{get_project_dir, project_lock};
use super::state::get_original_cwd;

fn goal_file() -> PathBuf {
    get_project_dir(&get_original_cwd()).join("goal.txt")
}

fn gates_file() -> PathBuf {
    get_project_dir(&get_original_cwd()).join("goal-gates.txt")
}

const GATE_TIMEOUT_S: u64 = 300; // per-gate hard timeout
const GATE_OUTPUT_TAIL: usize = 2000; // chars of a failed gate's output fed back

// ── goal text ───────────────────────────────────────────────────────────────

/// The active goal text, or "" if none is set. Never raises.
pub fn get_goal() -> String {
    let path = goal_file();
    if path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            return text.trim().to_string();
        }
    }
    String::new()
}

/// Set (and persist) the active goal.
pub fn set_goal(text: &str) {
    let text = text.trim();
    if text.is_empty() {
        clear_goal();
        return;
    }
    let path = goal_file();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Atomic: another session in this directory, or the turn that is about to
    // render the goal into its system prompt, must never read half of it. No lock
    // needed — this replaces the whole document rather than editing one line of
    // it, so there is nothing to merge.
    let _ = crate::utils::file::atomic_write_bytes(&path.to_string_lossy(), text.as_bytes());
}

/// Remove the active goal and its gates.
pub fn clear_goal() {
    for f in [goal_file(), gates_file()] {
        let _ = std::fs::remove_file(f);
    }
}

// ── completion gates ──────────────────────────────────────────────────────────

/// The gate commands, in order. Never raises.
pub fn get_gates() -> Vec<String> {
    let path = gates_file();
    if path.is_file() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            return text
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
        }
    }
    Vec::new()
}

/// Replace the gate list. Call it holding the gates lock (see edit_gates) when
/// the new list was derived from the old one.
fn save_gates(gates: &[String]) {
    let path = gates_file();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if !gates.is_empty() {
        let _ = crate::utils::file::atomic_write_bytes(
            &path.to_string_lossy(),
            format!("{}\n", gates.join("\n")).as_bytes(),
        );
    } else {
        let _ = std::fs::remove_file(&path);
    }
}

/// Read the gates, change them, write them back as one step.
///
/// Adding and removing are read-modify-write: two sessions in this directory
/// doing it at once would otherwise each write a list that never saw the other's
/// edit, and one gate — a test command the user means to be a release bar —
/// would vanish with nothing said.
fn edit_gates<F>(mutate: F) -> bool
where
    F: FnOnce(&mut Vec<String>) -> bool,
{
    let _lock = crate::utils::file::FileLock::acquire(&project_lock("goal-gates.txt", ""));
    let mut gates = get_gates();
    if !mutate(&mut gates) {
        return false;
    }
    save_gates(&gates);
    true
}

pub fn add_gate(command: &str) {
    let command = command.trim();
    if command.is_empty() {
        return;
    }
    edit_gates(|gates| {
        gates.push(command.to_string());
        true
    });
}

/// Remove the 1-based gate at `index`. Returns true if it existed.
pub fn remove_gate(index: usize) -> bool {
    edit_gates(|gates| {
        if index >= 1 && index <= gates.len() {
            gates.remove(index - 1);
            true
        } else {
            false
        }
    })
}

pub fn clear_gates() {
    save_gates(&[]);
}

fn tail_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        s.to_string()
    } else {
        chars[chars.len() - n..].iter().collect()
    }
}

/// Run one gate command through the shell. Returns (passed, exit_code, tail).
///
/// The 300s timeout kills the child (subprocess.run's kill-on-timeout) via a
/// polling wait — std::process has no native timeout.
pub fn run_gate(command: &str, cwd: Option<&str>) -> (bool, i32, String) {
    // shell=True: sh -c on POSIX, cmd /C on Windows (subprocess.run's mapping).
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = std::process::Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    #[cfg(windows)]
    let mut cmd = {
        let mut c = std::process::Command::new("cmd");
        c.arg("/C").arg(command);
        c
    };
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null());
    if let Some(cwd) = cwd {
        if !cwd.is_empty() {
            cmd.current_dir(cwd);
        }
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(exc) => {
            return (false, -1, format!("[gate could not run: {}: {exc}]", err_kind_name(&exc)))
        }
    };

    // Drain pipes on threads so a chatty gate can't dead-lock on a full pipe.
    let stdout_handle = child.stdout.take().map(|mut out| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buf = Vec::new();
            let _ = out.read_to_end(&mut buf);
            buf
        })
    });
    let stderr_handle = child.stderr.take().map(|mut err| {
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buf = Vec::new();
            let _ = err.read_to_end(&mut buf);
            buf
        })
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(GATE_TIMEOUT_S);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => break None,
        }
    };

    let stdout = stdout_handle
        .and_then(|h| h.join().ok())
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let stderr = stderr_handle
        .and_then(|h| h.join().ok())
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();

    let Some(status) = status else {
        return (false, -1, format!("[gate timed out after {GATE_TIMEOUT_S}s]"));
    };

    let combined = if stderr.is_empty() {
        stdout
    } else {
        format!("{stdout}\n{stderr}")
    };
    let code = status.code().unwrap_or(-1);
    (status.success(), code, tail_chars(&combined, GATE_OUTPUT_TAIL))
}

fn err_kind_name(e: &std::io::Error) -> String {
    format!("{:?}", e.kind())
}

/// Run every gate. Returns [(command, passed, exit_code, output_tail), ...].
pub fn run_gates(cwd: Option<&str>) -> Vec<(String, bool, i32, String)> {
    get_gates()
        .into_iter()
        .map(|g| {
            let (passed, code, tail) = run_gate(&g, cwd);
            (g, passed, code, tail)
        })
        .collect()
}

// ── injection ─────────────────────────────────────────────────────────────────

/// The text injected as the <active_goal> block each turn, or "" if no goal.
///
/// Includes the completion gates as the explicit "done" bar so the agent knows
/// not to declare the goal complete until they pass.
pub fn render_goal_context() -> String {
    let goal = get_goal();
    if goal.is_empty() {
        return String::new();
    }
    let gates = get_gates();
    if gates.is_empty() {
        return goal;
    }
    let mut lines = vec![
        goal,
        String::new(),
        "This goal is complete ONLY when ALL of these gate commands pass \
(exit 0). Run them to verify; do not declare the goal done while \
any gate fails:"
            .to_string(),
    ];
    for (i, g) in gates.iter().enumerate() {
        lines.push(format!("  {}. $ {g}", i + 1));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `run_gate` mirrors Python's `subprocess(shell=True)`: `sh -c` on POSIX,
    /// `cmd /C` on Windows. The gate script therefore has to be written in the
    /// platform's shell. `cmd` does not know `true`, and it treats `;` as a
    /// literal rather than a statement separator, so the POSIX script below
    /// used to echo "boom; exit 3" and exit 0 — the failure case never failed.
    /// `&` is cmd's separator.
    #[cfg(not(windows))]
    const GATE_OK: &str = "true";
    #[cfg(not(windows))]
    const GATE_FAIL: &str = "echo boom; exit 3";
    #[cfg(windows)]
    const GATE_OK: &str = "exit 0";
    #[cfg(windows)]
    const GATE_FAIL: &str = "echo boom& exit 3";

    #[test]
    fn gate_runs_and_tails() {
        let (passed, code, _tail) = run_gate(GATE_OK, None);
        assert!(passed);
        assert_eq!(code, 0);
        let (passed, code, tail) = run_gate(GATE_FAIL, None);
        assert!(!passed);
        assert_eq!(code, 3);
        assert!(tail.contains("boom"));
    }
}
