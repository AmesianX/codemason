//! Windows git-bash substrate for CodeMason (Tier 1).
//! Equivalent to codemason/utils/windows_shell.py (Claude Code
//! src/utils/windowsPaths.ts, updated v2.1.208 `nve`/`CHn` form), which
//! hardened three things over the earlier form:
//!
//!   1. CODEMASON_GIT_BASH_PATH is checked with a filesystem stat, not by
//!      shelling out to `dir` (no shell-parse surface).
//!   2. `where.exe` is invoked by absolute %SYSTEMROOT%\System32\where.exe, so
//!      a malicious where.exe on the cwd/PATH can't hijack discovery.
//!   3. `C:\Program Files\Git\bin\bash.exe` (and the x86 path) are checked
//!      directly, before the git-derived fallback.
//!
//! Tier 1 = "there is a POSIX shell on this host": genuine Linux/macOS/WSL2,
//! or native Windows WITH git-bash/MSYS2. The command bodies our substrate
//! builds are POSIX sh (`pwd -P`, `$?`, `/dev/null`), so on Windows we route
//! them through bash.exe unchanged rather than cmd.exe. Native cmd/PowerShell
//! is Tier 2.
//!
//! Everything here compiles on all platforms; POSIX branches never touch
//! Windows code paths and vice-versa.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::path::Path;

use crate::utils::platform::is_windows;

/// where.exe timeout (windowsPaths.ts Xwm = 5000ms).
#[allow(dead_code)]
const WHERE_TIMEOUT_S: f64 = 5.0;

/// Executable extensions worth trusting from a where.exe hit (Jwm).
const EXE_EXTS: &[&str] = &[".com", ".exe", ".bat", ".cmd"];

/// Known git-bash install locations (nve). 64-bit before 32-bit.
const KNOWN_BASH_PATHS: &[&str] = &[
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
];

static GIT_BASH_CACHE: Lazy<Mutex<(bool, Option<String>)>> =
    Lazy::new(|| Mutex::new((false, None)));

/// Path ends in a Windows executable extension (Qwm in windowsPaths.ts).
fn has_exe_ext(path: &str) -> bool {
    let base = path
        .to_lowercase()
        .replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or("")
        .trim_end_matches(['.', ' '])
        .to_string();
    match base.rfind('.') {
        Some(dot) if dot > 0 => EXE_EXTS.contains(&&base[dot..]),
        _ => false,
    }
}

/// True if `candidate` resolves inside `directory` (ISn cwd-hijack filter).
fn is_in_dir(candidate: &str, directory: &str) -> bool {
    let (Ok(cand), Ok(base)) = (
        std::fs::canonicalize(candidate),
        std::fs::canonicalize(directory),
    ) else {
        return false;
    };
    // normcase: Windows paths compare case-insensitively.
    let cand = cand.to_string_lossy().to_lowercase();
    let base = base.to_string_lossy().to_lowercase();
    cand == base || cand.starts_with(&format!("{base}{}", std::path::MAIN_SEPARATOR))
}

/// Locate git.exe with absolute System32\where.exe, skipping cwd hits (CHn).
///
/// Filters out any hit inside the current directory (anti-hijack) and requires
/// a real executable extension, mirroring the v2.1.208 discovery.
fn find_git_via_where() -> Option<String> {
    let where_exe = system32_exe("where.exe");
    let out = std::process::Command::new(&where_exe)
        .arg("git")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    // (std::process has no per-call timeout; where.exe on an absolute System32
    // path terminates promptly — the 5s guard existed for pathological hosts.)
    let stdout = String::from_utf8_lossy(&out.stdout);
    let cwd = std::env::current_dir().ok()?.to_string_lossy().into_owned();
    for line in stdout.lines() {
        let candidate = line.trim();
        if candidate.is_empty() {
            continue;
        }
        if !Path::new(candidate).exists() {
            continue;
        }
        if is_in_dir(candidate, &cwd) {
            // SECURITY: never a cwd-planted git
            continue;
        }
        if !has_exe_ext(candidate) {
            continue;
        }
        return Some(candidate.to_string());
    }
    None
}

/// Absolute path to git-bash's bash.exe, or None if not found (nve).
///
/// Order: CODEMASON_GIT_BASH_PATH → known install paths → git-derived path.
/// Memoized. Unlike the source we return None instead of process.exit() when
/// the override points nowhere — the caller decides how to surface it (see
/// git_bash_or_raise); a library must not kill the host.
pub fn find_git_bash_path() -> Option<String> {
    {
        let cache = GIT_BASH_CACHE.lock();
        if cache.0 {
            return cache.1.clone();
        }
    }
    let result = compute_git_bash_path();
    let mut cache = GIT_BASH_CACHE.lock();
    cache.0 = true;
    cache.1 = result.clone();
    result
}

fn compute_git_bash_path() -> Option<String> {
    if let Ok(env_path) = std::env::var("CODEMASON_GIT_BASH_PATH") {
        if !env_path.is_empty() {
            return if Path::new(&env_path).exists() {
                Some(env_path)
            } else {
                None
            };
        }
    }

    for known in KNOWN_BASH_PATHS {
        if Path::new(known).exists() {
            return Some(known.to_string());
        }
    }

    if let Some(git_path) = find_git_via_where() {
        // <git>/cmd/git.exe or <git>/bin/git.exe → <git>/bin/bash.exe
        let p = Path::new(&git_path);
        if let Some(root) = p.parent().and_then(|d| d.parent()) {
            let derived = root.join("bin").join("bash.exe");
            if derived.exists() {
                return Some(derived.to_string_lossy().into_owned());
            }
        }
    }
    None
}

pub const GIT_BASH_MISSING_MSG: &str = "CodeMason on native Windows requires git-bash (https://git-scm.com/downloads/win) for the Bash tool. If it is installed but not discovered, point CODEMASON_GIT_BASH_PATH at your bash.exe, e.g.\n  set CODEMASON_GIT_BASH_PATH=C:\\Program Files\\Git\\bin\\bash.exe";

/// Return bash.exe path or a clear, actionable error (Windows only).
pub fn git_bash_or_raise() -> Result<String, String> {
    find_git_bash_path().ok_or_else(|| GIT_BASH_MISSING_MSG.to_string())
}

// === POSIX <-> Windows path conversion (windowsPaths.ts) ===

/// C:\Users\foo -> /c/Users/foo ; \\server\share -> //server/share.
pub fn windows_path_to_posix(win_path: &str) -> String {
    if win_path.starts_with("\\\\") {
        return win_path.replace('\\', "/");
    }
    let chars: Vec<char> = win_path.chars().collect();
    if chars.len() >= 3
        && chars[1] == ':'
        && (chars[2] == '/' || chars[2] == '\\')
        && chars[0].is_alphabetic()
    {
        let rest: String = chars[2..].iter().collect();
        return format!("/{}{}", chars[0].to_lowercase(), rest.replace('\\', "/"));
    }
    win_path.replace('\\', "/")
}

/// /c/Users/foo -> C:\Users\foo ; //server/share -> \\server\share (MSYS/cygdrive).
pub fn posix_path_to_windows(posix_path: &str) -> String {
    if posix_path.starts_with("//") {
        return posix_path.replace('/', "\\");
    }
    let chars: Vec<char> = posix_path.chars().collect();
    if posix_path.starts_with("/cygdrive/") && chars.len() >= 11 && chars[10].is_alphabetic() {
        let drive: String = chars[10].to_uppercase().collect();
        let rest: String = chars[11..].iter().collect();
        let rest = if rest.is_empty() { "\\".to_string() } else { rest.replace('/', "\\") };
        return format!("{drive}:{rest}");
    }
    if chars.len() >= 2
        && chars[0] == '/'
        && chars[1].is_alphabetic()
        && (chars.len() == 2 || chars[2] == '/')
    {
        let drive: String = chars[1].to_uppercase().collect();
        let rest: String = chars[2..].iter().collect();
        let rest = if rest.is_empty() { "\\".to_string() } else { rest.replace('/', "\\") };
        return format!("{drive}:{rest}");
    }
    posix_path.replace('/', "\\")
}

// === binShell resolution: cmd / powershell / pwsh / bash → (exe, args) ===
// Port of the source's binShell resolver (e1i). Maps a shell token (bare name
// or absolute path) to the executable and its leading args, so Tier-1 (bash)
// and Tier-2 (cmd/PowerShell) share one shell-selection seam.

/// PowerShell invocation args. The interactive form is -NoProfile -Command;
/// the actual tool run uses EncodedCommand (see powershell_tool). fUc in source.
const PWSH_ARGS: &[&str] = &["-NoProfile", "-Command"];

/// Resolved shell: `exe` (absolute or bare) + `args` (leading args before the
/// command). Mirrors the source's {exe, args} object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinShell {
    pub exe: String,
    pub args: Vec<String>,
}

// ntpath equivalents — Windows path semantics on ANY host (std::path is
// platform-native, so backslashes are opaque on unix; the source used
// path/win32 for the same reason).

fn nt_isabs(p: &str) -> bool {
    let chars: Vec<char> = p.chars().collect();
    if p.starts_with("\\\\") || p.starts_with("//") {
        return true;
    }
    // Drive-letter absolute: C:\... or C:/...
    chars.len() >= 3
        && chars[0].is_alphabetic()
        && chars[1] == ':'
        && (chars[2] == '\\' || chars[2] == '/')
}

fn nt_basename(p: &str) -> &str {
    p.rsplit(['\\', '/']).next().unwrap_or("")
}

fn nt_join(parts: &[&str]) -> String {
    let mut out = String::new();
    for part in parts {
        if out.is_empty() {
            out.push_str(part);
        } else {
            if !out.ends_with('\\') && !out.ends_with('/') {
                out.push('\\');
            }
            out.push_str(part);
        }
    }
    out
}

fn system_root() -> String {
    std::env::var("SystemRoot")
        .or_else(|_| std::env::var("SYSTEMROOT"))
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| r"C:\Windows".to_string())
}

fn default_cmd() -> BinShell {
    BinShell {
        exe: nt_join(&[&system_root(), "System32", "cmd.exe"]),
        args: vec!["/d".into(), "/s".into(), "/c".into()],
    }
}

/// A shell spec for resolve_bin_shell: nothing, a pre-resolved BinShell, or a
/// token/absolute path string.
pub enum BinShellSpec<'a> {
    Default,
    Shell(BinShell),
    Token(&'a str),
}

/// Resolve a shell spec to a BinShell (e1i port). Windows-oriented.
///
/// Default → cmd.exe; a BinShell → validated passthrough; or a token/absolute
/// path — 'bash'/'sh'(.exe) (must be absolute → args ['-c']), 'pwsh'(.exe) →
/// pwsh.exe -NoProfile -Command, 'powershell'(.exe) → 5.1 path, 'cmd'(.exe) →
/// cmd.exe /d /s /c.
pub fn resolve_bin_shell(spec: BinShellSpec) -> Result<BinShell, String> {
    let spec_str = match spec {
        BinShellSpec::Default => return Ok(default_cmd()),
        BinShellSpec::Shell(sh) => {
            if !nt_isabs(&sh.exe) {
                return Err(format!("binShell.exe must be an absolute path (got {:?})", sh.exe));
            }
            return Ok(sh);
        }
        BinShellSpec::Token(s) => s,
    };

    let base = nt_basename(spec_str);
    let lower = base.to_lowercase();
    let is_abs = nt_isabs(spec_str);
    if !is_abs && spec_str != base {
        return Err(format!(
            "binShell string must be a bare token or an absolute path (got {spec_str:?})"
        ));
    }

    match lower.as_str() {
        "bash" | "bash.exe" | "sh" | "sh.exe" => {
            if !is_abs {
                return Err(format!(
                    "binShell bash path must be absolute (got {spec_str:?}); pass the resolved Git Bash install path"
                ));
            }
            Ok(BinShell { exe: spec_str.to_string(), args: vec!["-c".into()] })
        }
        "pwsh" | "pwsh.exe" => {
            // A bare token resolved to relative "pwsh.exe" → PATH/CWD lookup,
            // the same hijack class hardened elsewhere. Prefer the absolute
            // standard install path when it exists; fall back to the PATH name
            // only for a non-standard install (preserves old behavior).
            if is_abs {
                return Ok(BinShell {
                    exe: spec_str.to_string(),
                    args: PWSH_ARGS.iter().map(|s| s.to_string()).collect(),
                });
            }
            let pf = nt_join(&[
                &std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into()),
                "PowerShell",
                "7",
                "pwsh.exe",
            ]);
            let exe = if Path::new(&pf).is_file() { pf } else { "pwsh.exe".to_string() };
            Ok(BinShell { exe, args: PWSH_ARGS.iter().map(|s| s.to_string()).collect() })
        }
        "powershell" | "powershell.exe" => {
            let default_ps =
                nt_join(&[&system_root(), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"]);
            let exe = if is_abs { spec_str.to_string() } else { default_ps };
            Ok(BinShell { exe, args: PWSH_ARGS.iter().map(|s| s.to_string()).collect() })
        }
        "cmd" | "cmd.exe" => {
            if !is_abs {
                Ok(default_cmd())
            } else {
                Ok(BinShell {
                    exe: spec_str.to_string(),
                    args: vec!["/d".into(), "/s".into(), "/c".into()],
                })
            }
        }
        _ => Err(format!(
            "unrecognised binShell {spec_str:?}: expected 'cmd' | 'powershell' | 'pwsh' or an absolute path to bash.exe/sh.exe/pwsh.exe/powershell.exe"
        )),
    }
}

// === Spawn helpers shared by both substrate sites ===

/// CREATE_NEW_PROCESS_GROUP: Windows analogue of start_new_session (own group
/// so the whole tree can be signalled/killed).
#[allow(dead_code)]
pub const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
/// CREATE_NO_WINDOW: don't allocate a console for the child. Our shell
/// children are fully redirected, so they never need a console; without it a
/// console-subsystem child flashes a window on every tool use when the host
/// has no console (upstream #61051).
#[allow(dead_code)]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// argv to run a POSIX-sh `script` via exec, or None.
///
/// On Windows returns Ok(Some(['<bash.exe>', '-c', script])) so the POSIX
/// command body runs under git-bash instead of cmd.exe (Err if git-bash is
/// missing — the Python raised here). On POSIX returns Ok(None) — callers
/// spawn `sh -c script` (create_subprocess_shell) unchanged.
pub fn shell_argv(script: &str) -> Result<Option<Vec<String>>, String> {
    if is_windows() {
        let bash = git_bash_or_raise()?;
        return Ok(Some(vec![bash, "-c".to_string(), script.to_string()]));
    }
    Ok(None)
}

/// Make everything this process spawns from now on die when it does, however
/// it dies.
///
/// `taskkill /F` is TerminateProcess: no atexit handler, no cleanup registry,
/// no Drop. The one real child we have is a shell, and it was being left
/// behind -- measured on Windows 11 Pro 10.0.26300, a background shell was
/// still running after the session that owned it was killed. Sub-agents are
/// tasks inside this process and go when it goes, so the shell is the only
/// thing that can outlive us, and Windows users reach for taskkill /F far more
/// often than POSIX users reach for SIGKILL.
///
/// A job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is the Windows answer,
/// and the closest thing it has to a process group. We hold the only handle and
/// never close it, so the close that matters is the one the kernel does when
/// this process ends -- by return, by panic, by TerminateProcess, by taskkill
/// /F. That is the guarantee a Drop or an atexit cannot give.
///
/// This process puts ITSELF in the job, which is the part that matters. Windows
/// enrols every process a member creates in the same job, at creation, so a
/// shell spawned after this call -- and anything that shell spawns -- is a
/// member before it runs an instruction. Assigning the child after spawn
/// instead leaves a window, and the Python build measured a grandchild
/// (`bash -c "python ..."`) starting inside it and surviving the kill.
///
/// It rations nothing: three shells spawned concurrently with the job on and
/// with it off both reached 3 running at once in the same wall time. Nested
/// jobs have been allowed since Windows 8, so this still works when a terminal
/// or a CI runner has already put us in a job of its own.
///
/// Only the shells go through here, via apply_spawn_group. Anything the user is
/// meant to keep -- a browser, an editor -- must be spawned with
/// CREATE_BREAKAWAY_FROM_JOB, or it would close when codemason exits.
#[cfg(windows)]
pub fn reap_children_with_this_process() {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    // The handle is kept for the life of the process: it being OPEN is what
    // keeps the job alive, and its closing is what kills the members.
    static JOB: OnceLock<Option<usize>> = OnceLock::new();

    JOB.get_or_init(|| {
        #[repr(C)]
        struct IoCounters {
            read_operation_count: u64,
            write_operation_count: u64,
            other_operation_count: u64,
            read_transfer_count: u64,
            write_transfer_count: u64,
            other_transfer_count: u64,
        }
        #[repr(C)]
        struct BasicLimit {
            per_process_user_time_limit: i64,
            per_job_user_time_limit: i64,
            limit_flags: u32,
            minimum_working_set_size: usize,
            maximum_working_set_size: usize,
            active_process_limit: u32,
            affinity: usize,
            priority_class: u32,
            scheduling_class: u32,
        }
        #[repr(C)]
        struct ExtendedLimit {
            basic_limit_information: BasicLimit,
            io_info: IoCounters,
            process_memory_limit: usize,
            job_memory_limit: usize,
            peak_process_memory_used: usize,
            peak_job_memory_used: usize,
        }
        extern "system" {
            fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> *mut c_void;
            fn SetInformationJobObject(
                job: *mut c_void,
                class: i32,
                info: *mut c_void,
                len: u32,
            ) -> i32;
            fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
            fn GetCurrentProcess() -> *mut c_void;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }
        const KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;
        const EXTENDED_LIMIT_INFORMATION: i32 = 9;

        unsafe {
            // Unnamed: ours alone, never shared with another session's job.
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: ExtendedLimit = std::mem::zeroed();
            info.basic_limit_information.limit_flags = KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                EXTENDED_LIMIT_INFORMATION,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<ExtendedLimit>() as u32,
            ) == 0
            {
                CloseHandle(job);
                return None;
            }
            // Failure here is not fatal: without it the shells still get their
            // own process group and the tidy paths still tree-kill them.
            AssignProcessToJobObject(job, GetCurrentProcess());
            Some(job as usize)
        }
    });
}

/// No-op off Windows: setsid plus the cleanup registry and killpg cover the
/// tidy paths there, and SIGKILL leaves the same hole for the same reason.
#[cfg(not(windows))]
pub fn reap_children_with_this_process() {}

/// Platform config giving a fully-redirected child its own process group
/// (spawn_group_kwargs).
///
/// POSIX: setsid (start_new_session=True). Windows: CREATE_NEW_PROCESS_GROUP
/// (own group for tree-kill) | CREATE_NO_WINDOW (no console flash).
pub fn apply_spawn_group(cmd: &mut tokio::process::Command) {
    #[cfg(unix)]
    {
        // start_new_session=True == setsid() in the child.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        // Before the spawn, so the child is a job member from creation: see
        // reap_children_with_this_process for why after is too late.
        reap_children_with_this_process();
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = cmd;
    }
}

/// Windows-only config to suppress the child's console; a no-op on POSIX
/// (no_window_kwargs).
///
/// For spawns that must NOT change their POSIX process-group behaviour
/// (hooks), but should still avoid the console flash on Windows.
pub fn apply_no_window(cmd: &mut tokio::process::Command) {
    #[cfg(windows)]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

/// Absolute path to a Windows system executable under System32.
///
/// Raw `CreateProcess` with a NULL lpApplicationName searches the current
/// directory before System32, which is why the upstream rule exists: invoking a
/// system tool by bare name would let a malicious <name>.exe planted in a
/// checked-out repo hijack the call.
///
/// SCOPE, measured rather than assumed: that hazard does NOT reach
/// `std::process::Command`. Rust's std resolves the program itself and passes an
/// absolute path, and it does not put the current directory on the search list
/// at all — a binary sitting in the working directory is reported as not found,
/// even by its exact name. So a bare `Command::new("cmd")` elsewhere in this
/// tree is not a hijack, and an audit that reads the sentence above as covering
/// those call sites reaches the wrong conclusion (this one did).
///
/// What this function still buys is PATH-order robustness: a bare name resolves
/// through PATH, and an absolute path does not. Someone who controls PATH can
/// already do a great deal more than drop a file in a repo, so treat this as
/// defence in depth, not as the thing standing between a checkout and code
/// execution.
pub fn system32_exe(name: &str) -> String {
    let system_root = std::env::var("SYSTEMROOT")
        .or_else(|_| std::env::var("WINDIR"))
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| r"C:\Windows".to_string());
    if is_windows() {
        nt_join(&[&system_root, "System32", name])
    } else {
        // Off-Windows this is only reached in tests; use native joining.
        Path::new(&system_root).join("System32").join(name).to_string_lossy().into_owned()
    }
}

/// Default LC_CTYPE=C.UTF-8 for child processes on native Windows.
///
/// git-bash/MSYS2 and the Unix coreutils it ships default to the C locale when
/// no locale is set — the usual case when CodeMason is launched from cmd.exe
/// or PowerShell. In the C locale those tools treat non-ASCII path bytes as
/// opaque and mangle them in output. Setting LC_CTYPE=C.UTF-8 makes the
/// charset UTF-8 while every other locale category stays C, so parsed tool
/// output is unchanged and Unicode paths round-trip.
///
/// Only LC_CTYPE, only when the user set no locale of their own, and only on
/// native Windows. POSIX inherits the real system locale untouched.
pub fn ensure_utf8_locale() {
    if !is_windows() {
        return;
    }
    let unset = ["LANG", "LC_ALL", "LC_CTYPE"]
        .iter()
        .all(|k| std::env::var(k).map(|v| v.is_empty()).unwrap_or(true));
    if unset {
        std::env::set_var("LC_CTYPE", "C.UTF-8");
    }
}

/// The encoding for a Windows codepage number, when one exists.
///
/// Split out from the OS call so the mapping can be tested anywhere. Only the
/// WHATWG set is available: that covers every CJK OEM page, which is where the
/// mojibake actually bites, but NOT the DOS pages a Western install uses (437,
/// 850). For those the caller keeps the old lossy decode, so nothing regresses
/// — it simply stays as good as it was.
///
/// Compiled everywhere, not `#[cfg(windows)]`: it is a pure number-to-encoding
/// mapping with no OS call in it, and gating the CODE means it is never
/// type-checked or tested off Windows. That is the same shape as the drift that
/// broke the Windows build from Linux once already, so this repository gates the
/// LINT instead (see utils/mod.rs).
#[cfg_attr(not(windows), allow(dead_code))]
fn encoding_for_codepage(cp: u32) -> Option<&'static encoding_rs::Encoding> {
    let label: &str = match cp {
        949 => "euc-kr",    // windows-949, the Unified Hangul superset
        932 => "shift_jis", // windows-31j
        936 => "gbk",
        950 => "big5",
        866 => "ibm866",
        874 => "windows-874",
        1250 => "windows-1250",
        1251 => "windows-1251",
        1252 => "windows-1252",
        1253 => "windows-1253",
        1254 => "windows-1254",
        1255 => "windows-1255",
        1256 => "windows-1256",
        1257 => "windows-1257",
        1258 => "windows-1258",
        65001 => "utf-8",
        _ => return None,
    };
    encoding_rs::Encoding::for_label(label.as_bytes())
}

/// Decode captured child stdout/stderr to text.
///
/// git-bash builtins and Unix tools emit UTF-8, but native Windows console
/// programs emit the OEM codepage, and reading those bytes as UTF-8 destroys
/// them. Measured on this platform before the OEM step existed, running
/// `tasklist` with a filter that matches nothing:
///
///     ����: ���� ���� �۾� �� ������ ���ǿ� ��ġϴ� �۾��� �����ϴ�.
///
/// Every Korean byte became U+FFFD. The text is not merely ugly at that point,
/// it is gone: the model can only guess what the command said.
///
/// Strict UTF-8 first, because git-bash tools and everything POSIX emit it and
/// that must not be second-guessed. Only when that fails is the OEM codepage
/// tried — and the codepage is not a guess, GetOEMCP is the OS telling us what
/// its console programs write.
///
/// Deliberately NOT verified by re-encoding, unlike the file-encoding
/// detection: that check exists because Edit writes the decoded text back, so a
/// wrong guess corrupts a file. Console output is displayed and never written
/// back, the encoding is known rather than inferred, and demanding a perfect
/// round-trip would throw away output over one odd byte and hand back the
/// U+FFFD version instead.
pub fn decode_console_output(buf: &[u8]) -> String {
    // Fast path: the whole stream is UTF-8 and nothing else matters.
    if let Ok(s) = std::str::from_utf8(buf) {
        return s.to_string();
    }

    #[cfg(windows)]
    {
        extern "system" {
            fn GetOEMCP() -> u32;
        }
        if let Some(enc) = encoding_for_codepage(unsafe { GetOEMCP() }) {
            // Run by run, not the whole buffer. A Windows program and cmd.exe
            // itself write into ONE stream, the first often in UTF-8 and the
            // second always in the console codepage, so decoding everything as
            // the OEM page mangles the UTF-8 half exactly as decoding
            // everything as UTF-8 mangled the legacy half. Native Windows had
            // this backwards until 2026-10-11: the run-by-run path was reached
            // only OFF Windows, where the OEM page cannot be asked for, and
            // here the whole-buffer OEM decode turned pktmon's UTF-8 Korean
            // into "�쒕씪�대쾭��" while cmd.exe's CP949 came through.
            //
            // The OEM page is known here, so it is used instead of the
            // configured legacy encoding: it is what the console actually
            // wrote, and it needs no user setting to be right.
            return decode_runs(buf, enc);
        }
    }

    decode_mixed_console_output(buf)
}

/// Decode a console stream that is UTF-8 in places and a legacy CJK codepage in
/// others, run by run.
///
/// This is what a Windows program invoked from WSL produces, and the two halves
/// arrive interleaved in ONE command's output. Reported 2026-09-07 running
/// pktmon through cmd.exe from Ubuntu-24.04: pktmon writes UTF-8 and its Korean
/// came through intact, while cmd.exe's own messages are in the console
/// codepage (949 on a Korean install) and arrived as
/// "���� ��θ� ���� ���͸���".
///
/// The old code could not do better anywhere but on Windows itself: the OEM
/// codepage branch above is `cfg(windows)`, and codemason under WSL is a LINUX
/// binary, so it compiled out and every non-UTF-8 byte became U+FFFD.
///
/// Guessing is still refused. A run that is not valid UTF-8 is decoded with the
/// configured legacy encoding ONLY if the result is plausible text — real CJK
/// or ASCII, no control characters and no replacement characters. Otherwise the
/// bytes were binary, not misencoded text, and they keep the lossy form they
/// had. So `cat` of a JPEG still looks like a JPEG rather than like hangul.
fn decode_mixed_console_output(buf: &[u8]) -> String {
    let legacy = {
        let label = crate::utils::config::get_config().grep_legacy_encoding;
        let label = label.trim();
        if label.is_empty() {
            None
        } else {
            encoding_rs::Encoding::for_label(label.as_bytes())
        }
    };
    let Some(legacy) = legacy else {
        return String::from_utf8_lossy(buf).into_owned();
    };
    decode_runs(buf, legacy)
}

/// The run-by-run decode itself, over a legacy encoding the caller has already
/// settled on: the OEM codepage on Windows, the configured one everywhere else.
fn decode_runs(buf: &[u8], legacy: &'static encoding_rs::Encoding) -> String {
    let mut out = String::with_capacity(buf.len());
    let mut rest = buf;
    loop {
        match std::str::from_utf8(rest) {
            Ok(s) => {
                out.push_str(s);
                return out;
            }
            Err(e) => {
                // Everything before the bad byte is real UTF-8; keep it exactly.
                out.push_str(unsafe { std::str::from_utf8_unchecked(&rest[..e.valid_up_to()]) });
                let bad_from = e.valid_up_to();
                // The invalid run reaches to the next byte that starts a valid
                // UTF-8 sequence — decoding the whole run at once is what lets a
                // multi-byte legacy character survive.
                let mut end = bad_from + 1;
                while end < rest.len() && std::str::from_utf8(&rest[end..]).is_err() {
                    // Advance to the first position whose tail parses; a shorter
                    // tail can only become valid, never less valid.
                    end += 1;
                }
                let run = &rest[bad_from..end];
                let (decoded, _, had_errors) = legacy.decode(run);
                let plausible = !had_errors
                    && !decoded.is_empty()
                    && !decoded.chars().any(|c| {
                        c == '\u{fffd}' || (c.is_control() && c != '\n' && c != '\r' && c != '\t')
                    });
                if plausible {
                    out.push_str(&decoded);
                } else {
                    out.push_str(&String::from_utf8_lossy(run));
                }
                rest = &rest[end..];
            }
        }
    }
}

/// Kill `pid`'s process tree (POSIX killpg equivalent).
///
/// POSIX: kill(-getpgid(pid), sig). Windows: taskkill /F /T (tree kill), then
/// a direct kill as fallback. Swallows the races the callers already tolerate
/// (already-exited / no-permission) — killing is best-effort; never let
/// cleanup crash the caller.
pub fn kill_process_tree(pid: u32, sig: Option<i32>) {
    if is_windows() {
        let _ = std::process::Command::new(system32_exe("taskkill.exe"))
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        #[cfg(windows)]
        {
            // proc.kill() fallback: TerminateProcess via taskkill already
            // covered the tree; a direct handle kill is done by the caller
            // holding the Child. Nothing further here.
        }
        return;
    }
    #[cfg(unix)]
    {
        let sig = sig.unwrap_or(libc::SIGKILL);
        unsafe {
            let pgid = libc::getpgid(pid as libc::pid_t);
            if pgid > 0 && libc::killpg(pgid, sig) == 0 {
                return;
            }
            // Broad fallback: killpg was denied/failed — try the process alone.
            let _ = libc::kill(pid as libc::pid_t, sig);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = sig;
    }
}

/// Clear memoized git-bash discovery (tests only).
pub fn reset_cache() {
    *GIT_BASH_CACHE.lock() = (false, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The override is ours alone. Another program's namespace must never
    /// configure this one — it sets its own variables in every process it
    /// spawns, and codemason is routinely launched from it.
    #[test]
    fn git_bash_override_is_read_from_our_own_namespace_only() {
        // The message tells the user which variable to set, so it is the one
        // place a foreign spelling would keep circulating. Counting rather than
        // naming: every mention of the variable must be the CODEMASON_ one, so
        // this fails on ANY other prefix without hard-coding one.
        let total = GIT_BASH_MISSING_MSG.matches("GIT_BASH_PATH").count();
        let ours = GIT_BASH_MISSING_MSG.matches("CODEMASON_GIT_BASH_PATH").count();
        assert!(ours > 0, "the message must name the variable to set");
        assert_eq!(ours, total, "every mention must carry our own prefix");
    }

    #[test]
    fn path_conversion_roundtrip() {
        assert_eq!(windows_path_to_posix(r"C:\Users\foo"), "/c/Users/foo");
        assert_eq!(windows_path_to_posix(r"\\server\share"), "//server/share");
        assert_eq!(posix_path_to_windows("/c/Users/foo"), r"C:\Users\foo");
        assert_eq!(posix_path_to_windows("//server/share"), r"\\server\share");
        assert_eq!(posix_path_to_windows("/cygdrive/d/x"), r"D:\x");
        assert_eq!(posix_path_to_windows("/c"), "C:\\");
    }

    #[test]
    fn exe_ext_detection() {
        assert!(has_exe_ext(r"C:\Git\cmd\git.exe"));
        assert!(has_exe_ext(r"C:\x\a.CMD"));
        assert!(!has_exe_ext(r"C:\x\git"));
        assert!(!has_exe_ext(r"C:\x\.exe")); // dot at pos 0 of basename
        assert!(has_exe_ext(r"C:\x\git.exe. ")); // trailing dot/space stripped → .exe kept
    }

    #[test]
    fn bin_shell_resolution() {
        assert!(resolve_bin_shell(BinShellSpec::Token("bash")).is_err()); // must be absolute
        let b = resolve_bin_shell(BinShellSpec::Token(r"C:\Git\bin\bash.exe")).unwrap();
        assert_eq!(b.args, vec!["-c"]);
        let c = resolve_bin_shell(BinShellSpec::Default).unwrap();
        assert!(c.exe.to_lowercase().ends_with("cmd.exe"));
        assert_eq!(c.args, vec!["/d", "/s", "/c"]);
        assert!(resolve_bin_shell(BinShellSpec::Token("fish")).is_err());
        assert!(resolve_bin_shell(BinShellSpec::Token(r"relative\pwsh.exe")).is_err());
        let p = resolve_bin_shell(BinShellSpec::Token("powershell")).unwrap();
        assert!(p.exe.to_lowercase().ends_with("powershell.exe"));
        assert_eq!(p.args, vec!["-NoProfile", "-Command"]);
    }

    #[test]
    fn nt_helpers() {
        assert!(nt_isabs(r"C:\x"));
        assert!(nt_isabs(r"\\server\share"));
        assert!(!nt_isabs("bash"));
        assert!(!nt_isabs(r"rel\x"));
        assert_eq!(nt_basename(r"C:\a\b.exe"), "b.exe");
        assert_eq!(nt_basename("bash"), "bash");
    }

    /// UTF-8 output must pass through untouched on every platform — git-bash
    /// tools and everything POSIX emit it, and second-guessing that would break
    /// the common case to fix the rare one.
    #[test]
    fn utf8_console_output_is_never_reinterpreted() {
        assert_eq!(decode_console_output("plain ascii".as_bytes()), "plain ascii");
        assert_eq!(decode_console_output("한글 출력".as_bytes()), "한글 출력");
        assert_eq!(decode_console_output(&[]), "");
    }

    /// The codepage-to-encoding mapping, apart from the OS call that feeds it.
    /// Testing it separately is the only way to check the pages this machine
    /// does not use — GetOEMCP reports one number and it is 949 here. It runs
    /// on every platform for the same reason the function compiles on every
    /// platform: a mapping only ever exercised on Windows is a mapping nobody
    /// notices breaking from Linux.
    #[test]
    fn oem_codepages_map_to_the_right_encodings() {
        let label = |cp| encoding_for_codepage(cp).map(|e| e.name());
        assert_eq!(label(949), Some("EUC-KR")); // this machine
        assert_eq!(label(932), Some("Shift_JIS"));
        assert_eq!(label(936), Some("GBK"));
        assert_eq!(label(950), Some("Big5"));
        assert_eq!(label(1252), Some("windows-1252"));
        assert_eq!(label(65001), Some("UTF-8"));

        // The DOS pages a Western install uses are outside the WHATWG set, so
        // they must answer None and leave the caller on its old lossy path
        // rather than silently decoding as something else.
        assert_eq!(label(437), None);
        assert_eq!(label(850), None);
        assert_eq!(label(0), None);
    }

    /// CP949 bytes are not valid UTF-8, so they take the OEM branch. Asserting
    /// the real text comes back — not merely "no U+FFFD" — is the difference
    /// between decoding and mangling quietly.
    #[cfg(windows)]
    #[test]
    fn legacy_console_bytes_decode_when_the_oem_page_is_known() {
        extern "system" {
            fn GetOEMCP() -> u32;
        }
        if unsafe { GetOEMCP() } != 949 {
            return; // a different OEM page: this fixture says nothing
        }
        // "정보:" as tasklist writes it on a Korean install.
        let bytes: &[u8] = &[0xC1, 0xA4, 0xBA, 0xB8, 0x3A];
        assert!(std::str::from_utf8(bytes).is_err(), "fixture must not be UTF-8");
        assert_eq!(decode_console_output(bytes), "정보:");
    }
}

#[cfg(test)]
mod mixed_console_decode_tests {
    use super::*;

    /// A Windows program invoked from WSL can put UTF-8 and the console
    /// codepage in ONE stream. Reported 2026-09-07: pktmon writes UTF-8 and its
    /// Korean arrived intact, while cmd.exe's own messages are CP949 and became
    /// "���� ��θ�". The OEM-codepage branch is cfg(windows) and codemason
    /// under WSL is a Linux binary, so every legacy byte turned into U+FFFD.
    #[test]
    fn utf8_and_cp949_in_one_stream_both_survive() {
        // Off Windows the legacy half is decoded with the CONFIGURED encoding,
        // read from the process-wide config. Pin it, or this test answers for
        // whatever another thread or the developer's config.json left there.
        let _enc = crate::utils::config::pin_legacy_encoding("euc-kr");
        let cp949 = encoding_rs::EUC_KR;
        let (legacy, _, err) = cp949.encode("UNC 경로는 지원되지 않습니다.");
        assert!(!err);

        let mut buf = Vec::new();
        buf.extend_from_slice("PktMon 드라이버와 통신하지 못했습니다.\n".as_bytes());
        buf.extend_from_slice(&legacy);
        buf.extend_from_slice("\netl2pcap 로그 파일을 변환합니다.\n".as_bytes());

        let out = decode_console_output(&buf);
        assert!(out.contains("PktMon 드라이버와 통신하지 못했습니다."), "utf-8 half: {out}");
        assert!(out.contains("UNC 경로는 지원되지 않습니다."), "cp949 half: {out}");
        assert!(out.contains("etl2pcap 로그 파일을 변환합니다."), "trailing utf-8: {out}");
        assert!(!out.contains('\u{fffd}'), "nothing replaced: {out}");
    }

    /// Pure UTF-8 takes the fast path and is returned byte-for-byte.
    #[test]
    fn plain_utf8_is_untouched() {
        let s = "ls: 그런 파일이나 디렉터리가 없습니다\n";
        assert_eq!(decode_console_output(s.as_bytes()), s);
    }

    /// Binary is not text in disguise. Bytes that decode to control characters
    /// must NOT be reinterpreted as hangul — `cat` of a JPEG still looks like a
    /// JPEG.
    #[test]
    fn binary_is_not_guessed_into_text() {
        let buf = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46];
        let out = decode_console_output(&buf);
        assert!(out.contains('\u{fffd}'), "binary keeps its lossy form: {out:?}");
    }
}
