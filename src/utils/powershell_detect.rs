//! PowerShell / cmd detection for CodeMason (Tier 2, native Windows shells).
//! Equivalent to codemason/utils/powershell_detect.py (Claude Code
//! src/utils/shell/powershellDetection.ts). Finds a PowerShell executable
//! (prefers pwsh 7+, falls back to Windows PowerShell 5.1) and infers its
//! edition so the tool prompt can give version-correct syntax guidance.
//!
//! Edition matters: PowerShell 7 (`pwsh`) supports `&&`/`||`/`?:`/`??`;
//! Windows PowerShell 5.1 (`powershell`) does not (using `&&` there is a parse
//! error) and has the stderr-sets-$? quirk plus UTF-16 default output encoding.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};

/// 'core' (pwsh 7+) | 'desktop' (Windows PowerShell 5.1)
pub type PowerShellEdition = &'static str;

static CACHED_PS_PATH: Lazy<Mutex<(bool, Option<String>)>> =
    Lazy::new(|| Mutex::new((false, None)));

/// Return p if it is an existing regular file, else None (probePath).
fn probe_file(p: &str) -> Option<String> {
    if Path::new(p).is_file() {
        Some(p.to_string())
    } else {
        None
    }
}

/// True if `path` lives in (or resolves into) the current working directory.
///
/// Used to reject a planted pwsh/powershell binary. Checks BOTH the lexical
/// location (catches a symlink placed in CWD that points elsewhere) and the
/// realpath (catches a junction/link that resolves back into CWD).
fn is_in_cwd(path: &str) -> bool {
    let Ok(cwd_abs) = std::env::current_dir() else {
        return false;
    };
    let cwd_real = std::fs::canonicalize(&cwd_abs).unwrap_or_else(|_| cwd_abs.clone());
    let abs = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        cwd_abs.join(path)
    };
    let real = std::fs::canonicalize(&abs).unwrap_or_else(|_| abs.clone());
    for base in [&cwd_abs, &cwd_real] {
        for cand in [&abs, &real] {
            if cand == base || cand.starts_with(base) {
                return true;
            }
        }
    }
    false
}

/// A minimal shutil.which(): search PATH for `name` (with PATHEXT variants on
/// Windows, and the CPython win32 quirk of also considering the current
/// directory — which the CWD-rejection below exists to defuse).
fn which(name: &str) -> Option<String> {
    let path_var = std::env::var_os("PATH")?;
    let mut dirs: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        // CPython prepends os.curdir on win32; mirrored so the hardening path
        // exercises the same candidates.
        if let Ok(cwd) = std::env::current_dir() {
            dirs.push(cwd);
        }
    }
    dirs.extend(std::env::split_paths(&path_var));
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
            .split(';')
            .filter(|e| !e.is_empty())
            .map(|e| e.to_lowercase())
            .collect()
    } else {
        vec![]
    };
    for dir in dirs {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand.to_string_lossy().into_owned());
        }
        for ext in &exts {
            let with_ext = dir.join(format!("{name}{ext}"));
            if with_ext.is_file() {
                return Some(with_ext.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// Windows PowerShell resolution, hardened against a CWD hijack.
///
/// A pwsh.exe/powershell.exe (or a PATHEXT variant like pwsh.cmd) planted in
/// the repository the agent operates in would otherwise be resolved FIRST and
/// then spawned with -EncodedCommand on every PowerShell command validation —
/// arbitrary code exec before any read-only check runs. So: prefer the
/// absolute, well-known install locations (unhijackable), and reject any PATH
/// hit that lives inside CWD. Mirrors the absolute-path hardening already in
/// utils/windows_shell.rs.
#[allow(dead_code)]
fn find_powershell_windows() -> Option<String> {
    let program_files =
        std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
    for abs_path in [
        format!(r"{program_files}\PowerShell\7\pwsh.exe"),
        format!(r"{system_root}\System32\WindowsPowerShell\v1.0\powershell.exe"),
    ] {
        if let Some(hit) = probe_file(&abs_path) {
            return Some(hit);
        }
    }
    // Fall back to PATH, but never accept a binary resolved out of CWD.
    for name in ["pwsh", "powershell"] {
        if let Some(hit) = which(name) {
            if !is_in_cwd(&hit) {
                return Some(hit);
            }
        }
    }
    None
}

/// Locate PowerShell: pwsh (7+) preferred, then powershell (5.1) (findPowerShell).
///
/// On Windows, resolution goes through the CWD-hijack-hardened path above. On
/// Linux a snap launcher (/snap/...) can hang in subprocesses while snapd
/// initialises confinement, so if PATH resolves there — directly or via a
/// symlink chain — probe the reliable apt/rpm binary instead. (POSIX PATH
/// lookup does not consider the current directory, so it is not
/// CWD-hijackable.)
pub fn find_powershell() -> Option<String> {
    if cfg!(windows) {
        #[cfg(windows)]
        {
            return find_powershell_windows();
        }
    }

    if let Some(pwsh) = which("pwsh") {
        if cfg!(target_os = "linux") {
            let resolved = std::fs::canonicalize(&pwsh)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| pwsh.clone());
            if pwsh.starts_with("/snap/") || resolved.starts_with("/snap/") {
                let direct =
                    probe_file("/opt/microsoft/powershell/7/pwsh").or_else(|| probe_file("/usr/bin/pwsh"));
                if let Some(direct) = direct {
                    let direct_resolved = std::fs::canonicalize(&direct)
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| direct.clone());
                    if !direct.starts_with("/snap/") && !direct_resolved.starts_with("/snap/") {
                        return Some(direct);
                    }
                }
            }
        }
        return Some(pwsh);
    }

    which("powershell")
}

/// Memoized find_powershell() (getCachedPowerShellPath).
pub fn get_cached_powershell_path() -> Option<String> {
    {
        let cache = CACHED_PS_PATH.lock();
        if cache.0 {
            return cache.1.clone();
        }
    }
    let result = find_powershell();
    let mut cache = CACHED_PS_PATH.lock();
    if !cache.0 {
        cache.0 = true;
        cache.1 = result.clone();
    }
    cache.1.clone()
}

/// Infer edition from the binary name without spawning (getPowerShellEdition).
///
/// pwsh/pwsh.exe → 'core' (7+); powershell/powershell.exe → 'desktop' (5.1).
/// None if no PowerShell found.
pub fn get_powershell_edition() -> Option<PowerShellEdition> {
    let p = get_cached_powershell_path()?;
    let mut base = p
        .replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_lowercase();
    if base.ends_with(".exe") {
        base.truncate(base.len() - 4);
    }
    Some(if base == "pwsh" { "core" } else { "desktop" })
}

/// Clear the memoized path (tests only).
pub fn reset_powershell_cache() {
    *CACHED_PS_PATH.lock() = (false, None);
}
