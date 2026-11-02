//! ripgrep binary resolution.
//! Equivalent to codemason/utils/ripgrep.py.
//!
//! The Glob/Grep tools shell out to `rg`. The original ships a per-platform
//! binary under vendor/ripgrep/<arch-os>/rg[.exe] and resolves it as:
//!
//!     system `rg` on PATH  →  else the bundled vendored binary  →  else None
//!
//! Resolving system-first (by the bare name 'rg', not an absolute path)
//! matches the original's PATH-hijacking guard; the vendored binary is the
//! fallback so a machine without ripgrep installed still gets the fast path
//! instead of collapsing to the slow walk fallback.

use once_cell::sync::Lazy;

/// Map the current platform to a vendor/<tool>/<arch-os> folder name.
/// None on an unmapped platform (=> no vendored fallback).
pub(crate) fn vendor_subdir() -> Option<String> {
    let arch = if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        return None;
    };

    if cfg!(target_os = "windows") {
        Some(format!("{arch}-win32"))
    } else if cfg!(target_os = "linux") {
        Some(format!("{arch}-linux"))
    } else if cfg!(target_os = "macos") {
        Some(format!("{arch}-darwin"))
    } else {
        None
    }
}

/// The directory holding this executable, with symlinks resolved.
///
/// Vendored binaries are found relative to the real executable, so the search
/// has to start from the real executable. On Linux `current_exe()` reads
/// /proc/self/exe, which already resolves the link, and the user's
/// `~/.local/bin/cm2 -> target/release/codemason` therefore finds vendor/ two
/// directories up. On Windows `current_exe()` hands back the path the process
/// was STARTED with, so the same arrangement searches beside the LINK and the
/// bundled binary is invisible. Measured on this platform:
///
///     direct       current_exe() = ...\exeprobe.exe
///     via symlink  current_exe() = ...\probelink.exe   <- the link itself
///                  canonicalize  = ...\exeprobe.exe    <- the target
///
/// Canonicalizing first makes both platforms search from the same place. The
/// verbatim prefix that canonicalize adds on Windows is stripped so the paths
/// built from this stay in their plain form. If canonicalize fails — a deleted
/// or replaced image — the unresolved path is still better than giving up.
pub(crate) fn exe_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let real = match std::fs::canonicalize(&exe) {
        #[cfg(windows)]
        Ok(p) => crate::utils::state::strip_verbatim_prefix(p),
        #[cfg(not(windows))]
        Ok(p) => p,
        Err(_) => exe,
    };
    real.parent().map(|p| p.to_path_buf())
}

/// shutil.which equivalent: is `name` executable somewhere on PATH?
pub(crate) fn which(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    for dir in std::env::split_paths(&paths) {
        let cand = dir.join(name);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&cand) {
                if meta.is_file() && meta.permissions().mode() & 0o111 != 0 {
                    return true;
                }
            }
        }
        #[cfg(windows)]
        {
            for ext in ["", ".exe", ".cmd", ".bat"] {
                let c = dir.join(format!("{name}{ext}"));
                if c.is_file() {
                    return true;
                }
            }
            let _ = cand;
        }
    }
    false
}

static RESOLVED: Lazy<Option<String>> = Lazy::new(|| {
    // 1. system `rg` on PATH (returned as the bare name 'rg' so the OS
    //    resolves it safely, matching Claude Code's PATH-hijacking guard);
    if which("rg") {
        return Some("rg".to_string());
    }

    // 2. else the bundled vendored binary for this platform. The Python
    //    package resolved vendor/ relative to the package dir; the binary
    //    equivalent is the executable's directory (substrate divergence,
    //    matching the bundled-tokenizer resolution in utils/tokens.rs).
    if let Some(sub) = vendor_subdir() {
        let exe_name = if sub.ends_with("-win32") { "rg.exe" } else { "rg" };
        if let Some(dir) = exe_dir() {
            let dir = dir.as_path();
            for root in [Some(dir), dir.parent(), dir.parent().and_then(|p| p.parent())]
                .into_iter()
                .flatten()
            {
                let cand = root.join("vendor").join("ripgrep").join(&sub).join(exe_name);
                if cand.is_file() {
                    return Some(cand.to_string_lossy().into_owned());
                }
            }
        }
    }

    // 3. else None -> caller falls back to its native walk path.
    None
});

/// Path to an rg executable, or None if none is available (memoized).
pub fn resolve_ripgrep() -> Option<String> {
    RESOLVED.clone()
}
