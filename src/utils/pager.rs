//! Pager binary resolution, on the shape utils/ripgrep.rs uses.
//!
//!     system `less` on PATH  →  git-bash's own less  →  beside the
//!     executable  →  the vendored binary  →  None
//!
//! Ctrl+O writes every recent tool output to a temp file and hands it to a
//! pager. That assumed `less`, which is fine on Linux and macOS and false on
//! Windows: `less` is not a Windows program, and whether it resolves depends
//! entirely on which shell started codemason. Measured on a stock machine —
//! from Git Bash `less` is found (it ships in Git's usr/bin), from PowerShell
//! or cmd.exe it is not on PATH at all. The same keystroke therefore either
//! paged or dumped every stored tool output straight into the scrollback, and
//! the dump still carried the document's first line telling the reader to press
//! `q` — with no pager running to receive it.
//!
//! The fix is to USE the less that is already there rather than ship a second
//! copy of it. On Windows the Bash tool requires git-bash, and Git for Windows
//! puts less.exe in its usr/bin, so any machine that can run the Bash tool at
//! all already has this pager — the step below just looks where we already
//! found bash. Falling back to `more.exe` was the other option and is worse:
//! `more` has no `/` search and no `g/G`, so the key hints would stop being
//! true and the hint string would need a Windows variant instead of staying
//! byte-identical to the Python oracle.
//!
//! The vendor step is kept for a platform where neither holds. No binary is
//! vendored today, so it is a miss that costs nothing.

use once_cell::sync::Lazy;

use crate::utils::ripgrep::{exe_dir, vendor_subdir, which};

static RESOLVED: Lazy<Option<String>> = Lazy::new(|| {
    // 1. A system `less` on PATH wins, returned as the bare name so the OS
    //    resolves it — the same PATH-hijacking posture as ripgrep resolution.
    //    Someone who installed their own less gets theirs.
    if which("less") {
        return Some("less".to_string());
    }

    // 2. else the less that git-bash already brought. On Windows the Bash tool
    //    REQUIRES git-bash, and Git for Windows installs less.exe in its
    //    usr/bin — so a machine that can run the Bash tool at all already has a
    //    pager, it is just not on PATH when codemason was started from
    //    PowerShell or cmd.exe. Using it is not a new dependency, it is the
    //    consequence of one we already impose, which beats vendoring a second
    //    copy of the same binary.
    //
    //    Measured on Windows from both PowerShell and cmd.exe: it pages, the
    //    keys the hint advertises all work, and it turns the console's VT mode
    //    on for -R colour and puts it back on exit.
    if let Some(less) = less_from_git_bash() {
        return Some(less);
    }

    // 3. else a copy sitting in codemason's own install directory. An
    //    installer that "copies the binary next to the executable" lands here;
    //    one that puts its directory on PATH is already handled above.
    //
    //    The EXECUTABLE's directory, never the working directory: the working
    //    directory is a repo the user asked codemason to open, so honouring a
    //    less.exe found there would run a binary that arrived with the
    //    checkout. The install directory is as trusted as codemason itself.
    let exe_name = if cfg!(target_os = "windows") { "less.exe" } else { "less" };
    if let Some(dir) = exe_dir() {
        let beside = dir.join(exe_name);
        if beside.is_file() {
            return Some(beside.to_string_lossy().into_owned());
        }
    }

    // 4. else the vendored per-platform layout, for a checkout or an installer
    //    that ships the whole vendor tree. Same shape ripgrep resolution uses,
    //    searched beside the executable and two directories up (a cargo
    //    target/release layout puts the repo root two above).
    if let Some(sub) = vendor_subdir() {
        if let Some(dir) = exe_dir() {
            let dir = dir.as_path();
            for root in [Some(dir), dir.parent(), dir.parent().and_then(|p| p.parent())]
                .into_iter()
                .flatten()
            {
                let cand = root.join("vendor").join("less").join(&sub).join(exe_name);
                if cand.is_file() {
                    return Some(cand.to_string_lossy().into_owned());
                }
            }
        }
    }

    // 5. else None -> the caller prints the document instead of paging it.
    None
});

/// `<git>/usr/bin/less[.exe]` for a `<git>/bin/bash.exe`, if that file exists.
///
/// Split out from the lookup so the derivation is testable on Linux: the
/// Windows-only half is FINDING bash, and a `#[cfg(windows)]` body would never
/// be type-checked here — which is exactly how a signature drifted and broke
/// the Windows build once already.
fn derive_less_path(bash: &std::path::Path) -> Option<std::path::PathBuf> {
    let git_root = bash.parent()?.parent()?;
    let exe_name = if cfg!(target_os = "windows") { "less.exe" } else { "less" };
    let cand = git_root.join("usr").join("bin").join(exe_name);
    cand.is_file().then_some(cand)
}

/// The pager that came with git-bash, if git-bash is where we expect it.
///
/// find_git_bash_path resolves three ways (CODEMASON_GIT_BASH_PATH, the known
/// install paths, a path derived from git.exe) and every one of them ends at
/// `...\bin\bash.exe`, so the two-level climb holds for all three. The
/// override is the one that can point somewhere arbitrary, which is why this
/// takes the result only when the derived file actually exists and otherwise
/// falls through to the next step.
fn less_from_git_bash() -> Option<String> {
    let bash = crate::utils::windows_shell::find_git_bash_path()?;
    derive_less_path(std::path::Path::new(&bash)).map(|p| p.to_string_lossy().into_owned())
}

/// Path to a `less` executable, or None if none is available (memoized).
pub fn resolve_pager() -> Option<String> {
    RESOLVED.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolution must never panic and must answer the same way every time —
    /// it is consulted on a keystroke, and a second answer that disagreed with
    /// the first would page once and dump the next time.
    #[test]
    fn resolution_is_stable() {
        let a = resolve_pager();
        let b = resolve_pager();
        assert_eq!(a, b);
        if let Some(p) = a {
            assert!(!p.is_empty());
            // Either the bare name for a PATH lookup, or a real file on disk.
            assert!(p == "less" || std::path::Path::new(&p).is_file(), "{p} should exist");
        }
    }

    /// The two-level climb from bash.exe, which is the whole Windows fix, has
    /// to be exercised somewhere that actually compiles and runs it — hence a
    /// fake tree rather than a real install, so this runs on Linux too.
    #[test]
    fn less_is_derived_from_the_git_bash_path() {
        let tmp = std::env::temp_dir().join(format!("cm-pager-{}", std::process::id()));
        let bin = tmp.join("bin");
        let usr_bin = tmp.join("usr").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&usr_bin).unwrap();
        let bash = bin.join(if cfg!(windows) { "bash.exe" } else { "bash" });
        std::fs::write(&bash, b"").unwrap();
        let less = usr_bin.join(if cfg!(windows) { "less.exe" } else { "less" });

        // Absent: the step must decline so resolution falls through, rather
        // than returning a path that is not there.
        assert_eq!(derive_less_path(&bash), None, "no less yet, must not claim one");

        std::fs::write(&less, b"").unwrap();
        assert_eq!(derive_less_path(&bash), Some(less), "must find git's own less");

        // A bash too shallow to climb from must not panic.
        assert_eq!(derive_less_path(std::path::Path::new("bash")), None);

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The vendored layout this looks for has to match what ripgrep uses, so a
    /// single vendor/ convention covers both.
    #[test]
    fn vendor_layout_matches_ripgrep() {
        if let Some(sub) = vendor_subdir() {
            assert!(
                sub.ends_with("-win32") || sub.ends_with("-linux") || sub.ends_with("-darwin"),
                "unexpected vendor subdir {sub}"
            );
            assert!(sub.starts_with("x64-") || sub.starts_with("arm64-"));
        }
    }
}
