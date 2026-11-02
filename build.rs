//! Bake the build's own identity into the binary.
//!
//! WHY: the banner printed a version string and nothing else, and the version
//! string had not moved in 197 commits. On 2026-10-03 two binaries existed --
//! one from before a day of fixes, one from after -- and both introduced
//! themselves as "CodeMason v2.0.0"; which one was actually running had to be
//! settled with mtime and md5sum from outside the program. A build that cannot
//! say what it is cannot be trusted about anything else it says.
//!
//! The hash answers it without anyone having to remember to bump a number, and
//! the -dirty suffix answers the follow-up question: whether what is running
//! was ever committed.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn main() {
    // Re-run on every build. A path that does not exist is the documented way
    // to ask for that, and it is wanted here: tracking only .git/HEAD would
    // keep the hash correct across commits but let the -dirty flag go stale
    // the moment a source file is edited without one -- which is exactly the
    // state this is meant to report. Two git invocations per build.
    println!("cargo:rerun-if-changed=.git-identity-always-rerun");

    let hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // Exit status only. `git diff --quiet` reports through the exit code and
    // writes NOTHING to stdout, which is the whole point of the flag -- so the
    // helper above, which treats empty output as failure, called every clean
    // tree dirty. Status: 0 clean, 1 dirty.
    let dirty = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--"])
        .status()
        .map(|st| !st.success())
        .unwrap_or(false);
    let hash = if dirty { format!("{hash}-dirty") } else { hash };
    let date = git(&["log", "-1", "--format=%cd", "--date=format:%Y-%m-%d"])
        .unwrap_or_else(|| "unknown".into());

    println!("cargo:rustc-env=CODEMASON_GIT_HASH={hash}");
    println!("cargo:rustc-env=CODEMASON_COMMIT_DATE={date}");
}
