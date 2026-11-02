//! State management for CodeMason
//! Equivalent to codemason/utils/state.py (src/utils/state.ts)

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct GlobalState {
    cwd: Option<String>,
    original_cwd: Option<String>,
    session_id: Option<String>,
}

static STATE: Lazy<Mutex<GlobalState>> = Lazy::new(|| Mutex::new(GlobalState::default()));

/// Get the current session id (bootstrap/state.ts getSessionId). Generated
/// lazily on first access; stable for the process lifetime unless switched by
/// resume ('clear' / --resume both go through set_session_id).
pub fn get_session_id() -> String {
    let mut s = STATE.lock();
    if s.session_id.is_none() {
        s.session_id = Some(uuid::Uuid::new_v4().to_string());
    }
    s.session_id.clone().unwrap()
}

/// Switch the active session id (switchActiveSession subset). Callers that
/// change the id must also reset the session-storage file pointer — use
/// session_storage::switch_session() rather than calling this directly.
pub fn set_session_id(session_id: &str) {
    STATE.lock().session_id = Some(session_id.to_string());
}

/// Per-thread tally of get_cwd() calls, for tests that need to prove a function
/// does NOT consult the session directory.
///
/// Thread-local on purpose. The obvious alternative — a test that MOVES the cwd
/// and checks the output did not change — mutates process-global state, and the
/// suite runs in parallel: measured on Windows, widening that probe's window to
/// 400ms turned two unrelated path-permission tests red 6 runs out of 6, with
/// "git log should not require approval" as the message. Only the microsecond
/// width of the window was keeping it green. A global counter would race the
/// same way; a thread-local one cannot, because libtest gives each test its own
/// thread and no other test's reads land on it.
#[cfg(test)]
thread_local! {
    static CWD_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many times THIS THREAD has called get_cwd(). Test-only.
#[cfg(test)]
pub fn cwd_reads() -> u64 {
    CWD_READS.with(|c| c.get())
}

/// Get current working directory.
pub fn get_cwd() -> String {
    #[cfg(test)]
    CWD_READS.with(|c| c.set(c.get() + 1));
    let mut s = STATE.lock();
    if s.cwd.is_none() {
        s.cwd = Some(
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".into()),
        );
    }
    s.cwd.clone().unwrap()
}

/// Set current working directory.
pub fn set_cwd(path: &str) {
    STATE.lock().cwd = Some(path.to_string());
}

/// Get original working directory (from startup).
pub fn get_original_cwd() -> String {
    let mut s = STATE.lock();
    if s.original_cwd.is_none() {
        s.original_cwd = Some(
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".into()),
        );
    }
    s.original_cwd.clone().unwrap()
}

/// Set original working directory.
pub fn set_original_cwd(path: &str) {
    STATE.lock().original_cwd = Some(path.to_string());
}

/// Reset all state to defaults.
pub fn reset_state() {
    *STATE.lock() = GlobalState::default();
}

/// Drop Windows' extended-length ("verbatim") prefix from a canonicalized path.
///
/// DIVERGENCE FIX (Windows): `std::fs::canonicalize` returns `\\?\C:\dir`,
/// while Python's `Path.resolve()` — the oracle for `resolve` below — returns
/// the plain `C:\dir`. The prefix only appears when the path EXISTS, so a
/// canonicalized base directory never compared equal to the lexically
/// normalized form that `resolve`'s fallback produces for a path that does not
/// exist yet. Every containment check on a not-yet-created file inside the
/// session directory therefore answered "outside" on Windows.
///
/// Returns the path untouched when it is not valid UTF-8, rather than going
/// through `to_string_lossy` and corrupting it.
#[cfg(windows)]
pub(crate) fn strip_verbatim_prefix(p: PathBuf) -> PathBuf {
    let Some(s) = p.to_str() else { return p };
    // UNC share: \\?\UNC\server\share -> \\server\share
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    // Drive path: \\?\C:\dir -> C:\dir. Only a real drive letter qualifies;
    // \\?\Volume{...} has no plain equivalent and is left alone.
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        let b = rest.as_bytes();
        if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
            return PathBuf::from(rest);
        }
    }
    p
}

/// Strip the verbatim prefix on Windows; identity elsewhere.
fn plain(p: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        strip_verbatim_prefix(p)
    }
    #[cfg(not(windows))]
    {
        p
    }
}

fn resolve(path: &Path) -> PathBuf {
    // Python Path.resolve(): absolutize against cwd and normalize, resolving
    // symlinks (and, on Windows, 8.3 short names) as far as the path exists.
    if let Ok(p) = std::fs::canonicalize(path) {
        return plain(p);
    }

    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
    };
    // Lexical normalization: fold '.' and '..'.
    let mut folded = PathBuf::new();
    for comp in abs.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                folded.pop();
            }
            c => folded.push(c.as_os_str()),
        }
    }

    // canonicalize only succeeds when the WHOLE path exists, and the common
    // case here is a file about to be created. Pure lexical folding then keeps
    // whatever spelling the caller used, while an existing base directory
    // canonicalizes to its real one — so the two never compared equal.
    //
    // On Windows that gap is a hole, not just noise: `C:\Users\u\CODEMA~1` is
    // the 8.3 alias of `C:\Users\u\.codemason`, and a not-yet-existing file
    // named through the alias was judged OUTSIDE the config directory. The
    // config-dir guard exists because files there are executed (settings.json
    // Stop hooks, goal-gates.txt), so the alias bypassed it. A symlink or
    // junction over the parent does the same thing on either platform.
    //
    // Python has no such gap — Path.resolve() is non-strict and resolves the
    // existing prefix, so `CODEMA~1\zz_new.txt` really does come back as
    // `.codemason\zz_new.txt` (verified against the oracle on this platform).
    // Match it: canonicalize the deepest ancestor that exists, then re-attach
    // the components that do not.
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = folded.as_path();
    while let (Some(name), Some(parent)) = (cur.file_name(), cur.parent()) {
        if parent.as_os_str().is_empty() {
            break;
        }
        tail.push(name.to_os_string());
        cur = parent;
        if let Ok(base) = std::fs::canonicalize(cur) {
            let mut out = plain(base);
            for t in tail.iter().rev() {
                out.push(t);
            }
            return out;
        }
    }
    folded
}

/// Anchor a path for a containment check the way the platform's shell will
/// resolve it.
///
/// POSIX: unchanged — an absolute path stands, anything else hangs off `cwd`.
///
/// DIVERGENCE FIX (Windows): `/etc/hosts` carries no drive letter, so both
/// Rust's `Path::is_absolute` and Python 3.13's `ntpath.isabs` call it
/// relative, and the validators pasted it onto the session cwd. That put it at
/// `C:\session\etc\hosts` — INSIDE the session directory, so the write was
/// approved. PowerShell resolves a leading slash against the CURRENT DRIVE
/// ROOT instead: `[System.IO.Path]::GetFullPath('/etc/hosts')` and the
/// PowerShell provider both answer `C:\etc\hosts`, which is outside. A
/// validator that locates the file somewhere the shell will not touch is
/// green-lighting a write it never inspected, so anchor a rooted-but-driveless
/// path to the cwd's drive and let the containment check see the real target.
pub fn resolve_for_containment(path: &str, cwd: &str) -> String {
    if Path::new(path).is_absolute() {
        return path.to_string();
    }
    #[cfg(windows)]
    {
        let b = path.as_bytes();
        let rooted = matches!(b.first(), Some(b'/') | Some(b'\\'));
        // A UNC path (\\server\share, //server/share) is already absolute in
        // every practical sense and must not gain a drive letter. Leave it for
        // the containment check, which will place it outside the session dir.
        let unc = b.len() > 1 && matches!(b[1], b'/' | b'\\');
        if rooted && !unc {
            if let Some(std::path::Component::Prefix(p)) = Path::new(cwd).components().next() {
                return format!("{}{path}", p.as_os_str().to_string_lossy());
            }
        }
    }
    format!("{cwd}/{path}")
}

/// One canonical spelling for a path, for use as a stable identity.
///
/// Same normalization the containment check runs on its inputs, exposed for
/// callers that need to compare two paths for "is this the same file" rather
/// than "is this inside that". Case is deliberately NOT folded: containment
/// folds it because not folding would wrongly report a path as outside, while
/// an identity used to decide what an approval covers must not quietly grow.
pub fn canonical_spelling(path: &str) -> String {
    resolve(Path::new(path)).to_string_lossy().into_owned()
}

/// Check if path is within base directory (inclusive of base itself).
///
/// Uses path-segment containment, NOT string startswith — the latter wrongly
/// matches sibling prefixes (e.g. treating '/home/user/work-evil' as inside
/// '/home/user/work'), which is the exact containment bypass
/// pathInAllowedWorkingPath guards against.
pub fn is_in_directory(path: &str, base_dir: &str) -> bool {
    let path_resolved = resolve(Path::new(path));
    let base_resolved = resolve(Path::new(base_dir));
    // Windows filesystems are case-insensitive while `starts_with` compares
    // segments byte-for-byte, so a cwd spelled `C:\Root\Proj` and a target
    // spelled `c:\root\proj\f.txt` name the same place but failed containment.
    // Fold case for the comparison ONLY; the reported path keeps its spelling.
    // Segment-wise comparison is preserved by folding whole components, never
    // by lowercasing the joined strings and calling `str::starts_with`.
    #[cfg(windows)]
    {
        let fold = |p: &Path| -> Vec<String> {
            p.components()
                .map(|c| c.as_os_str().to_string_lossy().to_lowercase())
                .collect()
        };
        let p = fold(&path_resolved);
        let b = fold(&base_resolved);
        return p.len() >= b.len() && p[..b.len()] == b[..];
    }
    #[cfg(not(windows))]
    {
        path_resolved == base_resolved || path_resolved.starts_with(&base_resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The base directory exists (so canonicalize succeeds for it) while the
    /// target file does not (so only the lexical fallback runs). On Windows
    /// those two paths used to come back in different spellings — `\\?\C:\dir`
    /// vs `C:\dir` — and containment answered false for a file plainly inside
    /// the directory. Every not-yet-created file in the session directory hit
    /// this, so writes asked for permission they should not have needed.
    #[test]
    fn missing_file_inside_existing_dir_is_contained() {
        let base = std::env::temp_dir().join("codemason_is_in_directory_test");
        std::fs::create_dir_all(&base).expect("create temp base dir");
        let base_s = base.to_string_lossy().into_owned();

        let missing = base.join("does_not_exist_yet.txt");
        assert!(
            is_in_directory(&missing.to_string_lossy(), &base_s),
            "a missing file directly inside the base must be contained"
        );

        let nested = base.join("sub").join("deeper").join("new.txt");
        assert!(
            is_in_directory(&nested.to_string_lossy(), &base_s),
            "a missing file in a missing subdirectory must be contained"
        );

        // The base itself, and an existing child, still pass.
        assert!(is_in_directory(&base_s, &base_s));
        let existing = base.join("exists.txt");
        std::fs::write(&existing, b"x").expect("write temp file");
        assert!(is_in_directory(&existing.to_string_lossy(), &base_s));

        // Containment must still REJECT an outside path, and must not be
        // fooled by a sibling that merely shares the base's string prefix.
        let outside = std::env::temp_dir().join("codemason_elsewhere.txt");
        assert!(!is_in_directory(&outside.to_string_lossy(), &base_s));
        let sibling_prefix = std::env::temp_dir()
            .join("codemason_is_in_directory_test-evil")
            .join("f.txt");
        assert!(
            !is_in_directory(&sibling_prefix.to_string_lossy(), &base_s),
            "a sibling sharing the base's string prefix must NOT be contained"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Windows keeps an 8.3 alias for a long directory name, so `CODEMA~1` and
    /// `.codemason` can be one directory. `canonicalize` sees through the alias
    /// only when the WHOLE path exists, so a file about to be created kept the
    /// alias spelling while its existing parent resolved to the real name, and
    /// containment answered "outside". Anything guarding a directory by
    /// containment — the config directory whose files get executed, the session
    /// directory the shell tools police — was bypassable by naming the target
    /// through the alias. Python's non-strict `Path.resolve()` has no such gap.
    /// The POSIX half of the same change. Resolving the existing prefix of a
    /// not-yet-created path cuts both ways here, and both directions matter:
    ///
    ///   - a symlink INSIDE the session pointing out of it can no longer launder
    ///     a write past the containment check by naming a file that does not
    ///     exist yet (previously the lexical fallback kept the in-session
    ///     spelling and answered "inside");
    ///   - a session directory REACHED through a symlink no longer reports its
    ///     own not-yet-created files as outside, because base and target now
    ///     resolve the same way.
    ///
    /// The second case was a live annoyance, not a hypothetical: the base
    /// existed and canonicalized, the missing file did not and stayed lexical,
    /// so they could never compare equal.
    #[cfg(unix)]
    #[test]
    fn containment_resolves_symlinks_for_missing_paths() {
        let root = std::env::temp_dir().join(format!("cm_symlink_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let session = root.join("session");
        let outside = root.join("outside");
        std::fs::create_dir_all(&session).expect("session dir");
        std::fs::create_dir_all(&outside).expect("outside dir");

        // An escape hatch planted inside the session directory.
        let escape = session.join("out");
        std::os::unix::fs::symlink(&outside, &escape).expect("symlink");

        let session_s = session.to_string_lossy().into_owned();
        let missing_via_link = escape.join("payload.sh");
        assert!(
            !is_in_directory(&missing_via_link.to_string_lossy(), &session_s),
            "a not-yet-created file behind an out-of-session symlink must be OUTSIDE"
        );
        // An ordinary missing file in the session is still inside.
        assert!(is_in_directory(
            &session.join("notes.md").to_string_lossy(),
            &session_s
        ));

        // Session reached through a symlink: its own missing files are inside.
        let alias = root.join("alias");
        std::os::unix::fs::symlink(&session, &alias).expect("alias symlink");
        assert!(
            is_in_directory(&alias.join("new.txt").to_string_lossy(), &alias.to_string_lossy()),
            "a session named through a symlink must contain its own new files"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn containment_sees_through_short_names() {
        // A name long enough to be given an 8.3 alias.
        let base = std::env::temp_dir().join("codemason_shortname_regression_dir");
        std::fs::create_dir_all(&base).expect("create temp base dir");
        let base_s = base.to_string_lossy().into_owned();

        let Some(short) = short_path_name(&base_s) else {
            let _ = std::fs::remove_dir_all(&base);
            return; // volume has 8.3 generation disabled: nothing to assert
        };
        if short.eq_ignore_ascii_case(&base_s) {
            let _ = std::fs::remove_dir_all(&base);
            return;
        }

        assert!(is_in_directory(&short, &base_s), "{short} IS {base_s}");
        assert!(
            is_in_directory(&format!(r"{short}\not_created_yet.txt"), &base_s),
            "a new file named through the 8.3 alias must still be inside"
        );
        assert!(
            is_in_directory(&format!(r"{short}\sub\deeper\new.txt"), &base_s),
            "missing intermediate directories must not reopen the gap"
        );
        // The alias must not drag in a sibling.
        let evil = std::env::temp_dir().join("codemason_shortname_regression_dir-evil");
        assert!(!is_in_directory(&format!("{}\\f.txt", evil.display()), &base_s));

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The OS's own answer for a path's 8.3 alias; None when it cannot give one.
    #[cfg(windows)]
    fn short_path_name(path: &str) -> Option<String> {
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &format!(
                    "(New-Object -ComObject Scripting.FileSystemObject).GetFolder('{path}').ShortPath"
                ),
            ])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() || s.contains('\n') {
            None
        } else {
            Some(s)
        }
    }

    /// Windows filesystems ignore case; `starts_with` does not. A cwd and a
    /// target that differ only in case name the same location and must be
    /// judged contained.
    #[cfg(windows)]
    #[test]
    fn containment_ignores_case_on_windows() {
        let base = std::env::temp_dir().join("codemason_case_test");
        std::fs::create_dir_all(&base).expect("create temp base dir");
        let base_s = base.to_string_lossy().into_owned();

        let upper = base_s.to_uppercase();
        let lower = base_s.to_lowercase();
        assert!(is_in_directory(&format!(r"{upper}\new.txt"), &base_s));
        assert!(is_in_directory(&format!(r"{lower}\new.txt"), &base_s));
        assert!(is_in_directory(&format!(r"{base_s}\new.txt"), &upper));

        // Case folding must not soften the sibling-prefix guard.
        let evil = format!("{}-EVIL", base_s.to_uppercase());
        assert!(
            !is_in_directory(&format!(r"{evil}\f.txt"), &base_s),
            "case folding must not let a prefix sibling in"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// A rooted-but-driveless path is what the shell puts at the drive root,
    /// not inside the session directory. Anchoring it to the cwd would place
    /// `/etc/hosts` at `<cwd>\etc\hosts` and report it as contained, which is
    /// how a write outside the session directory got approved.
    #[test]
    fn rooted_driveless_path_is_not_pulled_into_cwd() {
        #[cfg(windows)]
        {
            let cwd = r"C:\root\project";
            let got = resolve_for_containment("/etc/hosts", cwd);
            assert_eq!(got, r"C:/etc/hosts", "must anchor to the drive root");
            assert!(
                !is_in_directory(&got, cwd),
                "{got} must NOT be inside {cwd}"
            );

            // A backslash-rooted path resolves the same way.
            assert_eq!(resolve_for_containment(r"\etc\hosts", cwd), r"C:\etc\hosts");

            // A UNC path must not acquire a drive letter, and stays outside.
            let unc = resolve_for_containment(r"\\server\share\f.txt", cwd);
            assert_eq!(unc, r"\\server\share\f.txt");
            assert!(!is_in_directory(&unc, cwd));

            // Ordinary relative paths still land inside the session directory.
            let inside = resolve_for_containment("sub/f.txt", cwd);
            assert!(is_in_directory(&inside, cwd), "{inside} should be inside");

            // An already drive-absolute path is returned untouched.
            assert_eq!(resolve_for_containment(r"D:\x\y", cwd), r"D:\x\y");
        }
        #[cfg(not(windows))]
        {
            let cwd = "/root/project";
            assert_eq!(resolve_for_containment("/etc/hosts", cwd), "/etc/hosts");
            assert!(!is_in_directory("/etc/hosts", cwd));
            let inside = resolve_for_containment("sub/f.txt", cwd);
            assert!(is_in_directory(&inside, cwd), "{inside} should be inside");
        }
    }
}
