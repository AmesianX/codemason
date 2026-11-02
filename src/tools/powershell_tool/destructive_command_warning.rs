//! Destructive-command warnings for the permission dialog (destructiveCommandWarning.ts).
//!
//! Purely informational — returns a warning string, does NOT affect permission
//! logic or auto-approval.

use once_cell::sync::Lazy;
use regex::Regex;

/// (pattern, warning). Order matters (first match wins). The git-clean case
/// needs a negative lookahead the regex crate lacks, so it is handled
/// separately below at its original position in the ordered scan.
static PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    let ci = |p: &str| Regex::new(&format!("(?i){p}")).unwrap();
    vec![
        // Remove-Item -Recurse + -Force (either order), anchored to statement
        // start so `git rm --force` doesn't match.
        (ci(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Recurse\b[^|;&\n}]*-Force\b"),
         "Note: may recursively force-remove files"),
        (ci(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Force\b[^|;&\n}]*-Recurse\b"),
         "Note: may recursively force-remove files"),
        (ci(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Recurse\b"),
         "Note: may recursively remove files"),
        (ci(r"(?:^|[|;&\n({])\s*(Remove-Item|rm|del|rd|rmdir|ri)\b[^|;&\n}]*-Force\b"),
         "Note: may force-remove files"),
        // Clear-Content on broad paths
        (ci(r"\bClear-Content\b[^|;&\n]*\*"), "Note: may clear content of multiple files"),
        // Disk operations
        (ci(r"\bFormat-Volume\b"), "Note: may format a disk volume"),
        (ci(r"\bClear-Disk\b"), "Note: may clear a disk"),
        // Git destructive operations
        (ci(r"\bgit\s+reset\s+--hard\b"), "Note: may discard uncommitted changes"),
        (ci(r"\bgit\s+push\b[^|;&\n]*\s+(--force|--force-with-lease|-f)\b"),
         "Note: may overwrite remote history"),
        // (git clean handled specially — see get_destructive_command_warning)
        (ci(r"\bgit\s+stash\s+(drop|clear)\b"), "Note: may permanently remove stashed changes"),
        // Database operations
        (ci(r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)\b"),
         "Note: may drop or truncate database objects"),
        // System operations
        (ci(r"\bStop-Computer\b"), "Note: will shut down the computer"),
        (ci(r"\bRestart-Computer\b"), "Note: will restart the computer"),
        (ci(r"\bClear-RecycleBin\b"), "Note: permanently deletes recycled files"),
    ]
});

// git-clean sub-parts (the negative-lookahead form, restructured).
static GIT_CLEAN_HEAD: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bgit\s+clean\b").unwrap());
static GIT_CLEAN_F: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)-[a-zA-Z]*f").unwrap());
static GIT_CLEAN_N: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)-[a-zA-Z]*n").unwrap());

/// Reproduces `\bgit\s+clean\b(?![^|;&\n]*(?:-[a-zA-Z]*n|--dry-run))[^|;&\n]*-[a-zA-Z]*f`
/// without lookahead: after "git clean", within the same segment (no |;&\n
/// separator), there must be a `-…f` flag and NO dry-run flag (`-…n` /
/// `--dry-run`). Semantics verified equivalent by the tests below.
fn matches_git_clean_force(command: &str) -> bool {
    for m in GIT_CLEAN_HEAD.find_iter(command) {
        let rest = &command[m.end()..];
        // segment tail: up to the first statement separator
        let tail_end = rest.find(|c| matches!(c, '|' | ';' | '&' | '\n')).unwrap_or(rest.len());
        let tail = &rest[..tail_end];
        let has_dry_run = GIT_CLEAN_N.is_match(tail) || tail.to_lowercase().contains("--dry-run");
        if !has_dry_run && GIT_CLEAN_F.is_match(tail) {
            return true;
        }
    }
    false
}

/// Warning string for a known destructive pattern, or None.
pub fn get_destructive_command_warning(command: &str) -> Option<&'static str> {
    // The git-clean check sits between `git push --force` and `git stash` in the
    // original ordered list; replicate that position.
    for (regex, warning) in PATTERNS.iter() {
        if regex.is_match(command) {
            return Some(warning);
        }
        // Insert the git-clean check right after the `git push` pattern.
        if *warning == "Note: may overwrite remote history" && matches_git_clean_force(command) {
            return Some("Note: may permanently delete untracked files");
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurse_force_either_order() {
        assert_eq!(get_destructive_command_warning("Remove-Item x -Recurse -Force"),
                   Some("Note: may recursively force-remove files"));
        assert_eq!(get_destructive_command_warning("Remove-Item x -Force -Recurse"),
                   Some("Note: may recursively force-remove files"));
    }

    #[test]
    fn git_rm_force_not_matched_by_removeitem() {
        // `git rm --force` must NOT hit the Remove-Item anchored patterns.
        let w = get_destructive_command_warning("git rm --force x");
        assert_eq!(w, None);
    }

    #[test]
    fn git_clean_force_but_not_dry_run() {
        assert_eq!(get_destructive_command_warning("git clean -fd"),
                   Some("Note: may permanently delete untracked files"));
        // dry-run forms must NOT warn
        assert_eq!(get_destructive_command_warning("git clean -fdn"), None);
        assert_eq!(get_destructive_command_warning("git clean -f --dry-run"), None);
        assert_eq!(get_destructive_command_warning("git clean -n"), None);
    }

    #[test]
    fn reset_hard_and_disk() {
        assert_eq!(get_destructive_command_warning("git reset --hard"),
                   Some("Note: may discard uncommitted changes"));
        assert_eq!(get_destructive_command_warning("Format-Volume -DriveLetter D"),
                   Some("Note: may format a disk volume"));
    }

    #[test]
    fn none_for_safe() {
        assert_eq!(get_destructive_command_warning("Get-ChildItem"), None);
    }
}
