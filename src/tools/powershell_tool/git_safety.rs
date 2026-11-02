//! Git sandbox-escape guards (gitSafety.ts).
//!
//! Git can be weaponized two ways: (1) bare-repo attack — cwd with HEAD/objects/
//! refs/ but no .git makes git treat cwd as a bare repo and run hooks from cwd;
//! (2) git-internal write + git — a compound creates HEAD/objects/refs/hooks/
//! then runs git, executing the freshly-planted hooks. These detect args that
//! resolve to git-internal paths in cwd.

use once_cell::sync::Lazy;
use regex::Regex;

use crate::utils::powershell::parser::PS_TOKENIZER_DASH_CHARS;
use crate::utils::state::get_cwd;

const GIT_INTERNAL_PREFIXES: &[&str] = &["head", "objects", "refs", "hooks"];
static GIT_SHORTNAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^git~\d+($|/)").unwrap());
static PROVIDER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^(?:[A-Za-z0-9_.]+\\){0,3}FileSystem::").unwrap());
static DRIVE_ABS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-z]:").unwrap());

/// POSIX-style lexical path normalization (posixpath.normpath equivalent for
/// the inputs this module handles).
fn posix_normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_string();
    }
    let is_abs = path.starts_with('/');
    // exactly-two leading slashes are preserved by POSIX; 3+ collapse to one.
    let initial_slashes = if path.starts_with("//") && !path.starts_with("///") { 2 } else { usize::from(is_abs) };
    let mut new_comps: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".."
            || (initial_slashes == 0 && new_comps.is_empty())
            || new_comps.last() == Some(&"..")
        {
            new_comps.push(comp);
        } else if !new_comps.is_empty() {
            new_comps.pop();
        }
    }
    let mut out = "/".repeat(initial_slashes);
    out.push_str(&new_comps.join("/"));
    if out.is_empty() {
        ".".to_string()
    } else {
        out
    }
}

fn strip_matching_quotes(s: &str) -> String {
    // ^['"] | ['"]\z  (one leading and/or one trailing quote)
    static Q: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());
    Q.replace_all(s, "").to_string()
}

fn basename_lower(p: &str) -> String {
    std::path::Path::new(p)
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// `../<cwd-basename>/x` re-enters cwd via parent → strip to cwd-relative.
fn resolve_cwd_reentry(normalized: &str) -> String {
    if !normalized.starts_with("../") {
        return normalized.to_string();
    }
    let cwd_base = basename_lower(&get_cwd());
    if cwd_base.is_empty() {
        return normalized.to_string();
    }
    let prefix = format!("../{cwd_base}/");
    let mut s = normalized.to_string();
    while s.starts_with(&prefix) {
        s = s[prefix.len()..].to_string();
    }
    if s == format!("../{cwd_base}") {
        return ".".to_string();
    }
    s
}

fn split_first_char(s: &str) -> Option<(char, &str)> {
    let mut it = s.char_indices();
    let (_, c) = it.next()?;
    let rest_start = it.next().map(|(i, _)| i).unwrap_or(s.len());
    Some((c, &s[rest_start..]))
}

/// Normalize PS arg text → canonical path for git-internal matching.
fn normalize_git_path_arg(arg: &str) -> String {
    let mut s = arg.to_string();
    // Parameter prefix (dash chars / PS 5.1 `/`) with colon-bound value.
    if let Some((first, _rest)) = split_first_char(&s) {
        if PS_TOKENIZER_DASH_CHARS.contains(&first) || first == '/' {
            if let Some(c) = s.char_indices().skip(1).find(|(_, ch)| *ch == ':').map(|(i, _)| i) {
                if c > 0 {
                    s = s[c + 1..].to_string();
                }
            }
        }
    }
    s = strip_matching_quotes(&s);
    s = s.replace('`', "");
    s = PROVIDER_RE.replace(&s, "").to_string(); // FileSystem::path
    // _DRIVE_REL_RE: ^[A-Za-z]:(?![/\\]) — drive-relative `C:foo` → `foo`.
    if is_drive_relative(&s) {
        s = s[2..].to_string();
    }
    s = s.replace('\\', "/");
    // Win32 per-component: iteratively strip trailing spaces then dots (keeping ./..)
    let mut parts: Vec<String> = Vec::new();
    for comp in s.split('/') {
        if comp.is_empty() {
            parts.push(String::new());
            continue;
        }
        let mut c = comp.to_string();
        loop {
            let prev = c.clone();
            c = c.trim_end_matches(' ').to_string();
            if c == "." || c == ".." {
                break;
            }
            c = c.trim_end_matches('.').to_string();
            if c == prev {
                break;
            }
        }
        parts.push(if c.is_empty() { ".".to_string() } else { c });
    }
    s = parts.join("/");
    s = posix_normpath(&s);
    if let Some(stripped) = s.strip_prefix("./") {
        s = stripped.to_string();
    }
    s.to_lowercase()
}

/// `^[A-Za-z]:` NOT followed by `/` or `\` (the negative-lookahead form).
fn is_drive_relative(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 2 || !bytes[0].is_ascii_alphabetic() || bytes[1] != b':' {
        return false;
    }
    match bytes.get(2) {
        None => true,
        Some(&c) => c != b'/' && c != b'\\',
    }
}

/// Resolve an escaping (../, absolute, drive) path against cwd; return the
/// cwd-relative remainder if it lands back inside cwd, else None.
fn resolve_escaping_path_to_cwd_relative(n: &str) -> Option<String> {
    let cwd = get_cwd();
    let joined = if n.starts_with('/') { n.to_string() } else { format!("{cwd}/{n}") };
    let abs_path = posix_normpath(&joined);
    let sep = "/";
    let cwd_with_sep = if cwd.ends_with(sep) { cwd.clone() } else { format!("{cwd}{sep}") };
    let abs_lower = abs_path.to_lowercase();
    let cwd_lower = cwd.to_lowercase();
    let cwd_with_sep_lower = cwd_with_sep.to_lowercase();
    if abs_lower == cwd_lower {
        return Some(".".to_string());
    }
    if !abs_lower.starts_with(&cwd_with_sep_lower) {
        return None;
    }
    Some(abs_path[cwd_with_sep.len()..].replace('\\', "/").to_lowercase())
}

fn matches_git_internal_prefix(n: &str) -> bool {
    if n == "head" || n == ".git" {
        return true;
    }
    if n.starts_with(".git/") || GIT_SHORTNAME_RE.is_match(n) {
        return true;
    }
    for p in GIT_INTERNAL_PREFIXES {
        if *p == "head" {
            continue;
        }
        if n == *p || n.starts_with(&format!("{p}/")) {
            return true;
        }
    }
    false
}

fn matches_dot_git_prefix(n: &str) -> bool {
    if n == ".git" || n.starts_with(".git/") {
        return true;
    }
    GIT_SHORTNAME_RE.is_match(n)
}

/// True if arg resolves to a git-internal path in cwd (bare-repo hooks/refs/ or
/// standard .git/*).
pub fn is_git_internal_path_ps(arg: &str) -> bool {
    let n = resolve_cwd_reentry(&normalize_git_path_arg(arg));
    if matches_git_internal_prefix(&n) {
        return true;
    }
    if n.starts_with("../") || n.starts_with('/') || DRIVE_ABS_RE.is_match(&n) {
        if let Some(rel) = resolve_escaping_path_to_cwd_relative(&n) {
            if matches_git_internal_prefix(&rel) {
                return true;
            }
        }
    }
    false
}

/// True if arg resolves to a path inside .git/ (standard-repo metadata). Unlike
/// is_git_internal_path_ps, does NOT match bare-repo root-level hooks/refs/.
pub fn is_dot_git_path_ps(arg: &str) -> bool {
    let n = resolve_cwd_reentry(&normalize_git_path_arg(arg));
    if matches_dot_git_prefix(&n) {
        return true;
    }
    if n.starts_with("../") || n.starts_with('/') || DRIVE_ABS_RE.is_match(&n) {
        if let Some(rel) = resolve_escaping_path_to_cwd_relative(&n) {
            if matches_dot_git_prefix(&rel) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normpath_basics() {
        assert_eq!(posix_normpath("a/./b"), "a/b");
        assert_eq!(posix_normpath("a/../b"), "b");
        assert_eq!(posix_normpath("./a"), "a");
        assert_eq!(posix_normpath(""), ".");
        assert_eq!(posix_normpath("../a"), "../a");
    }

    #[test]
    fn detects_internal_paths() {
        assert!(is_git_internal_path_ps("HEAD"));
        assert!(is_git_internal_path_ps("hooks/pre-commit"));
        assert!(is_git_internal_path_ps("refs/heads/x"));
        assert!(is_git_internal_path_ps(".git/config"));
        assert!(!is_git_internal_path_ps("src/main.rs"));
    }

    #[test]
    fn dot_git_stricter() {
        assert!(is_dot_git_path_ps(".git/hooks/x"));
        // bare-repo root hooks/ is NOT a .git path
        assert!(!is_dot_git_path_ps("hooks/pre-commit"));
    }

    #[test]
    fn drive_relative_detection() {
        assert!(is_drive_relative("C:foo"));
        assert!(!is_drive_relative("C:/foo"));
        assert!(!is_drive_relative("C:\\foo"));
    }
}
