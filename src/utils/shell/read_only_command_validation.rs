//! Read-only external-command validation (adapted from readOnlyCommandValidation.ts).
//! Equivalent to codemason/utils/shell/read_only_command_validation.py.
//!
//! Shared engine + data for deciding whether an external command (git, gh,
//! docker, rg, pyright, ...) is read-only. This module is the ENGINE (flag
//! validation, UNC detection, config type); the per-command allowlists
//! (GIT/GH/DOCKER/...) live alongside and are consumed by both the Bash and
//! PowerShell tools.
//!
//! Cmdlets/commands are matched case-insensitively by their callers.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::utils::platform::is_windows;

/// FlagArgType: 'none' | 'number' | 'string' | 'char' | '{}' | 'EOF'
pub type FlagArgType = &'static str;

/// Returns true if the command is dangerous, false if safe. args = tokens
/// AFTER the command name (e.g. after "git branch").
pub type DangerCallback = fn(&str, &[String]) -> bool;

/// Safe-flag config for a read-only external command.
#[derive(Clone)]
pub struct ExternalCommandConfig {
    pub safe_flags: HashMap<&'static str, FlagArgType>,
    pub additional_command_is_dangerous_callback: Option<DangerCallback>,
    /// When false, do NOT respect POSIX `--` end-of-options (keep checking flags).
    pub respects_double_dash: bool,
}

impl ExternalCommandConfig {
    fn new(flags: &[(&'static str, FlagArgType)], cb: Option<DangerCallback>) -> Self {
        ExternalCommandConfig {
            safe_flags: flags.iter().cloned().collect(),
            additional_command_is_dangerous_callback: cb,
            respects_double_dash: true,
        }
    }

    fn no_double_dash(mut self) -> Self {
        self.respects_double_dash = false;
        self
    }
}

/// Cross-platform external tools identical in bash and PowerShell on Windows.
///
/// These two are allowed WITHOUT flag validation, unlike every other docker
/// subcommand (see DOCKER_READ_ONLY_COMMANDS, which puts `logs` and `inspect`
/// through validate_flags). That asymmetry is deliberate, and it has been filed
/// as an inconsistency once — the reasoning is written down here so it is not
/// filed again.
///
/// `ps` and `images` are read-only COMMANDS, not merely commands with read-only
/// flags: nothing in either flag set writes. ps takes -a/--all, -f/--filter,
/// --format, -n/--last, -l/--latest, --no-trunc, -q/--quiet, -s/--size; images
/// takes -a/--all, --digests, -f/--filter, --format, --no-trunc, -q/--quiet.
/// There is no mutating flag to omit from an allowlist, so an allowlist would
/// buy no safety.
///
/// It would cost something, though. Every flag docker adds later would become a
/// permission prompt for a command that cannot do anything, and the list would
/// need chasing upstream forever. Security that costs more usability than it
/// buys is not a trade worth making, so the state and its price are recorded
/// rather than "fixed".
///
/// `$` is still rejected before this is reached, so the substitution route in
/// is closed independently of the flag surface.
pub const EXTERNAL_READONLY_COMMANDS: &[&str] = &["docker ps", "docker images"];

// =============================================================================
// UNC path detection (shared across Bash and PowerShell).
// =============================================================================
static UNC_BACKSLASH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\\\\[^\s\\/]+(?:@(?:\d+|ssl))?(?:[\\/]|$|\s)").unwrap());
// Python used a `(?<!:)` lookbehind here; the regex crate has none, so the
// anchored form below is tried at each candidate position whose preceding char
// is not ':' — see contains_vulnerable_unc_path (identical semantics, including
// the position-by-position advance a failed lookbehind causes).
static UNC_FORWARDSLASH_ANCHORED_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)^//[^\s\\/]+(?:@(?:\d+|ssl))?(?:[\\/]|$|\s)").unwrap());
static UNC_MIXED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"/\\{2,}[^\s\\/]").unwrap());
static UNC_REVERSE_MIXED_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\\{2,}/[^\s\\/]").unwrap());
static UNC_SSL_PORT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)@SSL@\d+").unwrap());
static UNC_PORT_SSL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)@\d+@SSL").unwrap());
static UNC_DAV_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)DavWWWRoot").unwrap());
static UNC_IPV4_BS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\\\\(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\\/]").unwrap());
static UNC_IPV4_FS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\\/]").unwrap());
static UNC_IPV6_BS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\\\\(\[[\da-fA-F:]+\])[\\/]").unwrap());
static UNC_IPV6_FS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^//(\[[\da-fA-F:]+\])[\\/]").unwrap());

/// The lookbehind-free equivalent of Python's `(?<!:)//...` search: try the
/// anchored pattern at every position starting "//" whose preceding character
/// is not ':'.
fn unc_forwardslash_search(s: &str) -> bool {
    let bytes = s.as_bytes();
    for (i, w) in bytes.windows(2).enumerate() {
        if w == b"//" && (i == 0 || bytes[i - 1] != b':') {
            // Safe slice: '/' is ASCII so i is always a char boundary.
            if UNC_FORWARDSLASH_ANCHORED_RE.is_match(&s[i..]) {
                return true;
            }
        }
    }
    // A trailing single '/' can't start '//'; windows(2) covers all starts.
    false
}

/// True if the string contains a UNC path that could trigger a network request
/// (NTLM/Kerberos leak, WebDAV). Windows-only.
pub fn contains_vulnerable_unc_path(path_or_command: &str) -> bool {
    if !is_windows() {
        return false;
    }
    if UNC_BACKSLASH_RE.is_match(path_or_command) {
        return true;
    }
    if unc_forwardslash_search(path_or_command) {
        return true;
    }
    if UNC_MIXED_RE.is_match(path_or_command) {
        return true;
    }
    if UNC_REVERSE_MIXED_RE.is_match(path_or_command) {
        return true;
    }
    if UNC_SSL_PORT_RE.is_match(path_or_command) || UNC_PORT_SSL_RE.is_match(path_or_command) {
        return true;
    }
    if UNC_DAV_RE.is_match(path_or_command) {
        return true;
    }
    if UNC_IPV4_BS_RE.is_match(path_or_command) || UNC_IPV4_FS_RE.is_match(path_or_command) {
        return true;
    }
    if UNC_IPV6_BS_RE.is_match(path_or_command) || UNC_IPV6_FS_RE.is_match(path_or_command) {
        return true;
    }
    false
}

// =============================================================================
// Flag validation.
// =============================================================================
static FLAG_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-[a-zA-Z0-9_-]").unwrap());
// ASCII-only + full-string: JS `/^\d+$/.test()` matches only ASCII 0-9 and does
// not accept a trailing newline. (The regex crate's `[0-9]` is ASCII and `$`
// with an explicit `^...$` anchoring over the whole haystack does not permit a
// trailing '\n', so TS semantics hold.)
static NUMERIC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[0-9]+$").unwrap());
static GIT_NUM_SHORTHAND_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-[0-9]+$").unwrap());
static SORT_REVERSE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-[a-zA-Z]").unwrap());

/// Validate a flag argument against its expected type.
pub fn validate_flag_argument(value: &str, arg_type: FlagArgType) -> bool {
    match arg_type {
        "none" => false, // should not be called for 'none'
        "number" => NUMERIC_RE.is_match(value),
        "string" => true,
        "char" => value.chars().count() == 1,
        "{}" => value == "{}",
        "EOF" => value == "EOF",
        _ => false,
    }
}

/// Validate the flags/args of a tokenized command against `config`.
///
/// Faithful port of the flag-walking loop (validateFlags), including the
/// security-hardened cases: xargs target detection, `-E=` empty-inline-value
/// (no next-token consume), git `-<n>` shorthand, grep/rg attached-numeric
/// (`-A20`), and bundled-flag rejection when any bundled flag takes an arg.
pub fn validate_flags(
    tokens: &[String],
    start_index: usize,
    config: &ExternalCommandConfig,
    command_name: Option<&str>,
    _raw_command: Option<&str>,
    xargs_target_commands: Option<&[String]>,
) -> bool {
    let mut i = start_index;
    let n = tokens.len();
    while i < n {
        let mut token = tokens[i].as_str();
        if token.is_empty() {
            i += 1;
            continue;
        }

        // xargs: once the target command appears, stop validating flags.
        if let Some(targets) = xargs_target_commands.filter(|t| !t.is_empty()) {
            // (an empty target list is falsy in Python and skips this branch)
            if command_name == Some("xargs") && (!token.starts_with('-') || token == "--") {
                if token == "--" && i + 1 < n {
                    i += 1;
                    token = tokens[i].as_str();
                }
                if !token.is_empty() && targets.iter().any(|t| t == token) {
                    break;
                }
                return false;
            }
        }

        if token == "--" {
            if config.respects_double_dash {
                // everything after -- is arguments
                break;
            }
            i += 1;
            continue;
        }

        if token.starts_with('-') && token.len() > 1 && FLAG_PATTERN.is_match(token) {
            let has_equals = token.contains('=');
            let mut parts = token.splitn(2, '=');
            let flag = parts.next().unwrap_or("");
            let inline_value = parts.next().unwrap_or("").to_string();

            if flag.is_empty() {
                return false;
            }

            let flag_arg_type = config.safe_flags.get(flag).copied();

            let Some(flag_arg_type) = flag_arg_type else {
                // git -<number> shorthand for -n <number>
                if command_name == Some("git") && GIT_NUM_SHORTHAND_RE.is_match(flag) {
                    i += 1;
                    continue;
                }
                // grep/rg attached numeric (-A20)
                if matches!(command_name, Some("grep") | Some("rg"))
                    && flag.starts_with('-')
                    && !flag.starts_with("--")
                    && flag.len() > 2
                {
                    let potential_flag = &flag[..2];
                    let potential_value = &flag[2..];
                    if let Some(pf_type) = config.safe_flags.get(potential_flag).copied() {
                        if NUMERIC_RE.is_match(potential_value)
                            && matches!(pf_type, "number" | "string")
                        {
                            if validate_flag_argument(potential_value, pf_type) {
                                i += 1;
                                continue;
                            }
                            return false;
                        }
                    }
                }
                // bundled single-letter flags (-nr): all must be 'none'.
                if flag.starts_with('-') && !flag.starts_with("--") && flag.len() > 2 {
                    let mut all_ok = true;
                    for ch in flag.chars().skip(1) {
                        let single_flag = format!("-{ch}");
                        match config.safe_flags.get(single_flag.as_str()).copied() {
                            None => {
                                all_ok = false;
                                break;
                            }
                            Some(ft) if ft != "none" => {
                                all_ok = false; // arg-taking flag in a bundle
                                break;
                            }
                            Some(_) => {}
                        }
                    }
                    if all_ok {
                        i += 1;
                        continue;
                    }
                    return false;
                }
                return false; // unknown flag
            };

            if flag_arg_type == "none" {
                if has_equals {
                    return false; // flag should not have a value
                }
                i += 1;
            } else {
                let arg_value: String;
                if has_equals {
                    arg_value = inline_value;
                    i += 1;
                } else {
                    let nxt = tokens.get(i + 1).map(|s| s.as_str());
                    let nxt_is_flag = nxt
                        .map(|s| s.starts_with('-') && s.len() > 1 && FLAG_PATTERN.is_match(s))
                        .unwrap_or(false);
                    if i + 1 >= n || nxt_is_flag {
                        return false; // missing required argument
                    }
                    arg_value = tokens[i + 1].clone();
                    i += 2;
                }

                // string args starting with '-' rejected (type-confusion),
                // except git --sort reverse (-refname).
                if flag_arg_type == "string" && arg_value.starts_with('-') {
                    let reverse_sort = flag == "--sort"
                        && command_name == Some("git")
                        && SORT_REVERSE_RE.is_match(&arg_value);
                    if !reverse_sort {
                        return false;
                    }
                }

                if !validate_flag_argument(&arg_value, flag_arg_type) {
                    return false;
                }
            }
        } else {
            i += 1; // non-flag positional arg — allowed
        }
    }

    true
}

// =============================================================================
// Per-command danger callbacks (hand-ported; the data records below wire these).
// =============================================================================

/// Block reflog expire/delete/exists (write to .git/logs/**); show/refs safe.
fn git_reflog_cb(_raw_command: &str, args: &[String]) -> bool {
    for token in args {
        if token.is_empty() || token.starts_with('-') {
            continue;
        }
        if matches!(token.as_str(), "expire" | "delete" | "exists") {
            return true;
        }
        return false; // first positional is show/ref → safe
    }
    false // bare `git reflog` = show
}

static REMOTE_NAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap());

/// Allow only optional -n then exactly one remote-name-looking positional.
fn git_remote_show_cb(_raw_command: &str, args: &[String]) -> bool {
    let positional: Vec<&String> = args.iter().filter(|a| a.as_str() != "-n").collect();
    if positional.len() != 1 {
        return true;
    }
    // Full-string match, not match+'$' with trailing-newline tolerance: JS
    // `/^[..]+$/.test` rejects `origin\n` (the regex crate `^...$` does too).
    !REMOTE_NAME_RE.is_match(positional[0])
}

/// Allow only bare `git remote` or `-v`/`--verbose`.
fn git_remote_cb(_raw_command: &str, args: &[String]) -> bool {
    args.iter().any(|a| a != "-v" && a != "--verbose")
}

/// Block tag creation: a positional without -l/--list is a tag name to create.
fn git_tag_cb(_raw_command: &str, args: &[String]) -> bool {
    const FLAGS_WITH_ARGS: &[&str] = &[
        "--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort",
        "--format", "-n",
    ];
    let mut i = 0usize;
    let mut seen_list = false;
    let mut seen_dash_dash = false;
    while i < args.len() {
        let token = args[i].as_str();
        if token.is_empty() {
            i += 1;
            continue;
        }
        if token == "--" && !seen_dash_dash {
            seen_dash_dash = true;
            i += 1;
            continue;
        }
        if !seen_dash_dash && token.starts_with('-') {
            if token == "--list" || token == "-l" {
                seen_list = true;
            } else if token.len() > 2
                && token.as_bytes()[1] != b'-'
                && !token.contains('=')
                && token[1..].contains('l')
            {
                seen_list = true;
            }
            if token.contains('=') {
                i += 1;
            } else if FLAGS_WITH_ARGS.contains(&token) {
                i += 2;
            } else {
                i += 1;
            }
        } else {
            if !seen_list {
                return true; // positional without --list = tag creation
            }
            i += 1;
        }
    }
    false
}

/// Block branch creation: a positional without --list/filtering flag is a new
/// branch.
fn git_branch_cb(_raw_command: &str, args: &[String]) -> bool {
    const FLAGS_WITH_ARGS: &[&str] = &["--contains", "--no-contains", "--points-at", "--sort"];
    const FLAGS_WITH_OPTIONAL_ARGS: &[&str] = &["--merged", "--no-merged"];
    let mut i = 0usize;
    let mut last_flag = String::new();
    let mut seen_list = false;
    let mut seen_dash_dash = false;
    while i < args.len() {
        let token = args[i].as_str();
        if token.is_empty() {
            i += 1;
            continue;
        }
        if token == "--" && !seen_dash_dash {
            seen_dash_dash = true;
            last_flag.clear();
            i += 1;
            continue;
        }
        if !seen_dash_dash && token.starts_with('-') {
            if token == "--list" || token == "-l" {
                seen_list = true;
            } else if token.len() > 2
                && token.as_bytes()[1] != b'-'
                && !token.contains('=')
                && token[1..].contains('l')
            {
                seen_list = true;
            }
            if token.contains('=') {
                last_flag = token.split('=').next().unwrap_or("").to_string();
                i += 1;
            } else if FLAGS_WITH_ARGS.contains(&token) {
                last_flag = token.to_string();
                i += 2;
            } else {
                last_flag = token.to_string();
                i += 1;
            }
        } else {
            let last_flag_optional = FLAGS_WITH_OPTIONAL_ARGS.contains(&last_flag.as_str());
            if !seen_list && !last_flag_optional {
                return true; // positional without --list/filter = branch creation
            }
            i += 1;
        }
    }
    false
}

/// Block gh repo/URL exfil: HOST/OWNER/REPO (2+ slashes), URLs, SSH-style.
fn gh_is_dangerous_cb(_raw_command: &str, args: &[String]) -> bool {
    for token in args {
        if token.is_empty() {
            continue;
        }
        let value: &str = if token.starts_with('-') {
            match token.find('=') {
                None => continue,
                Some(eq) => {
                    let v = &token[eq + 1..];
                    if v.is_empty() {
                        continue;
                    }
                    v
                }
            }
        } else {
            token.as_str()
        };
        if !value.contains('/') && !value.contains("://") && !value.contains('@') {
            continue;
        }
        if value.contains("://") {
            return true;
        }
        if value.contains('@') {
            return true;
        }
        if value.matches('/').count() >= 2 {
            return true;
        }
    }
    false
}

/// Block pyright --watch/-w (long-running, not read-only-and-exit).
fn pyright_cb(_raw_command: &str, args: &[String]) -> bool {
    args.iter().any(|t| t == "--watch" || t == "-w")
}

// =============================================================================
// Read-only command allowlists (data extracted verbatim from source; callbacks
// wired by reference). GIT/GH/DOCKER/RIPGREP/PYRIGHT.
// =============================================================================
pub static GIT_READ_ONLY_COMMANDS: Lazy<HashMap<&'static str, ExternalCommandConfig>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, ExternalCommandConfig> = HashMap::new();
        m.insert("git diff", ExternalCommandConfig::new(&[("--stat", "none"), ("--numstat", "none"), ("--shortstat", "none"), ("--name-only", "none"), ("--name-status", "none"), ("--color", "none"), ("--no-color", "none"), ("--dirstat", "none"), ("--summary", "none"), ("--patch-with-stat", "none"), ("--word-diff", "none"), ("--word-diff-regex", "string"), ("--color-words", "none"), ("--no-renames", "none"), ("--no-ext-diff", "none"), ("--check", "none"), ("--ws-error-highlight", "string"), ("--full-index", "none"), ("--binary", "none"), ("--abbrev", "number"), ("--break-rewrites", "none"), ("--find-renames", "none"), ("--find-copies", "none"), ("--find-copies-harder", "none"), ("--irreversible-delete", "none"), ("--diff-algorithm", "string"), ("--histogram", "none"), ("--patience", "none"), ("--minimal", "none"), ("--ignore-space-at-eol", "none"), ("--ignore-space-change", "none"), ("--ignore-all-space", "none"), ("--ignore-blank-lines", "none"), ("--inter-hunk-context", "number"), ("--function-context", "none"), ("--exit-code", "none"), ("--quiet", "none"), ("--cached", "none"), ("--staged", "none"), ("--pickaxe-regex", "none"), ("--pickaxe-all", "none"), ("--no-index", "none"), ("--relative", "string"), ("--diff-filter", "string"), ("-p", "none"), ("-u", "none"), ("-s", "none"), ("-M", "none"), ("-C", "none"), ("-B", "none"), ("-D", "none"), ("-l", "none"), ("-S", "string"), ("-G", "string"), ("-O", "string"), ("-R", "none")], None));
        m.insert("git log", ExternalCommandConfig::new(&[("--oneline", "none"), ("--graph", "none"), ("--decorate", "none"), ("--no-decorate", "none"), ("--date", "string"), ("--relative-date", "none"), ("--all", "none"), ("--branches", "none"), ("--tags", "none"), ("--remotes", "none"), ("--since", "string"), ("--after", "string"), ("--until", "string"), ("--before", "string"), ("--max-count", "number"), ("-n", "number"), ("--stat", "none"), ("--numstat", "none"), ("--shortstat", "none"), ("--name-only", "none"), ("--name-status", "none"), ("--color", "none"), ("--no-color", "none"), ("--patch", "none"), ("-p", "none"), ("--no-patch", "none"), ("--no-ext-diff", "none"), ("-s", "none"), ("--author", "string"), ("--committer", "string"), ("--grep", "string"), ("--abbrev-commit", "none"), ("--full-history", "none"), ("--dense", "none"), ("--sparse", "none"), ("--simplify-merges", "none"), ("--ancestry-path", "none"), ("--source", "none"), ("--first-parent", "none"), ("--merges", "none"), ("--no-merges", "none"), ("--reverse", "none"), ("--walk-reflogs", "none"), ("--skip", "number"), ("--max-age", "number"), ("--min-age", "number"), ("--no-min-parents", "none"), ("--no-max-parents", "none"), ("--follow", "none"), ("--no-walk", "none"), ("--left-right", "none"), ("--cherry-mark", "none"), ("--cherry-pick", "none"), ("--boundary", "none"), ("--topo-order", "none"), ("--date-order", "none"), ("--author-date-order", "none"), ("--pretty", "string"), ("--format", "string"), ("--diff-filter", "string"), ("-S", "string"), ("-G", "string"), ("--pickaxe-regex", "none"), ("--pickaxe-all", "none")], None));
        m.insert("git show", ExternalCommandConfig::new(&[("--oneline", "none"), ("--graph", "none"), ("--decorate", "none"), ("--no-decorate", "none"), ("--date", "string"), ("--relative-date", "none"), ("--stat", "none"), ("--numstat", "none"), ("--shortstat", "none"), ("--name-only", "none"), ("--name-status", "none"), ("--color", "none"), ("--no-color", "none"), ("--patch", "none"), ("-p", "none"), ("--no-patch", "none"), ("--no-ext-diff", "none"), ("-s", "none"), ("--abbrev-commit", "none"), ("--word-diff", "none"), ("--word-diff-regex", "string"), ("--color-words", "none"), ("--pretty", "string"), ("--format", "string"), ("--first-parent", "none"), ("--raw", "none"), ("--diff-filter", "string"), ("-m", "none"), ("--quiet", "none")], None));
        m.insert("git shortlog", ExternalCommandConfig::new(&[("--all", "none"), ("--branches", "none"), ("--tags", "none"), ("--remotes", "none"), ("--since", "string"), ("--after", "string"), ("--until", "string"), ("--before", "string"), ("-s", "none"), ("--summary", "none"), ("-n", "none"), ("--numbered", "none"), ("-e", "none"), ("--email", "none"), ("-c", "none"), ("--committer", "none"), ("--group", "string"), ("--format", "string"), ("--no-merges", "none"), ("--author", "string")], None));
        m.insert("git reflog", ExternalCommandConfig::new(&[("--oneline", "none"), ("--graph", "none"), ("--decorate", "none"), ("--no-decorate", "none"), ("--date", "string"), ("--relative-date", "none"), ("--all", "none"), ("--branches", "none"), ("--tags", "none"), ("--remotes", "none"), ("--since", "string"), ("--after", "string"), ("--until", "string"), ("--before", "string"), ("--max-count", "number"), ("-n", "number"), ("--author", "string"), ("--committer", "string"), ("--grep", "string")], Some(git_reflog_cb)));
        m.insert("git stash list", ExternalCommandConfig::new(&[("--oneline", "none"), ("--graph", "none"), ("--decorate", "none"), ("--no-decorate", "none"), ("--date", "string"), ("--relative-date", "none"), ("--all", "none"), ("--branches", "none"), ("--tags", "none"), ("--remotes", "none"), ("--max-count", "number"), ("-n", "number")], None));
        m.insert("git ls-remote", ExternalCommandConfig::new(&[("--branches", "none"), ("-b", "none"), ("--tags", "none"), ("-t", "none"), ("--heads", "none"), ("-h", "none"), ("--refs", "none"), ("--quiet", "none"), ("-q", "none"), ("--exit-code", "none"), ("--get-url", "none"), ("--symref", "none"), ("--sort", "string")], None));
        m.insert("git status", ExternalCommandConfig::new(&[("--short", "none"), ("-s", "none"), ("--branch", "none"), ("-b", "none"), ("--porcelain", "none"), ("--long", "none"), ("--verbose", "none"), ("-v", "none"), ("--untracked-files", "string"), ("-u", "string"), ("--ignored", "none"), ("--ignore-submodules", "string"), ("--column", "none"), ("--no-column", "none"), ("--ahead-behind", "none"), ("--no-ahead-behind", "none"), ("--renames", "none"), ("--no-renames", "none"), ("--find-renames", "string"), ("-M", "string")], None));
        m.insert("git blame", ExternalCommandConfig::new(&[("--color", "none"), ("--no-color", "none"), ("-L", "string"), ("--porcelain", "none"), ("-p", "none"), ("--line-porcelain", "none"), ("--incremental", "none"), ("--root", "none"), ("--show-stats", "none"), ("--show-name", "none"), ("--show-number", "none"), ("-n", "none"), ("--show-email", "none"), ("-e", "none"), ("-f", "none"), ("--date", "string"), ("-w", "none"), ("--ignore-rev", "string"), ("--ignore-revs-file", "string"), ("-M", "none"), ("-C", "none"), ("--score-debug", "none"), ("--abbrev", "number"), ("-s", "none"), ("-l", "none"), ("-t", "none")], None));
        m.insert("git ls-files", ExternalCommandConfig::new(&[("--cached", "none"), ("-c", "none"), ("--deleted", "none"), ("-d", "none"), ("--modified", "none"), ("-m", "none"), ("--others", "none"), ("-o", "none"), ("--ignored", "none"), ("-i", "none"), ("--stage", "none"), ("-s", "none"), ("--killed", "none"), ("-k", "none"), ("--unmerged", "none"), ("-u", "none"), ("--directory", "none"), ("--no-empty-directory", "none"), ("--eol", "none"), ("--full-name", "none"), ("--abbrev", "number"), ("--debug", "none"), ("-z", "none"), ("-t", "none"), ("-v", "none"), ("-f", "none"), ("--exclude", "string"), ("-x", "string"), ("--exclude-from", "string"), ("-X", "string"), ("--exclude-per-directory", "string"), ("--exclude-standard", "none"), ("--error-unmatch", "none"), ("--recurse-submodules", "none")], None));
        m.insert("git config --get", ExternalCommandConfig::new(&[("--local", "none"), ("--global", "none"), ("--system", "none"), ("--worktree", "none"), ("--default", "string"), ("--type", "string"), ("--bool", "none"), ("--int", "none"), ("--bool-or-int", "none"), ("--path", "none"), ("--expiry-date", "none"), ("-z", "none"), ("--null", "none"), ("--name-only", "none"), ("--show-origin", "none"), ("--show-scope", "none")], None));
        m.insert("git remote show", ExternalCommandConfig::new(&[("-n", "none")], Some(git_remote_show_cb)));
        m.insert("git remote", ExternalCommandConfig::new(&[("-v", "none"), ("--verbose", "none")], Some(git_remote_cb)));
        m.insert("git merge-base", ExternalCommandConfig::new(&[("--is-ancestor", "none"), ("--fork-point", "none"), ("--octopus", "none"), ("--independent", "none"), ("--all", "none")], None));
        m.insert("git rev-parse", ExternalCommandConfig::new(&[("--verify", "none"), ("--short", "string"), ("--abbrev-ref", "none"), ("--symbolic", "none"), ("--symbolic-full-name", "none"), ("--show-toplevel", "none"), ("--show-cdup", "none"), ("--show-prefix", "none"), ("--git-dir", "none"), ("--git-common-dir", "none"), ("--absolute-git-dir", "none"), ("--show-superproject-working-tree", "none"), ("--is-inside-work-tree", "none"), ("--is-inside-git-dir", "none"), ("--is-bare-repository", "none"), ("--is-shallow-repository", "none"), ("--is-shallow-update", "none"), ("--path-prefix", "none")], None));
        m.insert("git rev-list", ExternalCommandConfig::new(&[("--all", "none"), ("--branches", "none"), ("--tags", "none"), ("--remotes", "none"), ("--since", "string"), ("--after", "string"), ("--until", "string"), ("--before", "string"), ("--max-count", "number"), ("-n", "number"), ("--author", "string"), ("--committer", "string"), ("--grep", "string"), ("--count", "none"), ("--reverse", "none"), ("--first-parent", "none"), ("--ancestry-path", "none"), ("--merges", "none"), ("--no-merges", "none"), ("--min-parents", "number"), ("--max-parents", "number"), ("--no-min-parents", "none"), ("--no-max-parents", "none"), ("--skip", "number"), ("--max-age", "number"), ("--min-age", "number"), ("--walk-reflogs", "none"), ("--oneline", "none"), ("--abbrev-commit", "none"), ("--pretty", "string"), ("--format", "string"), ("--abbrev", "number"), ("--full-history", "none"), ("--dense", "none"), ("--sparse", "none"), ("--source", "none"), ("--graph", "none")], None));
        m.insert("git describe", ExternalCommandConfig::new(&[("--tags", "none"), ("--match", "string"), ("--exclude", "string"), ("--long", "none"), ("--abbrev", "number"), ("--always", "none"), ("--contains", "none"), ("--first-match", "none"), ("--exact-match", "none"), ("--candidates", "number"), ("--dirty", "none"), ("--broken", "none")], None));
        m.insert("git cat-file", ExternalCommandConfig::new(&[("-t", "none"), ("-s", "none"), ("-p", "none"), ("-e", "none"), ("--batch-check", "none"), ("--allow-undetermined-type", "none")], None));
        m.insert("git for-each-ref", ExternalCommandConfig::new(&[("--format", "string"), ("--sort", "string"), ("--count", "number"), ("--contains", "string"), ("--no-contains", "string"), ("--merged", "string"), ("--no-merged", "string"), ("--points-at", "string")], None));
        m.insert("git grep", ExternalCommandConfig::new(&[("-e", "string"), ("-E", "none"), ("--extended-regexp", "none"), ("-G", "none"), ("--basic-regexp", "none"), ("-F", "none"), ("--fixed-strings", "none"), ("-P", "none"), ("--perl-regexp", "none"), ("-i", "none"), ("--ignore-case", "none"), ("-v", "none"), ("--invert-match", "none"), ("-w", "none"), ("--word-regexp", "none"), ("-n", "none"), ("--line-number", "none"), ("-c", "none"), ("--count", "none"), ("-l", "none"), ("--files-with-matches", "none"), ("-L", "none"), ("--files-without-match", "none"), ("-h", "none"), ("-H", "none"), ("--heading", "none"), ("--break", "none"), ("--full-name", "none"), ("--color", "none"), ("--no-color", "none"), ("-o", "none"), ("--only-matching", "none"), ("-A", "number"), ("--after-context", "number"), ("-B", "number"), ("--before-context", "number"), ("-C", "number"), ("--context", "number"), ("--and", "none"), ("--or", "none"), ("--not", "none"), ("--max-depth", "number"), ("--untracked", "none"), ("--no-index", "none"), ("--recurse-submodules", "none"), ("--cached", "none"), ("--threads", "number"), ("-q", "none"), ("--quiet", "none")], None));
        m.insert("git stash show", ExternalCommandConfig::new(&[("--stat", "none"), ("--numstat", "none"), ("--shortstat", "none"), ("--name-only", "none"), ("--name-status", "none"), ("--color", "none"), ("--no-color", "none"), ("--patch", "none"), ("-p", "none"), ("--no-patch", "none"), ("--no-ext-diff", "none"), ("-s", "none"), ("--word-diff", "none"), ("--word-diff-regex", "string"), ("--diff-filter", "string"), ("--abbrev", "number")], None));
        m.insert("git worktree list", ExternalCommandConfig::new(&[("--porcelain", "none"), ("-v", "none"), ("--verbose", "none"), ("--expire", "string")], None));
        m.insert("git tag", ExternalCommandConfig::new(&[("-l", "none"), ("--list", "none"), ("-n", "number"), ("--contains", "string"), ("--no-contains", "string"), ("--merged", "string"), ("--no-merged", "string"), ("--sort", "string"), ("--format", "string"), ("--points-at", "string"), ("--column", "none"), ("--no-column", "none"), ("-i", "none"), ("--ignore-case", "none")], Some(git_tag_cb)));
        m.insert("git branch", ExternalCommandConfig::new(&[("-l", "none"), ("--list", "none"), ("-a", "none"), ("--all", "none"), ("-r", "none"), ("--remotes", "none"), ("-v", "none"), ("-vv", "none"), ("--verbose", "none"), ("--color", "none"), ("--no-color", "none"), ("--column", "none"), ("--no-column", "none"), ("--abbrev", "number"), ("--no-abbrev", "none"), ("--contains", "string"), ("--no-contains", "string"), ("--merged", "none"), ("--no-merged", "none"), ("--points-at", "string"), ("--sort", "string"), ("--show-current", "none"), ("-i", "none"), ("--ignore-case", "none")], Some(git_branch_cb)));
        m
    });

pub static GH_READ_ONLY_COMMANDS: Lazy<HashMap<&'static str, ExternalCommandConfig>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, ExternalCommandConfig> = HashMap::new();
        m.insert("gh pr view", ExternalCommandConfig::new(&[("--json", "string"), ("--comments", "none"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh pr list", ExternalCommandConfig::new(&[("--state", "string"), ("-s", "string"), ("--author", "string"), ("--assignee", "string"), ("--label", "string"), ("--limit", "number"), ("-L", "number"), ("--base", "string"), ("--head", "string"), ("--search", "string"), ("--json", "string"), ("--draft", "none"), ("--app", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh pr diff", ExternalCommandConfig::new(&[("--color", "string"), ("--name-only", "none"), ("--patch", "none"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh pr checks", ExternalCommandConfig::new(&[("--watch", "none"), ("--required", "none"), ("--fail-fast", "none"), ("--json", "string"), ("--interval", "number"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh issue view", ExternalCommandConfig::new(&[("--json", "string"), ("--comments", "none"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh issue list", ExternalCommandConfig::new(&[("--state", "string"), ("-s", "string"), ("--assignee", "string"), ("--author", "string"), ("--label", "string"), ("--limit", "number"), ("-L", "number"), ("--milestone", "string"), ("--search", "string"), ("--json", "string"), ("--app", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh repo view", ExternalCommandConfig::new(&[("--json", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh run list", ExternalCommandConfig::new(&[("--branch", "string"), ("-b", "string"), ("--status", "string"), ("-s", "string"), ("--workflow", "string"), ("-w", "string"), ("--limit", "number"), ("-L", "number"), ("--json", "string"), ("--repo", "string"), ("-R", "string"), ("--event", "string"), ("-e", "string"), ("--user", "string"), ("-u", "string"), ("--created", "string"), ("--commit", "string"), ("-c", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh run view", ExternalCommandConfig::new(&[("--log", "none"), ("--log-failed", "none"), ("--exit-status", "none"), ("--verbose", "none"), ("-v", "none"), ("--json", "string"), ("--repo", "string"), ("-R", "string"), ("--job", "string"), ("-j", "string"), ("--attempt", "number"), ("-a", "number")], Some(gh_is_dangerous_cb)));
        m.insert("gh auth status", ExternalCommandConfig::new(&[("--active", "none"), ("-a", "none"), ("--hostname", "string"), ("-h", "string"), ("--json", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh pr status", ExternalCommandConfig::new(&[("--conflict-status", "none"), ("-c", "none"), ("--json", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh issue status", ExternalCommandConfig::new(&[("--json", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh release list", ExternalCommandConfig::new(&[("--exclude-drafts", "none"), ("--exclude-pre-releases", "none"), ("--json", "string"), ("--limit", "number"), ("-L", "number"), ("--order", "string"), ("-O", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh release view", ExternalCommandConfig::new(&[("--json", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh workflow list", ExternalCommandConfig::new(&[("--all", "none"), ("-a", "none"), ("--json", "string"), ("--limit", "number"), ("-L", "number"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh workflow view", ExternalCommandConfig::new(&[("--ref", "string"), ("-r", "string"), ("--yaml", "none"), ("-y", "none"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh label list", ExternalCommandConfig::new(&[("--json", "string"), ("--limit", "number"), ("-L", "number"), ("--order", "string"), ("--search", "string"), ("-S", "string"), ("--sort", "string"), ("--repo", "string"), ("-R", "string")], Some(gh_is_dangerous_cb)));
        m.insert("gh search repos", ExternalCommandConfig::new(&[("--archived", "none"), ("--created", "string"), ("--followers", "string"), ("--forks", "string"), ("--good-first-issues", "string"), ("--help-wanted-issues", "string"), ("--include-forks", "string"), ("--json", "string"), ("--language", "string"), ("--license", "string"), ("--limit", "number"), ("-L", "number"), ("--match", "string"), ("--number-topics", "string"), ("--order", "string"), ("--owner", "string"), ("--size", "string"), ("--sort", "string"), ("--stars", "string"), ("--topic", "string"), ("--updated", "string"), ("--visibility", "string")], None));
        m.insert("gh search issues", ExternalCommandConfig::new(&[("--app", "string"), ("--assignee", "string"), ("--author", "string"), ("--closed", "string"), ("--commenter", "string"), ("--comments", "string"), ("--created", "string"), ("--include-prs", "none"), ("--interactions", "string"), ("--involves", "string"), ("--json", "string"), ("--label", "string"), ("--language", "string"), ("--limit", "number"), ("-L", "number"), ("--locked", "none"), ("--match", "string"), ("--mentions", "string"), ("--milestone", "string"), ("--no-assignee", "none"), ("--no-label", "none"), ("--no-milestone", "none"), ("--no-project", "none"), ("--order", "string"), ("--owner", "string"), ("--project", "string"), ("--reactions", "string"), ("--repo", "string"), ("-R", "string"), ("--sort", "string"), ("--state", "string"), ("--team-mentions", "string"), ("--updated", "string"), ("--visibility", "string")], None));
        m.insert("gh search prs", ExternalCommandConfig::new(&[("--app", "string"), ("--assignee", "string"), ("--author", "string"), ("--base", "string"), ("-B", "string"), ("--checks", "string"), ("--closed", "string"), ("--commenter", "string"), ("--comments", "string"), ("--created", "string"), ("--draft", "none"), ("--head", "string"), ("-H", "string"), ("--interactions", "string"), ("--involves", "string"), ("--json", "string"), ("--label", "string"), ("--language", "string"), ("--limit", "number"), ("-L", "number"), ("--locked", "none"), ("--match", "string"), ("--mentions", "string"), ("--merged", "none"), ("--merged-at", "string"), ("--milestone", "string"), ("--no-assignee", "none"), ("--no-label", "none"), ("--no-milestone", "none"), ("--no-project", "none"), ("--order", "string"), ("--owner", "string"), ("--project", "string"), ("--reactions", "string"), ("--repo", "string"), ("-R", "string"), ("--review", "string"), ("--review-requested", "string"), ("--reviewed-by", "string"), ("--sort", "string"), ("--state", "string"), ("--team-mentions", "string"), ("--updated", "string"), ("--visibility", "string")], None));
        m.insert("gh search commits", ExternalCommandConfig::new(&[("--author", "string"), ("--author-date", "string"), ("--author-email", "string"), ("--author-name", "string"), ("--committer", "string"), ("--committer-date", "string"), ("--committer-email", "string"), ("--committer-name", "string"), ("--hash", "string"), ("--json", "string"), ("--limit", "number"), ("-L", "number"), ("--merge", "none"), ("--order", "string"), ("--owner", "string"), ("--parent", "string"), ("--repo", "string"), ("-R", "string"), ("--sort", "string"), ("--tree", "string"), ("--visibility", "string")], None));
        m.insert("gh search code", ExternalCommandConfig::new(&[("--extension", "string"), ("--filename", "string"), ("--json", "string"), ("--language", "string"), ("--limit", "number"), ("-L", "number"), ("--match", "string"), ("--owner", "string"), ("--repo", "string"), ("-R", "string"), ("--size", "string")], None));
        m
    });

pub static DOCKER_READ_ONLY_COMMANDS: Lazy<HashMap<&'static str, ExternalCommandConfig>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, ExternalCommandConfig> = HashMap::new();
        m.insert("docker logs", ExternalCommandConfig::new(&[("--follow", "none"), ("-f", "none"), ("--tail", "string"), ("-n", "string"), ("--timestamps", "none"), ("-t", "none"), ("--since", "string"), ("--until", "string"), ("--details", "none")], None));
        m.insert("docker inspect", ExternalCommandConfig::new(&[("--format", "string"), ("-f", "string"), ("--type", "string"), ("--size", "none"), ("-s", "none")], None));
        m
    });

pub static RIPGREP_READ_ONLY_COMMANDS: Lazy<HashMap<&'static str, ExternalCommandConfig>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, ExternalCommandConfig> = HashMap::new();
        m.insert("rg", ExternalCommandConfig::new(&[("-e", "string"), ("--regexp", "string"), ("-f", "string"), ("-i", "none"), ("--ignore-case", "none"), ("-S", "none"), ("--smart-case", "none"), ("-F", "none"), ("--fixed-strings", "none"), ("-w", "none"), ("--word-regexp", "none"), ("-v", "none"), ("--invert-match", "none"), ("-c", "none"), ("--count", "none"), ("-l", "none"), ("--files-with-matches", "none"), ("--files-without-match", "none"), ("-n", "none"), ("--line-number", "none"), ("-o", "none"), ("--only-matching", "none"), ("-A", "number"), ("--after-context", "number"), ("-B", "number"), ("--before-context", "number"), ("-C", "number"), ("--context", "number"), ("-H", "none"), ("-h", "none"), ("--heading", "none"), ("--no-heading", "none"), ("-q", "none"), ("--quiet", "none"), ("--column", "none"), ("-g", "string"), ("--glob", "string"), ("-t", "string"), ("--type", "string"), ("-T", "string"), ("--type-not", "string"), ("--type-list", "none"), ("--hidden", "none"), ("--no-ignore", "none"), ("-u", "none"), ("-m", "number"), ("--max-count", "number"), ("-d", "number"), ("--max-depth", "number"), ("-a", "none"), ("--text", "none"), ("-z", "none"), ("-L", "none"), ("--follow", "none"), ("--color", "string"), ("--json", "none"), ("--stats", "none"), ("--help", "none"), ("--version", "none"), ("--debug", "none"), ("--", "none")], None));
        m
    });

pub static PYRIGHT_READ_ONLY_COMMANDS: Lazy<HashMap<&'static str, ExternalCommandConfig>> =
    Lazy::new(|| {
        let mut m: HashMap<&'static str, ExternalCommandConfig> = HashMap::new();
        m.insert(
            "pyright",
            ExternalCommandConfig::new(&[("--outputjson", "none"), ("--project", "string"), ("-p", "string"), ("--pythonversion", "string"), ("--pythonplatform", "string"), ("--typeshedpath", "string"), ("--venvpath", "string"), ("--level", "string"), ("--stats", "none"), ("--verbose", "none"), ("--version", "none"), ("--dependencies", "none"), ("--warnings", "none")], Some(pyright_cb))
            .no_double_dash(),
        );
        m
    });

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn numeric_rejects_trailing_newline_and_unicode() {
        assert!(NUMERIC_RE.is_match("12"));
        assert!(!NUMERIC_RE.is_match("12\n"));
        assert!(!NUMERIC_RE.is_match("１２")); // fullwidth digits
        assert!(!GIT_NUM_SHORTHAND_RE.is_match("-5\n"));
        assert!(GIT_NUM_SHORTHAND_RE.is_match("-5"));
    }

    #[test]
    fn validate_flags_basic() {
        let cfg = &GIT_READ_ONLY_COMMANDS["git log"];
        assert!(validate_flags(&toks(&["--oneline", "-n", "5"]), 0, cfg, Some("git"), None, None));
        assert!(!validate_flags(&toks(&["-n"]), 0, cfg, Some("git"), None, None)); // missing arg
        assert!(!validate_flags(&toks(&["--exec", "x"]), 0, cfg, Some("git"), None, None)); // unknown
        // git -<n> shorthand
        assert!(validate_flags(&toks(&["-3"]), 0, cfg, Some("git"), None, None));
        // string arg starting with '-' rejected
        assert!(!validate_flags(&toks(&["--author", "-evil"]), 0, cfg, Some("git"), None, None));
    }

    #[test]
    fn validate_flags_bundle_and_attached_numeric() {
        let cfg = &RIPGREP_READ_ONLY_COMMANDS["rg"];
        // -in bundle: both 'none'
        assert!(validate_flags(&toks(&["-in", "pat"]), 0, cfg, Some("rg"), None, None));
        // -A20 attached numeric
        assert!(validate_flags(&toks(&["-A20", "pat"]), 0, cfg, Some("rg"), None, None));
        // -g in a bundle takes an arg → reject
        assert!(!validate_flags(&toks(&["-ig"]), 0, cfg, Some("rg"), None, None));
    }

    /// The attached-numeric branch splits a flag at BYTE 2, which is a
    /// character boundary only because FLAG_PATTERN (`^-[a-zA-Z0-9_-]`) has
    /// already pinned byte 1 to ASCII. That coupling is invisible at the slice
    /// — loosening the pattern to accept a unicode flag character would turn
    /// `&flag[..2]` into a mid-character split and abort the process from
    /// inside the permission check. These inputs are all plausible model
    /// output: an en-dash pasted from a document, a CJK argument, an emoji in
    /// a search pattern. The test exists to fail if the guard is ever relaxed.
    #[test]
    fn attached_numeric_flag_survives_multibyte() {
        let cfg = &RIPGREP_READ_ONLY_COMMANDS["rg"];
        for flag in [
            "-–20",    // hyphen + EN DASH: bytes 1..4 are one character
            "-—x",     // em dash
            "-한글",    // Hangul
            "-🔥x",    // emoji (4 bytes)
            "-é9",     // 2-byte Latin-1 supplement
        ] {
            // The only requirement is that a decision comes back at all.
            let _ = validate_flags(&toks(&[flag, "pat"]), 0, cfg, Some("rg"), None, None);
            let _ = validate_flags(&toks(&[flag, "pat"]), 0, cfg, Some("grep"), None, None);
        }
        // The ASCII behaviour it was written for is unchanged.
        assert!(validate_flags(&toks(&["-A20", "pat"]), 0, cfg, Some("rg"), None, None));
    }

    #[test]
    fn validate_flags_none_with_equals_rejected() {
        let cfg = &GIT_READ_ONLY_COMMANDS["git log"];
        assert!(!validate_flags(&toks(&["--oneline=yes"]), 0, cfg, Some("git"), None, None));
    }

    #[test]
    fn git_sort_reverse_allowed() {
        // Oracle-verified against the Python implementation: the reverse-sort
        // exception only applies to the `=` form. `--sort -refname` trips the
        // missing-required-argument check FIRST (the next token matches
        // FLAG_PATTERN), so the space form is rejected in both languages.
        let cfg = &GIT_READ_ONLY_COMMANDS["git branch"];
        assert!(!validate_flags(&toks(&["--sort", "-refname", "--list"]), 0, cfg, Some("git"), None, None));
        assert!(validate_flags(&toks(&["--sort=-refname", "--list"]), 0, cfg, Some("git"), None, None));
    }

    #[test]
    fn pyright_double_dash_not_respected() {
        let cfg = &PYRIGHT_READ_ONLY_COMMANDS["pyright"];
        // After --, flags are STILL checked (respects_double_dash=false):
        assert!(!validate_flags(&toks(&["--", "--danger"]), 0, cfg, Some("pyright"), None, None));
        assert!(validate_flags(&toks(&["--", "--stats"]), 0, cfg, Some("pyright"), None, None));
    }

    #[test]
    fn callbacks() {
        let a = |s: &[&str]| toks(s);
        assert!(!git_reflog_cb("", &a(&["show"])));
        assert!(git_reflog_cb("", &a(&["expire"])));
        assert!(!git_remote_show_cb("", &a(&["origin"])));
        assert!(git_remote_show_cb("", &a(&["origin\n"]))); // newline rejected
        assert!(git_remote_show_cb("", &a(&["a", "b"])));
        assert!(!git_remote_cb("", &a(&["-v"])));
        assert!(git_remote_cb("", &a(&["add"])));
        assert!(git_tag_cb("", &a(&["v1.0"]))); // creation
        assert!(!git_tag_cb("", &a(&["-l", "v*"])));
        assert!(git_branch_cb("", &a(&["new-branch"])));
        assert!(!git_branch_cb("", &a(&["--list", "main"])));
        assert!(!git_branch_cb("", &a(&["--merged", "main"]))); // optional-arg flag
        assert!(gh_is_dangerous_cb("", &a(&["https://evil.com"])));
        assert!(gh_is_dangerous_cb("", &a(&["a/b/c"])));
        assert!(!gh_is_dangerous_cb("", &a(&["123"])));
        assert!(gh_is_dangerous_cb("", &a(&["--repo=git@github.com:o/r"])));
        assert!(pyright_cb("", &a(&["--watch"])));
        assert!(!pyright_cb("", &a(&["--stats"])));
    }

    #[test]
    fn unc_forwardslash_lookbehind_equivalence() {
        // These assert the position-by-position scan itself (platform-gated
        // contains_vulnerable_unc_path returns false off-Windows).
        assert!(!unc_forwardslash_search("http://example.com")); // ':' before //
        assert!(unc_forwardslash_search("http://a//b ")); // later // still found
        assert!(unc_forwardslash_search("//server/share"));
        assert!(!unc_forwardslash_search("a/b/c"));
    }
}
