//! Bash Tool for CodeMason — based on Claude Code src/tools/BashTool/.
//! Equivalent to codemason/tools/bash_tool.py.
//!
//! Pairs the tool surface (schema, description, command-security model) with
//! CodeMason's LOCAL command-execution substrate (PersistentShell::exec — an
//! async subprocess with timeout / abort / SIGKILL / background-pipe handling,
//! plus cwd-reset). The security model (command splitting, path containment,
//! dangerous-removal blocking) is ported faithfully; the execution mechanism is
//! CodeMason-specific and preserved.
//!
//! Divergence from the Python port, documented at the call sites: the Python's
//! heredoc-delimiter regex uses a backreference (unsupported by the `regex`
//! crate), reimplemented here as a hand-rolled scan with identical semantics;
//! background streaming uses pipes rather than a PTY (see local_shell_task.rs).

use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::constants::tool_prompts::BASH_DESCRIPTION;
use crate::tools::base::{
    CanUseToolFn, PermissionBehavior, PermissionResult, Tool, ToolResult, ToolStream,
    ValidationResult,
};
use crate::types::query::ToolUseContext;
use crate::utils::log::log_event;
use crate::utils::platform::is_windows;
use crate::utils::state::{get_cwd, get_original_cwd, is_in_directory, set_cwd as set_global_cwd};
use crate::utils::windows_shell::{
    apply_spawn_group, decode_console_output, kill_process_tree, posix_path_to_windows,
    shell_argv, windows_path_to_posix,
};

// === Timeouts (utils/timeouts.ts) ===
const DEFAULT_TIMEOUT_MS: u64 = 120_000; // 2 minutes
const MAX_TIMEOUT_MS: u64 = 600_000; // 10 minutes

/// Display cap for the echoed command (the full command still runs).
const MAX_DISPLAY_COMMAND_CHARS: usize = 400;

/// The exact input_json_schema() output dumped from the Python build.
static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get("Bash")
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

// === Output Type ===

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct BashOutput {
    pub stdout: String,
    pub stdout_lines: usize,
    pub stderr: String,
    pub stderr_lines: usize,
    pub interrupted: bool,
}

// === Command Security Model (bashSecurity.ts / commands.ts / pathValidation.ts) ===

/// Control operators that separate one subcommand from the next. Ordered
/// longest-first so the scanner matches multi-char operators greedily.
const CONTROL_OPERATORS: &[&str] = &["&&", "||", ";;", ";", "|", "&", "(", ")", "\n"];

/// Wrapper commands whose first non-flag operand is the *real* command.
const WRAPPER_COMMANDS: &[&str] = &["time", "nohup", "timeout", "nice", "stdbuf", "env"];

static LINE_CONT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\\+\n").unwrap());

/// Join backslash-newline line continuations (commands.ts:106-139).
///
/// SECURITY: a shell joins continuation tokens with NO space, so `tr\<NL>aceroute`
/// executes as the single word `traceroute`. Only join on an ODD number of
/// backslashes before the newline.
fn join_line_continuations(command: &str) -> String {
    LINE_CONT_RE
        .replace_all(command, |caps: &regex::Captures| {
            let run = &caps[0];
            let backslashes = run.len() - 1; // minus the newline
            if backslashes % 2 == 1 {
                "\\".repeat(backslashes - 1) // consume the escaping backslash + newline
            } else {
                run.to_string() // even: keep as-is (separator)
            }
        })
        .into_owned()
}

/// Remove heredoc bodies so their contents aren't mistaken for commands.
///
/// Hand-rolled (the Python's `<<-?\s*(['\"]?)DELIM\1` uses a backreference the
/// `regex` crate lacks): scan for `<<`/`<<-`, an optionally-quoted identifier,
/// then the matching close quote, then drop through the terminating delimiter
/// line.
fn strip_heredoc_bodies(command: &str) -> String {
    let bytes: Vec<char> = command.chars().collect();
    let mut out = command.to_string();
    // Operate on the string repeatedly (matching the Python while-loop).
    loop {
        let chars: Vec<char> = out.chars().collect();
        let n = chars.len();
        let mut found: Option<(usize, usize, String)> = None; // (start, op_end, delim)
        let mut i = 0;
        while i + 1 < n {
            if chars[i] == '<' && chars[i + 1] == '<' {
                let start = i;
                let mut j = i + 2;
                if j < n && chars[j] == '-' {
                    j += 1;
                }
                while j < n && chars[j].is_whitespace() && chars[j] != '\n' {
                    j += 1;
                }
                let quote = if j < n && (chars[j] == '\'' || chars[j] == '"') {
                    let q = chars[j];
                    j += 1;
                    Some(q)
                } else {
                    None
                };
                let id_start = j;
                if j < n && (chars[j].is_ascii_alphabetic() || chars[j] == '_') {
                    j += 1;
                    while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                        j += 1;
                    }
                    let delim: String = chars[id_start..j].iter().collect();
                    // require the matching close quote if one was opened
                    let ok = match quote {
                        Some(q) => j < n && chars[j] == q,
                        None => true,
                    };
                    if ok {
                        if quote.is_some() {
                            j += 1;
                        }
                        found = Some((start, j, delim));
                        break;
                    }
                }
                i = start + 2;
                continue;
            }
            i += 1;
        }
        let Some((start, op_end, delim)) = found else {
            break;
        };
        // find terminating delimiter line after op_end
        let rest: String = chars[op_end..].iter().collect();
        let end_re = Regex::new(&format!(r"\n[ \t]*{}[ \t]*(\n|$)", regex::escape(&delim))).unwrap();
        let prefix: String = chars[..start].iter().collect();
        if let Some(m) = end_re.find(&rest) {
            let after: String = rest[m.end()..].to_string();
            out = format!("{prefix} {after}");
        } else {
            // Unterminated heredoc: drop everything from the operator on.
            out = prefix;
            break;
        }
    }
    let _ = bytes;
    out
}

/// Split a command string into subcommands on shell control operators.
fn split_on_operators(command: &str) -> Vec<String> {
    // Join odd-backslash line continuations BEFORE any newline tokenization.
    let command = join_line_continuations(command);
    let command = strip_heredoc_bodies(&command);

    let mut subcommands: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let chars: Vec<char> = command.chars().collect();
    let n = chars.len();
    let mut i = 0;

    let flush = |current: &mut String, subcommands: &mut Vec<String>| {
        let text = current.trim().to_string();
        if !text.is_empty() {
            subcommands.push(text);
        }
        current.clear();
    };

    while i < n {
        let ch = chars[i];
        // Quote handling (a backslash-escaped quote is a literal char).
        // NOT inside single quotes: bash treats a backslash there as an ordinary
        // character, so it cannot escape the closing quote. Running this branch
        // first made the validator read `echo 'a\'; rm -rf ~/x #'` as one
        // `echo` subcommand — it thought \' was an escaped quote and stayed in
        // the string — while bash closes at the second quote and runs the `rm`.
        // tokens[0] was then `echo`, not a path command, so the
        // dangerous-removal and containment checks never ran on the `rm`.
        if ch == '\\' && i + 1 < n && !in_single {
            current.push(ch);
            current.push(chars[i + 1]);
            i += 2;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            current.push(ch);
            i += 1;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            current.push(ch);
            i += 1;
            continue;
        }
        if in_single || in_double {
            current.push(ch);
            i += 1;
            continue;
        }
        // Comment: everything to end-of-line is ignored (matches bash).
        if ch == '#' && (current.is_empty() || current.chars().last().unwrap().is_whitespace()) {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        let mut matched = false;
        for op in CONTROL_OPERATORS {
            let op_chars: Vec<char> = op.chars().collect();
            if i + op_chars.len() <= n && chars[i..i + op_chars.len()] == op_chars[..] {
                flush(&mut current, &mut subcommands);
                i += op_chars.len();
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        current.push(ch);
        i += 1;
    }
    flush(&mut current, &mut subcommands);
    subcommands
}

static REDIRECT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d*(>>|>&|>|<<<|<<|<)").unwrap());

/// Drop redirection operators and their targets (commands.ts).
fn strip_redirections(tokens: Vec<String>) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        if let Some(m) = REDIRECT_RE.find(tok) {
            // If the operator is fused with its target (e.g. `>file`, `2>&1`),
            // the whole token is consumed; otherwise also consume the following
            // target token.
            if tok.as_str() == m.as_str() {
                i += 2; // operator + separate target
            } else {
                i += 1; // fused operator+target
            }
            continue;
        }
        result.push(tok.clone());
        i += 1;
    }
    result
}

static ASSIGN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*=").unwrap());
static DURATION_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\d+(?:\.\d+)?[smhd]?$").unwrap());

/// Strip leading `VAR=val` assignments and wrapper commands.
fn strip_env_and_wrappers(mut tokens: Vec<String>) -> Vec<String> {
    let mut changed = true;
    while changed && !tokens.is_empty() {
        changed = false;
        // Strip leading VAR=val assignments.
        while !tokens.is_empty() && ASSIGN_RE.is_match(&tokens[0]) && !tokens[0].starts_with('-') {
            tokens.remove(0);
            changed = true;
        }
        if tokens.is_empty() {
            break;
        }
        let root = basename(&tokens[0]);
        if WRAPPER_COMMANDS.contains(&root.as_str()) {
            let rest = tokens[1..].to_vec();
            let mut j = 0;
            while j < rest.len() {
                let a = &rest[j];
                if a == "--" {
                    j += 1;
                    break;
                }
                if a.starts_with('-') {
                    j += 1;
                    continue;
                }
                // A bare numeric/duration operand belongs to the wrapper.
                if DURATION_RE.is_match(a) {
                    j += 1;
                    continue;
                }
                break;
            }
            tokens = rest[j..].to_vec();
            changed = true;
        }
    }
    tokens
}

/// Resolve a command token to its bare name (closes the abs-path bypass).
fn basename(token: &str) -> String {
    let token = token.trim().trim_matches(|c| c == '\'' || c == '"');
    if token.is_empty() {
        return String::new();
    }
    std::path::Path::new(token)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| token.to_string())
}

/// Split a single subcommand into whitespace-separated tokens (quote-aware).
fn tokenize(subcommand: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let chars: Vec<char> = subcommand.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        // Same rule as split_on_operators: a backslash inside single quotes is
        // a literal character in bash and cannot escape the closing quote.
        if ch == '\\' && i + 1 < n && !in_single {
            current.push(chars[i + 1]);
            i += 2;
            continue;
        }
        if ch == '\'' && !in_double {
            in_single = !in_single;
            i += 1;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            i += 1;
            continue;
        }
        if ch.is_whitespace() && !in_single && !in_double {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            i += 1;
            continue;
        }
        current.push(ch);
        i += 1;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Return each subcommand as a cleaned token list (root-extractable).
pub fn get_command_subcommands(command: &str) -> Vec<Vec<String>> {
    let mut result = Vec::new();
    for sub in split_on_operators(command) {
        let tokens = tokenize(&sub);
        let tokens = strip_redirections(tokens);
        let tokens = strip_env_and_wrappers(tokens);
        if !tokens.is_empty() {
            result.push(tokens);
        }
    }
    result
}

// === Path-constraint security (pathValidation.ts checkPathConstraints) ===

const WRITE_PATH_COMMANDS: &[&str] = &["rm", "rmdir", "mv", "cp", "mkdir", "touch", "sed"];
const READ_PATH_COMMANDS: &[&str] = &[
    "cd", "ls", "find", "cat", "head", "tail", "sort", "uniq", "wc", "cut", "paste", "column",
    "tr", "file", "stat", "diff", "awk", "strings", "hexdump", "od", "base64", "nl", "grep", "rg",
    "git", "jq", "sha256sum", "sha1sum", "md5sum",
];

fn is_write_path_command(base: &str) -> bool {
    WRITE_PATH_COMMANDS.contains(&base)
}
fn is_supported_path_command(base: &str) -> bool {
    WRITE_PATH_COMMANDS.contains(&base) || READ_PATH_COMMANDS.contains(&base)
}

static PROC_SUB_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r">>\s*>\s*\(|>\s*>\s*\(|<\s*\(").unwrap());
static GLOB_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[*?\[\]{}]").unwrap());
static WIN_DRIVE_ROOT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z]:/?$").unwrap());
static WIN_DRIVE_CHILD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z]:/[^/]+$").unwrap());
static MULTI_SLASH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"/+").unwrap());

fn home_dir_str() -> String {
    dirs::home_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

/// expandTilde: only bare ~ and ~/ resolve to $HOME.
fn expand_tilde(path: &str) -> String {
    if path == "~" {
        home_dir_str()
    } else if let Some(rest) = path.strip_prefix("~/") {
        format!("{}/{rest}", home_dir_str())
    } else {
        path.to_string()
    }
}

/// isDangerousRemovalPath (pathValidation.ts).
pub fn is_dangerous_removal_path(resolved_path: &str) -> bool {
    let fwd = MULTI_SLASH_RE.replace_all(&resolved_path.replace('\\', "/"), "/").into_owned();
    if fwd == "*" || fwd.ends_with("/*") {
        return true;
    }
    let normalized = if fwd == "/" { fwd.clone() } else { fwd.trim_end_matches('/').to_string() };
    if normalized == "/" || normalized.is_empty() {
        return true;
    }
    if WIN_DRIVE_ROOT_RE.is_match(&normalized) {
        return true;
    }
    let home = MULTI_SLASH_RE
        .replace_all(&home_dir_str().replace('\\', "/"), "/")
        .trim_end_matches('/')
        .to_string();
    if normalized == home {
        return true;
    }
    // Direct children of root: /usr, /tmp, /etc (but not /usr/local).
    if std::path::Path::new(&normalized).parent().map(|p| p.to_string_lossy() == "/").unwrap_or(false) {
        return true;
    }
    if WIN_DRIVE_CHILD_RE.is_match(&normalized) {
        return true;
    }
    false
}

/// filterOutFlags: positional args, honoring the POSIX `--` delimiter.
fn filter_out_flags(args: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut after_dd = false;
    for arg in args {
        if after_dd {
            result.push(arg.clone());
        } else if arg == "--" {
            after_dd = true;
        } else if !arg.starts_with('-') {
            result.push(arg.clone());
        }
    }
    result
}

/// PATH_EXTRACTORS[command]: pull the path operands for a path command.
fn extract_paths(command: &str, args: &[String]) -> Vec<String> {
    match command {
        "cd" => {
            if args.is_empty() {
                vec![home_dir_str()]
            } else {
                vec![args.join(" ")]
            }
        }
        "ls" | "find" => {
            let paths = filter_out_flags(args);
            if paths.is_empty() {
                vec![".".to_string()]
            } else {
                paths
            }
        }
        "tr" => {
            let has_delete = args.iter().any(|a| {
                a == "-d" || a == "--delete" || (a.starts_with('-') && a.contains('d'))
            });
            let non_flags = filter_out_flags(args);
            let skip = if has_delete { 1 } else { 2 };
            non_flags.into_iter().skip(skip).collect()
        }
        _ => filter_out_flags(args),
    }
}

fn normpath(p: &str) -> String {
    // Lexical normalization matching os.path.normpath (no symlink resolution).
    let mut out: Vec<String> = Vec::new();
    let is_abs = p.starts_with('/');
    for comp in p.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                if matches!(out.last().map(|s| s.as_str()), Some("..")) || out.is_empty() {
                    if !is_abs {
                        out.push("..".to_string());
                    }
                } else {
                    out.pop();
                }
            }
            c => out.push(c.to_string()),
        }
    }
    let joined = out.join("/");
    if is_abs {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// validatePath: reject tilde-variants / shell-expansion / UNC, block globs in
/// write ops, and require containment in the allowed working directory.
/// Returns a deny/ask reason string, or None if allowed.
fn validate_path(path: &str, op_type: &str) -> Option<String> {
    let mut clean = expand_tilde(path.trim().trim_matches(|c| c == '\'' || c == '"'));
    if clean.is_empty() {
        return None;
    }
    // UNC network paths.
    if clean.starts_with("\\\\")
        || (clean.starts_with("//") && clean.chars().count() > 2 && clean.chars().nth(2) != Some('/'))
    {
        return Some("UNC network paths require manual approval".to_string());
    }
    // Unexpanded tilde variants (~user, ~+, ~-, ~N).
    if clean.starts_with('~') {
        return Some(
            "Tilde expansion variants (~user, ~+, ~-) in paths require manual approval".to_string(),
        );
    }
    // Shell expansion syntax (TOCTOU): $VAR, ${}, $(), %VAR%, =cmd.
    if clean.contains('$') || clean.contains('%') || clean.starts_with('=') {
        return Some("Shell expansion syntax in paths requires manual approval".to_string());
    }
    // Globs: blocked outright for write/create.
    if GLOB_RE.is_match(&clean) {
        if op_type == "write" || op_type == "create" {
            return Some(
                "Glob patterns are not allowed in write operations. Please specify an exact file path."
                    .to_string(),
            );
        }
        // Read globs: validate the base directory they would expand in.
        clean = std::path::Path::new(&clean)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".".to_string());
    }
    let full = crate::utils::state::resolve_for_containment(&clean, &get_cwd());
    let full = normpath(&full);
    if !is_in_directory(&full, &get_original_cwd()) {
        return Some(format!(
            "'{full}' is outside the allowed working directory for this session ({}).",
            get_original_cwd()
        ));
    }
    None
}

static OUTPUT_REDIR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:&|\d+)?(>>|&>|>)\s*([^\s;|&()<>]+)").unwrap());

/// extractOutputRedirections (commands.ts).
fn extract_output_redirections(command: &str) -> (Vec<(String, String)>, bool) {
    let mut redirections = Vec::new();
    let mut has_dangerous = false;
    for caps in OUTPUT_REDIR_RE.captures_iter(command) {
        let op_raw = &caps[1];
        let target = caps[2].to_string();
        if target.starts_with('&') {
            continue; // FD duplication (>&1), not a file target
        }
        if target.contains('$') || target.contains('`') || target.contains('%') {
            has_dangerous = true;
            continue;
        }
        let operator = if op_raw == ">>" { ">>" } else { ">" };
        redirections.push((target, operator.to_string()));
    }
    (redirections, has_dangerous)
}

/// validateOutputRedirections.
fn validate_output_redirections(
    redirections: &[(String, String)],
    compound_has_cd: bool,
) -> Option<PermissionResult> {
    if compound_has_cd && !redirections.is_empty() {
        return Some(PermissionResult::ask_with(
            "Commands that change directories and write via output redirection require explicit approval (the final working directory can't be determined safely).",
        ));
    }
    for (target, _op) in redirections {
        if target == "/dev/null" {
            continue;
        }
        if let Some(reason) = validate_path(target, "create") {
            return Some(PermissionResult::ask_with(format!(
                "Output redirection to '{target}' was blocked: {reason}"
            )));
        }
    }
    None
}

/// checkPathConstraints port → a PermissionResult.
pub fn check_path_constraints(command: &str) -> PermissionResult {
    let joined = join_line_continuations(command);

    // Process substitution can execute arbitrary commands.
    if PROC_SUB_RE.is_match(&joined) {
        return PermissionResult::ask_with(
            "Process substitution (>(...) or <(...)) can execute arbitrary commands and requires manual approval.",
        );
    }

    let (redirections, has_dangerous) = extract_output_redirections(&joined);
    if has_dangerous {
        return PermissionResult::ask_with(
            "Shell expansion syntax in redirection paths requires manual approval.",
        );
    }

    let subcommands = get_command_subcommands(command);
    let compound_has_cd = subcommands
        .iter()
        .any(|tokens| !tokens.is_empty() && basename(&tokens[0]) == "cd");

    if let Some(redir) = validate_output_redirections(&redirections, compound_has_cd) {
        return redir;
    }

    for tokens in &subcommands {
        if tokens.is_empty() {
            continue;
        }
        let base = basename(&tokens[0]);
        if !is_supported_path_command(&base) {
            continue;
        }
        let args = &tokens[1..];

        // Dangerous removal (rm/rmdir).
        if base == "rm" || base == "rmdir" {
            for p in extract_paths(&base, args) {
                let clean = expand_tilde(p.trim().trim_matches(|c| c == '\'' || c == '"'));
                let full = if std::path::Path::new(&clean).is_absolute() {
                    clean.clone()
                } else {
                    format!("{}/{clean}", get_cwd())
                };
                if is_dangerous_removal_path(&normpath(&full)) {
                    return PermissionResult::ask_with(format!(
                        "Dangerous {base} operation detected: '{full}'. This would remove a critical directory and requires explicit approval."
                    ));
                }
            }
        }

        let op_type = if is_write_path_command(&base) { "write" } else { "read" };
        for p in extract_paths(&base, args) {
            if let Some(reason) = validate_path(&p, op_type) {
                return PermissionResult::ask_with(format!("{base}: {reason}"));
            }
        }
    }

    PermissionResult { behavior: PermissionBehavior::Allow, message: "All path commands validated".to_string() }
}

// === Output Truncation ===

/// Return (total_lines, content) with NO head truncation (the query() tool-
/// result layer persists + previews large output).
fn format_output(content: &str) -> (usize, String) {
    let total_lines = content.matches('\n').count() + 1;
    (total_lines, content.to_string())
}

static SEGMENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\|\||&&|[|;&]").unwrap());

struct CommandInterpretation {
    is_error: bool,
    message: Option<String>,
}

/// Port of BashTool/commandSemantics.ts interpretCommandResult.
fn interpret_command_result(command: &str, exit_code: i32) -> CommandInterpretation {
    let segments: Vec<&str> = SEGMENT_RE.split(command).collect();
    let last = segments.last().map(|s| s.trim()).unwrap_or(command);
    let base = last
        .split_whitespace()
        .next()
        .unwrap_or("")
        .rsplit('/')
        .next()
        .unwrap_or("");

    match base {
        "grep" | "rg" => CommandInterpretation {
            is_error: exit_code >= 2,
            message: (exit_code == 1).then(|| "No matches found".to_string()),
        },
        "find" => CommandInterpretation {
            is_error: exit_code >= 2,
            message: (exit_code == 1).then(|| "Some directories were inaccessible".to_string()),
        },
        "diff" => CommandInterpretation {
            is_error: exit_code >= 2,
            message: (exit_code == 1).then(|| "Files differ".to_string()),
        },
        "test" | "[" => CommandInterpretation {
            is_error: exit_code >= 2,
            message: (exit_code == 1).then(|| "Condition is false".to_string()),
        },
        _ => CommandInterpretation {
            is_error: exit_code != 0,
            message: (exit_code != 0).then(|| format!("Command failed with exit code {exit_code}")),
        },
    }
}

// =====================================================================
// LOCAL SUBSTRATE (PRESERVED) — async subprocess execution
// =====================================================================

/// The result of one foreground command execution.
struct ExecResult {
    stdout: String,
    stderr: String,
    code: i32,
    interrupted: bool,
    backgrounded: bool,
}

/// Singleton persistent shell cwd (the shell's cwd IS the session cwd).
static SHELL_CWD: Lazy<parking_lot::Mutex<String>> = Lazy::new(|| parking_lot::Mutex::new(get_cwd()));

fn shell_cwd() -> String {
    SHELL_CWD.lock().clone()
}
fn set_shell_cwd(path: &str) {
    *SHELL_CWD.lock() = path.to_string();
}

const MAX_CAPTURE: usize = 20 * 1024 * 1024; // 20 MiB per stream

fn extend_capped(buf: &mut Vec<u8>, capped: &mut bool, chunk: &[u8]) {
    if buf.len() < MAX_CAPTURE {
        let room = MAX_CAPTURE - buf.len();
        buf.extend_from_slice(&chunk[..chunk.len().min(room)]);
        if buf.len() >= MAX_CAPTURE && !*capped {
            *capped = true;
            buf.extend_from_slice(b"\n... [output truncated: exceeded 20 MiB] ...\n");
        }
    }
}

async fn read_chunk<R: AsyncReadExt + Unpin>(reader: &mut Option<R>) -> Option<Vec<u8>> {
    let Some(r) = reader.as_mut() else {
        return None;
    };
    let mut buf = [0u8; 4096];
    match r.read(&mut buf).await {
        Ok(0) | Err(_) => {
            *reader = None;
            None
        }
        Ok(n) => Some(buf[..n].to_vec()),
    }
}

/// Execute a command and return the result. Reads stdout/stderr incrementally
/// so a timeout still has every byte so far and can hand the live process to
/// the background instead of killing the work.
async fn shell_exec(
    command: &str,
    timeout_ms: u64,
    abort: Option<CancellationToken>,
    description: Option<&str>,
    tool_use_id: Option<&str>,
) -> ExecResult {
    let timeout_sec = timeout_ms as f64 / 1000.0;

    // cwd PERSISTENCE sentinel (pwd -P written to a temp file off stdout/stderr).
    let pwd_file = std::env::temp_dir().join(format!("cm-pwd-{}", uuid::Uuid::new_v4().simple()));
    let _ = std::fs::File::create(&pwd_file);
    let pwd_file_str = pwd_file.to_string_lossy().into_owned();
    let pwd_redirect = if is_windows() { windows_path_to_posix(&pwd_file_str) } else { pwd_file_str.clone() };
    let quoted = shlex::try_quote(&pwd_redirect).map(|c| c.into_owned()).unwrap_or_else(|_| pwd_redirect.clone());
    let wrapped = format!("{command}\n__CM_EC=$?\ncommand pwd -P > {quoted} 2>/dev/null\nexit $__CM_EC\n");

    let cwd = shell_cwd();
    let mut cmd = match shell_argv(&wrapped) {
        Ok(Some(argv)) => {
            let mut c = Command::new(&argv[0]);
            c.args(&argv[1..]);
            c
        }
        Ok(None) => {
            let mut c = Command::new("/bin/sh");
            c.arg("-c").arg(&wrapped);
            c
        }
        // git-bash discovery failed, which only happens on Windows. This used
        // to share the POSIX arm through a catch-all `_`, so the fallback ran
        // /bin/sh — a path that does not exist there — and the user got
        // "The system cannot find the path specified (os error 3)" instead of
        // the message written for exactly this situation, which names the
        // download and the CODEMASON_GIT_BASH_PATH override. Measured on a
        // host with discovery forced to fail: the model retried the opaque
        // failure until the turn timed out, because nothing in it says what to
        // do. Python cannot lose this — its shell_argv RAISES and the exception
        // propagates — so the catch-all was a port regression, not shared
        // behaviour.
        Err(e) => {
            let _ = std::fs::remove_file(&pwd_file);
            return ExecResult {
                stdout: String::new(),
                stderr: e,
                code: -1,
                interrupted: false,
                backgrounded: false,
            };
        }
    };
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(&cwd)
        .env("TERM", "dumb")
        // A pipe, not a tty: a python child block-buffers without this, and
        // when the command times out and is adopted as a background task its
        // output file shows nothing while it visibly works.
        .env("PYTHONUNBUFFERED", "1");
    apply_spawn_group(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&pwd_file);
            return ExecResult { stdout: String::new(), stderr: e.to_string(), code: -1, interrupted: false, backgrounded: false };
        }
    };

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let mut out_buf: Vec<u8> = Vec::new();
    let mut err_buf: Vec<u8> = Vec::new();
    let mut out_capped = false;
    let mut err_capped = false;
    let mut interrupted = false;
    let mut aborted_handled = false;
    let mut timed_out = false;

    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);

    // _force_close_pipes_after_exit parity (bash_tool.py:826-843): the loop
    // below ends on double pipe-EOF, but a background grandchild ('cmd &')
    // inherits the pipes and keeps them open long after the shell exits, so
    // EOF never comes and the tool blocks for the grandchild's lifetime.
    // Python polls returncode every 0.1s and closes the pipe transports 0.5s
    // after exit; here the poll tick detects exit and the flush deadline drops
    // our read ends, which ends the loop.
    let mut pipe_close_at: Option<tokio::time::Instant> = None;

    loop {
        if stdout.is_none() && stderr.is_none() {
            break;
        }
        if pipe_close_at.is_none() {
            if let Ok(Some(_)) = child.try_wait() {
                pipe_close_at =
                    Some(tokio::time::Instant::now() + Duration::from_millis(500));
            }
        }
        let poll_or_flush_at = pipe_close_at
            .unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_millis(100));
        let abort_fut = {
            let abort = abort.clone();
            async move {
                match abort {
                    Some(t) => t.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            }
        };
        tokio::select! {
            biased;
            _ = abort_fut, if !aborted_handled => {
                interrupted = true;
                aborted_handled = true;
                // ESC cancel = "stop now, discard the result": SIGKILL the group.
                if let Some(pid) = child.id() { kill_process_tree(pid, None); }
            }
            _ = tokio::time::sleep_until(deadline), if !timed_out => {
                timed_out = true;
                // Only hand off if the process is genuinely still running.
                let still_running = matches!(child.try_wait(), Ok(None));
                if still_running && !interrupted {
                    let combined = {
                        let mut v = out_buf.clone();
                        if !out_buf.is_empty() && !err_buf.is_empty() { v.push(b'\n'); }
                        v.extend_from_slice(&err_buf);
                        v
                    };
                    let (task_id, out_path) = crate::tasks::local_shell_task::adopt_timed_out_process(
                        child, stdout.take(), stderr.take(), combined, description, tool_use_id, "bash",
                    );
                    let _ = std::fs::remove_file(&pwd_file);
                    return ExecResult {
                        stdout: format!(
                            "Command timed out after {timeout_sec}s and was moved to the background.\nTask ID: {task_id}\nOutput is being written to: {}\nUse Read to view it; you will be notified on completion.",
                            out_path.display()
                        ),
                        stderr: String::new(),
                        code: 0,
                        interrupted: false,
                        backgrounded: true,
                    };
                }
            }
            c = read_chunk(&mut stdout), if stdout.is_some() => {
                if let Some(chunk) = c { extend_capped(&mut out_buf, &mut out_capped, &chunk); }
            }
            c = read_chunk(&mut stderr), if stderr.is_some() => {
                if let Some(chunk) = c { extend_capped(&mut err_buf, &mut err_capped, &chunk); }
            }
            _ = tokio::time::sleep_until(poll_or_flush_at) => {
                if pipe_close_at.is_some() {
                    // Shell exited and the flush window passed: close our read
                    // ends so the loop ends even while a grandchild holds the
                    // write ends open.
                    stdout = None;
                    stderr = None;
                }
                // else: a 100ms poll tick — exit is re-checked at loop top.
            }
        }
    }

    let returncode = child.wait().await.ok().and_then(|s| s.code()).unwrap_or(-1);

    // Adopt the post-command cwd so `cd` persists to the next command.
    if let Ok(raw) = std::fs::read_to_string(&pwd_file) {
        let mut new_cwd = raw.trim().to_string();
        if !new_cwd.is_empty() && is_windows() {
            new_cwd = posix_path_to_windows(&new_cwd);
        }
        if !new_cwd.is_empty() && std::path::Path::new(&new_cwd).is_dir() {
            set_shell_cwd(&new_cwd);
            set_global_cwd(&new_cwd);
        }
    }
    let _ = std::fs::remove_file(&pwd_file);

    ExecResult {
        stdout: decode_console_output(&out_buf),
        stderr: decode_console_output(&err_buf),
        code: returncode,
        interrupted,
        backgrounded: false,
    }
}

// =====================================================================
// END LOCAL SUBSTRATE
// =====================================================================

static HEREDOC_DISPLAY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)^(.*?)"?\$\(cat <<'EOF'\n(.*?)\n\s*EOF\n\s*\)"(.*)$"#).unwrap()
});

/// Render the result for the assistant (BashTool.tsx mapToolResult...).
fn render_result(result: &BashOutput) -> String {
    let mut error_message = result.stderr.trim().to_string();
    if result.interrupted {
        if !error_message.is_empty() {
            error_message.push('\n');
        }
        error_message.push_str("<error>Command was aborted before completion</error>");
    }
    let stdout = result.stdout.trim();
    let has_both = !stdout.is_empty() && !error_message.is_empty();
    format!("{stdout}{}{}", if has_both { "\n" } else { "" }, error_message.trim())
}

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "Bash"
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    fn max_result_size_chars(&self) -> f64 {
        30_000.0
    }

    async fn description(&self, input: &Value) -> String {
        // BashTool.tsx passes `description || command`.
        let desc = input.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        if !desc.is_empty() {
            desc.to_string()
        } else if !command.is_empty() {
            command.to_string()
        } else {
            "Execute bash command".to_string()
        }
    }

    async fn prompt(&self) -> String {
        BASH_DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        false
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Bash".to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        if command.trim().is_empty() {
            return ValidationResult::failure("Command cannot be empty");
        }
        ValidationResult::success()
    }

    async fn check_permissions(&self, input: &Value, _context: &ToolUseContext) -> PermissionResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let result = check_path_constraints(command);
        // Bash's check only ever returns 'ask'/'allow'; forward a 'deny' too if
        // ever introduced, else fall through to Bash's default prompt.
        match result.behavior {
            PermissionBehavior::Deny | PermissionBehavior::Ask => result,
            PermissionBehavior::Allow => PermissionResult::ask(),
        }
    }

    /// "Running <description or command>" (BashTool getActivityDescription).
    ///
    /// The model's own one-line description wins over the command text — it is
    /// what the command is FOR, which is the useful thing to read while
    /// waiting.
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        let command = input
            .and_then(|i| i.get("command"))
            .and_then(|v| v.as_str())
            .filter(|c| !c.is_empty());
        let Some(command) = command else {
            return Some("Running command".to_string());
        };
        let described = input
            .and_then(|i| i.get("description"))
            .and_then(|v| v.as_str())
            .filter(|d| !d.is_empty())
            .map(|d| d.to_string())
            .unwrap_or_else(|| {
                crate::utils::formatting::truncate(
                    command,
                    crate::utils::formatting::TOOL_SUMMARY_MAX_LENGTH,
                )
            });
        Some(format!("Running {described}"))
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let mut command = input.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string();
        // Clean up HEREDOC patterns for display.
        if command.contains("$(cat <<'EOF'") {
            if let Some(caps) = HEREDOC_DISPLAY_RE.captures(&command) {
                let prefix = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
                let content = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim();
                let suffix = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim();
                command = format!("{prefix} \"{content}\"{suffix}");
            }
        }
        let command = command.trim();
        let command = if command.chars().count() > MAX_DISPLAY_COMMAND_CHARS {
            // Show the END as well as the beginning. call() runs the whole
            // string, and this rendering is the entire basis for the user's
            // consent — a trailing "..." after 400 plausible characters hid
            // whatever came next, which is exactly where an appended
            // `; curl ... | sh` would sit. The heredoc rewrite above runs
            // before this and can shorten the visible prefix, pushing more past
            // the cut. Char-wise throughout, so a multi-byte tail is safe.
            const TAIL_CHARS: usize = 120;
            let all: Vec<char> = command.chars().collect();
            let head: String =
                all[..MAX_DISPLAY_COMMAND_CHARS - TAIL_CHARS].iter().collect();
            let tail: String = all[all.len() - TAIL_CHARS..].iter().collect();
            let hidden = all.len() - (MAX_DISPLAY_COMMAND_CHARS - TAIL_CHARS) - TAIL_CHARS;
            format!("{head}\n… ({hidden} chars hidden) …\n{tail}")
        } else {
            command.to_string()
        };
        let lines: Vec<&str> = command.split('\n').filter(|l| !l.trim().is_empty()).collect();
        lines
            .iter()
            .enumerate()
            .map(|(i, line)| if i == 0 { format!("$ {line}") } else { format!("  {line}") })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_result_for_assistant(&self, result: &Value) -> String {
        let out: BashOutput = serde_json::from_value(result.clone()).unwrap_or_default();
        render_result(&out)
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut timeout = input.get("timeout").and_then(|v| v.as_u64()).unwrap_or(0);
            if timeout == 0 {
                timeout = DEFAULT_TIMEOUT_MS;
            }
            if timeout > MAX_TIMEOUT_MS {
                timeout = MAX_TIMEOUT_MS;
            }

            let tool_use_id = context.tool_use_id.clone().or_else(|| context.message_id.clone());
            let description = input.get("description").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());

            // run_in_background: spawn detached, return immediately.
            if input.get("run_in_background").and_then(|v| v.as_bool()).unwrap_or(false) {
                let desc = description.clone().unwrap_or_else(|| command.clone());
                let (task_id, output_path) = crate::tasks::local_shell_task::spawn_shell_task(
                    &command, &shell_cwd(), Some(&desc), tool_use_id.as_deref(), "bash",
                );
                let info = format!(
                    "Command running in background with ID: {task_id}. Output is being written to: {}\nUse Read to read the output later.",
                    output_path.display()
                );
                yield Ok(ToolResult::Result {
                    result_for_assistant: info,
                    data: Some(serde_json::json!({
                        "backgroundTaskId": task_id,
                        "output_path": output_path.to_string_lossy(),
                    })),
                });
                return;
            }

            let abort = Some(context.abort_controller.clone());
            let desc_for_exec = description.clone().unwrap_or_else(|| command.clone());
            // Say that the wait is bounded, and by what. A foreground command
            // runs up to `timeout` (2 minutes by default) before it is moved
            // to the background — worth knowing before deciding to interrupt.
            crate::utils::tokens::set_wait_note(Some(format!(
                "running, moves to background at {}s",
                timeout / 1000
            )));
            let _clear = crate::utils::tokens::WaitNoteGuard;
            let result = shell_exec(&command, timeout, abort, Some(&desc_for_exec), tool_use_id.as_deref()).await;

            let mut stdout = result.stdout.trim().to_string();
            let mut stderr = result.stderr.trim().to_string();

            let interp = interpret_command_result(&command, result.code);
            if interp.is_error {
                if let Some(msg) = &interp.message {
                    stderr.push('\n');
                    stderr.push_str(msg);
                }
            } else if result.code != 0 && interp.message.is_some() && stdout.is_empty() && stderr.is_empty() {
                stderr.push_str(interp.message.as_deref().unwrap_or(""));
            }

            // Reset cwd if it drifted outside the original directory.
            if !is_in_directory(&shell_cwd(), &get_original_cwd()) {
                set_shell_cwd(&get_original_cwd());
                set_global_cwd(&get_original_cwd());
                stderr = format!("{}\nShell cwd was reset to {}", stderr.trim(), get_original_cwd());
                log_event("bash_tool_reset_to_original_dir", None);
            }

            let (stdout_lines, stdout_content) = format_output(&stdout);
            let (stderr_lines, stderr_content) = format_output(&stderr);
            let data = BashOutput {
                stdout: stdout_content,
                stdout_lines,
                stderr: stderr_content,
                stderr_lines,
                interrupted: result.interrupted,
            };
            let _ = result.backgrounded;

            yield Ok(ToolResult::Result {
                result_for_assistant: render_result(&data),
                data: Some(serde_json::to_value(&data).unwrap_or(Value::Null)),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The splitter must agree with bash about quoting, or the path and
    /// dangerous-removal analysis runs on a different command than the shell
    /// executes. A backslash inside single quotes is literal to bash and cannot
    /// escape the closing quote.
    #[test]
    fn splitter_matches_bash_single_quote_rules() {
        // bash runs TWO commands here: `echo 'a\'` then the rm. Reading \' as
        // an escaped quote kept the validator inside the string, yielding one
        // `echo` subcommand — and `echo` is not a path command, so the rm was
        // never analysed at all.
        let got = split_on_operators(r"echo 'a\'; rm -rf /home/user/proj");
        assert_eq!(got.len(), 2, "expected two subcommands, got {got:?}");
        assert!(got[1].starts_with("rm "), "the rm must be its own subcommand: {got:?}");

        // A backslash inside DOUBLE quotes still escapes, as in bash.
        assert_eq!(split_on_operators(r#"echo "a\"b""#).len(), 1);
        // And an ordinary quoted string is still one command.
        assert_eq!(split_on_operators("echo 'hello; world'").len(), 1);
        assert_eq!(split_on_operators("ls; rm -rf /tmp/x").len(), 2);
    }

    #[test]
    fn schema_is_verbatim() {
        assert_eq!(SCHEMA.get("title").and_then(|t| t.as_str()), Some("BashInput"));
        assert!(SCHEMA.get("properties").and_then(|p| p.get("run_in_background")).is_some());
    }

    #[test]
    fn splits_on_all_control_operators() {
        let subs = get_command_subcommands("a && b | c ; d");
        let roots: Vec<String> = subs.iter().map(|t| basename(&t[0])).collect();
        assert_eq!(roots, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn line_continuation_join_is_security_correct() {
        // tr\<NL>aceroute must become the single word `traceroute`.
        let joined = join_line_continuations("tr\\\naceroute");
        assert_eq!(joined, "traceroute");
    }

    #[test]
    fn strips_env_and_wrappers_abs_path() {
        let subs = get_command_subcommands("env FOO=1 timeout 5 /usr/bin/curl x");
        assert_eq!(basename(&subs[0][0]), "curl");
    }

    #[test]
    fn dangerous_removal_paths() {
        assert!(is_dangerous_removal_path("/"));
        assert!(is_dangerous_removal_path("/usr"));
        assert!(is_dangerous_removal_path("/*"));
        assert!(!is_dangerous_removal_path("/usr/local/share"));
    }

    #[test]
    fn heredoc_body_stripped() {
        let out = strip_heredoc_bodies("cat <<EOF\nsecret rm -rf /\nEOF\necho done");
        assert!(!out.contains("rm -rf"));
        assert!(out.contains("echo done"));
    }

    #[test]
    fn interpret_grep_no_match_is_not_error() {
        let r = interpret_command_result("grep foo file", 1);
        assert!(!r.is_error);
        assert_eq!(r.message.as_deref(), Some("No matches found"));
        let r2 = interpret_command_result("grep foo file", 2);
        assert!(r2.is_error);
    }

    #[test]
    fn output_redirection_target_extracted() {
        let (redirs, dangerous) = extract_output_redirections("echo x > out.txt 2>&1");
        assert!(!dangerous);
        assert_eq!(redirs, vec![("out.txt".to_string(), ">".to_string())]);
    }
}
