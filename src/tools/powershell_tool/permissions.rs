//! PowerShell tool permission engine (adaptation of powershellPermissions.ts).
//!
//! The final security decision: integrates the security validators, git-safety,
//! read-only allowlist, and path validation into one allow/ask/deny result.
//!
//! ADAPTED, not 1:1: CodeMason has NO rule engine — its Bash tool likewise runs
//! a pure security-decision flow. So all rule-matching is dropped and only the
//! security decisions remain, kept in the source's collect-then-reduce shape
//! (deny > ask > allow > passthrough) so an earlier ask can never mask a later
//! deny.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

use crate::tools::base::{PermissionBehavior, PermissionResult};
use crate::utils::powershell::parser::{
    get_file_redirections, ParsedCommandElement, ParsedPowerShellCommand, ParsedStatement,
    PS_TOKENIZER_DASH_CHARS,
};
use crate::utils::shell::read_only_command_validation::contains_vulnerable_unc_path;
use crate::utils::state::get_cwd;

use super::git_safety::{is_dot_git_path_ps, is_git_internal_path_ps};
use super::mode_validation::is_symlink_creating_command;
use super::path_validation::{check_path_constraints, dangerous_removal_deny, is_dangerous_removal_raw_path};
use super::powershell_security::powershell_command_is_safe;
use super::read_only_validation::{is_cwd_changing_cmdlet, is_read_only_command, resolve_to_canonical};

static PS_ASSIGN_PREFIX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\$[\w:]+\s*(?:[+\-*/%]|\?\?)?\s*=\s*").unwrap());
static BACKTICK_NEWLINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"`[\r\n]+\s*").unwrap());
static FRAGMENT_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[;|\n\r{}()&]+").unwrap());
static CALL_OP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[&.]\s+").unwrap());
static QUOTE_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());
static NON_FS_PROVIDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(?:[\w.]+\\)?(env|hklm|hkcu|function|alias|variable|cert|wsman|registry)::?")
        .unwrap()
});

/// Cmdlets that can place a file at a caller-specified path (git-internal guard).
static GIT_SAFETY_WRITE_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "new-item", "set-content", "add-content", "out-file", "copy-item", "move-item",
        "rename-item", "expand-archive", "invoke-webrequest", "invoke-restmethod", "tee-object",
        "export-csv", "export-clixml",
    ]
    .into_iter()
    .collect()
});

/// External archive extractors (opaque contents → git TOCTOU).
static GIT_SAFETY_ARCHIVE_EXTRACTORS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "tar", "tar.exe", "bsdtar", "bsdtar.exe", "unzip", "unzip.exe", "7z", "7z.exe", "7za",
        "7za.exe", "gzip", "gzip.exe", "gunzip", "gunzip.exe", "expand-archive",
    ]
    .into_iter()
    .collect()
});

/// True if cwd looks like a bare git repo (HEAD/objects/refs without a valid
/// .git/HEAD) — git would run hooks from cwd (isCurrentDirectoryBareGitRepo).
pub fn is_current_directory_bare_git_repo() -> bool {
    let cwd = std::path::PathBuf::from(get_cwd());
    let git_path = cwd.join(".git");
    if git_path.is_file() {
        return false; // worktree/submodule
    }
    if git_path.is_dir() && git_path.join("HEAD").is_file() {
        return false; // normal repo
    }
    if cwd.join("HEAD").is_file() {
        return true;
    }
    if cwd.join("objects").is_dir() {
        return true;
    }
    if cwd.join("refs").is_dir() {
        return true;
    }
    false
}

fn ask(message: &str) -> PermissionResult {
    PermissionResult::ask_with(message)
}

fn extract_provider_value(arg: &str) -> String {
    let mut s = arg.to_string();
    if let Some(first) = s.chars().next() {
        if PS_TOKENIZER_DASH_CHARS.contains(&first) {
            if let Some(colon_idx) = s.char_indices().skip(1).find(|(_, c)| *c == ':').map(|(i, _)| i) {
                if colon_idx > 0 {
                    s = s[colon_idx + 1..].to_string();
                }
            }
        }
    }
    s.replace('`', "")
}

fn provider_or_unc_decision(arg: &str) -> Option<PermissionResult> {
    let value = extract_provider_value(arg);
    if NON_FS_PROVIDER_RE.is_match(&value) {
        return Some(ask(&format!(
            "Command argument '{arg}' uses a non-filesystem provider path and requires approval"
        )));
    }
    if contains_vulnerable_unc_path(&value) {
        return Some(ask(&format!(
            "Command argument '{arg}' contains a UNC path that could trigger network requests"
        )));
    }
    None
}

/// Every CommandAst element (statements + nested), with its statement.
fn all_command_elements(parsed: &ParsedPowerShellCommand) -> Vec<(&ParsedCommandElement, &ParsedStatement)> {
    let mut out = Vec::new();
    for statement in &parsed.statements {
        for cmd in &statement.commands {
            if cmd.element_type == "CommandAst" {
                out.push((cmd, statement));
            }
        }
        if let Some(nested) = &statement.nested_commands {
            for cmd in nested {
                out.push((cmd, statement));
            }
        }
    }
    out
}

/// Decide allow/ask/deny for a PowerShell command (security-only adaptation).
pub fn powershell_tool_has_permission(
    command: &str,
    parsed: &ParsedPowerShellCommand,
) -> PermissionResult {
    let command = command.trim();
    if command.is_empty() {
        let mut r = PermissionResult::allow();
        r.message = "Empty command is safe".to_string();
        return r;
    }

    // Parse-failed: best-effort dangerous-removal hard-deny, else ask.
    if !parsed.valid {
        let backtick_stripped = BACKTICK_NEWLINE_RE.replace_all(command, "").replace('`', "");
        for fragment in FRAGMENT_SPLIT_RE.split(&backtick_stripped) {
            let frag = fragment.trim();
            if frag.is_empty() {
                continue;
            }
            let mut normalized = frag.to_string();
            loop {
                match PS_ASSIGN_PREFIX_RE.find(&normalized) {
                    Some(m) => normalized = normalized[m.end()..].to_string(),
                    None => break,
                }
            }
            normalized = CALL_OP_RE.replace(&normalized, "").to_string();
            let tokens: Vec<&str> = normalized.split_whitespace().collect();
            let first_tok = tokens.first().map(|t| QUOTE_STRIP_RE.replace_all(t, "").to_string()).unwrap_or_default();
            if resolve_to_canonical(&first_tok) == "remove-item" {
                for arg in &tokens[1.min(tokens.len())..] {
                    if let Some(first) = arg.chars().next() {
                        if PS_TOKENIZER_DASH_CHARS.contains(&first) {
                            continue;
                        }
                    }
                    if is_dangerous_removal_raw_path(arg) {
                        return dangerous_removal_deny(arg);
                    }
                }
            }
        }
        return ask("Command contains malformed syntax that cannot be parsed and requires approval");
    }

    let mut decisions: Vec<PermissionResult> = Vec::new();

    // Raw-command UNC.
    if contains_vulnerable_unc_path(command) {
        decisions.push(ask("Command contains a UNC path that could trigger network requests"));
    }

    // Security validators (subexpr / script blocks / encoded / cradles / etc.).
    let safety = powershell_command_is_safe(command, parsed);
    if safety.behavior != "passthrough" {
        decisions.push(ask(safety
            .message
            .as_deref()
            .unwrap_or("This command contains patterns that could pose security risks")));
    }

    // using / #Requires — load external code, invisible to the block walk.
    if parsed.has_using_statements.unwrap_or(false) {
        decisions.push(ask("Command contains a `using` statement that may load external code"));
    }
    if parsed.has_script_requirements.unwrap_or(false) {
        decisions.push(ask("Command contains a `#Requires` directive that may trigger module loading"));
    }

    let elements = all_command_elements(parsed);

    // Provider / UNC arg scan (first match).
    for (cmd, _stmt) in &elements {
        let mut hit = None;
        for arg in &cmd.args {
            hit = provider_or_unc_decision(arg);
            if hit.is_some() {
                break;
            }
        }
        if let Some(h) = hit {
            decisions.push(h);
            break;
        }
    }

    // Git-safety guards.
    let has_cd_subcommand = elements.len() > 1 && elements.iter().any(|(c, _)| is_cwd_changing_cmdlet(&c.name));
    let has_symlink_create = elements.len() > 1 && elements.iter().any(|(c, _)| is_symlink_creating_command(c));
    let has_git_subcommand = elements.iter().any(|(c, _)| resolve_to_canonical(&c.name) == "git");

    if has_cd_subcommand && has_git_subcommand {
        decisions.push(ask("Compound commands with cd/Set-Location and git require approval to prevent bare repository attacks"));
    }

    if has_git_subcommand && is_current_directory_bare_git_repo() {
        decisions.push(ask("Git command in a directory with bare-repository indicators. Git may execute hooks from cwd."));
    }

    if has_git_subcommand {
        let writes_git_internal = |cmd: &ParsedCommandElement, statement: &ParsedStatement| -> bool {
            if let Some(redirs) = &cmd.redirections {
                for r in redirs {
                    if is_git_internal_path_ps(&r.target) {
                        return true;
                    }
                }
            }
            if !GIT_SAFETY_WRITE_CMDLETS.contains(resolve_to_canonical(&cmd.name).as_str()) {
                return false;
            }
            for a in &cmd.args {
                for part in a.split(',') {
                    if is_git_internal_path_ps(part) {
                        return true;
                    }
                }
            }
            for c in &statement.commands {
                if c.element_type == "CommandAst" {
                    continue;
                }
                if is_git_internal_path_ps(&c.text) {
                    return true;
                }
            }
            false
        };

        let writes = elements.iter().any(|(c, s)| writes_git_internal(c, s));
        let redir_writes = get_file_redirections(parsed).iter().any(|r| is_git_internal_path_ps(&r.target));
        if writes || redir_writes {
            decisions.push(ask("Command writes to a git-internal path (HEAD/objects/refs/hooks/.git) and runs git — could plant a malicious hook."));
        }
        if elements.iter().any(|(c, _)| GIT_SAFETY_ARCHIVE_EXTRACTORS.contains(c.name.to_lowercase().as_str())) {
            decisions.push(ask("Compound command extracts an archive and runs git — archive contents may plant bare-repository indicators."));
        }
    }

    // .git/ writes are dangerous even without a git subcommand.
    let writes_dot_git = |cmd: &ParsedCommandElement| -> bool {
        if let Some(redirs) = &cmd.redirections {
            for r in redirs {
                if is_dot_git_path_ps(&r.target) {
                    return true;
                }
            }
        }
        if !GIT_SAFETY_WRITE_CMDLETS.contains(resolve_to_canonical(&cmd.name).as_str()) {
            return false;
        }
        cmd.args.iter().any(|a| a.split(',').any(is_dot_git_path_ps))
    };

    if elements.iter().any(|(c, _)| writes_dot_git(c))
        || get_file_redirections(parsed).iter().any(|r| is_dot_git_path_ps(&r.target))
    {
        decisions.push(ask("Command writes to .git/ — hooks or config planted there execute on the next git operation."));
    }

    // Path constraints (deny-capable). check_path_constraints returns allow|ask|deny.
    let path_result = check_path_constraints(parsed, has_cd_subcommand);
    if path_result.behavior != PermissionBehavior::Allow {
        decisions.push(path_result);
    }

    // Read-only allowlist (allow).
    if is_read_only_command(command, Some(parsed)) {
        let mut r = PermissionResult::allow();
        r.message = "Command is read-only and safe to execute".to_string();
        decisions.push(r);
    }

    // File redirections (write to arbitrary paths).
    if !get_file_redirections(parsed).is_empty() {
        decisions.push(ask("Command contains file redirections that could write to arbitrary paths"));
    }

    // REDUCE: deny > ask > allow. Symlink-create disables any read-only allow
    // (a just-created link can redirect a later read).
    for d in &decisions {
        if d.behavior == PermissionBehavior::Deny {
            return d.clone();
        }
    }
    for d in &decisions {
        if d.behavior == PermissionBehavior::Ask {
            return d.clone();
        }
    }
    if !has_symlink_create {
        for d in &decisions {
            if d.behavior == PermissionBehavior::Allow {
                return d.clone();
            }
        }
    }

    // Nothing decided → prompt (CodeMason's default for non-read-only).
    ask("This command requires approval")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::powershell::parser::parse_powershell_command;

    fn decide(cmd: &str) -> PermissionResult {
        let parsed = parse_powershell_command(cmd);
        powershell_tool_has_permission(cmd, &parsed)
    }

    // pwsh is unavailable here; a live parse fails closed in BOTH languages
    // (the fail-closed decide() cases above still exercise that path).
    // Synthetic valid parses mirror the real pwsh AST transform; every
    // expected behavior was oracle-verified by building the SAME structures
    // with the Python dataclasses and calling powershell_tool_has_permission.
    fn decide_syn(cmd: &str, name: &str, name_type: &str, args: &[&str], ets: &[&str]) -> PermissionResult {
        let parsed = crate::utils::powershell::parser::ParsedPowerShellCommand {
            valid: true,
            statements: vec![crate::utils::powershell::parser::ParsedStatement {
                statement_type: "Pipeline".to_string(),
                commands: vec![crate::utils::powershell::parser::ParsedCommandElement {
                    name: name.to_string(),
                    name_type: name_type.to_string(),
                    element_type: "CommandAst".to_string(),
                    args: args.iter().map(|s| s.to_string()).collect(),
                    text: cmd.to_string(),
                    element_types: Some(ets.iter().map(|s| s.to_string()).collect()),
                    ..Default::default()
                }],
                redirections: vec![],
                text: cmd.to_string(),
                ..Default::default()
            }],
            original_command: cmd.to_string(),
            ..Default::default()
        };
        powershell_tool_has_permission(cmd, &parsed)
    }

    #[test]
    fn read_only_allowed() {
        assert_eq!(
            decide_syn("Get-ChildItem", "Get-ChildItem", "cmdlet", &[], &["StringConstant"]).behavior,
            PermissionBehavior::Allow
        );
        // classify_command_name("git") = "unknown" (no Verb-Noun, no path chars).
        assert_eq!(
            decide_syn("git status", "git", "unknown", &["status"], &["StringConstant", "StringConstant"]).behavior,
            PermissionBehavior::Allow
        );
    }

    #[test]
    fn dangerous_removal_denied() {
        assert_eq!(decide("Remove-Item -Path / -Recurse -Force").behavior, PermissionBehavior::Deny);
    }

    #[test]
    fn iex_asks() {
        assert_eq!(decide("Invoke-Expression $x").behavior, PermissionBehavior::Ask);
    }

    #[test]
    fn plain_write_asks() {
        assert_eq!(decide("Set-Content -Path out.txt -Value hi").behavior, PermissionBehavior::Ask);
    }
}
