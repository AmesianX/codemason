//! PowerShell-specific security analysis (adapted from powershellSecurity.ts).
//!
//! Detects dangerous patterns — code injection, download cradles, privilege
//! escalation, dynamic command names, COM objects, etc. All checks are
//! AST-based: if parsing failed (valid=False), no individual check matches and
//! powershell_command_is_safe returns 'ask'.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

use crate::utils::powershell::dangerous_cmdlets::{
    DANGEROUS_SCRIPT_BLOCK_CMDLETS, FILEPATH_EXECUTION_CMDLETS, MODULE_LOADING_CMDLETS,
    WMI_CIM_CMDLETS,
};
use crate::utils::powershell::parser::{
    command_has_arg_abbreviation, derive_security_flags, get_all_commands, get_variables_by_scope,
    has_command_named, ParsedCommandElement, ParsedPowerShellCommand, COMMON_ALIASES,
};

use super::clm_types::is_clm_allowed_type;

#[derive(Debug, Clone, PartialEq)]
pub struct PowerShellSecurityResult {
    pub behavior: String, // 'passthrough' | 'ask' | 'allow'
    pub message: Option<String>,
}

fn passthrough() -> PowerShellSecurityResult {
    PowerShellSecurityResult { behavior: "passthrough".into(), message: None }
}

fn ask(message: &str) -> PowerShellSecurityResult {
    PowerShellSecurityResult { behavior: "ask".into(), message: Some(message.to_string()) }
}

static POWERSHELL_EXECUTABLES: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ["pwsh", "pwsh.exe", "powershell", "powershell.exe"].into_iter().collect());

/// True if `name` (bare or a full path) resolves to a PowerShell executable.
pub fn is_powershell_executable(name: &str) -> bool {
    let lower = name.to_lowercase();
    if POWERSHELL_EXECUTABLES.contains(lower.as_str()) {
        return true;
    }
    let last_sep = lower.rfind('/').max(lower.rfind('\\'));
    if let Some(idx) = last_sep {
        return POWERSHELL_EXECUTABLES.contains(&lower[idx + 1..]);
    }
    false
}

/// Alt parameter-prefix chars PowerShell treats as `-` (plus 5.1's `/`).
const PS_ALT_PARAM_PREFIXES: &[char] = &['/', '–', '—', '―'];

/// command_has_arg_abbreviation that also normalises `/`/en-dash/em-dash/bar → `-`.
fn ps_exe_has_param_abbreviation(cmd: &ParsedCommandElement, full_param: &str, min_prefix: &str) -> bool {
    if command_has_arg_abbreviation(cmd, full_param, min_prefix) {
        return true;
    }
    let normalized_args: Vec<String> = cmd
        .args
        .iter()
        .map(|a| {
            let mut chars = a.chars();
            match chars.next() {
                Some(first) if PS_ALT_PARAM_PREFIXES.contains(&first) => {
                    format!("-{}", chars.as_str())
                }
                _ => a.clone(),
            }
        })
        .collect();
    let mut synthetic = cmd.clone();
    synthetic.args = normalized_args;
    command_has_arg_abbreviation(&synthetic, full_param, min_prefix)
}

fn check_invoke_expression(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if has_command_named(parsed, "Invoke-Expression") {
        return ask("Command uses Invoke-Expression which can execute arbitrary code");
    }
    passthrough()
}

fn check_dynamic_command_name(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if cmd.element_type != "CommandAst" {
            continue;
        }
        let name_et = cmd.element_types.as_ref().and_then(|t| t.first());
        if let Some(t) = name_et {
            if t != "StringConstant" {
                return ask("Command name is a dynamic expression which cannot be statically validated");
            }
        }
    }
    passthrough()
}

fn check_encoded_command(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if is_powershell_executable(&cmd.name)
            && ps_exe_has_param_abbreviation(cmd, "-encodedcommand", "-e")
        {
            return ask("Command uses encoded parameters which obscure intent");
        }
    }
    passthrough()
}

fn check_pwsh_command_or_file(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if is_powershell_executable(&cmd.name) {
            return ask("Command spawns a nested PowerShell process which cannot be validated");
        }
    }
    passthrough()
}

static DOWNLOADER_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["invoke-webrequest", "iwr", "invoke-restmethod", "irm", "new-object", "start-bitstransfer"]
        .into_iter()
        .collect()
});

fn is_downloader(name: &str) -> bool {
    DOWNLOADER_NAMES.contains(name.to_lowercase().as_str())
}

fn is_iex(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == "invoke-expression" || lower == "iex"
}

fn check_download_cradles(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    // Per-statement piped cradle (IWR ... | IEX)
    for statement in &parsed.statements {
        let cmds = &statement.commands;
        if cmds.len() < 2 {
            continue;
        }
        if cmds.iter().any(|c| is_downloader(&c.name)) && cmds.iter().any(|c| is_iex(&c.name)) {
            return ask("Command downloads and executes remote code");
        }
    }
    // Cross-statement split cradle
    let all_cmds = get_all_commands(parsed);
    if all_cmds.iter().any(|c| is_downloader(&c.name)) && all_cmds.iter().any(|c| is_iex(&c.name)) {
        return ask("Command downloads and executes remote code");
    }
    passthrough()
}

fn check_download_utilities(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        if lower == "start-bitstransfer" {
            return ask("Command downloads files via BITS transfer");
        }
        if lower == "certutil" || lower == "certutil.exe" {
            if cmd.args.iter().any(|a| matches!(a.to_lowercase().as_str(), "-urlcache" | "/urlcache")) {
                return ask("Command uses certutil to download from a URL");
            }
        }
        if lower == "bitsadmin" || lower == "bitsadmin.exe" {
            if cmd.args.iter().any(|a| a.to_lowercase() == "/transfer") {
                return ask("Command downloads files via BITS transfer");
            }
        }
    }
    passthrough()
}

fn check_add_type(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if has_command_named(parsed, "Add-Type") {
        return ask("Command compiles and loads .NET code");
    }
    passthrough()
}

fn check_com_object(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if cmd.name.to_lowercase() != "new-object" {
            continue;
        }
        if ps_exe_has_param_abbreviation(cmd, "-comobject", "-com") {
            return ask("Command instantiates a COM object which may have execution capabilities");
        }
        // Extract -TypeName (named / colon-bound / positional-0) → CLM check.
        let mut type_name: Option<String> = None;
        for (i, a) in cmd.args.iter().enumerate() {
            let lower = a.to_lowercase();
            if lower.starts_with("-t") && a.contains(':') {
                let colon_idx = a.find(':').unwrap();
                let param_part = &lower[..colon_idx];
                if "-typename".starts_with(param_part) {
                    type_name = Some(a[colon_idx + 1..].to_string());
                    break;
                }
            }
            if lower.starts_with("-t") && "-typename".starts_with(lower.as_str()) && i + 1 < cmd.args.len() {
                type_name = Some(cmd.args[i + 1].clone());
                break;
            }
        }
        if type_name.is_none() {
            let value_params: HashSet<&str> = ["-argumentlist", "-comobject", "-property"].into_iter().collect();
            let switch_params: HashSet<&str> = ["-strict"].into_iter().collect();
            let mut i = 0;
            while i < cmd.args.len() {
                let a = &cmd.args[i];
                if a.starts_with('-') {
                    let lower = a.to_lowercase();
                    if lower.starts_with("-t") && "-typename".starts_with(lower.as_str()) {
                        i += 2;
                        continue;
                    }
                    if lower.contains(':') {
                        i += 1;
                        continue;
                    }
                    if switch_params.contains(lower.as_str()) {
                        i += 1;
                        continue;
                    }
                    if value_params.contains(lower.as_str()) {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    continue;
                }
                type_name = Some(a.clone());
                break;
            }
        }
        if let Some(tn) = &type_name {
            if !is_clm_allowed_type(tn) {
                return ask(&format!(
                    "New-Object instantiates .NET type '{tn}' outside the ConstrainedLanguage allowlist"
                ));
            }
        }
    }
    passthrough()
}

fn check_dangerous_file_path_execution(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        let resolved = COMMON_ALIASES.get(lower.as_str()).map(|s| s.to_lowercase()).unwrap_or(lower);
        if !FILEPATH_EXECUTION_CMDLETS.contains(resolved.as_str()) {
            continue;
        }
        if ps_exe_has_param_abbreviation(cmd, "-filepath", "-f")
            || ps_exe_has_param_abbreviation(cmd, "-literalpath", "-l")
        {
            return ask(&format!("{} -FilePath executes an arbitrary script file", cmd.name));
        }
        for (i, arg) in cmd.args.iter().enumerate() {
            let arg_type = cmd.element_types.as_ref().and_then(|t| t.get(i + 1)).map(|s| s.as_str());
            if arg_type == Some("StringConstant") && !arg.is_empty() && !arg.starts_with('-') {
                return ask(&format!(
                    "{} with positional string argument binds to -FilePath and executes a script file",
                    cmd.name
                ));
            }
        }
    }
    passthrough()
}

fn check_for_each_member_name(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        let resolved = COMMON_ALIASES.get(lower.as_str()).map(|s| s.to_lowercase()).unwrap_or(lower);
        if resolved != "foreach-object" {
            continue;
        }
        if ps_exe_has_param_abbreviation(cmd, "-membername", "-m") {
            return ask("ForEach-Object -MemberName invokes methods by string name which cannot be validated");
        }
        for (i, arg) in cmd.args.iter().enumerate() {
            let arg_type = cmd.element_types.as_ref().and_then(|t| t.get(i + 1)).map(|s| s.as_str());
            if arg_type == Some("StringConstant") && !arg.is_empty() && !arg.starts_with('-') {
                return ask("ForEach-Object with positional string argument binds to -MemberName and invokes methods by name");
            }
        }
    }
    passthrough()
}

static VERB_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)^[-–—―/]v[a-z]*:").unwrap());
static VERB_RUNAS_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)^[-–—―/]v[a-z]*:['"` ]*runas['"` ]*$"#).unwrap());
static VERB_CLEAN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"['"`\s]"#).unwrap());
static QUOTE_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());

fn check_start_process(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        if !matches!(lower.as_str(), "start-process" | "saps" | "start") {
            continue;
        }
        // Vector 1: -Verb RunAs (space syntax)
        if ps_exe_has_param_abbreviation(cmd, "-Verb", "-v")
            && cmd.args.iter().any(|a| a.to_lowercase() == "runas")
        {
            return ask("Command requests elevated privileges");
        }
        // Colon syntax (a) structural via children
        if let Some(children) = &cmd.children {
            for (i, raw_arg) in cmd.args.iter().enumerate() {
                let arg_clean = raw_arg.replace('`', "");
                if !VERB_COLON_RE.is_match(&arg_clean) {
                    continue;
                }
                let kids = children.get(i).and_then(|c| c.as_ref());
                let Some(kids) = kids else { continue };
                for child in kids {
                    if VERB_CLEAN_RE.replace_all(&child.text, "").to_lowercase() == "runas" {
                        return ask("Command requests elevated privileges");
                    }
                }
            }
        }
        // Colon syntax (b) regex fallback
        for a in &cmd.args {
            let clean = a.replace('`', "");
            if VERB_RUNAS_RE.is_match(&clean) {
                return ask("Command requests elevated privileges");
            }
        }
        // Vector 2: target is a PowerShell executable
        for arg in &cmd.args {
            let stripped = QUOTE_STRIP_RE.replace_all(arg, "");
            if is_powershell_executable(&stripped) {
                return ask("Start-Process launches a nested PowerShell process which cannot be validated");
            }
        }
    }
    passthrough()
}

static SAFE_SCRIPT_BLOCK_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "where-object", "sort-object", "select-object", "group-object", "format-table",
        "format-list", "format-wide", "format-custom",
    ]
    .into_iter()
    .collect()
});

fn check_script_block_injection(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if !derive_security_flags(parsed).has_script_blocks {
        return passthrough();
    }
    for cmd in get_all_commands(parsed) {
        if DANGEROUS_SCRIPT_BLOCK_CMDLETS.contains(cmd.name.to_lowercase().as_str()) {
            return ask("Command contains script block with dangerous cmdlet that may execute arbitrary code");
        }
    }
    let safe = |cmd: &ParsedCommandElement| -> bool {
        let lower = cmd.name.to_lowercase();
        if SAFE_SCRIPT_BLOCK_CMDLETS.contains(lower.as_str()) {
            return true;
        }
        if let Some(alias) = COMMON_ALIASES.get(lower.as_str()) {
            if SAFE_SCRIPT_BLOCK_CMDLETS.contains(alias.to_lowercase().as_str()) {
                return true;
            }
        }
        false
    };
    if get_all_commands(parsed).iter().all(|c| safe(c)) {
        return passthrough();
    }
    ask("Command contains script block that may execute arbitrary code")
}

fn check_sub_expressions(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if derive_security_flags(parsed).has_sub_expressions {
        return ask("Command contains subexpressions $()");
    }
    passthrough()
}

fn check_expandable_strings(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if derive_security_flags(parsed).has_expandable_strings {
        return ask("Command contains expandable strings with embedded expressions");
    }
    passthrough()
}

fn check_splatting(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if derive_security_flags(parsed).has_splatting {
        return ask("Command uses splatting (@variable)");
    }
    passthrough()
}

fn check_stop_parsing(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if derive_security_flags(parsed).has_stop_parsing {
        return ask("Command uses stop-parsing token (--%)");
    }
    passthrough()
}

fn check_member_invocations(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if derive_security_flags(parsed).has_member_invocations {
        return ask("Command invokes .NET methods");
    }
    passthrough()
}

fn check_type_literals(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    if let Some(literals) = &parsed.type_literals {
        for t in literals {
            if !is_clm_allowed_type(t) {
                return ask(&format!("Command uses .NET type [{t}] outside the ConstrainedLanguage allowlist"));
            }
        }
    }
    passthrough()
}

fn check_invoke_item(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        if lower == "invoke-item" || lower == "ii" {
            return ask(
                "Invoke-Item opens files with the default handler (ShellExecute). On executable files this runs arbitrary code."
            );
        }
    }
    passthrough()
}

static SCHEDULED_TASK_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["register-scheduledtask", "new-scheduledtask", "new-scheduledtaskaction", "set-scheduledtask"]
        .into_iter()
        .collect()
});

fn check_scheduled_task(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let lower = cmd.name.to_lowercase();
        if SCHEDULED_TASK_CMDLETS.contains(lower.as_str()) {
            return ask(&format!("{} creates or modifies a scheduled task (persistence primitive)", cmd.name));
        }
        if lower == "schtasks" || lower == "schtasks.exe" {
            if cmd.args.iter().any(|a| matches!(a.to_lowercase().as_str(), "/create" | "/change" | "-create" | "-change")) {
                return ask("schtasks with create/change modifies scheduled tasks (persistence primitive)");
            }
        }
    }
    passthrough()
}

static ENV_WRITE_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "set-item", "si", "new-item", "ni", "remove-item", "ri", "del", "rm", "rd", "rmdir",
        "erase", "clear-item", "cli", "set-content", "add-content", "ac",
    ]
    .into_iter()
    .collect()
});

fn check_env_var_manipulation(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    let env_vars = get_variables_by_scope(parsed, "env");
    if env_vars.is_empty() {
        return passthrough();
    }
    for cmd in get_all_commands(parsed) {
        if ENV_WRITE_CMDLETS.contains(cmd.name.to_lowercase().as_str()) {
            return ask("Command modifies environment variables");
        }
    }
    if derive_security_flags(parsed).has_assignments && !env_vars.is_empty() {
        return ask("Command modifies environment variables");
    }
    passthrough()
}

fn check_module_loading(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if MODULE_LOADING_CMDLETS.contains(cmd.name.to_lowercase().as_str()) {
            return ask(
                "Command loads, installs, or downloads a PowerShell module or script, which can execute arbitrary code"
            );
        }
    }
    passthrough()
}

static RUNTIME_STATE_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["set-alias", "sal", "new-alias", "nal", "set-variable", "sv", "new-variable", "nv"]
        .into_iter()
        .collect()
});

fn check_runtime_state_manipulation(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        let raw = cmd.name.to_lowercase();
        let lower = match raw.rfind('\\') {
            Some(i) => &raw[i + 1..],
            None => raw.as_str(),
        };
        if RUNTIME_STATE_CMDLETS.contains(lower) {
            return ask(
                "Command creates or modifies an alias or variable that can affect future command resolution"
            );
        }
    }
    passthrough()
}

fn check_wmi_process_spawn(parsed: &ParsedPowerShellCommand) -> PowerShellSecurityResult {
    for cmd in get_all_commands(parsed) {
        if WMI_CIM_CMDLETS.contains(cmd.name.to_lowercase().as_str()) {
            return ask(&format!("{} can spawn arbitrary processes via WMI/CIM (Win32_Process Create)", cmd.name));
        }
    }
    passthrough()
}

/// Order-sensitive: mirrors the source's validators array exactly.
type Validator = fn(&ParsedPowerShellCommand) -> PowerShellSecurityResult;
static VALIDATORS: &[Validator] = &[
    check_invoke_expression,
    check_dynamic_command_name,
    check_encoded_command,
    check_pwsh_command_or_file,
    check_download_cradles,
    check_download_utilities,
    check_add_type,
    check_com_object,
    check_dangerous_file_path_execution,
    check_invoke_item,
    check_scheduled_task,
    check_for_each_member_name,
    check_start_process,
    check_script_block_injection,
    check_sub_expressions,
    check_expandable_strings,
    check_splatting,
    check_stop_parsing,
    check_member_invocations,
    check_type_literals,
    check_env_var_manipulation,
    check_module_loading,
    check_runtime_state_manipulation,
    check_wmi_process_spawn,
];

/// Main entry: validate a PowerShell command against dangerous patterns.
///
/// AST-based; a failed parse (valid=False) returns 'ask' as a safe default.
/// Returns the first validator that says 'ask', else 'passthrough'.
pub fn powershell_command_is_safe(
    _command: &str,
    parsed: &ParsedPowerShellCommand,
) -> PowerShellSecurityResult {
    if !parsed.valid {
        return ask("Could not parse command for security analysis");
    }
    for validator in VALIDATORS {
        let result = validator(parsed);
        if result.behavior == "ask" {
            return result;
        }
    }
    passthrough()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::powershell::parser::parse_powershell_command;

    fn safe(cmd: &str) -> PowerShellSecurityResult {
        let parsed = parse_powershell_command(cmd);
        powershell_command_is_safe(cmd, &parsed)
    }

    // pwsh is unavailable here; a live parse fails closed in BOTH languages.
    // Synthetic valid parse for the one test that needs one (plain_read_passes)
    // — oracle-verified against the Python validator on the same structure.
    fn safe_syn(cmd: &str, name: &str, ets: &[&str]) -> PowerShellSecurityResult {
        let parsed = crate::utils::powershell::parser::ParsedPowerShellCommand {
            valid: true,
            statements: vec![crate::utils::powershell::parser::ParsedStatement {
                statement_type: "Pipeline".to_string(),
                commands: vec![crate::utils::powershell::parser::ParsedCommandElement {
                    name: name.to_string(),
                    name_type: "cmdlet".to_string(),
                    element_type: "CommandAst".to_string(),
                    args: vec![],
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
        powershell_command_is_safe(cmd, &parsed)
    }

    #[test]
    fn invoke_expression_asks() {
        assert_eq!(safe("Invoke-Expression $x").behavior, "ask");
        assert_eq!(safe("iex (Get-Content x)").behavior, "ask");
    }

    #[test]
    fn nested_powershell_asks() {
        assert_eq!(safe("powershell -Command foo").behavior, "ask");
    }

    #[test]
    fn plain_read_passes() {
        assert_eq!(safe_syn("Get-ChildItem", "Get-ChildItem", &["StringConstant"]).behavior, "passthrough");
    }

    #[test]
    fn ps_executable_detection() {
        assert!(is_powershell_executable("pwsh"));
        assert!(is_powershell_executable("C:\\Windows\\System32\\powershell.exe"));
        assert!(!is_powershell_executable("bash"));
    }
}
