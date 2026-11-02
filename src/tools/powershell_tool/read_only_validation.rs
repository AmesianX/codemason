//! PowerShell read-only command validation (adapted from readOnlyValidation.ts).
//!
//! Decides whether a parsed PowerShell command is read-only (safe to
//! auto-allow), using a cmdlet allowlist plus the shared external-command
//! validation for git/gh/docker/dotnet. Cmdlets are matched case-insensitively
//! (lowercase keys).

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};

use crate::utils::platform::is_windows;
use crate::utils::powershell::parser::{
    derive_security_flags, get_pipeline_segments, is_null_redirection_target,
    is_powershell_parameter, ParsedCommandElement, ParsedPowerShellCommand, COMMON_ALIASES,
};
use crate::utils::shell::read_only_command_validation::{
    validate_flags, ExternalCommandConfig, DOCKER_READ_ONLY_COMMANDS, EXTERNAL_READONLY_COMMANDS,
    GH_READ_ONLY_COMMANDS, GIT_READ_ONLY_COMMANDS,
};

use super::common_parameters::COMMON_PARAMETERS;

/// dotnet read-only top-level flags (DOTNET_READ_ONLY_FLAGS).
static DOTNET_READ_ONLY_FLAGS: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ["--version", "--info", "--list-runtimes", "--list-sdks"].into_iter().collect());

/// Metachar class flagging a runtime-evaluated arg: $ ( @ { [ (matches
/// source /[$(@{[]/).
static METACHAR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[$(@{\[]").unwrap());

/// Callback: returns True if the command is dangerous (command, element).
type DangerCb = fn(&str, Option<&ParsedCommandElement>) -> bool;

/// Allowlist config for a read-only PowerShell cmdlet / native exe.
pub struct CommandConfig {
    pub safe_flags: Vec<&'static str>,
    /// When true, all flags are allowed (entire flag surface is read-only).
    pub allow_all_flags: bool,
    /// Regex constraint on the original command string.
    pub regex: Option<Regex>,
    /// Returns True if the command is dangerous.
    pub additional_command_is_dangerous_callback: Option<DangerCb>,
}

impl CommandConfig {
    fn flags(safe_flags: &[&'static str]) -> Self {
        CommandConfig {
            safe_flags: safe_flags.to_vec(),
            allow_all_flags: false,
            regex: None,
            additional_command_is_dangerous_callback: None,
        }
    }
    fn flags_cb(safe_flags: &[&'static str], cb: DangerCb) -> Self {
        CommandConfig {
            safe_flags: safe_flags.to_vec(),
            allow_all_flags: false,
            regex: None,
            additional_command_is_dangerous_callback: Some(cb),
        }
    }
    fn all_flags() -> Self {
        CommandConfig {
            safe_flags: vec![],
            allow_all_flags: true,
            regex: None,
            additional_command_is_dangerous_callback: None,
        }
    }
    fn all_flags_cb(cb: DangerCb) -> Self {
        CommandConfig {
            safe_flags: vec![],
            allow_all_flags: true,
            regex: None,
            additional_command_is_dangerous_callback: Some(cb),
        }
    }
}

/// True if a cmdlet arg leaks a value (prints/coerces a non-literal). Port of
/// argLeaksValue — shared by Write-Output/Host, Start-Sleep, and the
/// Format-*/Select-Object/etc. pipeline-tail cmdlets.
fn arg_leaks_value(_cmd: &str, element: Option<&ParsedCommandElement>) -> bool {
    let Some(element) = element else { return false };
    let arg_types: Vec<String> = match &element.element_types {
        Some(t) => t.iter().skip(1).cloned().collect(),
        None => vec![],
    };
    let args = &element.args;
    let children = &element.children;
    for i in 0..arg_types.len() {
        if arg_types[i] != "StringConstant" && arg_types[i] != "Parameter" {
            let arg = args.get(i).map(|s| s.as_str()).unwrap_or("");
            if !METACHAR_RE.is_match(arg) {
                continue;
            }
            return true;
        }
        if arg_types[i] == "Parameter" {
            let param_children = children.as_ref().and_then(|c| c.get(i)).and_then(|c| c.as_ref());
            match param_children {
                Some(pc) if !pc.is_empty() => {
                    if pc.iter().any(|c| c.element_type != "StringConstant") {
                        return true;
                    }
                }
                _ => {
                    let arg = args.get(i).map(|s| s.as_str()).unwrap_or("");
                    if let Some(colon_idx) = arg.find(':') {
                        if colon_idx > 0 && METACHAR_RE.is_match(&arg[colon_idx + 1..]) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// ipconfig: reject any positional (non-/ non-- arg) — `ipconfig set ...`
/// writes network config on macOS.
fn ipconfig_cb(_cmd: &str, element: Option<&ParsedCommandElement>) -> bool {
    element
        .map(|e| e.args.iter().any(|a| !a.starts_with('/') && !a.starts_with('-')))
        .unwrap_or(false)
}

/// hostname: reject any positional — `hostname NAME` sets the hostname.
fn hostname_cb(_cmd: &str, element: Option<&ParsedCommandElement>) -> bool {
    element.map(|e| e.args.iter().any(|a| !a.starts_with('-'))).unwrap_or(false)
}

/// route: the first non-flag positional is the verb; only `print` is read-only.
fn route_cb(_cmd: &str, element: Option<&ParsedCommandElement>) -> bool {
    let Some(element) = element else { return true };
    let verb = element.args.iter().find(|a| !a.starts_with('-'));
    verb.map(|v| v.to_lowercase()).as_deref() != Some("print")
}

pub static CMDLET_ALLOWLIST: Lazy<HashMap<&'static str, CommandConfig>> = Lazy::new(|| {
    let mut m: HashMap<&'static str, CommandConfig> = HashMap::new();
    m.insert("get-childitem", CommandConfig::flags(&["-Path", "-LiteralPath", "-Filter", "-Include", "-Exclude", "-Recurse", "-Depth", "-Name", "-Force", "-Attributes", "-Directory", "-File", "-Hidden", "-ReadOnly", "-System"]));
    m.insert("get-content", CommandConfig::flags(&["-Path", "-LiteralPath", "-TotalCount", "-Head", "-Tail", "-Raw", "-Encoding", "-Delimiter", "-ReadCount"]));
    m.insert("get-item", CommandConfig::flags(&["-Path", "-LiteralPath", "-Force", "-Stream"]));
    m.insert("get-itemproperty", CommandConfig::flags(&["-Path", "-LiteralPath", "-Name"]));
    m.insert("test-path", CommandConfig::flags(&["-Path", "-LiteralPath", "-PathType", "-Filter", "-Include", "-Exclude", "-IsValid", "-NewerThan", "-OlderThan"]));
    m.insert("resolve-path", CommandConfig::flags(&["-Path", "-LiteralPath", "-Relative"]));
    m.insert("get-filehash", CommandConfig::flags(&["-Path", "-LiteralPath", "-Algorithm", "-InputStream"]));
    m.insert("get-acl", CommandConfig::flags(&["-Path", "-LiteralPath", "-Audit", "-Filter", "-Include", "-Exclude"]));
    m.insert("set-location", CommandConfig::flags(&["-Path", "-LiteralPath", "-PassThru", "-StackName"]));
    m.insert("push-location", CommandConfig::flags(&["-Path", "-LiteralPath", "-PassThru", "-StackName"]));
    m.insert("pop-location", CommandConfig::flags(&["-PassThru", "-StackName"]));
    m.insert("select-string", CommandConfig::flags(&["-Path", "-LiteralPath", "-Pattern", "-InputObject", "-SimpleMatch", "-CaseSensitive", "-Quiet", "-List", "-NotMatch", "-AllMatches", "-Encoding", "-Context", "-Raw", "-NoEmphasis"]));
    m.insert("convertto-json", CommandConfig::flags(&["-InputObject", "-Depth", "-Compress", "-EnumsAsStrings", "-AsArray"]));
    m.insert("convertfrom-json", CommandConfig::flags(&["-InputObject", "-Depth", "-AsHashtable", "-NoEnumerate"]));
    m.insert("convertto-csv", CommandConfig::flags(&["-InputObject", "-Delimiter", "-NoTypeInformation", "-NoHeader", "-UseQuotes"]));
    m.insert("convertfrom-csv", CommandConfig::flags(&["-InputObject", "-Delimiter", "-Header", "-UseCulture"]));
    m.insert("convertto-xml", CommandConfig::flags(&["-InputObject", "-Depth", "-As", "-NoTypeInformation"]));
    m.insert("convertto-html", CommandConfig::flags(&["-InputObject", "-Property", "-Head", "-Title", "-Body", "-Pre", "-Post", "-As", "-Fragment"]));
    m.insert("format-hex", CommandConfig::flags(&["-Path", "-LiteralPath", "-InputObject", "-Encoding", "-Count", "-Offset"]));
    m.insert("get-member", CommandConfig::flags(&["-InputObject", "-MemberType", "-Name", "-Static", "-View", "-Force"]));
    m.insert("get-unique", CommandConfig::flags(&["-InputObject", "-AsString", "-CaseInsensitive", "-OnType"]));
    m.insert("compare-object", CommandConfig::flags(&["-ReferenceObject", "-DifferenceObject", "-Property", "-SyncWindow", "-CaseSensitive", "-Culture", "-ExcludeDifferent", "-IncludeEqual", "-PassThru"]));
    m.insert("join-string", CommandConfig::flags(&["-InputObject", "-Property", "-Separator", "-OutputPrefix", "-OutputSuffix", "-SingleQuote", "-DoubleQuote", "-FormatString"]));
    m.insert("get-random", CommandConfig::flags(&["-InputObject", "-Minimum", "-Maximum", "-Count", "-SetSeed", "-Shuffle"]));
    m.insert("convert-path", CommandConfig::flags(&["-Path", "-LiteralPath"]));
    m.insert("join-path", CommandConfig::flags(&["-Path", "-ChildPath", "-AdditionalChildPath"]));
    m.insert("split-path", CommandConfig::flags(&["-Path", "-LiteralPath", "-Qualifier", "-NoQualifier", "-Parent", "-Leaf", "-LeafBase", "-Extension", "-IsAbsolute"]));
    m.insert("get-hotfix", CommandConfig::flags(&["-Id", "-Description"]));
    m.insert("get-itempropertyvalue", CommandConfig::flags(&["-Path", "-LiteralPath", "-Name"]));
    m.insert("get-psprovider", CommandConfig::flags(&["-PSProvider"]));
    m.insert("get-process", CommandConfig::flags(&["-Name", "-Id", "-Module", "-FileVersionInfo", "-IncludeUserName"]));
    m.insert("get-service", CommandConfig::flags(&["-Name", "-DisplayName", "-DependentServices", "-RequiredServices", "-Include", "-Exclude"]));
    m.insert("get-computerinfo", CommandConfig::all_flags());
    m.insert("get-host", CommandConfig::all_flags());
    m.insert("get-date", CommandConfig::flags(&["-Date", "-Format", "-UFormat", "-DisplayHint", "-AsUTC"]));
    m.insert("get-location", CommandConfig::flags(&["-PSProvider", "-PSDrive", "-Stack", "-StackName"]));
    m.insert("get-psdrive", CommandConfig::flags(&["-Name", "-PSProvider", "-Scope"]));
    m.insert("get-module", CommandConfig::flags(&["-Name", "-ListAvailable", "-All", "-FullyQualifiedName", "-PSEdition"]));
    m.insert("get-alias", CommandConfig::flags(&["-Name", "-Definition", "-Scope", "-Exclude"]));
    m.insert("get-history", CommandConfig::flags(&["-Id", "-Count"]));
    m.insert("get-culture", CommandConfig::all_flags());
    m.insert("get-uiculture", CommandConfig::all_flags());
    m.insert("get-timezone", CommandConfig::flags(&["-Name", "-Id", "-ListAvailable"]));
    m.insert("get-uptime", CommandConfig::all_flags());
    m.insert("write-output", CommandConfig::flags_cb(&["-InputObject", "-NoEnumerate"], arg_leaks_value));
    m.insert("write-host", CommandConfig::flags_cb(&["-Object", "-NoNewline", "-Separator", "-ForegroundColor", "-BackgroundColor"], arg_leaks_value));
    m.insert("start-sleep", CommandConfig::flags_cb(&["-Seconds", "-Milliseconds", "-Duration"], arg_leaks_value));
    m.insert("format-table", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("format-list", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("format-wide", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("format-custom", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("measure-object", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("select-object", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("sort-object", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("group-object", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("where-object", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("out-string", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("out-host", CommandConfig::all_flags_cb(arg_leaks_value));
    m.insert("get-netadapter", CommandConfig::flags(&["-Name", "-InterfaceDescription", "-InterfaceIndex", "-Physical"]));
    m.insert("get-netipaddress", CommandConfig::flags(&["-InterfaceIndex", "-InterfaceAlias", "-AddressFamily", "-Type"]));
    m.insert("get-netipconfiguration", CommandConfig::flags(&["-InterfaceIndex", "-InterfaceAlias", "-Detailed", "-All"]));
    m.insert("get-netroute", CommandConfig::flags(&["-InterfaceIndex", "-InterfaceAlias", "-AddressFamily", "-DestinationPrefix"]));
    m.insert("get-dnsclientcache", CommandConfig::flags(&["-Entry", "-Name", "-Type", "-Status", "-Section", "-Data"]));
    m.insert("get-dnsclient", CommandConfig::flags(&["-InterfaceIndex", "-InterfaceAlias"]));
    m.insert("get-eventlog", CommandConfig::flags(&["-LogName", "-Newest", "-After", "-Before", "-EntryType", "-Index", "-InstanceId", "-Message", "-Source", "-UserName", "-AsBaseObject", "-List"]));
    m.insert("get-winevent", CommandConfig::flags(&["-LogName", "-ListLog", "-ListProvider", "-ProviderName", "-Path", "-MaxEvents", "-FilterXPath", "-Force", "-Oldest"]));
    m.insert("get-cimclass", CommandConfig::flags(&["-ClassName", "-Namespace", "-MethodName", "-PropertyName", "-QualifierName"]));
    m.insert("git", CommandConfig::flags(&[]));
    m.insert("gh", CommandConfig::flags(&[]));
    m.insert("docker", CommandConfig::flags(&[]));
    m.insert("ipconfig", CommandConfig::flags_cb(&["/all", "/displaydns", "/allcompartments"], ipconfig_cb));
    m.insert("netstat", CommandConfig::flags(&["-a", "-b", "-e", "-f", "-n", "-o", "-p", "-q", "-r", "-s", "-t", "-x", "-y"]));
    m.insert("systeminfo", CommandConfig::flags(&["/FO", "/NH"]));
    m.insert("tasklist", CommandConfig::flags(&["/M", "/SVC", "/V", "/FI", "/FO", "/NH"]));
    m.insert("where.exe", CommandConfig::all_flags());
    m.insert("hostname", CommandConfig::flags_cb(&["-a", "-d", "-f", "-i", "-I", "-s", "-y", "-A"], hostname_cb));
    m.insert("whoami", CommandConfig::flags(&["/user", "/groups", "/claims", "/priv", "/logonid", "/all", "/fo", "/nh"]));
    m.insert("ver", CommandConfig::all_flags());
    m.insert("arp", CommandConfig::flags(&["-a", "-g", "-v", "-N"]));
    m.insert("route", CommandConfig::flags_cb(&["print", "PRINT", "-4", "-6"], route_cb));
    m.insert("getmac", CommandConfig::flags(&["/FO", "/NH", "/V"]));
    m.insert("file", CommandConfig::flags(&["-b", "--brief", "-i", "--mime", "-L", "--dereference", "--mime-type", "--mime-encoding", "-z", "--uncompress", "-p", "--preserve-date", "-k", "--keep-going", "-r", "--raw", "-v", "--version", "-0", "--print0", "-s", "--special-files", "-l", "-F", "--separator", "-e", "-P", "-N", "--no-pad", "-E", "--extension"]));
    m.insert("tree", CommandConfig::flags(&["/F", "/A", "/Q", "/L"]));
    m.insert("findstr", CommandConfig::flags(&["/B", "/E", "/L", "/R", "/S", "/I", "/X", "/V", "/N", "/M", "/O", "/P", "/C", "/G", "/D", "/A"]));
    m.insert("dotnet", CommandConfig::flags(&[]));
    m
});

pub static SAFE_OUTPUT_CMDLETS: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ["out-null"].into_iter().collect());

/// External .exe names allowed past the nameType='application' gate. Each MUST
/// have a matching CMDLET_ALLOWLIST entry for flag validation.
static SAFE_EXTERNAL_EXES: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ["where.exe"].into_iter().collect());

/// Windows PATHEXT extensions PowerShell resolves via PATH (.ps1 excluded).
static WINDOWS_PATHEXT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.(exe|cmd|bat|com)$").unwrap());

/// Canonical cmdlet name via COMMON_ALIASES; strips .exe/.cmd/.bat/.com on bare
/// (path-free) names so git.exe → git. (resolveToCanonical)
pub fn resolve_to_canonical(name: &str) -> String {
    let mut lower = name.to_lowercase();
    if !lower.contains('\\') && !lower.contains('/') {
        lower = WINDOWS_PATHEXT.replace(&lower, "").to_string();
    }
    if let Some(alias) = COMMON_ALIASES.get(lower.as_str()) {
        return alias.to_lowercase();
    }
    lower
}

/// True if the cmdlet alters the path-resolution namespace for later statements
/// (Set/Push/Pop-Location, New-PSDrive; ndr/mount Windows-only).
pub fn is_cwd_changing_cmdlet(name: &str) -> bool {
    let canonical = resolve_to_canonical(name);
    canonical == "set-location"
        || canonical == "push-location"
        || canonical == "pop-location"
        || canonical == "new-psdrive"
        || (is_windows() && (canonical == "ndr" || canonical == "mount"))
}

pub fn is_safe_output_command(name: &str) -> bool {
    SAFE_OUTPUT_CMDLETS.contains(resolve_to_canonical(name).as_str())
}

fn lookup_allowlist(name: &str) -> Option<&'static CommandConfig> {
    let lower = name.to_lowercase();
    if let Some(direct) = CMDLET_ALLOWLIST.get(lower.as_str()) {
        return Some(direct);
    }
    let canonical = resolve_to_canonical(&lower);
    if canonical != lower {
        return CMDLET_ALLOWLIST.get(canonical.as_str());
    }
    None
}

/// True if the parsed PowerShell command is read-only (isReadOnlyCommand).
pub fn is_read_only_command(command: &str, parsed: Option<&ParsedPowerShellCommand>) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }
    let Some(parsed) = parsed else { return false };
    if !parsed.valid {
        return false;
    }

    let security = derive_security_flags(parsed);
    if security.has_script_blocks
        || security.has_sub_expressions
        || security.has_expandable_strings
        || security.has_splatting
        || security.has_member_invocations
        || security.has_assignments
        || security.has_stop_parsing
    {
        return false;
    }

    let segments = get_pipeline_segments(parsed);
    if segments.is_empty() {
        return false;
    }

    // Block compound commands containing a cwd-changing cmdlet (stale-cwd path
    // resolution — finding #27).
    let total_commands: usize = segments.iter().map(|s| s.commands.len()).sum();
    if total_commands > 1 {
        let has_cd = segments
            .iter()
            .flat_map(|s| s.commands.iter())
            .any(|cmd| is_cwd_changing_cmdlet(&cmd.name));
        if has_cd {
            return false;
        }
    }

    for pipeline in segments {
        if pipeline.commands.is_empty() {
            return false;
        }
        // Reject file redirections (not `> $null`).
        if !pipeline.redirections.is_empty()
            && pipeline
                .redirections
                .iter()
                .any(|r| !r.is_merging && !is_null_redirection_target(&r.target))
        {
            return false;
        }
        let first_cmd = &pipeline.commands[0];
        if !is_allowlisted_command(first_cmd, command) {
            return false;
        }
        for cmd in pipeline.commands.iter().skip(1) {
            if cmd.name_type == "application" {
                return false;
            }
            if is_safe_output_command(&cmd.name) && cmd.args.is_empty() {
                continue;
            }
            if !is_allowlisted_command(cmd, command) {
                return false;
            }
        }
        if let Some(nested) = &pipeline.nested_commands {
            if !nested.is_empty() {
                return false;
            }
        }
    }

    true
}

/// True if a single command element is allowlisted and passes flag/arg
/// validation (isAllowlistedCommand).
pub fn is_allowlisted_command(cmd: &ParsedCommandElement, original_command: &str) -> bool {
    // nameType='application' → raw name has path chars; never auto-allow except
    // explicit safe .exe matched on the raw first token of cmd.text.
    if cmd.name_type == "application" {
        let raw_first_token = cmd.text.split_whitespace().next().unwrap_or("").to_lowercase();
        if !SAFE_EXTERNAL_EXES.contains(raw_first_token.as_str()) {
            return false;
        }
    }

    let Some(config) = lookup_allowlist(&cmd.name) else { return false };

    if let Some(re) = &config.regex {
        if !re.is_match(original_command) {
            return false;
        }
    }

    if let Some(cb) = config.additional_command_is_dangerous_callback {
        if cb(original_command, Some(cmd)) {
            return false;
        }
    }

    // SECURITY: whitelist arg elementTypes — only StringConstant/Parameter are
    // statically verifiable; everything else evaluates at runtime (leak vectors).
    let Some(element_types) = &cmd.element_types else { return false };
    for i in 1..element_types.len() {
        let t = &element_types[i];
        if t != "StringConstant" && t != "Parameter" {
            let arg = cmd.args.get(i - 1).map(|s| s.as_str()).unwrap_or("");
            if !METACHAR_RE.is_match(arg) {
                continue;
            }
            return false;
        }
        if t == "Parameter" {
            let param_children =
                cmd.children.as_ref().and_then(|c| c.get(i - 1)).and_then(|c| c.as_ref());
            match param_children {
                Some(pc) if !pc.is_empty() => {
                    if pc.iter().any(|c| c.element_type != "StringConstant") {
                        return false;
                    }
                }
                _ => {
                    let arg = cmd.args.get(i - 1).map(|s| s.as_str()).unwrap_or("");
                    if let Some(colon_idx) = arg.find(':') {
                        if colon_idx > 0 && METACHAR_RE.is_match(&arg[colon_idx + 1..]) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    let canonical = resolve_to_canonical(&cmd.name);

    if matches!(canonical.as_str(), "git" | "gh" | "docker" | "dotnet") {
        return is_external_command_safe(&canonical, &cmd.args);
    }

    let is_cmdlet = canonical.contains('-');

    if config.allow_all_flags {
        return true;
    }
    if config.safe_flags.is_empty() {
        // No safeFlags + no allowAllFlags → reject any flags; positional args OK.
        let has_flags = cmd.args.iter().enumerate().any(|(i, arg)| {
            if is_cmdlet {
                let et = cmd
                    .element_types
                    .as_ref()
                    .and_then(|t| t.get(i + 1))
                    .map(|s| s.as_str());
                is_powershell_parameter(arg, et)
            } else {
                arg.starts_with('-') || (is_windows() && arg.starts_with('/'))
            }
        });
        return !has_flags;
    }

    for i in 0..cmd.args.len() {
        let arg = &cmd.args[i];
        let is_flag = if is_cmdlet {
            let et = cmd.element_types.as_ref().and_then(|t| t.get(i + 1)).map(|s| s.as_str());
            is_powershell_parameter(arg, et)
        } else {
            arg.starts_with('-') || (is_windows() && arg.starts_with('/'))
        };
        if is_flag {
            let mut param_name = if is_cmdlet {
                crate::utils::powershell::parser::normalize_dash_param(arg)
            } else {
                arg.clone()
            };
            if let Some(colon_index) = param_name.find(':') {
                if colon_index > 0 {
                    param_name = param_name[..colon_index].to_string();
                }
            }
            let param_lower = param_name.to_lowercase();
            if is_cmdlet && COMMON_PARAMETERS.contains(param_lower.as_str()) {
                continue;
            }
            if !config.safe_flags.iter().any(|flag| flag.to_lowercase() == param_lower) {
                return false;
            }
        }
    }

    true
}

// External command validation (git / gh / docker / dotnet).
fn is_external_command_safe(command: &str, args: &[String]) -> bool {
    match command {
        "git" => is_git_safe(args),
        "gh" => is_gh_safe(args),
        "docker" => is_docker_safe(args),
        "dotnet" => is_dotnet_safe(args),
        _ => false,
    }
}

static DANGEROUS_GIT_GLOBAL_FLAGS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["-c", "-C", "--exec-path", "--config-env", "--git-dir", "--work-tree", "--attr-source"]
        .into_iter()
        .collect()
});
static GIT_GLOBAL_FLAGS_WITH_VALUES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "-c", "-C", "--exec-path", "--config-env", "--git-dir", "--work-tree", "--namespace",
        "--super-prefix", "--shallow-file",
    ]
    .into_iter()
    .collect()
});
const DANGEROUS_GIT_SHORT_FLAGS_ATTACHED: &[&str] = &["-c", "-C"];

fn is_git_safe(args: &[String]) -> bool {
    if args.is_empty() {
        return true;
    }
    // Blanket $ rejection (variable-expansion parser differential).
    if args.iter().any(|arg| arg.contains('$')) {
        return false;
    }
    // Skip/reject global flags before the subcommand.
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if arg.is_empty() || !arg.starts_with('-') {
            break;
        }
        for short_flag in DANGEROUS_GIT_SHORT_FLAGS_ATTACHED {
            if arg.len() > short_flag.len()
                && arg.starts_with(short_flag)
                && (*short_flag == "-C" || arg.as_bytes()[short_flag.len()] != b'-')
            {
                return false;
            }
        }
        let has_inline_value = arg.contains('=');
        let flag_name = if has_inline_value { arg.split('=').next().unwrap() } else { arg.as_str() };
        if DANGEROUS_GIT_GLOBAL_FLAGS.contains(flag_name) {
            return false;
        }
        if !has_inline_value && GIT_GLOBAL_FLAGS_WITH_VALUES.contains(flag_name) {
            idx += 2;
        } else {
            idx += 1;
        }
    }

    if idx >= args.len() {
        return true;
    }

    let first = args[idx].to_lowercase();
    let second = args.get(idx + 1).map(|s| s.to_lowercase()).unwrap_or_default();
    let two_word_key = format!("git {first} {second}");
    let one_word_key = format!("git {first}");

    let mut subcommand_tokens = 2usize;
    let config: Option<&ExternalCommandConfig> = match GIT_READ_ONLY_COMMANDS.get(two_word_key.as_str()) {
        Some(c) => Some(c),
        None => {
            subcommand_tokens = 1;
            GIT_READ_ONLY_COMMANDS.get(one_word_key.as_str())
        }
    };
    let Some(config) = config else { return false };

    let flag_args = &args[idx + subcommand_tokens..];

    // git ls-remote URL/exfil rejection.
    if first == "ls-remote" {
        for arg in flag_args {
            if !arg.starts_with('-')
                && (arg.contains("://") || arg.contains('@') || arg.contains(':') || arg.contains('$'))
            {
                return false;
            }
        }
    }

    if let Some(cb) = config.additional_command_is_dangerous_callback {
        if cb("", flag_args) {
            return false;
        }
    }
    validate_flags(flag_args, 0, config, Some("git"), None, None)
}

fn is_gh_safe(args: &[String]) -> bool {
    // gh is network-dependent; ant-only.
    if std::env::var("USER_TYPE").ok().as_deref() != Some("ant") {
        return false;
    }
    if args.is_empty() {
        return true;
    }
    let mut config: Option<&ExternalCommandConfig> = None;
    let mut subcommand_tokens = 0usize;
    if args.len() >= 2 {
        let key = format!("gh {} {}", args[0].to_lowercase(), args[1].to_lowercase());
        config = GH_READ_ONLY_COMMANDS.get(key.as_str());
        subcommand_tokens = 2;
    }
    if config.is_none() && !args.is_empty() {
        let key = format!("gh {}", args[0].to_lowercase());
        config = GH_READ_ONLY_COMMANDS.get(key.as_str());
        subcommand_tokens = 1;
    }
    let Some(config) = config else { return false };
    let flag_args = &args[subcommand_tokens..];
    if flag_args.iter().any(|a| a.contains('$')) {
        return false;
    }
    if let Some(cb) = config.additional_command_is_dangerous_callback {
        if cb("", flag_args) {
            return false;
        }
    }
    validate_flags(flag_args, 0, config, None, None, None)
}

fn is_docker_safe(args: &[String]) -> bool {
    if args.is_empty() {
        return true;
    }
    if args.iter().any(|a| a.contains('$')) {
        return false;
    }
    let one_word_key = format!("docker {}", args[0].to_lowercase());
    if EXTERNAL_READONLY_COMMANDS.contains(&one_word_key.as_str()) {
        return true;
    }
    let Some(config) = DOCKER_READ_ONLY_COMMANDS.get(one_word_key.as_str()) else { return false };
    let flag_args = &args[1..];
    if let Some(cb) = config.additional_command_is_dangerous_callback {
        if cb("", flag_args) {
            return false;
        }
    }
    validate_flags(flag_args, 0, config, None, None, None)
}

fn is_dotnet_safe(args: &[String]) -> bool {
    if args.is_empty() {
        return false;
    }
    args.iter().all(|arg| DOTNET_READ_ONLY_FLAGS.contains(arg.to_lowercase().as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // pwsh is not installed on this (Linux) box, and BOTH implementations
    // fail closed identically without it (oracle-verified: Python's
    // parse_powershell_command returns valid=False/NoPowerShell too). These
    // fixtures are synthetic parses mirroring the real pwsh AST transform —
    // element_type="CommandAst", element_types "StringConstant" for the name
    // and bare words / "Parameter" for -Flags, and classify_command_name's
    // name_type ("cmdlet" for Verb-Noun, "unknown" for bare `git`). Each
    // expected value below was oracle-verified by building the SAME synthetic
    // structures with the Python dataclasses and calling the Python validator.
    fn syn_cmd(name: &str, name_type: &str, args: &[&str], ets: &[&str], text: &str) -> crate::utils::powershell::parser::ParsedCommandElement {
        crate::utils::powershell::parser::ParsedCommandElement {
            name: name.to_string(),
            name_type: name_type.to_string(),
            element_type: "CommandAst".to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            text: text.to_string(),
            element_types: Some(ets.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        }
    }

    fn syn_parsed(command: &str, name: &str, name_type: &str, args: &[&str], ets: &[&str]) -> crate::utils::powershell::parser::ParsedPowerShellCommand {
        crate::utils::powershell::parser::ParsedPowerShellCommand {
            valid: true,
            statements: vec![crate::utils::powershell::parser::ParsedStatement {
                statement_type: "Pipeline".to_string(),
                commands: vec![syn_cmd(name, name_type, args, ets, command)],
                redirections: vec![],
                text: command.to_string(),
                ..Default::default()
            }],
            original_command: command.to_string(),
            ..Default::default()
        }
    }

    fn syn(cmd: &str) -> crate::utils::powershell::parser::ParsedPowerShellCommand {
        match cmd {
            "Get-ChildItem" => syn_parsed(cmd, "Get-ChildItem", "cmdlet", &[], &["StringConstant"]),
            "Get-Content -Path foo.txt" => syn_parsed(cmd, "Get-Content", "cmdlet", &["-Path", "foo.txt"], &["StringConstant", "Parameter", "StringConstant"]),
            "git status" => syn_parsed(cmd, "git", "unknown", &["status"], &["StringConstant", "StringConstant"]),
            "git log --oneline" => syn_parsed(cmd, "git", "unknown", &["log", "--oneline"], &["StringConstant", "StringConstant", "StringConstant"]),
            "Remove-Item foo" => syn_parsed(cmd, "Remove-Item", "cmdlet", &["foo"], &["StringConstant", "StringConstant"]),
            "Set-Content -Path a -Value b" => syn_parsed(cmd, "Set-Content", "cmdlet", &["-Path", "a", "-Value", "b"], &["StringConstant", "Parameter", "StringConstant", "Parameter", "StringConstant"]),
            "git -C /tmp status" => syn_parsed(cmd, "git", "unknown", &["-C", "/tmp", "status"], &["StringConstant", "Parameter", "StringConstant", "StringConstant"]),
            "git -c core.pager=x log" => syn_parsed(cmd, "git", "unknown", &["-c", "core.pager=x", "log"], &["StringConstant", "Parameter", "StringConstant", "StringConstant"]),
            other => syn_parsed(other, other.split_whitespace().next().unwrap_or(""), "unknown", &[], &["StringConstant"]),
        }
    }

    fn ro(cmd: &str) -> bool {
        let parsed = syn(cmd);
        is_read_only_command(cmd, Some(&parsed))
    }

    #[test]
    fn simple_reads_allowed() {
        assert!(ro("Get-ChildItem"));
        assert!(ro("Get-Content -Path foo.txt"));
        assert!(ro("git status"));
        assert!(ro("git log --oneline"));
    }

    #[test]
    fn writes_and_unknown_rejected() {
        assert!(!ro("Remove-Item foo"));
        assert!(!ro("Set-Content -Path a -Value b"));
        assert!(!ro("Some-UnknownCmdlet"));
    }

    #[test]
    fn dangerous_git_global_flag_rejected() {
        assert!(!ro("git -C /tmp status"));
        assert!(!ro("git -c core.pager=x log"));
    }

    #[test]
    fn resolve_canonical_strips_exe_and_alias() {
        assert_eq!(resolve_to_canonical("git.exe"), "git");
        assert_eq!(resolve_to_canonical("gci"), "get-childitem");
    }

    #[test]
    fn dotnet_only_readonly_flags() {
        assert!(is_dotnet_safe(&["--version".to_string()]));
        assert!(!is_dotnet_safe(&["build".to_string()]));
        assert!(!is_dotnet_safe(&[]));
    }
}
