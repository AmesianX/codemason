//! PowerShell path validation (adaptation of pathValidation.ts).
//!
//! Extracts filesystem paths from a parsed PowerShell command (via
//! CMDLET_PATH_CONFIG + the AST) and validates they stay within the session
//! working directory.
//!
//! ADAPTED, not 1:1: the source ties into Claude Code's full permission-rule /
//! filesystem system. CodeMason simplified that in its Bash tool to
//! `validate_path` (containment in the session dir) + `is_dangerous_removal_path`
//! + a 3-state PermissionResult. This module mirrors that: it ports the
//! PowerShell-specific path EXTRACTION faithfully, then routes the actual check
//! through CodeMason's validators.
//!
//! NOTE (cross-cluster): the Python imported bash_tool's `_validate_path`.
//! In the Rust tree bash_tool's `validate_path` is private (cluster TF), so an
//! identical faithful port is kept here as `validate_path` to avoid editing
//! another cluster's file. `is_dangerous_removal_path` IS public and shared.
//! If bash_tool later exports `validate_path`, dedupe to that single copy.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

use crate::tools::base::{PermissionBehavior, PermissionResult};
use crate::tools::bash_tool::is_dangerous_removal_path;
use crate::utils::powershell::parser::{
    is_powershell_parameter, ParsedCommandElement, ParsedPowerShellCommand, ParsedStatement,
};
use crate::utils::state::{get_cwd, get_original_cwd, is_in_directory};

use super::common_parameters::{COMMON_SWITCHES, COMMON_VALUE_PARAMS};
use super::read_only_validation::resolve_to_canonical;

/// Only these AST element types are statically verifiable as a path arg.
const SAFE_PATH_ELEMENT_TYPES: &[&str] = &["StringConstant", "Parameter"];

/// bash_tool `_GLOB_RE` (faithful copy): globs are * ? [ ] { }.
static GLOB_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[*?\[\]{}]").unwrap());
static QUOTE_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());

/// Per-cmdlet parameter configuration (CmdletPathConfig).
pub struct CmdletPathConfig {
    pub operation_type: &'static str, // 'read' | 'write' | 'create'
    pub path_params: &'static [&'static str],
    pub known_switches: &'static [&'static str],
    pub known_value_params: &'static [&'static str],
    pub leaf_only_path_params: Option<&'static [&'static str]>,
    pub positional_skip: Option<usize>,
    pub optional_write: bool,
    /// Do not treat an unrecognised `-flag` as making the whole command
    /// unvalidatable.
    ///
    /// Off by default, and it must stay off for cmdlets whose parameters are
    /// enumerated here: a flag nobody listed might be the one carrying a path,
    /// and asking is the safe answer. It exists for a command whose flags are
    /// already vetted somewhere else, where re-flagging them here would only
    /// produce prompts for ordinary use. The flag's VALUE is still examined —
    /// an unconsumed value falls through to the positional branch and is
    /// containment-checked there.
    pub tolerate_unknown_params: bool,
}

impl CmdletPathConfig {
    const fn new(
        operation_type: &'static str,
        path_params: &'static [&'static str],
        known_switches: &'static [&'static str],
        known_value_params: &'static [&'static str],
    ) -> Self {
        CmdletPathConfig {
            operation_type,
            path_params,
            known_switches,
            known_value_params,
            leaf_only_path_params: None,
            positional_skip: None,
            optional_write: false,
            tolerate_unknown_params: false,
        }
    }
}

pub static CMDLET_PATH_CONFIG: Lazy<HashMap<&'static str, CmdletPathConfig>> = Lazy::new(|| {
    let mut m: HashMap<&'static str, CmdletPathConfig> = HashMap::new();
    m.insert("set-content", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-passthru", "-force", "-whatif", "-confirm", "-usetransaction", "-nonewline", "-asbytestream"], &["-value", "-filter", "-include", "-exclude", "-credential", "-encoding", "-stream"]));
    m.insert("add-content", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-passthru", "-force", "-whatif", "-confirm", "-usetransaction", "-nonewline", "-asbytestream"], &["-value", "-filter", "-include", "-exclude", "-credential", "-encoding", "-stream"]));
    m.insert("remove-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-recurse", "-force", "-whatif", "-confirm", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential", "-stream"]));
    m.insert("clear-content", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-whatif", "-confirm", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential", "-stream"]));
    m.insert("out-file", CmdletPathConfig::new("write", &["-filepath", "-path", "-literalpath", "-pspath", "-lp"], &["-append", "-force", "-noclobber", "-nonewline", "-whatif", "-confirm"], &["-inputobject", "-encoding", "-width"]));
    m.insert("tee-object", CmdletPathConfig::new("write", &["-filepath", "-path", "-literalpath", "-pspath", "-lp"], &["-append"], &["-inputobject", "-variable", "-encoding"]));
    m.insert("export-csv", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-append", "-force", "-noclobber", "-notypeinformation", "-includetypeinformation", "-useculture", "-noheader", "-whatif", "-confirm"], &["-inputobject", "-delimiter", "-encoding", "-quotefields", "-usequotes"]));
    m.insert("export-clixml", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-noclobber", "-whatif", "-confirm"], &["-inputobject", "-depth", "-encoding"]));
    m.insert("new-item", {
        let mut c = CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-whatif", "-confirm", "-usetransaction"], &["-itemtype", "-value", "-credential", "-type"]);
        c.leaf_only_path_params = Some(&["-name"]);
        c
    });
    m.insert("copy-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp", "-destination"], &["-container", "-force", "-passthru", "-recurse", "-whatif", "-confirm", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential", "-fromsession", "-tosession"]));
    m.insert("move-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp", "-destination"], &["-force", "-passthru", "-whatif", "-confirm", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential"]));
    m.insert("rename-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-passthru", "-whatif", "-confirm", "-usetransaction"], &["-newname", "-credential", "-filter", "-include", "-exclude"]));
    m.insert("set-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-passthru", "-whatif", "-confirm", "-usetransaction"], &["-value", "-credential", "-filter", "-include", "-exclude"]));
    m.insert("get-content", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-usetransaction", "-wait", "-raw", "-asbytestream"], &["-readcount", "-totalcount", "-tail", "-first", "-head", "-last", "-filter", "-include", "-exclude", "-credential", "-delimiter", "-encoding", "-stream"]));
    m.insert("get-childitem", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-recurse", "-force", "-name", "-usetransaction", "-followsymlink", "-directory", "-file", "-hidden", "-readonly", "-system"], &["-filter", "-include", "-exclude", "-depth", "-attributes", "-credential"]));
    m.insert("get-item", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential", "-stream"]));
    m.insert("get-itemproperty", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-usetransaction"], &["-name", "-filter", "-include", "-exclude", "-credential"]));
    m.insert("get-itempropertyvalue", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-usetransaction"], &["-name", "-filter", "-include", "-exclude", "-credential"]));
    m.insert("get-filehash", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &[], &["-algorithm", "-inputstream"]));
    m.insert("get-acl", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-audit", "-allcentralaccesspolicies", "-usetransaction"], &["-inputobject", "-filter", "-include", "-exclude"]));
    m.insert("format-hex", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-raw"], &["-inputobject", "-encoding", "-count", "-offset"]));
    m.insert("test-path", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-isvalid", "-usetransaction"], &["-filter", "-include", "-exclude", "-pathtype", "-credential", "-olderthan", "-newerthan"]));
    m.insert("resolve-path", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-relative", "-usetransaction", "-force"], &["-credential", "-relativebasepath"]));
    m.insert("convert-path", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-usetransaction"], &[]));
    m.insert("select-string", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-simplematch", "-casesensitive", "-quiet", "-list", "-notmatch", "-allmatches", "-noemphasis", "-raw"], &["-inputobject", "-pattern", "-include", "-exclude", "-encoding", "-context", "-culture"]));
    m.insert("set-location", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-passthru", "-usetransaction"], &["-stackname"]));
    m.insert("push-location", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &["-passthru", "-usetransaction"], &["-stackname"]));
    m.insert("pop-location", CmdletPathConfig::new("read", &[], &["-passthru", "-usetransaction"], &["-stackname"]));
    m.insert("select-xml", CmdletPathConfig::new("read", &["-path", "-literalpath", "-pspath", "-lp"], &[], &["-xml", "-content", "-xpath", "-namespace"]));
    m.insert("get-winevent", CmdletPathConfig::new("read", &["-path"], &["-force", "-oldest"], &["-listlog", "-logname", "-listprovider", "-providername", "-maxevents", "-computername", "-credential", "-filterxpath", "-filterxml", "-filterhashtable"]));
    m.insert("invoke-webrequest", {
        let mut c = CmdletPathConfig::new("write", &["-outfile", "-infile"], &["-allowinsecureredirect", "-allowunencryptedauthentication", "-disablekeepalive", "-nobodyprogress", "-passthru", "-preservefileauthorizationmetadata", "-resume", "-skipcertificatecheck", "-skipheadervalidation", "-skiphttperrorcheck", "-usebasicparsing", "-usedefaultcredentials"], &["-uri", "-method", "-body", "-contenttype", "-headers", "-maximumredirection", "-maximumretrycount", "-proxy", "-proxycredential", "-retryintervalsec", "-sessionvariable", "-timeoutsec", "-token", "-transferencoding", "-useragent", "-websession", "-credential", "-authentication", "-certificate", "-certificatethumbprint", "-form", "-httpversion"]);
        c.positional_skip = Some(1);
        c.optional_write = true;
        c
    });
    m.insert("invoke-restmethod", {
        let mut c = CmdletPathConfig::new("write", &["-outfile", "-infile"], &["-allowinsecureredirect", "-allowunencryptedauthentication", "-disablekeepalive", "-followrellink", "-nobodyprogress", "-passthru", "-preservefileauthorizationmetadata", "-resume", "-skipcertificatecheck", "-skipheadervalidation", "-skiphttperrorcheck", "-usebasicparsing", "-usedefaultcredentials"], &["-uri", "-method", "-body", "-contenttype", "-headers", "-maximumfollowrellink", "-maximumredirection", "-maximumretrycount", "-proxy", "-proxycredential", "-responseheaderstvariable", "-retryintervalsec", "-sessionvariable", "-statuscodevariable", "-timeoutsec", "-token", "-transferencoding", "-useragent", "-websession", "-credential", "-authentication", "-certificate", "-certificatethumbprint", "-form", "-httpversion"]);
        c.positional_skip = Some(1);
        c.optional_write = true;
        c
    });
    m.insert("expand-archive", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp", "-destinationpath"], &["-force", "-passthru", "-whatif", "-confirm"], &[]));
    m.insert("compress-archive", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp", "-destinationpath"], &["-force", "-update", "-passthru", "-whatif", "-confirm"], &["-compressionlevel"]));
    m.insert("set-itemproperty", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-passthru", "-force", "-whatif", "-confirm", "-usetransaction"], &["-name", "-value", "-type", "-filter", "-include", "-exclude", "-credential", "-inputobject"]));
    m.insert("new-itemproperty", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-whatif", "-confirm", "-usetransaction"], &["-name", "-value", "-propertytype", "-type", "-filter", "-include", "-exclude", "-credential"]));
    m.insert("remove-itemproperty", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-whatif", "-confirm", "-usetransaction"], &["-name", "-filter", "-include", "-exclude", "-credential"]));
    m.insert("clear-item", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-force", "-whatif", "-confirm", "-usetransaction"], &["-filter", "-include", "-exclude", "-credential"]));
    m.insert("export-alias", CmdletPathConfig::new("write", &["-path", "-literalpath", "-pspath", "-lp"], &["-append", "-force", "-noclobber", "-passthru", "-whatif", "-confirm"], &["-name", "-description", "-scope", "-as"]));
    // Native read exes (findstr/tree/file) that take POSITIONAL path/pattern args.
    // git reads files through path OPERANDS, not named parameters, and its
    // flag vocabulary is vetted before this runs: is_read_only_command puts
    // every flag through validate_flags against GIT_READ_ONLY_COMMANDS, and a
    // flag outside that list yields no allow at all. Without an entry here,
    // extract_paths_from_command returned early with no paths, so
    // check_path_constraints answered Allow and the read-only allow stood —
    // `git diff --no-index <absolute path outside the session> NUL` printed a
    // file from anywhere on disk with no prompt (measured).
    //
    // positional_skip drops the subcommand (grep/diff/log), which is not a
    // path. tolerate_unknown_params keeps ordinary git quiet: without it every
    // flag lands in the unknown branch and `git status --short` becomes a
    // prompt. Flag VALUES are still checked — an unconsumed value falls
    // through to the positional branch and is containment-checked there.
    m.insert("git", {
        let mut c = CmdletPathConfig::new("read", &[], &[], &[]);
        c.positional_skip = Some(1);
        c.tolerate_unknown_params = true;
        c
    });
    m.insert("findstr", CmdletPathConfig::new("read", &[], &[], &[]));
    m.insert("tree", CmdletPathConfig::new("read", &[], &[], &[]));
    m.insert("file", CmdletPathConfig::new("read", &[], &[], &[]));
    m
});

fn home_dir_string() -> String {
    dirs::home_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

/// bash_tool `_expand_tilde` (faithful): only bare ~ and ~/ resolve to $HOME.
/// PS form also expands `~\`.
fn expand_tilde(file_path: &str) -> String {
    if file_path == "~" || file_path.starts_with("~/") || file_path.starts_with("~\\") {
        return format!("{}{}", home_dir_string(), &file_path[1..]);
    }
    file_path.to_string()
}

/// Faithful port of bash_tool `_validate_path`: reject tilde-variants /
/// shell-expansion / UNC, block globs in write ops, and require containment in
/// the allowed working directory. Returns a deny/ask reason, or None if allowed.
fn validate_path(path: &str, op_type: &str) -> Option<String> {
    let trimmed = path.trim();
    let unquoted = trimmed.trim_matches(|c| c == '\'' || c == '"');
    let mut clean = expand_tilde(unquoted);
    if clean.is_empty() {
        return None;
    }
    // UNC network paths.
    if clean.starts_with("\\\\")
        || (clean.starts_with("//") && clean.len() > 2 && clean.as_bytes()[2] != b'/')
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
                "Glob patterns are not allowed in write operations. Please specify an exact file path.".to_string(),
            );
        }
        // Read globs: validate the base directory they would expand in.
        let parent = std::path::Path::new(&clean)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| ".".to_string());
        clean = parent;
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

/// os.path.normpath (POSIX, lexical) for the containment check.
fn normpath(path: &str) -> String {
    let is_abs = path.starts_with('/');
    let mut comps: Vec<&str> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                if comps.last().map(|s| *s != "..").unwrap_or(false) {
                    comps.pop();
                } else if !is_abs {
                    comps.push("..");
                }
            }
            other => comps.push(other),
        }
    }
    let joined = comps.join("/");
    if is_abs {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// A colon-bound value that isn't a plain literal (hasComplexColonValue).
fn has_complex_colon_value(raw_value: &str) -> bool {
    raw_value.contains(',')
        || raw_value.starts_with('(')
        || raw_value.starts_with('[')
        || raw_value.contains('`')
        || raw_value.contains("@(")
        || raw_value.starts_with("@{")
        || raw_value.contains('$')
}

/// Dangerous-removal check on the RAW path (pre-realpath), backslash-normalised
/// (isDangerousRemovalRawPath).
pub fn is_dangerous_removal_raw_path(file_path: &str) -> bool {
    // \z (absolute end), not $ — matches JS `.replace(/^['"]|['"]$/g,...)`.
    let stripped = QUOTE_STRIP_RE.replace_all(file_path, "");
    let expanded = expand_tilde(&stripped).replace('\\', "/");
    is_dangerous_removal_path(&expanded)
}

/// Hard-deny a removal targeting a protected system path (dangerousRemovalDeny).
pub fn dangerous_removal_deny(path: &str) -> PermissionResult {
    PermissionResult::deny(format!(
        "Remove-Item on system path '{path}' is blocked. This path is protected from removal."
    ))
}

fn matches_param(param_lower: &str, param_list: &[&str]) -> bool {
    param_list.iter().any(|p| {
        *p == param_lower || (param_lower.chars().count() > 1 && p.starts_with(param_lower))
    })
}

struct ExtractResult {
    paths: Vec<String>,
    operation_type: &'static str,
    has_unvalidatable_path_arg: bool,
    optional_write: bool,
}

fn extract_paths_from_command(cmd: &ParsedCommandElement) -> ExtractResult {
    let canonical = resolve_to_canonical(&cmd.name);
    let Some(config) = CMDLET_PATH_CONFIG.get(canonical.as_str()) else {
        return ExtractResult { paths: vec![], operation_type: "read", has_unvalidatable_path_arg: false, optional_write: false };
    };

    let mut switch_params: Vec<&str> = config.known_switches.to_vec();
    switch_params.extend_from_slice(COMMON_SWITCHES);
    let mut value_params: Vec<&str> = config.known_value_params.to_vec();
    value_params.extend_from_slice(COMMON_VALUE_PARAMS);

    let mut paths: Vec<String> = Vec::new();
    let args = &cmd.args;
    let element_types = &cmd.element_types;
    let mut has_unvalidatable = false;
    let mut positionals_seen = 0usize;
    let positional_skip = config.positional_skip.unwrap_or(0);

    let check_arg_element_type = |arg_idx: usize| -> bool {
        let Some(ets) = element_types else { return false };
        let et = ets.get(arg_idx + 1).map(|s| s.as_str());
        matches!(et, Some(e) if !SAFE_PATH_ELEMENT_TYPES.contains(&e))
    };

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg.is_empty() {
            i += 1;
            continue;
        }

        let arg_element_type = element_types.as_ref().and_then(|t| t.get(i + 1)).map(|s| s.as_str());
        if is_powershell_parameter(arg, arg_element_type) {
            let normalized = crate::utils::powershell::parser::normalize_dash_param(arg);
            let colon_idx = normalized.char_indices().skip(1).find(|(_, c)| *c == ':').map(|(idx, _)| idx);
            let param_name = match colon_idx {
                Some(c) if c > 0 => &normalized[..c],
                _ => normalized.as_str(),
            };
            let param_lower = param_name.to_lowercase();

            if matches_param(&param_lower, config.path_params) {
                let mut value: Option<String> = None;
                match colon_idx {
                    Some(c) if c > 0 => {
                        let raw_value = &arg[c + 1..];
                        if has_complex_colon_value(raw_value) {
                            has_unvalidatable = true;
                        } else {
                            value = Some(raw_value.to_string());
                        }
                    }
                    _ => {
                        let next_val = args.get(i + 1);
                        let next_type = element_types.as_ref().and_then(|t| t.get(i + 2)).map(|s| s.as_str());
                        if let Some(nv) = next_val {
                            if !nv.is_empty() && !is_powershell_parameter(nv, next_type) {
                                value = Some(nv.clone());
                                if check_arg_element_type(i + 1) {
                                    has_unvalidatable = true;
                                }
                                i += 1;
                            }
                        }
                    }
                }
                if let Some(v) = value {
                    if !v.is_empty() {
                        paths.push(v);
                    }
                }
            } else if config.leaf_only_path_params.map(|l| matches_param(&param_lower, l)).unwrap_or(false) {
                let mut value: Option<String> = None;
                match colon_idx {
                    Some(c) if c > 0 => {
                        let raw_value = &arg[c + 1..];
                        if has_complex_colon_value(raw_value) {
                            has_unvalidatable = true;
                        } else {
                            value = Some(raw_value.to_string());
                        }
                    }
                    _ => {
                        let next_val = args.get(i + 1);
                        let next_type = element_types.as_ref().and_then(|t| t.get(i + 2)).map(|s| s.as_str());
                        if let Some(nv) = next_val {
                            if !nv.is_empty() && !is_powershell_parameter(nv, next_type) {
                                value = Some(nv.clone());
                                if check_arg_element_type(i + 1) {
                                    has_unvalidatable = true;
                                }
                                i += 1;
                            }
                        }
                    }
                }
                if let Some(v) = value {
                    if v.contains('/') || v.contains('\\') || v == "." || v == ".." {
                        has_unvalidatable = true; // non-leaf can't resolve correctly
                    } else {
                        paths.push(v);
                    }
                }
            } else if matches_param(&param_lower, &switch_params) {
                // switch — takes no value, don't consume next arg
            } else if matches_param(&param_lower, &value_params) {
                match colon_idx {
                    Some(c) if c > 0 => {
                        let raw_value = &arg[c + 1..];
                        if has_complex_colon_value(raw_value) {
                            has_unvalidatable = true;
                        }
                    }
                    _ => {
                        let next_arg = args.get(i + 1);
                        let next_arg_type = element_types.as_ref().and_then(|t| t.get(i + 2)).map(|s| s.as_str());
                        if let Some(na) = next_arg {
                            if !na.is_empty() && !is_powershell_parameter(na, next_arg_type) {
                                if check_arg_element_type(i + 1) {
                                    has_unvalidatable = true;
                                }
                                i += 1;
                            }
                        }
                    }
                }
            } else {
                // Unknown parameter — flag the whole command unvalidatable,
                // unless this command's flags are vetted elsewhere.
                if !config.tolerate_unknown_params {
                    has_unvalidatable = true;
                }
                if let Some(c) = colon_idx {
                    if c > 0 {
                        let raw_value = &arg[c + 1..];
                        if !has_complex_colon_value(raw_value) {
                            paths.push(raw_value.to_string());
                        }
                    }
                }
            }
            i += 1;
            continue;
        }

        // Positional argument.
        if positionals_seen < positional_skip {
            positionals_seen += 1;
            i += 1;
            continue;
        }
        positionals_seen += 1;
        if check_arg_element_type(i) {
            has_unvalidatable = true;
        }
        paths.push(arg.clone());
        i += 1;
    }

    ExtractResult {
        paths,
        operation_type: config.operation_type,
        has_unvalidatable_path_arg: has_unvalidatable,
        optional_write: config.optional_write,
    }
}

fn check_statement(statement: &ParsedStatement, compound_command_has_cd: bool) -> PermissionResult {
    let mut first_ask: Option<PermissionResult> = None;

    if compound_command_has_cd {
        first_ask = Some(PermissionResult::ask_with(
            "Compound command changes working directory (Set-Location/Push-Location/Pop-Location/New-PSDrive) — relative paths cannot be validated against the original cwd and require manual approval",
        ));
    }

    let mut has_expression_pipeline_source = false;

    for cmd in &statement.commands {
        if cmd.element_type != "CommandAst" {
            has_expression_pipeline_source = true;
            continue;
        }

        let res = extract_paths_from_command(cmd);
        let canonical = resolve_to_canonical(&cmd.name);

        if has_expression_pipeline_source && first_ask.is_none() {
            first_ask = Some(PermissionResult::ask_with(format!(
                "{canonical} receives its path from a pipeline expression source that cannot be statically validated and requires manual approval"
            )));
        }

        if res.has_unvalidatable_path_arg && first_ask.is_none() {
            first_ask = Some(PermissionResult::ask_with(format!(
                "{canonical} uses a parameter or complex path expression (array literal, subexpression, unknown parameter, etc.) that cannot be statically validated and requires manual approval"
            )));
        }

        if res.operation_type != "read"
            && !res.optional_write
            && res.paths.is_empty()
            && CMDLET_PATH_CONFIG.contains_key(canonical.as_str())
        {
            if first_ask.is_none() {
                first_ask = Some(PermissionResult::ask_with(format!(
                    "{canonical} is a write operation but no target path could be determined; requires manual approval"
                )));
            }
            continue;
        }

        let is_removal = canonical == "remove-item";

        for file_path in &res.paths {
            // Hard-deny removal of dangerous system paths (raw, then resolved).
            if is_removal && is_dangerous_removal_raw_path(file_path) {
                return dangerous_removal_deny(file_path);
            }
            if is_removal {
                let clean = expand_tilde(file_path.trim().trim_matches(|c| c == '\'' || c == '"'));
                let full = normpath(&crate::utils::state::resolve_for_containment(
                    &clean,
                    &get_cwd(),
                ));
                if is_dangerous_removal_path(&full) {
                    return dangerous_removal_deny(&full);
                }
            }

            // Containment check via CodeMason's simplified validator.
            if let Some(reason) = validate_path(file_path, res.operation_type) {
                if first_ask.is_none() {
                    first_ask = Some(PermissionResult::ask_with(format!("{canonical}: {reason}")));
                }
            }
        }
    }

    first_ask.unwrap_or_else(|| {
        let mut r = PermissionResult::allow();
        r.message = "All path constraints validated successfully".to_string();
        r
    })
}

/// Validate paths across all statements (checkPathConstraints).
///
/// Two-pass: a deny (dangerous removal) wins immediately; otherwise the first
/// ask is returned; else 'allow'. On a failed parse, 'allow' (passthrough).
pub fn check_path_constraints(
    parsed: &ParsedPowerShellCommand,
    compound_command_has_cd: bool,
) -> PermissionResult {
    if !parsed.valid {
        let mut r = PermissionResult::allow();
        r.message = "Cannot validate paths for unparsed command".to_string();
        return r;
    }

    let mut first_ask: Option<PermissionResult> = None;
    for statement in &parsed.statements {
        let result = check_statement(statement, compound_command_has_cd);
        if result.behavior == PermissionBehavior::Deny {
            return result;
        }
        if result.behavior == PermissionBehavior::Ask && first_ask.is_none() {
            first_ask = Some(result);
        }
    }

    first_ask.unwrap_or_else(|| {
        let mut r = PermissionResult::allow();
        r.message = "All path constraints validated successfully".to_string();
        r
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    // pwsh is unavailable on this (Linux) box; a live parse fails closed in
    // BOTH languages (oracle-verified). Synthetic parses mirror the real pwsh
    // AST transform (element_type="CommandAst", element_types per element);
    // each expected behavior below was oracle-verified by building the SAME
    // structures with the Python dataclasses and calling
    // check_path_constraints.
    fn syn(cmd: &str, name: &str, args: &[&str], ets: &[&str]) -> crate::utils::powershell::parser::ParsedPowerShellCommand {
        crate::utils::powershell::parser::ParsedPowerShellCommand {
            valid: true,
            statements: vec![crate::utils::powershell::parser::ParsedStatement {
                statement_type: "Pipeline".to_string(),
                commands: vec![crate::utils::powershell::parser::ParsedCommandElement {
                    name: name.to_string(),
                    name_type: "cmdlet".to_string(),
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
        }
    }

    fn check(cmd: &str) -> PermissionResult {
        let parsed = match cmd {
            "Get-Content foo.txt" => syn(cmd, "Get-Content", &["foo.txt"], &["StringConstant", "StringConstant"]),
            "Set-Content -Path /etc/hosts -Value x" => syn(cmd, "Set-Content", &["-Path", "/etc/hosts", "-Value", "x"], &["StringConstant", "Parameter", "StringConstant", "Parameter", "StringConstant"]),
            "Remove-Item -Path / -Recurse" => syn(cmd, "Remove-Item", &["-Path", "/", "-Recurse"], &["StringConstant", "Parameter", "StringConstant", "Parameter"]),
            "Set-Content -Path *.txt -Value x" => syn(cmd, "Set-Content", &["-Path", "*.txt", "-Value", "x"], &["StringConstant", "Parameter", "StringConstant", "Parameter", "StringConstant"]),
            other => syn(other, other.split_whitespace().next().unwrap_or(""), &[], &["StringConstant"]),
        };
        check_path_constraints(&parsed, false)
    }

    #[test]
    fn read_inside_cwd_allowed() {
        assert_eq!(check("Get-Content foo.txt").behavior, PermissionBehavior::Allow);
    }

    #[test]
    fn write_outside_cwd_asks() {
        // absolute path outside session dir
        let r = check("Set-Content -Path /etc/hosts -Value x");
        assert_eq!(r.behavior, PermissionBehavior::Ask);
    }

    #[test]
    fn removal_of_system_path_denied() {
        let r = check("Remove-Item -Path / -Recurse");
        assert_eq!(r.behavior, PermissionBehavior::Deny);
    }

    #[test]
    fn glob_write_asks() {
        let r = check("Set-Content -Path *.txt -Value x");
        assert_eq!(r.behavior, PermissionBehavior::Ask);
    }

    /// PowerShell accepts an en-dash as a parameter prefix, and the permission
    /// check used to slice it at byte index 1 — inside the character — so
    /// `Set-Content –Path /etc/hosts -Value x` aborted the process from inside
    /// path validation rather than returning a decision. The dash must be
    /// normalized and the path still judged outside the session directory.
    #[test]
    fn unicode_dash_parameter_does_not_panic() {
        for dash in ['-', '–', '—', '―'] {
            let cmd = format!("Set-Content {dash}Path /etc/hosts -Value x");
            let parsed = syn(
                &cmd,
                "Set-Content",
                &[&format!("{dash}Path"), "/etc/hosts", "-Value", "x"],
                &["StringConstant", "Parameter", "StringConstant", "Parameter", "StringConstant"],
            );
            let r = check_path_constraints(&parsed, false);
            assert_eq!(
                r.behavior,
                PermissionBehavior::Ask,
                "{dash:?} should still gate the write outside the session dir"
            );
        }
    }

    /// git reaches files through operands, and it is on the read-only
    /// allowlist, so without a path config `git diff --no-index <absolute path>
    /// NUL` printed any file on disk with no prompt.
    #[test]
    fn git_operand_outside_the_session_is_gated() {
        let outside = if cfg!(windows) { r"C:\Windows\System32\drivers\etc\hosts" } else { "/etc/hosts" };
        let parsed = syn(
            "git diff --no-index x NUL",
            "git",
            &["diff", "--no-index", outside, "NUL"],
            &["StringConstant", "StringConstant", "Parameter", "StringConstant", "StringConstant"],
        );
        assert_eq!(
            check_path_constraints(&parsed, false).behavior,
            PermissionBehavior::Ask,
            "an operand outside the session directory must be gated"
        );
    }

    /// The other half of the same change: gating the escape must not turn
    /// everyday git into a prompt. The subcommand is skipped as a positional,
    /// and an unrecognised flag does not by itself make the command
    /// unvalidatable — git's flags are vetted by the read-only allowlist.
    #[test]
    fn everyday_git_is_not_gated() {
        let cases: &[(&[&str], &[&str])] = &[
            (
                &["status", "--short"],
                &["StringConstant", "StringConstant", "Parameter"],
            ),
            (
                &["log", "--oneline", "-n", "5"],
                &["StringConstant", "StringConstant", "Parameter", "Parameter", "StringConstant"],
            ),
            (
                &["diff", "--stat"],
                &["StringConstant", "StringConstant", "Parameter"],
            ),
            // A path operand INSIDE the session directory stays allowed.
            (
                &["diff", "tracked.txt"],
                &["StringConstant", "StringConstant", "StringConstant"],
            ),
        ];
        for (args, ets) in cases {
            let cmd = format!("git {}", args.join(" "));
            let parsed = syn(&cmd, "git", args, ets);
            assert_eq!(
                check_path_constraints(&parsed, false).behavior,
                PermissionBehavior::Allow,
                "{cmd} should not require approval"
            );
        }
    }
}
