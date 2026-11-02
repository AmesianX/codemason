//! PowerShell cmdlets that execute arbitrary code (adapted from dangerousCmdlets.ts).
//! Equivalent to codemason/utils/powershell/dangerous_cmdlets.py.
//!
//! Consumed by the permission-engine validators (powershell_security) and the
//! UI suggestion gate (static_prefix). Single source so the lists don't drift.

use once_cell::sync::Lazy;
use std::collections::HashSet;

use crate::utils::permissions::dangerous_patterns::CROSS_PLATFORM_CODE_EXEC;
use crate::utils::powershell::parser::COMMON_ALIASES;

/// Cmdlets that accept a -FilePath (or positional path) and execute the file.
pub static FILEPATH_EXECUTION_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from(["invoke-command", "start-job", "start-threadjob", "register-scheduledjob"])
});

/// Cmdlets where a scriptblock argument executes arbitrary code (not just
/// filtering).
pub static DANGEROUS_SCRIPT_BLOCK_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "invoke-command", "invoke-expression", "start-job", "start-threadjob",
        "register-scheduledjob", "register-engineevent", "register-objectevent",
        "register-wmievent", "new-pssession", "enter-pssession",
    ])
});

/// Cmdlets that load and execute module/script code (.psm1 runs on import).
pub static MODULE_LOADING_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "import-module", "ipmo", "install-module", "save-module", "update-module",
        "install-script", "save-script",
    ])
});

/// Shells and process spawners.
const SHELLS_AND_SPAWNERS: &[&str] = &[
    "pwsh", "powershell", "cmd", "bash", "wsl", "sh",
    "start-process", "start", "add-type", "new-object",
];

/// Network cmdlets — wildcard rules enable exfil/download without prompt.
pub static NETWORK_CMDLETS: Lazy<HashSet<&'static str>> =
    Lazy::new(|| HashSet::from(["invoke-webrequest", "invoke-restmethod"]));

/// Alias/variable mutation cmdlets (rebind command resolution / poison defaults).
pub static ALIAS_HIJACK_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from(["set-alias", "sal", "new-alias", "nal", "set-variable", "sv", "new-variable", "nv"])
});

/// WMI/CIM process spawn — Start-Process equivalent that bypasses
/// checkStartProcess (Invoke-WmiMethod -Class Win32_Process -Name Create).
/// Source finding #34.
pub static WMI_CIM_CMDLETS: Lazy<HashSet<&'static str>> =
    Lazy::new(|| HashSet::from(["invoke-wmimethod", "iwmi", "invoke-cimmethod"]));

/// Cmdlets in CMDLET_ALLOWLIST with additionalCommandIsDangerousCallback — a
/// `Cmdlet:*` wildcard would bypass the callback forever, so never suggest it.
pub static ARG_GATED_CMDLETS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "select-object", "sort-object", "group-object", "where-object", "measure-object",
        "write-output", "write-host", "start-sleep", "format-table", "format-list",
        "format-wide", "format-custom", "out-string", "out-host",
        // Native executables with callback-gated args (ipconfig /flushdns
        // rejected, /all allowed).
        "ipconfig", "hostname", "route",
    ])
});

/// COMMON_ALIASES keys whose canonical target is in `targets` (lowercased).
fn aliases_of(targets: &HashSet<String>) -> Vec<&'static str> {
    COMMON_ALIASES
        .iter()
        .filter(|(_, target)| targets.contains(&target.to_lowercase()))
        .map(|(alias, _)| *alias)
        .collect()
}

fn build_never_suggest() -> HashSet<String> {
    let mut core: HashSet<String> = HashSet::new();
    core.extend(SHELLS_AND_SPAWNERS.iter().map(|s| s.to_string()));
    core.extend(FILEPATH_EXECUTION_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(DANGEROUS_SCRIPT_BLOCK_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(MODULE_LOADING_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(NETWORK_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(ALIAS_HIJACK_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(WMI_CIM_CMDLETS.iter().map(|s| s.to_string()));
    core.extend(ARG_GATED_CMDLETS.iter().map(|s| s.to_string()));
    core.insert("foreach-object".to_string());
    // Interpreters/runners — bare `node:*` would auto-allow -e/-p code.
    // Multi-word entries ('npm run') are filtered: NEVER_SUGGEST is a
    // single-name lookup.
    core.extend(
        CROSS_PLATFORM_CODE_EXEC
            .iter()
            .filter(|p| !p.contains(' '))
            .map(|p| p.to_string()),
    );
    let mut result = core.clone();
    result.extend(aliases_of(&core).into_iter().map(|s| s.to_string()));
    result
}

/// Commands to never suggest as a wildcard prefix in the permission dialog.
/// NOT WIRED: codemason has not ported the permission-rule/wildcard-suggestion
/// UI, so NEVER_SUGGEST (and the *_CMDLETS sets that only feed it —
/// NETWORK_CMDLETS, ALIAS_HIJACK_CMDLETS, ARG_GATED_CMDLETS,
/// SHELLS_AND_SPAWNERS) has no live consumer today. Retained as reference for
/// when that UI is ported. The sets that ARE live (FILEPATH_EXECUTION /
/// MODULE_LOADING / DANGEROUS_SCRIPT_BLOCK / WMI_CIM) are imported directly by
/// powershell_security.
pub static NEVER_SUGGEST: Lazy<HashSet<String>> = Lazy::new(build_never_suggest);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_suggest_includes_aliases_and_filters_multiword() {
        // 'start' is an alias of Start-Process AND listed directly.
        assert!(NEVER_SUGGEST.contains("start"));
        // 'saps' is an alias of Start-Process (in core) → pulled in via aliases.
        assert!(NEVER_SUGGEST.contains("saps"));
        // 'iex' aliases Invoke-Expression (dangerous scriptblock cmdlet).
        assert!(NEVER_SUGGEST.contains("iex"));
        // Multi-word interpreter entries are filtered out.
        assert!(!NEVER_SUGGEST.contains("npm run"));
        assert!(NEVER_SUGGEST.contains("python"));
        assert!(NEVER_SUGGEST.contains("foreach-object"));
        // '%' aliases ForEach-Object → included via aliases_of.
        assert!(NEVER_SUGGEST.contains("%"));
    }
}
