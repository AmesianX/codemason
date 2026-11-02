//! Dangerous shell-tool allow-rule prefixes (adapted from dangerousPatterns.ts).
//! Equivalent to codemason/utils/permissions/dangerous_patterns.py.
//!
//! An allow rule like `Bash(python:*)` / `PowerShell(node:*)` would let the
//! model run arbitrary code via that interpreter, bypassing an auto-mode
//! classifier.
//!
//! STATUS — NOT WIRED: in the upstream these lists feed the
//! isDangerous{Bash,PowerShell}Permission predicates that strip such rules at
//! auto-mode entry, but codemason has NOT ported the permission-rule engine or
//! auto/acceptEdits modes (see state/app_state.rs). There is therefore no
//! allow-rule engine here for them to protect, so this module is currently
//! reference data only — retained verbatim so that if the rule engine is ever
//! ported, the predicate can be wired to it. Do NOT read this as active
//! protection: until it's wired, adding such an allow-rule would not be
//! neutralized.

use once_cell::sync::Lazy;

/// Cross-platform code-execution entry points present on both Unix and Windows.
/// Shared so the two lists don't drift on interpreter additions.
pub const CROSS_PLATFORM_CODE_EXEC: &[&str] = &[
    // Interpreters
    "python", "python3", "python2", "node", "deno", "tsx", "ruby", "perl", "php", "lua",
    // Package runners
    "npx", "bunx", "npm run", "yarn run", "pnpm run", "bun run",
    // Shells reachable from both (Git Bash / WSL on Windows, native on Unix)
    "bash", "sh",
    // Remote arbitrary-command wrapper (native OpenSSH on Win10+)
    "ssh",
];

fn dangerous_bash_patterns() -> Vec<&'static str> {
    let mut base: Vec<&'static str> = CROSS_PLATFORM_CODE_EXEC.to_vec();
    base.extend(["zsh", "fish", "eval", "exec", "env", "xargs", "sudo"]);
    // Internal (USER_TYPE=ant) additions — empirical over-allowlist risk,
    // not a universal "unsafe" judgment. External users get the base list.
    if std::env::var("USER_TYPE").as_deref() == Ok("ant") {
        base.extend([
            "fa run", "coo", "gh", "gh api", "curl", "wget", "git", "kubectl", "aws",
            "gcloud", "gsutil",
        ]);
    }
    base
}

pub static DANGEROUS_BASH_PATTERNS: Lazy<Vec<&'static str>> = Lazy::new(dangerous_bash_patterns);
