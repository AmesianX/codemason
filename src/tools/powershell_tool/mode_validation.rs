//! PowerShell mode validation (modeValidation.ts) — adapted.
//!
//! The source's checkPermissionMode auto-allows filesystem-modifying cmdlets in
//! acceptEdits mode. CodeMason has NO acceptEdits mode (its permission model is
//! simplified), so checkPermissionMode would always passthrough and is
//! intentionally omitted. is_symlink_creating_command is a standalone security
//! helper worth keeping — links redirect path resolution at runtime, a TOCTOU
//! vector.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

use crate::utils::powershell::parser::{ParsedCommandElement, PS_TOKENIZER_DASH_CHARS};

use super::read_only_validation::resolve_to_canonical;

/// New-Item -ItemType values that create filesystem links (redirect path
/// resolution).
static LINK_ITEM_TYPES: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ["symboliclink", "junction", "hardlink"].into_iter().collect());
static QUOTE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());

/// Unambiguous abbrev of New-Item -ItemType / -Type (min `-it` / `-ty`).
fn is_item_type_param_abbrev(p: &str) -> bool {
    let n = p.chars().count();
    (n >= 3 && "-itemtype".starts_with(p)) || (n >= 3 && "-type".starts_with(p))
}

/// Return the first char and the rest of a string.
fn split_first_char(s: &str) -> Option<(char, &str)> {
    let mut it = s.char_indices();
    let (_, c) = it.next()?;
    let rest_start = it.next().map(|(i, _)| i).unwrap_or(s.len());
    Some((c, &s[rest_start..]))
}

/// True if `cmd` is New-Item creating a SymbolicLink/Junction/HardLink.
///
/// Handles PS abbreviation (-it.. / -ty..), unicode-dash / `/` prefixes,
/// colon-bound values, backtick escapes, quotes.
pub fn is_symlink_creating_command(cmd: &ParsedCommandElement) -> bool {
    let canonical = resolve_to_canonical(&cmd.name);
    if canonical != "new-item" {
        return false;
    }
    let args = &cmd.args;
    for i in 0..args.len() {
        let raw = &args[i];
        if raw.is_empty() {
            continue;
        }
        let (first, rest) = match split_first_char(raw) {
            Some(v) => v,
            None => continue,
        };
        let normalized = if PS_TOKENIZER_DASH_CHARS.contains(&first) || first == '/' {
            format!("-{rest}")
        } else {
            raw.clone()
        };
        let lower = normalized.to_lowercase();
        // colon_idx = lower.find(":", 1): search from char index 1.
        let colon_idx = lower
            .char_indices()
            .skip(1)
            .find(|(_, c)| *c == ':')
            .map(|(i, _)| i);
        let param_raw = match colon_idx {
            Some(i) if i > 0 => &lower[..i],
            _ => lower.as_str(),
        };
        let param = param_raw.replace('`', "");
        if !is_item_type_param_abbrev(&param) {
            continue;
        }
        let raw_val: String = match colon_idx {
            Some(i) if i > 0 => lower[i + 1..].to_string(),
            _ => args
                .get(i + 1)
                .map(|s| s.to_lowercase())
                .unwrap_or_default(),
        };
        let val = QUOTE_RE.replace(&raw_val.replace('`', ""), "").to_string();
        if LINK_ITEM_TYPES.contains(val.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    // pwsh is not installed on this (Linux) box, and BOTH implementations
    // fail closed identically without it (Python's parse_powershell_command
    // returns valid=False/NoPowerShell too). Build the command element the
    // real pwsh AST transform would produce; each expected value was
    // oracle-verified by building the SAME structure with the Python
    // dataclasses and calling is_symlink_creating_command.
    fn first_cmd(cmd: &str) -> ParsedCommandElement {
        let mut it = cmd.split_whitespace();
        let name = it.next().unwrap_or("").to_string();
        ParsedCommandElement {
            name: name.clone(),
            name_type: "cmdlet".to_string(),
            element_type: "CommandAst".to_string(),
            args: it.map(|s| s.to_string()).collect(),
            text: cmd.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn detects_symlink() {
        assert!(is_symlink_creating_command(&first_cmd("New-Item -ItemType SymbolicLink -Path l -Target t")));
        assert!(is_symlink_creating_command(&first_cmd("New-Item -Type Junction -Path l -Target t")));
        assert!(is_symlink_creating_command(&first_cmd("New-Item -it:HardLink -Path l -Target t")));
    }

    #[test]
    fn non_link_not_detected() {
        assert!(!is_symlink_creating_command(&first_cmd("New-Item -ItemType File -Path f")));
        assert!(!is_symlink_creating_command(&first_cmd("Get-ChildItem")));
    }
}
