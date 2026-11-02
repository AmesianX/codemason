//! Exit-code semantics for external executables in PowerShell (commandSemantics.ts).
//!
//! PowerShell-native cmdlets signal failure via terminating errors ($?), not
//! exit codes, so they need no special handling. But EXTERNAL exes set
//! $LASTEXITCODE and many use non-zero to convey info, not failure (grep
//! 1=no-match, robocopy 0-7=ok). Without this, PowerShell tool would report
//! those as errors.

use once_cell::sync::Lazy;
use regex::Regex;

/// (is_error, message|None)
pub type SemanticResult = (bool, Option<String>);

fn default_semantic(exit_code: i32, _stdout: &str, _stderr: &str) -> SemanticResult {
    (
        exit_code != 0,
        if exit_code != 0 {
            Some(format!("Command failed with exit code {exit_code}"))
        } else {
            None
        },
    )
}

fn grep_semantic(exit_code: i32, _stdout: &str, _stderr: &str) -> SemanticResult {
    // 0 = matches, 1 = no matches, 2+ = error
    (
        exit_code >= 2,
        if exit_code == 1 { Some("No matches found".to_string()) } else { None },
    )
}

fn robocopy_semantic(exit_code: i32, _stdout: &str, _stderr: &str) -> SemanticResult {
    // Bitfield: 0-7 success, 8+ failure. bit0 = files copied.
    let msg = if exit_code == 0 {
        Some("No files copied (already in sync)".to_string())
    } else if (1..8).contains(&exit_code) {
        Some(
            if exit_code & 1 != 0 {
                "Files copied successfully"
            } else {
                "Robocopy completed (no errors)"
            }
            .to_string(),
        )
    } else {
        None
    };
    (exit_code >= 8, msg)
}

/// Keys: lowercase command names WITHOUT .exe. diff/fc/find/test omitted
/// (ambiguous native-vs-cmdlet aliasing); native cmdlets (select-string etc.)
/// exit 0.
fn command_semantic(base: &str) -> fn(i32, &str, &str) -> SemanticResult {
    match base {
        "grep" | "rg" | "findstr" => grep_semantic,
        "robocopy" => robocopy_semantic,
        _ => default_semantic,
    }
}

static CALL_OP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[&.]\s+").unwrap());
static QUOTE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^["']|["']$"#).unwrap());
static EXE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.exe$").unwrap());
static SEG_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[;|]").unwrap());
static PATH_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\\/]").unwrap());

/// Command name from a pipeline segment: strip call op / quotes / path / .exe.
fn extract_base_command(segment: &str) -> String {
    let stripped = CALL_OP_RE.replace(segment.trim(), "");
    let first_token = stripped.split_whitespace().next().unwrap_or("");
    let unquoted = QUOTE_RE.replace_all(first_token, "");
    let basename = PATH_SPLIT_RE
        .split(&unquoted)
        .last()
        .filter(|s| !s.is_empty())
        .unwrap_or(&unquoted)
        .to_string();
    EXE_RE.replace(&basename.to_lowercase(), "").to_string()
}

/// Base command of the LAST pipeline segment (it sets the exit code). Heuristic
/// split on ; and | — do NOT use for security.
fn heuristically_extract_base_command(command: &str) -> String {
    let segments: Vec<&str> = SEG_SPLIT_RE.split(command).filter(|s| !s.trim().is_empty()).collect();
    let last = segments.last().copied().unwrap_or(command);
    extract_base_command(last)
}

/// Interpret a command result → (is_error, message|None) per semantic rules.
pub fn interpret_command_result(
    command: &str,
    exit_code: i32,
    stdout: &str,
    stderr: &str,
) -> SemanticResult {
    let base = heuristically_extract_base_command(command);
    command_semantic(&base)(exit_code, stdout, stderr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grep_no_match_not_error() {
        assert_eq!(interpret_command_result("grep foo bar", 1, "", ""), (false, Some("No matches found".to_string())));
        assert_eq!(interpret_command_result("grep foo bar", 2, "", "").0, true);
    }

    #[test]
    fn robocopy_bitfield() {
        assert_eq!(interpret_command_result("robocopy a b", 1, "", "").0, false);
        assert_eq!(interpret_command_result("robocopy a b", 8, "", "").0, true);
    }

    #[test]
    fn last_segment_decides() {
        // pipeline: last segment is `grep` → no-match tolerant
        assert_eq!(interpret_command_result("cat x | grep y", 1, "", "").0, false);
    }

    #[test]
    fn default_nonzero_is_error() {
        assert_eq!(interpret_command_result("git status", 1, "", "").0, true);
    }
}
