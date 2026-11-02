//! PowerShell tool prompt (adapted from PowerShellTool/prompt.ts).
//!
//! The tool description handed to the model. Critically edition-aware:
//! PowerShell 7 (pwsh) supports `&&`/`||`/`?:`/`??`; Windows PowerShell 5.1
//! (powershell.exe) does not (using them is a parse error).
//!
//! NOTE (cross-cluster): the Python imported the sibling tools' name constants
//! (GLOB/GREP/READ/EDIT/WRITE) and bash_tool.get_max_output_length to stay in
//! sync. Those live in clusters TF/TM; to avoid cross-cluster stub coupling the
//! stable tool-identifier literals and a faithful copy of get_max_output_length
//! are inlined here. If those modules export the symbols, switch to them.

use crate::utils::powershell_detect::get_powershell_edition;

use super::tool_name::POWERSHELL_TOOL_NAME;

// Sibling tool names (stable identifiers).
const GLOB_TOOL_NAME: &str = "Glob";
const GREP_TOOL_NAME: &str = "Grep";
const FILE_READ_TOOL_NAME: &str = "Read";
const FILE_EDIT_TOOL_NAME: &str = "Edit";
const FILE_WRITE_TOOL_NAME: &str = "Write";

// Timeouts mirror the Bash tool (getDefault/getMaxBashTimeoutMs).
const DEFAULT_TIMEOUT_MS: i64 = 120_000;
const MAX_TIMEOUT_MS: i64 = 600_000;

/// bash_tool output cap (faithful copy of get_max_output_length).
const BASH_MAX_OUTPUT_UPPER_LIMIT: i64 = 150_000;
const BASH_MAX_OUTPUT_DEFAULT: i64 = 30_000;

pub fn get_default_timeout_ms() -> i64 {
    DEFAULT_TIMEOUT_MS
}

pub fn get_max_timeout_ms() -> i64 {
    MAX_TIMEOUT_MS
}

fn get_max_output_length() -> i64 {
    let value = match std::env::var("BASH_MAX_OUTPUT_LENGTH") {
        Ok(raw) => match raw.trim().parse::<i64>() {
            Ok(v) if v >= 0 => v,
            _ => BASH_MAX_OUTPUT_DEFAULT,
        },
        Err(_) => BASH_MAX_OUTPUT_DEFAULT,
    };
    value.min(BASH_MAX_OUTPUT_UPPER_LIMIT)
}

fn background_tasks_disabled() -> bool {
    matches!(
        std::env::var("CODEMASON_DISABLE_BACKGROUND_TASKS").unwrap_or_default().to_lowercase().as_str(),
        "1" | "true" | "yes"
    )
}

fn background_usage_note() -> Option<&'static str> {
    if background_tasks_disabled() {
        return None;
    }
    Some("  - You can use the `run_in_background` parameter to run the command in the background. Only use this if you don't need the result immediately and are OK being notified when the command completes later. You do not need to check the output right away - you'll be notified when it finishes.")
}

fn sleep_guidance() -> Option<&'static str> {
    if background_tasks_disabled() {
        return None;
    }
    Some("  - Avoid unnecessary `Start-Sleep` commands:\n    - Do not sleep between commands that can run immediately — just run them.\n    - If your command is long running and you would like to be notified when it finishes — simply run your command using `run_in_background`. There is no need to sleep in this case.\n    - Do not retry failing commands in a sleep loop — diagnose the root cause or consider an alternative approach.\n    - If waiting for a background task you started with `run_in_background`, you will be notified when it completes — do not poll.\n    - If you must poll an external process, use a check command rather than sleeping first.\n    - If you must sleep, keep the duration short (1-5 seconds) to avoid blocking the user.")
}

/// Version-specific syntax guidance (getEditionSection).
fn edition_section(edition: Option<&str>) -> &'static str {
    match edition {
        Some("desktop") => "PowerShell edition: Windows PowerShell 5.1 (powershell.exe)\n   - Pipeline chain operators `&&` and `||` are NOT available — they cause a parser error. To run B only if A succeeds: `A; if ($?) { B }`. To chain unconditionally: `A; B`.\n   - Ternary (`?:`), null-coalescing (`??`), and null-conditional (`?.`) operators are NOT available. Use `if/else` and explicit `$null -eq` checks instead.\n   - Avoid `2>&1` on native executables. In 5.1, redirecting a native command's stderr inside PowerShell wraps each line in an ErrorRecord (NativeCommandError) and sets `$?` to `$false` even when the exe returned exit code 0. stderr is already captured for you — don't redirect it.\n   - Default file encoding is UTF-16 LE (with BOM). When writing files other tools will read, pass `-Encoding utf8` to `Out-File`/`Set-Content`.\n   - `ConvertFrom-Json` returns a PSCustomObject, not a hashtable. `-AsHashtable` is not available.",
        Some("core") => "PowerShell edition: PowerShell 7+ (pwsh)\n   - Pipeline chain operators `&&` and `||` ARE available and work like bash. Prefer `cmd1 && cmd2` over `cmd1; cmd2` when cmd2 should only run if cmd1 succeeds.\n   - Ternary (`$cond ? $a : $b`), null-coalescing (`??`), and null-conditional (`?.`) operators are available.\n   - Default file encoding is UTF-8 without BOM.",
        _ => "PowerShell edition: unknown — assume Windows PowerShell 5.1 for compatibility\n   - Do NOT use `&&`, `||`, ternary `?:`, null-coalescing `??`, or null-conditional `?.`. These are PowerShell 7+ only and parser-error on 5.1.\n   - To chain commands conditionally: `A; if ($?) { B }`. Unconditionally: `A; B`.",
    }
}

/// Build the PowerShell tool description (getPrompt).
pub fn get_prompt() -> String {
    let background_note = background_usage_note();
    let sleep_g = sleep_guidance();
    let edition = get_powershell_edition();
    let max_timeout = get_max_timeout_ms();
    let default_timeout = get_default_timeout_ms();

    let bg = background_note.map(|n| format!("{n}\n")).unwrap_or_default();
    let sg = sleep_g.map(|n| format!("{n}\n")).unwrap_or_default();

    format!(
        r#"Executes a given PowerShell command with optional timeout. Working directory persists between commands; shell state (variables, functions) does not.

IMPORTANT: This tool is for terminal operations via PowerShell: git, npm, docker, and PS cmdlets. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

{edition}

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use `Get-ChildItem` (or `ls`) to verify the parent directory exists and is the correct location

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes
   - Capture the output of the command.

PowerShell Syntax Notes:
   - Variables use $ prefix: $myVar = "value"
   - Escape character is backtick (`), not backslash
   - Use Verb-Noun cmdlet naming: Get-ChildItem, Set-Location, New-Item, Remove-Item
   - Common aliases: ls (Get-ChildItem), cd (Set-Location), cat (Get-Content), rm (Remove-Item)
   - Pipe operator | works similarly to bash but passes objects, not text
   - Use Select-Object, Where-Object, ForEach-Object for filtering and transformation
   - String interpolation: "Hello $name" or "Hello $($obj.Property)"
   - Registry access uses PSDrive prefixes: `HKLM:\SOFTWARE\...`, `HKCU:\...` — NOT raw `HKEY_LOCAL_MACHINE\...`
   - Environment variables: read with `$env:NAME`, set with `$env:NAME = "value"` (NOT `Set-Variable` or bash `export`)
   - Call native exe with spaces in path via call operator: `& "C:\Program Files\App\app.exe" arg1 arg2`

Interactive and blocking commands (will hang — this tool runs with -NonInteractive):
   - NEVER use `Read-Host`, `Get-Credential`, `Out-GridView`, `$Host.UI.PromptForChoice`, or `pause`
   - Destructive cmdlets (`Remove-Item`, `Stop-Process`, `Clear-Content`, etc.) may prompt for confirmation. Add `-Confirm:$false` when you intend the action to proceed. Use `-Force` for read-only/hidden items.
   - Never use `git rebase -i`, `git add -i`, or other commands that open an interactive editor

Passing multiline strings (commit messages, file content) to native executables:
   - Use a single-quoted here-string so PowerShell does not expand `$` or backticks inside. The closing `'@` MUST be at column 0 (no leading whitespace) on its own line — indenting it is a parse error:
<example>
git commit -m @'
Commit message here.
Second line with $literal dollar signs.
'@
</example>
   - Use `@'...'@` (single-quoted, literal) not `@"..."@` (double-quoted, interpolated) unless you need variable expansion
   - For arguments containing `-`, `@`, or other characters PowerShell parses as operators, use the stop-parsing token: `git log --% --format=%H`

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in milliseconds (up to {max_timeout}ms / {max_timeout_min} minutes). If not specified, commands will timeout after {default_timeout}ms ({default_timeout_min} minutes).
  - It is very helpful if you write a clear, concise description of what this command does.
  - If the output exceeds {max_output} characters, output will be truncated before being returned to you.
{bg}  - Avoid using PowerShell to run commands that have dedicated tools, unless explicitly instructed:
    - File search: Use {glob} (NOT Get-ChildItem -Recurse)
    - Content search: Use {grep} (NOT Select-String)
    - Read files: Use {read} (NOT Get-Content)
    - Edit files: Use {edit}
    - Write files: Use {write} (NOT Set-Content/Out-File)
    - Communication: Output text directly (NOT Write-Output/Write-Host)
  - When issuing multiple commands:
    - If the commands are independent and can run in parallel, make multiple {psname} tool calls in a single message.
    - If the commands depend on each other and must run sequentially, chain them in a single {psname} call (see edition-specific chaining syntax above).
    - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail.
    - DO NOT use newlines to separate commands (newlines are ok in quoted strings and here-strings)
  - Do NOT prefix commands with `cd` or `Set-Location` -- the working directory is already set to the correct project directory automatically.
{sg}  - For git commands:
    - Prefer to create a new commit rather than amending an existing commit.
    - Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach.
    - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue."#,
        edition = edition_section(edition),
        max_timeout = max_timeout,
        max_timeout_min = max_timeout / 60000,
        default_timeout = default_timeout,
        default_timeout_min = default_timeout / 60000,
        max_output = get_max_output_length(),
        bg = bg,
        sg = sg,
        glob = GLOB_TOOL_NAME,
        grep = GREP_TOOL_NAME,
        read = FILE_READ_TOOL_NAME,
        edit = FILE_EDIT_TOOL_NAME,
        write = FILE_WRITE_TOOL_NAME,
        psname = POWERSHELL_TOOL_NAME,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_contains_key_sections() {
        let p = get_prompt();
        assert!(p.contains("Executes a given PowerShell command"));
        assert!(p.contains("PowerShell edition:"));
        assert!(p.contains("stop-parsing token"));
    }

    #[test]
    fn edition_variants() {
        assert!(edition_section(Some("desktop")).contains("5.1"));
        assert!(edition_section(Some("core")).contains("7+"));
        assert!(edition_section(None).contains("unknown"));
    }
}
