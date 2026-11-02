//! PowerShell tool surface + execution substrate (adapted from PowerShellTool.tsx).
//!
//! Native-Windows tier-2 tool: the model emits PowerShell (or cmd via `cmd /c`),
//! and this runs it through powershell.exe/pwsh with -EncodedCommand. Gated to
//! Windows (isPowerShellToolEnabled parity). The security decision lives in the
//! permission engine (permissions.rs); execution mirrors the Bash tool's
//! PersistentShell but for PowerShell, with a Get-Location cwd sentinel.
//!
//! NOTE (schema): the PowerShell input schema IS in schemas/tool_schemas.json
//! under the "PowerShell" key. The pydantic model imports cleanly on Linux even
//! though the tool only registers on Windows, so the dump was taken the same way
//! as the other 18 tools and `schema_matches_python_dump` below asserts equality.
//!
//! NOTE (cross-cluster): the run_in_background branch calls
//! `tasks::local_shell_task::spawn_shell_task` (cluster TB). That code sits under
//! `#[cfg(windows)]`, so it does not gate the Linux build; it must exist before a
//! Windows build. The whole execution path is Windows-only — off Windows,
//! is_enabled() is false and call() returns the not-available result.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::tools::base::{
    CanUseToolFn, PermissionBehavior, PermissionResult, Tool, ToolError, ToolResult, ToolStream,
    ValidationResult,
};
use crate::types::query::ToolUseContext;
use crate::utils::platform::is_windows;
use crate::utils::powershell::parser::parse_powershell_command;

use super::destructive_command_warning::get_destructive_command_warning;
use super::permissions::powershell_tool_has_permission;
use super::prompt::get_prompt;
use super::tool_name::POWERSHELL_TOOL_NAME;

const DEFAULT_TIMEOUT_MS: i64 = 120_000;
const MAX_TIMEOUT_MS: i64 = 600_000;
const MAX_DISPLAY_COMMAND_CHARS: usize = 400;

fn is_truthy(v: &str) -> bool {
    matches!(v, "1" | "true" | "yes" | "on")
}
fn is_falsy(v: &str) -> bool {
    matches!(v, "0" | "false" | "no" | "off")
}

/// Windows-only. Ant defaults on (opt-out via env=0); external opt-in (env=1).
pub fn is_powershell_tool_enabled() -> bool {
    if !is_windows() {
        return false;
    }
    let env = std::env::var("CODEMASON_USE_POWERSHELL_TOOL")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if std::env::var("USER_TYPE").ok().as_deref() == Some("ant") {
        !is_falsy(&env) // opt-out
    } else {
        is_truthy(&env) // opt-in
    }
}

/// Input schema literal mirroring the pydantic PowerShellInput model.
static SCHEMA: Lazy<Value> = Lazy::new(|| {
    serde_json::json!({
        "additionalProperties": false,
        "description": "Input schema for the PowerShell tool.",
        "properties": {
            "command": {
                "description": "The PowerShell command to execute",
                "title": "Command",
                "type": "string"
            },
            "timeout": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": null,
                "description": format!("Optional timeout in milliseconds (max {MAX_TIMEOUT_MS})"),
                "title": "Timeout"
            },
            "run_in_background": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": null,
                "description": "Set to true to run this command in the background. Use Read to read the output later.",
                "title": "Run In Background"
            },
            "description": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": null,
                "description": "Clear, concise description of what this command does in active voice.",
                "title": "Description"
            }
        },
        "required": ["command"],
        "title": "PowerShellInput",
        "type": "object"
    })
});

/// Output of a PowerShell run.
#[derive(Debug, Clone, Default)]
pub struct PowerShellOutput {
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
    pub interrupted: bool,
    pub backgrounded: bool,
}

/// Render one PowerShell result for the assistant (renderResultForAssistant).
fn render_result_for_assistant(result: &PowerShellOutput) -> String {
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

pub struct PowerShellTool;

#[async_trait]
impl Tool for PowerShellTool {
    fn name(&self) -> &'static str {
        POWERSHELL_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    fn max_result_size_chars(&self) -> f64 {
        30_000.0
    }

    async fn prompt(&self) -> String {
        get_prompt()
    }

    async fn description(&self, input: &Value) -> String {
        let description = input.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        if !description.is_empty() {
            description.to_string()
        } else if !command.is_empty() {
            command.to_string()
        } else {
            "Execute PowerShell command".to_string()
        }
    }

    fn is_read_only(&self) -> bool {
        false
    }

    async fn is_enabled(&self) -> bool {
        is_powershell_tool_enabled()
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        POWERSHELL_TOOL_NAME.to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        if command.trim().is_empty() {
            return ValidationResult::failure("Command cannot be empty");
        }
        ValidationResult::success()
    }

    /// Parse the command and run the PowerShell permission engine. A 'deny'
    /// hard-blocks; 'ask' prompts (with the security reason); 'allow'
    /// auto-approves read-only commands.
    async fn check_permissions(&self, input: &Value, _context: &ToolUseContext) -> PermissionResult {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("").trim();
        if command.is_empty() {
            let mut r = PermissionResult::allow();
            r.message = "Empty command is safe".to_string();
            return r;
        }
        let parsed = parse_powershell_command(command);
        let mut result = powershell_tool_has_permission(command, &parsed);
        // Attach a destructive-command note to the message if present.
        if let Some(warning) = get_destructive_command_warning(command) {
            if result.behavior != PermissionBehavior::Allow {
                result.message = format!("{} ({warning})", result.message);
            }
        }
        result
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("").trim();
        let display: String = if command.chars().count() > MAX_DISPLAY_COMMAND_CHARS {
            format!("{}...", command.chars().take(MAX_DISPLAY_COMMAND_CHARS).collect::<String>())
        } else {
            command.to_string()
        };
        let lines: Vec<&str> = display.split('\n').filter(|l| !l.trim().is_empty()).collect();
        lines
            .iter()
            .enumerate()
            .map(|(i, ln)| if i == 0 { format!("PS> {ln}") } else { format!("    {ln}") })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn call(&self, input: Value, context: ToolUseContext, _can_use_tool: Option<CanUseToolFn>) -> ToolStream {
        Box::pin(async_stream::stream! {
            let command = input.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut timeout = input.get("timeout").and_then(|v| v.as_i64()).unwrap_or(DEFAULT_TIMEOUT_MS);
            if timeout > MAX_TIMEOUT_MS {
                timeout = MAX_TIMEOUT_MS;
            }
            let run_in_background = input.get("run_in_background").and_then(|v| v.as_bool()).unwrap_or(false);

            match run_powershell(&command, timeout, run_in_background, &input, &context).await {
                Ok(item) => yield Ok(item),
                Err(e) => yield Err(e),
            }
        })
    }
}

// === Execution substrate (Windows-only runtime) ===

/// Off Windows the tool is disabled (is_enabled() = false); this fallback keeps
/// call() total and compile-clean on non-Windows targets.
#[cfg(not(windows))]
async fn run_powershell(
    _command: &str,
    _timeout_ms: i64,
    _run_in_background: bool,
    _input: &Value,
    _context: &ToolUseContext,
) -> Result<ToolResult, ToolError> {
    let out = PowerShellOutput {
        stderr: "PowerShell is not available on this system.".to_string(),
        ..Default::default()
    };
    Ok(ToolResult::Result {
        result_for_assistant: render_result_for_assistant(&out),
        data: Some(serde_json::json!({"stdout": out.stdout, "stderr": out.stderr, "code": out.code})),
    })
}

#[cfg(windows)]
async fn run_powershell(
    command: &str,
    timeout_ms: i64,
    run_in_background: bool,
    input: &Value,
    context: &ToolUseContext,
) -> Result<ToolResult, ToolError> {
    use crate::utils::state::{get_cwd, get_original_cwd, is_in_directory, set_cwd};
    use super::command_semantics::interpret_command_result;

    // run_in_background: spawn detached through the PowerShell substrate.
    if run_in_background {
        let description = input
            .get("description")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(command)
            .to_string();
        let tool_use_id = context.tool_use_id.clone().or_else(|| context.message_id.clone());
        let (task_id, output_path) = crate::tasks::local_shell_task::spawn_shell_task(
            command,
            &get_cwd(),
            Some(&description),
            tool_use_id.as_deref(),
            "powershell",
        );
        let info = format!(
            "Command running in background with ID: {task_id}. Output is being written to: {}\nUse Read to read the output later.",
            output_path.display()
        );
        return Ok(ToolResult::Result {
            result_for_assistant: info,
            data: Some(serde_json::json!({
                "backgroundTaskId": task_id,
                "output_path": output_path.to_string_lossy(),
            })),
        });
    }

    let mut result = powershell_exec(command, timeout_ms, &context.abort_controller).await;

    // Interpret exit code with external-exe semantics (grep/robocopy/etc.).
    let (is_error, semantic_msg) = interpret_command_result(command, result.code, &result.stdout, &result.stderr);
    if is_error && result.stderr.trim().is_empty() && semantic_msg.is_none() {
        result.stderr = format!("{}\nCommand failed with exit code {}", result.stderr, result.code).trim().to_string();
    } else if !is_error {
        if let Some(msg) = semantic_msg {
            if result.stdout.trim().is_empty() {
                result.stdout = msg;
            }
        }
    }

    // Keep the shell cwd inside the session dir (Bash-tool parity).
    let cwd = get_cwd();
    if !is_in_directory(&cwd, &get_original_cwd()) {
        set_cwd(&get_original_cwd());
    }

    Ok(ToolResult::Result {
        result_for_assistant: render_result_for_assistant(&result),
        data: Some(serde_json::json!({"stdout": result.stdout, "stderr": result.stderr, "code": result.code})),
    })
}

#[cfg(windows)]
fn to_utf16le_base64(text: &str) -> String {
    use base64::Engine;
    let bytes: Vec<u8> = text.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(windows)]
fn ps_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

#[cfg(windows)]
async fn powershell_exec(
    command: &str,
    timeout_ms: i64,
    abort: &tokio_util::sync::CancellationToken,
) -> PowerShellOutput {
    use crate::utils::state::{get_cwd, set_cwd};
    use crate::utils::windows_shell::{apply_spawn_group, kill_process_tree};
    use std::process::Stdio;

    let Some(pwsh) = crate::utils::powershell_detect::get_cached_powershell_path() else {
        return PowerShellOutput { stderr: "PowerShell is not available on this system.".into(), ..Default::default() };
    };

    let timeout = std::time::Duration::from_millis(timeout_ms.max(0) as u64);
    // Persist cwd via a Get-Location sentinel file.
    let pwd_file = std::env::temp_dir().join(format!("cm-psloc-{}", uuid::Uuid::new_v4().simple()));
    let pwd_path = pwd_file.to_string_lossy().to_string();
    let wrapped = format!(
        "{command}\n$__cmec = if ($LASTEXITCODE -ne $null) {{ $LASTEXITCODE }} elseif ($?) {{ 0 }} else {{ 1 }}\n(Get-Location).Path | Set-Content -Encoding utf8 -Path {}\nexit $__cmec\n",
        ps_single_quote(&pwd_path)
    );
    let encoded = to_utf16le_base64(&wrapped);

    let mut cmd = tokio::process::Command::new(&pwsh);
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-NoLogo",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        &encoded,
    ])
    .current_dir(get_cwd())
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    apply_spawn_group(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = std::fs::remove_file(&pwd_file);
            return PowerShellOutput { stderr: format!("Failed to spawn PowerShell: {e}"), ..Default::default() };
        }
    };
    let pid = child.id();

    let output = tokio::select! {
        res = tokio::time::timeout(timeout, child.wait_with_output()) => res,
        _ = abort.cancelled() => {
            if let Some(p) = pid { kill_process_tree(p, None); }
            let _ = std::fs::remove_file(&pwd_file);
            return PowerShellOutput {
                stderr: String::new(), code: -1, interrupted: true, ..Default::default()
            };
        }
    };

    let result = match output {
        Ok(Ok(out)) => {
            let code = out.status.code().unwrap_or(0);
            // Adopt the post-command cwd so Set-Location persists.
            if let Ok(new_cwd) = std::fs::read_to_string(&pwd_file) {
                let new_cwd = new_cwd.trim_start_matches('\u{feff}').trim().to_string();
                if !new_cwd.is_empty() && std::path::Path::new(&new_cwd).is_dir() {
                    set_cwd(&new_cwd);
                }
            }
            PowerShellOutput {
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                code,
                ..Default::default()
            }
        }
        Ok(Err(e)) => PowerShellOutput { stderr: format!("PowerShell execution failed: {e}"), ..Default::default() },
        Err(_) => {
            if let Some(p) = pid { kill_process_tree(p, None); }
            let secs = timeout_ms as f64 / 1000.0;
            PowerShellOutput { stderr: format!("Command timed out after {secs}s"), code: -1, interrupted: true, ..Default::default() }
        }
    };
    let _ = std::fs::remove_file(&pwd_file);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The input schema must be byte-equal to the pydantic model_json_schema()
    /// dump (schemas/tool_schemas.json) — the model-facing contract.
    #[test]
    fn schema_matches_python_dump() {
        let dump: serde_json::Value =
            serde_json::from_str(include_str!("../../../schemas/tool_schemas.json")).unwrap();
        let expected = &dump["PowerShell"]["schema"];
        let actual = &*SCHEMA;
        assert_eq!(actual, expected, "schema for PowerShell diverges from the Python dump");
    }

    #[test]
    fn schema_has_command_required() {
        let s = &*SCHEMA;
        assert_eq!(s["required"][0], "command");
        assert_eq!(s["properties"]["command"]["type"], "string");
        assert_eq!(s["additionalProperties"], false);
        assert_eq!(s["properties"]["timeout"]["description"], "Optional timeout in milliseconds (max 600000)");
    }

    #[test]
    fn disabled_off_windows() {
        // On this (non-Windows) build target the tool must be disabled.
        if !is_windows() {
            assert!(!is_powershell_tool_enabled());
        }
    }

    #[test]
    fn render_result_combines_streams() {
        let out = PowerShellOutput { stdout: "hi".into(), stderr: "err".into(), code: 1, ..Default::default() };
        assert_eq!(render_result_for_assistant(&out), "hi\nerr");
    }

    #[test]
    fn render_tool_use_prefixes_ps() {
        let t = PowerShellTool;
        let msg = t.render_tool_use_message(&serde_json::json!({"command": "git status"}), false);
        assert_eq!(msg, "PS> git status");
    }
}
