//! Hooks — based on the Stop-hook path from src/utils/hooks.ts in Claude Code
//! + src/query/stopHooks.ts. Equivalent to codemason/utils/hooks.py.
//!
//! What this is: the mechanism that lets a user-configured "Stop" hook keep the
//! model working past a natural turn-end. When the model ends a turn with no
//! tool call (query.ts:1267), the configured Stop hooks run; if one "blocks"
//! (exit code 2, or stdout {"decision":"block","reason":...}), that feedback is
//! injected back into the conversation and the loop CONTINUES — the model
//! resumes instead of stopping. A hook returning {"continue":false} ends the
//! turn instead. `stop_hook_active` is passed to the hook and set true on the
//! continue path so a hook can detect it is already in a stop-hook continuation
//! and avoid looping forever (the hook owns loop-termination — documented
//! contract).
//!
//! SUBSTRATE / SCOPE (NOT ported, deliberately): the original hooks subsystem
//! is ~4000 lines spanning many events (PreToolUse/PostToolUse/…), HTTP- and
//! MCP-transport hooks, elicitation, the trust/permission UI, analytics,
//! teammate/task hooks, and an Ink progress display. CodeMason ports ONLY the
//! Stop event with command-type hooks — the piece relevant to autonomous
//! continuation. Hooks are read from ~/.codemason/settings.json under
//! ["hooks"]["Stop"], matching Claude Code's settings.json schema.

use serde_json::{json, Value};

use crate::types::messages::{ContentBlock, ContentBlockParam};
use crate::types::query::{AssistantMessage, Message, ToolUseContext, UserMessage};
use crate::utils::config::SETTINGS_FILE;
use crate::utils::log::debug_log;
use crate::utils::platform::is_windows;

/// hooks.ts TOOL_HOOK_EXECUTION_TIMEOUT_MS = 10*60*1000 (600s), in seconds.
/// (Was 60 — that killed legitimate 1-10min Stop hooks, e.g. test suites.)
pub const DEFAULT_HOOK_TIMEOUT_SEC: f64 = 600.0;

/// HookBlockingError (hooks.ts:330-ish): a hook's blocking feedback.
#[derive(Debug, Clone)]
pub struct HookBlockingError {
    pub blocking_error: String,
    pub command: String,
}

fn read_settings() -> Value {
    if !SETTINGS_FILE.exists() {
        return json!({});
    }
    std::fs::read_to_string(&*SETTINGS_FILE)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or_else(|| json!({}))
}

/// settings.json["hooks"][event] -> list of matcher entries, each
/// {matcher?, hooks: [{type, command, timeout?}]}.
pub fn get_hooks_config(event: &str) -> Vec<Value> {
    read_settings()
        .get("hooks")
        .and_then(|h| h.get(event))
        .and_then(|e| e.as_array().cloned())
        .unwrap_or_default()
}

/// hasHookForEvent (hooks.ts:1582) — is any hook configured for this event?
pub fn has_hook_for_event(event: &str) -> bool {
    !get_hooks_config(event).is_empty()
}

/// Flatten configured command hooks for an event. Stop hooks carry no match
/// query (getMatchingHooks: the Stop case leaves matchQuery undefined, so
/// every matcher entry applies), so we run all command-type hooks under the
/// event.
fn matched_command_hooks(event: &str) -> Vec<Value> {
    let mut out = Vec::new();
    for matcher_entry in get_hooks_config(event) {
        let Some(hooks) = matcher_entry.get("hooks").and_then(|h| h.as_array()) else {
            continue;
        };
        for h in hooks {
            let is_command = h.get("type").and_then(|t| t.as_str()) == Some("command");
            let has_command = h
                .get("command")
                .and_then(|c| c.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if is_command && has_command {
                out.push(h.clone());
            }
        }
    }
    out
}

/// getStopHookMessage (hooks.ts:1894).
pub fn get_stop_hook_message(blocking_error: &str) -> String {
    format!("Stop hook feedback:\n{blocking_error}")
}

/// Partial result of one hook run.
#[derive(Debug, Default)]
struct HookRunResult {
    blocking_error: Option<HookBlockingError>,
    prevent_continuation: bool,
    #[allow(dead_code)]
    stop_reason: Option<String>,
    /// Non-blocking execution failure (logged, neither blocks nor prevents).
    error: Option<String>,
}

/// processHookJSONOutput (hooks.ts:518) — Stop event subset. continue:false ->
/// preventContinuation (+stopReason); decision:'block' -> blockingError
/// (reason); decision:'approve' is a no-op for Stop; any other decision value
/// is an error (the original THROWS 'Unknown hook decision type' — here logged
/// and returned as a non-blocking 'error' result, so it neither blocks nor
/// prevents continuation).
fn process_hook_json(js: &Value, command: &str) -> HookRunResult {
    let mut result = HookRunResult::default();
    // handleSyncJSONOutput (hooks.ts:518): continue === false -> preventContinuation.
    if js.get("continue") == Some(&Value::Bool(false)) {
        result.prevent_continuation = true;
        if let Some(sr) = js.get("stopReason").and_then(|s| s.as_str()) {
            if !sr.is_empty() {
                result.stop_reason = Some(sr.to_string());
            }
        }
    }
    // decision (hooks.ts:528-543).
    if let Some(decision) = js.get("decision") {
        if !decision.is_null() {
            let d = decision.as_str().unwrap_or("");
            if d == "block" {
                let reason = js
                    .get("reason")
                    .and_then(|r| r.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Blocked by hook");
                result.blocking_error = Some(HookBlockingError {
                    blocking_error: reason.to_string(),
                    command: command.to_string(),
                });
            } else if d != "approve" {
                debug_log(
                    "STOP_HOOK_UNKNOWN_DECISION",
                    Some(&json!({"decision": decision, "command": command})),
                );
                result.error = Some(format!(
                    "Unknown hook decision type: {}. Valid types are: approve, block",
                    match decision {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    }
                ));
            }
        }
    }
    result
}

/// Run one command hook: stdin = JSON hook_input. Parse its decision per
/// hooks.ts handleSyncJSONOutput (continue:false / decision:'block') plus the
/// exit-code-2 blocking convention.
async fn run_command_hook(hook: &Value, hook_input: &Value) -> HookRunResult {
    let command = hook.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let timeout = hook
        .get("timeout")
        .and_then(|t| t.as_f64())
        .unwrap_or(DEFAULT_HOOK_TIMEOUT_SEC);
    let payload = hook_input.to_string();

    // Hook commands are user-authored, usually POSIX-sh one-liners. On native
    // Windows a cmd.exe shell breaks them; route through git-bash instead (same
    // as the Bash substrate). POSIX/WSL2 keep the plain shell unchanged.
    // A git-bash discovery failure lands in the same "failed to run" arm as the
    // Python's spawn-time exception.
    let hook_argv = match crate::utils::windows_shell::shell_argv(&command) {
        Ok(argv) => argv,
        Err(e) => {
            return HookRunResult {
                error: Some(format!("Stop hook failed to run ({command}): {e}")),
                ..Default::default()
            }
        }
    };
    let mut cmd = match &hook_argv {
        Some(argv) if !argv.is_empty() => {
            let mut c = tokio::process::Command::new(&argv[0]);
            c.args(&argv[1..]);
            c
        }
        _ => {
            let mut c = tokio::process::Command::new("/bin/sh");
            c.arg("-c").arg(&command);
            c
        }
    };
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // On timeout the in-flight wait future is dropped; kill_on_drop makes
        // that drop kill the shell — the same "reap only the shell" POSIX
        // behaviour as the Python proc.kill().
        .kill_on_drop(true);
    // no_window: suppress the console flash on Windows (#61051) without
    // changing the POSIX behaviour of hooks.
    crate::utils::windows_shell::apply_no_window(&mut cmd);

    let mut proc = match cmd.spawn() {
        Ok(p) => p,
        Err(e) => {
            return HookRunResult {
                error: Some(format!("Stop hook failed to run ({command}): {e}")),
                ..Default::default()
            }
        }
    };

    let child_pid = proc.id();

    // Feed stdin and collect output INSIDE one timeout, like Python's
    // asyncio.wait_for(proc.communicate(payload), timeout) (hooks.py:160-161).
    // The write must be under the deadline too: a hook that never reads stdin
    // plus a payload larger than the OS pipe buffer blocks write_all forever,
    // and with the write outside the wrapper the timeout never armed.
    let stdin = proc.stdin.take();
    let payload_bytes = payload.as_bytes().to_vec();
    let output = tokio::time::timeout(std::time::Duration::from_secs_f64(timeout), async move {
        if let Some(mut stdin) = stdin {
            use tokio::io::AsyncWriteExt;
            let _ = stdin.write_all(&payload_bytes).await;
            drop(stdin); // EOF for the hook
        }
        proc.wait_with_output().await
    })
    .await;

    let (exit_code, stdout, stderr) = match output {
        Err(_) => {
            // Timeout. The spawned proc is the shell; the real hook runs as its
            // CHILD. On Windows taskkill /T kills the whole tree by PID
            // (upstream #77078: a suspended/hung hook survives its timeout for
            // 30-60 min otherwise); POSIX keeps a plain kill via kill_on_drop
            // (the hook shares our group, so killpg would take CodeMason down
            // too — unchanged, pre-existing behaviour).
            if is_windows() {
                if let Some(pid) = child_pid {
                    crate::utils::windows_shell::kill_process_tree(pid, None);
                }
            }
            return HookRunResult {
                error: Some(format!("Stop hook timed out after {timeout}s: {command}")),
                ..Default::default()
            };
        }
        Ok(Err(e)) => {
            return HookRunResult {
                error: Some(format!("Stop hook failed to run ({command}): {e}")),
                ..Default::default()
            }
        }
        Ok(Ok(out)) => (
            out.status.code(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ),
    };

    // Decision precedence mirrors runHook (hooks.ts:2499+): parse stdout JSON
    // FIRST — if a JSON object is present, its decision is honored regardless
    // of exit code; ONLY non-JSON output falls back to the exit-code
    // convention. parseHookOutput (hooks.ts:399): JSON is attempted only when
    // trimmed stdout starts with '{'; a '{' that fails to parse is treated as
    // plain text (falls through).
    let trimmed = stdout.trim();
    if trimmed.starts_with('{') {
        if let Ok(js) = serde_json::from_str::<Value>(trimmed) {
            if js.is_object() {
                return process_hook_json(&js, &command);
            }
        }
    }

    // Non-JSON output -> exit-code convention (hooks.ts:2613+):
    //   exit 2     -> blocking feedback (raw stderr is the message),
    //   other != 0 -> non-blocking error (logged, does not block),
    //   0          -> success / no decision.
    match exit_code {
        Some(2) => HookRunResult {
            blocking_error: Some(HookBlockingError {
                blocking_error: format!(
                    "[{command}]: {}",
                    if stderr.is_empty() { "No stderr output" } else { &stderr }
                ),
                command: command.clone(),
            }),
            ..Default::default()
        },
        Some(0) | None => HookRunResult::default(),
        Some(code) => {
            debug_log(
                "STOP_HOOK_NONBLOCKING_ERROR",
                Some(&json!({
                    "command": command,
                    "exit_code": code,
                    "stderr": stderr.chars().take(500).collect::<String>(),
                })),
            );
            HookRunResult::default()
        }
    }
}

/// Text of the final assistant message, so hooks can inspect the response
/// without reading a transcript file (executeStopHooks: last_assistant_message).
fn last_assistant_text(assistant_message: &AssistantMessage) -> Option<String> {
    let parts: Vec<String> = assistant_message
        .message
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    let text = parts.join("\n").trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// True iff any tool ran since the last GENUINE user message (i.e. the turn
/// did real agentic work rather than a pure conversational reply).
///
/// The engine reports this fact to Stop hooks so a keep-going policy can avoid
/// nudging chat turns. Decision stays in the hook; the engine only observes.
///
/// Walk history backward to the boundary user input, counting intermediate
/// assistant tool_use blocks. Skipped as non-boundary: tool_result user
/// messages (part of the chain) and isMeta user messages (the max-output
/// recovery nudge / a prior Stop-hook block injection).
fn used_tools_since_last_user(messages: &[Message]) -> bool {
    for m in messages.iter().rev() {
        match m {
            Message::User(u) => {
                let has_tool_result = u
                    .message
                    .content
                    .as_blocks()
                    .iter()
                    .any(|b| matches!(b, ContentBlockParam::ToolResult { .. }));
                if has_tool_result || u.is_meta {
                    continue; // part of the tool chain / injected meta — not a boundary
                }
                return false; // reached the genuine user input with no tool use in between
            }
            Message::Assistant(a) => {
                if a.message
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse(_)))
                {
                    return true;
                }
            }
            Message::Progress(_) => {}
        }
    }
    false
}

/// executeStopHooks (hooks.ts:3639) — Stop event, command hooks only. Runs all
/// configured Stop hooks concurrently (as the original does) and returns their
/// parsed results.
async fn execute_stop_hooks(
    stop_hook_active: bool,
    last_assistant_text: Option<&str>,
    used_tools: bool,
) -> Vec<HookRunResult> {
    if !has_hook_for_event("Stop") {
        return Vec::new();
    }
    let hooks = matched_command_hooks("Stop");
    if hooks.is_empty() {
        return Vec::new();
    }

    // Real session identity + transcript path (hooks.ts payload contract) —
    // session persistence exists, so hooks can read the conversation.
    let hook_input = json!({
        "hook_event_name": "Stop",
        "stop_hook_active": stop_hook_active,
        "session_id": crate::utils::state::get_session_id(),
        "transcript_path": crate::utils::session_storage::get_transcript_path()
            .to_string_lossy(),
        "cwd": std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default(),
        "last_assistant_message": last_assistant_text,
        // Did this turn do real agentic work? Lets a keep-going hook skip pure
        // conversational replies (greetings/Q&A used no tools -> don't nudge).
        "used_tools": used_tools,
    });

    futures::future::join_all(hooks.iter().map(|h| run_command_hook(h, &hook_input))).await
}

/// Aggregated Stop-hook outcome (handleStopHooks return shape).
#[derive(Debug, Default)]
pub struct StopHookOutcome {
    /// Blocking feedback as isMeta user messages the model resumes from.
    pub blocking_errors: Vec<UserMessage>,
    pub prevent_continuation: bool,
}

/// handleStopHooks (query/stopHooks.ts:65) — Stop event subset. Aggregates
/// hook results; preventContinuation wins over blocking (matches the
/// original's early returns).
pub async fn handle_stop_hooks(
    messages: &[Message],
    assistant_message: &AssistantMessage,
    _tool_use_context: &ToolUseContext,
    stop_hook_active: bool,
) -> StopHookOutcome {
    let last_text = last_assistant_text(assistant_message);
    let used_tools = used_tools_since_last_user(messages);
    let results = execute_stop_hooks(stop_hook_active, last_text.as_deref(), used_tools).await;

    let mut blocking_errors: Vec<UserMessage> = Vec::new();
    let mut prevent_continuation = false;
    for r in results {
        if let Some(err) = &r.error {
            debug_log("STOP_HOOK_ERROR", Some(&json!({"error": err})));
            continue;
        }
        if let Some(be) = &r.blocking_error {
            let mut user_message = crate::utils::messages::create_user_message_text(
                &get_stop_hook_message(&be.blocking_error),
            );
            user_message.is_meta = true; // in history for the model; hidden in UI
            blocking_errors.push(user_message);
        }
        if r.prevent_continuation {
            prevent_continuation = true;
        }
    }

    if prevent_continuation {
        return StopHookOutcome { blocking_errors: Vec::new(), prevent_continuation: true };
    }
    StopHookOutcome { blocking_errors, prevent_continuation: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_block_decision() {
        let r = process_hook_json(&json!({"decision": "block", "reason": "keep going"}), "cmd");
        assert_eq!(r.blocking_error.as_ref().unwrap().blocking_error, "keep going");
        assert!(!r.prevent_continuation);
    }

    #[test]
    fn json_continue_false() {
        let r = process_hook_json(&json!({"continue": false, "stopReason": "done"}), "cmd");
        assert!(r.prevent_continuation);
        assert_eq!(r.stop_reason.as_deref(), Some("done"));
    }

    #[test]
    fn json_unknown_decision() {
        let r = process_hook_json(&json!({"decision": "maybe"}), "cmd");
        assert!(r.error.unwrap().starts_with("Unknown hook decision type: maybe"));
    }

    #[test]
    fn json_block_without_reason_defaults() {
        let r = process_hook_json(&json!({"decision": "block"}), "cmd");
        assert_eq!(r.blocking_error.unwrap().blocking_error, "Blocked by hook");
    }
}
