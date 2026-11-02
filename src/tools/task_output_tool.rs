//! TaskOutput tool — based on src/tools/TaskOutputTool/ from Claude Code.
//! Equivalent to codemason/tools/task_output_tool.py.
//!
//! Retrieves output from a background task. block=true (default) waits for the
//! task to finish (polling AppState.tasks), block=false returns the current
//! state. The result is rendered as the same
//! `<retrieval_status>/<task_id>/<status>/<output>` XML the original maps.
//! Marked deprecated in favor of Read-ing the output file path surfaced in the
//! tool result and <task-notification>.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::state::app_state::get_app_state;
use crate::task::now_ms;
use crate::tasks::local_shell_task::is_local_shell_task;
use crate::tasks::types::TaskState;
use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::task::disk_output::{get_task_output, DEFAULT_MAX_READ_BYTES};
use crate::utils::task::framework::update_task_state;
use crate::utils::task::output_formatting::format_task_output;

/// Output lines shown in the terminal for a completed task (verbose: all).
const RESULT_PREVIEW_LINES: usize = 10;

pub const TASK_OUTPUT_TOOL_NAME: &str = "TaskOutput";

/// Returned (as a normal result, NOT an error) when a task_id isn't
/// registered. The in-memory task table is empty right after a session resume,
/// so a model resuming a turn that had launched tasks would otherwise poll ids
/// that no longer exist and get a wall of hard errors. This tells it what
/// happened and what to do instead of retrying, and it also corrects the
/// launch-then-poll mental model: codemason runs agents synchronously and
/// returns their results inline.
const NOT_FOUND_GUIDANCE: &str = "No task is registered with ID {task_id}. Either it never existed, or its runtime state was cleared by a session restart — task state is in-memory and is NOT restored on resume. Do not keep polling this id. In codemason, agents run synchronously and return their results inline, so there is nothing to retrieve with TaskOutput; if you were waiting on earlier work, re-run it.";

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(TASK_OUTPUT_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

/// Port of semanticBoolean: accept bool or the strings 'true'/'false', falling
/// back to `default` when the value is absent.
fn semantic_bool(value: Option<&Value>, default: bool) -> bool {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) if s == "true" => true,
        Some(Value::String(s)) if s == "false" => false,
        // Python: bool(value) — truthiness of whatever else arrived.
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(default),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Unified output for any task type (getTaskOutputData).
///
/// codemason streams shell output to disk, so the bash path reads the file
/// (the original's in-memory ShellCommand.taskOutput is substrate). exitCode
/// is pulled from the task's recorded result.
fn get_task_output_data(task: &TaskState) -> Value {
    let output = get_task_output(&task.base().id, DEFAULT_MAX_READ_BYTES);
    let mut base = serde_json::json!({
        "task_id": task.base().id,
        "task_type": task.base().task_type,
        "status": task.base().status,
        "description": task.base().description,
        "output": output,
    });
    if let Some(shell) = task.as_local_shell().filter(|t| is_local_shell_task(t)) {
        base["exitCode"] = match &shell.result {
            Some(r) => Value::from(r.code),
            None => Value::Null,
        };
        return base;
    }

    if let Some(agent) = task.as_local_agent() {
        // Prefer the finished answer the task recorded over whatever is on
        // disk. The source's reason holds here too: the output file is the full
        // transcript — every message and tool use — while `result` holds only
        // the agent's final text, which is what the parent asked for. The file
        // is the fallback, and for a still-running agent it is all there is.
        let clean_result = agent
            .result
            .as_ref()
            .and_then(|r| r.get("text"))
            .and_then(|t| t.as_str())
            .map(|t| t.to_string())
            .filter(|t| !t.is_empty());
        let answer = clean_result.unwrap_or_else(|| output.clone());
        base["prompt"] = Value::from(agent.prompt.clone());
        base["result"] = Value::from(answer.clone());
        base["output"] = Value::from(answer);
        if let Some(error) = &agent.error {
            base["error"] = Value::from(error.clone());
        }
        // The counts the parent accounts an agent by — its context size and how
        // much work it did. Not in the source's TaskOutput shape, which reports
        // them through an SDK event instead; with no such consumer here, the
        // tool the parent already calls is where they belong.
        if let Some(progress) = &agent.progress {
            base["usage"] = serde_json::json!({
                "total_tokens": progress.token_count,
                "tool_uses": progress.tool_use_count,
            });
            if let Some(summary) = &progress.summary {
                base["summary"] = Value::from(summary.clone());
            }
        }
    }
    base
}

/// Poll until the task leaves running/pending or the timeout elapses
/// (waitForTaskCompletion). Returns the task, or None if it vanished.
async fn wait_for_task_completion(
    task_id: &str,
    timeout_ms: i64,
    context: &ToolUseContext,
) -> Option<TaskState> {
    let start_time = now_ms();
    // Show the wait. The model deciding to wait here is correct; what went
    // wrong on 2026-09-07 was that nothing on screen said so, the silence
    // read as a hang ("prefill?"), and the wait was removed instead of shown.
    // The spinner already shows the tool and the elapsed time; this adds what
    // it is waiting on and the task's latest line, refreshed twice a second.
    let _clear = crate::utils::tokens::WaitNoteGuard;
    let mut last_note_at: i64 = 0;
    while now_ms() - start_time < timeout_ms {
        // Honor an interrupt: stop waiting and return the current snapshot.
        if context.abort_controller.is_cancelled() {
            break;
        }
        let task = get_app_state().tasks.get(task_id).cloned();
        let Some(task) = task else {
            return None;
        };
        if task.base().status != "running" && task.base().status != "pending" {
            return Some(task);
        }
        if now_ms() - last_note_at >= 500 {
            crate::utils::tokens::set_wait_note(Some(wait_note_for(&task)));
            last_note_at = now_ms();
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    get_app_state().tasks.get(task_id).cloned()
}

/// ' waiting on b1a2c3 · python scrape.py · last: 5500 saved' — the task, what
/// it is, and the last line it wrote so far.
fn wait_note_for(task: &TaskState) -> String {
    fn clip(s: &str, max: usize) -> String {
        let s = s.trim();
        if s.chars().count() <= max {
            s.to_string()
        } else {
            let head: String = s.chars().take(max.saturating_sub(1)).collect();
            format!("{head}…")
        }
    }
    // A shell task falls back to its command when it has no description; a
    // sub-agent always has one (the brief it was spawned with).
    let what = if !task.base().description.is_empty() {
        task.base().description.as_str()
    } else {
        task.as_local_shell().map(|t| t.command.as_str()).unwrap_or("")
    };
    let mut note = format!("waiting on {} · {}", task.base().id, clip(what, 40));
    let out = get_task_output(&task.base().id, 4096);
    if let Some(last) = out.lines().rev().find(|l| !l.trim().is_empty()) {
        note.push_str(&format!(" · last: {}", clip(last, 60)));
    }
    note
}

/// mapToolResultToToolResultBlockParam: render the result as XML.
fn map_result(data: &Value) -> String {
    let mut parts = vec![format!(
        "<retrieval_status>{}</retrieval_status>",
        data.get("retrieval_status").and_then(|s| s.as_str()).unwrap_or("")
    )];
    if let Some(message) = data.get("message").and_then(|m| m.as_str()) {
        if !message.is_empty() {
            parts.push(format!("<message>{message}</message>"));
        }
    }
    if let Some(task) = data.get("task").filter(|t| t.is_object()) {
        let s = |k: &str| task.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        parts.push(format!("<task_id>{}</task_id>", s("task_id")));
        parts.push(format!("<task_type>{}</task_type>", s("task_type")));
        parts.push(format!("<status>{}</status>", s("status")));
        if let Some(exit_code) = task.get("exitCode").filter(|v| !v.is_null()) {
            parts.push(format!("<exit_code>{exit_code}</exit_code>"));
        }
        let output = task.get("output").and_then(|o| o.as_str()).unwrap_or("");
        if !output.trim().is_empty() {
            let formatted = format_task_output(output, &s("task_id"));
            parts.push(format!(
                "<output>\n{}\n</output>",
                formatted.content.trim_end()
            ));
        }
        if let Some(err) = task.get("error").and_then(|e| e.as_str()) {
            if !err.is_empty() {
                parts.push(format!("<error>{err}</error>"));
            }
        }
        // The source's renderer stops here and routes an agent's token and tool
        // counts to an SDK consumer (the editor's subagent panel) instead.
        // There is no such consumer here, and the parent is the one that has to
        // decide whether an agent is growing where it should not be — so the
        // counts go into the result the parent already reads. Without this the
        // capability is simply lost rather than substituted.
        if let Some(usage) = task.get("usage") {
            parts.push(format!(
                "<usage><total_tokens>{}</total_tokens><tool_uses>{}</tool_uses></usage>",
                usage.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                usage.get("tool_uses").and_then(|v| v.as_i64()).unwrap_or(0),
            ));
        }
        if let Some(summary) = task.get("summary").and_then(|v| v.as_str()) {
            if !summary.is_empty() {
                parts.push(format!("<summary>{summary}</summary>"));
            }
        }
    }
    parts.join("\n\n")
}

fn mark_notified(task_id: &str) {
    update_task_state(task_id, |t| {
        let mut next = t.clone();
        next.base_mut().notified = true;
        Some(next)
    });
}

pub struct TaskOutputTool;

#[async_trait]
impl Tool for TaskOutputTool {
    fn name(&self) -> &'static str {
        TASK_OUTPUT_TOOL_NAME
    }

    /// Backward-compatible aliases for the renamed AgentOutput/BashOutput tools.
    fn aliases(&self) -> &'static [&'static str] {
        &["AgentOutputTool", "BashOutputTool"]
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    async fn description(&self, _input: &Value) -> String {
        "[Deprecated] — prefer Read on the task output file path".to_string()
    }

    /// TaskOutput has no `DESCRIPTION` in the Python, so the wire description
    /// resolves through TOOL_DESCRIPTIONS / the simple blurb rather than this
    /// tool's (much longer) system prompt. Returning None selects that chain.
    async fn wire_description(&self) -> Option<String> {
        None
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        // TaskOutputTool.isConcurrencySafe returns isReadOnly (true).
        self.is_read_only()
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Task Output".to_string()
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let task_id = input
            .get("task_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| input.get("shell_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
            .unwrap_or("");
        if task_id.is_empty() {
            String::new()
        } else {
            colorize(task_id, colors::DIM)
        }
    }

    /// Status line + the task's output. The model gets the XML form
    /// (map_result); the terminal must not.
    fn render_tool_result_message(&self, data: &Value, verbose: bool) -> Option<String> {
        if !data.is_object() {
            return None;
        }
        let task = data.get("task").filter(|t| t.is_object());
        let Some(task) = task else {
            if let Some(message) = data.get("message").and_then(|m| m.as_str()) {
                if !message.is_empty() {
                    // A not_found (e.g. after resume) reads as a dim note, not
                    // a red error — the model still gets the full guidance via
                    // the XML.
                    return Some(colorize(message, colors::DIM));
                }
            }
            let rs = data.get("retrieval_status").and_then(|s| s.as_str()).unwrap_or("");
            return if rs.is_empty() { None } else { Some(rs.to_string()) };
        };

        let status = task.get("status").and_then(|s| s.as_str()).unwrap_or("");
        let task_id = task.get("task_id").and_then(|s| s.as_str()).unwrap_or("");
        let mut head = format!("{task_id} · {status}");
        if let Some(exit_code) = task.get("exitCode").filter(|v| !v.is_null()) {
            head.push_str(&format!(" · exit {exit_code}"));
        }
        let mut lines = vec![colorize(&head, colors::DIM)];

        let output = task.get("output").and_then(|o| o.as_str()).unwrap_or("").trim_end();
        if !output.is_empty() {
            let out_lines: Vec<&str> = output.split('\n').collect();
            let shown =
                if verbose { out_lines.len() } else { out_lines.len().min(RESULT_PREVIEW_LINES) };
            lines.extend(out_lines[..shown].iter().map(|s| s.to_string()));
            if out_lines.len() > shown {
                let hidden = out_lines.len() - shown;
                lines.push(colorize(&format!("... ({hidden} more lines)"), colors::DIM));
            }
        }
        if let Some(err) = task.get("error").and_then(|e| e.as_str()) {
            if !err.is_empty() {
                lines.push(colorize(&format!("error: {err}"), colors::RED));
            }
        }
        Some(lines.join("\n"))
    }

    async fn prompt(&self) -> String {
        concat!(
            "DEPRECATED: Prefer using the Read tool on the task's output file path ",
            "instead. Background tasks return their output file path in the tool ",
            "result, and you receive a <task-notification> with the same path when ",
            "the task completes — Read that file directly.\n\n",
            "- Retrieves output from a running or completed task (background shell, ",
            "agent, or remote session)\n",
            "- Takes a task_id parameter identifying the task\n",
            "- Returns the task output along with status information\n",
            "- Use block=true (default) to wait for task completion\n",
            "- Use block=false for non-blocking check of current status\n",
            "- Task IDs can be found using the /tasks command\n",
            "- Works with all task types: background shells, async agents, and ",
            "remote sessions"
        )
        .to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let task_id = input.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
        if task_id.is_empty() {
            let mut v = ValidationResult::failure("Task ID is required");
            v.meta = Some(serde_json::json!({"errorCode": 1}));
            return v;
        }
        // An unregistered task_id is deliberately NOT a validation failure:
        // after a session resume the in-memory task table is empty, so polling
        // a pre-restart id would otherwise surface as a hard error. call()
        // returns an informative not_found result (with recovery guidance)
        // instead.
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let task_id = input
                .get("task_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let block = semantic_bool(input.get("block"), true);
            // The schema says 0..=600000, but the schema is advisory to the
            // model, not a validator here — clamp, or one call could hold an
            // unattended -dsp turn for as long as it likes.
            let timeout = input
                .get("timeout")
                .and_then(|t| t.as_i64())
                .unwrap_or(30000)
                .clamp(0, 600_000);

            let task = get_app_state().tasks.get(&task_id).cloned();
            let Some(task) = task else {
                // Not an error: return actionable guidance so a resumed
                // session doesn't spew hard failures polling tasks the restart
                // cleared (see NOT_FOUND_GUIDANCE).
                let data = serde_json::json!({
                    "retrieval_status": "not_found",
                    "task": null,
                    "message": NOT_FOUND_GUIDANCE.replace("{task_id}", &task_id),
                });
                yield Ok(ToolResult::Result {
                    result_for_assistant: map_result(&data),
                    data: Some(data),
                });
                return;
            };

            if !block {
                // Non-blocking: return current state.
                let data = if task.base().status != "running" && task.base().status != "pending" {
                    mark_notified(&task_id);
                    serde_json::json!({
                        "retrieval_status": "success",
                        "task": get_task_output_data(&task),
                    })
                } else {
                    serde_json::json!({
                        "retrieval_status": "not_ready",
                        "task": get_task_output_data(&task),
                    })
                };
                yield Ok(ToolResult::Result {
                    result_for_assistant: map_result(&data),
                    data: Some(data),
                });
                return;
            }

            // Blocking: wait for completion.
            let completed = wait_for_task_completion(&task_id, timeout, &context).await;
            let Some(completed) = completed else {
                let data = serde_json::json!({"retrieval_status": "timeout", "task": null});
                yield Ok(ToolResult::Result {
                    result_for_assistant: map_result(&data),
                    data: Some(data),
                });
                return;
            };
            if completed.base().status == "running" || completed.base().status == "pending" {
                let data = serde_json::json!({
                    "retrieval_status": "timeout",
                    "task": get_task_output_data(&completed),
                });
                yield Ok(ToolResult::Result {
                    result_for_assistant: map_result(&data),
                    data: Some(data),
                });
                return;
            }

            mark_notified(&task_id);
            let data = serde_json::json!({
                "retrieval_status": "success",
                "task": get_task_output_data(&completed),
            });
            yield Ok(ToolResult::Result {
                result_for_assistant: map_result(&data),
                data: Some(data),
            });
        })
    }
}

#[cfg(test)]
mod agent_tests {
    //! TaskOutput reads a sub-agent, not only a shell task.
    //!
    //! Two things have to be true for the parent to manage its agents: asking
    //! for an agent's output has to return the agent's ANSWER rather than the
    //! raw transcript, and the counts the parent would act on — context size,
    //! tool uses — have to reach the parent rather than a panel that does not
    //! exist here.
    use super::{get_task_output_data, map_result};
    use crate::state::app_state::{get_app_state, reset_app_state, test_lock};
    use crate::tasks::local_agent_task::{
        complete_agent_task, fail_agent_task, register_agent_foreground, update_agent_progress,
        update_agent_summary, AgentProgress,
    };
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::task::disk_output::get_task_output_path;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn fake_agent() -> AgentDefinition {
        AgentDefinition {
            agent_type: "Explore".into(),
            when_to_use: String::new(),
            get_system_prompt: Arc::new(|| String::new()),
            tools: None,
            disallowed_tools: Vec::new(),
            source: "built-in".into(),
        }
    }

    fn spawn() -> String {
        let id = create_agent_id(None);
        register_agent_foreground(
            &id, "look around", "find the thing", fake_agent(), None,
            Some(CancellationToken::new()), None,
        );
        id
    }

    fn data_for(id: &str) -> serde_json::Value {
        get_task_output_data(get_app_state().tasks.get(id).unwrap())
    }

    fn rendered(data: &serde_json::Value) -> String {
        map_result(&serde_json::json!({"retrieval_status": "success", "task": data}))
    }

    #[test]
    fn a_running_agent_reports_its_brief_counts_and_partial_output() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        let _ = std::fs::write(get_task_output_path(&id), b"noise from the transcript");
        update_agent_progress(
            &id,
            AgentProgress { tool_use_count: 3, token_count: 12345, ..Default::default() },
        );
        update_agent_summary(&id, "reading the lane config");

        let data = data_for(&id);
        assert_eq!(data["task_type"], "local_agent");
        assert_eq!(data["status"], "running");
        assert_eq!(data["prompt"], "find the thing");
        assert!(
            data["output"].as_str().unwrap().contains("noise"),
            "with no answer yet, the file is all there is"
        );
        assert_eq!(data["usage"]["total_tokens"], 12345);
        assert_eq!(data["usage"]["tool_uses"], 3);
        assert_eq!(data["summary"], "reading the lane config");

        let out = rendered(&data);
        assert!(out.contains("<total_tokens>12345</total_tokens>"),
                "the parent is actually told the context size: {out}");
        assert!(out.contains("<tool_uses>3</tool_uses>"));
        assert!(out.contains("<summary>reading the lane config</summary>"));
        reset_app_state();
    }

    #[test]
    fn a_finished_agent_answers_with_its_own_text() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        let _ = std::fs::write(get_task_output_path(&id), b"noise from the transcript");
        complete_agent_task(&id, serde_json::json!({"text": "the thing is in lane 3"}));

        let data = data_for(&id);
        assert_eq!(data["result"], "the thing is in lane 3");
        assert_eq!(data["output"], "the thing is in lane 3");
        assert_eq!(data["status"], "completed");
        let out = rendered(&data);
        assert!(out.contains("<output>\nthe thing is in lane 3\n</output>"), "{out}");
        assert!(!out.contains("noise"), "the transcript must not leak through");
        reset_app_state();
    }

    #[test]
    fn a_failed_agent_carries_its_error_to_the_parent() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        fail_agent_task(&id, "the lane refused");
        let data = data_for(&id);
        assert_eq!(data["error"], "the lane refused");
        assert_eq!(data["status"], "failed");
        assert!(rendered(&data).contains("<error>the lane refused</error>"));
        reset_app_state();
    }

    #[test]
    fn a_shell_task_grows_no_agent_fields() {
        let _guard = test_lock();
        reset_app_state();
        use crate::tasks::local_shell_task::{LocalShellTaskState, ShellResult};
        use crate::utils::task::framework::register_task;
        let mut base = crate::task::create_task_state_base("b12345678", "local_bash", "run tests", None);
        base.status = "completed".into();
        register_task(
            LocalShellTaskState {
                base,
                command: "pytest".into(),
                kind: "bash".into(),
                result: Some(ShellResult { code: 2, interrupted: false }),
                ..Default::default()
            }
            .into(),
        );
        let data = data_for("b12345678");
        assert_eq!(data["exitCode"], 2);
        assert!(data.get("prompt").is_none() && data.get("usage").is_none());
        reset_app_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[test]
    fn semantic_bool_variants() {
        assert!(semantic_bool(None, true));
        assert!(!semantic_bool(None, false));
        assert!(semantic_bool(Some(&Value::String("true".into())), false));
        assert!(!semantic_bool(Some(&Value::String("false".into())), true));
        assert!(semantic_bool(Some(&Value::Bool(true)), false));
    }

    #[test]
    fn xml_shape() {
        let data = serde_json::json!({
            "retrieval_status": "success",
            "task": {
                "task_id": "b1", "task_type": "local_bash", "status": "completed",
                "description": "d", "output": "line1\nline2", "exitCode": 0,
            },
        });
        let xml = map_result(&data);
        assert!(xml.starts_with("<retrieval_status>success</retrieval_status>"));
        assert!(xml.contains("<task_id>b1</task_id>"));
        assert!(xml.contains("<exit_code>0</exit_code>"));
        assert!(xml.contains("<output>\nline1\nline2\n</output>"));
    }

    #[tokio::test]
    async fn not_found_guidance() {
        let mut s = TaskOutputTool.call(
            serde_json::json!({"task_id": "zzz-none"}),
            ToolUseContext::default(),
            None,
        );
        let r = s.next().await.unwrap().unwrap();
        match r {
            ToolResult::Result { result_for_assistant, data } => {
                assert!(result_for_assistant
                    .starts_with("<retrieval_status>not_found</retrieval_status>"));
                assert!(result_for_assistant.contains("No task is registered with ID zzz-none"));
                assert_eq!(data.unwrap()["retrieval_status"], "not_found");
            }
            _ => panic!(),
        }
    }
}
