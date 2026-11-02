//! TaskStop tool — based on src/tools/TaskStopTool/ from Claude Code.
//! Equivalent to codemason/tools/task_stop_tool.py.
//!
//! Stops a running background task by id. `shell_id` is accepted as a
//! deprecated alias for `task_id` (KillShell backward compatibility).
//! Delegates to stop_task, which kills the task via its task impl and
//! suppresses the kill notification.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::state::app_state::get_app_state;
use crate::tasks::stop_task::stop_task;
use crate::tools::base::{
    CanUseToolFn, PermissionResult, Tool, ToolError, ToolResult, ToolStream,
};
use crate::types::query::{ToolUseContext, ValidationResult};

pub const TASK_STOP_TOOL_NAME: &str = "TaskStop";

pub const DESCRIPTION: &str = "\n- Stops a running background task by its ID\n- Takes a task_id parameter identifying the task to stop\n- Returns a success or failure status\n- Use this tool when you need to terminate a long-running task\n";

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(TASK_STOP_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

fn requested_id(input: &Value) -> Option<String> {
    // Support both task_id and shell_id (deprecated KillShell compat).
    let task_id = input.get("task_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    let shell_id = input.get("shell_id").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
    task_id.or(shell_id).map(|s| s.to_string())
}

pub struct TaskStopTool;

#[async_trait]
impl Tool for TaskStopTool {
    fn name(&self) -> &'static str {
        TASK_STOP_TOOL_NAME
    }

    /// KillShell is the deprecated name — kept as an alias for backward
    /// compatibility with existing transcripts and SDK users.
    fn aliases(&self) -> &'static [&'static str] {
        &["KillShell"]
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        // Stopping a task is safe to run alongside other concurrency-safe
        // calls even though it isn't read-only (TaskStopTool.isConcurrencySafe
        // → true).
        true
    }

    async fn description(&self, _input: &Value) -> String {
        "Stop a running background task by ID".to_string()
    }

    async fn prompt(&self) -> String {
        DESCRIPTION.to_string()
    }

    /// TaskStop has no `DESCRIPTION` in the Python, so the wire description
    /// resolves through TOOL_DESCRIPTIONS / the simple blurb rather than this
    /// tool's (much longer) system prompt. Returning None selects that chain.
    async fn wire_description(&self) -> Option<String> {
        None
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Stop Task".to_string()
    }

    fn render_tool_use_message(&self, _input: &Value, _verbose: bool) -> String {
        // TaskStopTool/UI.tsx renderToolUseMessage returns '': the stopped
        // task is named in the result, so the call itself shows nothing.
        String::new()
    }

    /// The human line, not the JSON the model gets (jsonStringify in call()).
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        if !data.is_object() {
            return None;
        }
        if let Some(message) = data.get("message").and_then(|m| m.as_str()) {
            if !message.is_empty() {
                return Some(message.to_string());
            }
        }
        data.get("task_id")
            .and_then(|t| t.as_str())
            .map(|task_id| format!("Stopped task: {task_id}"))
    }

    async fn check_permissions(&self, _input: &Value, _context: &ToolUseContext) -> PermissionResult {
        // The original's default checkPermissions auto-allows (no classifier
        // gate for stopping a task the model itself backgrounded). Mirror
        // that: stopping a self-owned background task needs no user prompt.
        PermissionResult::allow()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let Some(id) = requested_id(input) else {
            let mut v = ValidationResult::failure("Missing required parameter: task_id");
            v.meta = Some(serde_json::json!({"errorCode": 1}));
            return v;
        };

        let state = get_app_state();
        let Some(task) = state.tasks.get(&id) else {
            let mut v = ValidationResult::failure(format!("No task found with ID: {id}"));
            v.meta = Some(serde_json::json!({"errorCode": 1}));
            return v;
        };
        if task.base().status != "running" {
            let mut v = ValidationResult::failure(format!(
                "Task {id} is not running (status: {})",
                task.base().status
            ));
            v.meta = Some(serde_json::json!({"errorCode": 3}));
            return v;
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let Some(id) = requested_id(&input) else {
                // Python raises ValueError here (validate_input already
                // guards); surface as a tool error the dispatcher formats.
                yield Err(ToolError::new("Missing required parameter: task_id"));
                return;
            };

            match stop_task(&id).await {
                Ok(result) => {
                    let command_text = result.command.clone().unwrap_or_default();
                    let data = serde_json::json!({
                        "message": format!(
                            "Successfully stopped task: {} ({})",
                            result.task_id, command_text
                        ),
                        "task_id": result.task_id,
                        "task_type": result.task_type,
                        "command": result.command,
                    });
                    // mapToolResultToToolResultBlockParam: jsonStringify(output).
                    yield Ok(ToolResult::Result {
                        result_for_assistant:
                            crate::services::openai_common::py_json_dumps(&data),
                        data: Some(data),
                    });
                }
                Err(e) => {
                    // Python: StopTaskError propagates out of call() and the
                    // query dispatcher formats it as an error tool_result.
                    yield Err(ToolError::new(e.message));
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_matches() {
        assert!(TaskStopTool.matches_name("TaskStop"));
        assert!(TaskStopTool.matches_name("KillShell"));
        assert!(!TaskStopTool.matches_name("Bash"));
    }

    #[tokio::test]
    async fn missing_id_fails_validation() {
        let v = TaskStopTool
            .validate_input(&serde_json::json!({}), &ToolUseContext::default())
            .await;
        assert!(!v.result);
        assert_eq!(v.message, "Missing required parameter: task_id");
    }

    #[test]
    fn shell_id_alias_read() {
        assert_eq!(
            requested_id(&serde_json::json!({"shell_id": "b123"})),
            Some("b123".to_string())
        );
        assert_eq!(
            requested_id(&serde_json::json!({"task_id": "a1", "shell_id": "b2"})),
            Some("a1".to_string())
        );
    }
}
