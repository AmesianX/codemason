//! TodoWrite Tool for CodeMason
//! Based on src/tools/TodoWriteTool/ from Claude Code
//! Equivalent to codemason/tools/todo_write_tool.py.
//!
//! Schema, description, prompt, storage semantics (appState.todos[agentId ??
//! sessionId], clear-on-all-done) and the result message are ported verbatim
//! from the original. The reactive Ink rendering is replaced by a terminal
//! render of the list — the terminal interface is the preserved local
//! substrate.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::ToolUseContext;
use crate::utils::formatting::{colorize, colors, plain_mode};
use crate::utils::todo_store::{get_todos, set_todos};

pub const TODO_WRITE_TOOL_NAME: &str = "TodoWrite";

pub const DESCRIPTION: &str = "Update the todo list for the current session. To be used proactively and often to track progress and pending tasks. Make sure that at least one task is in_progress at all times. Always provide both content (imperative) and activeForm (present continuous) for each task.";

pub const PROMPT: &str = include_str!("prompts/todo_write_prompt.txt");

pub const RESULT_MESSAGE: &str = "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable";

/// The exact input_json_schema() output dumped from the Python build
/// (schemas/tool_schemas.json) — embedded verbatim, never hand-derived.
static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(TODO_WRITE_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

/// agentId ?? sessionId — codemason runs one session per process.
fn todo_key(context: &ToolUseContext) -> String {
    crate::utils::todo_store::todo_key(context.agent_id.as_deref())
}

pub struct TodoWriteTool;

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &'static str {
        TODO_WRITE_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    async fn description(&self, _input: &Value) -> String {
        DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        PROMPT.to_string()
    }

    fn is_read_only(&self) -> bool {
        // touches session state only, no filesystem; checkPermissions=allow
        true
    }

    async fn is_enabled(&self) -> bool {
        // original: !isTodoV2Enabled(); TodoV2 off in the local build
        true
    }

    fn needs_permissions(&self) -> bool {
        false
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "TodoWrite".to_string()
    }

    /// The list as checkboxes (TodoWriteTool/UI.tsx), not the raw todos array.
    ///
    /// completed ✓ (dim) · in_progress ◐ (bold) · pending ☐.
    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let todos = input.get("todos").and_then(|t| t.as_array()).cloned().unwrap_or_default();
        let plain = plain_mode();
        let mut lines = Vec::new();
        for todo in &todos {
            let content = todo.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let status = todo.get("status").and_then(|s| s.as_str()).unwrap_or("pending");
            let content = content.split_whitespace().collect::<Vec<_>>().join(" ");
            let content = if content.chars().count() > 100 {
                format!("{}...", content.chars().take(100).collect::<String>())
            } else {
                content
            };
            match status {
                "completed" => {
                    let mark =
                        if plain { "[x]".to_string() } else { colorize("✓", colors::GREEN) };
                    lines.push(format!("{mark} {}", colorize(&content, colors::DIM)));
                }
                "in_progress" => {
                    let mark =
                        if plain { "[~]".to_string() } else { colorize("◐", colors::CYAN) };
                    lines.push(format!("{mark} {}", colorize(&content, colors::BOLD)));
                }
                _ => {
                    let mark = if plain { "[ ]" } else { "☐" };
                    lines.push(format!("{mark} {content}"));
                }
            }
        }
        lines.join("\n")
    }

    fn render_tool_result_message(&self, _data: &Value, _verbose: bool) -> Option<String> {
        // Tool.ts: "Omit for tools whose results are surfaced elsewhere (e.g.,
        // TodoWrite updates the todo panel, not the transcript)." The list
        // above IS the result; echoing "Todos have been modified…" is noise.
        Some(String::new())
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let todos: Vec<Value> =
                input.get("todos").and_then(|t| t.as_array()).cloned().unwrap_or_default();

            let key = todo_key(&context);
            let old_todos = get_todos(&key);
            let all_done = !todos.is_empty()
                && todos
                    .iter()
                    .all(|t| t.get("status").and_then(|s| s.as_str()) == Some("completed"));
            let new_todos = if all_done { Vec::new() } else { todos.clone() };
            set_todos(&key, new_todos);

            // No printing from here: the list is drawn by
            // render_tool_use_message (the display layer owns display). call()
            // printing straight to stdout also leaked the checkboxes into
            // headless `-p` output, which must stay machine-readable.

            // verificationNudgeNeeded (TodoWriteTool.ts:75-114) is deliberately
            // not ported: it's gated behind feature('VERIFICATION_AGENT') + a
            // GrowthBook flag (both OFF in external/3P builds) AND it instructs
            // the model to spawn subagent_type="verification", an agent
            // codemason does not have — emitting it would make the Agent tool
            // raise "Agent type 'verification' not found". The base result text
            // below is the verbatim external-build output.
            yield Ok(ToolResult::Result {
                result_for_assistant: RESULT_MESSAGE.to_string(),
                data: Some(serde_json::json!({
                    "oldTodos": old_todos,
                    "newTodos": todos,
                })),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_verbatim_dump() {
        assert_eq!(SCHEMA.get("title").and_then(|t| t.as_str()), Some("TodoWriteInput"));
        assert!(SCHEMA.get("$defs").and_then(|d| d.get("TodoItem")).is_some());
        assert_eq!(
            SCHEMA.get("additionalProperties").and_then(|a| a.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn checkbox_render_plain() {
        std::env::set_var("CODEMASON_PLAIN", "1");
        let input = serde_json::json!({"todos": [
            {"content": "Run tests", "status": "completed", "activeForm": "Running tests"},
            {"content": "Build", "status": "in_progress", "activeForm": "Building"},
            {"content": "Ship", "status": "pending", "activeForm": "Shipping"},
        ]});
        let out = TodoWriteTool.render_tool_use_message(&input, false);
        std::env::remove_var("CODEMASON_PLAIN");
        assert_eq!(out, "[x] Run tests\n[~] Build\n[ ] Ship");
    }

    #[tokio::test]
    async fn all_done_clears_store() {
        use futures::StreamExt;
        let ctx = ToolUseContext { agent_id: Some("test-agent-td".into()), ..Default::default() };
        let input = serde_json::json!({"todos": [
            {"content": "a", "status": "completed", "activeForm": "a-ing"},
        ]});
        let mut s = TodoWriteTool.call(input, ctx, None);
        let item = s.next().await.unwrap().unwrap();
        match item {
            ToolResult::Result { result_for_assistant, data } => {
                assert_eq!(result_for_assistant, RESULT_MESSAGE);
                // newTodos in the DATA is the full list (display), but the
                // STORE was cleared (all completed).
                assert_eq!(
                    data.unwrap()["newTodos"].as_array().unwrap().len(),
                    1
                );
                assert!(crate::utils::todo_store::get_todos("test-agent-td").is_empty());
            }
            _ => panic!("expected result"),
        }
    }
}
