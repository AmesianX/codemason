//! SendMessage tool — based on src/tools/SendMessageTool/ from Claude Code.
//! Equivalent to codemason/tools/send_message_tool.py.
//!
//! Talk to a sub-agent that is already running. Its plain output is not visible
//! to another agent and the parent's is not visible to it, so a message between
//! them has to be an explicit tool call.
//!
//! Scope. The source's SendMessage is mostly a swarm tool: teammates by name,
//! `"*"` broadcast, shutdown and plan-approval protocol replies, and
//! cross-session delivery over a unix socket or the Remote Control bridge.
//! codemason has none of those subsystems, so none of that is reproduced. What
//! IS ported is the branch that routes to a local sub-agent, which is the one
//! the Agent tool's own description points at. That is also why `is_enabled` is
//! unconditional here while the source gates the whole tool behind its swarm
//! feature: the gate exists because teammates live there.
//!
//! Delivery is a queue, not a handoff. The agent is mid-turn and the list of
//! messages it is working from belongs to the loop running it, so the message
//! waits on the agent's own task until that loop reaches its next tool-round
//! boundary (the drain in query.rs). Nothing else can read it: the queue is
//! per-task, so the only reader is the agent it was addressed to.

use async_trait::async_trait;
use futures::stream;
use serde_json::{json, Value};

use crate::state::app_state::get_app_state;
use crate::tasks::local_agent_task::{is_panel_agent_task, queue_pending_message};
use crate::tools::base::{
    CanUseToolFn, PermissionResult, Tool, ToolError, ToolResult, ToolStream, ValidationResult,
};
use crate::types::query::ToolUseContext;
use crate::utils::agent_id::is_agent_id;

pub const SEND_MESSAGE_TOOL_NAME: &str = "SendMessage";

const DESCRIPTION: &str = r#"
# SendMessage

Send a message to a running sub-agent.

```json
{"to": "a1b2c3d4e5f60718", "summary": "narrow the search", "message": "skip the vendor directory"}
```

`to` is the name you gave the agent when you launched it, or its ID as
returned by the launch when you did not name it.

An agent that was stopped is resumed from its own transcript and told your
message, so you do not have to re-describe a task it had already started.

Your plain text output is NOT visible to the agent — to tell it something, you
MUST call this tool. The message is delivered at the agent's next tool round;
it is not an interrupt, and the agent keeps working in the meantime.
"#;

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(limit).collect::<String>())
    }
}

pub struct SendMessageTool;

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &'static str {
        SEND_MESSAGE_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        json!({
            "type": "object",
            "description": "Input schema for SendMessage tool",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient: the name or ID of a running sub-agent"
                },
                "summary": {
                    "type": "string",
                    "description": "A 5-10 word summary shown as a preview in the UI"
                },
                "message": {
                    "type": "string",
                    "description": "Plain text message content"
                }
            },
            "required": ["to", "message"],
            "additionalProperties": false
        })
    }

    async fn description(&self, _input: &Value) -> String {
        "Send a message to another agent".to_string()
    }

    async fn prompt(&self) -> String {
        DESCRIPTION.trim().to_string()
    }

    fn is_read_only(&self) -> bool {
        // SendMessageTool.isReadOnly(input) is `typeof input.message === 'string'`
        // — true for a plain message, false for the structured protocol messages
        // (shutdown / plan approval) that belong to the swarm. Only plain
        // messages exist here, so it is simply true.
        true
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        self.is_read_only()
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        false
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "SendMessage".to_string()
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let to = input.get("to").and_then(|v| v.as_str()).unwrap_or("");
        let preview = input
            .get("summary")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                truncate(input.get("message").and_then(|v| v.as_str()).unwrap_or(""), 50)
            });
        if preview.is_empty() {
            to.to_string()
        } else {
            format!("{preview} → {to}")
        }
    }

    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        data.get("message")
            .and_then(|m| m.as_str())
            .map(|m| m.to_string())
    }

    async fn check_permissions(&self, _input: &Value, _context: &ToolUseContext) -> PermissionResult {
        PermissionResult::allow()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let to = input.get("to").and_then(|v| v.as_str()).unwrap_or("").trim();
        if to.is_empty() {
            return ValidationResult::failure("Missing required parameter: to".to_string());
        }
        if input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .is_empty()
        {
            return ValidationResult::failure("Missing required parameter: message".to_string());
        }
        if !get_app_state().agent_name_registry.contains_key(to) && is_agent_id(to).is_none() {
            return ValidationResult::failure(format!(
                "\"{to}\" is neither a name you gave an agent nor an agent ID. Pass \
                 the name you spawned it under, or the ID the Agent tool returned."
            ));
        }
        ValidationResult::success()
    }

    fn call(&self, input: Value, context: ToolUseContext, can_use_tool: Option<CanUseToolFn>) -> ToolStream {
        let to = input
            .get("to")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // A name the spawn registered wins over reading `to` as a raw id, which
        // is the source's `registered ?? toAgentId(input.to)`. The model named
        // the agent so it would not have to carry a 16-hex id around; making it
        // pass the id anyway would waste the name it was offered.
        let state = get_app_state();
        let resolved = state
            .agent_name_registry
            .get(&to)
            .map(|s| s.as_str())
            .or_else(|| is_agent_id(&to));

        // Two shapes of answer: one this call already knows, and one that has
        // to await a resume. The resume cannot happen here — building `data` is
        // synchronous — so the decision is made now and carried out in the
        // stream body below.
        enum Plan {
            Ready(Value),
            Resume(String),
        }

        let plan = match resolved {
            Some(agent_id) => {
                match state
                    .tasks
                    .get(agent_id)
                    .and_then(|t| t.as_local_agent())
                    .filter(|t| is_panel_agent_task(t))
                {
                    Some(agent) if agent.base.status == "running" => {
                        queue_pending_message(agent_id, &message);
                        Plan::Ready(json!({
                            "success": true,
                            "message": format!(
                                "Message queued for delivery to {to} at its next tool round."
                            ),
                        }))
                    }
                    // Stopped, but not gone: its conversation is on disk and
                    // its task record says what kind of agent it was, so the
                    // message can be the next thing it hears
                    // (resumeAgentBackground). In the background, because this
                    // call sits inside someone else's turn and must not block
                    // it — the agent announces its own outcome through the
                    // notification queue.
                    Some(_) => Plan::Resume(agent_id.to_string()),
                    None => Plan::Ready(json!({
                        "success": false,
                        "message": format!(
                            "No running agent called \"{to}\". Check the name you spawned \
                             it under or the ID the Agent tool returned, or launch a \
                             new agent."
                        ),
                    })),
                }
            }
            None => Plan::Ready(json!({
                "success": false,
                "message": format!(
                    "No running agent called \"{to}\". Check the name you spawned it \
                     under or the ID the Agent tool returned, or launch a new agent."
                ),
            })),
        };

        Box::pin(stream::once(async move {
            let data = match plan {
                Plan::Ready(value) => value,
                Plan::Resume(agent_id) => {
                    let outcome = crate::tools::resume_agent::resume_agent_background(
                        &agent_id, &message, &context, can_use_tool,
                    )
                    .await;
                    json!({"success": outcome.resumed, "message": outcome.message})
                }
            };
            Ok(ToolResult::Result {
                result_for_assistant: serde_json::to_string(&data).unwrap_or_default(),
                data: Some(data),
            })
        }))
    }
}

/// The registered instance (tools/mod.rs ALL_TOOLS).
pub fn tool() -> std::sync::Arc<dyn Tool> {
    std::sync::Arc::new(SendMessageTool)
}

#[cfg(test)]
mod tests {
    //! A message to a running sub-agent reaches it, and reaches only it.
    //!
    //! The queue is the whole mechanism: the agent is mid-turn and the message
    //! list it is working from belongs to the loop running it, so a message
    //! waits on the agent's own task until that loop hits its next tool-round
    //! boundary. Being per-task is also what makes it safe — unlike the
    //! process-global notification queue, there is no way for one agent to
    //! drain another's mail.
    use super::*;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::tasks::local_agent_task::{
        append_message_to_local_agent, complete_agent_task, drain_pending_messages,
        register_agent_foreground,
    };
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::messages::wrap_command_text;
    use crate::utils::task::framework::update_task_state;
    use futures::StreamExt;
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
            &id, "look around", "find it", fake_agent(), None,
            Some(CancellationToken::new()), None,
        );
        id
    }

    async fn send(to: &str, message: &str) -> Value {
        let input = json!({"to": to, "message": message});
        let context = ToolUseContext::default();
        let validation = SendMessageTool.validate_input(&input, &context).await;
        if !validation.result {
            return json!({"invalid": validation.message});
        }
        let mut stream = SendMessageTool.call(input, context, None);
        let mut last = Value::Null;
        while let Some(Ok(ToolResult::Result { data, .. })) = stream.next().await {
            last = data.unwrap_or(Value::Null);
        }
        last
    }

    fn queue_of(id: &str) -> Vec<String> {
        get_app_state()
            .tasks
            .get(id)
            .and_then(|t| t.as_local_agent())
            .map(|a| a.pending_messages.clone())
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn a_message_queues_on_that_agent_and_drains_in_order() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();

        let data = send(&id, "skip the vendor directory").await;
        assert_eq!(data["success"], true, "{data}");
        assert!(
            data["message"].as_str().unwrap().contains("at its next tool round"),
            "{data}"
        );
        assert_eq!(queue_of(&id), vec!["skip the vendor directory".to_string()]);

        send(&id, "and the build directory").await;
        assert_eq!(
            queue_of(&id),
            vec![
                "skip the vendor directory".to_string(),
                "and the build directory".to_string()
            ],
            "a second message queues behind the first, in order"
        );

        let drained = drain_pending_messages(&id);
        assert_eq!(drained.len(), 2);
        assert!(queue_of(&id).is_empty(), "the drain leaves the queue empty");
        assert!(drain_pending_messages(&id).is_empty(), "draining again takes nothing");
        reset_app_state();
    }

    #[tokio::test]
    async fn one_agent_cannot_read_anothers_mail() {
        let _guard = test_lock();
        reset_app_state();
        let a = spawn();
        let b = spawn();
        send(&a, "for a only").await;
        assert_eq!(queue_of(&a), vec!["for a only".to_string()]);
        assert!(queue_of(&b).is_empty());
        assert!(drain_pending_messages(&b).is_empty(), "draining b takes nothing");
        assert_eq!(queue_of(&a), vec!["for a only".to_string()], "a's message is untouched");
        reset_app_state();
    }

    #[test]
    fn the_framing_is_the_instruction() {
        let framed = wrap_command_text("skip the vendor directory", Some("coordinator"));
        assert!(framed.starts_with("The coordinator sent a message while you were working:"));
        assert!(framed.ends_with("Address this before completing your current task."));
    }

    #[tokio::test]
    async fn what_cannot_be_sent() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        complete_agent_task(&id, json!({"text": "done"}));
        let data = send(&id, "one more thing").await;
        assert_eq!(data["success"], false);
        let said = data["message"].as_str().unwrap();
        assert!(
            said.contains("no transcript to resume from"),
            "told why, not silently dropped: {data}"
        );

        let data = send(&format!("a{}", "0".repeat(16)), "hello?").await;
        assert_eq!(data["success"], false, "an unknown id fails");

        let data = send("researcher", "hello?").await;
        assert!(
            data["invalid"].as_str().unwrap_or("").contains("neither a name"),
            "a name nobody was spawned under is rejected: {data}"
        );

        let data = send("", "hello?").await;
        assert!(!data["invalid"].is_null(), "an empty recipient is rejected");
        reset_app_state();
    }

    /// The whole point of naming an agent: the model should not have to carry a
    /// 16-hex id out of a tool result and quote it back correctly.
    #[tokio::test]
    async fn a_registered_name_reaches_the_agent_it_names() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        let named = id.clone();
        crate::state::app_state::set_app_state(move |prev| {
            let mut next = prev.clone();
            next.agent_name_registry
                .insert("researcher".into(), named.clone());
            next
        });

        let data = send("researcher", "skip the vendor directory").await;
        assert_eq!(data["success"], true, "{data}");
        assert_eq!(
            queue_of(&id),
            vec!["skip the vendor directory".to_string()],
            "the message lands on the agent, not on the name"
        );

        // The name only ADDS a route; the raw id keeps working.
        let data = send(&id, "and the build dir").await;
        assert_eq!(data["success"], true, "{data}");
        assert_eq!(
            queue_of(&id),
            vec![
                "skip the vendor directory".to_string(),
                "and the build dir".to_string()
            ],
            "both queued, in order"
        );

        // An unregistered name must not fall through onto some other agent.
        let data = send("reviewer", "hello?").await;
        assert!(!data["invalid"].is_null(), "{data}");
        reset_app_state();
    }

    #[tokio::test]
    async fn the_main_session_is_not_messageable() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        update_task_state(&id, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.agent_type = "main-session".into();
            Some(next.into())
        });
        let data = send(&id, "hello?").await;
        assert_eq!(data["success"], false, "{data}");
        reset_app_state();
    }

    #[test]
    fn the_display_list_is_separate_from_the_queue() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        crate::tasks::local_agent_task::queue_pending_message(&id, "routed to the agent");
        append_message_to_local_agent(&id, json!({"type": "user", "text": "shown to the user"}));
        let state = get_app_state();
        let agent = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(agent.pending_messages, vec!["routed to the agent".to_string()]);
        assert_eq!(
            agent.messages.as_ref().unwrap(),
            &vec![json!({"type": "user", "text": "shown to the user"})]
        );
        reset_app_state();
    }

    /// A sub-agent is not given this tool — it would let one steer a sibling,
    /// and neither of them is the one the user is talking to.
    #[test]
    fn a_sub_agent_is_not_given_send_message() {
        assert!(crate::tools::built_in_agents::all_agent_disallowed_tools()
            .contains(SEND_MESSAGE_TOOL_NAME));
    }

    /// Where the drain is wired. The drain itself runs inside query()'s turn
    /// loop, which these tests do not drive; its behaviour is covered above.
    #[test]
    fn the_drain_is_wired_at_the_turn_boundary_behind_the_right_gate() {
        let src = include_str!("../query.rs");
        assert!(src.contains("drain_pending_messages(agent_id)"));
        assert!(src.contains("wrap_command_text(&queued, Some(\"coordinator\"))"));
        assert!(src.contains("wrap_messages_in_system_reminder(vec![sent])"));
        assert!(
            src.contains("if let Some(agent_id) = tool_use_context.agent_id.as_deref()"),
            "only a sub-agent drains its own queue"
        );
        assert!(
            src.find("drain_pending_notifications()").unwrap()
                < src.find("drain_pending_messages(agent_id)").unwrap(),
            "the notification drain stays where it was, main-thread only"
        );
    }
}
