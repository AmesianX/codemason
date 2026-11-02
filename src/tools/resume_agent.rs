//! Resume a stopped sub-agent — adapted from src/tools/AgentTool/resumeAgent.ts.
//! Equivalent to codemason/tools/resume_agent.py.
//!
//! An agent that was killed, or that failed, still has everything needed to
//! carry on: its conversation is on disk (see
//! tasks::local_agent_task::append_agent_transcript) and its task record says
//! what kind of agent it was. So a message addressed to a stopped agent does
//! not have to be refused — it can be the next thing that agent hears.
//!
//! What the source does and this does not: worktree revival, fork-agent resume
//! with the parent's exact system prompt, and content-replacement
//! reconstruction. The first two are features codemason does not have; the
//! third has no state to rebuild. The source also reads the agent type and
//! description from a metadata sidecar it writes per agent. There is no sidecar
//! here, and none is needed: this path is only reachable while the task record
//! is still registered — SendMessage has just looked it up — and the record
//! carries both.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::state::app_state::get_app_state;
use crate::tasks::local_agent_task::{get_agent_transcript, register_async_agent};
use crate::tools::base::CanUseToolFn;
use crate::types::query::{ContextOptions, ToolUseContext};
use crate::utils::messages::{
    filter_orphaned_thinking_only_messages, filter_unresolved_tool_uses,
    filter_whitespace_only_assistant_messages,
};

/// What a resume attempt reports back.
pub struct ResumeResult {
    pub resumed: bool,
    pub agent_id: String,
    pub message: String,
}

/// The agent's conversation, in a shape that can be sent again.
///
/// Three filters, in the source's order, each removing a different way an
/// interrupted conversation is malformed:
///
///   * `filter_unresolved_tool_uses` — the turn whose tool calls ALL went
///     unanswered. That is the tail a stop leaves behind, and an orphaned
///     tool_use is rejected outright by the API. Note this is the ALL rule, not
///     the ANY rule `filter_incomplete_tool_calls` uses; a turn that got some
///     results back carries real work and is kept.
///   * `filter_orphaned_thinking_only_messages` — reasoning with nothing to
///     merge into.
///   * `filter_whitespace_only_assistant_messages` — a turn that said nothing.
///
/// None when there is no transcript, or nothing survives the filters, which the
/// caller must read as "cannot resume" rather than "resume from empty".
pub fn load_resumable_messages(agent_id: &str) -> Option<Vec<Value>> {
    let messages = get_agent_transcript(agent_id)?;
    let messages = filter_unresolved_tool_uses(&messages);
    let messages = filter_orphaned_thinking_only_messages(&messages);
    let messages = filter_whitespace_only_assistant_messages(&messages);
    if messages.is_empty() {
        None
    } else {
        Some(messages)
    }
}

/// Restart a stopped agent from its transcript, in the background
/// (resumeAgentBackground).
///
/// Background because that is the only shape that makes sense here: the caller
/// is a SendMessage inside someone else's turn, and it must not block while the
/// agent works. The agent announces its own outcome through the notification
/// queue, exactly as a run_in_background spawn does.
///
/// Registered under the SAME agent id, which makes this a REPLACEMENT in the
/// task store rather than a new task — that is what register_task's
/// carry-forward of held state was written for, and it is why the agent keeps
/// its name in the SendMessage registry without re-registering it.
pub async fn resume_agent_background(
    agent_id: &str,
    prompt: &str,
    context: &ToolUseContext,
    can_use_tool: Option<CanUseToolFn>,
) -> ResumeResult {
    let refuse = |message: String| ResumeResult {
        resumed: false,
        agent_id: agent_id.to_string(),
        message,
    };

    let state = get_app_state();
    let Some(agent) = state.tasks.get(agent_id).and_then(|t| t.as_local_agent()) else {
        return refuse(format!("No agent task is registered with ID {agent_id}."));
    };
    if agent.base.status == "running" {
        return refuse(format!(
            "Agent {agent_id} is still running; nothing to resume."
        ));
    }
    let agent_type = agent.agent_type.clone();
    let description = if agent.base.description.is_empty() {
        "(resumed)".to_string()
    } else {
        agent.base.description.clone()
    };

    let Some(resumed_messages) = load_resumable_messages(agent_id) else {
        return refuse(format!(
            "Agent {agent_id} has no transcript to resume from. Launch a new agent \
             with the full task instead."
        ));
    };
    let Some(can_use_tool) = can_use_tool else {
        return refuse("Resuming an agent requires a permission callback.".to_string());
    };

    // The agent type the task recorded. An unknown one is an error rather than
    // a silent fall-back to general-purpose, so it is resolved defensively: a
    // resume must not fail because a custom agent definition has since been
    // removed.
    let agent_def = crate::tools::built_in_agents::get_agent_definition(Some(&agent_type))
        .or_else(|_| crate::tools::built_in_agents::get_agent_definition(None));
    let Ok(agent_def) = agent_def else {
        return refuse("No agent definition is available to resume with.".to_string());
    };

    let available_tools = crate::tools::get_agent_tools().await;
    let tools = crate::tools::built_in_agents::resolve_agent_tools(&agent_def, &available_tools);
    let model = crate::utils::config::get_config().model_name;
    let mut agent_prompt: Vec<String> = vec![(agent_def.get_system_prompt)()];
    agent_prompt.extend(crate::constants::prompts::agent_prefix_sections(&model));

    // Its own scope again, and unlinked — this is a background relaunch, and
    // register_async_agent's whole point is that ESC on the main thread cancels
    // the turn without killing an agent the user put in the background. Under
    // the caller's token it would have died with the very turn that resumed it.
    // Stopping it stays explicit: TaskStop cancels this token.
    let agent_abort = tokio_util::sync::CancellationToken::new();
    let mut sub_ctx = ToolUseContext {
        abort_controller: agent_abort.clone(),
        options: ContextOptions {
            tools: tools.clone(),
            dangerously_skip_permissions: context.options.dangerously_skip_permissions,
            slow_and_capable_model: context.options.slow_and_capable_model.clone(),
        },
        on_token_update: context.on_token_update.clone(),
        on_thinking_update: context.on_thinking_update.clone(),
        input_router: context.input_router.clone(),
        ..Default::default()
    };
    sub_ctx.agent_id = Some(agent_id.to_string());

    register_async_agent(
        agent_id,
        &description,
        prompt,
        agent_def,
        context.tool_use_id.clone(),
        Some(agent_abort.clone()),
    );

    // The transcript, then what the caller wants done next — the source's
    // `[...resumedMessages, createUserMessage({ content: prompt })]`.
    let resumed_typed: Vec<crate::types::query::Message> = resumed_messages
        .iter()
        .filter_map(|v| serde_json::from_value(v.clone()).ok())
        .collect();
    let mut messages = resumed_typed.clone();
    messages.push(crate::types::query::Message::User(
        crate::utils::messages::create_user_message_text(prompt),
    ));

    let count = resumed_messages.len();
    let detached_id = agent_id.to_string();
    let detached_description = description.clone();
    let detached_tool_use_id = context.tool_use_id.clone();
    let detached_abort = agent_abort.clone();
    let detached_tools = tools.clone();
    tokio::spawn(async move {
        // The slot is taken in here, not in the call: waiting for it there
        // would make a resume block the turn that asked for it.
        let mut slot = crate::utils::subagent_gate::try_acquire();
        if slot.is_none() {
            slot = crate::utils::subagent_gate::acquire(&detached_abort).await;
            if slot.is_none() {
                crate::tasks::local_agent_task::kill_async_agent(&detached_id);
                crate::tasks::local_agent_task::enqueue_agent_notification(
                    &detached_id,
                    &detached_description,
                    "killed",
                    None,
                    None,
                    None,
                    detached_tool_use_id.as_deref(),
                    None,
                    None,
                );
                return;
            }
        }
        let stream = crate::query::query(crate::query::QueryParams {
            messages,
            system_prompt: agent_prompt,
            context: BTreeMap::new(),
            can_use_tool,
            tool_use_context: sub_ctx,
            max_output_recovery_count: 0,
            repetition_recovery_count: 0,
            suppress_reasoning: false,
            tool_parse_recovery_count: 0,
            empty_turn_recovery_count: 0,
            stop_hook_active: false,
            turn_count: 0,
        });
        crate::tools::agent_tool_utils::run_async_agent_lifecycle(
            crate::tools::agent_tool_utils::AsyncAgentLifecycle {
                agent_id: detached_id,
                description: detached_description,
                stream,
                abort: Some(detached_abort),
                tool_use_id: detached_tool_use_id,
                started: std::time::Instant::now(),
                slot,
                tools: detached_tools,
                // The transcript is what it continues from, and it is already in
                // `messages`; the tally is seeded from it so the counts do not
                // restart.
                initial_messages: resumed_typed,
            },
        )
        .await;
    });

    ResumeResult {
        resumed: true,
        agent_id: agent_id.to_string(),
        message: format!(
            "Agent \"{description}\" ({agent_id}) was stopped and has been resumed in \
             the background from its transcript ({count} messages). You will be \
             notified when it completes."
        ),
    }
}

#[cfg(test)]
mod tests {
    //! A stopped sub-agent can be told one more thing instead of being replaced.
    //!
    //! The three filters are the interesting part. An interrupted conversation
    //! is malformed in three different ways, and each filter removes one;
    //! getting the first one's rule wrong (ANY vs ALL unresolved tool calls)
    //! would silently drop a turn that had done real work.
    use super::*;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::tasks::local_agent_task::{
        append_agent_transcript, kill_async_agent, register_async_agent,
    };
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::task::disk_output::{cleanup_task_output, get_task_output_path};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn fake_agent() -> AgentDefinition {
        AgentDefinition {
            agent_type: "Explore".into(),
            when_to_use: String::new(),
            get_system_prompt: Arc::new(String::new),
            tools: None,
            disallowed_tools: Vec::new(),
            source: "built-in".into(),
        }
    }

    /// A unique uuid per call: the transcript reader dedups by uuid, so two
    /// entries sharing one would silently collapse into a single turn — which
    /// is exactly what an earlier version of this test did to itself.
    fn next_uuid() -> String {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        format!("u-{}", N.fetch_add(1, Ordering::Relaxed))
    }

    fn assistant(text: &str, tools: &[&str], base: usize) -> Value {
        let mut content: Vec<Value> = vec![serde_json::json!({"type": "text", "text": text})];
        for (i, name) in tools.iter().enumerate() {
            content.push(serde_json::json!({
                "type": "tool_use", "id": format!("tu_{}", base + i), "name": name, "input": {}
            }));
        }
        serde_json::json!({
            "type": "assistant",
            "uuid": next_uuid(),
            "message": {"role": "assistant", "content": content},
        })
    }

    fn tool_result(id: &str) -> Value {
        serde_json::json!({
            "type": "user",
            "uuid": next_uuid(),
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": id, "content": "ok"}
            ]},
        })
    }

    #[test]
    fn the_filters_remove_three_different_kinds_of_malformed_turn() {
        // ALL unresolved, not ANY: a turn that got some results back is kept.
        let partly = vec![assistant("two calls", &["Read", "Grep"], 0), tool_result("tu_0")];
        assert_eq!(
            filter_unresolved_tool_uses(&partly).len(),
            2,
            "a turn with SOME results back survives"
        );
        let wholly = vec![
            assistant("one call", &["Read"], 0),
            tool_result("tu_0"),
            assistant("reaching", &["Bash"], 5),
        ];
        assert_eq!(
            filter_unresolved_tool_uses(&wholly).len(),
            2,
            "the turn whose calls ALL went unanswered is dropped"
        );

        let blank = vec![assistant("   ", &[], 0), assistant("real", &[], 0)];
        assert_eq!(filter_whitespace_only_assistant_messages(&blank).len(), 1);

        let orphan = vec![serde_json::json!({
            "type": "assistant", "uuid": "u1",
            "message": {"id": "m1", "role": "assistant",
                        "content": [{"type": "thinking", "thinking": "hmm"}]},
        })];
        assert!(
            filter_orphaned_thinking_only_messages(&orphan).is_empty(),
            "thinking with nothing to merge into is dropped"
        );
        let mut paired = orphan.clone();
        paired.push(serde_json::json!({
            "type": "assistant", "uuid": "u2",
            "message": {"id": "m1", "role": "assistant",
                        "content": [{"type": "text", "text": "and here it is"}]},
        }));
        assert_eq!(
            filter_orphaned_thinking_only_messages(&paired).len(),
            2,
            "thinking that shares an id with real content is kept"
        );
    }

    /// The chain has to survive the stop.
    ///
    /// A terminal transition drops the in-memory chain, and the resume then
    /// appends to the SAME file. Writing a second root there made everything
    /// before the resume unreachable from the newest leaf — the reader walks
    /// parentUuid back from it — so a resumed agent was handed one message
    /// instead of its whole history, and a second resume lost the first
    /// resume's work too. Caught end to end on the Python line: a real run
    /// reported "(1 messages)" for an agent that had read two files, and the
    /// resumed agent hallucinated a third file it had never seen.
    #[test]
    fn the_transcript_chain_survives_a_stop_and_a_resume() {
        let _guard = test_lock();
        reset_app_state();
        let id = create_agent_id(None);
        cleanup_task_output(&id);
        register_async_agent(&id, "d", "p", fake_agent(), None, Some(CancellationToken::new()));

        for m in [
            assistant("step one", &["Read"], 0),
            tool_result("tu_0"),
            assistant("done one", &[], 0),
        ] {
            append_agent_transcript(&id, &m);
        }
        assert_eq!(
            load_resumable_messages(&id).map(|m| m.len()),
            Some(3),
            "three turns before the stop"
        );

        kill_async_agent(&id);
        register_async_agent(&id, "d", "next", fake_agent(), None, Some(CancellationToken::new()));
        append_agent_transcript(&id, &assistant("step two", &[], 0));

        let path = get_task_output_path(&id);
        let text = std::fs::read_to_string(&path).expect("the transcript is readable");
        let roots = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|e| e.get("parentUuid").map(|v| v.is_null()).unwrap_or(true))
            .count();
        assert_eq!(roots, 1, "ONE root, not one per resume");
        assert_eq!(
            load_resumable_messages(&id).map(|m| m.len()),
            Some(4),
            "the whole history is still reachable"
        );

        kill_async_agent(&id);
        register_async_agent(&id, "d", "again", fake_agent(), None, Some(CancellationToken::new()));
        append_agent_transcript(&id, &assistant("step three", &[], 0));
        assert_eq!(
            load_resumable_messages(&id).map(|m| m.len()),
            Some(5),
            "a SECOND resume keeps the first resume's work"
        );

        cleanup_task_output(&id);
        reset_app_state();
    }

    /// The refusals that must stay refusals.
    #[tokio::test]
    async fn a_running_or_unknown_agent_is_not_resumed() {
        let _guard = test_lock();
        reset_app_state();
        let ctx = ToolUseContext::default();

        let out = resume_agent_background(&format!("a{}", "0".repeat(16)), "x", &ctx, None).await;
        assert!(!out.resumed);
        assert!(out.message.contains("No agent task"), "{}", out.message);

        let id = create_agent_id(None);
        register_async_agent(&id, "busy", "p", fake_agent(), None, Some(CancellationToken::new()));
        let out = resume_agent_background(&id, "x", &ctx, None).await;
        assert!(!out.resumed);
        assert!(
            out.message.contains("still running"),
            "a RUNNING agent has nothing to resume: {}",
            out.message
        );
        reset_app_state();
    }
}
