//! Agent tool helpers — adapted from src/tools/AgentTool/agentToolUtils.ts.
//! Equivalent to codemason/tools/agent_tool_utils.py.
//!
//! Three things the foreground and background paths both need: pulling the
//! agent's answer out of its messages, dropping a tool call that never came
//! back before the conversation is sent again, and (for the background one)
//! driving the whole run from spawn to the notification that carries the result
//! back.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::tasks::local_agent_task::{
    complete_agent_task, create_progress_tracker, enqueue_agent_notification, fail_agent_task,
    get_progress_update, get_token_count_from_tracker, kill_async_agent, update_agent_progress,
    update_progress_from_message, AgentUsage,
};
use crate::types::messages::{ContentBlock, ContentBlockParam, MessageContent};
use crate::types::query::Message;
use crate::utils::task::framework::update_task_state;

/// The text blocks of one assistant message, joined (extractTextContent).
fn text_of(message: &Message) -> String {
    let Message::Assistant(assistant) = message else {
        return String::new();
    };
    assistant
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// The agent's answer: the last assistant message that HAS text
/// (extractPartialResult).
///
/// Walks BACKWARD, and that is the point — the loop can end on an assistant
/// turn that is nothing but tool_use blocks, and taking the last message
/// outright would report an empty answer for a run that produced one. Also what
/// makes a killed agent still able to hand back whatever it had reached.
pub fn extract_partial_result(messages: &[Message]) -> Option<String> {
    for message in messages.iter().rev() {
        let text = text_of(message);
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

/// Drop assistant messages holding a tool_use with no tool_result
/// (filterIncompleteToolCalls).
///
/// An orphaned tool call is an API error, not a wasted token: a conversation
/// that ends on an assistant turn whose tool never returned is rejected
/// outright. That is the exact shape a foreground agent is in at the instant it
/// is backgrounded — the race can fire while a tool is mid-flight — so the
/// resumed run has to be handed the conversation without it.
///
/// Collects every tool_use_id that HAS a result first, then keeps an assistant
/// message only if none of its tool_use blocks is missing from that set. The
/// whole message goes, text blocks and all; a partial assistant turn is not a
/// thing the wire format can express.
pub fn filter_incomplete_tool_calls(messages: &[Message]) -> Vec<Message> {
    let mut satisfied: std::collections::HashSet<String> = std::collections::HashSet::new();
    for message in messages {
        if let Message::User(user) = message {
            if let MessageContent::Blocks(blocks) = &user.message.content {
                for block in blocks {
                    if let ContentBlockParam::ToolResult { tool_use_id, .. } = block {
                        satisfied.insert(tool_use_id.clone());
                    }
                }
            }
        }
    }

    messages
        .iter()
        .filter(|message| match message {
            Message::Assistant(assistant) => !assistant.message.content.iter().any(|block| {
                matches!(block, ContentBlock::ToolUse(tu) if !satisfied.contains(&tu.id))
            }),
            _ => true,
        })
        .cloned()
        .collect()
}

/// What the detached driver needs to run an agent to its end.
pub struct AsyncAgentLifecycle {
    pub agent_id: String,
    pub description: String,
    pub stream: Pin<Box<dyn Stream<Item = Message> + Send>>,
    pub abort: Option<CancellationToken>,
    pub tool_use_id: Option<String>,
    pub started: std::time::Instant,
    /// Taken here rather than in the tool call: the concurrency gate is ours,
    /// not the source's, and waiting for a slot in the call would make a
    /// background launch block the parent — the one thing it must not do.
    ///
    /// On a hand-over from the foreground it is a slot the caller was ALREADY
    /// holding, moved rather than released and re-taken: a backgrounded agent
    /// is still running and still occupying the pool, so releasing it at the
    /// switch would admit a third agent while this one works, and re-acquiring
    /// could deadlock behind the very agent that handed it over.
    pub slot: Option<crate::utils::subagent_gate::SubagentSlot>,
    /// The tool pool this agent was given, used only to build the activity
    /// resolver — "Reading src/foo.rs" rather than "Read" in the progress tally
    /// the parent reads.
    pub tools: Vec<std::sync::Arc<dyn crate::tools::base::Tool>>,
    /// The conversation a backgrounded foreground agent had already reached.
    /// Carried so the final answer and the notification account for the whole
    /// run, and so the progress tally resumes instead of rewinding to zero —
    /// the source does the same, seeding its tracker from `agentMessages`
    /// before the backgrounded loop opens.
    pub initial_messages: Vec<Message>,
}

/// Drive a background agent from spawn to its terminal notification
/// (runAsyncAgentLifecycle).
///
/// Detached: the tool call that launched it has already returned, so nothing is
/// awaiting this. That is exactly why every exit has to announce itself — the
/// notification queue is the only route left back into the conversation, and an
/// agent that finishes without enqueueing one has done its work in silence.
///
/// The status transition comes BEFORE the notification on purpose: a
/// TaskOutput(block=true) waiting on this agent unblocks on the status, so
/// anything slower between the two would hold that caller for no reason.
pub async fn run_async_agent_lifecycle(lifecycle: AsyncAgentLifecycle) {
    use futures::future::FutureExt;
    // Nothing is awaiting this task, so a panic inside it is caught by the
    // tokio runtime and dropped on the floor: the JoinHandle is discarded at
    // the spawn site, and the agent would be left `running` for ever with
    // nothing announced — a TaskOutput(block=true) on it would wait on a run
    // that no longer exists. The Python lifecycle has had the equivalent arm
    // from the start (`except Exception` -> fail_agent_task + a 'failed'
    // notification); this is that arm, and the reason fail_async_agent exists.
    //
    // The partial transcript is NOT carried into the notification here: the
    // messages are owned by the frame that just unwound. Nothing is lost — the
    // agent's .output file holds every message it produced and the
    // notification names it, which is the same route TaskOutput reads.
    let agent_id = lifecycle.agent_id.clone();
    let description = lifecycle.description.clone();
    let tool_use_id = lifecycle.tool_use_id.clone();
    if let Err(panic) = std::panic::AssertUnwindSafe(lifecycle_body(lifecycle))
        .catch_unwind()
        .await
    {
        let error = panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "the agent panicked".to_string());
        crate::utils::log::log_error(&error, Some(&format!("ASYNC_AGENT_PANIC {agent_id}")));
        fail_async_agent(&agent_id, &description, &error, tool_use_id.as_deref(), None);
    }
}

async fn lifecycle_body(mut lifecycle: AsyncAgentLifecycle) {
    let agent_id = lifecycle.agent_id.clone();
    let description = lifecycle.description.clone();
    let tool_use_id = lifecycle.tool_use_id.clone();
    let mut tracker = create_progress_tracker();
    // Resolved once per run, not per message: the pool does not change, and the
    // description is stored on each tool use as it is recorded.
    let resolve_activity = crate::tasks::local_agent_task::create_activity_description_resolver(
        std::mem::take(&mut lifecycle.tools),
    );
    let mut agent_messages: Vec<Message> = std::mem::take(&mut lifecycle.initial_messages);
    // Fold what already happened into the tally before the stream opens, so a
    // handed-over agent's token and tool counts carry on from where the
    // foreground left them rather than appearing to rewind.
    for existing in &agent_messages {
        update_progress_from_message(&mut tracker, existing, Some(&resolve_activity));
    }

    // Wrapped in the one abort-aware iterator the rest of the codebase uses
    // rather than racing the stop here a second time. Nothing is awaiting this
    // run, so a stop that did not reach it would leave an agent working forever
    // with no one to report to.
    //
    // `abortable` STOPS on the signal, it does not fail — so the check after the
    // loop is what tells a finished run from an interrupted one.
    let mut stream = crate::utils::generators::abortable(
        std::mem::replace(&mut lifecycle.stream, Box::pin(futures::stream::empty())),
        lifecycle.abort.clone(),
        None,
    );

    while let Some(message) = stream.next().await {
        agent_messages.push(message.clone());

        // Append to the viewed transcript only while the UI holds the task.
        // Buffering for an agent nobody is watching is how a long background
        // run turns into a memory leak.
        let to_append = serde_json::to_value(&message).unwrap_or(serde_json::Value::Null);

        // The agent's own output file, so TaskOutput on a RUNNING agent returns
        // what it is doing rather than an empty string.
        crate::tasks::local_agent_task::append_agent_transcript(&agent_id, &to_append);

        update_task_state(&agent_id, |task| {
            let task = task.as_local_agent()?;
            if !task.retain {
                return None;
            }
            let mut next = task.clone();
            next.messages
                .get_or_insert_with(Vec::new)
                .push(to_append.clone());
            Some(next.into())
        });

        update_progress_from_message(&mut tracker, &message, Some(&resolve_activity));
        update_agent_progress(&agent_id, get_progress_update(&tracker));
        // The prose line, after the counts: update_agent_progress preserves an
        // existing summary rather than overwriting it, so the fresh one has to
        // be written through the path that owns it.
        if let Some(summary) =
            crate::tasks::local_agent_task::derive_agent_summary(&tracker)
        {
            crate::tasks::local_agent_task::update_agent_summary(&agent_id, &summary);
        }
    }

    let interrupted = lifecycle
        .abort
        .as_ref()
        .map(|t| t.is_cancelled())
        .unwrap_or(false);
    let final_message = extract_partial_result(&agent_messages);
    let progress = get_progress_update(&tracker);
    let elapsed_ms = lifecycle.started.elapsed().as_millis() as i64;
    let usage = AgentUsage {
        total_tokens: get_token_count_from_tracker(&tracker),
        tool_uses: progress.tool_use_count,
        duration_ms: elapsed_ms,
    };

    if interrupted {
        // Stopped, not broken. kill_async_agent is a no-op when TaskStop
        // already marked it killed — but only here are the messages held, so
        // the notification fires regardless, carrying whatever the agent had
        // reached. Dropping it would throw away work the user asked to stop,
        // not work that failed.
        kill_async_agent(&agent_id);
        enqueue_agent_notification(
            &agent_id,
            &description,
            "killed",
            None,
            final_message.as_deref(),
            None,
            tool_use_id.as_deref(),
            None,
            None,
        );
    } else {
        complete_agent_task(
            &agent_id,
            serde_json::json!({
                "agent_id": agent_id,
                "text": final_message.clone().unwrap_or_default(),
                "tool_use_count": progress.tool_use_count,
                "duration": lifecycle.started.elapsed().as_secs_f64(),
                "usage": {
                    "total_tokens": usage.total_tokens,
                    "tool_uses": usage.tool_uses,
                    "duration_ms": usage.duration_ms,
                },
            }),
        );
        enqueue_agent_notification(
            &agent_id,
            &description,
            "completed",
            None,
            final_message.as_deref(),
            Some(&usage),
            tool_use_id.as_deref(),
            None,
            None,
        );
    }

    // The slot is held for the whole run and released here, so a background
    // agent counts against the pool exactly as long as it is actually working.
    drop(lifecycle.slot.take());
}

/// Mark a background agent failed and announce it (the catch arm of
/// runAsyncAgentLifecycle).
///
/// Separate because the Rust stream cannot hand an error out of its iteration
/// the way an async generator throws: the query stream yields messages, and a
/// failure that has to be reported arrives from the caller instead.
pub fn fail_async_agent(
    agent_id: &str,
    description: &str,
    error: &str,
    tool_use_id: Option<&str>,
    partial: Option<&str>,
) {
    fail_agent_task(agent_id, error);
    enqueue_agent_notification(
        agent_id,
        description,
        "failed",
        Some(error),
        partial,
        None,
        tool_use_id,
        None,
        None,
    );
}

#[cfg(test)]
mod tests {
    //! A backgrounded agent answers the parent at once and reports back later.
    //!
    //! The tool call returns the moment the agent is registered, so nothing is
    //! awaiting the run — which is why every exit from the detached driver has
    //! to announce itself. The notification queue is the only route left back
    //! into the conversation, and an agent that finishes without enqueueing one
    //! has done its work in silence.
    use super::*;
    use crate::state::app_state::{get_app_state, reset_app_state, test_lock};
    use crate::tasks::local_agent_task::{auto_background_ms, register_async_agent};
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::types::messages::{ApiMessage, BetaUsage, TextBlock, ToolUseBlock};
    use crate::types::query::AssistantMessage;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::message_queue_manager as mq;
    use std::sync::Arc;

    /// A background agent that panics still becomes terminal and still speaks.
    ///
    /// Nothing awaits a detached agent, so tokio catches the panic and drops it
    /// — the task would stay `running` for ever and a TaskOutput(block=true) on
    /// it would wait on a run that no longer exists. The Python lifecycle has
    /// always had this arm; this pins the Rust one.
    #[tokio::test]
    async fn a_panicking_agent_is_marked_failed_and_announced() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();

        let id = crate::utils::agent_id::create_agent_id(None);
        let token = tokio_util::sync::CancellationToken::new();
        crate::tasks::local_agent_task::register_async_agent(
            &id,
            "the one that breaks",
            "p",
            crate::tools::built_in_agents::get_agent_definition(None)
                .expect("the default agent definition"),
            Some("toolu_panic".into()),
            Some(token.clone()),
        );

        // A stream whose first pull panics, which is what a bug inside the
        // agent's own run looks like from out here.
        let stream: Pin<Box<dyn Stream<Item = Message> + Send>> =
            Box::pin(futures::stream::once(async {
                panic!("the lane refused");
            }));

        // Spawned, so the panic unwinds a task exactly as it would in
        // production rather than failing this test directly.
        let task_id = id.clone();
        tokio::spawn(async move {
            super::run_async_agent_lifecycle(super::AsyncAgentLifecycle {
                agent_id: task_id,
                description: "the one that breaks".into(),
                stream,
                abort: Some(token),
                tool_use_id: Some("toolu_panic".into()),
                started: std::time::Instant::now(),
                slot: None,
                tools: Vec::new(),
                initial_messages: Vec::new(),
            })
            .await;
        })
        .await
        .expect("the lifecycle must absorb the panic, not propagate it");

        let task = crate::state::app_state::get_app_state()
            .tasks
            .get(&id)
            .and_then(|t| t.as_local_agent().cloned())
            .expect("the task is still registered");
        assert_eq!(task.base.status, "failed", "a panicking agent is not left running");
        assert!(
            task.error.as_deref().unwrap_or_default().contains("the lane refused"),
            "carrying what broke: {:?}",
            task.error
        );

        let notes = crate::utils::message_queue_manager::drain_pending_notifications();
        assert_eq!(notes.len(), 1, "the model is told exactly once: {notes:?}");
        assert!(notes[0].contains("failed"), "{}", notes[0]);
        assert!(
            notes[0].contains("toolu_panic"),
            "stamped with the tool use that launched it: {}",
            notes[0]
        );

        crate::state::app_state::reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();
    }

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

    fn assistant(text: Option<&str>, tools: &[&str]) -> Message {
        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(t) = text {
            content.push(ContentBlock::Text(TextBlock { text: t.to_string() }));
        }
        for (i, name) in tools.iter().enumerate() {
            content.push(ContentBlock::ToolUse(ToolUseBlock {
                id: format!("tu_{i}"),
                name: (*name).to_string(),
                input: serde_json::json!({}),
            }));
        }
        Message::Assistant(AssistantMessage {
            uuid: String::new(),
            message: ApiMessage {
                content,
                usage: BetaUsage {
                    input_tokens: 500,
                    output_tokens: 20,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
                ..Default::default()
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    /// Like `assistant`, but with tool ids starting at `base` — the default
    /// helper numbers from tu_0 per message, so two turns would claim the same
    /// id and an unanswered call would look answered.
    fn assistant_from(text: Option<&str>, tools: &[&str], base: usize) -> Message {
        let Message::Assistant(mut a) = assistant(text, tools) else {
            unreachable!()
        };
        let mut i = base;
        for block in a.message.content.iter_mut() {
            if let ContentBlock::ToolUse(tu) = block {
                tu.id = format!("tu_{i}");
                i += 1;
            }
        }
        Message::Assistant(a)
    }

    fn user_tool_result(tool_use_id: &str) -> Message {
        use crate::types::messages::{MessageParam, ToolResultContent};
        use crate::types::query::UserMessage;
        Message::User(UserMessage {
            uuid: String::new(),
            message: MessageParam {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlockParam::ToolResult {
                    tool_use_id: tool_use_id.to_string(),
                    content: ToolResultContent::default(),
                    is_error: false,
                    cache_control: None,
                    data: None,
                }]),
            },
            is_meta: false,
            tool_use_result: None,
        })
    }

    fn spawn_async(description: &str) -> String {
        let id = create_agent_id(None);
        register_async_agent(
            &id, description, "survey the lane", fake_agent(),
            Some("toolu_bg".into()), Some(CancellationToken::new()),
        );
        id
    }

    #[test]
    fn the_answer_is_the_last_assistant_message_that_has_text() {
        assert_eq!(
            extract_partial_result(&[
                assistant(Some("first"), &[]),
                assistant(Some("second"), &[]),
                assistant(None, &["Read"]),
            ])
            .as_deref(),
            Some("second"),
            "a tool_use-only ending must not report an empty answer"
        );
        assert_eq!(extract_partial_result(&[assistant(None, &["Read"])]), None);
        assert_eq!(extract_partial_result(&[]), None);
    }

    #[tokio::test]
    async fn a_completed_background_agent_announces_itself_with_its_answer() {
        let _guard = test_lock();
        reset_app_state();
        mq::reset_pending_notifications();
        let id = spawn_async("survey the lane");
        let abort = CancellationToken::new();

        run_async_agent_lifecycle(AsyncAgentLifecycle {
            agent_id: id.clone(),
            description: "survey the lane".into(),
            stream: Box::pin(futures::stream::iter(vec![assistant(
                Some("found it in lane 3"),
                &["Read", "Grep"],
            )])),
            abort: Some(abort),
            tool_use_id: Some("toolu_bg".into()),
            started: std::time::Instant::now(),
            slot: None,
            tools: Vec::new(),
            initial_messages: Vec::new(),
        })
        .await;

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(task.base.status, "completed");
        assert_eq!(task.result.as_ref().unwrap()["text"], "found it in lane 3");
        let progress = task.progress.as_ref().unwrap();
        assert_eq!((progress.token_count, progress.tool_use_count), (520, 2));

        let notes = mq::drain_pending_notifications();
        assert_eq!(notes.len(), 1, "nothing was awaiting it — it has to announce");
        let note = &notes[0];
        assert!(note.contains("<status>completed</status>"));
        assert!(
            note.contains("<result>found it in lane 3</result>"),
            "the answer travels in the notification — that IS the delivery: {note}"
        );
        assert!(note.contains("<total_tokens>520</total_tokens>"));
        assert!(note.contains("<tool_uses>2</tool_uses>"));
        assert!(note.contains("<tool-use-id>toolu_bg</tool-use-id>"));
        assert!(task.base.notified, "announced once");
        reset_app_state();
    }

    #[tokio::test]
    async fn a_stopped_background_agent_hands_back_what_it_reached() {
        let _guard = test_lock();
        reset_app_state();
        mq::reset_pending_notifications();
        let id = spawn_async("survey the lane");
        let abort = CancellationToken::new();
        abort.cancel();

        run_async_agent_lifecycle(AsyncAgentLifecycle {
            agent_id: id.clone(),
            description: "survey the lane".into(),
            stream: Box::pin(futures::stream::iter(vec![assistant(
                Some("still working"),
                &["Read"],
            )])),
            abort: Some(abort),
            tool_use_id: None,
            started: std::time::Instant::now(),
            slot: None,
            tools: Vec::new(),
            initial_messages: Vec::new(),
        })
        .await;

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(task.base.status, "killed");
        let notes = mq::drain_pending_notifications();
        assert_eq!(notes.len(), 1, "a stop is announced too");
        assert!(notes[0].contains("<status>killed</status>"));
        assert!(
            notes[0].contains("<result>still working</result>"),
            "work the user asked to stop is still work: {}",
            notes[0]
        );
        reset_app_state();
    }

    #[test]
    fn a_failed_background_agent_is_announced_rather_than_lost() {
        let _guard = test_lock();
        reset_app_state();
        mq::reset_pending_notifications();
        let id = spawn_async("try the kpool");
        fail_async_agent(&id, "try the kpool", "the lane refused", None, Some("partial"));

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(task.base.status, "failed");
        assert_eq!(task.error.as_deref(), Some("the lane refused"));
        let notes = mq::drain_pending_notifications();
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("failed: the lane refused"), "{}", notes[0]);
        reset_app_state();
    }

    #[test]
    fn the_auto_background_timer_is_off_unless_asked_for() {
        assert_eq!(auto_background_ms(), 0);
        std::env::set_var("CODEMASON_AUTO_BACKGROUND_TASKS", "1");
        let armed = auto_background_ms();
        std::env::remove_var("CODEMASON_AUTO_BACKGROUND_TASKS");
        assert_eq!(armed, 120_000);
        assert_eq!(auto_background_ms(), 0);
    }

    /// A conversation that still has a tool call in flight cannot be sent again.
    #[test]
    fn an_assistant_turn_whose_tool_never_returned_is_dropped() {
        let complete = vec![
            assistant(Some("one"), &["Read"]),
            user_tool_result("tu_0"),
            assistant(Some("two"), &[]),
        ];
        assert_eq!(
            filter_incomplete_tool_calls(&complete).len(),
            3,
            "a complete conversation passes through untouched"
        );

        let orphaned = vec![
            assistant(Some("one"), &["Read"]),
            user_tool_result("tu_0"),
            assistant_from(Some("reaching"), &["Bash"], 5),
        ];
        let kept = filter_incomplete_tool_calls(&orphaned);
        assert_eq!(
            kept.len(),
            2,
            "the turn holding an unanswered tool_use goes, text blocks and all"
        );
        assert!(
            matches!(&kept[0], Message::Assistant(_)) && matches!(&kept[1], Message::User(_)),
            "the rest of the conversation is kept, in order"
        );

        // Order does not matter: the result is collected before the filter runs.
        assert_eq!(
            filter_incomplete_tool_calls(&[
                user_tool_result("tu_0"),
                assistant(Some("one"), &["Read"]),
            ])
            .len(),
            2
        );
        assert!(filter_incomplete_tool_calls(&[]).is_empty());
    }

    /// A handed-over agent carries its past with it.
    #[tokio::test]
    async fn a_handed_over_agent_resumes_its_counts_instead_of_rewinding() {
        let _guard = test_lock();
        reset_app_state();
        mq::reset_pending_notifications();
        let id = spawn_async("survey the lane");

        // What the foreground had reached before the switch: one tool, one turn.
        let already = vec![assistant(Some("foreground step"), &["Read"])];

        run_async_agent_lifecycle(AsyncAgentLifecycle {
            agent_id: id.clone(),
            description: "survey the lane".into(),
            stream: Box::pin(futures::stream::iter(vec![assistant(
                Some("resumed and finished"),
                &["Grep"],
            )])),
            abort: Some(CancellationToken::new()),
            tool_use_id: Some("toolu_fg".into()),
            started: std::time::Instant::now(),
            slot: None,
            tools: Vec::new(),
            initial_messages: already,
        })
        .await;

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(task.base.status, "completed");
        assert_eq!(
            task.result.as_ref().unwrap()["text"],
            "resumed and finished",
            "the answer is the resumed run's, not the foreground's"
        );
        let progress = task.progress.as_ref().unwrap();
        assert_eq!(
            progress.tool_use_count, 2,
            "the whole run is counted, foreground steps included"
        );
        // latest input (500) + every turn's output summed across both runs.
        assert_eq!(progress.token_count, 540);
        reset_app_state();
    }

    /// What the tool wires up for the background path.
    #[test]
    fn the_agent_tool_launches_and_answers_at_once() {
        let src = include_str!("agent_tool.rs");
        assert!(src.contains("register_async_agent("));
        assert!(src.contains("\"status\": \"async_launched\""));
        assert!(src.contains("run_async_agent_lifecycle("));
        assert!(
            src.find("register_async_agent(").unwrap()
                < src.find("register_agent_foreground(").unwrap(),
            "a background launch must not also register as foreground"
        );
        assert!(
            src.find("tokio::spawn").unwrap() < src.find("\"async_launched\"").unwrap(),
            "the run is detached before the call answers"
        );
        // The slot is taken inside the detached task, never before the answer.
        let spawn_at = src.find("tokio::spawn").unwrap();
        let launched_at = src.find("\"async_launched\"").unwrap();
        let gate_at = src[spawn_at..launched_at].find("subagent_gate::try_acquire");
        assert!(gate_at.is_some(), "the slot is acquired inside the detached run");

        // The name the model gave the agent, recorded for SendMessage routing
        // AFTER the spawn — a spawn that fails must leave no entry pointing at
        // an agent that does not exist.
        let name_at = src
            .find("agent_name_registry.insert")
            .expect("a named spawn is addressable by that name");
        assert!(
            src.find("register_async_agent(").unwrap() < name_at && name_at < launched_at,
            "the name is registered after the spawn and before the answer"
        );
        assert!(
            name_at < src.find("register_agent_foreground(").unwrap(),
            "only a background spawn registers a name: a foreground agent is \
             blocking its parent, so there is nobody left to address it"
        );
    }

    /// What the tool wires up for the foreground hand-over.
    #[test]
    fn the_agent_tool_races_every_pull_against_the_background_signal() {
        let src = include_str!("agent_tool.rs");
        assert!(
            src.contains("agent_task.background_signal.clone()")
                && src.contains("tokio::pin!(notified)"),
            "one signal future for the whole run, not one per pull"
        );
        assert!(
            src.contains("filter_incomplete_tool_calls(&messages)"),
            "an orphaned tool_use must not travel with the conversation"
        );
        assert!(
            src.contains("initial_messages: resume_messages"),
            "the conversation so far is handed over, so the run resumes"
        );
        assert!(
            src.contains("slot: detached_slot"),
            "the gate slot travels with the agent, it is not released and re-taken"
        );
        // The re-check before committing to the switch: the signal and the flag
        // are set together, so a signal without the flag is not a backgrounding.
        let notified_at = src.find("_ = &mut notified").expect("the race arm");
        let tail = &src[notified_at..];
        assert!(
            tail.find("is_backgrounded").unwrap() < tail.find("was_backgrounded = true").unwrap(),
            "the task is re-read before the hand-over commits"
        );
    }
}
