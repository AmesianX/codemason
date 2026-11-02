//! local_agent task — adapted from tasks/LocalAgentTask/LocalAgentTask.tsx.
//! Equivalent to codemason/tasks/local_agent_task.py.
//!
//! Ported functions (original names in parentheses):
//!
//!     register_agent_foreground              (EXK)  — spawn a foreground agent task
//!     register_async_agent (Fm8)  — spawn one already backgrounded
//!     background_agent_task          (lXK)  — move a running one to background
//!     unregister_agent_foreground              (LXK)  — drop a finished foreground one
//!     kill_async_agent                (aq6)  — cancel + mark killed
//!     complete_agent_task            (hL8)  — mark completed with its result
//!     fail_agent_task                (SL8)  — mark failed with its error
//!     is_local_agent_task            (EJ)   — type guard
//!
//! Why this exists — the two lifecycle failures it rules out:
//!
//!   * **An agent vanishing before its result is collected.**
//!     `unregister_agent_foreground` refuses to delete a task whose `is_backgrounded` is
//!     set, because the parent collects that one later through TaskOutput
//!     rather than from the tool's own return value.
//!
//!   * **An agent lingering after its owner is gone.** Spawning registers a
//!     shutdown cleanup (utils::cleanup_registry) that kills the task; its
//!     handle is stored on the task, and every terminal transition — killed,
//!     completed, failed — releases it before dropping the runtime handles.
//!     Task removal releases it too, after the delete.
//!
//! `retain` is not a spawn option. In the source it is flipped on only by
//! entering the transcript view, and it means two things: the task is exempt
//! from automatic eviction (`evict_after` stays None) and the streaming loop
//! buffers the agent's messages into task state. codemason has no transcript
//! viewer — Ink substrate — so nothing flips it, which matches the original's
//! behaviour for an agent nobody is watching. The field and the eviction
//! arithmetic are ported because the terminal transitions and
//! evict_terminal_task both key off them.

use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::constants::xml::{
    OUTPUT_FILE_TAG, STATUS_TAG, SUMMARY_TAG, TASK_ID_TAG, TASK_NOTIFICATION_TAG,
    TOOL_USE_ID_TAG, WORKTREE_BRANCH_TAG, WORKTREE_PATH_TAG, WORKTREE_TAG,
};
use crate::state::app_state::{get_app_state, set_app_state};
use crate::types::messages::ContentBlock;
use crate::types::query::Message;
use crate::task::{create_task_state_base, now_ms, Task, TaskStateBase};
use crate::tools::built_in_agents::AgentDefinition;
use crate::utils::cleanup_registry::{register_cleanup, unregister_cleanup, CleanupHandle};
use crate::utils::task::framework::{register_task, update_task_state, PANEL_GRACE_MS};

/// Auto-background delay (getAutoBackgroundMs). The source returns 120s only when
/// CLAUDE_AUTO_BACKGROUND_TASKS is truthy or the tengu_auto_background_agents
/// gate is on (default false), and 0 otherwise — so the timer is OFF by
/// default. codemason has no gate service; the env var alone decides, under
/// codemason's own namespace, with the original name accepted as well.
pub const AUTO_BACKGROUND_DELAY_MS: u64 = 120_000;

/// Milliseconds before a foreground agent auto-backgrounds, 0 = never.
pub fn auto_background_ms() -> u64 {
    for name in ["CODEMASON_AUTO_BACKGROUND_TASKS", "CLAUDE_AUTO_BACKGROUND_TASKS"] {
        if let Ok(v) = std::env::var(name) {
            if matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on") {
                return AUTO_BACKGROUND_DELAY_MS;
            }
        }
    }
    0
}

/// How many recent tool activities a tracker keeps (MAX_RECENT_ACTIVITIES).
pub const MAX_RECENT_ACTIVITIES: usize = 5;

/// One tool use, as the progress view shows it (ToolActivity).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolActivity {
    pub tool_name: String,
    pub input: Value,
    /// Pre-computed description from the tool, e.g. "Reading src/foo.ts".
    pub activity_description: Option<String>,
    /// Pre-computed: a search (Grep, Glob, ...) / a read (Read, cat, ...).
    pub is_search: Option<bool>,
    pub is_read: Option<bool>,
}

/// What an agent has done so far, as stored on its task (AgentProgress).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentProgress {
    pub tool_use_count: i64,
    pub token_count: i64,
    pub last_activity: Option<ToolActivity>,
    pub recent_activities: Option<Vec<ToolActivity>>,
    pub summary: Option<String>,
}

/// The running tally an agent's loop keeps (ProgressTracker).
///
/// Input and output are tracked separately to avoid double counting: the prompt
/// token count is cumulative per turn (it includes all previous context), so
/// only the latest value is kept, while output is per-turn and therefore
/// summed. The cache fields belong to the same prompt as `input_tokens` and are
/// added to it, not accumulated.
#[derive(Debug, Clone, Default)]
pub struct ProgressTracker {
    pub tool_use_count: i64,
    pub latest_input_tokens: i64,
    pub cumulative_output_tokens: i64,
    pub recent_activities: Vec<ToolActivity>,
}

/// A fresh tracker (createProgressTracker).
pub fn create_progress_tracker() -> ProgressTracker {
    ProgressTracker::default()
}

/// The agent's context size so far (getTokenCountFromTracker).
pub fn get_token_count_from_tracker(tracker: &ProgressTracker) -> i64 {
    tracker.latest_input_tokens + tracker.cumulative_output_tokens
}

/// Given a tool name and its input, a human-readable line for the progress view
/// (ActivityDescriptionResolver). Resolved at record time from the tool's own
/// activity description, so the view never has to look a tool up again.
pub type ActivityDescriptionResolver<'a> = &'a dyn Fn(&str, &Value) -> Option<String>;

/// Resolve activity descriptions out of a tool pool
/// (createActivityDescriptionResolver).
///
/// Returned as an owned closure so the callers can hand it to
/// `update_progress_from_message` for the life of a run. A tool's display
/// helper must never be able to fail the turn it is only describing, so a name
/// that is not in the pool simply resolves to None and the view falls back to
/// the tool name.
pub fn create_activity_description_resolver(
    tools: Vec<std::sync::Arc<dyn crate::tools::base::Tool>>,
) -> impl Fn(&str, &Value) -> Option<String> {
    move |tool_name: &str, tool_input: &Value| {
        tools
            .iter()
            .find(|t| t.name() == tool_name)
            .and_then(|t| t.get_activity_description(Some(tool_input)))
    }
}

/// Fold one assistant message into the tracker (updateProgressFromMessage).
///
/// Only assistant messages carry usage and tool uses, so everything else is
/// ignored. The token arithmetic is the reason this is a tracker and not a
/// running sum: the prompt count already includes every earlier turn, so adding
/// it each time would multiply the context size by the turn count.
pub fn update_progress_from_message(
    tracker: &mut ProgressTracker,
    message: &Message,
    resolve_activity_description: Option<ActivityDescriptionResolver<'_>>,
) {
    let Message::Assistant(assistant) = message else {
        return;
    };
    let usage = &assistant.message.usage;
    // Keep the latest prompt count, sum the completions. The two cache fields
    // are part of the same prompt; codemason's own backend fills them in
    // BetaUsage and leaves them 0 when the server reports no cache, so reading
    // them costs nothing and counts the whole prompt when it does.
    tracker.latest_input_tokens =
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens;
    tracker.cumulative_output_tokens += usage.output_tokens;

    for content in &assistant.message.content {
        if let ContentBlock::ToolUse(tu) = content {
            tracker.tool_use_count += 1;
            tracker.recent_activities.push(ToolActivity {
                tool_name: tu.name.clone(),
                input: tu.input.clone(),
                activity_description: resolve_activity_description
                    .and_then(|resolve| resolve(&tu.name, &tu.input)),
                is_search: None,
                is_read: None,
            });
        }
    }

    while tracker.recent_activities.len() > MAX_RECENT_ACTIVITIES {
        tracker.recent_activities.remove(0);
    }
}

/// A one-line "what is it doing right now", from what we already know.
///
/// codemason's local equivalent of startAgentSummarization. The source forks the
/// agent's transcript every 30 seconds and asks the model for the line, with a
/// prompt whose own examples are "Reading runAgent.ts" and "Running auth module
/// tests" — which is exactly the shape each tool's get_activity_description
/// already produces, deterministically and for free. So the line is derived
/// rather than generated.
///
/// What that trades away: the model can describe an action its tool arguments do
/// not name — "Fixing null check in validate.ts" reads the edit, not just the
/// path. What it buys is the reason to prefer it here. A summary fork is another
/// tenant in the server's KV pool, every 30 seconds, per running agent; on this
/// deployment the pool is the binding constraint (see SUBAGENT_TOTAL_FRACTION in
/// services::compaction), and three agents would add six requests a minute to it
/// for a line we can already write.
///
/// None when nothing has happened yet, so the attachment that reads this simply
/// omits its Progress line rather than printing an empty one.
pub fn derive_agent_summary(tracker: &ProgressTracker) -> Option<String> {
    tracker
        .recent_activities
        .iter()
        .rev()
        .find_map(|a| a.activity_description.clone())
}

/// The snapshot to store on the task (getProgressUpdate).
pub fn get_progress_update(tracker: &ProgressTracker) -> AgentProgress {
    AgentProgress {
        tool_use_count: tracker.tool_use_count,
        token_count: get_token_count_from_tracker(tracker),
        last_activity: tracker.recent_activities.last().cloned(),
        recent_activities: Some(tracker.recent_activities.clone()),
        summary: None,
    }
}

/// State for a sub-agent task (LocalAgentTaskState).
///
/// `abort_controller` is the agent's own cancellation token — a child of the
/// parent's, so a stop aimed at this agent spares its siblings.
/// `selected_agent` holds the resolved agent definition and, like the two
/// runtime handles, is dropped on a terminal transition: the source nulls it so
/// a finished task stops pinning the definition and its tool pool.
#[derive(Clone, Default)]
pub struct LocalAgentTaskState {
    pub base: TaskStateBase,
    pub agent_id: String,
    pub prompt: String,
    pub selected_agent: Option<AgentDefinition>,
    pub agent_type: String,
    pub model: Option<String>,
    pub progress: Option<AgentProgress>,
    pub abort_controller: Option<CancellationToken>,
    pub unregister_cleanup: Option<CleanupHandle>,
    pub retrieved: bool,
    pub last_reported_tool_count: i64,
    pub last_reported_token_count: i64,
    pub is_backgrounded: bool,
    pub pending_messages: Vec<String>,
    pub retain: bool,
    pub disk_loaded: bool,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub evict_after: Option<i64>,
    /// Buffered agent messages — only ever populated for a retained task.
    pub messages: Option<Vec<Value>>,
}

// AgentDefinition holds an `Arc<dyn Fn>` system-prompt closure and cannot
// derive Debug, so the definition is reported by its agent type.
impl fmt::Debug for LocalAgentTaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalAgentTaskState")
            .field("base", &self.base)
            .field("agent_id", &self.agent_id)
            .field("agent_type", &self.agent_type)
            .field("model", &self.model)
            .field("progress", &self.progress)
            .field("selected_agent", &self.selected_agent.as_ref().map(|a| &a.agent_type))
            .field("is_backgrounded", &self.is_backgrounded)
            .field("retain", &self.retain)
            .field("retrieved", &self.retrieved)
            .field("last_reported_tool_count", &self.last_reported_tool_count)
            .field("last_reported_token_count", &self.last_reported_token_count)
            .field("evict_after", &self.evict_after)
            .field("result", &self.result)
            .field("error", &self.error)
            .finish()
    }
}

/// Type guard (isLocalAgentTask).
pub fn is_local_agent_task(task: &LocalAgentTaskState) -> bool {
    task.base.task_type == "local_agent"
}

/// A local_agent task that is a spawned agent, not the main session
/// (isPanelAgentTask).
///
/// The source reserves `agentType === 'main-session'` for the session
/// registering ITSELF as a local_agent task, which must not be listed or
/// stopped as if it were a sub-agent. codemason does not register the main
/// session yet, so today this agrees with is_local_agent_task on every task —
/// it exists so the one predicate every listing must share is already the right
/// one when that task appears.
pub fn is_panel_agent_task(task: &LocalAgentTaskState) -> bool {
    is_local_agent_task(task) && task.agent_type != "main-session"
}

/// Queue a message for a running agent to read (queuePendingMessage).
///
/// Queued, not delivered: the agent is mid-turn and its message list belongs to
/// the loop that is running it. The loop drains this at its next tool-round
/// boundary.
pub fn queue_pending_message(agent_id: &str, msg: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        let mut next = task.clone();
        next.pending_messages.push(msg.to_string());
        Some(next.into())
    });
}

/// Append a message to the agent's displayed transcript
/// (appendMessageToLocalAgent).
///
/// Separate from queuing on purpose: queuing routes a prompt to the agent's
/// input, this puts it in what the user sees.
///
/// A seam with no caller here, deliberately. The source's only call site is
/// `onAgentSubmit` — the user typing INTO an agent's transcript while watching
/// it in the teammate view, a screen codemason does not have. The two things
/// that call site then does, we do have and do reach: a running agent gets the
/// line queued (queue_pending_message) and a stopped one is relaunched
/// (resume_agent_background), both driven by the SendMessage tool. Kept because
/// the half that is missing is a view, not a behaviour.
pub fn append_message_to_local_agent(agent_id: &str, message: Value) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        let mut next = task.clone();
        next.messages.get_or_insert_with(Vec::new).push(message.clone());
        Some(next.into())
    });
}

/// Take everything queued for this agent, leaving the queue empty
/// (drainPendingMessages).
pub fn drain_pending_messages(agent_id: &str) -> Vec<String> {
    let drained = match get_app_state()
        .tasks
        .get(agent_id)
        .and_then(|t| t.as_local_agent())
    {
        Some(agent) if !agent.pending_messages.is_empty() => agent.pending_messages.clone(),
        _ => return Vec::new(),
    };
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        let mut next = task.clone();
        next.pending_messages.clear();
        Some(next.into())
    });
    drained
}

/// agentId -> the uuid of the last transcript entry written for that agent, so
/// each new line can name its parent and the file reads as a chain
/// (recordSidechainTranscript's lastRecordedUuid).
static AGENT_TRANSCRIPT_CHAINS: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Append one message to the agent's own output file
/// (recordSidechainTranscript, via initTaskOutputAsSymlink).
///
/// This is what makes TaskOutput mean something for a RUNNING agent. The source
/// points the task's output path at the agent's transcript JSONL — its own
/// comment calls it "the full session transcript (every message, tool use,
/// etc.), not just the subagent's answer" — and prefers the in-memory `result`
/// once the run is over precisely because the file is the raw transcript.
/// Without a writer on this side the path stayed an empty file forever, so a
/// parent watching a sub-agent could read its token count and its tool count
/// but never a word of what it was actually doing.
///
/// codemason has no per-agent session file to link to, so the output path is
/// written directly, in the same JSONL shape session_storage uses. The chain is
/// kept per agent: borrowing session_storage's own write state would splice
/// sub-agent entries into the main transcript's parent links.
///
/// Best-effort, like every other write to this file: an agent's run must not
/// fail because its log could not be appended to.
pub fn append_agent_transcript(agent_id: &str, message: &serde_json::Value) {
    let mtype = message.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if !crate::utils::session_storage::TRANSCRIPT_TYPES.contains(&mtype) {
        // progress and the like are not transcript entries, here as there.
        return;
    }
    let mut chains = AGENT_TRANSCRIPT_CHAINS.lock();
    let mut parent = chains.get(agent_id).cloned();
    if parent.is_none() {
        // No chain in memory, but the file may already hold this agent's
        // earlier turns: a terminal transition drops the chain, and a resume
        // then appends to the same file. Writing a second root there made
        // everything before the resume unreachable from the newest leaf —
        // get_agent_transcript walks parentUuid back from it — so a resumed
        // agent was handed one message instead of its whole history, and a
        // second resume would have lost the first resume's work too.
        parent = last_transcript_uuid(agent_id);
    }
    let mut entry =
        crate::utils::session_storage::serialize_entry(message, parent.as_deref(), true);
    if let Some(obj) = entry.as_object_mut() {
        // The agent this line belongs to, which the main transcript's shape has
        // no field for: a reader of this file is looking at one agent, and the
        // id is what ties a line back to the task it came from.
        obj.insert(
            "agentId".into(),
            serde_json::Value::String(agent_id.to_string()),
        );
    }
    let uuid_str = entry
        .get("uuid")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let path = crate::utils::task::disk_output::get_task_output_path(agent_id);
    if crate::utils::session_storage::append_entry_to_file(&path, &entry).is_ok()
        && !uuid_str.is_empty()
    {
        chains.insert(agent_id.to_string(), uuid_str);
    }
}

/// An agent's own conversation, read back off disk (getAgentTranscript).
///
/// The agent's output file IS its transcript — see append_agent_transcript — so
/// this is the session reader pointed at that path: parse the JSONL, find the
/// latest leaf, and walk its parentUuid chain back to the root. The chain walk
/// matters even for a file only one writer ever appended to: it is what turns
/// the entries back into an ordered conversation rather than a bag.
///
/// None when there is nothing to read, which the caller must treat as "cannot
/// resume" rather than "resume from empty".
pub fn get_agent_transcript(agent_id: &str) -> Option<Vec<Value>> {
    let path = crate::utils::task::disk_output::get_task_output_path(agent_id);
    let (messages, order) = crate::utils::session_storage::load_transcript_file(&path).ok()?;
    if messages.is_empty() || order.is_empty() {
        return None;
    }
    let leaf = crate::utils::session_storage::find_latest_leaf(&messages, &order)?;
    let chain = crate::utils::session_storage::build_conversation_chain(&messages, leaf);
    // Back to the in-memory message shape: the stored entry carries the
    // transcript's own bookkeeping (parentUuid, cwd, sessionId) around the
    // message itself, and only the message is a conversation turn. Images were
    // moved to the session's image store when the entry was written
    // (serialize_entry); they come back here as they do for a resumed session
    // (restore_messages), or a resumed agent would see references, not images.
    let out: Vec<Value> = chain
        .into_iter()
        .filter(|e| matches!(e.get("type").and_then(|v| v.as_str()), Some("user") | Some("assistant")))
        .map(|e| {
            let message = e.get("message").cloned().unwrap_or(serde_json::json!({}));
            let sid = e.get("sessionId").and_then(|v| v.as_str());
            serde_json::json!({
                "type": e.get("type").cloned().unwrap_or(Value::Null),
                "uuid": e.get("uuid").cloned().unwrap_or(Value::String(String::new())),
                "message": crate::utils::image_store::hydrate(&message, sid),
            })
        })
        .collect();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// The uuid of the newest entry already on disk for this agent, if any.
///
/// The leaf by the same rule the reader uses, so the entry written next hangs
/// off exactly what a read would walk back from. Best-effort: an unreadable or
/// absent file simply means this write starts the chain.
fn last_transcript_uuid(agent_id: &str) -> Option<String> {
    let path = crate::utils::task::disk_output::get_task_output_path(agent_id);
    let (messages, order) = crate::utils::session_storage::load_transcript_file(&path).ok()?;
    if messages.is_empty() || order.is_empty() {
        return None;
    }
    let leaf = crate::utils::session_storage::find_latest_leaf(&messages, &order)?;
    leaf.get("uuid")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Drop an agent's chain state (the tail of recordSidechainTranscript).
///
/// Called from the terminal transitions. The chain is one short string per
/// agent, but a long-lived session spawning thousands of them would keep every
/// one of them, and nothing would ever read them again.
pub fn forget_agent_transcript(agent_id: &str) {
    AGENT_TRANSCRIPT_CHAINS.lock().remove(agent_id);
}

/// agentId -> that agent's background signal (gR6).
static BACKGROUND_SIGNAL_RESOLVERS: Lazy<Mutex<HashMap<String, std::sync::Arc<Notify>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// What a foreground spawn hands back (EXK's return value).
///
/// `background_signal` fires once the task goes to the background, whether by
/// the auto-background timer or by an explicit `background_agent_task`; the
/// agent loop selects on it against the next message. `cancel_auto_background`
/// is None when no timer was armed.
pub struct AgentTaskHandle {
    pub task_id: String,
    pub background_signal: std::sync::Arc<Notify>,
    pub cancel_auto_background: Option<CancellationToken>,
}

/// The task record both spawn paths build (the shared object literal).
fn base_agent_task(
    agent_id: &str,
    description: &str,
    prompt: &str,
    selected_agent: AgentDefinition,
    abort_controller: Option<CancellationToken>,
    tool_use_id: Option<String>,
    is_backgrounded: bool,
) -> LocalAgentTaskState {
    let mut base = create_task_state_base(agent_id, "local_agent", description, tool_use_id);
    // The base helper starts every task 'pending'; the agent spawn paths
    // override it to 'running'. Not cosmetic: every terminal transition guards
    // on `status == "running"`, so a task left pending could never be
    // completed, failed or killed.
    base.status = "running".to_string();
    let agent_type = if selected_agent.agent_type.is_empty() {
        "general-purpose".to_string()
    } else {
        selected_agent.agent_type.clone()
    };
    LocalAgentTaskState {
        base,
        agent_id: agent_id.to_string(),
        prompt: prompt.to_string(),
        selected_agent: Some(selected_agent),
        agent_type,
        model: None,
        progress: None,
        abort_controller,
        unregister_cleanup: None,
        retrieved: false,
        last_reported_tool_count: 0,
        last_reported_token_count: 0,
        is_backgrounded,
        pending_messages: Vec::new(),
        retain: false,
        disk_loaded: false,
        result: None,
        error: None,
        evict_after: None,
        messages: None,
    }
}

/// Register a running foreground sub-agent task (EXK).
///
/// `abort_controller` is the token the caller already built for this agent (the
/// source's sync path makes its own here; codemason's agent tool derives a
/// child of the parent's token before this call and passes it in, so a stop on
/// this task cancels the agent that is actually running).
pub fn register_agent_foreground(
    agent_id: &str,
    description: &str,
    prompt: &str,
    selected_agent: AgentDefinition,
    tool_use_id: Option<String>,
    abort_controller: Option<CancellationToken>,
    auto_background_delay_ms: Option<u64>,
) -> AgentTaskHandle {
    crate::utils::task::disk_output::init_task_output(agent_id);

    let mut task = base_agent_task(
        agent_id, description, prompt, selected_agent,
        abort_controller, tool_use_id, false,
    );
    // Registered before the task is installed, exactly as the source does: the
    // window in which an agent exists but has no teardown registered must be
    // empty.
    let id_for_cleanup = agent_id.to_string();
    task.unregister_cleanup = Some(register_cleanup(move || {
        kill_async_agent(&id_for_cleanup);
    }));

    let signal = std::sync::Arc::new(Notify::new());
    BACKGROUND_SIGNAL_RESOLVERS.lock().insert(agent_id.to_string(), signal.clone());
    register_task(task.into());

    let delay_ms = auto_background_delay_ms.unwrap_or_else(auto_background_ms);
    let cancel = if delay_ms > 0 {
        let token = CancellationToken::new();
        let armed = token.clone();
        let id = agent_id.to_string();
        // Only inside a runtime; a spawn from a blocking context simply gets no
        // timer, as the source gets none when autoBackgroundMs is 0.
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(async move {
                tokio::select! {
                    _ = armed.cancelled() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {
                        auto_background(&id);
                    }
                }
            });
            Some(token)
        } else {
            None
        }
    } else {
        None
    };

    AgentTaskHandle {
        task_id: agent_id.to_string(),
        background_signal: signal,
        cancel_auto_background: cancel,
    }
}

/// The auto-background timer body (EXK's setTimeout callback).
fn auto_background(agent_id: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.is_backgrounded {
            return None;
        }
        let mut next = task.clone();
        next.is_backgrounded = true;
        Some(next.into())
    });
    if let Some(signal) = BACKGROUND_SIGNAL_RESOLVERS.lock().remove(agent_id) {
        signal.notify_waiters();
        // notify_one so a waiter that has not reached its await yet is still
        // woken: the signal is one-shot and nobody re-arms it.
        signal.notify_one();
    }
}

/// Register a sub-agent task that starts in the background (Fm8).
///
/// No background signal and no auto-background timer: it is already there. The
/// caller keeps the returned state so it can report the task id immediately —
/// the async spawn path answers the parent before the agent has run.
pub fn register_async_agent(
    agent_id: &str,
    description: &str,
    prompt: &str,
    selected_agent: AgentDefinition,
    tool_use_id: Option<String>,
    abort_controller: Option<CancellationToken>,
) -> LocalAgentTaskState {
    crate::utils::task::disk_output::init_task_output(agent_id);

    let mut task = base_agent_task(
        agent_id, description, prompt, selected_agent,
        abort_controller, tool_use_id, true,
    );
    let id_for_cleanup = agent_id.to_string();
    task.unregister_cleanup = Some(register_cleanup(move || {
        kill_async_agent(&id_for_cleanup);
    }));
    register_task(task.clone().into());
    task
}

/// Move a running foreground agent task to the background (lXK).
///
/// False when there is no such task or it is already backgrounded — the caller
/// uses that to tell "moved" from "nothing to move".
///
/// Also a seam with no caller here: the source reaches it from the UI's
/// "background this agent" affordance (the BackgroundHint a long foreground run
/// shows), which codemason does not draw. The timer path (auto_background, off
/// unless CODEMASON_AUTO_BACKGROUND_TASKS is set) drives the same transition,
/// and the tests use this entry directly.
pub fn background_agent_task(agent_id: &str) -> bool {
    let movable = get_app_state()
        .tasks
        .get(agent_id)
        .and_then(|t| t.as_local_agent().cloned())
        .map(|t| !t.is_backgrounded)
        .unwrap_or(false);
    if !movable {
        return false;
    }
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        let mut next = task.clone();
        next.is_backgrounded = true;
        Some(next.into())
    });
    if let Some(signal) = BACKGROUND_SIGNAL_RESOLVERS.lock().remove(agent_id) {
        signal.notify_waiters();
        signal.notify_one();
    }
    true
}

/// Drop a finished foreground agent task (LXK).
///
/// A BACKGROUNDED task is deliberately left in place: its result is collected
/// later through TaskOutput, so deleting it here is the "agent vanished before
/// anyone read it" bug. The captured handle is released after the delete,
/// matching the source's ordering — the registry no longer refers to the task
/// before its exit cleanup is cancelled.
pub fn unregister_agent_foreground(agent_id: &str) {
    BACKGROUND_SIGNAL_RESOLVERS.lock().remove(agent_id);
    // Shared, not borrowed: set_app_state takes an owned updater, so what the
    // updater captures has to outlive this frame.
    let captured: std::sync::Arc<Mutex<Option<CleanupHandle>>> =
        std::sync::Arc::new(Mutex::new(None));
    {
        let id = agent_id.to_string();
        let captured = captured.clone();
        set_app_state(move |mut prev| {
            let removable = prev
                .tasks
                .get(&id)
                .and_then(|t| t.as_local_agent())
                .filter(|t| !t.is_backgrounded)
                .map(|t| t.unregister_cleanup);
            if let Some(handle) = removable {
                *captured.lock() = handle;
                prev.tasks.remove(&id);
            }
            prev
        });
    }
    let handle = *captured.lock();
    if let Some(handle) = handle {
        unregister_cleanup(handle);
    }
}

/// Shared terminal transition (the common tail of the three settle paths).
///
/// The source also calls `evictTaskOutput(taskId)` here, which flushes and
/// drops that task's in-memory output writer. codemason writes task output
/// straight to disk and keeps no writer map, so there is nothing to evict —
/// this is a genuine no-op, not a missing call. What it DOES have to drop is
/// the transcript chain, which is per-agent state no one will read again.
fn into_terminal(task: &LocalAgentTaskState, status: &str) -> LocalAgentTaskState {
    if let Some(handle) = task.unregister_cleanup {
        unregister_cleanup(handle);
    }
    forget_agent_transcript(&task.base.id);
    let mut next = task.clone();
    next.base.status = status.to_string();
    next.base.end_time = Some(now_ms());
    next.evict_after = if task.retain { None } else { Some(now_ms() + PANEL_GRACE_MS) };
    next.abort_controller = None;
    next.unregister_cleanup = None;
    next.selected_agent = None;
    next
}

/// Cancel a running agent task and mark it killed (aq6).
///
/// The cancel happens inside the state updater, as in the source, so the task
/// is never observable as "killed but still un-cancelled".
pub fn kill_async_agent(agent_id: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        if let Some(token) = &task.abort_controller {
            token.cancel();
        }
        Some(into_terminal(task, "killed").into())
    });
}

/// Mark a running agent task completed with its result (hL8).
pub fn complete_agent_task(agent_id: &str, result: Value) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        let mut next = into_terminal(task, "completed");
        next.result = Some(result.clone());
        Some(next.into())
    });
}

/// Mark a running agent task failed with its error (SL8).
pub fn fail_agent_task(agent_id: &str, error: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        let mut next = into_terminal(task, "failed");
        next.error = Some(error.to_string());
        Some(next.into())
    });
}

/// Mark a task notified without announcing it (markAgentsNotified).
///
/// A bulk stop sends one aggregate message instead of one per agent; this is
/// what suppresses the per-agent notices without pretending they were sent.
pub fn mark_agents_notified(agent_id: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.notified {
            return None;
        }
        let mut next = task.clone();
        next.base.notified = true;
        Some(next.into())
    });
}

/// Stop every running agent task (killAllRunningAgentTasks).
///
/// Explicitly NOT what ESC does. In the source this is reached only from the
/// deliberate `chat:killAgents` gesture (and from Ctrl+C while the user is
/// looking at a teammate), under the comment "killing agents from the main
/// prompt stays a deliberate gesture (chat:killAgents), not a side-effect of
/// cancelling a turn." A background agent is meant to outlive the turn that
/// spawned it; this is the one thing that ends it in bulk.
pub fn kill_all_running_agent_tasks(tasks: &HashMap<String, crate::tasks::types::TaskState>) {
    for (agent_id, task) in tasks.iter() {
        if let Some(agent) = task.as_local_agent() {
            if agent.base.status == "running" {
                kill_async_agent(agent_id);
            }
        }
    }
}

/// The whole bulk-stop gesture (killAllAgentsAndNotify). Returns what the model
/// was told, or None when there was nothing running.
///
/// Three steps, in this order, and the order matters:
///   1. stop every running agent task,
///   2. mark each one notified — which SUPPRESSES its own kill notice,
///   3. enqueue ONE aggregate message naming them all.
/// Skipping step 2 would tell the model the same thing N+1 times; doing step 2
/// without step 3 would stop three agents and never say so.
///
/// Reads the task table itself rather than taking it as an argument: the
/// source's handler pulls `store.getState().tasks` for exactly this reason
/// ("Reads tasks from the store directly to avoid stale closures"), and a kill
/// gesture must act on what is running now.
///
/// The source also emits a per-task SDK `task_terminated` event here. There is
/// no SDK consumer in codemason — the notification queue is the whole audience
/// — so that step has no local equivalent to stand in for.
pub fn kill_all_agents_and_notify() -> Option<String> {
    let tasks = get_app_state().tasks;
    let running: Vec<(String, String)> = tasks
        .iter()
        .filter_map(|(id, task)| {
            let agent = task.as_local_agent()?;
            (agent.base.status == "running")
                .then(|| (id.clone(), agent.base.description.clone()))
        })
        .collect();
    if running.is_empty() {
        return None;
    }

    kill_all_running_agent_tasks(&tasks);

    let mut descriptions = Vec::with_capacity(running.len());
    for (agent_id, description) in running {
        mark_agents_notified(&agent_id);
        descriptions.push(description);
    }

    let summary = if descriptions.len() == 1 {
        format!(
            "Background agent \"{}\" was stopped by the user.",
            descriptions[0]
        )
    } else {
        let quoted = descriptions
            .iter()
            .map(|d| format!("\"{d}\""))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{} background agents were stopped by the user: {}.",
            descriptions.len(),
            quoted
        )
    };
    crate::utils::message_queue_manager::enqueue_pending_notification(
        &summary,
        Some("task-notification"),
    );
    Some(summary)
}

/// Unlink a backgrounded agent from the scope it was spawned under, returning
/// the token it now answers to.
///
/// For an agent that moves to the background mid-run. The source does the same
/// thing by swapping which controller the run uses: a foreground agent runs on
/// the turn's controller (so ESC ends it with the turn), and the moment it is
/// backgrounded the continuation is re-launched with `task.abortController` —
/// a controller `registerAgentForeground` built with `createAbortController()`,
/// linked to nothing. Its sibling `registerAsyncAgent` is called with no parent
/// at all, under the comment "Don't link to parent's abort controller --
/// background agents should survive when the user presses ESC to cancel the
/// main thread. They are killed explicitly via chat:killAgents."
///
/// A CancellationToken cannot be unlinked from its parent, so this mints a
/// fresh root one and stores it on the task — which is also what keeps the
/// explicit stop working, since kill_async_agent cancels whatever the task is
/// holding. The caller must run the continuation on the returned token.
pub fn detach_agent_abort(agent_id: &str) -> CancellationToken {
    let detached = CancellationToken::new();
    let stored = detached.clone();
    update_task_state(agent_id, move |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        let mut next = task.clone();
        next.abort_controller = Some(stored.clone());
        Some(next.into())
    });
    detached
}

/// Store a progress snapshot on a running agent task (updateAgentProgress).
///
/// An existing `summary` survives: it is written by the summarisation path on
/// its own schedule, and a counts update from the agent's next message must not
/// wipe prose it did not produce.
pub fn update_agent_progress(agent_id: &str, progress: AgentProgress) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        let existing_summary = task.progress.as_ref().and_then(|p| p.summary.clone());
        let mut next = task.clone();
        let mut next_progress = progress.clone();
        if existing_summary.is_some() {
            next_progress.summary = existing_summary;
        }
        next.progress = Some(next_progress);
        Some(next.into())
    });
}

/// Store a prose progress summary on a running agent (updateAgentSummary).
///
/// Keeps the counts as they are — this path only knows the prose. The source
/// additionally emits the summary to SDK consumers behind an opt-in flag;
/// codemason has no such consumer, so the store is the whole of it.
pub fn update_agent_summary(agent_id: &str, summary: &str) {
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.status != "running" {
            return None;
        }
        let previous = task.progress.clone().unwrap_or_default();
        let mut next = task.clone();
        next.progress = Some(AgentProgress {
            summary: Some(summary.to_string()),
            ..previous
        });
        Some(next.into())
    });
}

/// What a finished agent reports about its own run.
#[derive(Debug, Clone, Default)]
pub struct AgentUsage {
    pub total_tokens: i64,
    pub tool_uses: i64,
    pub duration_ms: i64,
}

/// Announce a background agent's outcome (enqueueAgentNotification).
///
/// Announced at most once, and the check-and-set is atomic for a reason: a
/// TaskStop marks the task notified to suppress the kill notice, so a
/// completion that lands at the same moment must not also be announced. The
/// loser of that race is dropped here rather than reaching the model twice.
///
/// A foreground agent needs none of this — its result is the tool's return
/// value. This is for the one that was backgrounded, where the tool call
/// returned long ago and the only way back to the conversation is the queue.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_agent_notification(
    agent_id: &str,
    description: &str,
    status: &str, // 'completed' | 'failed' | 'killed'
    error: Option<&str>,
    final_message: Option<&str>,
    usage: Option<&AgentUsage>,
    tool_use_id: Option<&str>,
    worktree_path: Option<&str>,
    worktree_branch: Option<&str>,
) {
    let should_enqueue = std::cell::Cell::new(false);
    update_task_state(agent_id, |task| {
        let task = task.as_local_agent()?;
        if task.base.notified {
            return None;
        }
        should_enqueue.set(true);
        let mut next = task.clone();
        next.base.notified = true;
        Some(next.into())
    });
    if !should_enqueue.get() {
        return;
    }

    let summary = match status {
        "completed" => format!("Agent \"{description}\" completed"),
        "failed" => format!(
            "Agent \"{description}\" failed: {}",
            error.unwrap_or("Unknown error")
        ),
        _ => format!("Agent \"{description}\" was stopped"),
    };

    let output_path = crate::utils::task::disk_output::get_task_output_path(agent_id);
    let tool_use_id_line = match tool_use_id {
        Some(id) => format!("\n<{TOOL_USE_ID_TAG}>{id}</{TOOL_USE_ID_TAG}>"),
        None => String::new(),
    };
    // The agent's answer travels IN the notification. For a backgrounded agent
    // this is the only delivery: nothing is going to call the tool back for it.
    let result_section = match final_message {
        Some(m) if !m.is_empty() => format!("\n<result>{m}</result>"),
        _ => String::new(),
    };
    let usage_section = match usage {
        Some(u) => format!(
            "\n<usage><total_tokens>{}</total_tokens><tool_uses>{}</tool_uses>\
             <duration_ms>{}</duration_ms></usage>",
            u.total_tokens, u.tool_uses, u.duration_ms
        ),
        None => String::new(),
    };
    let worktree_section = match worktree_path {
        Some(path) => {
            let branch = match worktree_branch {
                Some(b) => format!("<{WORKTREE_BRANCH_TAG}>{b}</{WORKTREE_BRANCH_TAG}>"),
                None => String::new(),
            };
            format!(
                "\n<{WORKTREE_TAG}><{WORKTREE_PATH_TAG}>{path}</{WORKTREE_PATH_TAG}>\
                 {branch}</{WORKTREE_TAG}>"
            )
        }
        None => String::new(),
    };

    let message = format!(
        "<{TASK_NOTIFICATION_TAG}>\n\
         <{TASK_ID_TAG}>{agent_id}</{TASK_ID_TAG}>{tool_use_id_line}\n\
         <{OUTPUT_FILE_TAG}>{}</{OUTPUT_FILE_TAG}>\n\
         <{STATUS_TAG}>{status}</{STATUS_TAG}>\n\
         <{SUMMARY_TAG}>{summary}</{SUMMARY_TAG}>\
         {result_section}{usage_section}{worktree_section}\n\
         </{TASK_NOTIFICATION_TAG}>",
        output_path.to_string_lossy(),
    );
    crate::utils::message_queue_manager::enqueue_pending_notification(
        &message,
        Some("task-notification"),
    );
}

/// Ties a spawned agent task to the scope that owns it.
///
/// Rust has no `finally`, and a stream can also be dropped mid-flight when the
/// consumer stops polling it — the exact shape of "the agent was abandoned".
/// Drop therefore does what the source's finally does, in the same order:
/// stop the auto-background timer, settle the task if it is somehow still
/// running (`kill_async_agent` is a no-op once it has completed or failed), then
/// remove it — which a BACKGROUNDED task survives, because its result is
/// collected later.
pub struct AgentTaskGuard {
    agent_id: String,
    cancel_auto_background: Option<CancellationToken>,
}

impl AgentTaskGuard {
    pub fn new(agent_id: &str, cancel_auto_background: Option<CancellationToken>) -> Self {
        Self { agent_id: agent_id.to_string(), cancel_auto_background }
    }
}

impl Drop for AgentTaskGuard {
    fn drop(&mut self) {
        if let Some(token) = self.cancel_auto_background.take() {
            token.cancel();
        }
        // A BACKGROUNDED task is not this scope's to settle. Its run outlives
        // the tool call by design and a detached driver will finish it, so
        // killing it here would stop an agent that is working as asked.
        // `unregister_agent_foreground` already refuses to remove one; the kill
        // has to make the same exception.
        let backgrounded = get_app_state()
            .tasks
            .get(&self.agent_id)
            .and_then(|t| t.as_local_agent())
            .map(|a| a.is_backgrounded)
            .unwrap_or(false);
        if backgrounded {
            return;
        }
        kill_async_agent(&self.agent_id);
        unregister_agent_foreground(&self.agent_id);
    }
}

/// The Task implementation registered for type 'local_agent'.
pub struct LocalAgentTask;

#[async_trait]
impl Task for LocalAgentTask {
    fn name(&self) -> &'static str {
        "LocalAgentTask"
    }
    fn task_type(&self) -> &'static str {
        "local_agent"
    }
    async fn kill(&self, task_id: &str) {
        kill_async_agent(task_id);
    }
}

pub static LOCAL_AGENT_TASK: Lazy<LocalAgentTask> = Lazy::new(|| LocalAgentTask);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::task::is_terminal_task_status;
    use crate::tasks::registry::get_task_by_type;
    use crate::utils::agent_id::{create_agent_id, is_agent_id};
    use crate::utils::task::framework::register_task;
    use crate::utils::cleanup_registry::{cleanup_function_count, run_cleanup_functions};
    use std::sync::Arc;

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

    fn spawn(token: &CancellationToken) -> (String, AgentTaskHandle) {
        let id = create_agent_id(None);
        let handle = register_agent_foreground(
            &id, "look around", "find the thing", fake_agent(),
            Some("toolu_1".into()), Some(token.clone()), None,
        );
        (id, handle)
    }

    fn task_of(id: &str) -> Option<LocalAgentTaskState> {
        get_app_state().tasks.get(id).and_then(|t| t.as_local_agent().cloned())
    }

    /// A resumed agent gets its images back, not store references
    /// (tests/test_vision_followups.py test_agent_transcript_hydrated).
    #[test]
    fn transcript_images_come_back() {
        const B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";
        let agent = format!("agent-{}", uuid::Uuid::new_v4().simple());
        let msg = serde_json::json!({"type": "user", "uuid": "u1", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}}]}]}});
        append_agent_transcript(&agent, &msg);
        let back = get_agent_transcript(&agent);
        forget_agent_transcript(&agent);
        let _ = std::fs::remove_file(crate::utils::task::disk_output::get_task_output_path(&agent));
        let back = back.expect("transcript");
        let src = &back[0]["message"]["content"][0]["content"][0]["source"];
        assert_eq!(src["type"], "base64", "{src}");
        assert_eq!(src["data"], B64);
    }

    #[test]
    fn a_spawn_registers_a_running_task_with_its_teardown_in_place() {
        let _guard = test_lock();
        reset_app_state();
        let before = cleanup_function_count();
        let parent = CancellationToken::new();
        let (id, handle) = spawn(&parent);

        let t = task_of(&id).expect("the agent is registered as a local_agent task");
        assert!(is_local_agent_task(&t));
        assert_eq!(t.base.status, "running", "a pending task could never settle");
        assert_eq!(t.base.id, id, "the task id IS the agent id");
        assert_eq!(handle.task_id, id);
        assert_eq!(is_agent_id(&id), Some(id.as_str()));
        assert_eq!(t.agent_type, "Explore");
        assert!(!t.is_backgrounded && t.pending_messages.is_empty());
        assert!(!t.retain, "nothing is watching it");
        assert_eq!(cleanup_function_count(), before + 1, "spawn registers the exit kill");
        assert!(t.unregister_cleanup.is_some(), "and the task holds its handle");
        assert!(handle.cancel_auto_background.is_none(), "no timer by default");
        assert_eq!(auto_background_ms(), 0);

        unregister_agent_foreground(&id);
    }

    #[test]
    fn completion_settles_the_task_and_releases_the_teardown() {
        let _guard = test_lock();
        reset_app_state();
        let before = cleanup_function_count();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);

        complete_agent_task(&id, serde_json::json!({"text": "found it"}));
        let t = task_of(&id).unwrap();
        assert_eq!(t.base.status, "completed");
        assert!(is_terminal_task_status(&t.base.status));
        assert_eq!(t.result.as_ref().unwrap()["text"], "found it");
        let deadline = t.evict_after.expect("a finished task gets an eviction deadline");
        assert!(deadline - t.base.end_time.unwrap() <= PANEL_GRACE_MS);
        assert!(t.abort_controller.is_none() && t.unregister_cleanup.is_none());
        assert!(t.selected_agent.is_none(), "a finished task stops pinning the definition");
        assert_eq!(cleanup_function_count(), before, "the exit kill is released");

        // A settled task cannot be re-settled.
        fail_agent_task(&id, "too late");
        kill_async_agent(&id);
        assert_eq!(task_of(&id).unwrap().base.status, "completed");

        unregister_agent_foreground(&id);
        assert!(task_of(&id).is_none(), "a finished foreground task is removed");
    }

    #[test]
    fn a_backgrounded_task_survives_removal() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);

        assert!(background_agent_task(&id), "a running agent moves to the background");
        assert!(task_of(&id).unwrap().is_backgrounded);
        assert!(!background_agent_task(&id), "moving it twice reports nothing to do");

        complete_agent_task(&id, serde_json::json!({"text": "bg done"}));
        unregister_agent_foreground(&id);
        let t = task_of(&id).expect("its result is collected later through TaskOutput");
        assert_eq!(t.base.status, "completed");
        reset_app_state();
    }

    #[test]
    fn the_async_spawn_starts_backgrounded_and_outlives_the_turn() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let id = create_agent_id(None);
        // What AgentTool now passes for a background spawn: a root token, not a
        // child of the turn's. The source says why where it calls this —
        // "Don't link to parent's abort controller -- background agents should
        // survive when the user presses ESC to cancel the main thread. They are
        // killed explicitly via chat:killAgents." This test used to assert the
        // opposite, which is exactly the bug it was pinning in place.
        let token = CancellationToken::new();
        let state = register_async_agent(
            &id, "async one", "go", fake_agent(), None, Some(token.clone()),
        );
        assert!(state.is_backgrounded);
        assert!(task_of(&id).unwrap().is_backgrounded);
        assert!(!token.is_cancelled());
        parent.cancel(); // ESC on the main thread
        assert!(!token.is_cancelled(), "a cancelled turn does not reach it");
        kill_async_agent(&id);
        assert!(token.is_cancelled(), "an explicit stop does");
        reset_app_state();
    }

    #[test]
    fn a_stop_cancels_that_agent_and_spares_its_sibling() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let a_token = parent.child_token();
        let b_token = parent.child_token();
        let a = create_agent_id(None);
        let b = create_agent_id(None);
        register_agent_foreground(&a, "a", "a", fake_agent(), None, Some(a_token.clone()), None);
        register_agent_foreground(&b, "b", "b", fake_agent(), None, Some(b_token.clone()), None);

        kill_async_agent(&a);
        assert!(a_token.is_cancelled());
        assert_eq!(task_of(&a).unwrap().base.status, "killed");
        assert!(!b_token.is_cancelled(), "its sibling keeps running");
        assert_eq!(task_of(&b).unwrap().base.status, "running");
        assert!(!parent.is_cancelled(), "and so does the parent");

        unregister_agent_foreground(&a);
        unregister_agent_foreground(&b);
        reset_app_state();
    }

    #[tokio::test]
    async fn the_registry_knows_how_to_stop_an_agent() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        let impl_ = get_task_by_type("local_agent").expect("local_agent is registered");
        assert_eq!(impl_.name(), "LocalAgentTask");
        impl_.kill(&id).await;
        assert_eq!(task_of(&id).unwrap().base.status, "killed");
        unregister_agent_foreground(&id);
        reset_app_state();
    }

    #[test]
    fn a_process_dying_mid_run_kills_the_agent_it_owns() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        let token = task_of(&id).unwrap().abort_controller.unwrap();
        run_cleanup_functions();
        assert_eq!(task_of(&id).unwrap().base.status, "killed");
        assert!(token.is_cancelled());
        unregister_agent_foreground(&id);
        reset_app_state();
    }

    #[test]
    fn a_retained_task_gets_no_eviction_deadline() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        update_task_state(&id, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.retain = true;
            Some(next.into())
        });
        complete_agent_task(&id, serde_json::json!({"text": "watched"}));
        assert!(task_of(&id).unwrap().evict_after.is_none());
        unregister_agent_foreground(&id);
        reset_app_state();
    }

    #[test]
    fn eviction_waits_for_the_deadline() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        complete_agent_task(&id, serde_json::json!({"text": "x"}));

        let set = |f: &dyn Fn(&mut LocalAgentTaskState)| {
            update_task_state(&id, |t| {
                let mut next = t.as_local_agent()?.clone();
                f(&mut next);
                Some(next.into())
            });
        };

        set(&|t| t.evict_after = Some(1));
        crate::utils::task::framework::evict_terminal_task(&id);
        assert!(task_of(&id).is_some(), "an un-notified completion is never dropped");

        set(&|t| t.base.notified = true);
        set(&|t| t.evict_after = None);
        crate::utils::task::framework::evict_terminal_task(&id);
        assert!(task_of(&id).is_some(), "no deadline means retained — never evict");

        set(&|t| t.evict_after = Some(now_ms() + 10_000));
        crate::utils::task::framework::evict_terminal_task(&id);
        assert!(task_of(&id).is_some(), "kept until its deadline passes");

        set(&|t| t.evict_after = Some(1));
        crate::utils::task::framework::evict_terminal_task(&id);
        assert!(task_of(&id).is_none(), "and evicted once it has");
        reset_app_state();
    }

    #[tokio::test]
    async fn the_auto_background_timer_fires_and_cancels() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let id = create_agent_id(None);
        let handle = register_agent_foreground(
            &id, "slow", "slow", fake_agent(), None, Some(parent.child_token()), Some(20),
        );
        assert!(handle.cancel_auto_background.is_some(), "an armed timer hands back its canceller");
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert!(task_of(&id).unwrap().is_backgrounded, "the timer backgrounds the agent");
        unregister_agent_foreground(&id);

        let id2 = create_agent_id(None);
        let handle2 = register_agent_foreground(
            &id2, "slow", "slow", fake_agent(), None, Some(parent.child_token()), Some(20),
        );
        handle2.cancel_auto_background.as_ref().unwrap().cancel();
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert!(!task_of(&id2).unwrap().is_backgrounded, "a cancelled timer leaves it in the foreground");
        unregister_agent_foreground(&id2);
        reset_app_state();
    }

    /// The guard is what stands in for the source's `finally`, and it has to
    /// cover the case Python's finally also covers: the agent is abandoned
    /// without ever settling.
    #[test]
    fn the_guard_settles_and_removes_an_abandoned_agent() {
        let _guard = test_lock();
        reset_app_state();
        let before = cleanup_function_count();
        let parent = CancellationToken::new();
        let (id, handle) = spawn(&parent);
        let token = task_of(&id).unwrap().abort_controller.unwrap();
        {
            let _task_guard = AgentTaskGuard::new(&id, handle.cancel_auto_background.clone());
        }
        assert!(task_of(&id).is_none(), "dropped: the task is gone");
        assert!(token.is_cancelled(), "and the agent was told to stop");
        assert_eq!(cleanup_function_count(), before, "with nothing left registered");
        reset_app_state();
    }

    #[test]
    fn the_guard_leaves_a_completed_task_completed() {
        let _guard = test_lock();
        reset_app_state();
        let before = cleanup_function_count();
        let parent = CancellationToken::new();
        let (id, handle) = spawn(&parent);
        complete_agent_task(&id, serde_json::json!({"text": "done"}));
        {
            let _task_guard = AgentTaskGuard::new(&id, handle.cancel_auto_background.clone());
        }
        assert!(task_of(&id).is_none(), "removed on the way out");
        assert_eq!(cleanup_function_count(), before);
        reset_app_state();
    }

    fn assistant(inp: i64, out: i64, tools: &[&str], cache_read: i64) -> Message {
        use crate::types::messages::{ApiMessage, BetaUsage, ToolUseBlock};
        use crate::types::query::AssistantMessage;
        Message::Assistant(AssistantMessage {
            message: ApiMessage {
                usage: BetaUsage {
                    input_tokens: inp,
                    output_tokens: out,
                    cache_read_input_tokens: cache_read,
                    cache_creation_input_tokens: 0,
                },
                content: tools
                    .iter()
                    .map(|t| {
                        ContentBlock::ToolUse(ToolUseBlock {
                            id: String::new(),
                            name: (*t).to_string(),
                            input: serde_json::json!({}),
                        })
                    })
                    .collect(),
                ..Default::default()
            },
            uuid: String::new(),
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    #[test]
    fn the_prompt_count_is_the_latest_and_completions_are_summed() {
        let mut tracker = create_progress_tracker();
        update_progress_from_message(&mut tracker, &assistant(100, 10, &["Read"], 0), None);
        update_progress_from_message(&mut tracker, &assistant(250, 20, &["Grep"], 50), None);
        assert_eq!(
            tracker.latest_input_tokens, 300,
            "the prompt already includes the turns before it"
        );
        assert_eq!(tracker.cumulative_output_tokens, 30);
        assert_eq!(get_token_count_from_tracker(&tracker), 330);
        assert_eq!(tracker.tool_use_count, 2);
    }

    #[test]
    fn a_non_assistant_message_changes_nothing() {
        let mut tracker = create_progress_tracker();
        update_progress_from_message(&mut tracker, &assistant(100, 10, &["Read"], 0), None);
        let before = get_token_count_from_tracker(&tracker);
        update_progress_from_message(
            &mut tracker,
            &Message::User(crate::utils::messages::create_user_message_text("hi")),
            None,
        );
        assert_eq!(get_token_count_from_tracker(&tracker), before);
        assert_eq!(tracker.tool_use_count, 1);
    }

    #[test]
    fn the_activity_list_is_capped() {
        let mut tracker = create_progress_tracker();
        for i in 0..(MAX_RECENT_ACTIVITIES + 3) {
            let name = format!("T{i}");
            update_progress_from_message(&mut tracker, &assistant(300, 0, &[&name], 0), None);
        }
        assert_eq!(tracker.recent_activities.len(), MAX_RECENT_ACTIVITIES);
        let snap = get_progress_update(&tracker);
        assert_eq!(
            snap.last_activity.unwrap().tool_name,
            format!("T{}", MAX_RECENT_ACTIVITIES + 2)
        );
    }

    #[test]
    fn progress_and_summary_do_not_clobber_each_other() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);

        update_agent_summary(&id, "reading the config");
        update_agent_progress(
            &id,
            AgentProgress { tool_use_count: 7, token_count: 1234, ..Default::default() },
        );
        let p = task_of(&id).unwrap().progress.unwrap();
        assert_eq!(p.summary.as_deref(), Some("reading the config"),
                   "a counts update keeps a summary it did not write");
        assert_eq!((p.tool_use_count, p.token_count), (7, 1234));

        update_agent_summary(&id, "now editing");
        let p = task_of(&id).unwrap().progress.unwrap();
        assert_eq!((p.tool_use_count, p.token_count), (7, 1234),
                   "a summary update keeps the counts it did not write");
        assert_eq!(p.summary.as_deref(), Some("now editing"));

        complete_agent_task(&id, serde_json::json!({"text": "x"}));
        update_agent_progress(&id, AgentProgress { tool_use_count: 99, ..Default::default() });
        update_agent_summary(&id, "too late");
        let p = task_of(&id).unwrap().progress.unwrap();
        assert_eq!(p.tool_use_count, 7, "a finished agent takes no more progress");
        assert_eq!(p.summary.as_deref(), Some("now editing"));

        unregister_agent_foreground(&id);
        reset_app_state();
    }

    #[test]
    fn a_bulk_stop_reaches_every_running_agent() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for _ in 0..3 {
            let id = create_agent_id(None);
            let token = parent.child_token();
            register_agent_foreground(
                &id, "x", "x", fake_agent(), None, Some(token.clone()), None,
            );
            ids.push(id);
            tokens.push(token);
        }
        kill_all_running_agent_tasks(&get_app_state().tasks);
        for id in &ids {
            assert_eq!(task_of(id).unwrap().base.status, "killed");
        }
        assert!(tokens.iter().all(|t| t.is_cancelled()));
        assert!(!parent.is_cancelled(), "without stopping the parent");
        reset_app_state();
    }

    /// The gesture ESC deliberately is not: stop them all, say it once.
    ///
    /// ESC cancels the turn and leaves background agents running on purpose
    /// ("killing agents from the main prompt stays a deliberate gesture
    /// (chat:killAgents), not a side-effect of cancelling a turn"), so this is
    /// the path that ends them — and it has to suppress the per-agent notices
    /// it would otherwise race, or the model hears the same news four times.
    #[test]
    fn the_bulk_gesture_stops_everything_and_reports_once() {
        let _guard = test_lock();
        reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();

        assert!(kill_all_agents_and_notify().is_none(), "nothing running, nothing to say");
        assert!(
            crate::utils::message_queue_manager::drain_pending_notifications().is_empty(),
            "and nothing said"
        );

        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for description in ["read the lane", "count the blocks", "write it up"] {
            let id = create_agent_id(None);
            // Unlinked, which is how a background agent is registered.
            let token = CancellationToken::new();
            register_async_agent(
                &id, description, "p", fake_agent(), None, Some(token.clone()),
            );
            ids.push(id);
            tokens.push(token);
        }

        let summary = kill_all_agents_and_notify().expect("three agents were running");
        assert!(tokens.iter().all(|t| t.is_cancelled()), "every scope stopped");
        for id in &ids {
            let task = task_of(id).unwrap();
            assert_eq!(task.base.status, "killed");
            assert!(task.base.notified, "its own kill notice is suppressed");
        }
        let notes = crate::utils::message_queue_manager::drain_pending_notifications();
        assert_eq!(notes.len(), 1, "told once, not three times: {notes:?}");
        assert_eq!(notes[0], summary);
        assert!(notes[0].starts_with("3 background agents"), "{}", notes[0]);
        for description in ["read the lane", "count the blocks", "write it up"] {
            assert!(notes[0].contains(&format!("\"{description}\"")), "{}", notes[0]);
        }

        reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();
        let only = create_agent_id(None);
        register_async_agent(
            &only, "the only one", "p", fake_agent(), None,
            Some(CancellationToken::new()),
        );
        kill_all_agents_and_notify();
        let notes = crate::utils::message_queue_manager::drain_pending_notifications();
        assert_eq!(
            notes[0],
            "Background agent \"the only one\" was stopped by the user.",
            "one agent gets the singular phrasing"
        );
        reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();
    }

    /// Moving an agent to the background cuts it loose from the turn's scope —
    /// and leaves the explicit stop working, because the task holds the new one.
    #[test]
    fn detaching_an_agent_survives_the_turns_cancel() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let id = create_agent_id(None);
        let linked = parent.child_token();
        register_agent_foreground(
            &id, "survey", "p", fake_agent(), None, Some(linked.clone()), None,
        );

        let detached = detach_agent_abort(&id);
        parent.cancel(); // ESC
        assert!(linked.is_cancelled(), "the scope it was spawned under is gone");
        assert!(!detached.is_cancelled(), "the one it now runs on is not");

        let stored = task_of(&id).unwrap().abort_controller.unwrap();
        assert!(!stored.is_cancelled());
        kill_async_agent(&id);
        assert!(detached.is_cancelled(), "an explicit stop still ends it");

        // A finished agent is not re-tokened: the continuation is over, and
        // handing it a live token would say otherwise.
        reset_app_state();
        let id2 = create_agent_id(None);
        register_async_agent(&id2, "done", "p", fake_agent(), None, Some(CancellationToken::new()));
        complete_agent_task(&id2, serde_json::json!({"text": "x"}));
        let before = task_of(&id2).unwrap().abort_controller;
        let _ = detach_agent_abort(&id2);
        let after = task_of(&id2).unwrap().abort_controller;
        assert_eq!(before.is_none(), after.is_none(), "a terminal task is left alone");
        reset_app_state();
    }

    #[test]
    fn marking_notified_is_idempotent() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        mark_agents_notified(&id);
        assert!(task_of(&id).unwrap().base.notified);
        mark_agents_notified(&id);
        assert!(task_of(&id).unwrap().base.notified);
        unregister_agent_foreground(&id);
        reset_app_state();
    }

    #[test]
    fn the_main_session_is_not_a_panel_agent() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        assert!(is_panel_agent_task(&task_of(&id).unwrap()));
        update_task_state(&id, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.agent_type = "main-session".into();
            Some(next.into())
        });
        let t = task_of(&id).unwrap();
        assert!(!is_panel_agent_task(&t), "the main session is not a sub-agent");
        assert!(is_local_agent_task(&t), "though it is still a local_agent task");
        unregister_agent_foreground(&id);
        reset_app_state();
    }

    /// A resume registers a fresh task over the old one. What the UI and the
    /// message history own has to survive that, or a queued message is dropped
    /// and the user's hold is reset.
    #[test]
    fn re_registering_carries_the_held_state_forward() {
        let _guard = test_lock();
        reset_app_state();
        let parent = CancellationToken::new();
        let (id, _handle) = spawn(&parent);
        update_task_state(&id, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.retain = true;
            next.base.start_time = 4242;
            next.pending_messages = vec!["keep me".to_string()];
            next.messages = Some(vec![serde_json::json!("transcript")]);
            next.disk_loaded = true;
            Some(next.into())
        });

        let mut replacement = LocalAgentTaskState {
            base: crate::task::create_task_state_base(&id, "local_agent", "resumed", None),
            agent_id: id.clone(),
            prompt: "resumed".into(),
            agent_type: "Explore".into(),
            ..Default::default()
        };
        replacement.base.status = "running".into();
        register_task(replacement.into());

        let t = task_of(&id).unwrap();
        assert!(t.retain, "the hold the user placed");
        assert_eq!(t.base.start_time, 4242, "the original start time");
        assert_eq!(t.pending_messages, vec!["keep me".to_string()],
                   "a queued message, which would otherwise be dropped");
        assert_eq!(t.messages, Some(vec![serde_json::json!("transcript")]));
        assert!(t.disk_loaded);
        assert_eq!(t.base.description, "resumed", "while taking the replacement's own fields");
        reset_app_state();
    }

    /// What the tool actually wires up.
    #[test]
    fn the_agent_tool_drives_the_lifecycle() {
        let src = include_str!("../tools/agent_tool.rs");
        for needle in [
            "register_agent_foreground(",
            "complete_agent_task(",
            "kill_async_agent(",
            "AgentTaskGuard::new(",
            "create_agent_id(None)",
            "create_progress_tracker()",
            "update_progress_from_message(",
            "update_agent_progress(",
        ] {
            assert!(src.contains(needle), "agent_tool should call {needle}");
        }
        // rfind, not find: the background branch above also acquires a slot,
        // and the claim here is about the FOREGROUND path's ordering.
        assert!(
            src.find("register_agent_foreground(").unwrap()
                < src.rfind("subagent_gate::try_acquire").unwrap(),
            "the task is registered BEFORE the slot wait, or a queued agent is unstoppable"
        );
        assert!(
            src.find("update_agent_progress(").unwrap() < src.find("complete_agent_task(").unwrap(),
            "progress is published while the agent runs, not only once it is done"
        );
    }

    /// Every tool says what it is doing, in words.
    ///
    /// The resolver has always looked for a tool's activity description — and
    /// no tool defined one, and all three production callers passed None, so
    /// `progress.recent_activities[].activity_description` was permanently
    /// empty. The field existed, the plumbing existed, and there was nothing at
    /// either end of it.
    ///
    /// This is what a parent watching a sub-agent reads. "Reading src/foo.rs"
    /// and "Running the test suite" are the difference between knowing what an
    /// agent is doing and knowing only which tool it picked.
    #[test]
    fn the_tally_records_what_each_tool_use_was_doing() {
        use crate::utils::formatting::TOOL_SUMMARY_MAX_LENGTH;

        let tools = crate::tools::all_tools();
        let by_name = |n: &str| {
            tools
                .iter()
                .find(|t| t.name() == n)
                .cloned()
                .unwrap_or_else(|| panic!("{n} is not in the pool"))
        };

        // The verbs and the fallbacks are the source's, tool by tool.
        let cases: &[(&str, serde_json::Value, &str, &str)] = &[
            ("Read", serde_json::json!({"file_path": "/etc/hosts"}),
             "Reading /etc/hosts", "Reading file"),
            ("Write", serde_json::json!({"file_path": "/etc/hosts"}),
             "Writing /etc/hosts", "Writing file"),
            ("Edit", serde_json::json!({"file_path": "/etc/hosts"}),
             "Editing /etc/hosts", "Editing file"),
            ("Glob", serde_json::json!({"pattern": "**/*.rs"}),
             "Finding **/*.rs", "Finding files"),
            ("Grep", serde_json::json!({"pattern": "TODO"}),
             "Searching for TODO", "Searching"),
        ];
        for (name, input, expected, fallback) in cases {
            let tool = by_name(name);
            assert_eq!(
                tool.get_activity_description(Some(input)).as_deref(),
                Some(*expected),
                "{name} describes its call"
            );
            assert_eq!(
                tool.get_activity_description(Some(&serde_json::json!({}))).as_deref(),
                Some(*fallback),
                "{name} falls back without its argument"
            );
            assert_eq!(
                tool.get_activity_description(None).as_deref(),
                Some(*fallback),
                "{name} survives a missing input"
            );
        }

        // Bash prefers the model's own description; Agent adds no verb at all.
        let bash = by_name("Bash");
        assert_eq!(
            bash.get_activity_description(Some(&serde_json::json!({
                "command": "bun test", "description": "run the test suite"
            })))
            .as_deref(),
            Some("Running run the test suite")
        );
        assert_eq!(
            bash.get_activity_description(Some(&serde_json::json!({"command": "bun test"})))
                .as_deref(),
            Some("Running bun test")
        );
        assert_eq!(
            bash.get_activity_description(Some(&serde_json::json!({}))).as_deref(),
            Some("Running command")
        );
        let agent = by_name("Agent");
        assert_eq!(
            agent
                .get_activity_description(Some(&serde_json::json!({"description": "survey the lane"})))
                .as_deref(),
            Some("survey the lane")
        );
        assert_eq!(
            agent.get_activity_description(Some(&serde_json::json!({}))).as_deref(),
            Some("Running task")
        );

        // A pasted 200-char pattern must not push the spinner off the screen.
        let long = "x".repeat(200);
        let cut = by_name("Grep")
            .get_tool_use_summary(Some(&serde_json::json!({"pattern": long})))
            .expect("a pattern resolves");
        assert!(
            crate::utils::formatting::display_width(&cut) <= TOOL_SUMMARY_MAX_LENGTH,
            "cut to at most the summary limit, in COLUMNS: {cut}"
        );
        assert!(cut.ends_with('…'), "and marked as cut: {cut}");

        // And the whole point: the tally carries the words, not just the name.
        let resolve = create_activity_description_resolver(tools.clone());
        assert_eq!(
            resolve("Read", &serde_json::json!({"file_path": "/etc/hosts"})).as_deref(),
            Some("Reading /etc/hosts")
        );
        assert_eq!(
            resolve("NoSuchTool", &serde_json::json!({})),
            None,
            "a name outside the pool resolves to None, it does not panic"
        );

        let mut tracker = create_progress_tracker();
        let msg = Message::Assistant(crate::types::query::AssistantMessage {
            uuid: String::new(),
            message: crate::types::messages::ApiMessage {
                content: vec![
                    ContentBlock::ToolUse(crate::types::messages::ToolUseBlock {
                        id: "tu_0".into(),
                        name: "Read".into(),
                        input: serde_json::json!({"file_path": "/etc/hosts"}),
                    }),
                    ContentBlock::ToolUse(crate::types::messages::ToolUseBlock {
                        id: "tu_1".into(),
                        name: "Bash".into(),
                        input: serde_json::json!({
                            "command": "bun test", "description": "run the test suite"
                        }),
                    }),
                ],
                usage: crate::types::messages::BetaUsage {
                    input_tokens: 10,
                    output_tokens: 1,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
                ..Default::default()
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        });
        update_progress_from_message(&mut tracker, &msg, Some(&resolve));
        let descs: Vec<Option<&str>> = tracker
            .recent_activities
            .iter()
            .map(|a| a.activity_description.as_deref())
            .collect();
        assert_eq!(
            descs,
            vec![Some("Reading /etc/hosts"), Some("Running run the test suite")],
            "the field that was permanently None before"
        );

        // And the prose line the compaction attachment reads: the most recent
        // action, walked backwards so a tool with no description does not blank
        // it. codemason derives this instead of forking the agent every 30s to
        // ask the model for it — see derive_agent_summary.
        assert_eq!(
            derive_agent_summary(&tracker).as_deref(),
            Some("Running run the test suite")
        );
        assert_eq!(
            derive_agent_summary(&create_progress_tracker()),
            None,
            "nothing done yet means no line, not an empty one"
        );
    }

    /// TaskOutput on a RUNNING sub-agent has to return what it is doing.
    ///
    /// The source points the task's output path at the agent's transcript
    /// JSONL and prefers the in-memory result only once the run is over. This
    /// side had the reader and the preference but no writer, so the path stayed
    /// an empty file for the whole life of the agent — a parent could read an
    /// agent's counts and never one word of its work.
    #[test]
    fn a_running_agent_writes_its_transcript_where_task_output_reads_it() {
        let _guard = test_lock();
        reset_app_state();
        let id = create_agent_id(None);
        crate::utils::task::disk_output::cleanup_task_output(&id);
        forget_agent_transcript(&id);
        register_async_agent(&id, "survey the lane", "survey the lane", fake_agent(), None, None);

        assert_eq!(
            crate::utils::task::disk_output::get_task_output(&id, crate::utils::task::disk_output::DEFAULT_MAX_READ_BYTES),
            "",
            "a freshly spawned agent has an output file, empty"
        );

        let msgs = [
            serde_json::json!({
                "type": "assistant", "uuid": "u-1",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "reading the lane config"},
                    {"type": "tool_use", "id": "tu_0", "name": "Grep", "input": {}}
                ]},
            }),
            serde_json::json!({
                "type": "user", "uuid": "u-2",
                "message": {"role": "user", "content": "tool said: three lanes"},
            }),
            // Not a transcript entry, here as in the source.
            serde_json::json!({"type": "progress", "uuid": "u-3", "data": {}}),
        ];
        for m in &msgs {
            append_agent_transcript(&id, m);
        }

        let body = crate::utils::task::disk_output::get_task_output(&id, crate::utils::task::disk_output::DEFAULT_MAX_READ_BYTES);
        let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 2, "progress is not written: {body}");
        let parsed: Vec<serde_json::Value> = lines
            .iter()
            .map(|l| serde_json::from_str(l).expect("each line is JSONL"))
            .collect();
        assert_eq!(parsed[0]["type"], "assistant");
        assert_eq!(parsed[1]["type"], "user");
        assert!(
            body.contains("reading the lane config") && body.contains("Grep"),
            "the file carries the work, not just a status: {body}"
        );
        assert_eq!(parsed[0]["isSidechain"], true, "a sub-agent IS a sidechain");
        assert_eq!(parsed[0]["agentId"], id.as_str());
        assert!(parsed[0]["parentUuid"].is_null(), "the first line has no parent");
        assert_eq!(
            parsed[1]["parentUuid"], parsed[0]["uuid"],
            "each entry names the one before it"
        );

        // Finishing drops the chain: it is per-agent state nobody reads again.
        complete_agent_task(&id, serde_json::json!({"text": "lane 3"}));
        assert!(
            !AGENT_TRANSCRIPT_CHAINS.lock().contains_key(&id),
            "the chain is dropped on the terminal transition"
        );
        forget_agent_transcript(&id); // idempotent
        crate::utils::task::disk_output::cleanup_task_output(&id);
        reset_app_state();
    }
}
