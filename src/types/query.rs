//! Query-related type definitions for CodeMason
//! Equivalent to codemason/types/query.py (src/query.ts types)

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::messages::{ApiMessage, MessageParam};
use crate::tools::base::Tool;

// === Response Types ===

#[derive(Debug, Clone)]
pub struct Response {
    pub cost_usd: f64,
    pub response: String,
}

// === Message Types ===

/// Full tool use result with data and assistant message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullToolUseResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_for_assistant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// User message type. In the Python port user messages were plain dicts built
/// by create_user_message; this struct is the same JSON shape, with `isMeta`
/// kept under its original key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    // NOTE: no `type` field here — the `Message` enum owns the "type" tag.
    // serde's internally-tagged enums CONSUME the tag and do not hand it to
    // the variant struct, so a duplicate `#[serde(rename="type")]` field made
    // every real transcript entry fail with "missing field `type`" — /resume
    // and -c silently restored ZERO messages.
    pub uuid: String,
    pub message: MessageParam,
    #[serde(rename = "isMeta", default, skip_serializing_if = "std::ops::Not::not")]
    pub is_meta: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_result: Option<FullToolUseResult>,
}

/// Assistant message type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    // no `type` field — the enum tag owns it (see UserMessage note)
    pub uuid: String,
    pub message: ApiMessage,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub duration_ms: i64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_api_error_message: bool,
}

/// Progress message type — a tool's live progress line, forwarded to the CLI
/// for display and never persisted to the API history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMessage {
    // no `type` field — the enum tag owns it (see UserMessage note)
    #[serde(default)]
    pub result_for_assistant: String,
    #[serde(default)]
    pub data: Value,
}

/// Union type for all conversation message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "progress")]
    Progress(ProgressMessage),
}

impl Message {
    pub fn message_type(&self) -> &'static str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::Progress(_) => "progress",
        }
    }

    pub fn uuid(&self) -> &str {
        match self {
            Message::User(m) => &m.uuid,
            Message::Assistant(m) => &m.uuid,
            Message::Progress(_) => "",
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(m) => Some(m),
            _ => None,
        }
    }
}

// === Read File State ===

/// A readFileState entry (fileStateCache.ts). Recorded by Read, consumed by
/// Edit/Write/NotebookEdit for read-before-edit + modified-since-read, and by
/// Read itself for same-range dedup. See the Python reference for the full
/// field-by-field notes (is_partial_view semantics included).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReadFileState {
    /// file mtime (seconds) at read time.
    pub timestamp: f64,
    /// the file text at read time (stored for FULL reads; "" otherwise), used
    /// for the modified-since-read content-equality fallback.
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub offset: Option<i64>,
    #[serde(default)]
    pub limit: Option<i64>,
    /// true ONLY for auto-injected memory/attachment state whose content may
    /// differ from disk; codemason has no such auto-injection, so it is always
    /// false here — kept so the read-before-edit check stays faithful.
    #[serde(default)]
    pub is_partial_view: bool,
}

impl ReadFileState {
    pub fn is_full_read(&self) -> bool {
        self.offset.is_none() && self.limit.is_none()
    }
}

// === Tool Use Context ===

/// The typed slice of the Python ToolUseContext.options dict (keys actually
/// read anywhere: 'tools', 'dangerously_skip_permissions',
/// 'slow_and_capable_model').
#[derive(Clone, Default)]
pub struct ContextOptions {
    pub tools: Vec<Arc<dyn Tool>>,
    pub dangerously_skip_permissions: bool,
    pub slow_and_capable_model: Option<String>,
}

/// Interactive REPL only: the ReplInputRouter owning the persistent prompt
/// session. Anything needing user input mid-turn (permission prompts) must
/// route through it — stdin belongs to the session. None in --plain and -p mode.
pub trait InputRouter: Send + Sync {
    /// Prompt the user for a line mid-turn (permission prompts, questions).
    fn prompt_line(&self, prompt: &str) -> Option<String>;
    /// A steering line was injected into the running turn — drop it from the
    /// parked-input display and echo it to scrollback.
    fn note_steer_injected(&self, text: &str);
}

/// Auto-compaction circuit-breaker state threaded across turns.
#[derive(Debug, Clone, Default)]
pub struct AutocompactTracking {
    pub consecutive_failures: i64,
}

/// Context for tool use operations.
///
/// Cloning shares the mutable read_file_state map + abort token (the Python
/// dataclasses.replace shared them the same way); only the per-call fields
/// (tool_use_id) differ between clones.
#[derive(Clone)]
pub struct ToolUseContext {
    /// AbortController equivalent.
    pub abort_controller: CancellationToken,
    pub options: ContextOptions,
    pub message_id: Option<String>,
    pub read_file_state: Arc<Mutex<HashMap<String, ReadFileState>>>,
    pub on_token_update: Option<Arc<dyn Fn(i64) + Send + Sync>>,
    /// -rt: live safe text.
    pub on_content_delta: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Reasoning tokens generated so far this turn, when the server thinks on a
    /// separate channel. Fired only while thinking actually happens.
    pub on_thinking_update: Option<Arc<dyn Fn(i64) + Send + Sync>>,
    /// Per-call tool_use id (the source clones the context with toolUseId per
    /// call; check_permissions_and_call_tool sets it).
    pub tool_use_id: Option<String>,
    /// Sub-agent id, set on the context when a sub-agent runs.
    pub agent_id: Option<String>,
    /// Auto-compaction circuit-breaker state (consecutive_failures counter),
    /// shared across the turn's recursions.
    pub autocompact_tracking: Arc<Mutex<AutocompactTracking>>,
    /// Goal-gate recovery counter for this user turn (context._gate_recovery).
    pub gate_recovery: Arc<Mutex<i64>>,
    /// Sub-agent steer-wrapup flag (context._steer_wrapup).
    pub steer_wrapup: Arc<Mutex<bool>>,
    /// Set when the repetition recovery's LAST RESORT microcompacted a history
    /// the decoder had already collapsed on — a reduction that then had nowhere
    /// to go, because it lives in the recursion's arguments and the history
    /// belongs to the REPL. Read and cleared by process_prompt's pre-turn
    /// check, which passes it as `context_proven_fatal` so the same reduction
    /// is recomputed at the one place whose write-back already works
    /// (cli_repl.rs, the `was_microcompacted` arm).
    ///
    /// Why a flag and not the rescued message set itself: that set is a
    /// mid-turn snapshot, taken before the rescued turn produced its own tool
    /// calls and its final answer. Installing it as the next turn's history
    /// would discard the work the rescue was spent on. Recomputing over the
    /// CURRENT history yields the same reduction (microcompact is
    /// deterministic and idempotent) with the rescued turn still in it.
    ///
    /// Measured 2026-10-02, 16:03-16:11: two turns collapsed back to back on
    /// the same history. Each spent all 7 nudges — prompt moving +46 a time,
    /// `cached` pinned at 287,744, every attempt dying identically (think=26,
    /// out=200, 200 characters of '!') — then stage 8 reduced 289,659 to
    /// 171,554 and the turn completed. The next turn's first request went out
    /// at prompt=290,984, cached=287,744: the pre-collapse prefix had never
    /// been left, so the turn that was killed by that history was the one
    /// carried forward, and it died in the same place. 105s and 8 discarded
    /// requests per turn, indefinitely.
    pub proven_fatal: Arc<Mutex<bool>>,
    /// A mid-turn reduction happened and the REPL must redo it on its own
    /// history. Raised by compact_before_recursion for EVERY microcompaction,
    /// not just a collapse-driven one: microcompaction stands on the threshold
    /// alone, and a collapse merely borrows it.
    ///
    /// Separate from `proven_fatal` on purpose. That flag asserts the decoder
    /// died on this history and drives collapse-specific reporting; this one
    /// asserts only that a reduction needs re-applying, which is the ordinary
    /// case and must not depend on a collapse ever happening.
    pub needs_reduction: Arc<Mutex<bool>>,
    /// A mid-turn FULL compaction happened and the REPL must summarize its own
    /// history. Separate from `needs_reduction` because the two are not
    /// interchangeable: microcompaction is a pure, idempotent function of the
    /// history and can simply be replayed at the turn boundary, while a summary
    /// is an API round trip and cannot be reproduced locally at all.
    pub needs_summary: Arc<Mutex<bool>>,
    /// Re-pin the caller's spinner label for work that interrupts the turn,
    /// `None` to hand the label back. Same shape and same wiring point as
    /// on_token_update/on_thinking_update -- the turn owns the spinner, and
    /// nothing below it is allowed to know that.
    pub on_status: Option<Arc<dyn Fn(Option<String>) + Send + Sync>>,
    pub input_router: Option<Arc<dyn InputRouter>>,
}

impl Default for ToolUseContext {
    fn default() -> Self {
        ToolUseContext {
            abort_controller: CancellationToken::new(),
            options: ContextOptions::default(),
            message_id: None,
            read_file_state: Arc::new(Mutex::new(HashMap::new())),
            on_token_update: None,
            on_content_delta: None,
            on_thinking_update: None,
            tool_use_id: None,
            agent_id: None,
            autocompact_tracking: Arc::new(Mutex::new(AutocompactTracking::default())),
            gate_recovery: Arc::new(Mutex::new(0)),
            steer_wrapup: Arc::new(Mutex::new(false)),
            proven_fatal: Arc::new(Mutex::new(false)),
            needs_reduction: Arc::new(Mutex::new(false)),
            needs_summary: Arc::new(Mutex::new(false)),
            on_status: None,
            input_router: None,
        }
    }
}

impl ToolUseContext {
    /// The source clones toolUseContext with toolUseId per call
    /// (dataclasses.replace): shared state, new per-call id.
    pub fn with_tool_use_id(&self, tool_use_id: &str) -> Self {
        let mut c = self.clone();
        c.tool_use_id = Some(tool_use_id.to_string());
        c
    }

    pub fn is_aborted(&self) -> bool {
        self.abort_controller.is_cancelled()
    }
}

// === Validation Types ===

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub result: bool,
    pub message: String,
    pub meta: Option<Value>,
}

impl ValidationResult {
    pub fn success() -> Self {
        ValidationResult { result: true, message: String::new(), meta: None }
    }
    pub fn failure(message: impl Into<String>) -> Self {
        ValidationResult { result: false, message: message.into(), meta: None }
    }
}

// === Constants ===

pub const MAX_TOOL_USE_CONCURRENCY: usize = 10;

// INTERRUPT_MESSAGE / INTERRUPT_MESSAGE_FOR_TOOL_USE live in utils::messages
// (the single verbatim source).

#[cfg(test)]
mod proven_fatal_tests {
    use super::*;

    /// The flag starts down: an ordinary turn must not trigger a reduction.
    #[test]
    fn proven_fatal_starts_down() {
        assert!(!*ToolUseContext::default().proven_fatal.lock());
    }

    /// The whole mechanism rests on this. query.rs raises the flag several
    /// recursions deep, on whatever context clone it was handed; cli_repl reads
    /// it on the context it owns. A plain bool would make that write invisible
    /// and the fix a no-op, so pin the sharing across BOTH clone paths — a bare
    /// clone and the per-call with_tool_use_id the tools use.
    #[test]
    fn the_flag_is_shared_by_every_clone_of_the_context() {
        let repl_ctx = ToolUseContext::default();
        let deep_in_the_recursion = repl_ctx.clone().with_tool_use_id("toolu_1").clone();

        *deep_in_the_recursion.proven_fatal.lock() = true;

        assert!(
            *repl_ctx.proven_fatal.lock(),
            "the turn boundary cannot see the collapse the recursion observed"
        );
    }

    /// cli_repl consumes with mem::replace rather than reading. One rescue must
    /// force exactly one reduction: left latched, a single collapse would keep
    /// microcompacting every remaining turn of the session.
    #[test]
    fn consuming_the_flag_leaves_it_down() {
        let ctx = ToolUseContext::default();
        *ctx.proven_fatal.lock() = true;

        let first = { let mut f = ctx.proven_fatal.lock(); std::mem::replace(&mut *f, false) };
        let second = { let mut f = ctx.proven_fatal.lock(); std::mem::replace(&mut *f, false) };

        assert!(first, "the turn after the rescue must reduce");
        assert!(!second, "and the turn after that must not");
    }
}
