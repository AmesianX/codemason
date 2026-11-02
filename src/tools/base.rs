//! Base Tool trait for CodeMason
//! Equivalent to codemason/tools/base.py (inferred from BashTool, GrepTool, etc.)

use async_trait::async_trait;
use futures::stream::Stream;
use serde_json::Value;
use std::pin::Pin;

use crate::types::query::ToolUseContext;
use crate::utils::formatting::summarize_tool_input;

// === Validation Result Types ===

pub use crate::types::query::ValidationResult;

// === Permission Result (Tool.ts checkPermissions -> PermissionResult) ===

/// 3-state decision a tool returns from check_permissions:
///
/// - 'allow' : auto-approve, no user prompt.
/// - 'ask'   : defer to the interactive prompt (can_use_tool). The optional
///             message is shown so the user knows WHY approval is needed.
/// - 'deny'  : hard-block with message, no prompt.
///
/// The whole permission flow is bypassed under dangerously_skip_permissions —
/// faithful to the original, where checkPermissions is never consulted in that
/// mode.
#[derive(Debug, Clone)]
pub struct PermissionResult {
    pub behavior: PermissionBehavior,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionBehavior {
    Allow,
    Ask,
    Deny,
}

impl PermissionResult {
    pub fn allow() -> Self {
        PermissionResult { behavior: PermissionBehavior::Allow, message: String::new() }
    }
    pub fn ask() -> Self {
        PermissionResult { behavior: PermissionBehavior::Ask, message: String::new() }
    }
    pub fn ask_with(message: impl Into<String>) -> Self {
        PermissionResult { behavior: PermissionBehavior::Ask, message: message.into() }
    }
    pub fn deny(message: impl Into<String>) -> Self {
        PermissionResult { behavior: PermissionBehavior::Deny, message: message.into() }
    }
}

// === Tool Result Types ===

/// Successful tool result (ToolResultSuccess) or a progress update
/// (ToolResultProgress). Python tools yielded either from an async generator;
/// Rust tools yield them from their result stream.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Result {
        result_for_assistant: String,
        /// The tool's structured Output (rendered by
        /// render_tool_result_message; never sent to the model).
        data: Option<Value>,
    },
    Progress {
        content: Option<Value>,
        normalized_messages: Vec<Value>,
    },
    /// A result whose model-facing content carries image blocks next to the
    /// text (Read on an image when the server can see). `images` are
    /// Anthropic image blocks ({"type": "image", "source": {...}}).
    ResultWithImages {
        result_for_assistant: String,
        images: Vec<Value>,
        data: Option<Value>,
    },
}

impl ToolResult {
    pub fn result(result_for_assistant: impl Into<String>, data: Option<Value>) -> Self {
        ToolResult::Result { result_for_assistant: result_for_assistant.into(), data }
    }
}

/// A tool error raised mid-execution (the Python generator `raise` path).
/// Carried as a stream item so the dispatcher can format it exactly like the
/// original's exception handler (format_error).
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ToolError {
    pub message: String,
    pub stderr: Option<String>,
    pub stdout: Option<String>,
}

impl ToolError {
    pub fn new(message: impl Into<String>) -> Self {
        ToolError { message: message.into(), stderr: None, stdout: None }
    }
}

/// The stream a tool's call() returns — the Rust shape of the Python async
/// generator. Err items are the generator raising.
pub type ToolStream = Pin<Box<dyn Stream<Item = Result<ToolResult, ToolError>> + Send>>;

/// can_use_tool callback: the interactive permission prompt
/// (CanUseToolFn in query.py). Returns {'result': bool, 'message': str}.
pub type CanUseToolFn = std::sync::Arc<
    dyn Fn(
            &'static str,        // tool name
            Value,               // input_data
            ToolUseContext,      // context
        ) -> futures::future::BoxFuture<'static, PermissionDecision>
        + Send
        + Sync,
>;

/// Result of the interactive permission prompt.
#[derive(Debug, Clone)]
pub struct PermissionDecision {
    pub result: bool,
    pub message: String,
}

impl PermissionDecision {
    pub fn allowed() -> Self {
        PermissionDecision { result: true, message: String::new() }
    }
    pub fn denied(message: impl Into<String>) -> Self {
        PermissionDecision { result: false, message: message.into() }
    }
}

// === Base Tool Trait ===

/// Base trait for all tools in CodeMason.
///
/// Each tool must implement `name`, `input_json_schema`, and `call`.
/// Everything else has the same defaults the Python base class carried; see
/// codemason/tools/base.py for the full method-by-method rationale.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool identifier.
    fn name(&self) -> &'static str;

    /// Optional aliases for backward compatibility when a tool is renamed
    /// (ToolDef.aliases). A tool_use naming any alias resolves to this tool.
    fn aliases(&self) -> &'static [&'static str] {
        &[]
    }

    /// Max chars of result before it is persisted to disk and replaced with a
    /// <persisted-output> preview. The effective threshold is min(this, 50k);
    /// f64::INFINITY opts out entirely (e.g. Read).
    fn max_result_size_chars(&self) -> f64 {
        100_000.0
    }

    /// The JSON schema the model fills in (the `parameters` of the tool call).
    /// A hook, not a constant, so a tool can adapt a field's description to
    /// the session. Backends must call this.
    fn input_json_schema(&self) -> Value;

    /// Generate a description for this tool use.
    async fn description(&self, _input: &Value) -> String {
        format!("Execute {}", self.name())
    }

    /// Get the system prompt for this tool.
    async fn prompt(&self) -> String {
        String::new()
    }

    /// The description sent to the model ON THE WIRE, in the tool definition.
    ///
    /// This is the Python's `DESCRIPTION` class attribute (and, where a tool
    /// has one, its dynamic `build_description()`) — NOT `description()`,
    /// which is the short display blurb. Getting these two confused silently
    /// strips thousands of tokens of tool guidance from every request, which
    /// nothing on screen would show.
    ///
    /// Every codemason tool whose `DESCRIPTION` equals its system prompt is
    /// served by this default; the four whose wire text DIFFERS from their
    /// prompt (AskUserQuestion, SessionSearch, TaskOutput, TaskStop) override
    /// it. `None` means "no tool-provided description", which the backend
    /// resolves through TOOL_DESCRIPTIONS and then the simple blurb — the same
    /// chain the Python walks.
    async fn wire_description(&self) -> Option<String> {
        let p = self.prompt().await;
        if p.is_empty() {
            None
        } else {
            Some(p)
        }
    }

    /// Whether this tool only reads data (doesn't modify anything).
    fn is_read_only(&self) -> bool {
        false
    }

    /// Whether this tool may run in parallel with other concurrency-safe tools.
    /// Defaults to is_read_only(); a tool that is safe while NOT read-only
    /// (e.g. TaskStop) overrides this to return true.
    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        self.is_read_only()
    }

    /// True if `name` is this tool's name or one of its aliases
    /// (toolMatchesName, Tool.ts).
    fn matches_name(&self, name: &str) -> bool {
        name == self.name() || self.aliases().contains(&name)
    }

    /// Whether this tool is currently enabled.
    async fn is_enabled(&self) -> bool {
        true
    }

    /// Whether this tool requires user permission.
    fn needs_permissions(&self) -> bool {
        true
    }

    /// True when this tool's prompt is the user's decision, not a grant —
    /// it keeps asking even under dangerously_skip_permissions. Only
    /// ExitPlanMode sets it today.
    fn always_ask(&self) -> bool {
        false
    }

    /// Tool-specific permission decision (Tool.ts checkPermissions).
    /// Default: read-only tools auto-allow; everything else defers to the
    /// interactive prompt ('ask').
    async fn check_permissions(
        &self,
        _input: &Value,
        _context: &ToolUseContext,
    ) -> PermissionResult {
        if self.is_read_only() {
            PermissionResult::allow()
        } else {
            PermissionResult::ask()
        }
    }

    /// Validate the input before execution.
    async fn validate_input(&self, _input: &Value, _context: &ToolUseContext) -> ValidationResult {
        ValidationResult::success()
    }

    /// Get the user-facing name for this tool.
    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        self.name().to_string()
    }

    /// The one argument that identifies this call (Tool.ts getToolUseSummary).
    ///
    /// A path, a pattern, a URL — whatever a reader would use to tell two calls
    /// of the same tool apart, already shortened for display. None when the
    /// input does not carry it, which is what lets the caller fall back to the
    /// tool's name.
    fn get_tool_use_summary(&self, _input: Option<&Value>) -> Option<String> {
        None
    }

    /// Present-tense line for a progress view (Tool.ts getActivityDescription).
    ///
    /// "Reading src/foo.rs", "Running bun test", "Searching for pattern" — the
    /// verb belongs to the tool, the argument comes from get_tool_use_summary.
    /// None falls back to the tool name.
    ///
    /// This is what a parent watching a sub-agent reads: it is stored on each
    /// tool use in the agent's progress tally (tasks/local_agent_task.rs), so
    /// "what is it doing right now" has an answer that is not just a tool name.
    fn get_activity_description(&self, _input: Option<&Value>) -> Option<String> {
        None
    }

    /// Render the tool use for display (Tool.ts renderToolUseMessage). The tool
    /// owns its own display: the CLI prints the header and this body, and never
    /// inspects the input itself. May be multi-line; "" shows the header alone.
    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        summarize_tool_input(input, 160)
    }

    /// Render the tool RESULT for display (Tool.ts renderToolResultMessage).
    /// Takes the tool's structured Output (ToolResult::Result.data), NOT the
    /// serialized result_for_assistant. None = no custom rendering; Some("") =
    /// render nothing.
    fn render_tool_result_message(&self, _data: &Value, _verbose: bool) -> Option<String> {
        None
    }

    /// Render the result for the assistant.
    fn render_result_for_assistant(&self, result: &Value) -> String {
        match result {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// Execute the tool. `input` is the validated input; the stream yields
    /// progress updates and finally the result (or an error item, matching the
    /// Python generator raising).
    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream;
}

/// Find a tool by name OR alias in a tool list (toolMatchesName resolution).
pub fn find_tool<'a>(
    tools: &'a [std::sync::Arc<dyn Tool>],
    name: &str,
) -> Option<&'a std::sync::Arc<dyn Tool>> {
    tools.iter().find(|t| t.matches_name(name))
}
