//! Tools for CodeMason — aligned to Claude Code's getAllBaseTools().
//! Equivalent to codemason/tools/__init__.py.
//!
//! Only tools that exist in Claude Code are registered. SessionSearch is the
//! one deliberate addition: it is claude.ai's conversation_search brought
//! local — the past chats are JSONL transcripts on this disk.

pub mod agent_tool;
pub mod agent_tool_utils;
pub mod ask_user_question_tool;
pub mod base;
pub mod bash_tool;
pub mod built_in_agents;
pub mod file_edit_tool;
pub mod file_read_tool;
pub mod file_write_tool;
pub mod glob_tool;
pub mod grep_tool;
pub mod mcp_tool;
pub mod notebook_edit_tool;
pub mod plan_mode_tools;
#[cfg_attr(not(windows), allow(dead_code))]
pub mod powershell_tool;
pub mod session_search_tool;
pub mod skill_tool;
pub mod task_output_tool;
pub mod resume_agent;
pub mod send_message_tool;
pub mod task_stop_tool;
pub mod todo_write_tool;
pub mod web_fetch_tool;
pub mod web_search_tool;

use std::collections::HashSet;
use std::sync::Arc;

use base::Tool;

/// All available tools. PowerShellTool is Windows-only and self-gates via
/// is_enabled() (get_all_tools filters on it). Order matches ALL_TOOLS in the
/// Python — it is the order the model sees the tool list in.
pub fn all_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(agent_tool::AgentTool),
        Arc::new(bash_tool::BashTool),
        Arc::new(powershell_tool::tool::PowerShellTool),
        Arc::new(glob_tool::GlobTool),
        Arc::new(grep_tool::GrepTool),
        Arc::new(file_read_tool::FileReadTool),
        Arc::new(file_edit_tool::FileEditTool),
        Arc::new(file_write_tool::FileWriteTool),
        Arc::new(todo_write_tool::TodoWriteTool),
        Arc::new(ask_user_question_tool::AskUserQuestionTool),
        Arc::new(notebook_edit_tool::NotebookEditTool),
        Arc::new(web_fetch_tool::WebFetchTool),
        Arc::new(web_search_tool::WebSearchTool),
        Arc::new(session_search_tool::SessionSearchTool),
        Arc::new(plan_mode_tools::EnterPlanModeTool),
        Arc::new(plan_mode_tools::ExitPlanModeTool),
        Arc::new(skill_tool::SkillTool),
        Arc::new(task_output_tool::TaskOutputTool),
        Arc::new(send_message_tool::SendMessageTool),
        Arc::new(task_stop_tool::TaskStopTool),
    ]
}

/// Tool names disabled for this process (DISABLED_TOOL_NAMES).
fn disabled_tool_names() -> HashSet<&'static str> {
    HashSet::new()
}

/// Per-process tool restriction, set once at startup.
///
/// An embedder that only needs to read (a log analyser, a reviewer) should not
/// have to trust the model not to call Write or WebFetch — the call simply must
/// not be offered. Tools the model never sees are tools it cannot misuse, and
/// the restriction is auditable from the command line rather than buried in a
/// prompt.
///
/// Set from `--allowed-tools` / `--disallowed-tools`, or the environment
/// (`CODEMASON_ALLOWED_TOOLS`, `CODEMASON_DISALLOWED_TOOLS`) so a parent process
/// can pin it without touching argv.
static TOOL_FILTER: std::sync::OnceLock<ToolFilter> = std::sync::OnceLock::new();

#[derive(Default, Clone)]
pub struct ToolFilter {
    /// When non-empty, only these names survive.
    pub allowed: HashSet<String>,
    /// Always removed, even if listed in `allowed`.
    pub disallowed: HashSet<String>,
}

fn split_names(raw: &str) -> HashSet<String> {
    raw.split(&[',', ' '][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Install the filter. First call wins; later calls are ignored so a nested
/// component cannot widen what the process was started with.
pub fn set_tool_filter(allowed: &str, disallowed: &str) {
    let _ = TOOL_FILTER.set(ToolFilter {
        allowed: split_names(allowed),
        disallowed: split_names(disallowed),
    });
}

fn tool_filter() -> ToolFilter {
    if let Some(f) = TOOL_FILTER.get() {
        return f.clone();
    }
    // Not set from argv: fall back to the environment so an embedder that
    // spawns this binary can restrict it without owning the command line.
    ToolFilter {
        allowed: split_names(&std::env::var("CODEMASON_ALLOWED_TOOLS").unwrap_or_default()),
        disallowed: split_names(&std::env::var("CODEMASON_DISALLOWED_TOOLS").unwrap_or_default()),
    }
}

/// Is this tool offered to the model in this process?
fn tool_permitted(name: &str, f: &ToolFilter) -> bool {
    if f.disallowed.contains(name) {
        return false;
    }
    if f.allowed.is_empty() {
        return true;
    }
    f.allowed.contains(name)
}

/// Tools discovered from configured MCP servers (empty if none).
///
/// Discovery is cached in tools::mcp_tool and never raises out here, so a
/// misconfigured or unreachable MCP server can't stop CodeMason from starting.
fn mcp_tools() -> Vec<Arc<dyn Tool>> {
    mcp_tool::build_mcp_tools()
}

/// Get all available tools (native + any MCP-server tools).
pub async fn get_all_tools() -> Vec<Arc<dyn Tool>> {
    let disabled = disabled_tool_names();
    let filter = tool_filter();
    let mut enabled: Vec<Arc<dyn Tool>> = Vec::new();
    for tool in all_tools() {
        if disabled.contains(tool.name()) || !tool_permitted(tool.name(), &filter) {
            continue;
        }
        if tool.is_enabled().await {
            enabled.push(tool);
        }
    }
    for tool in mcp_tools() {
        if disabled.contains(tool.name()) || !tool_permitted(tool.name(), &filter) {
            continue;
        }
        if tool.is_enabled().await {
            enabled.push(tool);
        }
    }
    enabled
}

/// Get read-only tools only.
pub async fn get_read_only_tools() -> Vec<Arc<dyn Tool>> {
    let disabled = disabled_tool_names();
    let mut enabled: Vec<Arc<dyn Tool>> = Vec::new();
    for tool in all_tools() {
        if disabled.contains(tool.name()) {
            continue;
        }
        if tool.is_enabled().await && tool.is_read_only() {
            enabled.push(tool);
        }
    }
    enabled
}

/// Tool pool available to sub-agents (no recursive Agent tool, no asking).
///
/// Faithful to resolveAgentTools: start from the FULL enabled tool pool —
/// write capability is gated at CALL TIME by the permission flow, NOT by
/// pre-filtering the pool to read-only. resolve_agent_tools() then applies the
/// agent definition's allow/deny policy.
///
/// AskUserQuestion is withheld on top of that: a sub-agent has no user on the
/// other end. Its local half reads a stdin nobody is typing into, and its
/// remote half puts the question on the phone and waits out remote_ask_timeout
/// (1800s by default) before giving up — half an hour of a parent turn spent
/// on a question whose answer was never going to arrive. An agent is supposed
/// to come back with a report: it should decide, or say what it could not
/// decide, and let the caller ask.
pub async fn get_agent_tools() -> Vec<Arc<dyn Tool>> {
    get_all_tools()
        .await
        .into_iter()
        .filter(|t| {
            t.name() != "Agent"
                && t.name() != crate::constants::prompts::ASK_USER_QUESTION_TOOL_NAME
        })
        .collect()
}

/// Get a tool by its name or one of its aliases (toolMatchesName).
pub fn get_tool_by_name(name: &str) -> Result<Arc<dyn Tool>, String> {
    for tool in all_tools() {
        if tool.matches_name(name) {
            return Ok(tool);
        }
    }
    for tool in mcp_tools() {
        if tool.matches_name(name) {
            return Ok(tool);
        }
    }
    Err(format!("Unknown tool: {name}"))
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    /// Resolve a tool's wire description exactly as the backend does.
    async fn resolved_wire(t: &Arc<dyn Tool>) -> String {
        match t.wire_description().await {
            Some(d) if !d.is_empty() => d,
            _ => crate::constants::tool_prompts::tool_description(t.name())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    crate::services::openai_common::get_simple_description(t.name())
                }),
        }
    }

    /// The wire description must be the tool's real DESCRIPTION, never the
    /// short display blurb `description()` returns.
    ///
    /// This is the invariant a regression here breaks silently: sending the
    /// blurb instead cost ~7k tokens of tool guidance per request and was
    /// invisible on screen — only the server's input_tokens showed it. Lengths
    /// are not asserted (several descriptions embed live values: the agent
    /// list, the current month, gate-dependent sections); the shape is.
    #[tokio::test]
    async fn wire_description_is_not_the_display_blurb() {
        // Tools whose Python DESCRIPTION is deliberately SHORT — verified
        // byte-for-byte against the Python dump, so their exact text is safe
        // to pin.
        const SHORT: &[(&str, &str)] = &[
            ("AskUserQuestion", "Asks the user multiple choice questions to gather information, clarify ambiguity, understand preferences, make decisions or offer them choices."),
            ("SessionSearch", "Use the SessionSearch tool"),
            ("TaskOutput", "Use the TaskOutput tool"),
            ("TaskStop", "Use the TaskStop tool"),
        ];
        let tools = get_all_tools().await;
        for t in &tools {
            let wire = resolved_wire(t).await;
            assert!(!wire.is_empty(), "{} has no wire description", t.name());
            if let Some((_, expected)) = SHORT.iter().find(|(n, _)| *n == t.name()) {
                assert_eq!(&wire, expected, "{} wire description", t.name());
                continue;
            }
            // Every other tool sends its full DESCRIPTION, which for these is
            // exactly its system prompt. (Not compared against description():
            // several tools legitimately return the same long text there too,
            // so equality with prompt() is the load-bearing check.)
            assert_eq!(
                wire,
                t.prompt().await,
                "{} wire description drifted from its prompt",
                t.name()
            );
            assert!(
                wire.len() > 300,
                "{} wire description is suspiciously short ({})",
                t.name(),
                wire.len()
            );
        }
    }

    /// The registry is the tool list the model sees; a tool silently dropped
    /// from it is invisible until the model fails to use it.
    #[tokio::test]
    async fn registry_matches_python_tool_set() {
        // ALL_TOOLS from codemason/tools/__init__.py, in order (PowerShell is
        // Windows-only and filtered by is_enabled off Windows).
        const EXPECTED_ALL: &[&str] = &[
            "Agent",
            "Bash",
            "PowerShell",
            "Glob",
            "Grep",
            "Read",
            "Edit",
            "Write",
            "TodoWrite",
            "AskUserQuestion",
            "NotebookEdit",
            "WebFetch",
            "WebSearch",
            "SessionSearch",
            "EnterPlanMode",
            "ExitPlanMode",
            "Skill",
            "TaskOutput",
            "SendMessage",
            "TaskStop",
        ];
        let names: Vec<&str> = all_tools().iter().map(|t| t.name()).collect();
        assert_eq!(names, EXPECTED_ALL);

        // Sub-agents never get the Agent tool (the recursion guard).
        let agent_tools = get_agent_tools().await;
        assert!(agent_tools.iter().all(|t| t.name() != "Agent"));
    }
}
