//! Built-in agent definitions + tool resolution for the Agent tool.
//! Equivalent to codemason/tools/built_in_agents.py (src/tools/AgentTool/built-in/*
//! + agentToolUtils.ts + prompt.ts).
//!
//! Each subagent_type maps to an AgentDefinition with its own system prompt and
//! an allow/deny tool policy. general-purpose / Explore / Plan are the portable
//! set; the SDK-only agents are omitted (no local substrate).

use std::collections::HashSet;
use std::sync::Arc;

use crate::tools::base::Tool;

// Tool-name constants (match the registered tool names).
pub const AGENT_TOOL_NAME: &str = "Agent";
pub const FILE_READ_TOOL_NAME: &str = "Read";
pub const FILE_WRITE_TOOL_NAME: &str = "Write";
/// tools::send_message_tool::SEND_MESSAGE_TOOL_NAME, named here for the same
/// reason the others are — importing the tool module would close a cycle.
pub const SEND_MESSAGE_TOOL_NAME: &str = "SendMessage";
pub const FILE_EDIT_TOOL_NAME: &str = "Edit";
pub const GLOB_TOOL_NAME: &str = "Glob";
pub const GREP_TOOL_NAME: &str = "Grep";
pub const BASH_TOOL_NAME: &str = "Bash";
pub const NOTEBOOK_EDIT_TOOL_NAME: &str = "NotebookEdit";
pub const ENTER_PLAN_MODE_TOOL_NAME: &str = "EnterPlanMode";
pub const EXIT_PLAN_MODE_TOOL_NAME: &str = "ExitPlanMode";
pub const ASK_USER_QUESTION_TOOL_NAME: &str = "AskUserQuestion";

/// Tools no sub-agent may use (ALL_AGENT_DISALLOWED_TOOLS). See the Python
/// reference for why each is barred.
pub fn all_agent_disallowed_tools() -> HashSet<&'static str> {
    [
        AGENT_TOOL_NAME,
        ASK_USER_QUESTION_TOOL_NAME,
        ENTER_PLAN_MODE_TOOL_NAME,
        EXIT_PLAN_MODE_TOOL_NAME,
        "TaskOutput",
        "TaskStop",
        // SendMessage: the source does not bar it — it simply never grants it.
        // A sub-agent's pool there is an ALLOWLIST
        // (ASYNC_AGENT_ALLOWED_TOOLS) that does not contain SendMessage; the
        // tool is granted only to in-process teammates and the coordinator.
        // codemason's pool is a denylist, so producing the same effect means
        // naming it here. A sub-agent holding SendMessage could steer a
        // sibling, and neither of them is the one the user is talking to.
        "SendMessage",
        // Skill: a deliberate divergence. The source ALLOWS Skill for async
        // agents (it is in ASYNC_AGENT_ALLOWED_TOOLS) — it can afford to,
        // because its fork path is behind a feature flag. codemason's
        // `context: fork` skills always spawn, so this guard is load-bearing:
        // without it a forked skill's sub-agent could fork again, unbounded,
        // around the Agent guard.
        "Skill",
    ]
    .into_iter()
    .collect()
}

/// A sub-agent definition. `get_system_prompt` is a closure so a custom agent
/// can capture its markdown body (the Python used a lambda closing over `body`).
#[derive(Clone)]
pub struct AgentDefinition {
    pub agent_type: String,
    pub when_to_use: String,
    pub get_system_prompt: Arc<dyn Fn() -> String + Send + Sync>,
    /// allowlist; None / ["*"] = all.
    pub tools: Option<Vec<String>>,
    pub disallowed_tools: Vec<String>,
    pub source: String,
}

// === general-purpose (generalPurposeAgent.ts) ===

const SHARED_PREFIX: &str = "You are an agent for CodeMason, an AI-powered coding assistant CLI. Given the user's message, you should use the tools available to complete the task. Complete the task fully—don't gold-plate, but don't leave it half-done.";

const SHARED_GUIDELINES: &str = "Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested.";

fn general_purpose_prompt() -> String {
    format!(
        "{SHARED_PREFIX} When you complete the task, respond with a concise report covering what was done and any key findings — the caller will relay this to the user, so it only needs the essentials.\n\n{SHARED_GUIDELINES}"
    )
}

pub const EXPLORE_AGENT_MIN_QUERIES: usize = 3;

fn explore_prompt() -> String {
    format!("You are a file search specialist for CodeMason. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use {GLOB_TOOL_NAME} for broad file pattern matching
- Use {GREP_TOOL_NAME} for searching file contents with regex
- Use {FILE_READ_TOOL_NAME} when you know the specific file path you need to read
- Use {BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use {BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly.")
}

fn plan_prompt() -> String {
    let search_hint = format!("{GLOB_TOOL_NAME}, {GREP_TOOL_NAME}, and {FILE_READ_TOOL_NAME}");
    format!("You are a software architect and planning specialist for CodeMason. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools - attempting to edit files will fail.

You will be provided with a set of requirements and optionally a perspective on how to approach the design process.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout the design process.

2. **Explore Thoroughly**:
   - Read any files provided to you in the initial prompt
   - Find existing patterns and conventions using {search_hint}
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use {BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
   - NEVER use {BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification

3. **Design Solution**:
   - Create implementation approach based on your assigned perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1.ts
- path/to/file2.ts
- path/to/file3.ts

REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or modify any files. You do NOT have access to file editing tools.")
}

fn general_purpose_agent() -> AgentDefinition {
    AgentDefinition {
        agent_type: "general-purpose".into(),
        when_to_use: "General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you.".into(),
        tools: Some(vec!["*".into()]),
        disallowed_tools: Vec::new(),
        get_system_prompt: Arc::new(general_purpose_prompt),
        source: "built-in".into(),
    }
}

fn explore_agent() -> AgentDefinition {
    AgentDefinition {
        agent_type: "Explore".into(),
        when_to_use: "Fast agent specialized for exploring codebases. Use this when you need to quickly find files by patterns (eg. \"src/components/**/*.tsx\"), search code for keywords (eg. \"API endpoints\"), or answer questions about the codebase (eg. \"how do API endpoints work?\"). When calling this agent, specify the desired thoroughness level: \"quick\" for basic searches, \"medium\" for moderate exploration, or \"very thorough\" for comprehensive analysis across multiple locations and naming conventions.".into(),
        tools: None,
        disallowed_tools: vec![
            AGENT_TOOL_NAME.into(), EXIT_PLAN_MODE_TOOL_NAME.into(), FILE_EDIT_TOOL_NAME.into(),
            FILE_WRITE_TOOL_NAME.into(), NOTEBOOK_EDIT_TOOL_NAME.into(),
        ],
        get_system_prompt: Arc::new(explore_prompt),
        source: "built-in".into(),
    }
}

fn plan_agent() -> AgentDefinition {
    AgentDefinition {
        agent_type: "Plan".into(),
        when_to_use: "Software architect agent for designing implementation plans. Use this when you need to plan the implementation strategy for a task. Returns step-by-step plans, identifies critical files, and considers architectural trade-offs.".into(),
        // Plan mirrors Explore's tools (None => all remaining after filters).
        tools: None,
        disallowed_tools: vec![
            AGENT_TOOL_NAME.into(), EXIT_PLAN_MODE_TOOL_NAME.into(), FILE_EDIT_TOOL_NAME.into(),
            FILE_WRITE_TOOL_NAME.into(), NOTEBOOK_EDIT_TOOL_NAME.into(),
        ],
        get_system_prompt: Arc::new(plan_prompt),
        source: "built-in".into(),
    }
}

/// getBuiltInAgents (external portable set).
pub fn get_built_in_agents() -> Vec<AgentDefinition> {
    vec![general_purpose_agent(), explore_agent(), plan_agent()]
}

/// Load user + project `agents/*.md` as custom sub-agent definitions
/// (loadAgentsDir.ts). First-wins by agent_type in user -> project order.
fn load_custom_agents() -> Vec<AgentDefinition> {
    use crate::utils::skills::parse_frontmatter;
    // Our own directories first. Until now ONLY the other product's were read,
    // so a codemason user had nowhere of their own to put a sub-agent — and one
    // dropped in another tool's directory was picked up whether they meant it or
    // not. Importing that layout is now opt-in (config.import_claude_dirs /
    // CODEMASON_IMPORT_CLAUDE_DIRS), matching how skills are discovered.
    let cwd = std::path::PathBuf::from(crate::utils::state::get_cwd());
    let import_foreign = crate::utils::config::get_config().import_claude_dirs;
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    dirs.push(crate::utils::config::CONFIG_DIR.join("agents"));
    dirs.push(cwd.join(".codemason").join("agents"));
    if import_foreign {
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join(".claude").join("agents"));
        }
        dirs.push(cwd.join(".claude").join("agents"));
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<AgentDefinition> = Vec::new();
    for d in dirs {
        if !d.is_dir() {
            continue;
        }
        let mut files: Vec<std::path::PathBuf> = match std::fs::read_dir(&d) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|x| x == "md").unwrap_or(false))
                .collect(),
            Err(_) => continue,
        };
        files.sort();
        for f in files {
            let Ok(text) = std::fs::read_to_string(&f) else {
                continue;
            };
            let (fm, body) = parse_frontmatter(&text);
            let name = fm
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .or_else(|| f.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_default();
            if name.is_empty() || seen.contains(&name) {
                continue;
            }
            seen.insert(name.clone());
            // Frontmatter values are serde_yaml::Value (utils::skills::parse_frontmatter).
            let tools_list: Option<Vec<String>> = match fm.get("tools") {
                Some(serde_yaml::Value::String(s)) => {
                    let v: Vec<String> = s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
                    if v.is_empty() { None } else { Some(v) }
                }
                Some(serde_yaml::Value::Sequence(a)) => {
                    let v: Vec<String> = a.iter().filter_map(|t| t.as_str().map(|s| s.trim().to_string())).filter(|t| !t.is_empty()).collect();
                    if v.is_empty() { None } else { Some(v) }
                }
                _ => None,
            };
            let when_to_use = fm.get("description").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            let body_owned = body.to_string();
            out.push(AgentDefinition {
                agent_type: name,
                when_to_use,
                get_system_prompt: Arc::new(move || body_owned.clone()),
                tools: tools_list,
                disallowed_tools: Vec::new(),
                        source: "project".into(),
            });
        }
    }
    out
}

/// Built-in agents plus custom `agents/*.md` definitions.
pub fn get_all_agents() -> Vec<AgentDefinition> {
    let mut all = get_built_in_agents();
    all.extend(load_custom_agents());
    all
}

/// Resolve a subagent_type to its definition. An omitted type defaults to
/// general-purpose; an explicit but UNKNOWN type is an error (never a silent
/// fall-back to general-purpose).
pub fn get_agent_definition(subagent_type: Option<&str>) -> Result<AgentDefinition, String> {
    let Some(st) = subagent_type.filter(|s| !s.is_empty()) else {
        return Ok(general_purpose_agent());
    };
    let agents = get_all_agents();
    if let Some(a) = agents.iter().find(|a| a.agent_type == st) {
        return Ok(a.clone());
    }
    let available: Vec<String> = agents.iter().map(|a| a.agent_type.clone()).collect();
    Err(format!(
        "Agent type '{st}' not found. Available agents: {}",
        available.join(", ")
    ))
}

/// Port of resolveAgentTools: ALL_AGENT_DISALLOWED filter -> per-agent
/// disallowed -> allowlist (["*"]/None = all remaining).
pub fn resolve_agent_tools(
    agent_def: &AgentDefinition,
    available_tools: &[Arc<dyn Tool>],
) -> Vec<Arc<dyn Tool>> {
    let global_disallowed = all_agent_disallowed_tools();
    let disallowed: HashSet<&str> = agent_def.disallowed_tools.iter().map(|s| s.as_str()).collect();
    let allowed: Vec<Arc<dyn Tool>> = available_tools
        .iter()
        .filter(|t| !global_disallowed.contains(t.name()))
        .filter(|t| !disallowed.contains(t.name()))
        .cloned()
        .collect();
    match &agent_def.tools {
        None => allowed,
        Some(t) if t.len() == 1 && t[0] == "*" => allowed,
        Some(t) => {
            let allow_set: HashSet<&str> = t.iter().map(|s| s.as_str()).collect();
            allowed.into_iter().filter(|tool| allow_set.contains(tool.name())).collect()
        }
    }
}

/// Port of getToolsDescription (prompt.ts:15).
fn tools_description(agent_def: &AgentDefinition) -> String {
    let has_allowlist = match &agent_def.tools {
        Some(t) => !(t.len() == 1 && t[0] == "*"),
        None => false,
    };
    let has_denylist = !agent_def.disallowed_tools.is_empty();

    if has_allowlist && has_denylist {
        let deny: HashSet<&str> = agent_def.disallowed_tools.iter().map(|s| s.as_str()).collect();
        let effective: Vec<&str> = agent_def
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.as_str())
            .filter(|t| !deny.contains(t))
            .collect();
        if effective.is_empty() {
            return "None".into();
        }
        return effective.join(", ");
    }
    if has_allowlist {
        return agent_def.tools.as_ref().unwrap().join(", ");
    }
    if has_denylist {
        return format!("All tools except {}", agent_def.disallowed_tools.join(", "));
    }
    "All tools".into()
}

pub fn format_agent_line(agent_def: &AgentDefinition) -> String {
    format!(
        "- {}: {} (Tools: {})",
        agent_def.agent_type,
        agent_def.when_to_use,
        tools_description(agent_def)
    )
}

/// Faithful getPrompt() (non-fork, non-coordinator), scoped to codemason's
/// Agent params.
pub fn build_agent_description() -> String {
    let agent_list: String = get_all_agents()
        .iter()
        .map(format_agent_line)
        .collect::<Vec<_>>()
        .join("\n");
    format!("Launch a new agent to handle complex, multi-step tasks autonomously.

The {AGENT_TOOL_NAME} tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{agent_list}

When using the {AGENT_TOOL_NAME} tool, specify a subagent_type parameter to select which agent type to use. If omitted, the general-purpose agent is used.

When NOT to use the {AGENT_TOOL_NAME} tool:
- If you want to read a specific file path, use the {FILE_READ_TOOL_NAME} tool or the {GLOB_TOOL_NAME} tool instead, to find the match more quickly
- If you are searching for a specific class definition like \"class Foo\", use the {GLOB_TOOL_NAME} tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the {FILE_READ_TOOL_NAME} tool instead of the {AGENT_TOOL_NAME} tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above

Usage notes:
- Always include a short `description` (3-5 words) summarizing what the agent will do.
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses.
- IMPORTANT: Write the agent prompt in the language the user is writing in — whatever that language is. Never default to English because the task is technical; the sub-agent reports back to the same person.
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- You can optionally run agents in the background using the run_in_background parameter. When an agent runs in the background, you will be automatically notified when it completes — do NOT sleep, poll, or proactively check on its progress. Continue with other work or respond to the user instead.
- **Foreground vs background**: Use foreground (default) when you need the agent's results before you can proceed — e.g., research agents whose findings inform your next steps. Use background when you have genuinely independent work to do in parallel.
- To send a message to a running agent, use {SEND_MESSAGE_TOOL_NAME} with the agent's ID as the `to` field. The message is delivered at the agent's next tool round. Each Agent invocation starts fresh — provide a complete task description.
- The agent's outputs should generally be trusted.
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent.
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents \"in parallel\", you MUST send a single message with multiple {AGENT_TOOL_NAME} tool use content blocks. For example, if you need to launch both a build-validator agent and a test-runner agent in parallel, send a single message with both tool calls.

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction.
- If you need a short response, say so (\"report in under 200 words\").
- Lookups: hand over the exact command. Investigations: hand over the question — prescribed steps become dead weight when the premise is wrong.

Terse command-style prompts produce shallow, generic work.

**Never delegate understanding.** Don't write \"based on your findings, fix the bug\" or \"based on the research, implement it.\" Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change.

Example usage:

<example_agent_descriptions>
\"test-runner\": use this agent after you are done writing code to run tests
\"greeting-responder\": use this agent to respond to user greetings with a friendly joke
</example_agent_descriptions>

<example>
user: \"Please write a function that checks if a number is prime\"
assistant: I'm going to use the {FILE_WRITE_TOOL_NAME} tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since a significant piece of code was written and the task was completed, now use the test-runner agent to run the tests
</commentary>
assistant: Uses the {AGENT_TOOL_NAME} tool to launch the test-runner agent
</example>

<example>
user: \"Hello\"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: \"I'm going to use the {AGENT_TOOL_NAME} tool to launch the greeting-responder agent\"
</example>")
}

#[cfg(test)]
mod tests {

    /// The prompt-writing section the earlier port compressed into one
    /// paragraph, and the examples it dropped entirely. "Never delegate
    /// understanding" is the part that earns its tokens: without it the model
    /// writes "based on your findings, fix the bug" and pushes the synthesis
    /// onto an agent that cannot do it.
    #[test]
    fn the_description_keeps_the_prompt_writing_guidance_and_examples() {
        let d = super::build_agent_description();
        for needle in [
            "## Writing the prompt",
            "report in under 200 words",
            "Lookups: hand over the exact command",
            "prescribed steps become dead weight when the premise is wrong",
            "**Never delegate understanding.**",
            "Example usage:",
            "<example_agent_descriptions>",
            "\"test-runner\": use this agent after you are done writing code to run tests",
            "launch the greeting-responder agent",
        ] {
            assert!(d.contains(needle), "the description dropped: {needle}");
        }
        // The code example must render as code, not as leftover format braces.
        assert!(d.contains("if (n <= 1) return false"));
        assert!(!d.contains("{{"), "doubled braces leaked into the rendered text");
    }

    /// getPrompt lists four notes after "outputs should generally be trusted";
    /// the port carried two and stopped. The "in parallel" one is the only place
    /// the model is told what the words mean, so it is pinned here.
    #[test]
    fn the_description_carries_the_source_usage_notes() {
        let d = build_agent_description();
        assert!(d.contains("should be used proactively"), "proactive note");
        assert!(
            d.contains("you MUST send a single message with multiple Agent tool use content blocks"),
            "the parallel note, with the tool name substituted"
        );
        assert!(d.contains("web fetches, etc."), "the research line lists fetches");
        assert!(!d.contains("{AGENT_TOOL_NAME}"), "no unsubstituted placeholder");
    }
    use super::*;

    #[test]
    fn unknown_agent_type_errors() {
        assert!(get_agent_definition(Some("nope")).is_err());
        assert!(get_agent_definition(None).is_ok());
        assert_eq!(get_agent_definition(None).unwrap().agent_type, "general-purpose");
    }

    #[test]
    fn tools_description_variants() {
        let ex = explore_agent();
        assert!(tools_description(&ex).starts_with("All tools except"));
        let gp = general_purpose_agent();
        assert_eq!(tools_description(&gp), "All tools");
    }
}
