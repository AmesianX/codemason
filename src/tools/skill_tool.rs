//! Skill Tool for CodeMason
//! Based on src/tools/SkillTool/ (getPrompt description + schema) from Claude
//! Code. Equivalent to codemason/tools/skill_tool.py.
//!
//! A skill is a markdown file with frontmatter (see utils/skills.rs). Invoking
//! a skill with `context: inline` (default) expands its body into the
//! conversation (returned as the tool result); `context: fork` runs the body
//! in a sub-agent.
//!
//! Substrate note: the original lists available skills via per-turn
//! system-reminder attachments; codemason has no attachment pipeline, so the
//! skill listing is appended to the tool description (built dynamically).

use async_trait::async_trait;
use futures::StreamExt;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::tools::base::{find_tool, CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::skills::{discover_skills, get_skill, model_invocable_skills, Skill};

pub const SKILL_TOOL_NAME: &str = "Skill";

/// argumentSubstitution.ts parseArguments: shell-quote split, falling back to
/// a whitespace split if parsing fails.
fn parse_arguments(args: &str) -> Vec<String> {
    if args.trim().is_empty() {
        return Vec::new();
    }
    match shlex::split(args) {
        Some(parts) => parts,
        None => args.split_whitespace().map(|t| t.to_string()).collect(),
    }
}

/// Port of argumentSubstitution.ts substituteArguments (lines 94-145).
///
/// Replaces $name / $ARGUMENTS[n] / $n / $ARGUMENTS placeholders in `content`.
/// Only when NO placeholder was found (and args is non-empty) does it append
/// "\n\nARGUMENTS: {args}".
pub fn substitute_arguments(
    content: &str,
    args: Option<&str>,
    append_if_no_placeholder: bool,
    argument_names: &[String],
) -> String {
    // undefined/null means no args provided - return content unchanged.
    let Some(args) = args else {
        return content.to_string();
    };

    let parsed = parse_arguments(args);
    let original = content.to_string();
    let mut content = content.to_string();

    // Named arguments ($foo, $bar) map to positions: names[i] -> parsed[i].
    // Python matched `\$name(?![\[\w])`; the regex crate has no lookahead, so
    // the blocking char is captured and reinserted — a blocked match consumes
    // one extra char, which can never hide a later `$` (names are literal).
    // The replacement value is inserted literally (the Python used a function
    // repl so arg text like C:\Users\x or \1 is never parsed as group refs).
    for (i, name) in argument_names.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        let val = parsed.get(i).map(|s| s.as_str()).unwrap_or("");
        if let Ok(re) = Regex::new(&format!(r"\${}([\[\w])?", regex::escape(name))) {
            content = re
                .replace_all(&content, |caps: &regex::Captures| match caps.get(1) {
                    Some(blocker) => format!("${}{}", name, blocker.as_str()),
                    None => val.to_string(),
                })
                .into_owned();
        }
    }

    // Indexed arguments $ARGUMENTS[0], $ARGUMENTS[1], ...
    static INDEXED_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\$ARGUMENTS\[(\d+)\]").unwrap());
    let indexed = |caps: &regex::Captures| -> String {
        let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
        parsed.get(idx).cloned().unwrap_or_default()
    };
    content = INDEXED_RE.replace_all(&content, indexed).into_owned();

    // Shorthand indexed arguments $0, $1, ... (not followed by word char —
    // same captured-blocker restructure as above; note \w includes digits, so
    // a blocked `$12x` reinserts verbatim exactly like the Python's failed
    // backtrack).
    static SHORTHAND_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\$(\d+)(\w)?").unwrap());
    content = SHORTHAND_RE
        .replace_all(&content, |caps: &regex::Captures| match caps.get(2) {
            Some(blocker) => format!("${}{}", &caps[1], blocker.as_str()),
            None => {
                let idx: usize = caps[1].parse().unwrap_or(usize::MAX);
                parsed.get(idx).cloned().unwrap_or_default()
            }
        })
        .into_owned();

    // Full arguments string.
    content = content.replace("$ARGUMENTS", args);

    // If no placeholder was substituted and args is non-empty, append.
    if content == original && append_if_no_placeholder && !args.is_empty() {
        content = format!("{content}\n\nARGUMENTS: {args}");
    }

    content
}

const BASE_PROMPT: &str = include_str!("prompts/skill_base_prompt.txt");

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(SKILL_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

// Embedded-shell syntax (promptShellExecution.ts): fenced ```! blocks and
// inline !`cmd`. Inline requires start-of-line or preceding whitespace so a
// literal `!` mid-word doesn't match. Python used `(?<=\s)` (lookbehind); the
// regex crate has none, so the leading char is captured and reinserted at
// splice time.
static SKILL_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"```!\s*\n?([\s\S]*?)\n?```").unwrap());
static SKILL_INLINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)(^|\s)!`([^`]+)`").unwrap());

/// Port of executeShellCommandsInPrompt: run ```! blocks and !`inline`
/// commands via Bash and splice their output into the skill body. Permission
/// is gated through can_use_tool (fail-closed: no callback or a deny leaves
/// the marker untouched, so a skill file can't silently run shell without
/// approval).
async fn execute_shell_commands_in_prompt(
    text: &str,
    context: &ToolUseContext,
    can_use_tool: Option<&CanUseToolFn>,
) -> String {
    use crate::utils::tool_result_storage::process_tool_result_content;

    // (full match text, leading prefix to keep, command)
    let mut matches: Vec<(String, String, String)> = Vec::new();
    for m in SKILL_BLOCK_RE.captures_iter(text) {
        matches.push((
            m.get(0).map(|w| w.as_str().to_string()).unwrap_or_default(),
            String::new(),
            m.get(1).map(|c| c.as_str().to_string()).unwrap_or_default(),
        ));
    }
    if text.contains("!`") {
        for m in SKILL_INLINE_RE.captures_iter(text) {
            matches.push((
                m.get(0).map(|w| w.as_str().to_string()).unwrap_or_default(),
                m.get(1).map(|p| p.as_str().to_string()).unwrap_or_default(),
                m.get(2).map(|c| c.as_str().to_string()).unwrap_or_default(),
            ));
        }
    }
    if matches.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    for (matched, prefix, command) in matches {
        let command = command.trim().to_string();
        if command.is_empty() {
            continue;
        }
        let mut allowed = false;
        if let Some(gate) = can_use_tool {
            let decision = gate(
                "Bash",
                serde_json::json!({"command": command}),
                context.clone(),
            )
            .await;
            allowed = decision.result;
        }
        if !allowed {
            continue; // permission denied / no gate -> leave the marker as-is
        }
        let mut out = String::new();
        if let Some(bash) = find_tool(&context.options.tools, "Bash") {
            let mut stream = bash.call(
                serde_json::json!({"command": command}),
                context.clone(),
                can_use_tool.cloned(),
            );
            while let Some(item) = stream.next().await {
                if let Ok(ToolResult::Result { result_for_assistant, .. }) = item {
                    out = result_for_assistant;
                }
            }
        }
        let tool_use_id = context
            .tool_use_id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "skill-inline".to_string());
        let out = process_tool_result_content(&out, "Bash", &tool_use_id, 30_000.0);
        // Splice: replace the first occurrence of the matched text, keeping
        // the captured leading char the Python's lookbehind never consumed.
        result = result.replacen(&matched, &format!("{prefix}{out}"), 1);
    }
    result
}

fn normalized_name(input: &Value) -> String {
    let name = input.get("skill").and_then(|s| s.as_str()).unwrap_or("").trim();
    // `/commit` -> `commit` (SkillTool.ts commandName normalization)
    name.strip_prefix('/').unwrap_or(name).to_string()
}

pub struct SkillTool;

impl SkillTool {
    pub fn build_description() -> String {
        let skills = model_invocable_skills();
        let listing = if skills.is_empty() {
            "(no skills are currently available)".to_string()
        } else {
            skills
                .iter()
                .map(|s| format!("- {}: {}", s.name, s.description))
                .collect::<Vec<_>>()
                .join("\n")
        };
        format!("{BASE_PROMPT}\nAvailable skills:\n{listing}\n")
    }

    /// Run a context:fork skill in a sub-agent, returning its final report.
    async fn run_forked(
        skill: &Skill,
        body: String,
        context: &ToolUseContext,
        can_use_tool: Option<CanUseToolFn>,
    ) -> Result<String, String> {
        use crate::tools::built_in_agents::{get_agent_definition, resolve_agent_tools};
        use crate::types::query::{ContextOptions, Message};
        use crate::utils::messages::{create_user_message_text, get_text_from_message};

        let skip = context.options.dangerously_skip_permissions;
        let agent_def = get_agent_definition(skill.agent.as_deref())?;
        let tools = resolve_agent_tools(&agent_def, &crate::tools::get_agent_tools().await);
        let model = context
            .options
            .slow_and_capable_model
            .clone()
            .unwrap_or_else(|| crate::utils::config::get_config().model_name);
        // Definition prompt + the same session-constant tail the Agent tool
        // uses (notes, env, language) — one pinned copy per process, so a
        // forked skill and a sub-agent share the cached prefix instead of each
        // spawn rebuilding it from the current cwd and effective_language().
        let mut system_prompt = vec![(agent_def.get_system_prompt)()];
        system_prompt.extend(crate::constants::prompts::agent_prefix_sections(&model));

        // Own abort scope under the parent's, same as AgentTool: a forked skill is a

        // sub-agent tenant and must be stoppable without stopping its siblings.

        let skill_abort = context.abort_controller.child_token();

        let mut sub_ctx = ToolUseContext {
            abort_controller: skill_abort.clone(),
            options: ContextOptions {
                tools,
                dangerously_skip_permissions: skip,
                slow_and_capable_model: Some(model),
            },
            on_token_update: context.on_token_update.clone(),
            on_thinking_update: context.on_thinking_update.clone(),
            // Same reason as the sub-agent context: without the parent's
            // router a permission prompt raised inside the skill's own query
            // loop asks with a bare line read and never reaches the phone.
            input_router: context.input_router.clone(),
            ..Default::default()
        };
        sub_ctx.agent_id = Some(uuid::Uuid::new_v4().simple().to_string()[..8].to_string());

        let Some(can_use_tool) = can_use_tool else {
            return Err("Skill fork requires a permission callback".to_string());
        };

        // Same slot the Agent tool takes: a forked skill is a sub-agent tenant.
        let _slot = match crate::utils::subagent_gate::acquire(&skill_abort).await {
            Some(s) => s,
            None => return Err("Aborted while waiting for a sub-agent slot".to_string()),
        };

        let mut messages: Vec<Message> = vec![Message::User(create_user_message_text(&body))];
        let mut stream = crate::query::query(crate::query::QueryParams {
            messages: messages.clone(),
            system_prompt,
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
        while let Some(msg) = stream.next().await {
            messages.push(msg);
        }

        // Final assistant text (faithful to finalizeAgentTool: last assistant
        // with text).
        for m in messages.iter().rev() {
            if let Message::Assistant(_) = m {
                let joined = get_text_from_message(m).trim().to_string();
                if !joined.is_empty() {
                    return Ok(joined);
                }
            }
        }
        Ok(format!("Skill \"{}\" completed.", skill.name))
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &'static str {
        SKILL_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    async fn description(&self, _input: &Value) -> String {
        Self::build_description()
    }

    async fn prompt(&self) -> String {
        Self::build_description()
    }

    fn is_read_only(&self) -> bool {
        false // a skill may run write tools
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        // [ours] Same split as AgentTool: not read-only (a skill may write), but
        // safe to schedule alongside other concurrency-safe calls. See agent_tool.rs.
        true
    }

    async fn is_enabled(&self) -> bool {
        // Only expose the Skill tool when model-invocable skills exist.
        !model_invocable_skills().is_empty()
    }

    fn needs_permissions(&self) -> bool {
        false
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Skill".to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let name = normalized_name(input);
        if name.is_empty() {
            return ValidationResult::failure("skill name is required");
        }
        // Reject disable-model-invocation skills with the original's message
        // (SkillTool.ts:412-418) before the generic unknown-skill path.
        for s in discover_skills() {
            if s.name == name && s.disable_model_invocation {
                return ValidationResult::failure(format!(
                    "Skill {name} cannot be used with {SKILL_TOOL_NAME} tool due to disable-model-invocation"
                ));
            }
        }
        if get_skill(&name).is_none() {
            let names: Vec<String> =
                model_invocable_skills().iter().map(|s| s.name.clone()).collect();
            let available =
                if names.is_empty() { "(none)".to_string() } else { names.join(", ") };
            return ValidationResult::failure(format!(
                "Unknown skill \"{name}\". Available: {available}"
            ));
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let name = normalized_name(&input);
            let args = input.get("args").and_then(|a| a.as_str()).map(|s| s.to_string());
            let Some(skill) = get_skill(&name) else {
                yield Ok(ToolResult::Result {
                    result_for_assistant: format!("Unknown skill \"{name}\"."),
                    data: Some(serde_json::json!({})),
                });
                return;
            };

            // Faithful to loadSkillsDir.ts:343-362:
            //   1. prefix "Base directory for this skill: {baseDir}\n\n" (so
            //      the model knows where the skill's bundled files live),
            //   2. substitute $name (named args) / $ARGUMENTS[n] / $n /
            //      $ARGUMENTS placeholders, passing the frontmatter argument
            //      names,
            //   3. replace ${CODEMASON_SKILL_DIR} with the skill's own directory.
            let mut content = skill.content.clone();
            if !skill.base_dir.is_empty() {
                content = format!("Base directory for this skill: {}\n\n{content}", skill.base_dir);
            }
            let mut body =
                substitute_arguments(&content, args.as_deref(), true, &skill.argument_names);
            if !skill.base_dir.is_empty() {
                body = body.replace("${CODEMASON_SKILL_DIR}", &skill.base_dir);
            }
            // Execute embedded shell commands (```! blocks and !`inline`) and
            // splice their output in — the dynamic-context feature
            // (promptShellExecution.ts).
            let body =
                execute_shell_commands_in_prompt(&body, &context, can_use_tool.as_ref()).await;

            if skill.context == "fork" {
                match SkillTool::run_forked(&skill, body, &context, can_use_tool).await {
                    Ok(report) => {
                        yield Ok(ToolResult::Result {
                            result_for_assistant: report,
                            data: Some(serde_json::json!({"skill": name, "context": "fork"})),
                        });
                    }
                    Err(e) => {
                        // get_agent_definition's unknown-agent error (the
                        // Python let the ValueError propagate to the
                        // dispatcher's error formatting).
                        yield Err(crate::tools::base::ToolError::new(e));
                    }
                }
            } else {
                // Inline: the skill body expands into the conversation; the
                // model follows it directly.
                yield Ok(ToolResult::Result {
                    result_for_assistant: body,
                    data: Some(serde_json::json!({"skill": name, "context": "inline"})),
                });
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_and_indexed_substitution() {
        let out = substitute_arguments(
            "name=$first idx=$ARGUMENTS[1] short=$0 all=$ARGUMENTS",
            Some("alpha beta"),
            true,
            &["first".to_string()],
        );
        assert_eq!(out, "name=alpha idx=beta short=alpha all=alpha beta");
    }

    #[test]
    fn blocked_placeholders_left_alone() {
        // $name followed by [ or word char must NOT substitute (Python
        // lookahead semantics).
        let out = substitute_arguments(
            "$firstname and $first[0]",
            Some("alpha"),
            false,
            &["first".to_string()],
        );
        assert_eq!(out, "$firstname and $first[0]");
        // $12x: \w blocker includes digits → verbatim.
        let out2 = substitute_arguments("$0x stays", Some("a"), false, &[]);
        assert_eq!(out2, "$0x stays");
    }

    #[test]
    fn append_when_no_placeholder() {
        let out = substitute_arguments("no placeholders here", Some("-m 'Fix'"), true, &[]);
        assert_eq!(out, "no placeholders here\n\nARGUMENTS: -m 'Fix'");
        // none appended when args empty or when a placeholder consumed them
        assert_eq!(substitute_arguments("plain", Some(""), true, &[]), "plain");
        assert_eq!(substitute_arguments("plain", None, true, &[]), "plain");
    }

    #[test]
    fn missing_index_is_empty() {
        let out = substitute_arguments("a=$5 b=$ARGUMENTS[9]", Some("one"), true, &[]);
        assert_eq!(out, "a= b=");
    }

    #[test]
    fn shlex_split_with_fallback() {
        assert_eq!(parse_arguments("-m 'Fix bug'"), vec!["-m", "Fix bug"]);
        // unterminated quote → whitespace fallback
        assert_eq!(parse_arguments("a 'b c"), vec!["a", "'b", "c"]);
        assert!(parse_arguments("   ").is_empty());
    }

    #[test]
    fn shell_marker_regexes() {
        assert!(SKILL_BLOCK_RE.is_match("```!\ngit status\n```"));
        let caps = SKILL_INLINE_RE.captures("run !`date` now").unwrap();
        assert_eq!(&caps[2], "date");
        // mid-word ! must not match
        assert!(SKILL_INLINE_RE.captures("mid!`x`").is_none());
        // start of line matches
        assert!(SKILL_INLINE_RE.captures("!`ls`").is_some());
    }
}
