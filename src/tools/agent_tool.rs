//! Agent Tool for CodeMason
//! Equivalent to codemason/tools/agent_tool.py (src/tools/AgentTool/AgentTool.tsx)

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::tools::base::{
    CanUseToolFn, PermissionResult, Tool, ToolResult, ToolStream,
};
use crate::tools::built_in_agents::{
    build_agent_description, get_agent_definition, resolve_agent_tools,
};
use crate::types::messages::ContentBlock;
use crate::types::query::{ContextOptions, Message, ToolUseContext, ValidationResult};
use crate::utils::formatting::render_markdown;

pub const TOOL_NAME: &str = "Agent";

/// Tool-use display: cap the echoed brief and wrap it (the full prompt still runs).
const MAX_PROMPT_CHARS: usize = 800;
const WRAP_COLS: usize = 80;

/// The rule also stated in the target language: an instruction written IN the
/// language it is asking for is a far stronger cue than the same sentence in
/// English (measured: the model kept writing English briefs otherwise).
fn native_prompt_rule(language: &str) -> Option<&'static str> {
    match language.trim().to_lowercase().as_str() {
        "korean" => Some("이 도구의 prompt와 description은 반드시 한국어로 작성하세요. 영어로 쓰지 마세요."),
        "japanese" => Some("このツールの prompt と description は必ず日本語で書いてください。英語で書かないでください。"),
        "chinese" => Some("此工具的 prompt 和 description 必须用中文书写，不要使用英文。"),
        _ => None,
    }
}

/// The per-turn language rule. PUBLIC because it is injected from the REPL's
/// turn message (cli_repl process_prompt), NEVER from a tool description or
/// schema: the tool block is the very front of the cached prefix, so a rule
/// that changes with the detected language invalidates the WHOLE conversation
/// behind it. Measured 2026-09-06: one flip of this string on a 121k-token
/// session cost a full re-prefill, ~123 seconds of silence.
pub fn prompt_language_rule(language: &str) -> String {
    let rule = format!(
        "IMPORTANT: write `prompt` and `description` in {language} — the user's language — never in English, however technical the task is."
    );
    match native_prompt_rule(language) {
        Some(native) => format!("{rule} {native}"),
        None => rule,
    }
}

pub struct AgentTool;

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &'static str {
        TOOL_NAME
    }

    fn aliases(&self) -> &'static [&'static str] {
        // The source tool is `name: "Agent", aliases: ["Task"]` (cli.js 2.1.88).
        // constants::prompts::LEGACY_AGENT_TOOL_NAME was ported and then wired to
        // nothing, so a tool_use naming Task matched no tool and the call died as
        // unknown. Models trained on the source emit Task. Nothing else claims it:
        // the only Task-prefixed tools are TaskOutput and TaskStop.
        &[crate::constants::prompts::LEGACY_AGENT_TOOL_NAME]
    }

    fn input_json_schema(&self) -> Value {
        // Verbatim from schemas/tool_schemas.json (the pydantic-generated
        // shape). CONSTANT — nothing session-dependent may enter here.
        //
        // The language rule used to be appended to the `prompt` field's
        // description from effective_language(). That reads the LAST user
        // message, which inside a tool loop is a tool RESULT, so the detected
        // language flipped with whatever a command happened to print: an
        // English log erased the rule, a file with Korean in it restored it.
        // Each flip rewrote the tool block, which sits ahead of the system
        // prompt and the whole history, so a 121k-token conversation lost every
        // cached token and re-prefilled for ~123 seconds. Measured 2026-09-06:
        // the two cold prefills in a 27-request session were exactly the two
        // requests where this string changed, and nothing else.
        //
        // The rule now rides the turn message (cli_repl process_prompt), where
        // cwd and the date already live for the same reason — appended, so it
        // can never move the prefix, and nearer the end, which reads stronger.
        let schema = json!({
            "additionalProperties": false,
            "description": "Input schema for Agent tool",
            "properties": {
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": null,
                    "description": "A short (3-5 word) description of the task",
                    "title": "Description"
                },
                "prompt": {
                    "description": "The task for the agent to perform",
                    "title": "Prompt",
                    "type": "string"
                },
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": null,
                    "description": "Name for the spawned agent. Makes it addressable via SendMessage({to: name}) while running.",
                    "title": "Name"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run this agent in the background. You will be notified when it completes."
                },
                "subagent_type": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": null,
                    "description": "The type of specialized agent to use for this task",
                    "title": "Subagent Type"
                }
            },
            "required": ["prompt"],
            "title": "AgentInput",
            "type": "object"
        });
        schema
    }

    async fn description(&self, _input: &Value) -> String {
        "Launch a new task".into()
    }

    async fn prompt(&self) -> String {
        // The faithful getPrompt port is static — it does not vary with
        // dangerously_skip_permissions.
        build_agent_description()
    }

    fn is_read_only(&self) -> bool {
        // A sub-agent can run write tools when policy allows, so launching one
        // is NOT side-effect-free.
        false
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        // [ours] The source Agent tool (cli.js 2.1.88) sets BOTH isReadOnly and
        // isConcurrencySafe to true. We keep is_read_only false on purpose — the
        // permission label must not call launching a write-capable agent harmless
        // — but the scheduler decision is a separate flag by design (base.rs, as
        // TaskStop already does), and letting it default to is_read_only
        // serialized every spawn while the system prompt told the model to batch
        // independent agents "so they run concurrently".
        //
        // The flag alone changes nothing: utils::subagent_gate rations spawns with
        // per-endpoint file locks and config.max_concurrent_subagents defaults to
        // 1, which is a KV budget set per install, not here.
        true
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        false
    }

    /// Spawning a sub-agent is auto-approved (AgentTool checkPermissions).
    ///
    /// The source has one branch above this — route through the auto-mode
    /// security classifier — guarded by `"external" === 'ant'`, a compile-time
    /// constant that is false in the shipped external build. Dead-code
    /// elimination leaves exactly one behaviour there: allow. There is no
    /// classifier here either, so this is that behaviour, not a relaxation.
    ///
    /// Unreachable today, and kept anyway. The permission flow is entered only
    /// when `needs_permissions()` is true, and this tool returns false there —
    /// so spawning was already never prompted, by that gate rather than this
    /// answer. Worth having regardless: the source implements the method, and
    /// `is_read_only` is deliberately false (a sub-agent can run
    /// Write/Edit/Bash when policy allows, so launching one is not
    /// side-effect-free), which means the trait default would send this tool to
    /// the interactive prompt if the `needs_permissions` gate ever moved.
    async fn check_permissions(&self, _input: &Value, _ctx: &ToolUseContext) -> PermissionResult {
        PermissionResult::allow()
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Agent".into()
    }

    async fn validate_input(&self, _input: &Value, _ctx: &ToolUseContext) -> ValidationResult {
        ValidationResult::success()
    }

    /// The brief we hand the sub-agent, wrapped to WRAP_COLS.
    /// The task's own description (AgentTool getActivityDescription).
    ///
    /// No verb added: `description` is already a 3-5 word phrase for what the
    /// sub-agent is doing, which is exactly what this line wants.
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(
            input
                .and_then(|i| i.get("description"))
                .and_then(|v| v.as_str())
                .filter(|d| !d.is_empty())
                .unwrap_or("Running task")
                .to_string(),
        )
    }

    fn render_tool_use_message(&self, input: &Value, verbose: bool) -> String {
        let prompt = input.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
        // Models sometimes emit literal "\n" inside the JSON string argument.
        let mut prompt = prompt
            .replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\r\n", "\n")
            .trim()
            .to_string();
        if !verbose && prompt.chars().count() > MAX_PROMPT_CHARS {
            let head: String = prompt.chars().take(MAX_PROMPT_CHARS).collect();
            prompt = format!("{head}...");
        }
        let mut out: Vec<String> = Vec::new();
        for line in prompt.split('\n') {
            let mut line = line.trim().to_string();
            while line.chars().count() > WRAP_COLS {
                let head: String = line.chars().take(WRAP_COLS).collect();
                out.push(head);
                line = line.chars().skip(WRAP_COLS).collect();
            }
            if !line.is_empty() {
                out.push(line);
            }
        }
        out.join("\n")
    }

    /// The sub-agent's report is markdown prose — render it.
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        let text = data.get("text").and_then(|t| t.as_str()).unwrap_or("");
        if text.trim().is_empty() {
            None
        } else {
            Some(render_markdown(text.trim()))
        }
    }

    fn call(&self, input: Value, context: ToolUseContext, can_use_tool: Option<CanUseToolFn>) -> ToolStream {
        Box::pin(async_stream::stream! {
            use crate::utils::messages::{create_assistant_message, create_user_message_text, normalize_messages};

            let prompt = input.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();
            let start_time = std::time::Instant::now();

            let mut messages: Vec<Message> = vec![Message::User(create_user_message_text(&prompt))];
            let mut tool_use_count: i64 = 0;

            // Yield initial progress.
            yield Ok(ToolResult::Progress {
                content: Some(json!({
                    "content": create_assistant_message("Initializing...", None),
                    "normalized_messages": normalize_messages(&messages)
                        .iter()
                        .map(crate::utils::serialization::to_jsonable)
                        .collect::<Vec<_>>(),
                    "result_for_assistant": "Initializing...",
                })),
                normalized_messages: Vec::new(),
            });

            // Resolve the agent definition (default general-purpose), then its
            // tool pool + system prompt.
            let subagent_type = input.get("subagent_type").and_then(|s| s.as_str());
            let agent_def = match get_agent_definition(subagent_type) {
                Ok(d) => d,
                Err(e) => {
                    yield Err(crate::tools::base::ToolError::new(e));
                    return;
                }
            };

            let skip_permissions = context.options.dangerously_skip_permissions;
            let available_tools = crate::tools::get_agent_tools().await;
            let tools = resolve_agent_tools(&agent_def, &available_tools);

            // Agent system prompt = the definition's prompt + agent notes + env.
            let model = context
                .options
                .slow_and_capable_model
                .clone()
                .unwrap_or_else(|| crate::utils::config::get_config().model_name);
            // Definition prompt + the session-constant tail (notes, env,
            // language). The tail is pinned per process — see
            // agent_prefix_sections for why it must not be rebuilt per spawn.
            let mut agent_prompt: Vec<String> = vec![(agent_def.get_system_prompt)()];
            agent_prompt.extend(crate::constants::prompts::agent_prefix_sections(&model));

            // Sub-agent context: shares the parent's abort token, callbacks and
            // input_router (a permission prompt inside the sub-agent must reach
            // the same terminal/phone the parent is holding).
            // Its own abort scope, under the parent's. clone() handed the agent the
            // parent's token, which with one agent at a time cost nothing and with
            // three is wrong both ways: cancelling one cancels every sibling and
            // the parent (subagent_gate::acquire returns None on it, so a stop
            // aimed at one agent kills a sibling that had only queued), and a
            // child could end the session. child_token() is the source's linkage
            // exactly — the parent's cancel reaches the child, never the reverse.
            //
            // ...unless it starts in the background, where the link is exactly
            // what must not exist. registerAsyncAgent is called with no parent
            // under the comment "Don't link to parent's abort controller --
            // background agents should survive when the user presses ESC to
            // cancel the main thread. They are killed explicitly via
            // chat:killAgents." ESC is a cancel of the TURN; an agent the user
            // deliberately detached is not part of that turn any more. Linking
            // it here meant pressing ESC on the main prompt silently killed
            // every background agent — the opposite of the reason for
            // backgrounding one. Stopping it stays possible and stays explicit:
            // TaskStop and kill_async_agent cancel this token directly.
            let is_background =
                input.get("run_in_background").and_then(|v| v.as_bool()) == Some(true);
            let agent_abort = if is_background {
                tokio_util::sync::CancellationToken::new()
            } else {
                context.abort_controller.child_token()
            };
            // One id for the agent and its task: the source builds the task
            // with `id: agentId`, and that id names the agent's output file.
            let agent_id = crate::utils::agent_id::create_agent_id(None);
            let mut sub_ctx = ToolUseContext {
                abort_controller: agent_abort.clone(),
                options: ContextOptions {
                    tools,
                    dangerously_skip_permissions: skip_permissions,
                    slow_and_capable_model: context.options.slow_and_capable_model.clone(),
                },
                on_token_update: context.on_token_update.clone(),
                on_thinking_update: context.on_thinking_update.clone(),
                input_router: context.input_router.clone(),
                ..Default::default()
            };
            // Distinct agent_id so agent-keyed state (TodoWrite's store) does
            // not collide with the parent's session key.
            sub_ctx.agent_id = Some(agent_id.clone());

            let description = input
                .get("description")
                .and_then(|d| d.as_str())
                .map(|d| d.to_string())
                .unwrap_or_else(|| {
                    if prompt.len() > 60 { prompt[..60].to_string() } else { prompt.clone() }
                });

            // --- run_in_background: launch it and answer at once -------------
            // The parent gets the agent's id and output path immediately and
            // carries on; the run is driven by a detached task that announces
            // its own outcome through the notification queue. Registered as
            // backgrounded from the start (registerAsyncAgent), which is also
            // what stops the foreground teardown from touching it.
            if is_background {
                let Some(can_use_tool) = can_use_tool.clone() else {
                    yield Err(crate::tools::base::ToolError::new(
                        "Agent tool requires a permission callback",
                    ));
                    return;
                };
                crate::tasks::local_agent_task::register_async_agent(
                    &agent_id,
                    &description,
                    &prompt,
                    agent_def.clone(),
                    context.tool_use_id.clone(),
                    Some(agent_abort.clone()),
                );

                // Register name -> agentId for SendMessage routing. AFTER the
                // spawn, exactly as the source does, so a spawn that fails
                // leaves no entry pointing at an agent that does not exist.
                // Foreground agents are skipped on purpose: one is blocking its
                // parent, so there is nobody left to address it.
                if let Some(given) = input.get("name").and_then(|v| v.as_str()) {
                    let given = given.trim().to_string();
                    if !given.is_empty() {
                        let id = agent_id.clone();
                        crate::state::app_state::set_app_state(move |prev| {
                            let mut next = prev.clone();
                            next.agent_name_registry.insert(given.clone(), id.clone());
                            next
                        });
                    }
                }

                let can_read_output_file = sub_ctx
                    .options
                    .tools
                    .iter()
                    .any(|t| matches!(t.name(), "Read" | "Bash"));
                let output_file = crate::utils::task::disk_output::get_task_output_path(&agent_id)
                    .to_string_lossy()
                    .into_owned();

                let detached_id = agent_id.clone();
                let detached_description = description.clone();
                let detached_tool_use_id = context.tool_use_id.clone();
                let detached_abort = agent_abort.clone();
                let detached_messages = messages.clone();
                // Arc clones — the pool itself is shared, not copied. Taken
                // before sub_ctx moves into the detached run.
                let detached_tools = sub_ctx.options.tools.clone();
                let detached_prompt_sections = agent_prompt.clone();
                tokio::spawn(async move {
                    // The slot is taken in here, not in the call: waiting for
                    // it there would make a background launch block the parent,
                    // which is the one thing it must not do.
                    let mut slot = crate::utils::subagent_gate::try_acquire();
                    if slot.is_none() {
                        slot = crate::utils::subagent_gate::acquire(&detached_abort).await;
                        if slot.is_none() {
                            // Stopped while queued. Still terminal, still
                            // announced — the parent is waiting for word either
                            // way.
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
                        messages: detached_messages,
                        system_prompt: detached_prompt_sections,
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
                            // Launched from scratch — nothing to carry over.
                            initial_messages: Vec::new(),
                        },
                    )
                    .await;
                });

                let launched = json!({
                    "isAsync": true,
                    "status": "async_launched",
                    "agentId": agent_id,
                    "description": description,
                    "prompt": prompt,
                    "outputFile": output_file,
                    "canReadOutputFile": can_read_output_file,
                });
                yield Ok(ToolResult::Result {
                    result_for_assistant: serde_json::to_string(&launched).unwrap_or_default(),
                    data: Some(launched),
                });
                return;
            }

            // Register the agent as a `local_agent` task for its whole life
            // (EXK). Deliberately BEFORE the slot wait below: the gate is
            // codemason's own addition, and an agent queued on it is already
            // alive as far as the parent is concerned. Registering after the
            // wait would leave a queued agent invisible to the parent and
            // unstoppable by TaskStop — while registering here makes a stop
            // land, because the gate waits on this agent's own token and
            // kill_async_agent cancels it.
            let agent_task = crate::tasks::local_agent_task::register_agent_foreground(
                &agent_id,
                &description,
                &prompt,
                agent_def.clone(),
                context.tool_use_id.clone(),
                Some(agent_abort.clone()),
                None,
            );
            // Settles and removes the task however this stream ends, including
            // being dropped mid-run.
            let _task_guard = crate::tasks::local_agent_task::AgentTaskGuard::new(
                &agent_id,
                agent_task.cancel_auto_background.clone(),
            );

            // The running tally the parent reads this agent by. Stored on the
            // task after every message, which is what lets the parent see an
            // agent's context grow — and stop one that is growing where it
            // should not be.
            //
            // The source gates this store on an SDK opt-in in the foreground
            // branch and stores unconditionally in the backgrounded one. There
            // is no SDK consumer to ask here and the task store IS our
            // consumer, so it is unconditional, matching the branch whose
            // reader is local.
            let mut tracker = crate::tasks::local_agent_task::create_progress_tracker();

            let Some(can_use_tool) = can_use_tool else {
                yield Err(crate::tools::base::ToolError::new(
                    "Agent tool requires a permission callback",
                ));
                return;
            };

            // One sub-agent at a time (config.max_concurrent_subagents), across
            // every codemason process on this machine — a second one evicts the
            // parent from a KV pool that holds 1.62 conversations. Say so while
            // waiting: an unattended run must never look hung.
            let _slot = {
                let mut slot = crate::utils::subagent_gate::try_acquire();
                if slot.is_none() {
                    yield Ok(ToolResult::Progress {
                        content: Some(json!({
                            "normalized_messages": [],
                            "result_for_assistant": "Waiting for another sub-agent to finish (max_concurrent_subagents)",
                        })),
                        normalized_messages: Vec::new(),
                    });
                    slot = crate::utils::subagent_gate::acquire(&agent_abort).await;
                }
                match slot {
                    Some(s) => s,
                    None => {
                        yield Err(crate::tools::base::ToolError::new("Aborted while waiting for a sub-agent slot"));
                        return;
                    }
                }
            };

            // Kept back for the resumed run: if this agent is backgrounded
            // mid-flight, the detached continuation opens a second query from
            // the same prompt, permission callback and sub-context — on a
            // detached abort token, swapped in at the hand-over below, because
            // it stops being the parent's agent at that moment.
            let resume_prompt = agent_prompt.clone();
            let resume_can_use_tool = can_use_tool.clone();
            let mut resume_ctx = sub_ctx.clone();
            // "Reading src/foo.rs" rather than "Read" in the tally the parent
            // reads. Resolved once per run: the pool does not change, and the
            // description is stored on each tool use as it is recorded.
            let activity_tools = sub_ctx.options.tools.clone();
            let resolve_activity =
                crate::tasks::local_agent_task::create_activity_description_resolver(
                    activity_tools.clone(),
                );

            let mut stream = crate::query::query(crate::query::QueryParams {
                messages: messages.clone(),
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

            // Every pull is raced against this agent's background signal
            // (EXK's backgroundSignal), set either by the auto-background timer
            // the spawn armed or by an explicit background_agent_task. One
            // future for the whole run, not one per pull — the source's own
            // comment warns that re-arming it each iteration "accumulat[es]
            // callbacks for the lifetime of the agent".
            let background_signal = agent_task.background_signal.clone();
            let notified = background_signal.notified();
            tokio::pin!(notified);
            let mut racing = true;
            let mut was_backgrounded = false;
            let mut slot_holder = Some(_slot);

            loop {
                let pulled = if racing {
                    tokio::select! {
                        _ = &mut notified => {
                            // The signal won. Re-read the task: the signal and
                            // the flag are set together, so a signal without
                            // the flag is not a backgrounding and must not be
                            // treated as one.
                            let backgrounded = crate::state::app_state::get_app_state()
                                .tasks
                                .get(&agent_id)
                                .and_then(|t| t.as_local_agent())
                                .map(|a| a.is_backgrounded)
                                .unwrap_or(false);
                            if !backgrounded {
                                // Give up on the race rather than re-arming a
                                // one-shot signal, and pull again. Nothing is
                                // lost: StreamExt::next is cancel-safe, so the
                                // dropped pull left the stream exactly where it
                                // was — which is why this needs none of the
                                // "hold the pending pull" care the Python side
                                // takes with a cancelled task.
                                racing = false;
                                continue;
                            }
                            was_backgrounded = true;
                            None
                        }
                        message = stream.next() => message,
                    }
                } else {
                    stream.next().await
                };

                if was_backgrounded {
                    // Drop the foreground stream so the query's own cleanup
                    // runs, then hand the conversation over — minus any tool
                    // call that never came back, because the switch can land
                    // mid-tool and an orphaned tool_use is rejected outright
                    // when the conversation is sent again.
                    //
                    // The source restarts the agent from its original prompt
                    // here, because its promptMessages array is never appended
                    // to. codemason's `messages` IS the live transcript (the
                    // loop below pushes every turn onto it) and query() takes
                    // it by value, so handing it back RESUMES the agent instead
                    // of making it redo work it has already paid for.
                    drop(stream);
                    let resume_messages =
                        crate::tools::agent_tool_utils::filter_incomplete_tool_calls(&messages);

                    // Read off the sub-context before it moves into the
                    // detached run: whether the parent can look at the partial
                    // output itself, rather than only being told where it is.
                    let can_read_output_file = resume_ctx
                        .options
                        .tools
                        .iter()
                        .any(|t| matches!(t.name(), "Read" | "Bash"));

                    // It is not the parent's agent any more: unlink it from
                    // the parent's scope so ESC on the main thread cancels the
                    // turn without taking this agent with it. The source
                    // arrives here by re-launching the continuation with
                    // `task.abortController`, which registerAgentForeground
                    // created unlinked; a CancellationToken cannot be detached
                    // in place, so detach_agent_abort mints a fresh root one
                    // and stores it on the task — which is what keeps the
                    // explicit stop working, since kill_async_agent cancels
                    // whatever the task is holding.
                    let detached_abort =
                        crate::tasks::local_agent_task::detach_agent_abort(&agent_id);
                    resume_ctx.abort_controller = detached_abort.clone();
                    let detached_id = agent_id.clone();
                    let detached_description = description.clone();
                    let detached_tool_use_id = context.tool_use_id.clone();
                    let detached_messages = resume_messages.clone();
                    let detached_slot = slot_holder.take();
                    let detached_tools = activity_tools.clone();
                    tokio::spawn(async move {
                        let stream = crate::query::query(crate::query::QueryParams {
                            messages: detached_messages,
                            system_prompt: resume_prompt,
                            context: BTreeMap::new(),
                            can_use_tool: resume_can_use_tool,
                            tool_use_context: resume_ctx,
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
                                // The slot travels with the agent: it is still
                                // running and still occupying the pool.
                                slot: detached_slot,
                                tools: detached_tools,
                                initial_messages: resume_messages,
                            },
                        )
                        .await;
                    });

                    let launched = json!({
                        "isAsync": true,
                        "status": "async_launched",
                        "agentId": agent_id,
                        "description": description,
                        "prompt": prompt,
                        "outputFile": crate::utils::task::disk_output::get_task_output_path(&agent_id)
                            .to_string_lossy()
                            .into_owned(),
                        "canReadOutputFile": can_read_output_file,
                    });
                    yield Ok(ToolResult::Result {
                        result_for_assistant: serde_json::to_string(&launched).unwrap_or_default(),
                        data: Some(launched),
                    });
                    return;
                }

                let Some(message) = pulled else { break };
                messages.push(message.clone());

                // The agent's own output file, so a parent that reads this
                // agent mid-run with TaskOutput sees what it is doing.
                crate::tasks::local_agent_task::append_agent_transcript(
                    &agent_id,
                    &serde_json::to_value(&message).unwrap_or(serde_json::Value::Null),
                );

                // Fold the message into the tally and publish it. Every
                // message, not only the ones carrying a tool use: the token
                // count is what the parent watches, and it moves on messages
                // that use no tool.
                crate::tasks::local_agent_task::update_progress_from_message(
                    &mut tracker, &message, Some(&resolve_activity),
                );
                crate::tasks::local_agent_task::update_agent_progress(
                    &agent_id,
                    crate::tasks::local_agent_task::get_progress_update(&tracker),
                );
                // The prose line, after the counts: update_agent_progress
                // preserves an existing summary rather than overwriting it, so
                // the fresh one has to be written through the path that owns it.
                if let Some(summary) =
                    crate::tasks::local_agent_task::derive_agent_summary(&tracker)
                {
                    crate::tasks::local_agent_task::update_agent_summary(&agent_id, &summary);
                }
                tool_use_count = tracker.tool_use_count;

                if let Message::Assistant(a) = &message {
                    for content in &a.message.content {
                        if let ContentBlock::ToolUse(tu) = content {
                            yield Ok(ToolResult::Progress {
                                content: Some(json!({
                                    "normalized_messages": normalize_messages(&messages)
                                        .iter()
                                        .map(crate::utils::serialization::to_jsonable)
                                        .collect::<Vec<_>>(),
                                    "tool_name": tu.name,
                                    "tool_input": tu.input,
                                    "tool_use_count": tool_use_count,
                                    "result_for_assistant": format!("Agent → {} ({tool_use_count})", tu.name),
                                })),
                                normalized_messages: Vec::new(),
                            });
                        }
                    }
                }
            }

            // Final result: walk BACKWARD to the most recent assistant message
            // that HAS text blocks (finalizeAgentTool) — taking only the last
            // message loses the real answer when the loop ends on a tool_use turn.
            let elapsed = start_time.elapsed().as_secs_f64();
            // The backward walk lives in agent_tool_utils now: the background
            // lifecycle needs the same answer out of the same messages, and two
            // copies of "which message actually holds the result" is exactly
            // the kind of duplication that drifts.
            let combined_text =
                crate::tools::agent_tool_utils::extract_partial_result(&messages)
                    .unwrap_or_default();
            let text_blocks: Vec<String> = if combined_text.is_empty() {
                Vec::new()
            } else {
                vec![combined_text.clone()]
            };

            // The run is over: settle the task before the final yield, so a
            // consumer that reads task state on receiving the result sees a
            // finished task rather than a running one. A cancel is not a
            // failure — the source reports it as "user_cancel_sync", and the
            // task's own status has to say which, because that is what the
            // parent reads back.
            if agent_abort.is_cancelled() {
                crate::tasks::local_agent_task::kill_async_agent(&agent_id);
            } else {
                let progress = crate::tasks::local_agent_task::get_progress_update(&tracker);
                crate::tasks::local_agent_task::complete_agent_task(
                    &agent_id,
                    json!({
                        "agent_id": agent_id,
                        "text": combined_text,
                        "tool_use_count": progress.tool_use_count,
                        "duration": elapsed,
                        // What the source puts in the completion's usage block:
                        // the agent's final context size, its tool count and
                        // how long it took.
                        "usage": {
                            "total_tokens": progress.token_count,
                            "tool_uses": progress.tool_use_count,
                            "duration_ms": (elapsed * 1000.0) as i64,
                        },
                    }),
                );
            }

            if !combined_text.is_empty() {
                yield Ok(ToolResult::Result {
                    result_for_assistant: combined_text.clone(),
                    data: Some(json!({
                        // 'text' is what render_tool_result_message displays.
                        "text": combined_text,
                        "text_blocks": text_blocks,
                        "tool_use_count": tool_use_count,
                        "duration": elapsed,
                    })),
                });
            } else {
                // No assistant message had any text — synthetic summary so the
                // parent still receives a non-empty result.
                let result_text = format!("Done ({tool_use_count} tool uses, {elapsed:.1}s)");
                yield Ok(ToolResult::Result {
                    result_for_assistant: result_text,
                    data: Some(json!({"tool_use_count": tool_use_count, "duration": elapsed})),
                });
            }
        })
    }
}

/// The registered instance (tools/mod.rs ALL_TOOLS).
pub fn tool() -> Arc<dyn Tool> {
    Arc::new(AgentTool)
}

#[cfg(test)]
mod tests {

    /// A parent's cancel reaches its sub-agents; a sub-agent's reaches no one else.
    ///
    /// While one ran at a time, cloning the parent's token cost nothing. With
    /// three it decides whether stopping one is possible: subagent_gate::acquire
    /// returns None on the token it is given, so a stop that reached a sibling
    /// would cancel an agent that had only queued.
    #[test]
    fn a_stop_aimed_at_one_agent_spares_its_siblings() {
        use tokio_util::sync::CancellationToken;
        let parent = CancellationToken::new();
        let a = parent.child_token();
        let b = parent.child_token();
        assert!(!a.is_cancelled() && !b.is_cancelled());

        a.cancel();
        assert!(a.is_cancelled(), "the agent that was stopped");
        assert!(!b.is_cancelled(), "its sibling keeps running");
        assert!(!parent.is_cancelled(), "and so does the parent");

        parent.cancel();
        assert!(b.is_cancelled(), "a parent's stop reaches every child");

        let late = parent.child_token();
        assert!(late.is_cancelled(), "a child of a stopped parent starts stopped");

        // A forked skill inside a sub-agent is a token under a token.
        assert!(b.child_token().is_cancelled(), "the link is transitive");
    }

    /// A FOREGROUND spawn hands the sub-agent a child, and waits for the slot
    /// on that child — not on the parent, which is what made a queued sibling
    /// die. A BACKGROUND one is deliberately not a child at all: it has to
    /// outlive the turn, so it starts on a root token and the hand-over swaps
    /// the moved agent onto one (detach_agent_abort). A resumed agent is a
    /// background relaunch and follows the same rule.
    #[test]
    fn the_spawn_paths_link_the_foreground_only() {
        let agent = include_str!("agent_tool.rs");
        assert!(agent.contains("context.abort_controller.child_token()"));
        assert!(agent.contains("subagent_gate::acquire(&agent_abort)"));
        assert!(
            agent.contains("let agent_abort = if is_background {"),
            "the background branch must choose its own root token"
        );
        assert!(
            agent.contains("detach_agent_abort(&agent_id)"),
            "and the hand-over must cut the moved agent loose"
        );
        let resume = include_str!("resume_agent.rs");
        assert!(
            !resume.contains("context.abort_controller.child_token()"),
            "a resumed agent must not die with the turn that resumed it"
        );
        let skill = include_str!("skill_tool.rs");
        assert!(skill.contains("context.abort_controller.child_token()"));
        assert!(skill.contains("subagent_gate::acquire(&skill_abort)"));
        // Built from pieces: spelling it out would plant the needle in this very
        // file, which include_str! then reads back.
        let handover = format!("abort_controller: context.{}", "abort_controller.clone()");
        assert!(
            !skill.contains(&handover) && !agent.contains(&handover),
            "no path may hand over the parent's own token"
        );
    }

    /// The backend falls back to tool_prompts::tool_description only when a tool
    /// hands it nothing. This one hands it the built description, which is why
    /// the map has no Agent entry to drift out of.
    #[tokio::test]
    async fn the_tool_needs_no_fallback_description() {
        let d = AgentTool.wire_description().await.expect("a wire description");
        assert!(!d.is_empty());
        assert_eq!(d, build_agent_description());
        assert_eq!(
            crate::constants::tool_prompts::tool_description(AgentTool.name()),
            None,
            "and the fallback is unreachable for it"
        );
    }

    /// The source gives this tool two names; a tool_use may arrive under either.
    #[test]
    fn the_tool_answers_to_task() {
        let t = AgentTool;
        assert!(t.matches_name("Agent"));
        assert!(t.matches_name("Task"), "the source aliases Agent as Task");
        assert!(!t.matches_name("TaskStop"), "a prefix is not an alias");
    }
    use super::*;

    #[test]
    fn wraps_and_caps_prompt() {
        let long = "x".repeat(200);
        let input = json!({"prompt": long});
        let rendered = AgentTool.render_tool_use_message(&input, false);
        assert!(rendered.lines().all(|l| l.chars().count() <= WRAP_COLS));
        assert_eq!(rendered.lines().count(), 3); // 80 + 80 + 40
    }

    #[test]
    fn schema_matches_dump_shape() {
        let s = AgentTool.input_json_schema();
        assert_eq!(s["title"], "AgentInput");
        assert_eq!(s["required"][0], "prompt");
        assert_eq!(s["additionalProperties"], false);
    }
}
