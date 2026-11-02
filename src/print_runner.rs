//! Non-interactive print mode (`-p` / `--print`) — adapted from
//! cli/print.ts runHeadless. Equivalent to codemason/print_runner.py.
//!
//! Implements the SDK OUTPUT CONTRACT across the three output formats:
//!   - text        : the final assistant text, plain (default).
//!   - json        : a single `result` message object (the SDK result schema).
//!   - stream-json : NDJSON — a system/init message, then each assistant/user
//!                   message as it streams, then the final `result` message.
//! Field-by-field substrate notes live in the Python reference.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::{json, Value};

use crate::query::{query, QueryParams};
use crate::tools::base::{CanUseToolFn, Tool};
use crate::types::messages::ContentBlock;
use crate::types::query::{ContextOptions, Message, ToolUseContext};
use crate::utils::messages::create_user_message_text;
use crate::utils::serialization::to_jsonable;

pub const VALID_OUTPUT_FORMATS: &[&str] = &["text", "json", "stream-json"];
pub const VALID_INPUT_FORMATS: &[&str] = &["text", "stream-json"];

fn empty_usage() -> Value {
    json!({
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    })
}

/// Write one NDJSON line to stdout (stream-json / json output).
fn emit(obj: &Value) {
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{}", serde_json::to_string(obj).unwrap_or_default());
    let _ = out.flush();
}

fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// system/init (buildSystemInitMessage), adapted to the local substrate.
pub fn build_init_message(
    tools: &[Arc<dyn Tool>],
    model: &str,
    skip_permissions: bool,
    session_id: &str,
) -> Value {
    let skills: Vec<String> = crate::utils::skills::model_invocable_skills()
        .into_iter()
        .map(|s| s.name)
        .collect();
    let agents: Vec<String> = crate::tools::built_in_agents::get_built_in_agents()
        .into_iter()
        .map(|a| a.agent_type)
        .collect();
    json!({
        "type": "system",
        "subtype": "init",
        "cwd": crate::utils::state::get_cwd(),
        "session_id": session_id,
        "tools": tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
        "mcp_servers": [],
        "model": model,
        "permissionMode": if skip_permissions { "bypassPermissions" } else { "default" },
        "slash_commands": [],
        "agents": agents,
        "skills": skills,
        "uuid": new_uuid(),
    })
}

/// Wrap a streamed assistant/user message in the SDK envelope.
fn wrap_message(msg: &Message, msg_type: &str, session_id: &str) -> Value {
    let (inner, uuid) = match msg {
        Message::Assistant(a) => (to_jsonable(&a.message), a.uuid.clone()),
        Message::User(u) => (to_jsonable(&u.message), u.uuid.clone()),
        Message::Progress(_) => (Value::Null, new_uuid()),
    };
    json!({
        "type": msg_type,
        "message": inner,
        "parent_tool_use_id": Value::Null,
        "session_id": session_id,
        "uuid": if uuid.is_empty() { new_uuid() } else { uuid },
    })
}

/// The SDK `result` message (cli/print.ts result schema).
#[allow(clippy::too_many_arguments)]
pub fn build_result_message(
    subtype: &str,
    result_text: Option<&str>,
    num_turns: i64,
    usage: Option<&Value>,
    duration_ms: i64,
    session_id: &str,
    is_error: bool,
    errors: Option<&[String]>,
) -> Value {
    let mut msg = json!({
        "type": "result",
        "subtype": subtype,
        "duration_ms": duration_ms,
        "duration_api_ms": 0,
        "is_error": is_error,
        "num_turns": num_turns,
        "session_id": session_id,
        "total_cost_usd": 0,
        "usage": usage.cloned().unwrap_or_else(empty_usage),
        "modelUsage": {},
        "permission_denials": [],
        "uuid": new_uuid(),
    });
    if !is_error {
        msg["result"] = Value::String(result_text.unwrap_or("").to_string());
    }
    if let Some(errs) = errors {
        if !errs.is_empty() {
            msg["errors"] = json!(errs);
        }
    }
    msg
}

/// Pull text blocks + usage from a codemason assistant message.
fn extract_text_and_usage(msg: &Message) -> (String, Option<Value>) {
    let Message::Assistant(a) = msg else {
        return (String::new(), None);
    };
    let parts: Vec<String> = a
        .message
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) if !t.text.is_empty() => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    (parts.join("\n"), Some(to_jsonable(&a.message.usage)))
}

/// Resolve the prompt: from the positional arg, else stdin (NDJSON of SDK user
/// messages for input_format='stream-json'). Returns None when no input.
pub fn read_prompt_from_input(prompt_arg: &str, input_format: &str) -> Option<String> {
    if !prompt_arg.is_empty() {
        return Some(prompt_arg.to_string());
    }
    use std::io::{IsTerminal, Read};
    if std::io::stdin().is_terminal() {
        return None;
    }
    let mut raw = String::new();
    if std::io::stdin().read_to_string(&mut raw).is_err() {
        return None;
    }
    if input_format == "stream-json" {
        let mut texts: Vec<String> = Vec::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(obj) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let content = obj.get("message").and_then(|m| m.get("content"));
            match content {
                Some(Value::String(s)) => texts.push(s.clone()),
                Some(Value::Array(a)) => {
                    for b in a {
                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let joined = texts.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n");
        let joined = joined.trim().to_string();
        return if joined.is_empty() { None } else { Some(joined) };
    }
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn now_ms() -> i64 {
    crate::task::now_ms()
}

/// Run one non-interactive turn and emit it in the requested format. Returns
/// the process exit code (0 ok, 1 error).
pub async fn run_print(
    prompt: &str,
    tools: Vec<Arc<dyn Tool>>,
    output_format: &str,
    verbose: bool,
    dangerously_skip_permissions: bool,
    resume_session_id: Option<&str>,
    can_use_tool: CanUseToolFn,
) -> i32 {
    // AskUserQuestion is withheld from -p for the same reason it is withheld
    // from sub-agents: nobody is on the other end. The local half reads a stdin
    // that is normally closed, and the remote half puts the question on the
    // phone and waits out remote_ask_timeout (1800s) before giving up. A -p
    // caller wants an answer or a stated inability, not a half-hour pause on a
    // question it cannot see. The interactive REPL keeps the tool.
    let tools: Vec<Arc<dyn Tool>> = tools
        .into_iter()
        .filter(|t| t.name() != crate::constants::prompts::ASK_USER_QUESTION_TOOL_NAME)
        .collect();

    // Self-heal a stale/renamed model id before the first request.
    crate::utils::tokens::resolve_model();
    let model = crate::utils::config::get_config().model_name;

    // Resume: restore the stored conversation and keep appending.
    let mut restored: Vec<Message> = Vec::new();
    if let Some(rid) = resume_session_id {
        match crate::utils::session_storage::load_conversation_for_resume(rid) {
            Ok(r) => {
                crate::utils::session_storage::adopt_resumed_session(rid);
                restored = crate::utils::messages::messages_from_transcript(&r);
            }
            Err(e) => {
                eprintln!("Error: --resume {rid}: {e}");
                return 1;
            }
        }
    }
    let session_id = crate::utils::state::get_session_id();
    let is_stream = output_format == "stream-json";
    let is_json = output_format == "json";

    if is_stream {
        emit(&build_init_message(&tools, &model, dangerously_skip_permissions, &session_id));
    }

    let system_prompt = crate::constants::prompts::get_system_prompt(&tools, Some(&model), None).await;
    let mut context: BTreeMap<String, String> = BTreeMap::new();
    context.insert("cwd".into(), crate::utils::state::get_cwd());
    context.insert("date".into(), chrono::Local::now().format("%Y-%m-%d").to_string());
    let goal = crate::utils::goal::render_goal_context();
    if !goal.is_empty() {
        context.insert("active_goal".into(), goal);
    }

    let tool_use_context = ToolUseContext {
        options: ContextOptions {
            tools: tools.clone(),
            dangerously_skip_permissions,
            slow_and_capable_model: None,
        },
        ..Default::default()
    };

    let mut messages: Vec<Message> = restored.clone();
    messages.push(Message::User(create_user_message_text(prompt)));
    crate::utils::session_storage::record_transcript(&messages, false);

    let mut text_parts: Vec<String> = Vec::new();
    let mut total_usage: Option<serde_json::Map<String, Value>> = None;
    let mut num_turns: i64 = 0;
    let mut tool_counts: BTreeMap<String, i64> = BTreeMap::new();
    let mut is_error = false;
    let mut errors: Vec<String> = Vec::new();
    let start = now_ms();

    let mut stream = query(QueryParams {
        messages: messages.clone(),
        system_prompt: system_prompt.clone(),
        context: context.clone(),
        can_use_tool,
        tool_use_context: tool_use_context.clone(),
        max_output_recovery_count: 0,
            repetition_recovery_count: 0,
            suppress_reasoning: false,
            tool_parse_recovery_count: 0,
            empty_turn_recovery_count: 0,
        stop_hook_active: false,
        turn_count: 0,
    });

    while let Some(message) = stream.next().await {
        match &message {
            Message::Assistant(a) => {
                num_turns += 1;
                for b in &a.message.content {
                    if let ContentBlock::ToolUse(tu) = b {
                        if !tu.name.is_empty() {
                            *tool_counts.entry(tu.name.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let (text, usage) = extract_text_and_usage(&message);
                if !text.is_empty() {
                    text_parts.push(text);
                }
                if let Some(Value::Object(u)) = usage {
                    // ACCUMULATE across turns (SDK result.usage is cumulative).
                    match &mut total_usage {
                        None => total_usage = Some(u),
                        Some(acc) => {
                            for (k, v) in u {
                                if let Some(n) = v.as_i64() {
                                    let cur = acc.get(&k).and_then(|x| x.as_i64()).unwrap_or(0);
                                    acc.insert(k, json!(cur + n));
                                } else {
                                    acc.entry(k).or_insert(v);
                                }
                            }
                        }
                    }
                }
                messages.push(message.clone());
                crate::utils::session_storage::record_transcript(&messages, false);
                if is_stream {
                    emit(&wrap_message(&message, "assistant", &session_id));
                }
            }
            Message::User(_) => {
                messages.push(message.clone());
                crate::utils::session_storage::record_transcript(&messages, false);
                if is_stream {
                    emit(&wrap_message(&message, "user", &session_id));
                }
            }
            Message::Progress(_) => {} // not part of the SDK stream — skip
        }
    }

    // waitingForAgents: keep draining background-task completions and do NOT
    // exit until no background task is running AND the queue is empty. Process
    // exit SIGTERMs still-running tasks, so exiting early would drop a
    // completion.
    loop {
        for block in crate::utils::message_queue_manager::drain_pending_notifications() {
            let nmsg = create_user_message_text(&block);
            messages.push(Message::User(nmsg.clone()));
            crate::utils::session_storage::record_transcript(&messages, false);
            if is_stream {
                emit(&wrap_message(&Message::User(nmsg), "user", &session_id));
            }
        }
        let state = crate::state::app_state::get_app_state();
        let has_running_bg = crate::utils::task::framework::get_running_tasks(&state)
            .iter()
            .any(|t| t.is_backgrounded());
        let pending = crate::utils::message_queue_manager::has_pending_notifications();
        if !has_running_bg && !pending {
            break;
        }
        if !pending {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    let duration_ms = now_ms() - start;
    let result_text = text_parts.join("\n").trim().to_string();

    // Best-effort remote notify; never affects the exit code.
    let error_str = if is_error && !errors.is_empty() {
        errors[0].clone()
    } else if is_error {
        "execution error".to_string()
    } else {
        String::new()
    };
    let cwd = crate::utils::state::get_cwd();
    let tools_map: std::collections::HashMap<String, i64> =
        tool_counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let status = crate::utils::session_status::status_line(&messages);
    crate::remote::hooks::notify_turn_end(
        duration_ms,
        &result_text,
        crate::remote::hooks::TurnEndInfo {
            cwd: Some(&cwd),
            error: &error_str,
            tools: Some(tools_map),
            prompt,
            status: &status,
            ..Default::default()
        },
    )
    .await;

    let usage_val = total_usage.map(Value::Object);
    if is_stream || is_json {
        emit(&build_result_message(
            if is_error { "error_during_execution" } else { "success" },
            Some(&result_text),
            num_turns,
            usage_val.as_ref(),
            duration_ms,
            &session_id,
            is_error,
            if errors.is_empty() { None } else { Some(&errors) },
        ));
    } else {
        // text
        if is_error {
            eprintln!("Error: {}", errors.first().map(|s| s.as_str()).unwrap_or("execution error"));
        } else {
            println!("{result_text}");
        }
    }

    let _ = verbose;
    if is_error {
        1
    } else {
        0
    }
}
