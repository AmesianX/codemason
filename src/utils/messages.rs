//! Message utilities for CodeMason
//! Equivalent to codemason/utils/messages.py (src/utils/messages.tsx)

use std::collections::HashSet;
use serde_json::Value;
use uuid::Uuid;

use crate::types::messages::{
    ApiMessage, BetaUsage, ContentBlock, ContentBlockParam, MessageContent, MessageParam,
};
use crate::types::query::{Message, UserMessage};

/// Verbatim from utils/messages.ts:207-209 — the SINGLE source of truth.
pub const INTERRUPT_MESSAGE: &str = "[Request interrupted by user]";
pub const INTERRUPT_MESSAGE_FOR_TOOL_USE: &str = "[Request interrupted by user for tool use]";

/// constants/messages.ts NO_CONTENT_MESSAGE.
pub const NO_CONTENT_MESSAGE: &str = "(no content)";

/// Verbatim from utils/messages.ts:213 — the tool_result the model receives
/// when the USER rejects a tool at the permission prompt. The 'STOP … wait for
/// the user' phrasing is load-bearing: a weak 'User denied permission' reads
/// as an environment error, and the model keeps retrying instead of stopping.
pub const REJECT_MESSAGE: &str = "The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.";

/// Create a user message from plain text.
/// Wrap text as a system reminder (wrapInSystemReminder).
pub fn wrap_in_system_reminder(content: &str) -> String {
    format!("<system-reminder>\n{content}\n</system-reminder>")
}

/// Wrap each message's text in a system reminder (wrapMessagesInSystemReminder).
///
/// String content is wrapped whole; block content has each text block wrapped
/// and every other block left alone.
pub fn wrap_messages_in_system_reminder(messages: Vec<UserMessage>) -> Vec<UserMessage> {
    messages
        .into_iter()
        .map(|mut msg| {
            match &msg.message.content {
                crate::types::messages::MessageContent::Text(t) => {
                    msg.message.content =
                        crate::types::messages::MessageContent::Text(wrap_in_system_reminder(t));
                }
                crate::types::messages::MessageContent::Blocks(blocks) => {
                    let wrapped = blocks
                        .iter()
                        .map(|b| match serde_json::to_value(b) {
                            Ok(mut v) if v.get("type").and_then(|t| t.as_str()) == Some("text") => {
                                if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                                    v["text"] = serde_json::Value::from(wrap_in_system_reminder(text));
                                }
                                serde_json::from_value(v).unwrap_or_else(|_| b.clone())
                            }
                            _ => b.clone(),
                        })
                        .collect();
                    msg.message.content = crate::types::messages::MessageContent::Blocks(wrapped);
                }
            }
            msg
        })
        .collect()
}

/// Frame a message that arrived mid-turn, by who sent it (wrapCommandText).
///
/// The framing is the instruction: a queued line is not just context, it has to
/// say what the reader is expected to do about it and when.
pub fn wrap_command_text(raw: &str, origin: Option<&str>) -> String {
    match origin {
        Some("task-notification") => format!("A background agent completed a task:\n{raw}"),
        Some("coordinator") => format!(
            "The coordinator sent a message while you were working:\n{raw}\n\n\
             Address this before completing your current task."
        ),
        _ => format!(
            "The user sent a new message while you were working:\n{raw}\n\n\
             IMPORTANT: After completing your current task, you MUST address the \
             user's message above. Do not ignore it."
        ),
    }
}

/// A task_status attachment, as the messages the model reads
/// (the `task_status` arm of getAttachmentMessages).
///
/// Only emitted after a compaction, where the message that spawned the agent is
/// gone — which is why the running branch says not to spawn a duplicate, and
/// why it names SendMessage: the model has just lost the record of the spawn,
/// so it needs to be told the agent is still reachable, not only readable.
pub fn render_task_status_attachment(attachment: &serde_json::Value) -> Vec<UserMessage> {
    let string_of = |key: &str| {
        attachment
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let optional = |key: &str| {
        attachment
            .get(key)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(|v| v.to_string())
    };

    let status = string_of("status");
    let task_id = string_of("taskId");
    let description = string_of("description");
    let delta_summary = optional("deltaSummary");
    let output_file_path = optional("outputFilePath");
    let display_status = if status == "killed" { "stopped" } else { status.as_str() };

    let meta = |text: String| {
        let mut message = create_user_message_text(&wrap_in_system_reminder(&text));
        message.is_meta = true;
        vec![message]
    };

    // For stopped tasks, keep it brief — the work was interrupted and the raw
    // transcript delta isn't useful context.
    if status == "killed" {
        return meta(format!(
            "Task \"{description}\" ({task_id}) was stopped by the user."
        ));
    }

    // For running tasks, warn against spawning a duplicate.
    if status == "running" {
        let mut parts = vec![format!(
            "Background agent \"{description}\" ({task_id}) is still running."
        )];
        if let Some(summary) = &delta_summary {
            parts.push(format!("Progress: {summary}"));
        }
        match &output_file_path {
            Some(path) => parts.push(format!(
                "Do NOT spawn a duplicate. You will be notified when it completes. \
                 You can read partial output at {path} or send it a message with {}.",
                crate::tools::send_message_tool::SEND_MESSAGE_TOOL_NAME
            )),
            None => parts.push(format!(
                "Do NOT spawn a duplicate. You will be notified when it completes. \
                 You can check its progress with the {} tool or send it a message \
                 with {}.",
                crate::tools::task_output_tool::TASK_OUTPUT_TOOL_NAME,
                crate::tools::send_message_tool::SEND_MESSAGE_TOOL_NAME
            )),
        }
        return meta(parts.join(" "));
    }

    // For completed/failed tasks, include the full delta.
    let mut parts = vec![
        format!("Task {task_id}"),
        format!("(type: {})", string_of("taskType")),
        format!("(status: {display_status})"),
        format!("(description: {description})"),
    ];
    if let Some(summary) = &delta_summary {
        parts.push(format!("Delta: {summary}"));
    }
    match &output_file_path {
        Some(path) => parts.push(format!(
            "Read the output file to retrieve the result: {path}"
        )),
        None => parts.push(format!(
            "You can check its output using the {} tool.",
            crate::tools::task_output_tool::TASK_OUTPUT_TOOL_NAME
        )),
    }
    meta(parts.join(" "))
}

pub fn create_user_message_text(content: &str) -> UserMessage {
    UserMessage {
        uuid: Uuid::new_v4().to_string(),
        message: MessageParam {
            role: "user".into(),
            content: MessageContent::Blocks(vec![ContentBlockParam::text(content)]),
        },
        is_meta: false,
        tool_use_result: None,
    }
}

/// Create a user message from explicit content blocks (the tool_result path).
pub fn create_user_message_blocks(blocks: Vec<ContentBlockParam>) -> UserMessage {
    UserMessage {
        uuid: Uuid::new_v4().to_string(),
        message: MessageParam { role: "user".into(), content: MessageContent::Blocks(blocks) },
        is_meta: false,
        tool_use_result: None,
    }
}

/// Create an assistant message (createAssistantMessage). Empty string content
/// becomes NO_CONTENT_MESSAGE; the model is taken from config, not hardcoded.
pub fn create_assistant_message(content: &str, usage: Option<BetaUsage>) -> Value {
    let text = if content.is_empty() { NO_CONTENT_MESSAGE } else { content };
    let usage = usage.unwrap_or_default();
    serde_json::json!({
        "type": "assistant",
        "uuid": Uuid::new_v4().to_string(),
        "message": {
            "id": format!("msg_{}", &Uuid::new_v4().simple().to_string()[..24]),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": crate::utils::config::get_config().model_name,
            "stop_reason": "end_turn",
            "usage": serde_json::to_value(usage).unwrap(),
        }
    })
}

/// deriveUUID (messages.ts:725): parent[:24] + index as zero-padded 12-hex.
/// Used to give each split content block a stable, unique uuid.
pub fn derive_uuid(parent_uuid: &str, index: usize) -> String {
    let head: String = parent_uuid.chars().take(24).collect();
    format!("{head}{index:012x}")
}

/// Get the ID of the last assistant message.
pub fn get_last_assistant_message_id(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|m| match m {
        Message::Assistant(a) => Some(a.message.id.clone()),
        _ => None,
    })
}

/// normalizeMessages (messages.ts:741): split each message so EVERY content
/// block (text and tool_use, for both roles) becomes its own single-block
/// message. Once any message has >1 block, all SUBSEQUENT messages get fresh
/// derive_uuid()-based uuids (isNewChain) to prevent duplicate uuids.
/// progress passes through unchanged.
pub fn normalize_messages(messages: &[Message]) -> Vec<Message> {
    let mut normalized: Vec<Message> = Vec::with_capacity(messages.len());
    let mut is_new_chain = false;

    for message in messages {
        match message {
            Message::Progress(_) => {
                normalized.push(message.clone());
                continue;
            }
            Message::User(u) => {
                let content = u.message.content.as_blocks();
                is_new_chain = is_new_chain || content.len() > 1;
                if content.len() <= 1 && !is_new_chain {
                    normalized.push(message.clone());
                    continue;
                }
                for (index, block) in content.iter().enumerate() {
                    let new_uuid = if is_new_chain {
                        derive_uuid(&u.uuid, index)
                    } else {
                        u.uuid.clone()
                    };
                    let mut split = u.clone();
                    split.uuid = new_uuid;
                    split.message = MessageParam {
                        role: u.message.role.clone(),
                        content: MessageContent::Blocks(vec![block.clone()]),
                    };
                    normalized.push(Message::User(split));
                }
            }
            Message::Assistant(a) => {
                let content = &a.message.content;
                is_new_chain = is_new_chain || content.len() > 1;
                if content.len() <= 1 && !is_new_chain {
                    normalized.push(message.clone());
                    continue;
                }
                for (index, block) in content.iter().enumerate() {
                    let new_uuid = if is_new_chain {
                        derive_uuid(&a.uuid, index)
                    } else {
                        a.uuid.clone()
                    };
                    let mut split = a.clone();
                    split.uuid = new_uuid;
                    split.message.content = vec![block.clone()];
                    normalized.push(Message::Assistant(split));
                }
            }
        }
    }

    normalized
}

/// Extract text content from a message (text blocks joined by '\n').
/// The content blocks of a message, when it has a list of them.
fn content_blocks(message: &Value) -> Option<&Vec<Value>> {
    message.get("message")?.get("content")?.as_array()
}

/// Drop assistant turns whose tool calls ALL went unanswered
/// (filterUnresolvedToolUses).
///
/// Not the same rule as filter_incomplete_tool_calls, and the difference is
/// deliberate in the source: that one drops a turn if ANY of its tool calls is
/// unanswered, this one only if EVERY one is. A turn that got some results back
/// carries real work, and resume keeps it; a turn that got none is the
/// interrupted tail and goes.
///
/// The collection walks content blocks directly rather than normalising first,
/// for the reason the source gives: normalising mints new uuids, and returning
/// those would defeat the transcript's uuid dedup and grow the file on every
/// resume.
pub fn filter_unresolved_tool_uses(messages: &[Value]) -> Vec<Value> {
    let mut tool_use_ids: HashSet<String> = HashSet::new();
    let mut tool_result_ids: HashSet<String> = HashSet::new();
    for message in messages {
        let kind = message.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }
        for block in content_blocks(message).into_iter().flatten() {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("tool_use") => {
                    if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                        tool_use_ids.insert(id.to_string());
                    }
                }
                Some("tool_result") => {
                    if let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) {
                        tool_result_ids.insert(id.to_string());
                    }
                }
                _ => {}
            }
        }
    }
    let unresolved: HashSet<&String> = tool_use_ids
        .iter()
        .filter(|id| !tool_result_ids.contains(*id))
        .collect();
    if unresolved.is_empty() {
        return messages.to_vec();
    }

    messages
        .iter()
        .filter(|message| {
            if message.get("type").and_then(|v| v.as_str()) != Some("assistant") {
                return true;
            }
            let Some(blocks) = content_blocks(message) else {
                return true;
            };
            let ids: Vec<&str> = blocks
                .iter()
                .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
                .filter_map(|b| b.get("id").and_then(|v| v.as_str()))
                .collect();
            ids.is_empty() || !ids.iter().all(|id| unresolved.contains(&id.to_string()))
        })
        .cloned()
        .collect()
}

/// True when every block is text and none of it says anything
/// (hasOnlyWhitespaceTextContent).
fn has_only_whitespace_text_content(blocks: &[Value]) -> bool {
    if blocks.is_empty() {
        return false;
    }
    for block in blocks {
        if block.get("type").and_then(|v| v.as_str()) != Some("text") {
            return false;
        }
        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
            if !text.trim().is_empty() {
                return false;
            }
        }
    }
    true
}

/// Drop assistant turns that are nothing but blank text
/// (filterWhitespaceOnlyAssistantMessages).
///
/// An empty content list is kept: that shape is handled elsewhere and means
/// something different from "said only spaces".
pub fn filter_whitespace_only_assistant_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .filter(|message| {
            if message.get("type").and_then(|v| v.as_str()) != Some("assistant") {
                return true;
            }
            match content_blocks(message) {
                None => true,
                Some(blocks) if blocks.is_empty() => true,
                Some(blocks) => !has_only_whitespace_text_content(blocks),
            }
        })
        .cloned()
        .collect()
}

const THINKING_TYPES: &[&str] = &["thinking", "redacted_thinking"];

/// Drop thinking-only assistant turns with nothing to merge into
/// (filterOrphanedThinkingOnlyMessages).
///
/// A streamed turn can arrive as several messages sharing one message id — the
/// thinking in one, the text and tool calls in another — and the API serializer
/// merges them again. So a thinking-only message is only orphaned when no other
/// message with the SAME id carries non-thinking content. Dropping it anyway
/// would throw away reasoning that still belongs to a turn.
pub fn filter_orphaned_thinking_only_messages(messages: &[Value]) -> Vec<Value> {
    let non_thinking = |blocks: &Vec<Value>| {
        blocks.iter().any(|b| {
            !THINKING_TYPES.contains(&b.get("type").and_then(|v| v.as_str()).unwrap_or(""))
        })
    };
    let msg_id = |message: &Value| {
        message
            .get("message")
            .and_then(|m| m.get("id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    let mut ids_with_content: HashSet<String> = HashSet::new();
    for message in messages {
        if message.get("type").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        if let Some(blocks) = content_blocks(message) {
            if non_thinking(blocks) {
                if let Some(id) = msg_id(message) {
                    ids_with_content.insert(id);
                }
            }
        }
    }

    messages
        .iter()
        .filter(|message| {
            if message.get("type").and_then(|v| v.as_str()) != Some("assistant") {
                return true;
            }
            let Some(blocks) = content_blocks(message) else {
                return true;
            };
            if blocks.is_empty() || non_thinking(blocks) {
                return true;
            }
            msg_id(message)
                .map(|id| ids_with_content.contains(&id))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

pub fn get_text_from_message(message: &Message) -> String {
    match message {
        Message::Assistant(a) => a
            .message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Message::User(u) => u
            .message
            .content
            .as_blocks()
            .iter()
            .filter_map(|b| match b {
                ContentBlockParam::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Message::Progress(_) => String::new(),
    }
}

/// Check if message is an interrupt message.
pub fn is_interrupt_message(message: &Message) -> bool {
    get_text_from_message(message).contains(INTERRUPT_MESSAGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_uuid_shape() {
        let parent = "0123456789abcdef0123456789abcdef";
        assert_eq!(derive_uuid(parent, 1), format!("{}{:012x}", &parent[..24], 1));
        assert_eq!(derive_uuid(parent, 1).len(), 36);
    }
}

/// Convert restored transcript entries (raw JSON, as `/resume` reads them off
/// disk) into typed conversation messages.
///
/// The Python port kept resumed history as plain dicts flowing through the same
/// list as live messages (session_storage.py:423-436 restores every entry
/// tolerantly); Rust's `Message` is typed, so the boundary converts once here.
///
/// Parsing is LENIENT per message: an entry whose typed parse fails is not
/// dropped wholesale — dropping a whole assistant message while keeping its
/// user tool_result leaves an orphaned result and the server 400s on the first
/// resumed request. Instead the content array is re-filtered element-wise
/// (unknown block shapes are skipped, null content becomes empty), which
/// matches what the Python wire converter effectively does with blocks it does
/// not recognise while never losing the message itself.
pub fn messages_from_transcript(values: &[Value]) -> Vec<Message> {
    values
        .iter()
        .filter_map(|v| match v.get("type").and_then(|t| t.as_str()) {
            Some(kind @ ("user" | "assistant")) => {
                serde_json::from_value::<Message>(v.clone())
                    .ok()
                    .or_else(|| lenient_transcript_message(v, kind))
            }
            _ => None,
        })
        .collect()
}

/// Second-chance parse for a transcript entry the strict shape rejected:
/// keep only the content blocks that individually parse, coerce null content
/// to an empty list, and try again. Returns None only if the entry is broken
/// beyond its content (missing uuid/message — genuinely not a message).
fn lenient_transcript_message(v: &Value, kind: &str) -> Option<Message> {
    let mut v2 = v.clone();
    let content = v2.get_mut("message")?.get_mut("content")?;
    match content {
        Value::Array(arr) => {
            arr.retain(|b| {
                if kind == "assistant" {
                    serde_json::from_value::<crate::types::messages::ContentBlock>(b.clone())
                        .is_ok()
                } else {
                    serde_json::from_value::<crate::types::messages::ContentBlockParam>(b.clone())
                        .is_ok()
                }
            });
        }
        Value::Null => *content = Value::Array(Vec::new()),
        _ => {}
    }
    serde_json::from_value::<Message>(v2).ok()
}

#[cfg(test)]
mod lenient_transcript_tests {
    use super::*;
    use serde_json::json;

    /// Regression: an unknown content-block type used to fail the whole typed
    /// parse and silently DROP the assistant message, leaving its user
    /// tool_result orphaned — the server then 400s on the first resumed
    /// request. The lenient pass keeps the message, minus the alien block.
    #[test]
    fn unknown_block_loses_the_block_not_the_message() {
        let entries = vec![
            json!({"type":"assistant","uuid":"a1","message":{
                "id":"m1","type":"message","role":"assistant",
                "content":[
                    {"type":"server_tool_use","weird":true},
                    {"type":"tool_use","id":"t1","name":"Bash","input":{}}
                ],
                "model":"x","stop_reason":null,"stop_sequence":null}}),
            json!({"type":"user","uuid":"u1","message":{
                "role":"user",
                "content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}),
        ];
        let out = messages_from_transcript(&entries);
        assert_eq!(out.len(), 2, "both messages must survive");
        let a = out[0].as_assistant().expect("assistant kept");
        assert_eq!(a.message.content.len(), 1, "alien block skipped, tool_use kept");
    }

    #[test]
    fn null_content_becomes_empty_not_a_drop() {
        let entries = vec![json!({"type":"user","uuid":"u2","message":{
            "role":"user","content":null}})];
        assert_eq!(messages_from_transcript(&entries).len(), 1);
    }
}

#[cfg(test)]
mod strict_parse_probe {
    use super::*;
    #[test]
    fn real_entry_shape_parses_strict() {
        let v = serde_json::json!({"parentUuid":null,"isSidechain":false,
            "userType":"external","cwd":"/x","sessionId":"s","version":"2.0.0",
            "type":"user","message":{"role":"user","content":[{"type":"text","text":"hi"}]},
            "uuid":"u9","timestamp":"2026-08-27T00:00:00.000Z"});
        let r = serde_json::from_value::<Message>(v);
        eprintln!("REAL-SHAPE strict: {:?}", r.as_ref().err());
        assert!(r.is_ok(), "a real transcript entry must parse strictly");
    }
}

#[cfg(test)]
mod tag_roundtrip_tests {
    use super::*;
    /// The enum owns the "type" tag: it must still be WRITTEN on serialize
    /// (transcript compatibility with Python) and consumed on parse.
    #[test]
    fn enum_tag_round_trips() {
        let m = Message::User(create_user_message_text("hi"));
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["type"], "user", "serialized entry must carry type");
        let back: Message = serde_json::from_value(v).unwrap();
        assert_eq!(back.message_type(), "user");
    }
}

/// Mark a user message as meta: carried to the model, hidden from the user.
///
/// Used for turn-scoped environment facts (cwd, date) that must reach the model
/// every turn but must NOT sit in the system prompt, which is the cached prefix.
pub fn as_meta(mut msg: UserMessage) -> UserMessage {
    msg.is_meta = true;
    msg
}
