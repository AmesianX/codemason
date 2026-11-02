//! OpenAI-compatible model service for CodeMason — shared implementation.
//! Equivalent to codemason/services/_openai_common.py.
//!
//! Both backends (llama.cpp and vLLM) speak the same OpenAI /v1/chat/completions
//! protocol, so all of the logic — message conversion, context management,
//! leaked tool-call recovery, the streaming loop, live token counting, body
//! encryption — lives HERE and is server-agnostic. The two thin shims that
//! import it, services/llamacpp.rs and services/vllm.rs, differ by exactly one
//! request field:
//!
//!   - vLLM sets stream_options.continuous_usage_stats -> a running
//!     completion_tokens on every chunk, so the live spinner shows the server's
//!     exact count instead of a per-chunk tally.
//!   - llama.cpp emits usage only in the final chunk, so it leaves the flag off
//!     and relies on the per-chunk tally until that final exact count arrives.
//!
//! Callers never import from here directly; they go through services/backend.rs,
//! which routes to the right shim based on the detected/configured server type.

use futures::StreamExt;
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::tools::base::Tool;
use crate::types::messages::{
    ApiMessage, BetaUsage, ContentBlock, ContentBlockParam, MessageContent, TextBlock,
    ThinkingBlock, ToolResultContent, ToolUseBlock,
};
use crate::types::query::{AssistantMessage, Message};
use crate::utils::log::debug_log;
use crate::utils::tokens::{count_text_tokens, get_max_tokens, has_model_tokenizer};

// === Configuration ===

static THINKING_MODE: Lazy<bool> = Lazy::new(|| {
    std::env::var("THINKING_MODE").map(|v| v.to_lowercase() == "true").unwrap_or(false)
});

/// Tool result truncate settings
pub const MAX_TOOL_RESULT_LENGTH: usize = 30000;

/// Minimum seconds between exact re-counts of the reasoning text (see
/// recount_thinking). The spinner repaints at 10fps, so a 0.2s floor keeps the
/// thinking counter visibly live while capping the re-tokenise cost.
const THINK_COUNT_EVERY: f64 = 0.2;

/// Build vLLM's RepetitionDetectionParams from config, or None if disabled.
///
/// Measured against the live server (2026-08-06): a forced 40x repetition was
/// cut at ~140 tokens with finish_reason='repetition', while legitimately
/// repetitive answers ran to the full budget untouched under every setting
/// tried. The failure mode this guards is loud; legitimate repetition is not
/// n-gram-identical.
fn repetition_detection_params(cfg: &crate::utils::config::Config) -> Option<Value> {
    // Model-gated: the detector's premise does not hold for every model.
    // See Config::repetition_detection_enabled.
    if !cfg.repetition_detection_enabled() {
        return None;
    }
    let lo = cfg.repetition_min_pattern;
    let hi = cfg.repetition_max_pattern;
    let n = cfg.repetition_min_count;
    if hi <= 0 || n < 2 {
        return None;
    }
    Some(json!({"min_pattern_size": lo, "max_pattern_size": hi, "min_count": n}))
}

/// Count tokens precisely. Prefers the served model's own tokenizer (exact,
/// matches the server) and falls back to the CJK-aware heuristic via
/// count_text_tokens.
pub fn count_tokens_precise(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    // count_text_tokens never raises in the Rust substrate; the Python
    // len(text)//3 fallback covered a tokenizer import failure.
    count_text_tokens(text)
}

// === Utility Functions ===

/// Python-compatible json.dumps with default separators (", ", ": ") and
/// ensure_ascii=True — used where the serialized string is model-visible (the
/// tool_calls `arguments` field), so the wire bytes match the Python's.
/// KNOWN DIVERGENCE: serde_json's Map is sorted (BTreeMap), so object key
/// order is alphabetical where Python preserved model insertion order —
/// semantically identical JSON; enable serde_json "preserve_order" to close it.
pub fn py_json_dumps(v: &Value) -> String {
    let mut out = String::new();
    py_json_write(v, &mut out);
    out
}

fn py_json_write(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => py_json_write_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_write(item, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, item)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_write_str(k, out);
                out.push_str(": ");
                py_json_write(item, out);
            }
            out.push('}');
        }
    }
}

fn py_json_write_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) < 0x7f => out.push(c),
            c => {
                // ensure_ascii: non-ASCII escaped, astral chars as surrogate pairs.
                let cp = c as u32;
                if cp <= 0xFFFF {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    let v = cp - 0x10000;
                    let hi = 0xD800 + (v >> 10);
                    let lo = 0xDC00 + (v & 0x3FF);
                    out.push_str(&format!("\\u{hi:04x}\\u{lo:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// Truncate tool result if too long, keeping start and end. (Char-based, like
/// the Python len()/slice semantics.)
pub fn truncate_tool_result(content: &str) -> String {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= MAX_TOOL_RESULT_LENGTH {
        return content.to_string();
    }
    let half_length = MAX_TOOL_RESULT_LENGTH / 2;
    let start: String = chars[..half_length].iter().collect();
    let end: String = chars[chars.len() - half_length..].iter().collect();
    let truncated_lines = chars[half_length..chars.len() - half_length]
        .iter()
        .filter(|c| **c == '\n')
        .count();
    format!("{start}\n\n... [{truncated_lines} lines truncated] ...\n\n{end}")
}

/// Get simple description for a tool.
pub fn get_simple_description(tool_name: &str) -> String {
    match tool_name {
        "dispatch_agent" => "Launch agent for complex searches".into(),
        "Agent" => "Launch a sub-agent for batch operations or open-ended searches. IMPORTANT: write the agent prompt in the language the user is writing in, whatever that language is. Never default to English.".into(),
        "Architect" => "Analyze technical requirements".into(),
        "Bash" => "Execute bash commands".into(),
        "Glob" => "Find files by pattern".into(),
        "Grep" => "Search file contents".into(),
        "List" => "List directory contents".into(),
        "Read" => "Read file contents".into(),
        "Edit" => "Edit existing files".into(),
        "Write" => "Create or overwrite files".into(),
        "ReadNotebook" => "Read Jupyter notebooks".into(),
        "NotebookEditCell" => "Edit notebook cells".into(),
        "TodoWrite" => "Create task lists".into(),
        "TodoRead" => "Read task lists".into(),
        other => format!("Use the {other} tool"),
    }
}

static RE_CHANNEL: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(r"(?is)<\|channel\|>[^<]*<\|message\|>[\s\S]*?<\|end\|>").unwrap()
});
static RE_SPECIAL: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"<\|[^|]+\|>").unwrap());

/// Head-room left between the estimated prompt and the context window, as a
/// FLOOR — the live margin scales with the input (see the output cap).
const CONTEXT_SAFETY_MARGIN: i64 = 512;
/// A turn always gets at least this much room to answer, even when the estimate
/// says there is none: a too-short answer is recoverable, a 400 is not.
const MIN_OUTPUT_TOKENS: i64 = 512;

/// Built tool blocks, keyed by the tool-name set they were built from.
///
/// The tool block is the FIRST thing in the request, ahead of the system prompt
/// and the entire message history, so it is the base of the server's prefix
/// cache. Anything that changes it throws away every cached token behind it.
///
/// It used to be rebuilt on every request, and `wire_description()` /
/// `input_json_schema()` are resolved fresh each time — so any tool that read
/// live state silently invalidated the whole conversation. An audit on
/// 2026-09-06 found three doing exactly that: the Agent tool's schema varied
/// with the detected conversation language (which, inside a tool loop, is
/// detected from tool OUTPUT), while the Skill tool's description enumerated
/// the skills under the CURRENT directory and the Agent tool's description
/// enumerated custom agents the same way. Each was worth a full re-prefill —
/// ~123 seconds of silence on a 121k-token session.
///
/// Building once per tool-set and reusing it makes the block structurally
/// incapable of drifting mid-session, which is a stronger guarantee than
/// auditing every tool forever. Sub-agents legitimately run different tool
/// sets, hence the key rather than a single slot.
///
/// The cost is that a skill or agent added mid-session is not visible until
/// restart. That is the intended trade: the alternative charges a re-prefill
/// for a listing that almost never changes while a session is running.
static TOOL_BLOCK_CACHE: Lazy<parking_lot::Mutex<std::collections::HashMap<String, Vec<Value>>>> =
    Lazy::new(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
static RE_THINK_TAGS: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(?i)[ \t]*</?think>[ \t]*").unwrap());
static RE_THINK_BLOCK: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(?is)<think>[\s\S]*?</think>").unwrap());
static RE_THINK_EMPTY: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(?is)<think>\s*</think>").unwrap());
static RE_MINIMAX_JSON: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(
        r#"\[?\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{[^}]*\}\s*\}\s*\]?"#,
    )
    .unwrap()
});

/// Clean special tokens from model output.
///
/// reasoning_separated: true when the server already delivered this turn's
/// thinking via the separate `reasoning_content` channel. In that case any
/// <think>…</think> left in `content` is NOT thinking — it's a leaked fragment
/// where the model wrapped the head of its own answer in stray tags. Deleting
/// the block then truncates the answer. So we strip only the TAGS and keep the
/// inner text. Full block deletion is reserved for servers that inline ALL
/// reasoning (no separate channel) under THINKING_MODE.
pub fn clean_special_tokens(content: &str, reasoning_separated: bool) -> String {
    if content.is_empty() {
        return content.to_string();
    }

    // Remove gpt-oss style channel/message tokens
    let cleaned = RE_CHANNEL.replace_all(content, "");
    // Remove any remaining special tokens like <|...|>
    let cleaned = RE_SPECIAL.replace_all(&cleaned, "");

    // Handle <think> tags. With a separate reasoning channel, never delete the
    // enclosed text (it's answer content) — drop the tags only. Otherwise fall
    // back to THINKING_MODE: strip whole inline-think blocks, or just empty ones.
    let cleaned = if reasoning_separated {
        // Drop the stray tags but keep the text. Collapse each tag plus the
        // horizontal whitespace touching it into a single space — the tag often
        // sits where a word break belongs (`word</think>next` -> `word next`).
        RE_THINK_TAGS.replace_all(&cleaned, " ")
    } else if *THINKING_MODE {
        RE_THINK_BLOCK.replace_all(&cleaned, "")
    } else {
        RE_THINK_EMPTY.replace_all(&cleaned, "")
    };

    // Remove any JSON tool calls from the text (MiniMax specific)
    let cleaned = RE_MINIMAX_JSON.replace_all(&cleaned, "");

    cleaned.trim().to_string()
}

// ── Leaked tool-call recovery ────────────────────────────────────────────────
// DeepSeek-V4 intermittently SERIALISES a tool call into the assistant
// `content` instead of emitting it on the structured tool_calls channel
// (deepseek-ai DeepSeek-V3#1244; vLLM#41240). finish_reason comes back 'stop',
// tool_calls is empty, and the invocation would be silently lost — the model
// then narrates as if it ran, fabricating output. DeepSeek ships no fix and no
// official Jinja template, so recovery is on us: parse the leaked markup back
// into real tool calls so NOT ONE invocation is dropped.
//
// Two leak shapes are handled:
//   1. DSML (the native V4 tool markup) — XML with the FULL-WIDTH pipe ｜
//      (U+FF5C); the half-width | is accepted as a tokenizer variant, and a
//      stream truncated before the closing tag is tolerated.
//   2. Bare JSON — {"name": "...", "arguments"|"parameters": {...}} — some
//      builds leak this shape instead (the MiniMax-style fallthrough).

const PIPE: &str = r"[|｜]";

static RE_DSML_INVOKE: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(&format!(
        r#"(?s)<{PIPE}DSML{PIPE}invoke\s+name="([^"]+)"\s*>(.*?)</{PIPE}DSML{PIPE}invoke\s*>"#
    ))
    .unwrap()
});
/// truncated: opened invoke, no closing tag
static RE_DSML_INVOKE_OPEN: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(&format!(r#"(?s)<{PIPE}DSML{PIPE}invoke\s+name="([^"]+)"\s*>(.*)\z"#))
        .unwrap()
});
static RE_DSML_PARAM: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(&format!(
        r#"(?s)<{PIPE}DSML{PIPE}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</{PIPE}DSML{PIPE}parameter\s*>"#
    ))
    .unwrap()
});
/// truncated final parameter, no closing tag
static RE_DSML_PARAM_OPEN: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(&format!(
        r#"(?s)<{PIPE}DSML{PIPE}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)\z"#
    ))
    .unwrap()
});
static RE_DSML_ANY_TAG: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(&format!(r"</?{PIPE}DSML{PIPE}[^>]*>")).unwrap());
static RE_DSML_WRAP_SPAN: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(&format!(
        r"(?s)<{PIPE}DSML{PIPE}tool_calls\s*>.*?(?:</{PIPE}DSML{PIPE}tool_calls\s*>|\z)"
    ))
    .unwrap()
});

/// A tool call recovered from leaked content markup.
#[derive(Debug, Clone)]
pub struct RecoveredCall {
    pub name: String,
    pub input: Value,
}

/// Parse <｜DSML｜parameter …> elements inside one invoke body into a map.
/// string="true" keeps the raw value; string="false" is JSON (fallback: raw).
fn parse_dsml_params(body: &str) -> Value {
    let mut args = serde_json::Map::new();
    for cap in RE_DSML_PARAM.captures_iter(body) {
        let name = cap[1].to_string();
        let is_str = &cap[2];
        let val = &cap[3];
        if is_str == "true" {
            args.insert(name, Value::String(val.to_string()));
        } else {
            match serde_json::from_str::<Value>(val.trim()) {
                Ok(v) => {
                    args.insert(name, v);
                }
                Err(_) => {
                    args.insert(name, Value::String(val.trim().to_string()));
                }
            }
        }
    }
    // A stream cut off mid-parameter leaves one unterminated element; recover
    // it too, but only if it wasn't already captured by the closed-tag pass.
    if let Some(mo) = RE_DSML_PARAM_OPEN.captures(body) {
        let name = mo[1].to_string();
        if !args.contains_key(&name) {
            let is_str = &mo[2];
            // strip a dangling partial open-tag of the next element if present
            let val = RE_DSML_ANY_TAG.replace_all(&mo[3], "").into_owned();
            if is_str == "true" {
                args.insert(name, Value::String(val));
            } else {
                match serde_json::from_str::<Value>(val.trim()) {
                    Ok(v) => {
                        args.insert(name, v);
                    }
                    Err(_) => {
                        args.insert(name, Value::String(val.trim().to_string()));
                    }
                }
            }
        }
    }
    Value::Object(args)
}

/// Yield (name, args, start, end) for every bare JSON tool-call object in
/// text. Uses an incremental JSON parse (not regex) so nested braces parse
/// correctly (the Python raw_decode equivalent).
fn scan_json_tool_calls(text: &str) -> Vec<(String, Value, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let n = text.len();
    while i < n {
        let Some(rel) = text[i..].find('{') else {
            break;
        };
        let s = i + rel;
        let mut iter = serde_json::Deserializer::from_str(&text[s..]).into_iter::<Value>();
        match iter.next() {
            Some(Ok(obj)) => {
                let end = s + iter.byte_offset();
                if let Value::Object(map) = &obj {
                    if map.contains_key("name")
                        && (map.contains_key("arguments") || map.contains_key("parameters"))
                    {
                        let mut args = map
                            .get("arguments")
                            .or_else(|| map.get("parameters"))
                            .cloned()
                            .unwrap_or(Value::Null);
                        if let Value::String(raw) = &args {
                            args = serde_json::from_str::<Value>(raw)
                                .unwrap_or_else(|_| Value::Object(Default::default()));
                        }
                        if args.is_object() {
                            let name = match map.get("name") {
                                Some(Value::String(s2)) => s2.clone(),
                                Some(other) => other.to_string(),
                                None => String::new(),
                            };
                            out.push((name, args, s, end));
                        }
                    }
                }
                i = end;
            }
            _ => {
                i = s + 1;
            }
        }
    }
    out
}

/// Extract tool calls that leaked into `content`.
///
/// Returns (recovered, residual_text):
///   recovered      -> list of {'name', 'input'}
///   residual_text  -> content with the leaked markup removed (the real answer)
///
/// No-op (returns [], content) when no leak markers are present, so the normal
/// path — where the model used the structured tool_calls channel — is untouched.
pub fn recover_leaked_tool_calls(content: &str) -> (Vec<RecoveredCall>, String) {
    if content.is_empty() {
        return (Vec::new(), content.to_string());
    }
    let has_dsml = content.contains("DSML");
    let has_json = content.contains("\"name\"")
        && (content.contains("\"arguments\"") || content.contains("\"parameters\""));
    if !has_dsml && !has_json {
        return (Vec::new(), content.to_string());
    }

    let mut recovered: Vec<RecoveredCall> = Vec::new();
    let mut text = content.to_string();

    if has_dsml {
        let mut matched = false;
        for cap in RE_DSML_INVOKE.captures_iter(&text) {
            recovered.push(RecoveredCall {
                name: cap[1].to_string(),
                input: parse_dsml_params(&cap[2]),
            });
            matched = true;
        }
        if matched {
            text = RE_DSML_INVOKE.replace_all(&text, "").into_owned();
        }
        // A trailing invoke can be truncated by a stream cut whether or not any
        // fully-closed invoke preceded it. Check the residual for an
        // unterminated one in BOTH cases (previously this ran only when NO
        // closed invoke matched, so a truncated invoke AFTER >=1 complete
        // invoke was dropped).
        if let Some(mo) = RE_DSML_INVOKE_OPEN.captures(&text) {
            let start = mo.get(0).map(|m| m.start()).unwrap_or(0);
            recovered.push(RecoveredCall {
                name: mo[1].to_string(),
                input: parse_dsml_params(&mo[2]),
            });
            text.truncate(start);
        }
        // Strip any leftover wrapper / stray DSML tags from the residual answer.
        text = RE_DSML_WRAP_SPAN.replace_all(&text, "").into_owned();
        text = RE_DSML_ANY_TAG.replace_all(&text, "").into_owned();
    }

    // Only fall back to bare-JSON scanning when DSML yielded nothing, so a
    // normal DSML answer that happens to quote JSON isn't mis-scavenged.
    if recovered.is_empty() && has_json {
        let spans = scan_json_tool_calls(&text);
        for (name, args, _s, _e) in &spans {
            recovered.push(RecoveredCall { name: name.clone(), input: args.clone() });
        }
        for (_name, _args, s, e) in spans.iter().rev() {
            text = format!("{}{}", &text[..*s], &text[*e..]);
        }
    }

    (recovered, text)
}

/// Build system prompt with cwd emphasis at the top for better LLM awareness.
/// Splits into dynamic (cwd) + static (core prompt) + rest (env info).
pub fn build_system_prompt_with_cwd_emphasis(system_prompt: &[String]) -> String {
    // The name is kept; the cwd emphasis is NOT.
    //
    // This used to put the live cwd in the first bytes of the system prompt —
    // ahead of everything — and rebuild it on EVERY request, tool-loop turns
    // included. The system prompt is the cached prefix, so a single `cd` moved
    // byte ~40 of the request and the server re-prefilled the entire
    // conversation. Measured directly on this backend at 96k tokens: 72.1s when
    // the front changes, against 3.4s for the same information appended to the
    // end. The cost scales with the whole context, so at 200k it is minutes.
    //
    // (A 146.4s stall observed live while diagnosing this was NOT this bug —
    // cwd had not moved for that request. It was almost certainly the
    // diagnosis itself evicting the session's prefix with test traffic. Recorded
    // so the number is not later mistaken for evidence of this code path.)
    //
    // The emphasis was there so the model would not lose track of where "." is,
    // which is a real need — the model does need the CURRENT directory, not the
    // one the session opened in. It now rides the turn (cli_repl's <turn-env>),
    // where it is appended and costs nothing. Nothing is lost: the value is
    // still per-turn and still current, and <env> continues to state the
    // session's starting directory once.
    //
    // Do not put it back. Anything that varies within a session belongs after
    // the history, never before it.
    system_prompt.join("\n\n")
}

// === Chat Message Types ===

/// One OpenAI-wire chat message ('system' | 'user' | 'assistant' | 'tool').
///
/// reasoning_content: thinking-mode round-trip. DeepSeek-V4's thinking API
/// requires the prior turn's reasoning to come back on every assistant message
/// that made a tool call; without it the model loses the conclusion it reached
/// while thinking and re-derives it from the same tool output, which is how an
/// agent ends up issuing the identical command turn after turn. Only set when
/// the backend asked for the round-trip (vLLM) — see reasoning_roundtrip.
/// Serialized under both wire names (to_value).
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
    pub reasoning_content: Option<String>,
    /// Multimodal content. OpenAI's chat API takes `content` as EITHER a string
    /// OR an array of typed parts; images only exist in the array form. Keeping
    /// them in a separate field means every text-only message serializes exactly
    /// as before (a bare string), so enabling vision changes no existing request.
    pub content_parts: Option<Vec<Value>>,
}

impl ChatMessage {
    pub fn to_value(&self) -> Value {
        let mut result = serde_json::Map::new();
        result.insert("role".into(), Value::String(self.role.clone()));
        if let Some(parts) = self.content_parts.as_ref().filter(|p| !p.is_empty()) {
            result.insert("content".into(), Value::Array(parts.clone()));
        } else if let Some(c) = &self.content {
            result.insert("content".into(), Value::String(c.clone()));
        }
        // Truthy gates, matching the Python to_dict exactly: empty string /
        // empty list / None are all omitted.
        if let Some(rc) = &self.reasoning_content {
            if !rc.is_empty() {
                // vLLM reads `reasoning` (0.26 parses only that key and ignores
                // `reasoning_content`, which it lists as deprecated); older
                // servers and the DeepSeek API read `reasoning_content`. Send both.
                result.insert("reasoning".into(), Value::String(rc.clone()));
                result.insert("reasoning_content".into(), Value::String(rc.clone()));
            }
        }
        if let Some(tc) = &self.tool_calls {
            if !tc.is_empty() {
                result.insert("tool_calls".into(), Value::Array(tc.clone()));
            }
        }
        if let Some(id) = &self.tool_call_id {
            if !id.is_empty() {
                result.insert("tool_call_id".into(), Value::String(id.clone()));
            }
        }
        Value::Object(result)
    }
}

// === Message Conversion ===

/// Upper bound of what one image costs in the prompt (server VISION_MAX_TOKENS).
pub const VISION_TOKENS_PER_IMAGE_MAX: i64 = 4096;

/// Anthropic image `source` -> OpenAI `image_url` data URL.
///
/// Anthropic nests the bytes under `source`: {"type": "base64", "media_type":
/// "image/png", "data": "<b64>"}. OpenAI wants one `data:` URL. A url-sourced
/// block is passed through as-is. None when the block carries neither, so the
/// caller falls back to the placeholder instead of sending a part the server
/// will reject.
fn image_source_to_data_url(source: &Value) -> Option<String> {
    if source.get("type").and_then(|t| t.as_str()) == Some("url") {
        return source.get("url").and_then(|u| u.as_str()).filter(|u| !u.is_empty()).map(String::from);
    }
    let data = source.get("data").and_then(|d| d.as_str()).filter(|d| !d.is_empty())?;
    let media_type = source.get("media_type").and_then(|m| m.as_str()).unwrap_or("image/png");
    Some(format!("data:{media_type};base64,{data}"))
}

fn image_url_part(url: String) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

/// How many of the oldest images to send as text instead of pixels.
///
/// The server takes at most `limit` images per request (vLLM
/// --limit-mm-per-prompt). Dropping exactly the overflow would change the
/// request prefix at a new place every time an image is added, and the server
/// re-prefills from that point each turn. Dropping in batches of limit/2 keeps
/// the evicted set fixed until the next batch is due, so the prefix moves once
/// per limit/2 new images.
///
/// `seen` is how many of the oldest images were already in a request the model
/// answered. The batch rounding only takes from those: an image that arrived
/// since the last answer has not been looked at yet, and dropping it early
/// means the model describes it without having seen it. (It is not in any
/// cached prefix yet either, so batching it buys nothing.) Unseen images go
/// only when the overflow itself demands it — so a burst of new images past the
/// limit is dropped exactly, and once the model has answered, the next request
/// rounds the evicted set up to the batch: the prefix moves twice for that
/// burst, then again once per limit/2 images.
pub fn images_to_evict(total: usize, limit: i64, seen: usize) -> usize {
    if limit <= 0 || total as i64 <= limit {
        return 0;
    }
    let batch = (limit / 2).max(1);
    let over = total as i64 - limit;
    let batched = (((over + batch - 1) / batch) * batch).min(total as i64);
    over.max(batched.min(seen as i64)) as usize
}

/// Images the next request would carry. Counted on the converted request
/// itself, so whatever convert_to_chat_messages skips (a duplicate tool result,
/// a dropped recovery nudge, an image it cannot send) or evicts is not counted
/// either. For the display only.
pub fn attached_image_count(messages: &[Message], limit: i64) -> usize {
    convert_to_chat_messages(messages, &[], false, limit)
        .iter()
        .filter_map(|m| m.content_parts.as_ref())
        .flatten()
        .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"))
        .count()
}

/// An assistant message the model wrote, as opposed to one codemason made (an
/// API error, an interrupt notice — model "<synthetic>"). Only the former means
/// the images before it were looked at.
fn is_model_answer(a: &crate::types::query::AssistantMessage) -> bool {
    !a.is_api_error_message && a.message.model != "<synthetic>"
}

/// Text that stands in for an image not attached. It says whether the model was
/// ever shown it — without that, a model that had described an image concluded
/// from "no longer attached" that it never saw it, and re-read it — and names
/// the copy in the session's image store (utils/image_store.rs) so it can be
/// Read. Deterministic, so the request prefix stays the same.
fn evicted_image_label(n: usize, source: &Value, limit: i64, shown: bool) -> String {
    let mut place = String::new();
    if source.get("type").and_then(|t| t.as_str()) == Some("url") {
        place = format!("; source {}", source.get("url").and_then(|u| u.as_str()).unwrap_or(""));
    } else if let Some(path) = crate::utils::image_store::stored_path(source, None) {
        place = format!(
            "; saved at {} (Read it to {})",
            path.display(),
            if shown { "look again" } else { "see it" }
        );
    }
    if shown {
        format!(
            "[Image {n}: no longer attached — you saw it earlier; dropped to stay within \
             the server limit of {limit} images per request{place}]"
        )
    } else {
        format!(
            "[Image {n}: not attached — over the server limit of {limit} images per \
             request, so you have not seen it{place}]"
        )
    }
}

/// Plain text of a ChatMessage, whether it is a bare string or typed parts.
fn chat_message_text(msg: &ChatMessage) -> String {
    if let Some(c) = msg.content.as_ref().filter(|c| !c.is_empty()) {
        return c.clone();
    }
    msg.content_parts
        .as_ref()
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string()
        })
        .unwrap_or_default()
}

/// Convert Claude-style messages to OpenAI chat format.
///
/// reasoning_roundtrip: when true, ThinkingBlocks stored on assistant turns are
/// sent back as `reasoning_content`. Off for llama.cpp, which has no such
/// channel and would only see an unknown field.

/// Layer-2 repetition-recovery nudges ("You started repeating…", isMeta user
/// messages) exist to rescue ONE turn. Left in history they are re-read on
/// every later turn as a standing reprimand — observed on GLM-5.3 as the
/// model growing defensive ("변심 없음, 확신" appended to every answer) and
/// re-doing work. Drop stale nudges (and the empty assistant stub the cut
/// generation left behind) from the API-bound history; the one that is the
/// LAST message is the live recovery attempt and stays.
const REPETITION_NUDGE_PREFIX: &str = "You started repeating the same output";
fn stale_recovery_indices(messages: &[Message]) -> std::collections::HashSet<usize> {
    let mut stale = std::collections::HashSet::new();
    let last = messages.len().saturating_sub(1);
    for (i, m) in messages.iter().enumerate() {
        let Message::User(u) = m else { continue };
        if !u.is_meta || i == last {
            continue;
        }
        let is_nudge = match &u.message.content {
            MessageContent::Text(t) => t.starts_with(REPETITION_NUDGE_PREFIX),
            MessageContent::Blocks(bs) => bs.iter().any(|b| {
                matches!(b, ContentBlockParam::Text { text, .. } if text.starts_with(REPETITION_NUDGE_PREFIX))
            }),
        };
        if !is_nudge {
            continue;
        }
        stale.insert(i);
        if i > 0 {
            if let Message::Assistant(a) = &messages[i - 1] {
                let empty = a.message.content.iter().all(|b| match b {
                    ContentBlock::Text(t) => t.text.trim().is_empty(),
                    _ => false,
                });
                if empty {
                    stale.insert(i - 1);
                }
            }
        }
    }
    stale
}

/// `max_images`: images the server takes per request (services/vision.rs);
/// 0 = no image channel. When > 0, image blocks are sent as OpenAI `image_url`
/// parts instead of a placeholder sentence. 0 for llama.cpp, which has no image
/// channel — an image_url part there is an unknown field at best and a hard 400
/// at worst.
///
/// Every attached image gets an `[Image N]` label, numbered in conversation
/// order, so the model can tell images apart and refer to them (unlabelled, it
/// miscounts a run of 8). Exceeding the server cap is a 400, so when the
/// conversation holds more, the oldest are sent as text (images_to_evict).
/// Only this request copy changes; the stored history keeps every image.
pub fn convert_to_chat_messages(
    messages: &[Message],
    system_prompt: &[String],
    reasoning_roundtrip: bool,
    max_images: i64,
) -> Vec<ChatMessage> {
    let mut chat_messages: Vec<ChatMessage> = Vec::new();
    let vision = max_images > 0;
    // Source of each attached image, in the order its image_url part is placed.
    // Parts land in chat_messages in that same order, which is how the eviction
    // pass below finds image k again.
    let mut attached: Vec<&Value> = Vec::new();
    // Images before the last assistant message (see images_to_evict).
    let mut seen = 0usize;
    // Numbers of the images that were attached to a request the model answered.
    // Eviction only grows, so this is fixed once an image's first answered
    // request is past, and its label never flips.
    let mut shown: std::collections::HashSet<usize> = Default::default();

    // Add system prompt with cwd emphasis
    if !system_prompt.is_empty() {
        chat_messages.push(ChatMessage {
            role: "system".into(),
            content: Some(build_system_prompt_with_cwd_emphasis(system_prompt)),
            ..Default::default()
        });
    }

    debug_log(
        "CONVERTING MESSAGES",
        Some(&json!({
            "total_messages": messages.len(),
            "message_types": messages.iter().map(|m| m.message_type()).collect::<Vec<_>>(),
        })),
    );

    // Track tool calls for matching results
    let mut processed_tool_call_ids: std::collections::HashSet<String> = Default::default();
    let mut processed_assistant_uuids: std::collections::HashSet<String> = Default::default();
    let mut added_tool_use_ids: std::collections::HashSet<String> = Default::default();

    let stale = stale_recovery_indices(messages);
    if !stale.is_empty() {
        debug_log("STALE_RECOVERY_NUDGES_DROPPED", Some(&json!({"count": stale.len()})));
    }
    for (msg_idx, msg) in messages.iter().enumerate() {
        if stale.contains(&msg_idx) {
            continue;
        }
        match msg {
            Message::User(u) => match &u.message.content {
                MessageContent::Blocks(blocks) => {
                    // Check for tool_result blocks
                    let tool_result_blocks: Vec<&ContentBlockParam> = blocks
                        .iter()
                        .filter(|b| matches!(b, ContentBlockParam::ToolResult { .. }))
                        .collect();
                    let text_blocks: Vec<&ContentBlockParam> = blocks
                        .iter()
                        .filter(|b| matches!(b, ContentBlockParam::Text { .. }))
                        .collect();

                    if !tool_result_blocks.is_empty() {
                        // Convert tool_result to role: 'tool'
                        for block in &tool_result_blocks {
                            // Images this tool returned (Read on a PNG, an MCP
                            // screenshot) go inside its own tool message, after the
                            // text. Not in a user message after the results: the
                            // chat template treats the last user message as the
                            // start of the turn and blanks the reasoning of every
                            // assistant message before it (clear_thinking), so each
                            // image read erased what the model had worked out in
                            // the tool loop.
                            let mut tool_images: Vec<Value> = Vec::new();
                            let ContentBlockParam::ToolResult { tool_use_id, content, .. } = block
                            else {
                                continue;
                            };
                            if processed_tool_call_ids.contains(tool_use_id) {
                                debug_log(
                                    "SKIPPING DUPLICATE TOOL RESULT",
                                    Some(&json!({"tool_call_id": tool_use_id})),
                                );
                                continue;
                            }
                            processed_tool_call_ids.insert(tool_use_id.clone());

                            let result_content = match content {
                                ToolResultContent::Text(s) => s.clone(),
                                ToolResultContent::Blocks(items) => items
                                    .iter()
                                    // An attached image adds no text piece; its
                                    // [Image N] label rides with the image part.
                                    .filter_map(|b| {
                                        if b.get("type").and_then(|t| t.as_str()) == Some("image") {
                                            // Never stringify an image block: that pastes
                                            // its whole base64 payload into the tool text.
                                            let source = b.get("source");
                                            let url = if vision {
                                                source.and_then(image_source_to_data_url)
                                            } else {
                                                None
                                            };
                                            match (url, source) {
                                                (Some(u), Some(src)) => {
                                                    attached.push(src);
                                                    let n = attached.len();
                                                    tool_images.push(json!({
                                                        "type": "text",
                                                        "text": format!("[Image {n}]")
                                                    }));
                                                    tool_images.push(image_url_part(u));
                                                    None
                                                }
                                                _ => Some("[Image content not supported]".to_string()),
                                            }
                                        } else {
                                            Some(
                                                b.get("text")
                                                    .and_then(|t| t.as_str())
                                                    .map(|s| s.to_string())
                                                    .unwrap_or_else(|| b.to_string()),
                                            )
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };

                            // Truncate if needed
                            let mut truncated_content = truncate_tool_result(&result_content);

                            if !tool_images.is_empty() {
                                let mut parts = Vec::new();
                                if !truncated_content.trim().is_empty() {
                                    parts.push(json!({"type": "text", "text": truncated_content}));
                                }
                                parts.extend(tool_images);
                                chat_messages.push(ChatMessage {
                                    role: "tool".into(),
                                    tool_call_id: Some(tool_use_id.clone()),
                                    content_parts: Some(parts),
                                    ..Default::default()
                                });
                                continue;
                            }

                            // Add success message for empty results to prevent
                            // LLM stalling
                            if truncated_content.trim().is_empty() {
                                truncated_content = "(success, no output)".into();
                            }

                            chat_messages.push(ChatMessage {
                                role: "tool".into(),
                                tool_call_id: Some(tool_use_id.clone()),
                                content: Some(truncated_content),
                                ..Default::default()
                            });
                        }

                        // Add text blocks as user message if present
                        let mut text_content = text_blocks
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlockParam::Text { text, .. } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                            .trim()
                            .to_string();
                        // An image the user added alongside the results (typed in
                        // mid-turn) rides in that user message; it used to be
                        // dropped without a word.
                        let mut extra_parts: Vec<Value> = Vec::new();
                        for b in blocks {
                            if let ContentBlockParam::Image { source, .. } = b {
                                match if vision { image_source_to_data_url(source) } else { None } {
                                    Some(url) => {
                                        attached.push(source);
                                        let n = attached.len();
                                        extra_parts.push(json!({"type": "text", "text": format!("[Image {n}]")}));
                                        extra_parts.push(image_url_part(url));
                                    }
                                    None => {
                                        text_content = format!("{text_content}\n[Image content not supported]")
                                            .trim()
                                            .to_string();
                                    }
                                }
                            }
                        }
                        if !extra_parts.is_empty() {
                            let mut parts = Vec::new();
                            if !text_content.is_empty() {
                                parts.push(json!({"type": "text", "text": text_content}));
                            }
                            parts.extend(extra_parts);
                            chat_messages.push(ChatMessage {
                                role: "user".into(),
                                content_parts: Some(parts),
                                ..Default::default()
                            });
                        } else if !text_content.is_empty() {
                            chat_messages.push(ChatMessage {
                                role: "user".into(),
                                content: Some(text_content),
                                ..Default::default()
                            });
                        }
                    } else {
                        // Regular user message with content blocks
                        let mut text_parts: Vec<String> = Vec::new();
                        let mut mm_parts: Vec<Value> = Vec::new();
                        for c in blocks {
                            match c {
                                ContentBlockParam::Text { text, .. } => {
                                    text_parts.push(text.clone());
                                    if !text.is_empty() {
                                        mm_parts.push(json!({"type": "text", "text": text}));
                                    }
                                }
                                ContentBlockParam::Image { source, .. } => {
                                    match if vision { image_source_to_data_url(source) } else { None } {
                                        Some(url) => {
                                            attached.push(source);
                                            let n = attached.len();
                                            mm_parts.push(json!({"type": "text", "text": format!("[Image {n}]")}));
                                            mm_parts.push(image_url_part(url));
                                        }
                                        None => {
                                            // No image channel (or an unreadable source):
                                            // say so rather than dropping the block
                                            // silently, so the model knows something
                                            // was there.
                                            text_parts.push("[Image content not supported]".into());
                                            mm_parts.push(json!({
                                                "type": "text",
                                                "text": "[Image content not supported]"
                                            }));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        let has_image = mm_parts
                            .iter()
                            .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"));
                        if has_image {
                            chat_messages.push(ChatMessage {
                                role: "user".into(),
                                content_parts: Some(mm_parts),
                                ..Default::default()
                            });
                        } else {
                            // Text-only: keep the bare-string shape byte-identical
                            // to the pre-vision path.
                            let text_content = text_parts.join("\n").trim().to_string();
                            if !text_content.is_empty() {
                                chat_messages.push(ChatMessage {
                                    role: "user".into(),
                                    content: Some(text_content),
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }
                MessageContent::Text(s) => {
                    // String content
                    if !s.trim().is_empty() {
                        chat_messages.push(ChatMessage {
                            role: "user".into(),
                            content: Some(s.trim().to_string()),
                            ..Default::default()
                        });
                    }
                }
            },
            Message::Assistant(a) => {
                // This message answered a request holding `attached` minus the
                // images that request evicted.
                if is_model_answer(a) {
                    for k in images_to_evict(attached.len(), max_images, seen) + 1..=attached.len() {
                        shown.insert(k);
                    }
                    seen = attached.len();
                }
                // Check for duplicate by UUID
                let msg_uuid = a.uuid.clone();
                if !msg_uuid.is_empty() && processed_assistant_uuids.contains(&msg_uuid) {
                    debug_log(
                        "SKIPPING DUPLICATE ASSISTANT BY UUID",
                        Some(&json!({"uuid": msg_uuid})),
                    );
                    continue;
                }
                if !msg_uuid.is_empty() {
                    processed_assistant_uuids.insert(msg_uuid.clone());
                }

                let mut content_text = String::new();
                let mut reasoning_text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut current_tool_use_ids: Vec<String> = Vec::new();

                for block in &a.message.content {
                    match block {
                        ContentBlock::Thinking(t) => reasoning_text.push_str(&t.thinking),
                        ContentBlock::Text(t) => {
                            content_text.push_str(&t.text);
                            content_text.push('\n');
                        }
                        ContentBlock::ToolUse(tu) => {
                            if added_tool_use_ids.contains(&tu.id) {
                                debug_log(
                                    "SKIPPING DUPLICATE TOOL_USE",
                                    Some(&json!({"id": tu.id})),
                                );
                                continue;
                            }
                            current_tool_use_ids.push(tu.id.clone());
                            tool_calls.push(json!({
                                "id": tu.id,
                                "type": "function",
                                "function": {
                                    "name": tu.name,
                                    "arguments": py_json_dumps(&tu.input),
                                }
                            }));
                        }
                    }
                }

                // Mark tool_use IDs as added
                for tid in current_tool_use_ids {
                    added_tool_use_ids.insert(tid);
                }

                // Skip empty assistant messages
                let content_text = content_text.trim().to_string();
                if content_text.is_empty() && tool_calls.is_empty() {
                    debug_log("SKIPPING EMPTY ASSISTANT MESSAGE", None);
                    continue;
                }

                let reasoning_trimmed = reasoning_text.trim().to_string();
                let assistant_msg = ChatMessage {
                    role: "assistant".into(),
                    content: if content_text.is_empty() { None } else { Some(content_text.clone()) },
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls.clone()) },
                    reasoning_content: if reasoning_roundtrip && !reasoning_trimmed.is_empty() {
                        Some(reasoning_trimmed)
                    } else {
                        None
                    },
                    ..Default::default()
                };
                chat_messages.push(assistant_msg);

                debug_log(
                    "ADDED ASSISTANT MESSAGE",
                    Some(&json!({
                        "uuid": msg_uuid,
                        "content_len": content_text.chars().count(),
                        "reasoning_len": if reasoning_roundtrip { reasoning_text.chars().count() } else { 0 },
                        "tool_calls_count": tool_calls.len(),
                    })),
                );
            }
            // progress / system / attachment — never sent to the model.
            Message::Progress(_) => {}
        }
    }

    let evict = images_to_evict(attached.len(), max_images, seen);
    if evict > 0 {
        let mut seen = 0usize;
        for m in chat_messages.iter_mut() {
            let Some(parts) = m.content_parts.as_mut() else { continue };
            let mut i = 0;
            while i < parts.len() && seen < evict {
                if parts[i].get("type").and_then(|t| t.as_str()) != Some("image_url") {
                    i += 1;
                    continue;
                }
                seen += 1;
                // The [Image N] label sits right before its image part.
                if i > 0 {
                    parts[i - 1] = json!({
                        "type": "text",
                        "text": evicted_image_label(
                            seen,
                            attached[seen - 1],
                            max_images,
                            shown.contains(&seen)
                        )
                    });
                }
                parts.remove(i);
            }
        }
        debug_log("IMAGES_EVICTED", Some(&json!({"total": attached.len(), "evicted": evict})));
    }

    // Clean up empty messages and merge consecutive assistants
    let cleaned_messages = clean_messages(chat_messages);

    // Context management
    let managed_messages = manage_context(cleaned_messages);

    // Final pass: merge consecutive assistant messages
    let final_messages = merge_consecutive_assistants(managed_messages);

    debug_log(
        "FINAL MESSAGE SEQUENCE",
        Some(&json!({
            "count": final_messages.len(),
            "roles": final_messages.iter().map(|m| json!({
                "role": m.role,
                "has_tool_calls": m.tool_calls.as_ref().map(|t| !t.is_empty()).unwrap_or(false),
            })).collect::<Vec<_>>(),
        })),
    );

    final_messages
}

/// Clean empty messages and merge consecutive assistant messages.
fn clean_messages(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut cleaned: Vec<ChatMessage> = Vec::new();

    for msg in messages {
        // Skip empty messages (but keep assistant messages with tool_calls)
        let has_tool_calls = msg.role == "assistant"
            && msg.tool_calls.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
        let content_empty =
            msg.content.as_ref().map(|c| c.trim().is_empty()).unwrap_or(true);
        // A vision message keeps everything in content_parts (content is None);
        // it is not empty.
        let has_parts = msg.content_parts.as_ref().map(|p| !p.is_empty()).unwrap_or(false);
        if content_empty && !has_tool_calls && !has_parts {
            continue;
        }

        // Merge consecutive assistant messages
        if let Some(prev_msg) = cleaned.last_mut() {
            if msg.role == "assistant" && prev_msg.role == "assistant" {
                let prev_has_tool_calls =
                    prev_msg.tool_calls.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
                let curr_has_tool_calls = has_tool_calls;

                if !prev_has_tool_calls && !curr_has_tool_calls {
                    // Merge content
                    let prev_content = prev_msg.content.clone().unwrap_or_default();
                    let curr_content = msg.content.clone().unwrap_or_default();
                    prev_msg.content =
                        Some(format!("{prev_content}\n\n{curr_content}").trim().to_string());
                    if let Some(rc) = &msg.reasoning_content {
                        if !rc.is_empty() {
                            let prev_reasoning =
                                prev_msg.reasoning_content.clone().unwrap_or_default();
                            prev_msg.reasoning_content =
                                Some(format!("{prev_reasoning}\n\n{rc}").trim().to_string());
                        }
                    }
                    continue;
                }
            }
        }

        cleaned.push(msg);
    }

    cleaned
}

/// Manage context by keeping recent messages within token limit.
///
/// NOTE: this pre-send trim is a local stopgap for servers that hard-error on
/// overflow. It silently drops the oldest messages (by heuristic count, which
/// mis-sizes the window) and, by keeping every request under the limit, also
/// renders compaction's PTL retry unreachable. When the exact model tokenizer
/// is available, auto-compaction counts accurately and fires in time, so we
/// defer to it exactly like upstream and skip this trim entirely (this also
/// lets the PTL retry actually run).
fn manage_context(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    if has_model_tokenizer() {
        return messages;
    }

    fn count_message_tokens(msg: &ChatMessage) -> i64 {
        let mut tokens = count_tokens_precise(&chat_message_text(msg));
        if let Some(parts) = &msg.content_parts {
            // Each image is at most VISION_TOKENS_PER_IMAGE_MAX on the server.
            tokens += VISION_TOKENS_PER_IMAGE_MAX
                * parts
                    .iter()
                    .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"))
                    .count() as i64;
        }
        if let Some(rc) = &msg.reasoning_content {
            if !rc.is_empty() {
                tokens += count_tokens_precise(rc);
            }
        }
        if let Some(tc) = &msg.tool_calls {
            if !tc.is_empty() {
                tokens += count_tokens_precise(&py_json_dumps(&Value::Array(tc.clone())));
            }
        }
        tokens + 4 // Overhead for message structure
    }

    let (system_messages, conversation_messages): (Vec<ChatMessage>, Vec<ChatMessage>) =
        messages.into_iter().partition(|m| m.role == "system");

    let mut total_tokens: i64 = system_messages.iter().map(count_message_tokens).sum();
    let mut managed_messages: Vec<ChatMessage> = system_messages.clone();
    let n_system = system_messages.len();

    // Add recent messages from the end
    for i in (0..conversation_messages.len()).rev() {
        let msg = &conversation_messages[i];
        let msg_tokens = count_message_tokens(msg);

        if total_tokens + msg_tokens > get_max_tokens() {
            debug_log(
                "CONTEXT LIMIT REACHED",
                Some(&json!({
                    "dropped_messages": i + 1,
                    "kept_messages": conversation_messages.len() - i - 1,
                    "total_tokens": total_tokens,
                })),
            );
            break;
        }

        total_tokens += msg_tokens;
        managed_messages.insert(n_system, msg.clone());
    }

    // Truncating from the front can orphan tool-result messages whose assistant
    // tool_calls were dropped (a leading role='tool' with no preceding
    // tool_calls). Many servers 400 on that, so strip any orphan tool results
    // at the head of the conversation. (Assistant tool_calls can't be orphaned
    // the other way: their results are newer, so they're always kept first.)
    while managed_messages.len() > n_system && managed_messages[n_system].role == "tool" {
        managed_messages.remove(n_system);
    }

    managed_messages
}

/// Final pass to merge any consecutive assistant messages.
fn merge_consecutive_assistants(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut safe_messages: Vec<ChatMessage> = Vec::new();

    for msg in messages {
        let mergeable = safe_messages
            .last()
            .map(|p| msg.role == "assistant" && p.role == "assistant")
            .unwrap_or(false);
        if mergeable {
            let prev_msg = safe_messages.last_mut().expect("checked non-empty");
            let prev_has_tool_calls =
                prev_msg.tool_calls.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
            let curr_has_tool_calls =
                msg.tool_calls.as_ref().map(|t| !t.is_empty()).unwrap_or(false);

            // Merge content
            if let Some(c) = &msg.content {
                if !c.is_empty() {
                    let prev_content = prev_msg.content.clone().unwrap_or_default();
                    prev_msg.content =
                        Some(format!("{prev_content}\n\n{c}").trim().to_string());
                }
            }

            // Merge reasoning alongside it — dropping it here would reintroduce
            // the very gap this field exists to close.
            if let Some(rc) = &msg.reasoning_content {
                if !rc.is_empty() {
                    let prev_reasoning = prev_msg.reasoning_content.clone().unwrap_or_default();
                    prev_msg.reasoning_content =
                        Some(format!("{prev_reasoning}\n\n{rc}").trim().to_string());
                }
            }

            // Merge tool_calls
            if curr_has_tool_calls && !prev_has_tool_calls {
                prev_msg.tool_calls = msg.tool_calls.clone();
            } else if curr_has_tool_calls && prev_has_tool_calls {
                let prev_tc = prev_msg.tool_calls.get_or_insert_with(Vec::new);
                let existing_ids: std::collections::HashSet<String> = prev_tc
                    .iter()
                    .filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect();
                for tc in msg.tool_calls.iter().flatten() {
                    let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    if !existing_ids.contains(id) {
                        prev_tc.push(tc.clone());
                    }
                }
            }

            debug_log(
                "MERGED CONSECUTIVE ASSISTANTS",
                Some(&json!({
                    "prev_has_tool_calls": prev_has_tool_calls,
                    "curr_has_tool_calls": curr_has_tool_calls,
                })),
            );
        } else {
            safe_messages.push(msg);
        }
    }

    safe_messages
}

// === Main Query Function ===

/// chat_template_kwargs for the thinking gate, or None to send no kwargs at
/// all. Split out so the decision is testable without a server.
fn thinking_template_kwargs(clear_thinking: bool, keep_thinking: bool) -> Option<Value> {
    if !clear_thinking {
        return None;
    }
    let clear = !keep_thinking;
    Some(json!({"clear_thinking": clear, "drop_thinking": clear}))
}

/// The typed slice of the Python `options` dict passed to query_model
/// (keys read anywhere: 'model', 'max_tokens', 'no_reasoning', 'keep_thinking';
/// 'dangerously_skip_permissions' is carried for parity).
#[derive(Clone, Default)]
pub struct QueryOptions {
    pub dangerously_skip_permissions: bool,
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    /// Ask for the least thinking the server will do on THIS request.
    pub no_reasoning: bool,
    /// Keep the thinking already done — do not let the template delete it.
    ///
    /// Separate from `no_reasoning` on purpose, though the recovery paths set
    /// both. The two are different moments: no_reasoning is about the reasoning
    /// this request may produce, keep_thinking about the reasoning previous
    /// requests already produced. Compaction wants the first and not the
    /// second — it sets no_reasoning to save the wait, and carrying every
    /// earlier turn's reasoning into the one request whose purpose is to shrink
    /// the history is the opposite of what it is for.
    pub keep_thinking: bool,
}

/// The parameter bundle every backend query takes (the Python function's
/// keyword signature as a struct).
#[derive(Clone, Default)]
pub struct QueryModelParams {
    pub messages: Vec<Message>,
    pub system_prompt: Vec<String>,
    pub tools: Vec<Arc<dyn Tool>>,
    pub signal: Option<CancellationToken>,
    pub options: QueryOptions,
    pub on_token_update: Option<Arc<dyn Fn(i64) + Send + Sync>>,
    pub on_content_delta: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    pub on_thinking_update: Option<Arc<dyn Fn(i64) + Send + Sync>>,
}

/// Incremental UTF-8 decode: a multibyte char (a 3-byte CJK codepoint) can
/// straddle two byte chunks. Decoding each chunk in isolation would turn the
/// split halves into U+FFFD. This holds an incomplete trailing sequence until
/// the next chunk completes it (errors='replace' semantics otherwise).
/// (Encrypted mode is base64/ASCII on the wire so it never split; plaintext
/// mode did — this makes both paths safe.)
struct IncrementalUtf8 {
    pending: Vec<u8>,
}

impl IncrementalUtf8 {
    fn new() -> Self {
        IncrementalUtf8 { pending: Vec::new() }
    }

    fn decode(&mut self, chunk: &[u8]) -> String {
        self.pending.extend_from_slice(chunk);
        let data = &self.pending;
        let mut hold = 0usize;
        for i in 1..=data.len().min(3) {
            let b = data[data.len() - i];
            if b & 0b1100_0000 == 0b1000_0000 {
                continue; // continuation byte — keep scanning back
            }
            let need = if b >= 0xF0 {
                4
            } else if b >= 0xE0 {
                3
            } else if b >= 0xC0 {
                2
            } else {
                1
            };
            if need > i {
                hold = i; // incomplete multi-byte sequence: hold it back
            }
            break;
        }
        let split = data.len() - hold;
        let out = String::from_utf8_lossy(&data[..split]).into_owned();
        self.pending.drain(..split);
        out
    }
}

/// Accumulated streaming state for one completion.
struct StreamState {
    full_content: String,
    /// True once the server emits any reasoning_content this turn — i.e.
    /// thinking arrived on its own channel. Any <think> left in `content` is
    /// then a known Qwen3.6-MTP glitch (answer text wrongly tag-wrapped;
    /// llama.cpp#22398), so clean_special_tokens keeps the inner text instead
    /// of deleting it.
    got_reasoning: bool,
    tool_calls: Vec<Value>,
    usage: Option<Value>,
    /// last non-null finish_reason from the stream; mapped to the Anthropic
    /// stop_reason below so query.rs can detect a max_output truncation
    /// ('length' -> 'max_tokens') and run recovery.
    final_finish_reason: Option<String>,
    /// live per-chunk tally over answer + reasoning deltas, for the spinner
    /// only; the final include_usage chunk supplies the exact end count.
    streamed_tokens: i64,
    /// prompt_tokens_details.cached_tokens, tracked SEPARATELY from `usage`
    /// because that field is overwritten wholesale by every chunk carrying
    /// usage. With continuous_usage_stats the server sends usage on EVERY
    /// chunk and only some carry the details block, so a final chunk without
    /// it would erase a real cache hit. Keep the largest value ever reported:
    /// cached_tokens describes the prompt, which does not change mid-stream,
    /// so it never legitimately shrinks.
    cached_prompt_tokens: i64,
    reasoning_text: String,
    thinking_tokens: i64,
    last_think_count_at: Option<Instant>,
    /// Rolling window of the most recent output characters (answer AND
    /// reasoning), with the per-character tally that goes with it. Used only
    /// by note_output.
    run_window: std::collections::VecDeque<char>,
    run_counts: std::collections::HashMap<char, usize>,
    /// Set once the window has collapsed; the read loop turns this into an
    /// abort. Carries the message the user is shown.
    degenerate: Option<String>,
    /// When the server last made PROGRESS: an answer/reasoning delta, or a
    /// rising completion_tokens under continuous_usage_stats. `None` until the
    /// first of either, which is what separates "still prefilling" from
    /// "started, then went silent". Drives the stall guards in the read loop.
    last_output_at: Option<Instant>,
    /// Set when a stall guard cut the stream; carries the line the user is
    /// shown. Kept separate from `degenerate` because the two end the turn
    /// differently: a collapsed decoder is handed to Layer-2 recovery, a
    /// server that simply stopped sending is not.
    stall_reason: Option<String>,
    /// One warning per turn from check_silent_reasoning.
    silent_reasoning_logged: bool,
}

impl StreamState {
    fn new() -> Self {
        StreamState {
            full_content: String::new(),
            got_reasoning: false,
            tool_calls: Vec::new(),
            usage: None,
            final_finish_reason: None,
            streamed_tokens: 0,
            cached_prompt_tokens: 0,
            reasoning_text: String::new(),
            thinking_tokens: 0,
            last_think_count_at: None,
            run_window: std::collections::VecDeque::new(),
            run_counts: std::collections::HashMap::new(),
            degenerate: None,
            last_output_at: None,
            stall_reason: None,
            silent_reasoning_logged: false,
        }
    }

    /// Degenerate-run guard. Feed every output character, answer and reasoning
    /// alike; trip when the last N of them hold almost no distinct characters.
    ///
    /// Observed 2026-09-10, twice: GLM-5.3 collapsed onto one token and
    /// emitted it until max_tokens — 53,859 consecutive '!' on the reasoning
    /// channel with no answer text at all. Nothing stopped it. The turn took
    /// 63 minutes, and the max_tokens recovery then ran it three more times,
    /// for 4h12m of a session spent generating one character.
    ///
    /// The server-side n-gram detector is not the answer here. It is off for
    /// GLM on purpose (config::repetition_detection_enabled): enumerating an
    /// API holds a command skeleton fixed and varies one field, so byte-
    /// identical tool calls are ORDINARY there, and turning it on cut a live
    /// probe mid-enumeration on 2026-09-06. That detector judges a loop on
    /// n-gram shape, which real work can share.
    ///
    /// Window for the SINGLE-character case. A collapse onto one token is the
    /// only shape ever actually observed here, and real work does not repeat
    /// one character anywhere near this far (measured max 40), so this tier
    /// catches it in a fraction of the time the two-character window needs:
    /// ~12s against ~126s on the turns measured 2026-09-13.
    const SINGLE_CHAR_RUN_CHARS: usize = 200;

    /// This one asks a different question — how many DISTINCT characters are
    /// in the recent output — and real work cannot answer it the same way. Two
    /// distinct characters across two thousand of them is not enumeration, not
    /// a table, not base64, not minified code, not CJK prose. It is a stuck
    /// decoder. So the guard is family-independent and always on.
    fn note_output(&mut self, s: &str, window: usize) {
        // Recorded even when the collapse guard is off: the stall guards in the
        // read loop need to know a token arrived, which is a separate question.
        if !s.is_empty() {
            self.last_output_at = Some(Instant::now());
        }
        if window == 0 || self.degenerate.is_some() {
            return;
        }
        for ch in s.chars() {
            self.run_window.push_back(ch);
            *self.run_counts.entry(ch).or_insert(0) += 1;
            if self.run_window.len() > window {
                if let Some(old) = self.run_window.pop_front() {
                    if let Some(n) = self.run_counts.get_mut(&old) {
                        *n -= 1;
                        if *n == 0 {
                            self.run_counts.remove(&old);
                        }
                    }
                }
            }
            // Two tiers, because one distinct character and two are not the
            // same claim. Measured 2026-09-13 over 449 text blocks of this
            // deployment's own transcripts: the longest run real work produces
            // is 40 characters when they are all ONE character, and 47 when
            // two are allowed. Every collapse actually observed — eight in one
            // session that day, plus the 53,859 above — was a single character.
            // So the single-character case can trip 10x sooner without coming
            // near anything real, while the two-character case keeps the wide
            // window that protects rules, indentation and separator lines.
            let single_window = std::cmp::min(Self::SINGLE_CHAR_RUN_CHARS, window);
            let tripped = if self.run_counts.len() <= 1 {
                self.run_window.len() >= single_window
            } else {
                self.run_window.len() >= window && self.run_counts.len() <= 2
            };
            if tripped {
                let n = if self.run_counts.len() <= 1 { single_window } else { window };
                let mut seen: Vec<String> =
                    self.run_counts.keys().map(|c| format!("{c:?}")).collect();
                seen.sort();
                self.degenerate = Some(format!(
                    "the model stopped making progress: the last {n} characters of output contain only {} ({}). Generation was cut here.",
                    seen.len(),
                    seen.join(", ")
                ));
                return;
            }
        }
    }

    /// Record that the server made progress, WITHOUT feeding the collapse
    /// guard.
    ///
    /// Every delta that carries characters — answer, reasoning, tool-call
    /// arguments — goes through note_output, which also fills the
    /// distinct-character window. vLLM's continuous_usage_stats running
    /// completion_tokens is the one source with no text behind it: it proves
    /// the server decoded something, but feeding it a synthetic character
    /// would poison a guard whose whole test is how many DISTINCT characters
    /// the recent output holds. So it marks progress here, and only that.
    ///
    /// Measured 2026-09-12: a turn whose entire output was tool-call arguments
    /// had the server counting 40 completion tokens while not one delta of
    /// either kind arrived. The spinner climbed (it reads the usage counter)
    /// and last_output_at stayed None, so the read loop kept the 600s
    /// first-token budget instead of the 120s gap budget and the turn sat
    /// wedged for 603.1s. The screen said "generating"; the guard said
    /// "nothing has ever arrived". They were reading different signals, and
    /// this is the one line that makes them agree.
    fn note_progress(&mut self) {
        self.last_output_at = Some(Instant::now());
    }

    /// Hand the turn to Layer-2 recovery instead of ending it.
    ///
    /// Drops what the collapsed decoder produced — it has already been written
    /// to disk — and raises the one signal query.rs keys its recovery on.
    /// Everything downstream then behaves exactly as it did when the SERVER
    /// raised it: looping tool calls discarded, a nudge injected, up to
    /// REPETITION_RECOVERY_LIMIT attempts, stop_reason normalised to end_turn
    /// so the transcript stays standard.
    ///
    /// The buffers are cleared rather than kept because a collapsed generation
    /// in history is read back on every later turn. That is the same hazard
    /// stale_recovery_indices prunes for spent nudges, and it is worse here:
    /// the model would be re-reading its own garbage as if it were work.
    fn cut_as_repetition(&mut self) {
        self.reasoning_text.clear();
        self.full_content.clear();
        self.tool_calls.clear();
        self.final_finish_reason = Some("repetition".into());
    }

    /// Refresh the exact reasoning-token count, throttled.
    ///
    /// Reasoning-token accounting: the server gives NO breakdown — vLLM's
    /// usage carries only completion_tokens — so counting the reasoning text
    /// ourselves is the only way to know. Counting CHUNKS instead would be
    /// badly wrong: with speculative decoding a delta routinely carries
    /// several tokens. Re-tokenising the whole accumulated string is exact and
    /// cheap, but doing it per delta is O(n²), so it is throttled and always
    /// finalised once when the stream ends.
    fn recount_thinking(&mut self, force: bool, on_thinking_update: &Option<Arc<dyn Fn(i64) + Send + Sync>>) {
        if self.reasoning_text.is_empty() {
            return;
        }
        let now = Instant::now();
        if !force {
            if let Some(last) = self.last_think_count_at {
                if now.duration_since(last).as_secs_f64() < THINK_COUNT_EVERY {
                    return;
                }
            }
        }
        self.last_think_count_at = Some(now);
        // never let token accounting break a stream
        let counted = count_text_tokens(&self.reasoning_text);
        if counted == self.thinking_tokens {
            return;
        }
        self.thinking_tokens = counted;
        if let Some(cb) = on_thinking_update {
            cb(self.thinking_tokens);
        }
        self.check_silent_reasoning();
    }

    /// Cut a turn that is thinking without ever answering.
    ///
    /// The degenerate-run guard above asks whether the recent output has
    /// collapsed onto one character. A turn can fail the other way: reasoning
    /// that keeps producing varied, plausible text and never emits a single
    /// character of content. Nothing was watching for that.
    ///
    /// Measured 2026-09-13 in a live session: two turns generated the full
    /// 65,536-token output budget entirely on the reasoning channel — think=
    /// 65,537 and 65,536 against out=0 — and ran 71 and 28 minutes before the
    /// server's own max_tokens stopped them. The damage did not end there.
    /// Each cut fired MAX_OUTPUT_RECOVERY_PROMPT, which tells the model to
    /// resume mid-thought with no recap, so it resumed without the state the
    /// cut had just destroyed and redid the same work for hours. In that
    /// session's transcript the density of "!!!" runs went from 0.00 per line
    /// before the second cut to 0.99 after it.
    ///
    /// The ceiling is set against the largest legitimate reasoning pass this
    /// deployment has on record: a hard geometry question at
    /// reasoning_effort=high spent 26,271 completion tokens before answering
    /// (see config.max_tokens). SILENT_REASONING_TOKENS is comfortably above
    /// that and far below the 65,536 the runaways reached.
    ///
    /// This only RECORDS. It does not cut, because there is no way to tell a
    /// runaway from a hard question by volume alone: the largest legitimate
    /// pass on record is 26,271 tokens and the runaways reached 65,536, which
    /// is a factor of two, not a cliff. Both turns were stopped by the server's
    /// own max_tokens rather than by finishing, so neither says where the
    /// thinking would have ended. Throwing away an hour of reasoning on a guess
    /// is the more expensive mistake; the fix that matters is carrying it into
    /// the recovery request, which the clear_thinking gate now does.
    fn check_silent_reasoning(&mut self) {
        if self.degenerate.is_some() || !self.full_content.trim().is_empty() {
            return;
        }
        if self.thinking_tokens >= SILENT_REASONING_TOKENS && !self.silent_reasoning_logged {
            self.silent_reasoning_logged = true;
            crate::utils::log::log_error(
                &format!(
                    "the model has produced {} reasoning tokens and no answer yet. Not cut —                      a hard question can legitimately think this long, and the reasoning is                      carried into the recovery request if the server's max_tokens ends the turn.",
                    self.thinking_tokens
                ),
                Some("openai_common: silent reasoning"),
            );
        }
    }
}

/// Reasoning-token ceiling for a turn that has emitted no content — see
/// StreamState::check_silent_reasoning. 40,000 sits above the largest
/// legitimate reasoning pass on record here (26,271) and well below the
/// 65,536 two runaway turns reached on 2026-09-13.
const SILENT_REASONING_TOKENS: i64 = 40_000;

struct SseContext<'a> {
    /// How many trailing characters the degenerate-run guard looks at. 0 = off.
    degenerate_window: usize,
    aes_key: Option<[u8; 32]>,
    on_token_update: &'a Option<Arc<dyn Fn(i64) + Send + Sync>>,
    on_content_delta: &'a Option<Arc<dyn Fn(&str) + Send + Sync>>,
    on_thinking_update: &'a Option<Arc<dyn Fn(i64) + Send + Sync>>,
    rt_filter: Option<crate::utils::realtime_render::RealtimeStreamFilter>,
}

/// Write what a cut generation had produced to logs/<kind>-<ts>.txt.
///
/// Same shape as the aborted-turn dump and for the same reason: the output is
/// about to be dropped from the conversation on purpose, and dropping it from
/// the record too is how a 2,500-second stall ends up needing a core dump to
/// explain. Best effort — a failure here is logged, never raised.
fn write_cut_dump(kind: &str, why: &str, state: &StreamState, elapsed_s: f64) {
    let dir = crate::utils::config::LOG_DIR.clone();
        // ensure_log_dir and atomic_write_bytes, not a bare create_dir_all and
    // fs::write: the bare pair lands the directory 0755 and the file 0644,
    // and these dumps hold a generation's whole reasoning and content — the
    // same request material every other file in logs/ is 0600 for. The
    // atomic write also means a reader never catches a half-written dump.
    crate::utils::log::ensure_log_dir();
    let name = format!(
        "{kind}-{}.txt",
        crate::utils::log::now_iso().replace([':', '.'], "-")
    );
    let body = format!(
        "cut after {elapsed_s:.1}s
reason: {why}
reasoning: {} chars
content: {} chars

===== REASONING =====
{}

===== CONTENT =====
{}
",
        state.reasoning_text.chars().count(),
        state.full_content.chars().count(),
        state.reasoning_text,
        state.full_content,
    );
    if let Err(e) = crate::utils::file::atomic_write_bytes(
        &dir.join(&name).to_string_lossy(),
        body.as_bytes(),
    ) {
        crate::utils::log::log_error(
            &format!("could not write the {kind} dump: {e}"),
            Some("openai_common: cut generation"),
        );
    }
}

/// What the read loop should do once a line has been handled.
///
/// `[DONE]` used to be swallowed here, leaving the loop to learn the response
/// was over only when the byte stream ended. A server that holds the
/// connection open after sending it therefore left the loop parked inside
/// `stream.next()` until a stall guard fired: measured 2026-09-12 as a 600.0s
/// wait that began ~3.1s into a turn whose usage chunk had ALREADY reported
/// the final token count. The terminator the protocol hands us is now used as
/// one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SseFlow {
    Continue,
    Done,
}

/// Handle one `data: ...` SSE line. Extracted so the trailing buffer (a final
/// line with no newline) can be processed too.
fn process_sse_line(
    line: &str,
    state: &mut StreamState,
    ctx: &mut SseContext,
) -> Result<SseFlow, String> {
    if line.trim().is_empty() || !line.starts_with("data: ") {
        return Ok(SseFlow::Continue);
    }

    let mut data = line[6..].to_string();
    if data == "[DONE]" {
        return Ok(SseFlow::Done);
    }

    // Encrypted stream: each SSE event is an independent AES-GCM message;
    // decrypt back to the original JSON line.
    if let Some(key) = &ctx.aes_key {
        match crate::utils::body_crypto::decrypt(key, &data) {
            Ok(plain) => data = String::from_utf8_lossy(&plain).into_owned(),
            Err(e) => return Err(format!("stream body decryption failed: {e}")),
        }
    }

    let chunk_data: Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return Ok(SseFlow::Continue),
    };
    // Server-side error delivered INSIDE a 200 SSE stream (the HTTP status was
    // already 200). vLLM emits {"error": {...}} as a data chunk — with no
    // `choices` — for e.g. an unsupported sampling param under speculative
    // decoding. Without catching it here the loop drains to EMPTY content (no
    // delta, no finish_reason) — the invisible blank-response failure.
    if let Some(err) = chunk_data.get("error") {
        if !err.is_null() {
            let emsg = err
                .get("message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| err.to_string());
            debug_log("STREAM_ERROR_CHUNK", Some(&json!({"error": err})));
            return Err(format!("Model API error (in-stream): {emsg}"));
        }
    }
    // The final include_usage chunk has choices: [] (empty) — guard against
    // an index error; its usage is captured below.
    let empty_choice = json!({});
    let choice = chunk_data
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .unwrap_or(&empty_choice);
    let empty_delta = json!({});
    let delta = choice.get("delta").unwrap_or(&empty_delta);

    if let Some(finish_reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
        state.final_finish_reason = Some(finish_reason.to_string());
        debug_log(
            "STREAM_FINISH",
            Some(&json!({
                "finish_reason": finish_reason,
                "content_length": state.full_content.chars().count(),
                "tool_calls_count": state.tool_calls.len(),
            })),
        );
    }

    // vLLM reasoning parsers differ by model: DeepSeek-V4 streams `reasoning_content`,
    // the GLM-5.3 fork streams `reasoning`. Accept both (same channel).
    let reasoning_delta = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(|r| r.as_str())
        .filter(|r| !r.is_empty());
    if let Some(rc) = reasoning_delta {
        state.got_reasoning = true;
        state.reasoning_text.push_str(rc);
        state.note_output(rc, ctx.degenerate_window);
        state.recount_thinking(false, ctx.on_thinking_update);
        debug_log(
            "REASONING_CONTENT_CHUNK",
            Some(&json!({"length": rc.chars().count()})),
        );
    }

    let content_delta =
        delta.get("content").and_then(|c| c.as_str()).filter(|c| !c.is_empty());
    if let Some(c) = content_delta {
        state.full_content.push_str(c);
        state.note_output(c, ctx.degenerate_window);
        // -rt: stream the safe portion to the screen now.
        if let Some(rt) = ctx.rt_filter.as_mut() {
            let safe = rt.feed(c);
            if !safe.is_empty() {
                if let Some(cb) = ctx.on_content_delta {
                    cb(&safe);
                }
            }
        }
    }

    // Advance the live spinner counter for BOTH answer and reasoning deltas —
    // reasoning is real generation, so the count must not freeze during long
    // thinking. Per-chunk tally; the exact end count comes from the final
    // include_usage chunk.
    if content_delta.is_some() || reasoning_delta.is_some() {
        crate::utils::tokens::mark_output_started();
        state.streamed_tokens += 1;
        if let Some(cb) = ctx.on_token_update {
            cb(state.streamed_tokens);
        }
    }

    if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        // A tool-call delta is output too. Without this a turn whose whole
        // answer is a tool call looked, to the stall guards, like a server that
        // never answered (2026-09-12: 603.1s on the first-token budget), and a
        // decoder that collapsed INSIDE an argument string was invisible to the
        // degenerate-run guard.
        state.note_progress();
        for tc in tcs {
            let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            while state.tool_calls.len() <= idx {
                state.tool_calls.push(json!({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                }));
            }
            if let Some(id) = tc.get("id").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                state.tool_calls[idx]["id"] = Value::String(id.to_string());
            }
            if let Some(name) = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                state.tool_calls[idx]["function"]["name"] = Value::String(name.to_string());
            }
            if let Some(args) = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                let existing = state.tool_calls[idx]["function"]["arguments"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                state.tool_calls[idx]["function"]["arguments"] =
                    Value::String(format!("{existing}{args}"));
                state.note_output(args, ctx.degenerate_window);
            }
        }
    }

    if let Some(usage) = chunk_data.get("usage") {
        if !usage.is_null() {
            state.usage = Some(usage.clone());
            // OpenAI-compatible servers report a prefix-cache hit as
            // prompt_tokens_details.cached_tokens (a SUBSET of prompt_tokens).
            // Nothing here ever read it, so every response recorded
            // cache_read_input_tokens=0 and a fully warm 234k-token turn was
            // indistinguishable in the logs from a cold one that re-prefilled
            // all 234k — exactly the measurement needed when a turn stalls for
            // minutes, since a cold prefill of a ~200k prompt is ~3 minutes of
            // silence that looks identical to a hung server.
            if let Some(cached) = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_i64())
            {
                if cached > state.cached_prompt_tokens {
                    state.cached_prompt_tokens = cached;
                }
            }
            // vLLM continuous_usage_stats: running completion_tokens on each
            // chunk -> use it as the EXACT live count (the tally above
            // under-counts multi-token chunks). Monotonic guard; harmless on
            // servers that only send usage in the final chunk.
            if let Some(ct) = usage.get("completion_tokens").and_then(|v| v.as_i64()) {
                if ct > state.streamed_tokens {
                    state.streamed_tokens = ct;
                    // The stall guards have to see this too, or a turn that
                    // emits only tool-call arguments looks identical to a
                    // server that never answered. See note_progress.
                    state.note_progress();
                    if let Some(cb) = ctx.on_token_update {
                        cb(state.streamed_tokens);
                    }
                }
            }
        }
    }

    Ok(SseFlow::Continue)
}

/// Query an OpenAI-compatible chat-completions server (llama.cpp or vLLM).
///
/// continuous_usage_stats: when true (vLLM), ask the server for a running
/// completion_tokens on every chunk via stream_options; when false
/// (llama.cpp), usage arrives only in the final chunk and the live count uses
/// a per-chunk tally until then.
///
/// reasoning_roundtrip: when true (vLLM serving DeepSeek-V4), this turn's
/// reasoning is stored on the assistant message and sent back as
/// `reasoning_content` on every later request. DeepSeek's thinking API
/// requires it for turns that made a tool call; a self-hosted server simply
/// drops the reasoning otherwise and the model silently loses what it worked
/// out — it then repeats the same tool call to rediscover it. Off for
/// llama.cpp, which has no reasoning channel.
pub async fn query_openai_compatible(
    params: QueryModelParams,
    continuous_usage_stats: bool,
    repetition_detection: bool,
    reasoning_roundtrip: bool,
    max_images: i64,
) -> AssistantMessage {
    let start_time = Instant::now();
    match query_openai_compatible_inner(
        &params,
        continuous_usage_stats,
        repetition_detection,
        reasoning_roundtrip,
        max_images,
        start_time,
    )
    .await
    {
        Ok(msg) => msg,
        Err(e) => {
            // The error string is the repr-style form: many network errors have
            // an empty display message, and the type-carrying form is what
            // makes the log and the user-facing message diagnosable.
            debug_log("ERROR", Some(&json!({"error": e})));
            let duration_ms = start_time.elapsed().as_millis() as i64;

            let error_message = ApiMessage {
                id: Uuid::new_v4().to_string(),
                message_type: "message".into(),
                role: "assistant".into(),
                content: vec![ContentBlock::Text(TextBlock {
                    text: format!("Error connecting to the model server: {e}"),
                })],
                // API-error message carries zero usage; flag synthetic so it
                // can't anchor the token count
                model: "<synthetic>".into(),
                stop_reason: Some("end_turn".into()),
                stop_sequence: None,
                usage: BetaUsage::default(),
            };

            AssistantMessage {
                message: error_message,
                cost_usd: 0.0,
                duration_ms,
                uuid: Uuid::new_v4().to_string(),
                is_api_error_message: true,
            }
        }
    }
}

/// Turn a finished stream into the assistant message body and its stop_reason.
///
/// Extracted from the streaming function so it can be measured without a model
/// turn: the rules below (what survives assembly, and what the stop_reason is
/// allowed to claim as a result) are exactly the ones a live session got wrong,
/// and they are only checkable here.
struct AssembledMessage {
    content: Vec<ContentBlock>,
    stop_reason: String,
    /// Tool calls the server sent that produced no invocation, one line each.
    /// Returned rather than only logged so the rule that separates a real loss
    /// from a harmless index filler is measurable without reading a log file.
    lost_tool_calls: Vec<String>,
}

fn assemble_content_and_stop_reason(
    state: &StreamState,
    reasoning_roundtrip: bool,
    aborted: bool,
) -> AssembledMessage {
    // Build response content
    let mut content_blocks: Vec<ContentBlock> = Vec::new();

    // Keep this turn's thinking on the message so the next request can send it
    // back (see ChatMessage.reasoning_content). It leads the block list
    // because that is the order the model produced it in: think, then answer
    // or call a tool. Storing it costs context but losing it costs the
    // conclusion. Gated so a backend without the channel never stores it.
    if reasoning_roundtrip && !state.reasoning_text.trim().is_empty() {
        content_blocks.push(ContentBlock::Thinking(ThinkingBlock {
            thinking: state.reasoning_text.clone(),
        }));
    }

    // Recover any tool call that leaked into `content` (DeepSeek-V4 #1244)
    // BEFORE cleaning strips the markup. On the normal path — model used the
    // structured tool_calls channel — this is a no-op and residual == raw.
    let (recovered_calls, residual_content) = recover_leaked_tool_calls(&state.full_content);
    if !recovered_calls.is_empty() {
        debug_log(
            "RECOVERED_LEAKED_TOOL_CALLS",
            Some(&json!({
                "count": recovered_calls.len(),
                "names": recovered_calls.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
                "finish_reason": state.final_finish_reason,
                "n_structured": state.tool_calls.len(),
            })),
        );
    }

    // Clean and add text content (from the residual, sans leaked tool markup)
    if !residual_content.is_empty() {
        let cleaned_content = clean_special_tokens(&residual_content, state.got_reasoning);
        if state.got_reasoning && residual_content.contains("<think>") {
            // A leaked-think glitch (llama.cpp#22398): record it so the
            // frequency is visible, and confirm we preserved the inner text.
            debug_log(
                "INLINE_THINK_RECOVERED",
                Some(&json!({
                    "raw_len": residual_content.chars().count(),
                    "cleaned_len": cleaned_content.chars().count(),
                })),
            );
        }
        if !cleaned_content.is_empty() {
            content_blocks.push(ContentBlock::Text(TextBlock { text: cleaned_content }));
        }
    }

    // Signatures of structured tool calls, so a recovered duplicate (same call
    // echoed both on-channel and in content) isn't dispatched twice.
    let mut seen_sigs: std::collections::HashSet<(String, String)> = Default::default();

    // Tool calls the server sent that could not be turned into an invocation.
    // Collected rather than discarded: a dropped call is how a turn ends having
    // ANNOUNCED work it never did, which reads to the user as the model quietly
    // stopping. Everything that can go wrong upstream — a stream cut mid
    // 'arguments' so the JSON is truncated, an 'arguments' that arrives as an
    // object instead of a string (as_str() yields None), a server that numbers
    // tool_calls from a nonzero index — converges HERE, so this one count
    // covers all of them without having to tell them apart.
    let mut lost_tool_calls: Vec<String> = Vec::new();

    // Add structured tool calls (the normal channel)
    for tc in &state.tool_calls {
        let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
        let args = tc["function"]["arguments"].as_str().unwrap_or("");
        // An index filler from the delta loop: a server that opens at index 1
        // leaves an untouched slot at 0. It is not a call the model made, so it
        // is neither dispatched nor counted as lost.
        if name.is_empty() && args.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(args) {
            Ok(tool_input) => {
                let id = tc["id"].as_str().unwrap_or("");
                let id = if id.is_empty() { Uuid::new_v4().to_string() } else { id.to_string() };
                seen_sigs.insert((name.clone(), sorted_json_dump(&tool_input)));
                content_blocks.push(ContentBlock::ToolUse(ToolUseBlock {
                    id,
                    name,
                    input: tool_input,
                }));
            }
            Err(e) => {
                lost_tool_calls.push(format!("{}: {e}", if name.is_empty() { "<unnamed>" } else { &name }));
                debug_log("FAILED TO PARSE TOOL ARGUMENTS", Some(&json!({"tool_call": tc})));
            }
        }
    }

    // Add recovered (leaked) tool calls, de-duplicated against structured ones
    for rc in &recovered_calls {
        let sig = (rc.name.clone(), sorted_json_dump(&rc.input));
        if seen_sigs.contains(&sig) {
            continue;
        }
        seen_sigs.insert(sig);
        content_blocks.push(ContentBlock::ToolUse(ToolUseBlock {
            id: Uuid::new_v4().to_string(),
            name: rc.name.clone(),
            input: rc.input.clone(),
        }));
    }

    // A lost call goes to the ERROR log, not just the debug log: debug_log only
    // writes when CODEMASON_DEBUG is set, so under an unattended run the one
    // record of why the turn went quiet is the one record that does not exist.
    //
    // But NOT when the user aborted. Cutting the stream mid-tool-call leaves
    // the arguments half-written by definition, so the parse "failure" is the
    // expected outcome of pressing ESC, not a defect. Logging it as one cost
    // real time: 56 of 60 such records in this user's log sat at column 0 or
    // column 9 — the first argument fragment, the signature of a cut, not of a
    // model that emits bad JSON — and they were read as evidence of a
    // truncation bug that did not exist.
    if !lost_tool_calls.is_empty() && !aborted {
        crate::utils::log::log_error(
            &format!(
                "dropped {} tool call(s) the model sent — arguments did not parse: {}",
                lost_tool_calls.len(),
                lost_tool_calls.join("; ")
            ),
            Some("openai_common: tool call assembly"),
        );
    }

    // Ensure at least one content block
    if content_blocks.is_empty() {
        content_blocks.push(ContentBlock::Text(TextBlock { text: String::new() }));
    }

    debug_log(
        "FINAL CONTENT ARRAY",
        Some(&json!({
            "content_length": content_blocks.len(),
            "content_types": content_blocks.iter().map(|c| c.block_type()).collect::<Vec<_>>(),
        })),
    );

    // Map the OpenAI/llama.cpp finish_reason to the Anthropic stop_reason
    // (substrate boundary). 'length' means the output hit the token cap and
    // was cut mid-stream -> 'max_tokens', which query.rs's max_output recovery
    // keys on to resume the model instead of ending the turn. A completed tool
    // call -> 'tool_use'; everything else -> 'end_turn'. A recovered leaked
    // call is a real invocation too — it must drive the 'tool_use' stop_reason
    // so query.rs dispatches it instead of ending the turn.
    // 'repetition' (vLLM) is checked FIRST and always ends the turn: it must
    // not reach the two branches below — 'max_tokens' would resume the loop,
    // and 'tool_use' would dispatch a tool call the model emitted while
    // already repeating itself. The cut also leaves the text ending
    // mid-thought, so it is labelled: the note names the cause rather than
    // leaving the model to infer one.
    // An aborted stream ends the turn. Never 'tool_parse_failed': that is the
    // signal query.rs keys its retry policy on, and re-issuing a call the user
    // just stopped is the opposite of what they asked for.
    if aborted {
        // Two different events arrive here and they must not end the turn the
        // same way: the user pressed Esc, or a stall guard cut a server that
        // had stopped sending. Only the first is an interrupt.
        if let Some(why) = &state.stall_reason {
            // Say what happened IN THE TRANSCRIPT, where the person reading it
            // is. Until this line the only record was one entry in error.log,
            // so a stalled turn showed on screen as "[Request interrupted by
            // user]" and left the user to conclude they had stopped it —
            // measured 2026-09-12, where the turn's real cause (600s with no
            // token) was reachable only by opening the log directory.
            content_blocks
                .retain(|b| !matches!(b, ContentBlock::Text(t) if t.text.is_empty()));
            content_blocks.push(ContentBlock::Text(TextBlock {
                text: format!("\n\n[Stopped: {why}]"),
            }));
            // A tool call the model finished emitting is real work: fully
            // received, fully parsed, and the stall happened AFTER it. Ending
            // on end_turn threw it away without a word — the turn announced a
            // step and never ran it, which is the exact failure lost_tool_calls
            // exists to make impossible. A truncated call never becomes a
            // ToolUse block, so what survives this far is only ever complete.
            if content_blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse(_))) {
                debug_log(
                    "STALL_KEPT_TOOL_USE",
                    Some(&json!({"tool_calls": state.tool_calls.len()})),
                );
                return AssembledMessage {
                    content: content_blocks,
                    stop_reason: "tool_use".to_string(),
                    lost_tool_calls,
                };
            }
            // Nothing to dispatch: surface the raw signal and let query.rs own
            // the policy, exactly as "repetition" and "tool_parse_failed" do.
            return AssembledMessage {
                content: content_blocks,
                stop_reason: "stalled".to_string(),
                lost_tool_calls,
            };
        }
        return AssembledMessage {
            content: content_blocks,
            stop_reason: "end_turn".to_string(),
            lost_tool_calls,
        };
    }

    let stop_reason: &str = match state.final_finish_reason.as_deref() {
        // Surface the raw vLLM signal; query.rs owns the policy (layer-2
        // recovery: nudge the model to self-correct N times before giving up).
        // The baked-in end_turn + "[Generation stopped]" note moved there.
        Some("repetition") => "repetition",
        Some("length") => "max_tokens",
        _ => {
            // Derived from what was actually BUILT, never from what arrived.
            // The two used to be able to disagree: the old test was "did the
            // server send tool calls", so a call that failed to parse still set
            // 'tool_use' while producing no ToolUse block, and query.rs read
            // that as a turn with nothing to dispatch and ended silently.
            // Asking the content array removes the disagreement by construction.
            if content_blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse(_))) {
                "tool_use"
            } else if !lost_tool_calls.is_empty() {
                // The model DID call a tool and none of it survived. Not an
                // end_turn — say so, and let query.rs own the recovery policy.
                "tool_parse_failed"
            } else if state.full_content.trim().is_empty() {
                // Finished cleanly having produced nothing at all: no text, no
                // tool call, nothing lost in parsing. That is not a turn ending,
                // it is a turn that never happened.
                //
                // Observed 2026-09-07 in an unattended run: after a command
                // returned an empty result (a sed range past the end of the
                // file) the model spent 125 reasoning tokens and answered with
                // zero characters. finish_reason said "stop", so the turn ended
                // and the session sat waiting for the user — the silent stop
                // that -dsp exists to avoid, with nothing on screen to say why.
                //
                // The tool-call check above has to come first: answering with a
                // tool call and no prose is normal and must not land here.
                "empty_turn"
            } else {
                "end_turn"
            }
        }
    };
    // The SERVER-raised repetition signal used to leave the collapsed
    // generation in the message, while the CLIENT-side detector
    // (cut_as_repetition) clears its buffers for exactly the reason documented
    // there: a collapsed generation in history is read back on every later
    // turn. The asymmetry meant a server-side cut put the "!!!!!"/"★★★★" run
    // into history and re-sent it on every subsequent request — and because
    // stale_recovery_indices only prunes an EMPTY assistant stub, it removed
    // the explaining nudge and left the garbage behind. Same treatment on both
    // paths: the run has already been streamed to the screen, so nothing the
    // user should see is lost by not storing it.
    if stop_reason == "repetition" {
        content_blocks.clear();
    }
    AssembledMessage {
        content: content_blocks,
        stop_reason: stop_reason.to_string(),
        lost_tool_calls,
    }
}

async fn query_openai_compatible_inner(
    params: &QueryModelParams,
    continuous_usage_stats: bool,
    repetition_detection: bool,
    reasoning_roundtrip: bool,
    max_images: i64,
    start_time: Instant,
) -> Result<AssistantMessage, String> {
    let cfg = crate::utils::config::get_config();
    let options = &params.options;

    let chat_messages = convert_to_chat_messages(
        &params.messages,
        &params.system_prompt,
        reasoning_roundtrip,
        max_images,
    );

    // Derive the language to instruct the model in FROM THE CONVERSATION (the
    // last thing the user actually typed), so it works on a machine whose
    // config.json never set `language`. config.language still wins when set.
    {
        // A vision message carries its text in content_parts, not content, so
        // reading only `content` would skip the newest turn and detect the
        // language from an older one.
        let last_user = chat_messages
            .iter()
            .rev()
            .filter(|m| m.role == "user")
            .map(chat_message_text)
            .find(|t| !t.is_empty())
            .unwrap_or_default();
        crate::utils::language::set_request_language(
            crate::utils::language::detect_language(&last_user).map(String::from),
        );
    }

    let mut tool_schemas: Vec<Value> = Vec::new();
    // Convert tools to OpenAI format. Prefer the faithfully-ported per-tool
    // description (tool_prompts); fall back to the short blurb.
    //
    // Python resolution chain: tool.build_description() (dynamic, e.g. Skill
    // lists available skills) -> tool.DESCRIPTION -> TOOL_DESCRIPTIONS map ->
    // get_simple_description. In the Rust trait both dynamic and static live in
    // Tool::description(); its base default ("Execute {name}") marks "no
    // tool-provided description", which falls through to the map/blurb.
    // Keyed on the tool-set identity. Names are unique per set, so the joined
    // list identifies the block without hashing the (not yet built) schemas.
    let tool_cache_key: String =
        crate::utils::tokens::oob_scope_key(params.tools.iter().map(|t| t.name()));
    if let Some(cached) = TOOL_BLOCK_CACHE.lock().get(&tool_cache_key) {
        tool_schemas = cached.clone();
    }
    if tool_schemas.is_empty() {
    for tool in &params.tools {
        // Resolution chain, verbatim from the Python: the tool's own wire
        // description (build_description() / DESCRIPTION) -> the
        // TOOL_DESCRIPTIONS map -> the simple blurb. NOTE this is
        // wire_description(), never description() — the latter is the short
        // display blurb, and sending it instead silently strips ~7k tokens of
        // tool guidance from every request.
        let description = match tool.wire_description().await {
            Some(d) if !d.is_empty() => d,
            _ => crate::constants::tool_prompts::tool_description(tool.name())
                .map(|s| s.to_string())
                .unwrap_or_else(|| get_simple_description(tool.name())),
        };
        tool_schemas.push(json!({
            "type": "function",
            "function": {
                "name": tool.name(),
                "description": description,
                "parameters": tool.input_json_schema(),
            }
        }));
    }
        if !tool_schemas.is_empty() {
            TOOL_BLOCK_CACHE
                .lock()
                .insert(tool_cache_key.clone(), tool_schemas.clone());
        }
    }

    // Per-call output cap override (e.g. compaction reserves a fixed summary
    // budget); falls back to the configured default.
    let mut max_output_tokens = options.max_tokens.filter(|t| *t != 0).unwrap_or(cfg.max_tokens);

    // Register what the estimate cannot see, now that both parts are built.
    // Cheap: the tool block is cached per tool-set and the system prompt is
    // fixed for the session, so this recomputes the same number each turn and
    // every consumer (compaction included) reads one consistent value.
    {
        let mut oob = 0i64;
        if !tool_schemas.is_empty() {
            oob += count_tokens_precise(&py_json_dumps(&Value::Array(tool_schemas.clone())));
        }
        if !params.system_prompt.is_empty() {
            oob += count_tokens_precise(&params.system_prompt.join("\n\n"));
        }
        crate::utils::tokens::set_out_of_band_tokens(&tool_cache_key, oob);
    }

    // Never let (prompt + requested output) exceed the model's context window.
    // The server rejects that outright — measured on the Python port as
    // 458,753 input + 65,536 output = 524,289 against a 524,288 window, one
    // token over, on a one-line request. The Rust port had no cap at all, so
    // the only thing standing between a full history and a 400 was compaction
    // firing in time.
    //
    // The margin scales with the input: the model's OWN tokenizer counts a
    // little more than any estimate of ours, and that drift grows with the
    // prompt (chat-template wrappers, per-message role tags), so a constant is
    // too small exactly where it matters. ~1.5%, floored at CONTEXT_SAFETY_MARGIN
    // so short prompts are unaffected.
    {
        let ctx = crate::utils::tokens::get_context_size();
        let input_est =
            crate::utils::tokens::estimate_input_tokens(&params.messages, &tool_cache_key);
        let margin = CONTEXT_SAFETY_MARGIN.max(input_est / 64);
        let room = ctx - input_est - margin;
        if room < max_output_tokens {
            max_output_tokens = room.max(MIN_OUTPUT_TOKENS);
            debug_log(
                "OUTPUT_CAPPED_TO_FIT_CONTEXT",
                Some(&json!({
                    "context": ctx,
                    "input_est": input_est,
                    "margin": margin,
                    "capped_max_tokens": max_output_tokens,
                })),
            );
        }
    }

    // Ask the server to emit a final usage chunk with REAL token counts
    // (computed by the model's own tokenizer). include_usage -> final exact
    // counts. continuous_usage_stats (vLLM only) -> a running completion_tokens
    // on EVERY chunk, so the live spinner can show the server's exact count
    // instead of a per-chunk tally that under-counts multi-token
    // (speculative/MTP) chunks. llama.cpp ignores the flag.
    let mut stream_options = json!({"include_usage": true});
    if continuous_usage_stats {
        stream_options["continuous_usage_stats"] = Value::Bool(true);
    }

    // Sampling comes from Config::effective_sampling, not the raw fields: the
    // served family decides. GLM gets the vendor's own numbers (temperature
    // 1.0 / top_p 0.95, no top_k); everything else keeps the stored values it
    // has always used. A `None` here means the key is not sent at all, so the
    // server's default applies instead of one we invented.
    let sampling = cfg.effective_sampling();
    let mut request_body = json!({
        "model": cfg.model_name,
        "messages": chat_messages.iter().map(|m| m.to_value()).collect::<Vec<_>>(),
        "temperature": sampling.temperature,
        "max_tokens": max_output_tokens,
        "stream": true,
        "stream_options": stream_options,
        "top_p": sampling.top_p,
        // llama.cpp prompt-cache hint. vLLM ignores it (prefix caching is
        // automatic/server-side), so it's harmless there and still helps if
        // the backend is llama.cpp.
        "cache_prompt": true,
    });
    // Optional knobs. Anti-runaway penalties were the DeepSeek-V4 8K-repeat
    // guard: not in the server-side DSV4_SERVER_LOCK whitelist, so they pass
    // through even while temperature is locked server-side, and mild by design
    // so code generation isn't distorted.
    if let Some(v) = sampling.top_k {
        request_body["top_k"] = json!(v);
    }
    if let Some(v) = sampling.min_p {
        request_body["min_p"] = json!(v);
    }
    if let Some(v) = sampling.repetition_penalty {
        request_body["repetition_penalty"] = json!(v);
    }
    if let Some(v) = sampling.presence_penalty {
        request_body["presence_penalty"] = json!(v);
    }
    if let Some(v) = sampling.frequency_penalty {
        request_body["frequency_penalty"] = json!(v);
    }

    if !tool_schemas.is_empty() {
        request_body["tools"] = Value::Array(tool_schemas);
    }

    // Thinking opt-in: only carry reasoning_effort when explicitly configured
    // (cfg.reasoning_effort in {'low','medium','high','max'}). Empty => omit the key
    // entirely, which does NOT mean "no thinking" — it hands the decision to
    // the server's own default (a launch flag we do not control from here).
    // options.no_reasoning asks for the least thinking the server will do, for
    // calls that are bookkeeping rather than work — compaction above all, where
    // the reasoning pass runs before the first summary token and doubles the
    // wait, and repetition retries, where the reasoning channel is the thing
    // that collapsed.
    //
    // It sends 'low' rather than omitting the key. Omitting does NOT mean "no
    // thinking": the key is simply absent and the server applies its own launch
    // default, which here is default_chat_template_kwargs={'enable_thinking':
    // True, 'reasoning_effort': 'high'} — so omission bought nothing.
    //
    // Measured 2026-09-13 against this deployment, one geometry question,
    // completion tokens: reasoning_effort=high 737, key omitted 636,
    // chat_template_kwargs={'enable_thinking': False} 728, reasoning_effort=low
    // 417. Omission and enable_thinking=false both land next to high; only
    // 'low' actually moves it. (An earlier note here said low "injects an empty
    // string" on this model family and so meant "no instruction" — that is no
    // longer true of the served model, and it is why this call omitted the key
    // for a month without effect.)
    let eff = cfg.effective_reasoning_effort();
    if options.no_reasoning {
        request_body["reasoning_effort"] = Value::String("low".into());
    } else if !eff.is_empty() {
        request_body["reasoning_effort"] = Value::String(eff);
    }
    // clear_thinking: drop prior-turn reasoning at the template (see Config).
    //
    // Held OFF for recovery requests (options.keep_thinking), because the
    // template decides "prior turn" by user message and every recovery path
    // injects one. The nudge that asks the model to carry on is itself what
    // deletes the thinking it needs to carry on FROM.
    //
    // Traced 2026-09-13 from a live session. A turn spent its whole 65,536-token
    // budget on the reasoning channel and produced no content, so the server cut
    // it at max_tokens and MAX_OUTPUT_RECOVERY_PROMPT fired. The next request's
    // prompt should have grown by that reasoning; instead it FELL from 95,825 to
    // 79,420 — the cut turn's 65,537 tokens gone, and with them the 16,405
    // tokens of in-turn reasoning the tool loop had accumulated, which is the
    // drop exactly. The prompt the model then received told it to "resume
    // mid-thought, no recap" with the thought deleted, and it redid the same
    // work for hours.
    //
    // The trade is prompt size against state: this one request carries the
    // thinking it would otherwise lose. The cost is bounded — the reasoning
    // dies at the next ordinary user turn, as before.
    //
    // The kwarg is named per family and both are sent, because a server that
    // does not know a name ignores it:
    //   GLM-5.3     `clear_thinking` — read by its chat_template.jinja.
    //   DeepSeek-V4 `drop_thinking`  — read by the model's own encoder
    //     (encoding_dsv4.apply_chat_template -> kwargs.get("drop_thinking",
    //     true)). The encoder then forces it OFF whenever any message carries
    //     tools, which is every request this client makes, so on DSV4 prior
    //     reasoning survives regardless and the key is inert.
    // Semantics match: true = keep only this turn's reasoning.
    if let Some(kwargs) = thinking_template_kwargs(cfg.clear_thinking, options.keep_thinking) {
        request_body["chat_template_kwargs"] = kwargs;
    }

    // Server-side repetition detection (vLLM RepetitionDetectionParams). Cuts
    // a generation that has started repeating an n-gram, and reports
    // finish_reason='repetition'. Sent only when the backend advertises it,
    // because llama.cpp has no such field.
    //
    // This is the answer to the loop shape the repeated-tool-call cap cannot
    // see: paragraphs alternating with no tool call between them. Catching it
    // at the sampler stops it mid-generation instead of after the tokens are
    // gone.
    if repetition_detection {
        if let Some(rd) = repetition_detection_params(&cfg) {
            request_body["repetition_detection"] = rd;
        }
    }

    debug_log("REQUEST BODY", Some(&request_body));

    // Prefix fingerprint. Set CODEMASON_PREFIX_DUMP=/path/to/file to append one
    // compact line per request: the per-message hashes of the EXACT payload
    // being sent, in order. Diffing two consecutive lines says precisely which
    // message index first differs, which is the only way to tell a server-side
    // cache eviction (prefix identical, cache gone) apart from a client-side
    // prefix break (prefix changed, so the cache could never have matched).
    //
    // Guessing between those two is what this exists to stop: a 121k-token
    // conversation measured 99.1% cached during a run and 0.0% on the next
    // question with ZERO requests in between — and with no traffic there is no
    // LRU pressure, so "something else evicted it" cannot be the whole story.
    // The bodies themselves are far too large to log (debug_log writes the
    // whole 121k pretty-printed and then rotates it away); hashes are not.
    // Fresh numbers for this request, then ask the server what it is about to
    // re-read (services::prefill_probe). Our own estimate below fills the gap
    // until the server answers, and llama.cpp has no counters so it is all
    // there ever is there.
    crate::utils::tokens::reset_prefill_state();
    crate::services::prefill_probe::probe_next_request();

    // Prefix-break detector — ALWAYS ON, and cheap (two hashes per message).
    //
    // The cached prefix survives only while each request is the previous one
    // plus new messages on the end. Anything else — a message rewritten,
    // dropped, reordered, or shortened — moves the front of the prompt and
    // costs a full re-prefill of the whole conversation. On this deployment
    // that is ~1,335 tok/s, so a 73k conversation pays 55 seconds and a 233k
    // one pays 175, with nothing on screen to say why.
    //
    // It happens for real: a measured session went 73,164 -> 67,117 prompt
    // tokens between two consecutive requests while its transcript only grew,
    // and that turn spent 43 unexplained seconds. Reading it back afterwards
    // could show THAT the history had shrunk but never WHICH message went, so
    // the cause stayed a guess. Recording the divergence point at the moment it
    // happens is the difference between a diagnosis and another theory.
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        static PREV: Lazy<parking_lot::Mutex<Vec<(u64, usize)>>> =
            Lazy::new(|| parking_lot::Mutex::new(Vec::new()));
        let mut now: Vec<(u64, usize)> = Vec::new();
        if let Some(arr) = request_body.get("messages").and_then(|m| m.as_array()) {
            for m in arr {
                let text = py_json_dumps(m);
                let mut h = DefaultHasher::new();
                text.hash(&mut h);
                now.push((h.finish(), text.len()));
            }
        }
        let mut prev = PREV.lock();
        // How much of this request the server cannot serve from its cache:
        // everything from the first message that differs. On a normal append
        // that is just the new tail; on a break it is the whole conversation.
        let diverge = (0..now.len().min(prev.len()))
            .find(|&i| prev[i] != now[i])
            .unwrap_or(prev.len().min(now.len()));
        let tail: String = request_body
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|arr| arr[diverge.min(arr.len())..].iter().map(py_json_dumps).collect())
            .unwrap_or_default();
        crate::utils::tokens::set_pending_prefill(
            crate::utils::tokens::count_text_tokens(&tail),
        );

        // Only compare against a strictly shorter history: a NEW conversation
        // (sub-agent, compaction summary, a fresh session) legitimately starts
        // over and is not a break.
        if !prev.is_empty() && now.len() >= prev.len() {
            if let Some(i) = (0..prev.len()).find(|&i| prev[i] != now[i]) {
                let kept: usize = prev[..i].iter().map(|(_, n)| n).sum();
                let lost: usize = prev[i..].iter().map(|(_, n)| n).sum();
                crate::utils::log::log_error(
                    &format!(
                        "prefix break: message {i} of {} changed, so {lost} chars of cached prompt \
                         must be re-read ({kept} chars still shared). was {}/{} bytes, now {}/{} bytes",
                        prev.len(),
                        prev[i].0,
                        prev[i].1,
                        now[i].0,
                        now[i].1,
                    ),
                    Some("openai_common: cached prefix"),
                );
            }
        }
        *prev = now;
    }

    if let Ok(dump_path) = std::env::var("CODEMASON_PREFIX_DUMP") {
        if !dump_path.is_empty() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut parts: Vec<String> = Vec::new();
            if let Some(arr) = request_body.get("messages").and_then(|m| m.as_array()) {
                for m in arr {
                    let text = py_json_dumps(m);
                    let mut h = DefaultHasher::new();
                    text.hash(&mut h);
                    parts.push(format!("{:x}:{}", h.finish(), text.len()));
                }
            }
            let mut th = DefaultHasher::new();
            request_body
                .get("tools")
                .map(py_json_dumps)
                .unwrap_or_default()
                .hash(&mut th);
            // Per-tool detail, because the aggregate hash only says THAT the
            // tool block changed, never which tool. Both observed cold prefills
            // followed a change here, and descriptions are resolved fresh on
            // every request (wire_description().await), so a tool appearing,
            // vanishing, or merely rewording itself moves the very front of the
            // prompt and invalidates the entire cached prefix behind it.
            let mut tool_detail: Vec<String> = Vec::new();
            if let Some(arr) = request_body.get("tools").and_then(|t| t.as_array()) {
                for t in arr {
                    let name = t
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("?");
                    let mut h = DefaultHasher::new();
                    py_json_dumps(t).hash(&mut h);
                    tool_detail.push(format!("{}:{:x}", name, h.finish()));
                }
            }
            let line = format!(
                "{} n={} tools={:x} ntools={} [{}] {}\n",
                crate::utils::log::now_iso(),
                parts.len(),
                th.finish(),
                tool_detail.len(),
                tool_detail.join(","),
                parts.join(" ")
            );
            use std::io::Write;
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&dump_path)
                .and_then(|mut f| f.write_all(line.as_bytes()));
        }
    }

    let mut state = StreamState::new();

    // Omit the Authorization header entirely when no key is set, so a default
    // (empty-key) install sends an anonymous request rather than a malformed
    // empty header. Matches tokens.rs behaviour.
    // Optional end-to-end BODY encryption (utils/body_crypto). When a key is
    // configured, the request body is AES-256-GCM sealed and base_url should
    // point at the decrypt gateway; when empty this is a no-op and the
    // plaintext body is sent exactly as before.
    let aes_key: Option<[u8; 32]> = if !cfg.model_aes_key.is_empty() {
        Some(
            crate::utils::body_crypto::load_key(&cfg.model_aes_key)
                .map_err(|e| format!("BodyCryptoError({e})"))?,
        )
    } else {
        None
    };

    let url = format!("{}/v1/chat/completions", cfg.model_base_url);
    // No per-request .timeout(): in reqwest that is a TOTAL deadline over the
    // whole streamed body, which cuts any completion longer than the window
    // (a ~1M-token compaction on the slow 2-node setup exceeds 30 minutes with
    // chunks still flowing). The shared client already carries the httpx-parity
    // connect/read-gap timeouts (backend.rs SHARED_CLIENT) — a stream stays
    // alive exactly as long as chunks keep arriving within the window.
    let client = super::backend::get_shared_client();
    let api_key = cfg.model_api_key.clone();
    let body_json = request_body.clone();
    let encrypted_body: Option<String> = aes_key
        .as_ref()
        .map(|k| crate::utils::body_crypto::encrypt(k, request_body.to_string().as_bytes()));

    // Persistent keep-alive client + connect-phase retry (services::backend):
    // reuse one connection across rapid turns and absorb a momentary connect
    // blip instead of dying on the first connect error.
    //
    // The token goes in for the same reason it wraps the read loop below: the
    // ladder waits up to two hours, and until it was handed the abort token
    // there was NOTHING watching Esc or Ctrl+C during that wait.
    let retry_signal = params.signal.clone();
    let response = super::backend::send_with_retry(|| {
        let mut req = client.post(&url);
        if let Some(enc) = &encrypted_body {
            req = req
                .header("Content-Type", "application/octet-stream")
                .header("X-Body-Encrypted", "aesgcm")
                .body(enc.clone().into_bytes());
        } else {
            req = req.header("Content-Type", "application/json").json(&body_json);
        }
        if !api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {api_key}"));
        }
        req
    }, retry_signal.as_ref())
    .await?;

    if response.status().as_u16() != 200 {
        let status = response.status().as_u16();
        let error_text = response.bytes().await.map(|b| b.to_vec()).unwrap_or_default();
        // A rejected auth with no key configured is the one failure whose cause
        // the status alone hides: the request went out with no Authorization
        // header at all, so the server is answering "who are you", not "your
        // key is wrong". This port deliberately ships no baked-in credential
        // (see config.rs), and its note that existing installs already have one
        // does not hold for a fresh config — which leaves a bare 401 and no way
        // to tell it apart from a server that is down or a key that expired.
        //
        // Only on 401/403 and only when the key is empty: an empty key is
        // legitimate against a local server with no auth, so warning up front
        // would be noise on the setup that is working fine.
        let hint = if matches!(status, 401 | 403)
            && crate::utils::config::get_config().model_api_key.is_empty()
        {
            "\n  No API key is configured, so the request was sent unauthenticated. \
             Set model_api_key in ~/.codemason/config.json, or the environment \
             variable CODEMASON_MODEL_API_KEY."
        } else {
            ""
        };
        return Err(format!(
            "Model API error: {status} - {}{hint}",
            String::from_utf8_lossy(&error_text)
        ));
    }

    let mut buffer = String::new();
    // -rt/--realtime: DSML/think/special-token-safe live renderer. Holds back a
    // trailing fragment that could still grow into a marker, so nothing raw
    // leaks even across chunk boundaries.
    let mut ctx = SseContext {
        degenerate_window: cfg.degenerate_run_chars.max(0) as usize,
        aes_key,
        on_token_update: &params.on_token_update,
        on_content_delta: &params.on_content_delta,
        on_thinking_update: &params.on_thinking_update,
        rt_filter: params
            .on_content_delta
            .as_ref()
            .map(|_| crate::utils::realtime_render::RealtimeStreamFilter::new()),
    };
    let mut decoder = IncrementalUtf8::new();

    // Race every read against the abort signal (utils::generators::abortable)
    // instead of only polling BETWEEN chunks. The byte stream blocks until the
    // server sends something, and during PREFILL it sends nothing at all —
    // 122k tokens is ~80s at ~1.5k tok/s — so a between-chunk poll never ran
    // and Esc/Ctrl+C looked completely dead until the first token landed.
    // Racing force-cancels the wedged read after the grace window, tearing the
    // connection down at once, so an interrupt takes effect immediately no
    // matter where the stream is.
    let mut aborted = false;
    let mut stream_done = false;
    // A stall guard has to tear down THIS stream without ending the turn.
    // params.signal is tool_use_context.abort_controller — the single token
    // shared by every tool, every subagent, the subagent gate, compaction,
    // TaskOutput and the Esc key — so cancelling it to unwedge a read also
    // cancelled work that had nothing to do with this request (measured
    // 2026-09-12 13:46:42: four aborted-turn dumps inside 0.7ms) and left the
    // turn indistinguishable from one the user had stopped. A child token
    // propagates cancellation DOWNWARD only: Esc still reaches the stream
    // through the parent, while a stall stops at the stream.
    let stream_signal = params
        .signal
        .as_ref()
        .map(|s| s.child_token())
        .unwrap_or_else(tokio_util::sync::CancellationToken::new);
    let byte_stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>,
    > = Box::pin(response.bytes_stream());
    let mut stream =
        crate::utils::generators::abortable(byte_stream, Some(stream_signal.clone()), None);

    // Stall guards. A request that produces NOTHING is a different failure from
    // one that starts and then goes silent, so they get separate budgets: the
    // first is generous because prefill scales with the uncached prompt, the
    // second is tight because a stream in flight has no reason to pause.
    let first_token_budget = Duration::from_secs_f64(cfg.first_token_timeout_secs.max(0.0));
    let token_gap_budget = Duration::from_secs_f64(cfg.token_gap_timeout_secs.max(0.0));
    loop {
        let started = state.last_output_at.is_some();
        let budget = if started { token_gap_budget } else { first_token_budget };
        let next = if budget.is_zero() {
            stream.next().await
        } else {
            match tokio::time::timeout(budget, stream.next()).await {
                Ok(v) => v,
                Err(_) => {
                    let why = if started {
                        format!(
                            "the model went silent mid-answer: no output for {:.0}s after \
                             {} characters. Generation was cut here.",
                            budget.as_secs_f64(),
                            state.full_content.chars().count()
                                + state.reasoning_text.chars().count()
                        )
                    } else {
                        format!(
                            "the model produced nothing for {:.0}s — the prompt was \
                             accepted but no token ever arrived. Generation was cut here.",
                            budget.as_secs_f64()
                        )
                    };
                    crate::utils::log::log_error(&why, Some("openai_common: stalled stream"));
                    debug_log("STREAM_STALLED", Some(&json!({"reason": why, "started": started})));
                    // To disk under its own name, so a stall is separable from
                    // an Esc in the log directory without opening either file.
                    // The Python original already did this; the port did not.
                    write_cut_dump("stalled", &why, &state, start_time.elapsed().as_secs_f64());
                    // Carried into assemble_content_and_stop_reason: a stall
                    // ends the turn with its own stop_reason and its own line
                    // on screen, never as "[Request interrupted by user]".
                    state.stall_reason = Some(why);
                    // The STREAM only. See the child-token comment above.
                    stream_signal.cancel();
                    aborted = true;
                    break;
                }
            }
        };
        let Some(item) = next else {
            break;
        };
        // Still poll per chunk: a signal set while THIS chunk was in flight
        // must not be processed as if it arrived before it. Reading the CHILD
        // catches both causes — Esc reaches it from the parent.
        if stream_signal.is_cancelled() {
            debug_log(
                "STREAM_ABORTED",
                Some(&json!({"content_length": state.full_content.chars().count()})),
            );
            aborted = true;
            break;
        }
        let chunk_bytes = match item {
            Ok(b) => b,
            Err(e) => {
                // A cancelled read surfaces as a transport error on the way
                // out. During an abort that is the expected outcome, not a
                // failure — the connection was torn down on purpose. Only a
                // stream that broke with no abort pending is a real error.
                if stream_signal.is_cancelled() {
                    debug_log("STREAM_ABORTED", Some(&json!({"via": "reqwest::Error"})));
                    aborted = true;
                    break;
                }
                return Err(format!("{e:?}"));
            }
        };

        let chunk = decoder.decode(&chunk_bytes);
        buffer.push_str(&chunk);
        let mut lines: Vec<String> = buffer.split('\n').map(String::from).collect();
        buffer = lines.pop().unwrap_or_default();

        for line in lines {
            if process_sse_line(&line, &mut state, &mut ctx)? == SseFlow::Done {
                // Every line in THIS batch is still handled: a server that
                // packs its final usage chunk behind [DONE] must not lose it.
                stream_done = true;
            }
        }

        // Degenerate-run guard tripped. This does NOT abort the turn: it takes
        // over the signal the server-side detector used to send, so the Layer-2
        // recovery in query.rs runs exactly as it always did — discard the
        // looping tool calls, nudge the model to do something different, up to
        // REPETITION_RECOVERY_LIMIT times, without the user seeing a stopped
        // turn at all.
        //
        // That ladder went dark for GLM on 2026-09-06, when repetition
        // detection was defaulted off for the family to stop it cutting a live
        // API enumeration. Nothing else can produce stop_reason "repetition",
        // so switching the detector off also switched the RECOVERY off, and a
        // loop that used to be caught at ~140 tokens and quietly nudged instead
        // ran to max_tokens: 63 minutes a request, three recoveries behind it.
        // The guard is the client-side replacement for that signal, and it
        // cannot false-positive the way the n-gram detector did.
        //
        // The collapsed output itself is written to disk and then dropped. Left
        // in history the model reads its own garbage back on every later turn —
        // the same reason stale_recovery_indices prunes spent nudges.
        if let Some(why) = state.degenerate.clone() {
            crate::utils::log::log_error(&why, Some("openai_common: degenerate run"));
            debug_log("DEGENERATE_RUN", Some(&json!({"reason": why})));
            write_cut_dump("degenerate", &why, &state, start_time.elapsed().as_secs_f64());
            state.cut_as_repetition();
            break;
        }

        // The response said it was over. Checked AFTER the degenerate guard, so
        // a collapse spotted in the same batch still wins, and after the whole
        // batch, so nothing queued behind [DONE] is dropped.
        if stream_done {
            debug_log(
                "STREAM_DONE",
                Some(&json!({"content_length": state.full_content.chars().count()})),
            );
            break;
        }
    }

    // abortable() ends the stream when it force-cancels a wedged read, so the
    // in-loop check above never runs in exactly the prefill case this fixes.
    // Re-derive the flag from the signal, or the trailing buffer below would
    // be parsed as a clean finish.
    if !aborted && stream_signal.is_cancelled() {
        debug_log(
            "STREAM_ABORTED",
            Some(&json!({"content_length": state.full_content.chars().count()})),
        );
        aborted = true;
    }

    // (An aborted stream's half-read connection is torn down when the response
    // stream drops here — Rust's drop replaces the Python aclose()-inside-
    // guard dance, and nothing can raise past this function from a drop.)

    // Process a final SSE line the stream left WITHOUT a trailing newline.
    // Some servers send the last include_usage chunk unterminated; dropping it
    // lost the exact token counts. Skip on abort — the stream was cut.
    if !aborted && !buffer.trim().is_empty() {
        let final_line = buffer.clone();
        process_sse_line(&final_line, &mut state, &mut ctx)?;
    }

    // Settle the reasoning count: the throttle above may have skipped the last
    // deltas, and on a SHORT think it may never have fired at all.
    state.recount_thinking(true, ctx.on_thinking_update);
    if state.thinking_tokens > 0 {
        debug_log(
            "THINKING_TOKENS",
            Some(&json!({
                "tokens": state.thinking_tokens,
                "chars": state.reasoning_text.chars().count(),
            })),
        );
    }

    // -rt: emit any held-back tail (never resolved into a marker → real text).
    if let Some(rt) = ctx.rt_filter.as_mut() {
        let tail = rt.flush();
        if !tail.is_empty() {
            if let Some(cb) = ctx.on_content_delta {
                cb(&tail);
            }
        }
    }

    let duration_ms = start_time.elapsed().as_millis() as i64;

    // [DIAGNOSTIC PROBE] DSML content-leak capture. Env-gated, default off.
    // Set CODEMASON_RAW_DUMP=/path/to/file to append the PRE-clean raw content
    // plus finish_reason and the count of STRUCTURED tool_calls the stream
    // parser saw. Records only; changes no behaviour.
    if let Ok(dump_path) = std::env::var("CODEMASON_RAW_DUMP") {
        if !dump_path.is_empty() {
            let rec = json!({
                "ts": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
                "finish_reason": state.final_finish_reason,
                "n_structured_tool_calls": state.tool_calls.len(),
                "raw_content": state.full_content,
            });
            use std::io::Write;
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&dump_path)
                .and_then(|mut f| writeln!(f, "{rec}"));
        }
    }

    // An interrupted turn drops everything it had produced. That is correct for
    // the CONVERSATION — folding a half-finished answer into the history would
    // change the next request's prompt and cost tokens for work the user just
    // stopped — but it also means the evidence goes with it.
    //
    // Measured 2026-09-07: a turn ran 37.7 minutes and produced 36,946 tokens
    // (4,619 of them reasoning) with no text and no tool call on screen, and
    // ESC left one line in the transcript: "[Request interrupted by user]".
    // Whether it was a genuine long derivation or a loop repeating itself could
    // only be argued from the token rate, because not one character survived.
    //
    // So: to disk, always. The history is untouched.
    //
    // This was behind -dbg until 2026-09-10, which put it in the same trap the
    // comment above describes: the flag has to be set BEFORE the run that goes
    // wrong, and nobody launches an ordinary session expecting to need it. That
    // day a turn sat for 2,500s emitting nothing; the flag was not on, so ESC
    // would again have left one transcript line. The evidence came from a core
    // dump of the live process instead, which is not a procedure anyone should
    // need. An abort is a rare, one-shot event — it costs nothing to record
    // unconditionally, exactly like turns.log.
    //
    // A stall is not written here a second time: write_cut_dump already put it
    // in stalled-<ts>.txt with its reason, and two files for one event is
    // what makes the log directory hard to read at a glance.
    if aborted && state.stall_reason.is_none() {
        let dir = crate::utils::config::LOG_DIR.clone();
        // Same two helpers, same reason, as the cut dump above.
        crate::utils::log::ensure_log_dir();
        let name = format!(
            "aborted-{}.txt",
            crate::utils::log::now_iso().replace([':', '.'], "-")
        );
        let body = format!(
            "aborted after {:.1}s
cause: interrupted by the user
reasoning: {} chars
content: {} chars
             
===== REASONING =====
{}

===== CONTENT =====
{}
",
            start_time.elapsed().as_secs_f64(),
            state.reasoning_text.chars().count(),
            state.full_content.chars().count(),
            state.reasoning_text,
            state.full_content,
        );
        if let Err(e) = crate::utils::file::atomic_write_bytes(
            &dir.join(&name).to_string_lossy(),
            body.as_bytes(),
        ) {
            // Must be visible too: a dump that silently failed to write is the
            // same lost evidence as no dump at all.
            crate::utils::log::log_error(
                &format!("could not write the aborted-turn dump: {e}"),
                Some("openai_common: aborted turn"),
            );
        } else {
            crate::utils::log::log_error(
                &format!(
                    "turn interrupted after {:.0}s — {} reasoning + {} content chars saved to {}",
                    start_time.elapsed().as_secs_f64(),
                    state.reasoning_text.chars().count(),
                    state.full_content.chars().count(),
                    dir.join(&name).display()
                ),
                Some("openai_common: aborted turn"),
            );
        }
    }

    let AssembledMessage { content: content_blocks, stop_reason, .. } =
        assemble_content_and_stop_reason(&state, reasoning_roundtrip, aborted);


    // Prompt-token split for BetaUsage below (see the comment there).
    let prompt_tokens = state
        .usage
        .as_ref()
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let cached_tokens = state.cached_prompt_tokens.min(prompt_tokens);

    // Create response message
    let api_message = ApiMessage {
        id: Uuid::new_v4().to_string(),
        message_type: "message".into(),
        role: "assistant".into(),
        content: content_blocks,
        model: cfg.model_name.clone(),
        stop_reason: Some(stop_reason.to_string()),
        stop_sequence: None,
        usage: BetaUsage {
            // Anthropic accounting, which every consumer in this tree assumes:
            // input_tokens EXCLUDES the cached prefix, and the total is
            // input + cache_read + cache_creation + output (utils/tokens.rs,
            // utils/formatting.rs). OpenAI accounting is the opposite —
            // prompt_tokens INCLUDES cached_tokens — so the cached part is
            // SUBTRACTED here, not added. Reporting the raw prompt_tokens next
            // to a non-zero cache_read would double-count the prefix and
            // inflate the context-fill percentage that drives auto-compact: a
            // warm 234k turn would read as 467k and summarise away a
            // conversation that had not filled the window. The sum is
            // unchanged by this split, so anchoring and the default footer read
            // exactly as before; only the previously invisible cache figure is
            // new. Clamped because a server reporting more cached than prompt
            // tokens would drive input_tokens negative.
            input_tokens: (prompt_tokens - cached_tokens).max(0),
            output_tokens: state
                .usage
                .as_ref()
                .and_then(|u| u.get("completion_tokens"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            cache_read_input_tokens: cached_tokens,
            ..Default::default()
        },
    };

    // A collapsed decoder is the only direct evidence of where THIS model stops
    // working, as opposed to where the server stops accepting. Recorded here
    // because this is the one place holding both halves of the measurement —
    // the stop reason and the prompt size that earned it — and recorded before
    // the turn is handed back, so the repetition recovery that query.rs is
    // about to run already sees the lowered threshold on its first retry.
    //
    // Only "repetition" counts. A max_tokens or stalled turn says nothing about
    // the window, and treating either as a collapse would ratchet the ceiling
    // down on evidence that is not about length at all.
    if api_message.stop_reason.as_deref() == Some("repetition") {
        crate::services::compaction::note_decoder_collapse(prompt_tokens);
    }

    // One line per request, always. See utils::log::log_turn for why this is
    // unconditional rather than threshold- or debug-gated.
    //
    // The three numbers a slow turn is actually about, side by side: how much
    // of the prompt the server had to recompute (prefill, straight from vLLM's
    // own counters), how much it thought, and how much it wrote. At the
    // measured 1,335 tok/s prefill and 35 tok/s decode, those three explain
    // essentially all of any wait, and having them written down is what
    // settles "it was prefilling" without anyone having to reconstruct it from
    // timings afterwards.
    {
        let out = api_message.usage.output_tokens;
        let prefill = crate::utils::tokens::pending_prefill();
        let cached_now = crate::utils::tokens::pending_cached();
        let secs = duration_ms as f64 / 1000.0;
        crate::utils::log::log_turn(&json!({
            "prompt": prompt_tokens,
            "prefill": prefill,
            "cached": cached_now,
            "measured": crate::utils::tokens::prefill_is_measured(),
            "think": state.thinking_tokens,
            "out": out,
            "dur_s": format!("{secs:.1}"),
            // What the time was spent on, at the rates measured on this
            // deployment. A turn whose seconds are not covered by these is the
            // interesting one.
            "est_prefill_s": format!("{:.1}", prefill as f64 / 1335.0),
            "est_decode_s": format!("{:.1}", out as f64 / 35.0),
            "stop": api_message.stop_reason.clone().unwrap_or_default(),
            "aborted": aborted,
        }));
    }

    Ok(AssistantMessage {
        message: api_message,
        cost_usd: 0.0,
        duration_ms,
        uuid: Uuid::new_v4().to_string(),
        is_api_error_message: false,
    })
}

/// json.dumps(v, sort_keys=True) equivalent for dedup signatures (serde_json's
/// Map is already sorted; compact form is fine — signatures only compare to
/// each other).
fn sorted_json_dump(v: &Value) -> String {
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_short() {
        assert_eq!(truncate_tool_result("short"), "short");
    }

    #[test]
    fn truncate_long_marks_lines() {
        let long: String = (0..40000).map(|i| if i % 100 == 0 { '\n' } else { 'x' }).collect();
        let out = truncate_tool_result(&long);
        assert!(out.contains("lines truncated"));
        assert!(out.chars().count() < long.chars().count());
    }

    #[test]
    fn clean_strips_special_tokens() {
        assert_eq!(clean_special_tokens("a<|foo|>b", false), "ab");
        // empty think block removed when reasoning not separated
        assert_eq!(clean_special_tokens("x<think>  </think>y", false), "xy");
        // separated reasoning: tags dropped, inner text kept
        assert_eq!(clean_special_tokens("word</think>next", true), "word next");
    }

    #[test]
    fn dsml_recovery_basic() {
        let content = "before <｜DSML｜tool_calls><｜DSML｜invoke name=\"Bash\">\
<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\
</｜DSML｜invoke></｜DSML｜tool_calls> after";
        let (calls, residual) = recover_leaked_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].input["command"], "ls -la");
        assert!(residual.contains("before"));
        assert!(residual.contains("after"));
        assert!(!residual.contains("DSML"));
    }

    #[test]
    fn dsml_recovery_truncated_invoke() {
        let content = "<｜DSML｜tool_calls><｜DSML｜invoke name=\"Read\">\
<｜DSML｜parameter name=\"file_path\" string=\"true\">/tmp/x";
        let (calls, _residual) = recover_leaked_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].input["file_path"], "/tmp/x");
    }

    #[test]
    fn dsml_json_param_parsed() {
        let content = "<｜DSML｜invoke name=\"T\">\
<｜DSML｜parameter name=\"n\" string=\"false\">42</｜DSML｜parameter>\
</｜DSML｜invoke>";
        let (calls, _)= recover_leaked_tool_calls(content);
        assert_eq!(calls[0].input["n"], 42);
    }

    #[test]
    fn json_leak_recovery() {
        let content = r#"I'll run it {"name": "Bash", "arguments": {"command": "echo {nested}"}} done"#;
        let (calls, residual) = recover_leaked_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].input["command"], "echo {nested}");
        assert!(!residual.contains("arguments"));
    }

    #[test]
    fn no_leak_is_noop() {
        let (calls, residual) = recover_leaked_tool_calls("plain answer");
        assert!(calls.is_empty());
        assert_eq!(residual, "plain answer");
    }

    #[test]
    fn incremental_utf8_split_char() {
        let mut d = IncrementalUtf8::new();
        let bytes = "한".as_bytes(); // 3 bytes
        let a = d.decode(&bytes[..1]);
        let b = d.decode(&bytes[1..]);
        assert_eq!(format!("{a}{b}"), "한");
    }

    #[test]
    fn py_json_dumps_matches_python_defaults() {
        let v = json!({"a": 1, "b": "한", "c": [true, null]});
        // Python: json.dumps -> '{"a": 1, "b": "\ud55c", "c": [true, null]}'
        // (default ensure_ascii=True escapes non-ASCII)
        assert_eq!(py_json_dumps(&v), r#"{"a": 1, "b": "\ud55c", "c": [true, null]}"#);
    }

    #[test]
    fn py_json_dumps_astral_surrogates() {
        let v = json!("😀");
        assert_eq!(py_json_dumps(&v), r#""\ud83d\ude00""#);
    }

    #[test]
    fn sse_line_accumulates_tool_call() {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 0,
        };
        let l1 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Bash","arguments":"{\"comm"}}]}}]}"#;
        let l2 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\": \"ls\"}"}}]},"finish_reason":"tool_calls"}]}"#;
        process_sse_line(l1, &mut state, &mut ctx).unwrap();
        process_sse_line(l2, &mut state, &mut ctx).unwrap();
        assert_eq!(state.tool_calls.len(), 1);
        assert_eq!(state.tool_calls[0]["function"]["arguments"], "{\"command\": \"ls\"}");
        assert_eq!(state.final_finish_reason.as_deref(), Some("tool_calls"));
    }

    /// A decoder stuck on one character must be cut, not ridden to
    /// max_tokens. 2026-09-10: 53,859 consecutive '!' cost 63 minutes a
    /// request, four times over.
    ///
    /// One character trips at SINGLE_CHAR_RUN_CHARS rather than at the full
    /// window: measured 2026-09-13, real work never repeats a single character
    /// past 40, and waiting for 2000 cost ~126s per collapse.
    #[test]
    fn a_collapsed_decoder_is_cut() {
        let n = StreamState::SINGLE_CHAR_RUN_CHARS;
        let mut state = StreamState::new();
        state.note_output(&"!".repeat(n - 1), 2000);
        assert!(state.degenerate.is_none(), "under the single-char window: not yet");
        state.note_output("!", 2000);
        assert!(state.degenerate.is_some(), "at the single-char window: must trip");
    }

    /// The silent-reasoning watch RECORDS and never cuts. Volume alone cannot
    /// tell a runaway from a hard question — the largest legitimate pass on
    /// record is 26,271 tokens against the 65,536 two runaways reached — and
    /// throwing away an hour of reasoning on a guess is the worse mistake.
    #[test]
    fn silent_reasoning_is_recorded_but_never_cuts() {
        let mut state = StreamState::new();
        state.thinking_tokens = SILENT_REASONING_TOKENS * 2;
        state.check_silent_reasoning();
        assert!(
            state.degenerate.is_none(),
            "the silent-reasoning watch must never cut a turn"
        );
        assert!(state.silent_reasoning_logged, "past the ceiling: must be recorded");

        // Under the ceiling nothing is recorded.
        let mut quiet = StreamState::new();
        quiet.thinking_tokens = SILENT_REASONING_TOKENS - 1;
        quiet.check_silent_reasoning();
        assert!(!quiet.silent_reasoning_logged, "under the ceiling: nothing to say");

        // A turn that has answered is not remarked on at any volume.
        let mut answered = StreamState::new();
        answered.thinking_tokens = SILENT_REASONING_TOKENS * 4;
        answered.full_content.push_str("the vault address is 0x144a");
        answered.check_silent_reasoning();
        assert!(!answered.silent_reasoning_logged, "content present: not silent");
    }

    /// The measured legitimate maximum must stay under the ceiling: a hard
    /// question at high effort spent 26,271 completion tokens before it
    /// answered, and cutting that would break real work.
    #[test]
    fn the_longest_real_reasoning_pass_stays_under_the_ceiling() {
        assert!(
            26_271 < SILENT_REASONING_TOKENS,
            "ceiling must clear the largest legitimate reasoning pass on record"
        );
    }

    /// The single-character tier must not swallow the two-character case.
    /// Separator rules and indentation legitimately alternate two characters,
    /// so those keep the wide window (measured real maximum: 47).
    #[test]
    fn two_distinct_characters_keep_the_wide_window() {
        let mut state = StreamState::new();
        // 1998 characters, two distinct, well past the single-char window.
        state.note_output(&"-=".repeat(999), 2000);
        assert!(
            state.degenerate.is_none(),
            "two distinct characters must not trip at the single-char window"
        );
        state.note_output("-=", 2000);
        assert!(state.degenerate.is_some(), "at the full window: must trip");
    }

    /// The measured ceiling for real work, on both tiers, must stay quiet.
    #[test]
    fn measured_real_world_runs_do_not_trip() {
        let mut state = StreamState::new();
        state.note_output(&"=".repeat(40), 2000);
        assert!(state.degenerate.is_none(), "40 identical: the measured real maximum");
        let mut state = StreamState::new();
        state.note_output(&"-=".repeat(24), 2000);
        assert!(state.degenerate.is_none(), "48 alternating: past the measured real maximum");
    }

    /// The guard must not fire on real work. Ordinary prose, code and CJK all
    /// carry far more than two distinct characters per two thousand.
    #[test]
    fn real_output_does_not_trip_the_guard() {
        for sample in [
            "fn main() { println!(\"hello\"); } // ordinary code, repeated. ",
            "안녕하세요 반갑습니다. 오늘은 날씨가 좀 춥습니다. ",
            "curl -s https://example.com/a/b/c?x=1 | jq .data[] ",
        ] {
            let mut state = StreamState::new();
            let mut fed = String::new();
            while fed.chars().count() < 8000 {
                fed.push_str(sample);
            }
            state.note_output(&fed, 2000);
            assert!(
                state.degenerate.is_none(),
                "false positive on real output: {sample}"
            );
        }
    }

    /// Both family kwarg names go out together: GLM-5.3 reads `clear_thinking`
    /// in its chat_template.jinja, DeepSeek-V4 reads `drop_thinking` in its own
    /// encoder, and each ignores the other's name.
    #[test]
    fn an_ordinary_turn_clears_prior_thinking_under_both_names() {
        let k = thinking_template_kwargs(true, false).expect("kwargs sent");
        assert_eq!(k["clear_thinking"], serde_json::json!(true));
        assert_eq!(k["drop_thinking"], serde_json::json!(true));
    }

    /// A recovery request keeps the thinking it has to resume from — the nudge
    /// that asks the model to carry on is a user message, and the template
    /// judges "prior turn" by exactly that (traced 2026-09-13).
    #[test]
    fn a_recovery_request_keeps_the_thinking_it_resumes_from() {
        let k = thinking_template_kwargs(true, true).expect("kwargs sent");
        assert_eq!(k["clear_thinking"], serde_json::json!(false));
        assert_eq!(k["drop_thinking"], serde_json::json!(false));
    }

    /// Compaction sets no_reasoning but NOT keep_thinking, so it must still
    /// clear: dragging every earlier turn's reasoning into the one request whose
    /// purpose is to shrink the history is the opposite of what it is for.
    #[test]
    fn compaction_still_clears_prior_thinking() {
        let k = thinking_template_kwargs(true, false).expect("kwargs sent");
        assert_eq!(k["clear_thinking"], serde_json::json!(true));
    }

    /// clear_thinking off in config means no kwargs at all, recovery or not.
    #[test]
    fn the_config_switch_still_turns_the_whole_thing_off() {
        assert!(thinking_template_kwargs(false, false).is_none());
        assert!(thinking_template_kwargs(false, true).is_none());
    }

    /// A collapsed generation must ask for recovery, not end the turn, and it
    /// must leave nothing behind. Before 2026-09-06 the SERVER raised this
    /// signal for GLM and the loop was nudged away invisibly; defaulting the
    /// detector off for the family took the recovery with it.
    #[test]
    fn a_collapsed_generation_asks_for_recovery_and_leaves_nothing() {
        let mut state = StreamState::new();
        state.reasoning_text = "!".repeat(2000);
        state.full_content = "half a sentence".into();
        state.tool_calls = vec![json!({"id": "x"})];

        state.cut_as_repetition();

        assert_eq!(
            state.final_finish_reason.as_deref(),
            Some("repetition"),
            "query.rs keys Layer-2 recovery on exactly this"
        );
        assert!(state.reasoning_text.is_empty(), "garbage must not reach history");
        assert!(state.full_content.is_empty(), "garbage must not reach history");
        assert!(state.tool_calls.is_empty(), "a looping call must not be dispatched");
    }

    /// A two-character cycle is just as stuck as a one-character one, and the
    /// distinct-count test catches it for free.
    #[test]
    fn a_two_character_cycle_also_trips() {
        let mut state = StreamState::new();
        state.note_output(&"ab".repeat(1500), 2000);
        assert!(state.degenerate.is_some());
    }

    #[test]
    fn sse_error_chunk_raises() {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 0,
        };
        let l = r#"data: {"error":{"message":"bad param"}}"#;
        let e = process_sse_line(l, &mut state, &mut ctx).unwrap_err();
        assert!(e.contains("bad param"));
    }
}

#[cfg(test)]
mod lost_tool_call_tests {
    /// A turn that finished cleanly having produced nothing at all is not an
    /// end_turn — it is a turn that never happened, and under -dsp it stops the
    /// run with nothing on screen to say why. Observed 2026-09-07: after a
    /// command returned an empty result the model spent 125 reasoning tokens
    /// and answered with zero characters.
    #[test]
    fn a_finish_with_no_text_and_no_tool_call_is_an_empty_turn() {
        let empty = sse(&[r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#]);
        assert_eq!(
            assemble_content_and_stop_reason(&empty, false, false).stop_reason,
            "empty_turn"
        );

        // A tool call with no prose is normal and must NOT land there.
        let tool_only = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Bash","arguments":"{\"comm"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\": \"ls\"}"}}]},"finish_reason":"stop"}]}"#,
        ]);
        assert_eq!(
            assemble_content_and_stop_reason(&tool_only, false, false).stop_reason,
            "tool_use"
        );

        // Text with no tool call is an ordinary end.
        let text_only = sse(&[r#"data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}"#]);
        assert_eq!(
            assemble_content_and_stop_reason(&text_only, false, false).stop_reason,
            "end_turn"
        );

        // An abort produced nothing either, but the user asked for that.
        assert_eq!(
            assemble_content_and_stop_reason(&empty, false, true).stop_reason,
            "end_turn"
        );
    }

    /// A stream cut mid-tool-call leaves half-written arguments. That is what
    /// pressing ESC looks like from here, and it must not be reported as the
    /// model emitting bad JSON — nor drive query.rs's re-issue policy, which
    /// would repeat the call the user just stopped.
    #[test]
    fn aborted_mid_tool_call_ends_the_turn_instead_of_blaming_the_parser() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"content":"Writing the file now."}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Write","arguments":"{\"file_pa"}}]}}]}"#,
        ]);
        let aborted = assemble_content_and_stop_reason(&state, false, true);
        assert_eq!(aborted.stop_reason, "end_turn", "an abort ends the turn");
        assert!(tool_names(&aborted.content).is_empty());

        // Same bytes, no abort: still a genuine lost call.
        let not_aborted = assemble_content_and_stop_reason(&state, false, false);
        assert_eq!(not_aborted.stop_reason, "tool_parse_failed");
    }

    use super::*;

    fn sse(lines: &[&str]) -> StreamState {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 0,
        };
        for l in lines {
            process_sse_line(l, &mut state, &mut ctx).unwrap();
        }
        state
    }

    fn tool_names(blocks: &[ContentBlock]) -> Vec<String> {
        blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(t) => Some(t.name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The live failure, reproduced: the model announced a tool, the arguments
    /// were cut mid-JSON, and the turn ended looking like a plain answer.
    ///
    /// The assertion is that the message SAYS a call was lost — not merely that
    /// it lacks a tool block, which an ordinary text answer also satisfies.
    #[test]
    fn a_cut_off_tool_call_is_reported_as_lost_not_as_a_finished_turn() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"content":"Writing the PoC now."}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Write","arguments":"{\"file_pa"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let a = assemble_content_and_stop_reason(&state, false, false);
        let (blocks, stop) = (&a.content, a.stop_reason.as_str());
        assert!(tool_names(blocks).is_empty(), "nothing was dispatchable");
        assert_eq!(
            stop, "tool_parse_failed",
            "a turn that lost its only tool call must say so; 'tool_use' claims a \
             dispatch that is not in the content, and 'end_turn' claims the model \
             chose to stop"
        );
    }

    /// 'arguments' arriving as an object rather than a string: as_str() gives
    /// None, so nothing accumulates and the empty string fails to parse. Same
    /// silent drop, different cause — and the same single signal covers it.
    #[test]
    fn a_non_string_arguments_field_is_reported_as_lost() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Bash","arguments":{"command":"ls"}}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let a = assemble_content_and_stop_reason(&state, false, false);
        let (blocks, stop) = (&a.content, a.stop_reason.as_str());
        assert!(tool_names(blocks).is_empty());
        assert_eq!(stop, "tool_parse_failed");
    }

    /// A server that opens its tool_calls at a nonzero index leaves an untouched
    /// filler slot behind it. That slot is not a call the model made, so it must
    /// not trigger recovery — the real call at index 1 dispatches normally.
    #[test]
    fn an_index_filler_slot_is_not_mistaken_for_a_lost_call() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"x2","function":{"name":"Bash","arguments":"{\"command\": \"ls\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        assert_eq!(state.tool_calls.len(), 2, "slot 0 is the filler");
        let a = assemble_content_and_stop_reason(&state, false, false);
        let (blocks, stop) = (&a.content, a.stop_reason.as_str());
        assert_eq!(tool_names(blocks), vec!["Bash".to_string()]);
        assert_eq!(stop, "tool_use");
        assert!(
            a.lost_tool_calls.is_empty(),
            "an untouched filler slot must not be counted as a call the model \
             made — counting it puts a phantom loss in the error log every time \
             this server answers, which is the log the diagnosis depends on: {:?}",
            a.lost_tool_calls
        );
    }

    /// A partial loss keeps what parsed. It deliberately does NOT report
    /// 'tool_parse_failed': query.rs re-asks on that signal, which would run the
    /// surviving call a second time.
    #[test]
    fn a_partial_loss_keeps_the_surviving_call_and_does_not_ask_for_a_retry() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Bash","arguments":"{\"command\": \"ls\"}"}},{"index":1,"id":"x2","function":{"name":"Write","arguments":"{\"file_pa"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let a = assemble_content_and_stop_reason(&state, false, false);
        let (blocks, stop) = (&a.content, a.stop_reason.as_str());
        assert_eq!(tool_names(blocks), vec!["Bash".to_string()]);
        assert_eq!(stop, "tool_use", "re-asking here would re-run Bash");
    }

    /// The ordinary path stays ordinary: no tool calls at all is still end_turn.
    #[test]
    fn a_plain_text_answer_is_still_an_ordinary_end_turn() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"content":"Here is the answer."}}]}"#,
            r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let a = assemble_content_and_stop_reason(&state, false, false);
        let stop = a.stop_reason.as_str();
        assert_eq!(stop, "end_turn");
    }
}

#[cfg(test)]
mod system_prompt_stability_tests {
    use super::*;

    /// The system prompt is the cached prefix. Building it twice from the same
    /// input must give the same bytes — if it reads ANY live state (cwd, clock,
    /// git head), it does not, and every request re-prefills the conversation.
    ///
    /// Asserted as "the same input gives the same output" rather than "cwd is
    /// absent", because the next thing to get this wrong will not be cwd.
    #[test]
    fn the_system_prompt_does_not_read_live_state() {
        let sp = vec!["core prompt".to_string(), "<env>…</env>".to_string()];

        // Ask the state module how many times THIS THREAD read the session
        // directory, and require the build to add none. That states the
        // property directly: a prefix builder must not consult anything that
        // moves.
        //
        // The earlier version proved the same thing by MOVING the tracked
        // directory and checking the bytes did not change. It worked, but it
        // mutated process-global state while the rest of the suite ran in
        // parallel. Measured on Windows: widening its window to 400ms turned
        // path_validation's everyday_git_is_not_gated and read_inside_cwd_allowed
        // red 6 runs out of 6. Those two never call get_cwd themselves — the
        // code they exercise resolves paths against it. At the real width it
        // never fired (0 in 20 runs), so the only thing keeping it green was
        // that the move and the restore sat next to each other. Anything
        // inserted between them — a log line, an extra assertion, a sleep added
        // while debugging — brings it back, reported as "git log should not
        // require approval", which reads like a permission bug and points
        // nowhere near this file.
        //
        // Counting reads needs no mutation, so there is nothing left to race.
        let before = crate::utils::state::cwd_reads();
        let first = build_system_prompt_with_cwd_emphasis(&sp);
        let after = crate::utils::state::cwd_reads();

        assert_eq!(
            after, before,
            "the prefix builder read the session directory {} time(s); whatever \
             it just took from the environment belongs on the user turn, after \
             the history, not in the system prompt",
            after - before
        );

        // And nothing else varies per call either: same input, same bytes.
        let second = build_system_prompt_with_cwd_emphasis(&sp);
        assert_eq!(
            first, second,
            "the cached prefix is not stable across two identical builds"
        );
    }

    /// And it still carries the prompt it was given — a build that returned a
    /// constant would pass the test above while breaking the agent.
    #[test]
    fn the_system_prompt_still_contains_what_it_was_given() {
        let sp = vec!["core prompt".to_string(), "<env>working dir</env>".to_string()];
        let out = build_system_prompt_with_cwd_emphasis(&sp);
        assert!(out.contains("core prompt"));
        assert!(out.contains("<env>working dir</env>"));
    }
}

#[cfg(test)]
mod stale_nudge_prefix_tests {
    use super::*;
    use crate::types::messages::MessageParam;
    use crate::types::query::UserMessage;

    fn user(text: &str, meta: bool) -> Message {
        Message::User(UserMessage {
            uuid: uuid::Uuid::new_v4().to_string(),
            message: MessageParam {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlockParam::text(text)]),
            },
            is_meta: meta,
            tool_use_result: None,
        })
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant(crate::types::query::AssistantMessage {
            uuid: uuid::Uuid::new_v4().to_string(),
            message: ApiMessage {
                id: "m".into(),
                message_type: "message".into(),
                role: "assistant".into(),
                content: vec![ContentBlock::Text(TextBlock { text: text.into() })],
                model: "m".into(),
                stop_reason: Some("end_turn".into()),
                stop_sequence: None,
                usage: BetaUsage::default(),
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    fn wire(msgs: &[Message]) -> Vec<String> {
        convert_to_chat_messages(msgs, &["sys".to_string()], false, 0)
            .iter()
            .map(|m| m.to_value().to_string())
            .collect()
    }

    /// Dropping a stale nudge rewrites the message list — but the question that
    /// matters is WHERE. A cache keeps every whole block before the first
    /// difference, so a removal at the tail costs the tail, and a removal in
    /// the middle costs everything after it.
    ///
    /// This measures the split point directly: how much of the sent history is
    /// still byte-identical once the nudge is gone.
    #[test]
    fn dropping_a_stale_nudge_only_disturbs_the_tail() {
        // A long settled conversation.
        let mut history: Vec<Message> = Vec::new();
        for i in 0..20 {
            history.push(user(&format!("question {i}"), false));
            history.push(assistant(&format!("answer {i}")));
        }

        // The repetition turn: the cut left an empty assistant, then the nudge.
        // The nudge is LAST here, so it is live and kept.
        let mut during = history.clone();
        during.push(assistant(""));
        during.push(user(REPETITION_NUDGE_PREFIX, true));
        let sent_during = wire(&during);

        // Next request: the model answered, so the nudge is no longer last and
        // is dropped along with the empty assistant before it.
        let mut after = during.clone();
        after.push(assistant("recovered answer"));
        let sent_after = wire(&after);

        let common = sent_during
            .iter()
            .zip(sent_after.iter())
            .take_while(|(a, b)| a == b)
            .count();

        assert!(
            sent_after.len() < after.len(),
            "the stale nudge must actually be dropped, or this measures nothing"
        );
        // Asserted as a RULE, not a count. The exact number of disturbed
        // messages is incidental — the Python side merges the recovered answer
        // into the assistant before it and disturbs one where this disturbs two
        // — and pinning it here would make the two ports disagree over
        // something neither of them is actually promising. What both promise is
        // that the disturbance stays at the tail.
        let disturbed = sent_during.len() - common;
        assert!(
            disturbed <= 3,
            "the removal must stay at the tail: {disturbed} messages changed, so \
             the conversation behind them re-prefills"
        );
        assert!(
            common as f64 / sent_during.len() as f64 > 0.9,
            "the surviving prefix must be nearly the whole conversation, not a fraction"
        );
    }
}

/// Regression cover for the 2026-09-12 wedge: a turn whose whole output was
/// tool-call arguments sat for 603.1s and came out as "[Request interrupted by
/// user]", with the tool call it had finished emitting thrown away.
#[cfg(test)]
mod stall_tests {
    use super::*;

    fn sse(lines: &[&str]) -> StreamState {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 0,
        };
        for l in lines {
            process_sse_line(l, &mut state, &mut ctx).unwrap();
        }
        state
    }

    /// The stall guards pick their budget from last_output_at. vLLM's running
    /// completion_tokens is the ONLY progress signal a tool-call-only turn
    /// emits, so a turn that has it must be on the 120s gap budget, not the
    /// 600s first-token one.
    #[test]
    fn the_usage_counter_alone_counts_as_progress() {
        let state = sse(&[r#"data: {"choices":[],"usage":{"completion_tokens":40}}"#]);
        assert_eq!(state.streamed_tokens, 40, "the spinner's number moved");
        assert!(
            state.last_output_at.is_some(),
            "the server counted 40 tokens, so the stall guards must not still \
             be waiting for the first one"
        );
    }

    /// The delta itself is the progress signal, usage counter or not: a
    /// llama.cpp server sends no running completion_tokens, and a tool-call-
    /// only turn there must still leave the first-token budget.
    #[test]
    fn a_tool_call_delta_alone_counts_as_progress() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Bash","arguments":"{\"comm"}}]}}]}"#,
        ]);
        assert!(state.last_output_at.is_some(), "a tool-call delta is output");
    }

    /// A decoder that collapses inside an argument string is the same stuck
    /// decoder as one that collapses in the answer, and must be cut the same.
    #[test]
    fn a_collapse_inside_tool_call_arguments_trips_the_guard() {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 2000,
        };
        let head = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Write","arguments":"{\"content\": \""}}]}}]}"#;
        process_sse_line(head, &mut state, &mut ctx).unwrap();
        let bang = format!(
            r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{}"}}}}]}}}}]}}"#,
            "!".repeat(500)
        );
        for _ in 0..3 {
            process_sse_line(&bang, &mut state, &mut ctx).unwrap();
        }
        assert!(state.degenerate.is_none(), "1,500 of the last 2,000 are '!', the rest is the JSON head: not yet");
        process_sse_line(&bang, &mut state, &mut ctx).unwrap();
        assert!(state.degenerate.is_some(), "2,000 consecutive '!' inside an argument must trip");
    }

    /// A counter that only ever goes backwards is not progress.
    #[test]
    fn a_non_advancing_usage_chunk_is_not_progress() {
        let state = sse(&[r#"data: {"choices":[],"usage":{"completion_tokens":0}}"#]);
        assert!(state.last_output_at.is_none());
    }

    /// [DONE] is the terminator the protocol provides; the read loop needs to
    /// hear it rather than wait for a keep-alive connection to close.
    #[test]
    fn done_ends_the_stream_and_nothing_else_does() {
        let mut state = StreamState::new();
        let mut ctx = SseContext {
            aes_key: None,
            on_token_update: &None,
            on_content_delta: &None,
            on_thinking_update: &None,
            rt_filter: None,
            degenerate_window: 0,
        };
        let ordinary = r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        assert_eq!(
            process_sse_line(ordinary, &mut state, &mut ctx).unwrap(),
            SseFlow::Continue
        );
        assert_eq!(
            process_sse_line("data: [DONE]", &mut state, &mut ctx).unwrap(),
            SseFlow::Done
        );
    }

    /// The headline bug: a server that stopped sending is not a person who
    /// pressed Esc, and the transcript has to say which one happened.
    #[test]
    fn a_stalled_turn_is_not_reported_as_a_user_interrupt() {
        let mut state = sse(&[r#"data: {"choices":[{"delta":{"content":"Writing the report."}}]}"#]);
        state.stall_reason = Some("the model produced nothing for 600s".into());

        let out = assemble_content_and_stop_reason(&state, false, true);
        assert_eq!(out.stop_reason, "stalled", "not end_turn, which reads as Esc");

        let text: String = out
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();
        assert!(
            text.contains("600s"),
            "the cause belongs in the transcript, not only in error.log: {text:?}"
        );
    }

    /// A fully parsed tool call is work the model finished before the stall.
    /// Dropping it is how a turn announces a step and never runs it — here,
    /// the final report that was never written.
    #[test]
    fn a_stall_keeps_a_tool_call_the_model_had_finished() {
        let mut state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Write","arguments":"{\"file_path\": \"r.md\", \"content\": \"done\"}"}}]}}]}"#,
        ]);
        state.stall_reason = Some("the model produced nothing for 600s".into());

        let out = assemble_content_and_stop_reason(&state, false, true);
        assert_eq!(out.stop_reason, "tool_use", "the call must still be dispatched");
        let names: Vec<String> = out
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(t) => Some(t.name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["Write".to_string()]);
    }

    /// The other half of the same rule: Esc must keep behaving exactly as it
    /// did. Reviving a tool the user just stopped is the opposite of the ask.
    #[test]
    fn a_real_user_interrupt_still_ends_the_turn() {
        let state = sse(&[
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x1","function":{"name":"Write","arguments":"{\"file_path\": \"r.md\", \"content\": \"done\"}"}}]}}]}"#,
        ]);
        assert!(state.stall_reason.is_none());
        let out = assemble_content_and_stop_reason(&state, false, true);
        assert_eq!(out.stop_reason, "end_turn");
    }
}

/// Image blocks in convert_to_chat_messages (tests/test_vision_convert.py).
#[cfg(test)]
mod vision_convert_tests {
    use super::*;
    use crate::types::messages::{MessageParam, ToolUseBlock};
    use crate::types::query::UserMessage;

    const B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

    fn img() -> Value {
        json!({"type": "base64", "media_type": "image/png", "data": B64})
    }

    fn user(blocks: Vec<ContentBlockParam>) -> Message {
        Message::User(UserMessage {
            uuid: uuid::Uuid::new_v4().to_string(),
            message: MessageParam { role: "user".into(), content: MessageContent::Blocks(blocks) },
            is_meta: false,
            tool_use_result: None,
        })
    }

    fn assistant_tool(id: &str, name: &str) -> Message {
        assistant(vec![ContentBlock::ToolUse(ToolUseBlock {
            id: id.into(),
            name: name.into(),
            input: json!({}),
        })])
    }

    fn assistant(content: Vec<ContentBlock>) -> Message {
        Message::Assistant(crate::types::query::AssistantMessage {
            uuid: String::new(),
            message: ApiMessage {
                id: "m".into(),
                message_type: "message".into(),
                role: "assistant".into(),
                content,
                model: "m".into(),
                stop_reason: Some("tool_use".into()),
                stop_sequence: None,
                usage: BetaUsage::default(),
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    fn tool_result(id: &str, items: Vec<Value>) -> ContentBlockParam {
        ContentBlockParam::ToolResult {
            tool_use_id: id.into(),
            content: ToolResultContent::Blocks(items),
            is_error: false,
            cache_control: None,
            data: None,
        }
    }

    fn wire(msgs: &[Message], max_images: i64) -> Vec<Value> {
        convert_to_chat_messages(msgs, &[], false, max_images).iter().map(|m| m.to_value()).collect()
    }

    #[test]
    fn text_only_unchanged_by_vision() {
        let conv = vec![
            user(vec![ContentBlockParam::text("hello")]),
            assistant_tool("t1", "Bash"),
            user(vec![tool_result("t1", vec![json!({"type": "text", "text": "a.txt"})])]),
            user(vec![ContentBlockParam::text("next")]),
        ];
        assert_eq!(wire(&conv, 0), wire(&conv, 8));
        assert_eq!(wire(&conv, 8).last().unwrap()["content"], json!("next"));
    }

    #[test]
    fn user_image_part() {
        let conv = vec![user(vec![
            ContentBlockParam::text("what?"),
            ContentBlockParam::Image { source: img(), cache_control: None },
        ])];
        let out = wire(&conv, 8);
        assert_eq!(out.len(), 1, "{out:?}");
        let parts = out[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["text"], "[Image 1]");
        assert_eq!(parts[2]["type"], "image_url");
        assert_eq!(parts[2]["image_url"]["url"], format!("data:image/png;base64,{B64}"));
    }

    #[test]
    fn tool_result_image_stays_in_tool_message() {
        let conv = vec![
            user(vec![ContentBlockParam::text("read it")]),
            assistant_tool("r1", "Read"),
            user(vec![tool_result("r1", vec![
                json!({"type": "text", "text": "Read /x.png"}),
                json!({"type": "image", "source": img()}),
            ])]),
        ];
        let out = wire(&conv, 8);
        // No user message after it: that would blank the loop's reasoning.
        let tool = out.last().unwrap();
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "r1");
        let parts = tool["content"].as_array().unwrap();
        let kinds: Vec<&str> = parts.iter().map(|p| p["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, vec!["text", "text", "image_url"]);
        assert_eq!(parts[0]["text"], "Read /x.png");
        assert_eq!(parts[1]["text"], "[Image 1]");
    }

    #[test]
    fn vision_off_sends_no_payload() {
        let conv = vec![
            user(vec![
                ContentBlockParam::text("see"),
                ContentBlockParam::Image { source: img(), cache_control: None },
            ]),
            assistant_tool("r1", "Read"),
            user(vec![tool_result("r1", vec![json!({"type": "image", "source": img()})])]),
        ];
        let s = serde_json::to_string(&wire(&conv, 0)).unwrap();
        assert!(!s.contains(B64), "{s}");
        assert!(!s.contains("image_url"), "{s}");
        assert!(s.contains("[Image content not supported]"), "{s}");
    }

    #[test]
    fn evict_count() {
        // All seen: limit 8 -> batches of 4; at least 5 stay attached.
        for (total, limit, want) in [(0, 8, 0), (8, 8, 0), (9, 8, 4), (12, 8, 4), (13, 8, 8),
                                     (16, 8, 8), (17, 8, 12), (3, 0, 0), (2, 1, 1), (5, 1, 4)] {
            assert_eq!(images_to_evict(total, limit, total), want, "{total} {limit}");
        }
        // Unseen images go only as far as the overflow demands.
        for (total, seen, want) in [(10, 0, 2), (9, 8, 4), (10, 9, 4), (14, 10, 8), (14, 4, 6), (12, 2, 4)] {
            assert_eq!(images_to_evict(total, 8, seen), want, "{total} {seen}");
        }
    }

    /// Ten images from one parallel Read burst, not answered yet: the request
    /// carries the full 8, only the overflow (2) goes as text.
    #[test]
    fn unseen_burst_keeps_the_limit() {
        let calls = (0..10)
            .map(|i| ContentBlock::ToolUse(ToolUseBlock { id: format!("r{i}"), name: "Read".into(), input: json!({}) }))
            .collect();
        let results = (0..10)
            .map(|i| tool_result(&format!("r{i}"), vec![json!({"type": "image", "source": img()})]))
            .collect();
        let conv = vec![user(vec![ContentBlockParam::text("read all")]), assistant(calls), user(results)];
        let out = wire(&conv, 8);
        let images = out.iter()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter(|p| p["type"] == "image_url").count();
        assert_eq!(images, 8);
    }

    fn image_conv(n: usize) -> Vec<Message> {
        let mut conv = Vec::new();
        for i in 0..n {
            conv.push(user(vec![
                ContentBlockParam::text(&format!("look {}", i + 1)),
                ContentBlockParam::Image { source: img(), cache_control: None },
            ]));
            conv.push(assistant(vec![ContentBlock::text("ok")]));
        }
        conv
    }

    #[test]
    fn oldest_images_go_as_text() {
        let out = wire(&image_conv(12), 8);
        let users: Vec<&Value> = out.iter().filter(|m| m["role"] == "user").collect();
        let images = users.iter()
            .flat_map(|m| m["content"].as_array().unwrap())
            .filter(|p| p["type"] == "image_url").count();
        assert_eq!(images, 8);
        for (i, m) in users.iter().enumerate() {
            let i = i + 1;
            let labels: Vec<&str> = m["content"].as_array().unwrap().iter()
                .filter_map(|p| p["text"].as_str()).filter(|t| t.starts_with("[Image")).collect();
            if i <= 4 {
                assert!(
                    labels[0].starts_with(&format!("[Image {i}: no longer attached — you saw it earlier")),
                    "{labels:?}"
                );
                assert!(labels[0].contains("saved at"), "{labels:?}");
            } else {
                assert_eq!(labels, vec![format!("[Image {i}]")]);
            }
        }
    }

    #[test]
    fn prefix_stable_between_batches() {
        // 9..12 images evict the same 4, so each request starts with the previous one.
        let mut prev: Option<Vec<Value>> = None;
        for n in 9..13 {
            let cur = wire(&image_conv(n), 8);
            if let Some(p) = &prev {
                assert_eq!(&cur[..p.len()], &p[..], "{n}");
            }
            prev = Some(cur);
        }
        // The 13th image is the next batch: the prefix changes once, at image 5.
        let (prev, next) = (prev.unwrap(), wire(&image_conv(13), 8));
        assert_eq!(next[..8], prev[..8]);
        assert_ne!(next[8], prev[8]);
    }

    /// The display count is what the request carries (tests/test_session_status.py).
    #[test]
    fn attached_count_matches_request() {
        let burst = |n: usize| -> Vec<Message> {
            let calls = (0..n)
                .map(|i| ContentBlock::ToolUse(ToolUseBlock { id: format!("r{i}"), name: "Read".into(), input: json!({}) }))
                .collect();
            let results = (0..n)
                .map(|i| tool_result(&format!("r{i}"), vec![json!({"type": "image", "source": img()})]))
                .collect();
            vec![assistant(calls), user(results)]
        };
        for (answered, b) in [(0, 0), (3, 0), (8, 0), (9, 0), (12, 0), (13, 0), (0, 10), (8, 2), (5, 7)] {
            let mut conv = image_conv(answered);
            if b > 0 {
                conv.extend(burst(b));
            }
            let sent = wire(&conv, 8)
                .iter()
                .filter_map(|m| m["content"].as_array())
                .flatten()
                .filter(|p| p["type"] == "image_url")
                .count();
            assert_eq!(attached_image_count(&conv, 8), sent, "{answered} answered + burst {b}");
        }
    }

    /// Same request text as the Python port for the same conversation.
    #[test]
    fn evicted_label_matches_python() {
        let out = wire(&image_conv(9), 8);
        let t = out[0]["content"][1]["text"].as_str().unwrap();
        let prefix = "[Image 1: no longer attached — you saw it earlier; dropped to stay within \
                      the server limit of 8 images per request; saved at ";
        assert!(t.starts_with(prefix), "{t}");
        assert!(t.ends_with(".png (Read it to look again)]"), "{t}");
    }

    /// The burst of ten, answered once and then one more image: images 1-2
    /// were never attached to an answered request, 3-8 were.
    #[test]
    fn unshown_image_says_so() {
        let calls = (0..10)
            .map(|i| ContentBlock::ToolUse(ToolUseBlock { id: format!("r{i}"), name: "Read".into(), input: json!({}) }))
            .collect();
        let results = (0..10)
            .map(|i| tool_result(&format!("r{i}"), vec![json!({"type": "image", "source": img()})]))
            .collect();
        let conv = vec![
            user(vec![ContentBlockParam::text("read all")]),
            assistant(calls),
            user(results),
            assistant_tool("x", "Read"),
            user(vec![tool_result("x", vec![json!({"type": "image", "source": img()})])]),
        ];
        let out = wire(&conv, 8);
        let labels: Vec<String> = out.iter()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter_map(|p| p["text"].as_str())
            .filter(|t| t.starts_with("[Image"))
            .map(String::from)
            .collect();
        assert!(labels[0].starts_with("[Image 1: not attached") && labels[0].contains("you have not seen it"), "{labels:?}");
        assert!(labels[1].starts_with("[Image 2: not attached"), "{labels:?}");
        assert!(labels[2].starts_with("[Image 3: no longer attached — you saw it earlier"), "{labels:?}");
        assert!(labels[3].starts_with("[Image 4: no longer attached — you saw it earlier"), "{labels:?}");
        assert_eq!(labels[4], "[Image 5]");
    }

    fn labels(chat: &[Value]) -> Vec<String> {
        chat.iter()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter_map(|p| p["text"].as_str())
            .filter(|t| t.starts_with("[Image"))
            .map(String::from)
            .collect()
    }

    fn img_item() -> Value {
        json!({"type": "image", "source": img()})
    }

    /// tests/test_vision_followups.py test_synthetic_answer_not_seen.
    #[test]
    fn api_error_does_not_mark_images_seen() {
        let calls: Vec<ContentBlock> = (0..10)
            .map(|i| ContentBlock::ToolUse(ToolUseBlock { id: format!("r{i}"), name: "Read".into(), input: json!({}) }))
            .collect();
        let results: Vec<ContentBlockParam> = (0..10).map(|i| tool_result(&format!("r{i}"), vec![img_item()])).collect();
        let mut error = assistant(vec![ContentBlock::Text(TextBlock { text: "Error connecting".into() })]);
        if let Message::Assistant(a) = &mut error {
            a.message.model = "<synthetic>".into();
            a.is_api_error_message = true;
        }
        let conv = vec![
            user(vec![ContentBlockParam::text("read all")]),
            assistant(calls),
            user(results),
            error,
            user(vec![ContentBlockParam::text("again"), ContentBlockParam::Image { source: img(), cache_control: None }]),
        ];
        let l = labels(&wire(&conv, 8));
        assert!(!l.iter().any(|t| t.contains("you saw it earlier")), "{l:?}");
    }

    #[test]
    fn image_beside_tool_results_is_sent_and_counted() {
        let conv = vec![
            user(vec![ContentBlockParam::text("x")]),
            assistant_tool("t", "Bash"),
            user(vec![
                ContentBlockParam::ToolResult {
                    tool_use_id: "t".into(),
                    content: ToolResultContent::Text("ok".into()),
                    is_error: false,
                    cache_control: None,
                    data: None,
                },
                ContentBlockParam::text("and this"),
                ContentBlockParam::Image { source: img(), cache_control: None },
            ]),
        ];
        let chat = wire(&conv, 8);
        let last = chat.last().unwrap();
        assert_eq!(last["role"], "user");
        assert!(last["content"].as_array().unwrap().iter().any(|p| p["type"] == "image_url"), "{last}");
        assert_eq!(attached_image_count(&conv, 8), 1);
    }

    #[test]
    fn count_skips_a_duplicate_result() {
        let conv = vec![
            user(vec![ContentBlockParam::text("x")]),
            assistant_tool("t", "Read"),
            user(vec![tool_result("t", vec![img_item()])]),
            user(vec![tool_result("t", vec![img_item()])]),
        ];
        assert_eq!(attached_image_count(&conv, 8), 1);
    }
}
