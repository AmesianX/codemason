//! Message type definitions for CodeMason
//! Equivalent to codemason/types/messages.py (src/types/messages.ts)

use serde::{Deserialize, Serialize};
use serde_json::Value;

// === Cache Control ===

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String, // 'ephemeral'
}

impl Default for CacheControl {
    fn default() -> Self {
        CacheControl { cache_type: "ephemeral".into() }
    }
}

// === Block Param Types (for sending to API) ===
//
// The Python dataclasses' to_dict() omitted absent optional fields; the serde
// skip_serializing_if attributes reproduce exactly that wire shape.

fn is_false(b: &bool) -> bool {
    !*b
}

/// tool_result content: a plain string or a list of text/image blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<Value>),
}

impl Default for ToolResultContent {
    fn default() -> Self {
        ToolResultContent::Text(String::new())
    }
}

impl ToolResultContent {
    /// Flatten to the text the model reads (list blocks joined by newline,
    /// matching check_permissions_and_call_tool's list handling).
    pub fn as_text(&self) -> String {
        match self {
            ToolResultContent::Text(s) => s.clone(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| {
                    b.get("text")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| b.to_string())
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: ToolResultContent,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// The tool's structured Output, carried alongside the model-facing
        /// content (the original's message.toolUseResult). The display layer
        /// renders FROM this via Tool::render_tool_result_message; the API
        /// serializers only read type/content/tool_use_id, so it never reaches
        /// the model.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    Document {
        source: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

impl ContentBlockParam {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlockParam::Text { text: text.into(), cache_control: None }
    }

    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        ContentBlockParam::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: ToolResultContent::Text(content.into()),
            is_error,
            cache_control: None,
            data: None,
        }
    }

    pub fn block_type(&self) -> &'static str {
        match self {
            ContentBlockParam::Text { .. } => "text",
            ContentBlockParam::Image { .. } => "image",
            ContentBlockParam::ToolUse { .. } => "tool_use",
            ContentBlockParam::ToolResult { .. } => "tool_result",
            ContentBlockParam::Document { .. } => "document",
            ContentBlockParam::Thinking { .. } => "thinking",
        }
    }
}

// === Block Types (received from API) ===

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TextBlock {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ThinkingBlock {
    pub thinking: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    Thinking(ThinkingBlock),
}

impl ContentBlock {
    pub fn block_type(&self) -> &'static str {
        match self {
            ContentBlock::Text(_) => "text",
            ContentBlock::ToolUse(_) => "tool_use",
            ContentBlock::Thinking(_) => "thinking",
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text(TextBlock { text: text.into() })
    }
}

/// Parse a content block from API response; unknown types default to a text
/// block carrying the raw payload (parse_content_block).
pub fn parse_content_block(data: &Value) -> ContentBlock {
    match data.get("type").and_then(|t| t.as_str()) {
        Some("text") => ContentBlock::Text(TextBlock {
            text: data.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
        }),
        Some("tool_use") => ContentBlock::ToolUse(ToolUseBlock {
            id: data.get("id").and_then(|t| t.as_str()).unwrap_or("").to_string(),
            name: data.get("name").and_then(|t| t.as_str()).unwrap_or("").to_string(),
            input: data.get("input").cloned().unwrap_or_else(|| Value::Object(Default::default())),
        }),
        Some("thinking") => ContentBlock::Thinking(ThinkingBlock {
            thinking: data.get("thinking").and_then(|t| t.as_str()).unwrap_or("").to_string(),
        }),
        _ => ContentBlock::Text(TextBlock { text: data.to_string() }),
    }
}

// === Message Types ===

/// Message content — a plain string, or a list of blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlockParam>),
}

impl MessageContent {
    pub fn as_blocks(&self) -> Vec<ContentBlockParam> {
        match self {
            MessageContent::Text(s) => vec![ContentBlockParam::text(s.clone())],
            MessageContent::Blocks(b) => b.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageParam {
    pub role: String, // 'user' | 'assistant' | 'system'
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct BetaUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub cache_creation_input_tokens: i64,
}

/// Message type (API response). Alias in Python: APIMessage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ApiMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String, // 'message'
    pub role: String, // 'assistant'
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: BetaUsage,
}

impl Default for ApiMessage {
    fn default() -> Self {
        ApiMessage {
            id: String::new(),
            message_type: "message".into(),
            role: "assistant".into(),
            content: Vec::new(),
            model: String::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: BetaUsage::default(),
        }
    }
}

// === Error Classes ===

/// Base API Error (APIError / APIConnectionError).
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{message}")]
    Api {
        message: String,
        status: Option<u16>,
        headers: std::collections::HashMap<String, String>,
    },
    #[error("{0}")]
    Connection(String),
}

impl ApiError {
    pub fn api(message: impl Into<String>, status: Option<u16>) -> Self {
        ApiError::Api { message: message.into(), status, headers: Default::default() }
    }
}
