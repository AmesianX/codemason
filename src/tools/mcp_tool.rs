//! Wrap tools discovered from MCP servers as CodeMason `Tool`s.
//! Equivalent to codemason/tools/mcp_tool.py.
//!
//! `build_mcp_tools()` connects to every configured MCP server (see
//! utils/mcp_client), lists their tools, and returns one dynamically-created
//! `Tool` per remote tool. tools::get_all_tools() appends these to the native
//! tool set, so from the model's point of view an MCP tool is just another
//! tool — named `mcp__<server>__<tool>` (the same convention Claude Code
//! uses), described by the server, its parameters taken verbatim from the
//! server's input schema, and its call forwarded over the live session.
//!
//! Substrate divergence (documented): the Python fabricated a class per remote
//! tool; Rust uses one McpTool struct per remote tool, with the tool name
//! interned via Box::leak (the Tool trait's `name()` returns &'static str) —
//! bounded, one leak per discovered tool per process, the same lifetime the
//! Python classes had. Divergence 2: utils::mcp_client's McpToolInfo does not
//! surface MCP `annotations.readOnlyHint`, so every MCP tool reports
//! is_read_only() = false — the Python's default when annotations are absent;
//! a server that DOES hint read-only now asks for permission instead of
//! auto-allowing, which fails closed (safe direction).

use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::sync::Arc;

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::ToolUseContext;
use crate::utils::log::log_error;
use crate::utils::mcp_client::{self, McpToolInfo};

/// Python discover(connect_timeout=20.0) / call_tool(timeout=120.0) defaults.
const CONNECT_TIMEOUT: f64 = 20.0;
const CALL_TIMEOUT: f64 = 120.0;

fn tool_schema(mcp_tool: &McpToolInfo) -> Value {
    let schema = mcp_tool.input_schema.clone();
    let mut schema = if schema.is_object() && !schema.as_object().map(|o| o.is_empty()).unwrap_or(true)
    {
        schema
    } else {
        serde_json::json!({"type": "object", "properties": {}})
    };
    if let Some(obj) = schema.as_object_mut() {
        // setdefault("type", "object") / setdefault("properties", {})
        obj.entry("type").or_insert_with(|| Value::String("object".into()));
        obj.entry("properties").or_insert_with(|| serde_json::json!({}));
    }
    schema
}

/// Flatten an MCP CallToolResult into text for the assistant.
fn render_result(result: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    for block in result.get("content").and_then(|c| c.as_array()).map(|a| a.as_slice()).unwrap_or(&[])
    {
        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
            parts.push(text.to_string());
            continue;
        }
        match block.get("type").and_then(|t| t.as_str()) {
            Some("image") => {
                let mime = block
                    .get("mimeType")
                    .or_else(|| block.get("mime_type"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("");
                if mime.is_empty() {
                    parts.push("[image content omitted]".to_string());
                } else {
                    parts.push(format!("[image content {mime} omitted]"));
                }
            }
            Some("resource") => {
                let resource = block.get("resource").map(|r| r.to_string()).unwrap_or_default();
                parts.push(format!("[resource: {resource}]"));
            }
            _ => parts.push(block.to_string()),
        }
    }
    let out = parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n");
    let is_error = result
        .get("is_error")
        .or_else(|| result.get("isError"))
        .and_then(|e| e.as_bool())
        .unwrap_or(false);
    if is_error {
        if out.is_empty() {
            "MCP tool returned an error.".to_string()
        } else {
            format!("MCP tool error:\n{out}")
        }
    } else if out.is_empty() {
        "(no output)".to_string()
    } else {
        out
    }
}

/// One wrapped remote tool (the Python's dynamically-created class).
pub struct McpTool {
    /// `mcp__<server>__<tool>` — interned so name() can hand out &'static str.
    full_name: &'static str,
    description: String,
    schema: Value,
    /// kept for diagnostics / the `mcp` CLI
    pub mcp_server: String,
    pub mcp_tool_name: String,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &'static str {
        self.full_name
    }

    fn input_json_schema(&self) -> Value {
        self.schema.clone()
    }

    async fn description(&self, _input: &Value) -> String {
        self.description.clone()
    }

    fn is_read_only(&self) -> bool {
        // See module doc: readOnlyHint is not surfaced by mcp_client, so this
        // is the Python's annotations-absent default (false) for every tool.
        false
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        format!("{}:{}", self.mcp_server, self.mcp_tool_name)
    }

    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        let server = self.mcp_server.clone();
        let tool_name = self.mcp_tool_name.clone();
        let full_name = self.full_name;
        Box::pin(async_stream::stream! {
            let args = if input.is_object() { input } else { serde_json::json!({}) };
            match mcp_client::call_tool(&server, &tool_name, args, CALL_TIMEOUT).await {
                Ok(result) => {
                    yield Ok(ToolResult::Result {
                        result_for_assistant: render_result(&result),
                        data: Some(serde_json::json!({"server": server, "tool": tool_name})),
                    });
                }
                Err(exc) => {
                    let msg = format!("MCP call to {full_name} failed: {exc}");
                    log_error(&msg, None);
                    yield Ok(ToolResult::Result {
                        result_for_assistant: msg,
                        data: Some(serde_json::json!({"error": exc})),
                    });
                }
            }
        })
    }
}

fn make_tool(server: &str, mcp_tool: &McpToolInfo) -> Arc<dyn Tool> {
    let full_name: &'static str =
        Box::leak(format!("mcp__{server}__{}", mcp_tool.name).into_boxed_str());
    let desc = if mcp_tool.description.trim().is_empty() {
        format!("MCP tool {} from {server}", mcp_tool.name)
    } else {
        mcp_tool.description.trim().to_string()
    };
    Arc::new(McpTool {
        full_name,
        description: desc,
        schema: tool_schema(mcp_tool),
        mcp_server: server.to_string(),
        mcp_tool_name: mcp_tool.name.clone(),
    })
}

static CACHE: Lazy<Mutex<Option<Vec<Arc<dyn Tool>>>>> = Lazy::new(|| Mutex::new(None));

/// Discover MCP tools once and return their wrappers (cached).
///
/// Sync API, matching the Python (whose mcp_client bridged its own asyncio
/// loop behind a sync discover()); the registry (tools::get_all_tools) calls
/// this without awaiting. The async discovery runs on a dedicated scoped
/// thread with its own runtime so this is callable from any context —
/// including inside a current-thread tokio runtime, where block_in_place
/// would panic. Discovery happens once per process (cached), so the thread
/// spawn cost is paid once.
pub fn build_mcp_tools_forced(force: bool) -> Vec<Arc<dyn Tool>> {
    if !force {
        if let Some(cached) = CACHE.lock().clone() {
            return cached;
        }
    }
    let discovered: Vec<(String, McpToolInfo)> = std::thread::scope(|s| {
        s.spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map(|rt| rt.block_on(mcp_client::discover(CONNECT_TIMEOUT)))
                .unwrap_or_default()
        })
        .join()
        .unwrap_or_default()
    });
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    // Python wraps discovery in try/except and logs; the Rust discover()
    // already logs-and-skips per-server failures internally.
    for (server, mcp_tool) in discovered {
        tools.push(make_tool(&server, &mcp_tool));
    }
    *CACHE.lock() = Some(tools.clone());
    tools
}

/// build_mcp_tools(force=False) — the registry's entry point.
pub fn build_mcp_tools() -> Vec<Arc<dyn Tool>> {
    build_mcp_tools_forced(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_defaults_filled() {
        let info = McpToolInfo {
            name: "t".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"properties": {"a": {"type": "string"}}}),
        };
        let s = tool_schema(&info);
        assert_eq!(s["type"], "object");
        assert!(s["properties"]["a"].is_object());
        // empty/non-object schema → bare object schema
        let empty = McpToolInfo { name: "t".into(), description: String::new(), input_schema: Value::Null };
        let s2 = tool_schema(&empty);
        assert_eq!(s2, serde_json::json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn result_flattening() {
        let r = serde_json::json!({"content": [
            {"type": "text", "text": "hello"},
            {"type": "image", "mimeType": "image/png"},
            {"type": "resource", "resource": {"uri": "x"}},
        ]});
        let out = render_result(&r);
        assert!(out.starts_with("hello\n[image content image/png omitted]\n[resource: "));
        let err = serde_json::json!({"content": [{"type": "text", "text": "boom"}], "isError": true});
        assert_eq!(render_result(&err), "MCP tool error:\nboom");
        assert_eq!(render_result(&serde_json::json!({"content": []})), "(no output)");
    }

    #[test]
    fn naming_convention() {
        let info = McpToolInfo { name: "search".into(), description: "d".into(), input_schema: Value::Null };
        let t = make_tool("myserver", &info);
        assert_eq!(t.name(), "mcp__myserver__search");
        assert!(t.matches_name("mcp__myserver__search"));
        assert_eq!(t.user_facing_name(None), "myserver:search");
    }
}
