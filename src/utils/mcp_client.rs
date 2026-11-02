//! Minimal MCP (Model Context Protocol) client for CodeMason.
//! Equivalent to codemason/utils/mcp_client.py.
//!
//! CodeMason mirrors Claude Code's *native* tool set (see tools/mod.rs) and had
//! no way to talk to external MCP servers. This module adds one: it reads the
//! same ``mcpServers`` configuration Claude Code uses, connects to each server
//! over stdio / Streamable-HTTP, discovers its tools, and exposes them so
//! tools/mcp_tool.rs can wrap each as a CodeMason ``Tool``.
//!
//! SUBSTRATE (Python → Rust divergences, behaviour-preserving):
//!
//!   * The Python build delegated the protocol to the ``mcp`` SDK and parked
//!     every session on a dedicated background asyncio loop (the SDK's anyio
//!     task groups demanded same-task entry/exit). Rust has no SDK dependency:
//!     the JSON-RPC 2.0 protocol (initialize handshake, notifications/
//!     initialized, tools/list, tools/call) is implemented here directly, and
//!     tokio tasks own each connection — the loop-isolation machinery has no
//!     equivalent problem to solve and is not reproduced.
//!   * Transports: stdio (newline-delimited JSON-RPC over the child's pipes)
//!     and Streamable-HTTP (POST per message; JSON or SSE response bodies;
//!     Mcp-Session-Id header) are implemented. Legacy SSE is NOT — a config
//!     asking for it gets the same shape the Python produced when its SDK
//!     lacked the transport: the server is skipped with
//!     "sse transport unavailable" logged.
//!
//! Discovery and per-server connection failures are non-fatal: a broken server
//! is logged and skipped so it can never block CodeMason from starting.
//!
//! Config sources (later wins on name collision), each an object whose
//! ``mcpServers`` maps a server name to its entry:
//!
//!     ~/.codemason/mcp.json           →  {"mcpServers": {...}}   (Claude Code .mcp.json shape)
//!     ~/.codemason/settings.json      →  {"mcpServers": {...}}
//!     <cwd>/.mcp.json                 →  {"mcpServers": {...}}   (project-scoped)
//!
//! Server entry shapes (Claude Code compatible):
//!
//!     stdio:  {"command": "npx", "args": ["-y", "pkg"], "env": {"K": "v"}}
//!     http :  {"type": "http", "url": "https://host/mcp", "headers": {"Authorization": "Bearer .."}}
//!     sse  :  {"type": "sse",  "url": "https://host/sse", "headers": {...}}
//!
//! Transport is inferred when ``type`` is absent: ``command`` → stdio, ``url``
//! → http. ``"disabled": true`` skips an entry.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::utils::config::{CONFIG_DIR, SETTINGS_FILE};
use crate::utils::log::log_error;

/// MCP protocol revision this client speaks in the initialize handshake.
const PROTOCOL_VERSION: &str = "2025-03-26";

/// A discovered tool, mirroring the SDK Tool fields mcp_tool reads
/// (name / description / inputSchema).
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ── transports ──────────────────────────────────────────────────────────────

enum Transport {
    Stdio {
        /// Held so the child dies with us; the writer half feeds requests.
        child: tokio::process::Child,
        stdin: tokio::process::ChildStdin,
        pending: Arc<Mutex<HashMap<i64, tokio::sync::oneshot::Sender<Value>>>>,
    },
    Http {
        url: String,
        headers: Vec<(String, String)>,
        client: reqwest::Client,
        session_id: Mutex<Option<String>>,
    },
}

/// A live MCP session plus its discovered tools (the Python _Server).
pub struct Server {
    pub name: String,
    pub config: Value,
    transport: tokio::sync::Mutex<Option<Transport>>,
    next_id: AtomicI64,
    pub tools: Mutex<Vec<McpToolInfo>>,
    pub error: Mutex<Option<String>>,
    pub connected: std::sync::atomic::AtomicBool,
}

fn transport_type(cfg: &Value) -> String {
    cfg.get("type")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            if cfg.get("command").and_then(|c| c.as_str()).is_some() {
                "stdio".to_string()
            } else {
                "http".to_string()
            }
        })
        .to_lowercase()
}

impl Server {
    fn new(name: &str, config: Value) -> Self {
        Server {
            name: name.to_string(),
            config,
            transport: tokio::sync::Mutex::new(None),
            next_id: AtomicI64::new(1),
            tools: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Open the transport, run the initialize handshake, list tools.
    async fn connect(&self) -> Result<(), String> {
        let cfg = &self.config;
        let ttype = transport_type(cfg);
        let transport = match ttype.as_str() {
            "stdio" => self.open_stdio(cfg)?,
            "http" | "streamable-http" | "streamable_http" => {
                let url = cfg
                    .get("url")
                    .and_then(|u| u.as_str())
                    .ok_or_else(|| "http transport requires 'url'".to_string())?
                    .to_string();
                let headers: Vec<(String, String)> = cfg
                    .get("headers")
                    .and_then(|h| h.as_object())
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| {
                                (k.clone(), v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let client = reqwest::Client::builder()
                    .build()
                    .map_err(|e| format!("http client: {e}"))?;
                Transport::Http { url, headers, client, session_id: Mutex::new(None) }
            }
            "sse" => return Err("sse transport unavailable".to_string()),
            other => return Err(format!("unknown MCP transport type: '{other}'")),
        };
        *self.transport.lock().await = Some(transport);

        // initialize handshake, then notifications/initialized.
        let init = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "codemason", "version": crate::cli::VERSION},
                }),
                Duration::from_secs(30),
            )
            .await?;
        let _ = init; // server capabilities not consumed (matches Python usage)
        self.notify("notifications/initialized", json!({})).await?;

        // Discover tools.
        let result = self
            .request("tools/list", json!({}), Duration::from_secs(30))
            .await?;
        let tools: Vec<McpToolInfo> = result
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        Some(McpToolInfo {
                            name: t.get("name")?.as_str()?.to_string(),
                            description: t
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string(),
                            input_schema: t
                                .get("inputSchema")
                                .cloned()
                                .unwrap_or_else(|| json!({"type": "object"})),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        *self.tools.lock() = tools;
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn open_stdio(&self, cfg: &Value) -> Result<Transport, String> {
        let command = cfg
            .get("command")
            .and_then(|c| c.as_str())
            .ok_or_else(|| "stdio transport requires 'command'".to_string())?;
        let args: Vec<String> = cfg
            .get("args")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let mut cmd = tokio::process::Command::new(command);
        cmd.args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        // Inherit the environment MINUS this program's own secrets. An MCP
        // server is third-party code; passing it the model API key, the body
        // AES key and the bot token gives every one of them the ability to
        // impersonate the user to the model server, decrypt captured traffic,
        // or take over the remote channel. Nothing in the MCP protocol needs
        // them, and an entry that genuinely wants a credential still declares
        // it in its own `env` below.
        cmd.env_clear();
        for (k, v) in std::env::vars() {
            let secret = k.ends_with("_API_KEY")
                || k.ends_with("_AES_KEY")
                || k == "TELEGRAM_BOT_TOKEN"
                || ((k.starts_with("CODEMASON_")
                    || k.starts_with("LLAMACPP_")
                    || k.starts_with("TELEGRAM_"))
                    && (k.ends_with("_TOKEN") || k.ends_with("_SECRET") || k.ends_with("_PASSWORD")));
            if !secret {
                cmd.env(k, v);
            }
        }
        // Overlay the config's own env last, so an entry can still pass one.
        for (k, v) in cfg
            .get("env")
            .and_then(|e| e.as_object())
            .map(|m| m.iter().collect::<Vec<_>>())
            .unwrap_or_default()
        {
            cmd.env(k, v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()));
        }
        if let Some(cwd) = cfg.get("cwd").and_then(|c| c.as_str()) {
            if !cwd.is_empty() {
                cmd.current_dir(cwd);
            }
        }

        let mut child = cmd.spawn().map_err(|e| format!("spawn {command}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin pipe")?;
        let stdout = child.stdout.take().ok_or("no stdout pipe")?;

        let pending: Arc<Mutex<HashMap<i64, tokio::sync::oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Reader task: newline-delimited JSON-RPC; route responses by id.
        // Server-initiated requests/notifications are ignored (this client
        // advertises no capabilities that would invite them).
        let pending_reader = pending.clone();
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                if let Some(id) = msg.get("id").and_then(|i| i.as_i64()) {
                    if msg.get("result").is_some() || msg.get("error").is_some() {
                        if let Some(tx) = pending_reader.lock().remove(&id) {
                            let _ = tx.send(msg);
                        }
                    }
                }
            }
        });

        Ok(Transport::Stdio { child, stdin, pending })
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        let mut guard = self.transport.lock().await;
        match guard.as_mut() {
            Some(Transport::Stdio { stdin, .. }) => {
                use tokio::io::AsyncWriteExt;
                let line = format!("{msg}\n");
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| format!("stdio write: {e}"))?;
                stdin.flush().await.map_err(|e| format!("stdio flush: {e}"))?;
                Ok(())
            }
            Some(Transport::Http { url, headers, client, session_id }) => {
                let mut req = client
                    .post(url.as_str())
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream");
                for (k, v) in headers.iter() {
                    req = req.header(k.as_str(), v.as_str());
                }
                if let Some(sid) = session_id.lock().clone() {
                    req = req.header("Mcp-Session-Id", sid);
                }
                let _ = req.json(&msg).send().await; // 202 Accepted; body ignored
                Ok(())
            }
            None => Err("transport not open".to_string()),
        }
    }

    async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});

        let mut guard = self.transport.lock().await;
        let response: Value = match guard.as_mut() {
            Some(Transport::Stdio { stdin, pending, .. }) => {
                let (tx, rx) = tokio::sync::oneshot::channel();
                pending.lock().insert(id, tx);
                {
                    use tokio::io::AsyncWriteExt;
                    let line = format!("{msg}\n");
                    stdin
                        .write_all(line.as_bytes())
                        .await
                        .map_err(|e| format!("stdio write: {e}"))?;
                    stdin.flush().await.map_err(|e| format!("stdio flush: {e}"))?;
                }
                // Release the transport lock while waiting so concurrent calls
                // interleave (the reader task fills the oneshot).
                drop(guard);
                match tokio::time::timeout(timeout, rx).await {
                    Ok(Ok(v)) => v,
                    Ok(Err(_)) => return Err("server closed the connection".to_string()),
                    Err(_) => {
                        return Err(format!("request timed out after {:.0}s", timeout.as_secs_f64()))
                    }
                }
            }
            Some(Transport::Http { url, headers, client, session_id }) => {
                let mut req = client
                    .post(url.as_str())
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream")
                    .timeout(timeout);
                for (k, v) in headers.iter() {
                    req = req.header(k.as_str(), v.as_str());
                }
                if let Some(sid) = session_id.lock().clone() {
                    req = req.header("Mcp-Session-Id", sid);
                }
                let resp = req
                    .json(&msg)
                    .send()
                    .await
                    .map_err(|e| format!("http request: {e}"))?;
                if let Some(sid) = resp
                    .headers()
                    .get("mcp-session-id")
                    .and_then(|v| v.to_str().ok())
                {
                    *session_id.lock() = Some(sid.to_string());
                }
                let status = resp.status();
                if !status.is_success() {
                    return Err(format!("http status {status}"));
                }
                let content_type = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let body = resp.text().await.map_err(|e| format!("http body: {e}"))?;
                if content_type.contains("text/event-stream") {
                    // Take the SSE message whose id matches ours.
                    parse_sse_response(&body, id)
                        .ok_or_else(|| "no matching SSE response".to_string())?
                } else {
                    serde_json::from_str(&body).map_err(|e| format!("bad JSON response: {e}"))?
                }
            }
            None => return Err("transport not open".to_string()),
        };

        if let Some(err) = response.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
            return Err(format!("server error {code}: {msg}"));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Invoke a tool (session.call_tool). Returns the raw CallToolResult value
    /// ({"content": [...], "isError": bool, ...}).
    pub async fn call(&self, tool_name: &str, arguments: Value, timeout: Duration) -> Result<Value, String> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(format!("MCP server '{}' is not connected", self.name));
        }
        let args = if arguments.is_object() { arguments } else { json!({}) };
        self.request(
            "tools/call",
            json!({"name": tool_name, "arguments": args}),
            timeout,
        )
        .await
    }
}

/// name -> Server registry.
static SERVERS: Lazy<Mutex<HashMap<String, Arc<Server>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// ── config loading ──────────────────────────────────────────────────────────

fn read_json(path: &Path) -> Value {
    if path.is_file() {
        match std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string()))
        {
            Ok(v) => return v,
            Err(exc) => {
                log_error(&format!("MCP: failed to read {}: {exc}", path.display()), None);
            }
        }
    }
    json!({})
}

/// Merge ``mcpServers`` from all config sources (later source wins).
/// Merge `mcpServers` from all config sources (later source wins).
///
/// The project-local `.mcp.json` is the exception to "later wins", and is
/// opt-in. discover() runs at tool-registry construction — before the user has
/// typed anything — and spawns each entry's `command`, so reading that file
/// from whatever directory codemason started in made `cd` into a cloned repo
/// enough to execute code from it. It also sat at the HIGHEST precedence, so a
/// repo could reuse a server name from the user's own config and silently
/// re-point it, sending the user's bearer token to its own host.
pub fn load_server_configs() -> HashMap<String, Value> {
    let mut merged: HashMap<String, Value> = HashMap::new();
    let sources: Vec<PathBuf> = vec![CONFIG_DIR.join("mcp.json"), SETTINGS_FILE.clone()];
    for src_path in sources {
        let src = read_json(&src_path);
        let servers = src
            .get("mcpServers")
            .or_else(|| src.get("mcp_servers"))
            .and_then(|s| s.as_object());
        if let Some(servers) = servers {
            for (name, entry) in servers {
                if entry.is_object() {
                    merged.insert(name.clone(), entry.clone());
                }
            }
        }
    }

    let project_file =
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(".mcp.json");
    if !project_file.is_file() {
        return merged;
    }
    let trusted = std::env::var("CODEMASON_TRUST_PROJECT_MCP")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if !trusted {
        crate::cmprintln!(
            "  Skipped {} — a project MCP config runs its commands at startup. \
             Set CODEMASON_TRUST_PROJECT_MCP=1 to load it.",
            project_file.display()
        );
        return merged;
    }
    let project = read_json(&project_file);
    if let Some(servers) =
        project.get("mcpServers").or_else(|| project.get("mcp_servers")).and_then(|s| s.as_object())
    {
        for (name, entry) in servers {
            // Never allowed to replace a server the user configured themselves.
            if entry.is_object() && !merged.contains_key(name) {
                merged.insert(name.clone(), entry.clone());
            }
        }
    }
    merged
}

fn entry_disabled(cfg: &Value) -> bool {
    cfg.get("disabled").and_then(|d| d.as_bool()).unwrap_or(false)
        || cfg.get("enabled") == Some(&Value::Bool(false))
}

// ── public API ──────────────────────────────────────────────────────────────

/// Connect to every enabled server and return ``(server_name, tool)``.
///
/// Per-server connection failures are logged and skipped (non-fatal).
/// Idempotent: an already-connected server is reused, not re-spawned.
pub async fn discover(connect_timeout: f64) -> Vec<(String, McpToolInfo)> {
    let configs = load_server_configs();
    if configs.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<(String, McpToolInfo)> = Vec::new();
    for (name, cfg) in configs {
        if entry_disabled(&cfg) {
            continue;
        }
        let existing = SERVERS.lock().get(&name).cloned();
        let srv = match existing {
            Some(s) if s.connected.load(Ordering::SeqCst) => s,
            _ => {
                let srv = Arc::new(Server::new(&name, cfg));
                SERVERS.lock().insert(name.clone(), srv.clone());
                let connect = tokio::time::timeout(
                    Duration::from_secs_f64(connect_timeout),
                    srv.connect(),
                )
                .await;
                match connect {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        *srv.error.lock() = Some(e.clone());
                        log_error(&format!("MCP: server '{name}' unavailable ({e})"), None);
                        continue;
                    }
                    Err(_) => {
                        let e = format!("connect timed out after {connect_timeout:.0}s");
                        *srv.error.lock() = Some(e.clone());
                        log_error(&format!("MCP: server '{name}' unavailable ({e})"), None);
                        continue;
                    }
                }
                srv
            }
        };
        for tool in srv.tools.lock().iter() {
            out.push((name.clone(), tool.clone()));
        }
    }
    out
}

/// Invoke a tool on a connected server.
pub async fn call_tool(
    server_name: &str,
    tool_name: &str,
    arguments: Value,
    timeout: f64,
) -> Result<Value, String> {
    let srv = SERVERS.lock().get(server_name).cloned();
    let Some(srv) = srv else {
        return Err(format!("MCP server '{server_name}' is not connected"));
    };
    srv.call(tool_name, arguments, Duration::from_secs_f64(timeout)).await
}

/// The Python gated on the optional SDK; the Rust client is built in.
pub fn is_available() -> bool {
    true
}

/// Lightweight status snapshot for the `mcp` CLI / diagnostics.
#[derive(Debug, Clone)]
pub struct ServerStatus {
    pub name: String,
    pub transport: String,
    pub connected: bool,
    pub tools: usize,
    pub error: Option<String>,
    pub disabled: bool,
}

pub fn server_status() -> Vec<ServerStatus> {
    let mut rows = Vec::new();
    for (name, cfg) in load_server_configs() {
        let srv = SERVERS.lock().get(&name).cloned();
        rows.push(ServerStatus {
            name: name.clone(),
            transport: transport_type(&cfg),
            connected: srv
                .as_ref()
                .map(|s| s.connected.load(Ordering::SeqCst))
                .unwrap_or(false),
            tools: srv.as_ref().map(|s| s.tools.lock().len()).unwrap_or(0),
            error: srv.as_ref().and_then(|s| s.error.lock().clone()),
            disabled: entry_disabled(&cfg),
        });
    }
    rows
}

/// Extract the JSON-RPC response with `id` from an SSE body
/// (`data: {...}` lines; multi-line data concatenated per SSE spec).
fn parse_sse_response(body: &str, id: i64) -> Option<Value> {
    let mut data = String::new();
    let check = |data: &mut String| -> Option<Value> {
        if data.is_empty() {
            return None;
        }
        let parsed = serde_json::from_str::<Value>(data).ok();
        data.clear();
        let msg = parsed?;
        if msg.get("id").and_then(|i| i.as_i64()) == Some(id) {
            Some(msg)
        } else {
            None
        }
    };
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.trim_start());
        } else if line.is_empty() {
            if let Some(msg) = check(&mut data) {
                return Some(msg);
            }
        }
    }
    check(&mut data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_parse_matching_id() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n";
        let v = parse_sse_response(body, 7).unwrap();
        assert_eq!(v["result"]["ok"], Value::Bool(true));
        assert!(parse_sse_response(body, 8).is_none());
    }

    #[test]
    fn transport_inference() {
        assert_eq!(transport_type(&json!({"command": "npx"})), "stdio");
        assert_eq!(transport_type(&json!({"url": "https://x/mcp"})), "http");
        assert_eq!(transport_type(&json!({"type": "sse", "url": "u"})), "sse");
    }

    #[test]
    fn disabled_entries() {
        assert!(entry_disabled(&json!({"disabled": true})));
        assert!(entry_disabled(&json!({"enabled": false})));
        assert!(!entry_disabled(&json!({"command": "x"})));
    }
}
