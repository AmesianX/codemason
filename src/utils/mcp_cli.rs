//! ``codemason mcp`` management subcommand.
//! Equivalent to codemason/utils/mcp_cli.py.
//!
//! Intercepted at the top of cli::main() (before arg parsing, which owns the
//! positional ``prompt``). Edits ~/.codemason/mcp.json — the same
//! ``mcpServers`` map the runtime client (utils/mcp_client) reads — and can
//! live-probe a server.
//!
//!     codemason mcp list
//!     codemason mcp add <name> --command npx --args "-y,firecrawl-mcp" --env K=V --env K2=V2
//!     codemason mcp add <name> --url https://host/mcp [--transport http|sse] --header "Authorization: Bearer .."
//!     codemason mcp remove <name>
//!     codemason mcp test <name>

use serde_json::{json, Value};
use std::path::PathBuf;

use crate::utils::config::CONFIG_DIR;
use crate::utils::mcp_client;

fn mcp_json_path() -> PathBuf {
    CONFIG_DIR.join("mcp.json")
}

fn load() -> Result<Value, i32> {
    let path = mcp_json_path();
    if path.is_file() {
        match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string()))
        {
            Ok(v) => return Ok(v),
            Err(exc) => {
                eprintln!("error: could not read {}: {exc}", path.display());
                return Err(1);
            }
        }
    }
    Ok(json!({}))
}

/// Apply `mutate` to mcp.json as a single step, atomically.
///
/// A shared document, and the edits are per-SERVER: `mcp add a` and `mcp remove
/// b` in two terminals both read the whole file and both write it back, so the
/// later one reverted the other's server outright. Reading inside the lock —
/// rather than passing in a copy loaded earlier — is what makes each edit see the
/// other's. The plain write it replaces also truncated first, so a crash or a
/// concurrent reader could see an mcp.json with no servers at all.
///
/// `mutate` gets the document (always an object) and returns it, or None for
/// "no change". Returns whether the file was written.
fn edit<F>(mutate: F) -> bool
where
    F: FnOnce(Value) -> Option<Value>,
{
    let path = mcp_json_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    crate::utils::file::update_json_document(
        &path,
        &crate::utils::config::lock_path("mcp.json"),
        |current| {
            mutate(match current {
                Some(v) if v.is_object() => v,
                _ => json!({}),
            })
        },
        true,
    )
}

fn servers(cfg: &mut Value) -> &mut serde_json::Map<String, Value> {
    if !cfg.is_object() {
        *cfg = json!({});
    }
    let obj = cfg.as_object_mut().expect("just coerced to object");
    let entry = obj
        .entry("mcpServers".to_string())
        .or_insert_with(|| json!({}));
    if !entry.is_object() {
        *entry = json!({});
    }
    entry.as_object_mut().expect("just coerced to object")
}

async fn cmd_list(args: &[String]) -> i32 {
    if !mcp_client::is_available() {
        eprintln!("MCP SDK not installed (pip install 'mcp>=1.9').");
    }
    let probe = !args.iter().any(|a| a == "--no-probe");
    if probe && mcp_client::is_available() && !mcp_client::load_server_configs().is_empty() {
        // Actively connect so the status reflects reality (tool counts,
        // errors), not just what a prior in-process discovery populated.
        eprintln!("Probing servers ...");
        let _ = mcp_client::discover(60.0).await;
    }
    let rows = mcp_client::server_status();
    if rows.is_empty() {
        println!("No MCP servers configured. Add one with: codemason mcp add <name> ...");
        return 0;
    }
    for r in rows {
        let state = if r.disabled {
            "disabled"
        } else if r.connected {
            "connected"
        } else {
            "not connected"
        };
        let mut line = format!("  {:<20} {:<6} {state}", r.name, r.transport);
        if r.tools > 0 {
            line += &format!("  ({} tools)", r.tools);
        }
        if let Some(err) = &r.error {
            line += &format!("  error: {err}");
        }
        println!("{line}");
    }
    0
}

fn parse_kv(items: &[String]) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for it in items {
        if let Some((k, v)) = it.split_once('=') {
            out.insert(k.trim().to_string(), Value::String(v.to_string()));
        }
    }
    out
}

fn parse_headers(items: &[String]) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for it in items {
        if let Some((k, v)) = it.split_once(':') {
            out.insert(k.trim().to_string(), Value::String(v.trim().to_string()));
        }
    }
    out
}

fn cmd_add(args: &[String]) -> i32 {
    let Some(name) = args.first().cloned() else {
        eprintln!(
            "usage: codemason mcp add <name> [--command C --args a,b | --url U] \
             [--transport http|sse] [--env K=V ...] [--header 'H: v' ...]"
        );
        return 2;
    };
    let rest = &args[1..];
    let mut command: Option<String> = None;
    let mut url: Option<String> = None;
    let mut transport: Option<String> = None;
    let mut arg_list: Vec<String> = Vec::new();
    let mut envs: Vec<String> = Vec::new();
    let mut headers: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        let tok = rest[i].as_str();
        macro_rules! next {
            () => {{
                i += 1;
                match rest.get(i) {
                    Some(v) => v.clone(),
                    None => {
                        eprintln!("unknown option: {tok}");
                        return 2;
                    }
                }
            }};
        }
        match tok {
            "--command" => command = Some(next!()),
            "--args" => {
                arg_list = next!()
                    .split(',')
                    .filter(|a| !a.is_empty())
                    .map(|a| a.to_string())
                    .collect()
            }
            "--url" => url = Some(next!()),
            "--transport" => transport = Some(next!()),
            "--env" => envs.push(next!()),
            "--header" => headers.push(next!()),
            _ => {
                eprintln!("unknown option: {tok}");
                return 2;
            }
        }
        i += 1;
    }

    let mut entry = serde_json::Map::new();
    if let Some(command) = command.filter(|c| !c.is_empty()) {
        entry.insert("command".to_string(), Value::String(command));
        if !arg_list.is_empty() {
            entry.insert(
                "args".to_string(),
                Value::Array(arg_list.into_iter().map(Value::String).collect()),
            );
        }
        let env = parse_kv(&envs);
        if !env.is_empty() {
            entry.insert("env".to_string(), Value::Object(env));
        }
    } else if let Some(url) = url.filter(|u| !u.is_empty()) {
        entry.insert(
            "type".to_string(),
            Value::String(transport.unwrap_or_else(|| "http".to_string())),
        );
        entry.insert("url".to_string(), Value::String(url));
        let hdr = parse_headers(&headers);
        if !hdr.is_empty() {
            entry.insert("headers".to_string(), Value::Object(hdr));
        }
    } else {
        eprintln!("error: provide either --command or --url");
        return 2;
    }

    let entry_for_file = Value::Object(entry.clone());
    let added_name = name.clone();
    let wrote = edit(move |mut cfg| {
        servers(&mut cfg).insert(added_name, entry_for_file);
        Some(cfg)
    });
    if !wrote {
        // edit() refuses to overwrite an mcp.json it could not read, so saying
        // "Added" here would be a lie about a file we left alone.
        eprintln!("error: {} could not be read; nothing was changed", mcp_json_path().display());
        return 1;
    }
    println!("Added MCP server '{name}' to {}", mcp_json_path().display());
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(entry)).unwrap_or_default()
    );
    println!("\nProbe it with: codemason mcp test {name}");
    0
}

fn cmd_remove(args: &[String]) -> i32 {
    let Some(name) = args.first() else {
        eprintln!("usage: codemason mcp remove <name>");
        return 2;
    };
    // "is it there" and "remove it" have to be the same step as the write, or a
    // concurrent edit decides the answer after we looked.
    let removed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = removed.clone();
    let target = name.clone();
    let wrote = edit(move |mut cfg| {
        let map = servers(&mut cfg);
        if map.remove(target.as_str()).is_none() {
            return None;
        }
        seen.store(true, std::sync::atomic::Ordering::SeqCst);
        Some(cfg)
    });
    if !removed.load(std::sync::atomic::Ordering::SeqCst) {
        eprintln!("error: no such MCP server: {name}");
        return 1;
    }
    if !wrote {
        eprintln!("error: {} could not be read; nothing was changed", mcp_json_path().display());
        return 1;
    }
    println!("Removed MCP server '{name}'");
    0
}

async fn cmd_test(args: &[String]) -> i32 {
    let Some(name) = args.first() else {
        eprintln!("usage: codemason mcp test <name>");
        return 2;
    };
    let mut cfg = match load() {
        Ok(c) => c,
        Err(code) => return code,
    };
    if !servers(&mut cfg).contains_key(name.as_str()) {
        eprintln!("error: no such MCP server: {name}");
        return 1;
    }
    println!("Connecting to '{name}' ...");
    let found: Vec<(String, mcp_client::McpToolInfo)> = mcp_client::discover(90.0)
        .await
        .into_iter()
        .filter(|(s, _t)| s == name)
        .collect();
    let st = mcp_client::server_status()
        .into_iter()
        .find(|r| &r.name == name);
    if let Some(st) = &st {
        if let Some(err) = &st.error {
            eprintln!("FAILED: {err}");
            return 1;
        }
    }
    println!("OK — {} tools:", found.len());
    for (_s, t) in found {
        let desc = t.description.trim().lines().next().unwrap_or("").to_string();
        println!("  {:<32} {desc}", t.name);
    }
    0
}

/// Dispatch ``codemason mcp <sub> ...``. `argv` is the args after "mcp".
/// Sync entry (called before the REPL's runtime exists): spins a small tokio
/// runtime for the probing subcommands.
pub fn handle_mcp_cli(argv: &[String]) -> i32 {
    if argv.is_empty() || matches!(argv[0].as_str(), "-h" | "--help" | "help") {
        println!(
            "codemason mcp — manage MCP servers\n\n  list                     list configured servers and status\n  add <name> ...           add a server (--command/--args/--env or --url/--header)\n  remove <name>            remove a server\n  test <name>              connect and list the server's tools"
        );
        return 0;
    }
    let sub = argv[0].as_str();
    let rest: Vec<String> = argv[1..].to_vec();
    match sub {
        "list" | "ls" => run_async(cmd_list(&rest)),
        "add" => cmd_add(&rest),
        "remove" | "rm" => cmd_remove(&rest),
        "test" => run_async(cmd_test(&rest)),
        _ => {
            eprintln!("unknown mcp subcommand: {sub}");
            2
        }
    }
}

fn run_async<F: std::future::Future<Output = i32>>(fut: F) -> i32 {
    match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt.block_on(fut),
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}
