//! CLI Entry Point for CodeMason
//! Equivalent to codemason/cli.py (src/entrypoints/cli.tsx).
//!
//! Substrate divergence (documented): the Python's prompt_toolkit
//! PromptSession + patch_stdout pair is replaced by utils::prompt_engine (the
//! persistent bottom-block editor) plus utils::term_out (the output router
//! that writes above it). Every behavior contract — Enter submit vs mid-turn
//! steering, the double-Esc arming, single-keypress permission answers, the
//! Ctrl+O pager, the context gauge, the token footer — is carried over.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;

use crate::tools::base::{PermissionDecision, Tool};
use crate::types::query::{ContextOptions, Message, ToolUseContext};
use crate::utils::config::{get_config, save_config};
use crate::utils::formatting::{
    colorize, colors, display_width, format_tokens_compact, format_token_display, plain_mode,
    render_markdown, safe_print, sanitize_terminal_text, summarize_tool_input, sym,
};
use crate::utils::i18n::{tr, trf, FmtArg};
use crate::utils::log::{debug_log, log_error};
use crate::utils::messages::REJECT_MESSAGE;
use crate::utils::spinner::StreamingSpinner;
use crate::{cmprint, cmprintln};

pub const PRODUCT_NAME: &str = "CodeMason";
/// Taken from the package, never spelled out again.
/// One source of truth: Cargo.toml. This was a second literal, and two
/// literals holding the same version is a version that can disagree with
/// itself without anything noticing.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The commit this binary was built from, with `-dirty` when the tree had
/// uncommitted changes, and the commit's date. Baked by build.rs -- see the
/// note there for why a binary has to be able to identify itself.
pub const BUILD_ID: &str = concat!(
    env!("CODEMASON_GIT_HASH"),
    " ",
    env!("CODEMASON_COMMIT_DATE")
);

/// Get context warning message if approaching limits (thresholds).
pub fn get_token_warning(token_usage: i64) -> String {
    let state = crate::services::compaction::calculate_token_warning_state(token_usage);
    if !state.is_above_warning_threshold {
        return String::new();
    }
    let used_pct = 100 - state.percent_left;

    // Progress bar
    let bar_width = 20i64;
    let filled = (bar_width * used_pct / 100) as usize;
    let bar = format!(
        "{}{}",
        sym("bar_full").repeat(filled),
        sym("bar_empty").repeat(bar_width as usize - filled)
    );

    if state.is_above_error_threshold {
        colorize(
            &format!("{} Context: [{bar}] {used_pct}% - Run '/compact' to continue", sym("warn")),
            colors::RED,
        )
    } else {
        colorize(&format!("{} Context: [{bar}] {used_pct}%", sym("bolt")), colors::YELLOW)
    }
}

/// True when the response actually carried token counts.
fn reported_usage(usage: Option<&crate::types::messages::BetaUsage>) -> bool {
    match usage {
        Some(u) => u.input_tokens != 0 || u.output_tokens != 0,
        None => false,
    }
}

/// The dim per-response line ('⎯ ↓ 51k / 232.8k (22%) · 3.2s'), or '' when
/// there is no duration to report yet.
///
/// The fallback is the point of having this in one place: when a response
/// carries no usage, the spinner's streamed tokens are OUTPUT ONLY, and
/// printing that as context fill reads falsely reassuring. The honest stand-in
/// is the estimate compaction itself sizes on.
fn render_token_footer(
    usage: Option<&crate::types::messages::BetaUsage>,
    duration_ms: i64,
    spinner: &StreamingSpinner,
    fallback: i64,
    verbose: bool,
) -> String {
    let (elapsed, _streamed) = spinner.get_final_stats();
    let display_duration = if duration_ms > 0 { duration_ms } else { (elapsed * 1000.0) as i64 };
    if display_duration <= 0 {
        return String::new();
    }
    colorize(
        &format_token_display(usage, display_duration, verbose, fallback),
        colors::DIM,
    )
}

// === Tool Display ===

/// Print the tool-use header, then the tool's own rendering of its input.
///
/// Dispatcher only: each tool owns its display via
/// Tool::render_tool_use_message. No per-tool switch here.
pub fn display_tool_use(api_tool_name: &str, display_name: &str, tool_input: &Value, verbose: bool) {
    cmprintln!("[{display_name}]");
    let body = match crate::tools::get_tool_by_name(api_tool_name) {
        Ok(t) => t.render_tool_use_message(tool_input, verbose),
        // Unknown tool (alias not in the registry) — never dump the raw input.
        Err(_) => summarize_tool_input(tool_input, 160),
    };
    // Sanitize before printing. The body is rendered from tool INPUT, which the
    // model writes, and this is drawn immediately above the Allow? prompt — the
    // one control standing between the model and execution. An unfiltered
    // ESC[2K\r in a command string erases the line the user just read and draws
    // a harmless-looking one over it, so the command approved is not the command
    // shown. sanitize_terminal_text was already written for untrusted text and
    // applied only to tool RESULTS; input is no less model-controlled.
    let body = sanitize_terminal_text(&body);
    for line in body.split('\n') {
        if !line.is_empty() {
            safe_print(&format!("  {line}"));
        }
    }
}

/// Width of the argument echoed on a sub-agent's progress line.
const AGENT_PROGRESS_COLS: usize = 80;

// === Permission Handling ===

/// 'a' (always) answers at the permission prompt land here — session-scoped.
/// Keys are per tool; Bash is keyed by the command's FIRST WORD, and compound
/// commands never match — a blanket 'ls' allow must not wave through
/// 'ls && rm -rf' NOR 'echo x > ~/.bashrc'.
static SESSION_TOOL_ALLOWS: Lazy<Mutex<HashSet<(String, String)>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

static BASH_COMPOUND_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"[;|&`\n<>]|\$\(").unwrap());

fn permission_key(tool_name: &str, input: &Value) -> Option<(String, String)> {
    // Every command-executing tool gets per-command keying, not just the
    // literal "Bash" (DIVERGENCE from cli.py:654, which predates the Windows
    // port): with the old check, one 'always' on the PowerShell tool inserted
    // ("PowerShell","") and blanket-allowed every later command of any content
    // — exactly the hole the Bash special case exists to close.  The compound
    // guard is shared; PowerShell's backtick is its escape char, and treating
    // it as compound only errs toward asking.
    if matches!(tool_name, "Bash" | "PowerShell") {
        let cmd = input.get("command").and_then(|c| c.as_str()).unwrap_or("").trim().to_string();
        if cmd.is_empty() || BASH_COMPOUND_RE.is_match(&cmd) {
            return None; // compound/empty: always ask
        }
        let head = cmd.split_whitespace().next().unwrap_or("");
        // An interpreter turns one grant into unattended arbitrary code: 'a' on
        // `python build.py` would otherwise auto-approve `python -c "<anything>"`,
        // which contains no character the compound guard looks for.
        // DANGEROUS_BASH_PATTERNS was written for exactly this and had no
        // consumer; its module doc claimed there was no allow-rule engine to
        // protect, but SESSION_TOOL_ALLOWS is one.
        let base = std::path::Path::new(head)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(head);
        let blocked = crate::utils::permissions::dangerous_patterns::DANGEROUS_BASH_PATTERNS
            .iter()
            .any(|p| *p == head || *p == base);
        if blocked {
            return None;
        }
        return Some((tool_name.to_string(), head.to_string()));
    }

    // File-mutating tools: keyed on the resolved path, not the tool. The user
    // approves a specific file; keying on the name alone silently extended that
    // to every later path, including ~/.bashrc, .git/hooks/* and the config dir
    // (whose contents this program executes as shell — see is_config_path).
    if matches!(tool_name, "Write" | "Edit" | "NotebookEdit") {
        let raw = input
            .get("file_path")
            .or_else(|| input.get("notebook_path"))
            .and_then(|p| p.as_str())
            .unwrap_or("");
        if raw.is_empty() {
            return None;
        }
        // Anchor to the SESSION cwd, then reduce to one spelling. Without the
        // second step the key is whatever the model happened to type, and on
        // Windows the same file reached the grant set under several names:
        // a relative path became `C:\dir/f.txt` (the join uses '/') while the
        // absolute form was `C:\dir\f.txt`, and `.\f.txt` kept its unfolded
        // '.'. Those never matched, so 'always' on a file still re-prompted on
        // the next write to it. POSIX hid the bug because '/' is the separator
        // there and the two forms already collided.
        let anchored = crate::utils::state::resolve_for_containment(
            raw,
            &crate::utils::state::get_cwd(),
        );
        return Some((tool_name.to_string(), crate::utils::state::canonical_spelling(&anchored)));
    }

    Some((tool_name.to_string(), String::new()))
}

/// True for anything under the config dir.
///
/// Files there are executed, not just read: settings.json's Stop hooks and
/// goal-gates.txt both run through a shell with no permission check of their
/// own, and mcp.json names a process spawned at startup. A write there is a
/// code-execution request wearing a file-write prompt, so it is never covered
/// by a session grant and is always shown for its own decision.
fn is_config_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    // is_in_directory resolves both sides and compares by path segment, so a
    // sibling like ~/.codemason-evil is correctly outside.
    let target = crate::utils::state::resolve_for_containment(
        path,
        &crate::utils::state::get_cwd(),
    );
    let dir = crate::utils::config::CONFIG_DIR.to_string_lossy().into_owned();
    crate::utils::state::is_in_directory(&target, &dir)
}

/// The REPL router the permission prompt routes through (set for the
/// interactive loop, None in --plain and -p mode).
static ROUTER: Lazy<Mutex<Option<Arc<ReplInputRouter>>>> = Lazy::new(|| Mutex::new(None));

fn set_router(r: Option<Arc<ReplInputRouter>>) {
    *ROUTER.lock() = r;
}

fn router() -> Option<Arc<ReplInputRouter>> {
    ROUTER.lock().clone()
}

/// Put the question on the terminal and (if configured) on the phone at the
/// same time; the first answer wins.
///
/// Racing rather than assigning ownership is the point: neither side owns the
/// decision, so nothing has to be handed over. A remote timeout is not an
/// answer — it drops back to waiting on the terminal.
async fn ask_permission_local_or_remote(
    router: &Arc<ReplInputRouter>,
    tool: &dyn Tool,
    input: &Value,
    remote_ok: bool,
    prompt_text: &str,
) -> String {
    let local = router.request_permission(prompt_text);
    if !remote_ok {
        let response = local.await.trim().to_lowercase();
        cmprintln!("{prompt_text}{response}");
        return response;
    }

    let cwd = crate::utils::state::get_cwd();
    let remote = crate::remote::permissions::ask(tool, input, Some(&cwd), None);
    tokio::pin!(local);
    tokio::pin!(remote);

    tokio::select! {
        response = &mut local => {
            let response = response.trim().to_lowercase();
            cmprintln!("{prompt_text}{response}");
            response
        }
        answer = &mut remote => {
            match answer {
                Some(a) if !a.is_empty() => {
                    cmprintln!("{prompt_text}{a}  {}", tr("(remote)"));
                    a
                }
                // Phone timed out. Keep waiting on the terminal — never convert
                // silence into a decision.
                _ => {
                    let response = local.await.trim().to_lowercase();
                    cmprintln!("{prompt_text}{response}");
                    response
                }
            }
        }
    }
}

/// Check if we have permission to use a tool (has_permissions_to_use_tool).
///
/// dangerously_skip_permissions allows EVERYTHING, with no exception: in that
/// mode no permission prompt may ever appear (user instruction). The previous
/// always_ask() carve-out meant ExitPlanMode still blocked an unattended run on
/// a prompt nobody was there to answer, which is the opposite of what the flag
/// is for. The only question the user should still see is the model's own
/// AskUserQuestion, a deliberate tool call rather than a permission gate.
///
/// Everything after that early return is therefore non-dsp only, by
/// construction — which is where the hardening below belongs.
pub async fn has_permissions_to_use_tool(
    tool_name: &'static str,
    input: Value,
    context: ToolUseContext,
) -> PermissionDecision {
    let Ok(tool) = crate::tools::get_tool_by_name(tool_name) else {
        // Fail CLOSED. An unresolvable tool name is not a reason to allow;
        // every other default around here denies, and this one only looked
        // safe because the current call graph cannot reach it.
        return PermissionDecision::denied(format!("Unknown tool: {tool_name}"));
    };
    if context.options.dangerously_skip_permissions {
        return PermissionDecision::allowed();
    }

    // Read-only tools are auto-approved — unless the tool itself asks to be
    // asked. The default check_permissions returns allow() for a read-only
    // tool, so every existing one is unaffected; the exception is a tool whose
    // "read" reaches OUT of this machine.
    //
    // WebFetch and WebSearch are read-only in the sense that they change
    // nothing locally, but the URL is model-controlled, so a read of any file
    // (Read has no path restriction) can be posted to an arbitrary host in the
    // next call, with neither step ever prompting. Both being auto-approved
    // here made that a silent egress channel. Under -dsp nothing reaches this
    // line at all, which is the intended behaviour: skipping permissions means
    // skipping them.
    if tool.is_read_only() {
        let pre = tool.check_permissions(&input, &context).await;
        match pre.behavior {
            crate::tools::base::PermissionBehavior::Allow => {
                return PermissionDecision::allowed()
            }
            crate::tools::base::PermissionBehavior::Deny => {
                return PermissionDecision::denied(pre.message)
            }
            crate::tools::base::PermissionBehavior::Ask => {}
        }
    }

    // Session 'always' grants ('a' at a previous ask). A write into the config
    // dir is never covered by one: those files are executed, so each is its own
    // decision no matter what was granted earlier.
    let mut pkey = permission_key(tool.name(), &input);
    let cfg_target = if matches!(tool.name(), "Write" | "Edit" | "NotebookEdit") {
        input
            .get("file_path")
            .or_else(|| input.get("notebook_path"))
            .and_then(|p| p.as_str())
            .unwrap_or("")
    } else {
        ""
    };
    if is_config_path(cfg_target) {
        pkey = None;
    } else if let Some(k) = &pkey {
        if SESSION_TOOL_ALLOWS.lock().contains(k) {
            return PermissionDecision::allowed();
        }
    }

    // Display tool info.
    let display = tool.user_facing_name(Some(&input));
    display_tool_use(tool.name(), &display, &input, false);

    let router = router();
    let remote_ok = crate::remote::permissions::is_available();
    let prompt = format!("  {}", tr("Allow? (y/n, a=always): "));

    loop {
        let response = match &router {
            Some(r) => {
                // Interactive: the persistent prompt owns stdin, so route the
                // ask through it — a single y/n/a keypress answers.
                ask_permission_local_or_remote(r, tool.as_ref(), &input, remote_ok, &prompt).await
            }
            None => {
                // Headless (-p): there is no terminal to answer with. Ask the
                // phone first when one is configured; fall through to stdin.
                let mut response: Option<String> = None;
                use std::io::IsTerminal;
                if remote_ok && !std::io::stdin().is_terminal() {
                    let cwd = crate::utils::state::get_cwd();
                    response = crate::remote::permissions::ask(tool.as_ref(), &input, Some(&cwd), None).await;
                    if let Some(r) = &response {
                        cmprintln!("{prompt}{r}  {}", tr("(remote)"));
                    }
                }
                match response {
                    Some(r) => r,
                    None => {
                        cmprint!("{prompt}");
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                        let mut line = String::new();
                        match std::io::stdin().read_line(&mut line) {
                            // Ctrl+D / EOF at the permission prompt = decline.
                            Ok(0) | Err(_) => {
                                return PermissionDecision::denied(REJECT_MESSAGE);
                            }
                            Ok(_) => line.trim().to_lowercase(),
                        }
                    }
                }
            }
        };

        match response.as_str() {
            "y" | "yes" => return PermissionDecision::allowed(),
            "a" | "always" => {
                match &pkey {
                    Some(k) => {
                        SESSION_TOOL_ALLOWS.lock().insert(k.clone());
                        let what = if !k.1.is_empty() {
                            format!("'{}' Bash commands", k.1)
                        } else {
                            format!("{} calls", k.0)
                        };
                        cmprintln!("{}", colorize(&format!("  (always allowing {what} this session)"), colors::DIM));
                    }
                    None => {
                        // Compound Bash command — no safe prefix to generalize.
                        cmprintln!("{}", colorize("  (compound command — allowed once, will ask again)", colors::DIM));
                    }
                }
                return PermissionDecision::allowed();
            }
            "n" | "no" => return PermissionDecision::denied(REJECT_MESSAGE),
            _ => {
                // Re-prompt on anything else instead of silently looping.
                cmprintln!("  {}", tr("Please answer 'y', 'n' or 'a'."));
            }
        }
    }
}

/// The can_use_tool callback the engine calls (CanUseToolFn).
pub fn permission_callback() -> crate::tools::base::CanUseToolFn {
    Arc::new(|tool_name, input, context| {
        Box::pin(has_permissions_to_use_tool(tool_name, input, context))
    })
}

// === REPL helpers ===

/// The keep-going Stop hook asks the model to end a completed task with the
/// exact line `TASK_COMPLETE`. It belongs in history (the hook reads it) but
/// must not be shown to the user.
static SENTINEL_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\n?[ \t]*TASK_COMPLETE[ \t]*$").unwrap());

fn strip_completion_sentinel(text: &str) -> String {
    SENTINEL_RE.replace(text, "").trim_end().to_string()
}

/// REPL built-ins, slash-invoked (/help, /clear, ...).
pub const BUILTIN_COMMANDS: &[&str] = &[
    "exit", "quit", "clear", "compact", "help", "keepgoing", "dsp", "effort", "reasoning",
    "think", "goal", "resume", "expand", "export", "language", "lang", "ab",
    // Claude Code names this /remote-control (/rc); keep both.
    "remote", "remote-control", "rc",
];

/// Effective keep-going state — same precedence as the Stop hook script:
/// CODEMASON_KEEP_GOING env (session override) > config.json keep_going > True.
fn keep_going_enabled() -> bool {
    match std::env::var("CODEMASON_KEEP_GOING") {
        Ok(v) => !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no" | "off" | ""),
        Err(_) => get_config().keep_going,
    }
}

/// Render the in-memory conversation to Markdown (for /export).
fn export_conversation_md(messages: &[Message]) -> String {
    use crate::types::messages::{ContentBlock, ContentBlockParam};
    let mut lines: Vec<String> = vec!["# CodeMason conversation export".into(), String::new()];
    for m in messages {
        match m {
            Message::User(u) => {
                let blocks = u.message.content.as_blocks();
                let only_results = !blocks.is_empty()
                    && blocks.iter().all(|b| matches!(b, ContentBlockParam::ToolResult { .. }));
                if !only_results {
                    lines.push("## 🧑 User".into());
                    lines.push(String::new());
                }
                for b in blocks {
                    match b {
                        ContentBlockParam::Text { text, .. } if !text.trim().is_empty() => {
                            lines.push(text.trim_end().to_string());
                            lines.push(String::new());
                        }
                        ContentBlockParam::ToolResult { content, .. } => {
                            let mut res = content.as_text();
                            if res.chars().count() > 4000 {
                                let total = res.chars().count();
                                res = res.chars().take(4000).collect::<String>()
                                    + &format!("\n… (truncated, {total} chars total)");
                            }
                            lines.push("  ⎿".into());
                            lines.push("```".into());
                            lines.push(res.trim_end().to_string());
                            lines.push("```".into());
                            lines.push(String::new());
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant(a) => {
                lines.push("## 🤖 Assistant".into());
                lines.push(String::new());
                for b in &a.message.content {
                    match b {
                        ContentBlock::Text(t) if !t.text.trim().is_empty() => {
                            lines.push(t.text.trim_end().to_string());
                            lines.push(String::new());
                        }
                        ContentBlock::ToolUse(tu) => {
                            let arg = tu
                                .input
                                .get("command")
                                .and_then(|c| c.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| tu.input.to_string());
                            let name = if tu.name.is_empty() { "tool" } else { &tu.name };
                            lines.push(format!("**⏺ {name}**"));
                            lines.push(String::new());
                            lines.push("```".into());
                            lines.push(arg);
                            lines.push("```".into());
                            lines.push(String::new());
                        }
                        _ => {}
                    }
                }
            }
            Message::Progress(_) => {}
        }
    }
    format!("{}\n", lines.join("\n").trim_end())
}

// --- Full tool-output viewer (Ctrl+O / /expand) ---
//
// The display shows tool results truncated (5 lines) and very large outputs
// are persisted to disk with only a 2KB preview in context. This ring buffer
// keeps the FULL result text (and the persisted file path when one exists) so
// Ctrl+O can open the real thing in a pager without touching the scrollback.

#[derive(Clone)]
struct ToolOutputEntry {
    tool: String,
    input: Value,
    text: String,
    path: Option<String>,
    time: String,
}

static RECENT_TOOL_OUTPUTS: Lazy<Mutex<Vec<ToolOutputEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));
const MAX_RECENT_TOOL_OUTPUTS: usize = 20;
/// /expand prints inline into the scrollback — keep it small and capped.
const EXPAND_DEFAULT_COUNT: usize = 3;
const EXPAND_PAGES_PER_ENTRY: usize = 3;
const INLINE_EXPAND_BYTE_GUARD: usize = 200 * 1024;

static PERSISTED_PATH_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"Full output saved to: (\S+)").unwrap());

fn record_tool_output(tool_name: &str, tool_input: &Value, text: &str) {
    if text.is_empty() {
        return;
    }
    let path = PERSISTED_PATH_RE
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string());
    let mut buf = RECENT_TOOL_OUTPUTS.lock();
    buf.push(ToolOutputEntry {
        tool: if tool_name.is_empty() { "Tool".into() } else { tool_name.to_string() },
        input: tool_input.clone(),
        text: text.to_string(),
        path,
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
    });
    let len = buf.len();
    if len > MAX_RECENT_TOOL_OUTPUTS {
        buf.drain(..len - MAX_RECENT_TOOL_OUTPUTS);
    }
}

/// One picker row: time, tool, input preview, size.
fn tool_output_row(entry: &ToolOutputEntry) -> String {
    let preview = entry
        .input
        .get("command")
        .or_else(|| entry.input.get("file_path"))
        .or_else(|| entry.input.get("pattern"))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| entry.input.to_string());
    let mut preview = preview.split_whitespace().collect::<Vec<_>>().join(" ");
    if preview.chars().count() > 60 {
        preview = preview.chars().take(60).collect::<String>() + "…";
    }
    let size = entry.text.len();
    format!(
        "{}  [{}] {preview}  ({}B)",
        entry.time,
        entry.tool,
        crate::utils::formatting::format_number(size as i64)
    )
}

/// The untruncated output: the persisted original file when one exists (the
/// in-context text is only a 2KB preview then), else the full result.
fn entry_full_text(entry: &ToolOutputEntry) -> String {
    if let Some(path) = &entry.path {
        if std::path::Path::new(path).is_file() {
            if let Ok(s) = std::fs::read_to_string(path) {
                return s;
            }
        }
    }
    entry.text.clone()
}

/// Show tool outputs in full in `less -R`. Falls back to plain printing when
/// less is missing or in --plain mode (dumb consoles can't run a pager).
async fn run_output_pager(entries: &[ToolOutputEntry]) {
    use std::io::Write;
    let tmp = std::env::temp_dir().join(format!("cm-output-{}.txt", uuid::Uuid::new_v4().simple()));
    {
        let Ok(mut f) = std::fs::File::create(&tmp) else {
            return;
        };
        // Key hint where it's needed — first line of the document.
        let _ = writeln!(f, "[q quit · / search · n/N match · Space/b page · g/G top/bottom]\n");
        for (i, entry) in entries.iter().enumerate() {
            if i > 0 {
                let _ = writeln!(f);
            }
            let _ = writeln!(f, "{} {} {}\n", "=".repeat(8), tool_output_row(entry), "=".repeat(8));
            let _ = writeln!(f, "{}", entry_full_text(entry));
        }
    }
    if plain_mode() {
        if let Ok(s) = std::fs::read_to_string(&tmp) {
            cmprintln!("{s}");
        }
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    // `less` by whatever route it is available: on PATH, else the one
    // git-bash brought, else a copy beside the executable, else the
    // vendored tree (empty today). See utils/pager for the full order. Windows has no system less, and before this the same keystroke
    // paged when codemason was started from Git Bash and dumped every stored
    // tool output into the scrollback when it was started from PowerShell.
    let paged = match crate::utils::pager::resolve_pager() {
        Some(less) => tokio::process::Command::new(less)
            .arg("-R")
            .arg(&tmp)
            .status()
            .await
            .is_ok(),
        None => false,
    };
    if !paged {
        if let Ok(s) = std::fs::read_to_string(&tmp) {
            cmprintln!("{s}");
        }
    }
    let _ = std::fs::remove_file(&tmp);
}

/// '(Bash completed with no output)' — the empty-result marker; useful in
/// context, noise on the terminal.
fn is_tool_result_marker(text: &str) -> bool {
    text.starts_with('(') && text.ends_with("completed with no output)")
}

/// Replay a resumed transcript's typed turns as UI-language evidence.
/// Only what the human wrote counts.
fn observe_restored_turns(restored: &[Value]) {
    use crate::utils::i18n::observe_user_text;
    for msg in restored {
        if msg.get("type").and_then(|t| t.as_str()) != Some("user") {
            continue;
        }
        if msg.get("isMeta").and_then(|b| b.as_bool()).unwrap_or(false)
            || msg.get("isCompactSummary").and_then(|b| b.as_bool()).unwrap_or(false)
        {
            continue;
        }
        let content = msg.get("message").and_then(|m| m.get("content"));
        match content {
            Some(Value::String(s)) => observe_user_text(s),
            Some(Value::Array(blocks)) => {
                if blocks
                    .iter()
                    .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
                {
                    continue;
                }
                for block in blocks {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                        observe_user_text(block.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Short orientation after a resume: what session this is and where it left off.
/// Build the resume banner as text. RETURNS rather than prints, so the caller
/// can render it after dropping the `messages` lock.
///
/// Printing runs through the prompt engine, which takes its `paint` lock and,
/// while holding it, calls the rprompt hook — and that hook locks `messages` to
/// size the context gauge. So a thread that prints while holding `messages` is
/// asking for `paint` from under `messages`, in the exact opposite order, and
/// the two threads wedge each other. Every thread then sits in futex_do_wait,
/// the terminal is still in raw mode so Ctrl+C and Ctrl+Z are read as ordinary
/// bytes nobody is left to read, and the session is unkillable from its own
/// window. That is the failure observed 2026-09-06.
fn render_resume_banner(messages: &[Message]) -> String {
    let mut out = String::new();
    macro_rules! line {
        ($($a:tt)*) => {{ out.push_str(&format!($($a)*)); out.push('\n'); }};
    }
    let n = messages.len();
    let values: Vec<Value> = messages.iter().map(crate::utils::serialization::to_jsonable).collect();
    let first = crate::utils::session_storage::extract_first_prompt(&values);
    line!("Resumed session {} ({n} messages)", crate::utils::state::get_session_id());
    line!("  First prompt: {first}");
    // Last assistant text = where the conversation left off.
    for m in messages.iter().rev() {
        if let Message::Assistant(_) = m {
            let tail = strip_completion_sentinel(&crate::utils::messages::get_text_from_message(m));
            let tail = tail.trim();
            if !tail.is_empty() {
                let tail = if tail.chars().count() > 500 {
                    tail.chars().take(500).collect::<String>() + "…"
                } else {
                    tail.to_string()
                };
                line!("{}", colorize("  Last response:", colors::DIM));
                for l in tail.split('\n').take(8) {
                    line!("{}", colorize(&format!("    {l}"), colors::DIM));
                }
                break;
            }
        }
    }
    out
}

/// Surface background-shell completions and inject them as <task-notification>
/// user messages.
fn drain_bg_notifications(messages: &Arc<Mutex<Vec<Message>>>) {
    let blocks = crate::utils::message_queue_manager::drain_pending_notifications();
    if blocks.is_empty() {
        return;
    }
    let count = blocks.len();
    // Mutate under the lock, print AFTER it is dropped. It used to print from
    // inside, and since the caller held `messages` across the whole call, the
    // print asked the prompt engine for its `paint` lock while holding
    // `messages` — while the engine's repaint holds `paint` and asks the
    // rprompt hook for `messages` to size the context gauge. Opposite order,
    // so the two wedged each other: every thread in futex_do_wait, the terminal
    // still in raw mode (Ctrl+C and Ctrl+Z arrive as bytes nobody is left to
    // read), and the session only killable from another window. It needed a
    // background task to finish at the same moment as a repaint, which is why
    // it struck rarely and looked random. Observed 2026-09-06; the /export that
    // appeared to hang was only the next caller to want `messages`.
    {
        let mut m = messages.lock();
        for block in blocks {
            m.push(Message::User(crate::utils::messages::create_user_message_text(&block)));
        }
        crate::utils::session_storage::record_transcript(&m, false);
    }
    for _ in 0..count {
        cmprintln!("{}", colorize(&format!("  {} Background task completed", sym("connector")), colors::CYAN));
    }
}

/// Say that configured credentials will not be used in --plain mode. Only when
/// something is actually configured.
fn warn_remote_off_in_plain() {
    let reason = crate::remote::notify::off_reason();
    if reason.is_empty() || !crate::remote::notify::adapters().iter().any(|a| a.is_configured()) {
        return;
    }
    // ASCII only: this is the mode for terminals that cannot draw anything else.
    cmprintln!("[!] {reason}: remote channel off (no phone alerts, approvals or questions in this mode)");
}

// The REPL input router, the turn loop, the slash-command dispatcher and main()
// live in cli_repl.rs (this file would otherwise be one 4,000-line module, and
// the split follows the Python's own section markers).
pub use crate::cli_repl::*;

// --- entry points cli_repl.rs calls back into (the Python had them as
// module-level functions in the same file) ---

pub const EXPAND_DEFAULT_COUNT_PUB: usize = EXPAND_DEFAULT_COUNT;

pub fn observe_restored_turns_pub(restored: &[Value]) {
    observe_restored_turns(restored);
}

/// Render the resume banner. The caller must print it AFTER releasing the
/// `messages` lock — see render_resume_banner.
pub fn render_resume_banner_pub(messages: &[Message]) -> String {
    render_resume_banner(messages)
}

pub fn export_conversation_md_pub(messages: &[Message]) -> String {
    export_conversation_md(messages)
}

pub fn recent_tool_outputs_is_empty() -> bool {
    RECENT_TOOL_OUTPUTS.lock().is_empty()
}

pub fn record_tool_output_pub(tool_name: &str, tool_input: &Value, text: &str) {
    record_tool_output(tool_name, tool_input, text);
}

pub fn drain_bg_notifications_pub(messages: &Arc<Mutex<Vec<Message>>>) {
    drain_bg_notifications(messages);
}

pub fn warn_remote_off_in_plain_pub() {
    warn_remote_off_in_plain();
}

pub fn keep_going_enabled_pub() -> bool {
    keep_going_enabled()
}

pub fn strip_completion_sentinel_pub(text: &str) -> String {
    strip_completion_sentinel(text)
}

/// `fallback` is the token estimate to show when the response carried no usage.
/// It is taken as a NUMBER rather than the message list on purpose: this call
/// reaches get_context_size(), which probes the model server over HTTP with a
/// 5s timeout and deliberately does not cache a failure. Passing `&messages
/// .lock()` made that guard an argument temporary alive for the whole call, so
/// with the backend unreachable every footer render pinned a worker thread for
/// the timeout WHILE HOLDING the conversation mutex — and the engine's context
/// gauge, which locks the same mutex, stopped repainting and stopped taking
/// keys for that long, every turn. Compute the estimate under a short lock at
/// the call site (see footer_fallback) and hand the number over.
pub fn render_token_footer_pub(
    usage: Option<&crate::types::messages::BetaUsage>,
    duration_ms: i64,
    spinner: &StreamingSpinner,
    fallback: i64,
    verbose: bool,
) -> String {
    render_token_footer(usage, duration_ms, spinner, fallback, verbose)
}

/// Token estimate for the footer, taking the conversation lock only as long as
/// the estimate itself needs. Zero when the response reported real usage, in
/// which case the estimate is unused and the lock is never taken at all.
pub fn footer_fallback(
    usage: Option<&crate::types::messages::BetaUsage>,
    messages: &Mutex<Vec<Message>>,
) -> i64 {
    if reported_usage(usage) {
        return 0;
    }
    let guard = messages.lock();
    crate::utils::tokens::token_count_with_estimation(&guard)
}

pub fn reported_usage_pub(usage: Option<&crate::types::messages::BetaUsage>) -> bool {
    reported_usage(usage)
}

pub fn set_router_pub(r: Option<Arc<ReplInputRouter>>) {
    set_router(r);
}

pub fn permission_key_pub(tool_name: &str, input: &Value) -> Option<(String, String)> {
    permission_key(tool_name, input)
}

pub fn session_tool_allows_contains(key: &(String, String)) -> bool {
    SESSION_TOOL_ALLOWS.lock().contains(key)
}

pub const AGENT_PROGRESS_COLS_PUB: usize = AGENT_PROGRESS_COLS;

/// /expand: print the last N tool outputs in full, inline into the scrollback.
/// Bounded per entry (SCREEN PAGES) so it can't flood the terminal.
pub fn expand_recent_outputs(n: usize) {
    let entries: Vec<ToolOutputEntry> = {
        let buf = RECENT_TOOL_OUTPUTS.lock();
        let start = buf.len().saturating_sub(n);
        buf[start..].to_vec() // chronological, newest last
    };
    let rows = crossterm::terminal::size().map(|(_, r)| r as usize).unwrap_or(24);
    let max_lines = (rows * EXPAND_PAGES_PER_ENTRY).max(20);
    cmprintln!();
    for entry in &entries {
        cmprintln!("{}", colorize(&format!("--- {}", tool_output_row(entry)), colors::CYAN));
        let mut text = entry_full_text(entry);
        if text.len() > INLINE_EXPAND_BYTE_GUARD {
            let mut cut = INLINE_EXPAND_BYTE_GUARD;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text.truncate(cut);
        }
        let tlines: Vec<&str> = text.split('\n').collect();
        if tlines.len() > max_lines {
            let cut = tlines.len() - max_lines;
            let head = tlines[..max_lines].join("\n");
            text = format!(
                "{head}{}",
                colorize(&format!("\n... ({cut} more lines - Ctrl+O shows the full output)"), colors::DIM)
            );
        }
        cmprintln!("{text}");
        cmprintln!();
    }
}

/// Show tool outputs in full in the pager (Ctrl+O): ALL recent outputs,
/// newest first, in one document.
pub async fn open_output_pager() {
    let entries: Vec<ToolOutputEntry> = {
        let buf = RECENT_TOOL_OUTPUTS.lock();
        buf.iter().rev().cloned().collect()
    };
    if entries.is_empty() {
        cmprintln!("(Ctrl+O: no tool output to show yet)");
        return;
    }
    run_output_pager(&entries).await;
}

/// Lines for the pending-message preview shown just ABOVE the '> ' input while
/// a turn runs — the messages you've typed and will send next.
pub fn pending_preview_lines(pending: &[String], cols: usize) -> Vec<String> {
    if pending.is_empty() {
        return Vec::new();
    }
    const MAX_BODY: usize = 4;
    let width = cols.saturating_sub(4).max(20);
    let mut body: Vec<String> = Vec::new();
    for msg in pending {
        for ln in msg.split('\n') {
            let mut ln = ln.trim_end().to_string();
            if ln.chars().count() > width {
                ln = ln.chars().take(width - 1).collect::<String>() + "…";
            }
            body.push(format!("⋯ {ln}"));
        }
    }
    if body.len() > MAX_BODY {
        let omitted = body.len() - MAX_BODY;
        body.truncate(MAX_BODY);
        body.push(format!("⋯ … (+{omitted} more)"));
    }
    body
}

/// Build the scrollback echo of a submitted prompt: '> ' marker + the text.
pub fn user_echo_line(text: &str) -> String {
    use crate::utils::formatting::{fg256_hex, supports_color};
    let body = format_user_echo(text);
    let theme = crate::utils::terminal_theme::active_theme_colors();
    let line = if !theme.is_empty() && supports_color() {
        let marker = match theme.get("prompt") {
            Some(p) => format!("{}{}> {}", fg256_hex(p), colors::BOLD, colors::RESET),
            None => colorize("> ", colors::DIM),
        };
        let body = match theme.get("input") {
            Some(i) => format!("{}{body}{}", fg256_hex(i), colors::RESET),
            None => body,
        };
        format!("{marker}{body}")
    } else {
        format!("{}{body}", colorize("> ", colors::DIM))
    };
    shade_user_echo(&line)
}

/// Lay a dim background behind the echoed prompt.
///
/// Per line, and only behind the characters: padding to the terminal width
/// would make the block copy out with trailing spaces. The reset is re-armed
/// after every embedded reset (the theme colours emit their own).
fn shade_user_echo(line: &str) -> String {
    use crate::utils::formatting::{hex_to_ansi256, supports_color};
    let cfg = get_config();
    if !supports_color() || !cfg.input_highlight {
        return line.to_string();
    }
    let theme = cfg.theme.to_lowercase();
    let bg_hex = if cfg.input_highlight_bg.is_empty() {
        if theme == "light" { "#d0d0d0" } else { "#444444" }
    } else {
        &cfg.input_highlight_bg
    };
    let fg_hex = if cfg.input_highlight_fg.is_empty() {
        if theme == "light" { "#1c1c1c" } else { "#ffffff" }
    } else {
        &cfg.input_highlight_fg
    };
    let on = format!("\x1b[48;5;{}m\x1b[38;5;{}m", hex_to_ansi256(bg_hex), hex_to_ansi256(fg_hex));
    line.split('\n')
        .map(|ln| {
            // A reset inside the line would drop the background for the rest of
            // it, so every reset is followed by re-arming it.
            let ln = ln.replace("\x1b[0m", &format!("\x1b[0m{on}"));
            format!("{on}{ln}\x1b[0m")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Scrollback echo for a submitted prompt: a pasted block is shown EXPANDED
/// but with the middle elided when it's long.
fn format_user_echo(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    const HEAD: usize = 12;
    const TAIL: usize = 6;
    const MAX_LINES: usize = 22;
    const MAX_CHARS: usize = 2400;
    if lines.len() <= MAX_LINES && text.chars().count() <= MAX_CHARS {
        return text.to_string();
    }
    // The guard above is an AND, so we also land here for a SHORT-line-count but
    // over-long text — a pasted paragraph or one long log line, where
    // lines.len() can be 1. Eliding a middle needs a middle to exist; without
    // this check lines[..HEAD] indexes past the end and the whole REPL panics on
    // submit. Only the char cap below applies to those.
    let out = if lines.len() > HEAD + TAIL {
        let mut kept: Vec<String> = lines[..HEAD].iter().map(|s| s.to_string()).collect();
        kept.push(format!("  … ({} lines elided) …", lines.len() - HEAD - TAIL));
        kept.extend(lines[lines.len() - TAIL..].iter().map(|s| s.to_string()));
        kept.join("\n")
    } else {
        text.to_string()
    };
    if out.chars().count() > MAX_CHARS * 2 {
        // a few enormous lines, not many lines
        let head: String = out.chars().take(MAX_CHARS).collect();
        return format!("{head}\n  … ({} chars elided) …", text.chars().count() - MAX_CHARS);
    }
    out
}

/// constants/xml.ts LOCAL_COMMAND_CAVEAT_TAG + messages.ts:566-568, verbatim.
pub const LOCAL_COMMAND_CAVEAT: &str = "<local-command-caveat>Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.</local-command-caveat>";

/// Print help information. The tool list shows the SESSION's active tools when
/// given, falling back to the registry — never a hardcoded list.
pub fn print_help(tools: Option<&[Arc<dyn Tool>]>) {
    let owned;
    let tools: &[Arc<dyn Tool>] = match tools {
        Some(t) => t,
        None => {
            owned = crate::tools::all_tools();
            &owned
        }
    };
    let tool_names: String = tools.iter().map(|t| format!("  {}", t.name())).collect::<Vec<_>>().join("\n");
    cmprintln!(
        "
{PRODUCT_NAME} Commands (slash-only; bare words go to the model):
  /exit, /quit  - Exit the program (Ctrl+D also works)
  /clear        - Clear conversation history (starts a new session)
  /resume       - Pick a stored session and continue it (also: --resume <id>)
  /compact      - Summarize the conversation to free context
  /remote       - /remote [on|off]  (remote channel status/toggle, alias /rc)
  /dsp          - /dsp on|off|status  (toggle skipping tool approvals)
  /effort       - /effort low|medium|high|off|status  (reasoning effort; also /reasoning, /think)
  /goal         - /goal <objective> | show | clear  (persistent goal injected each turn)
  /goal gate    - /goal gate add <cmd> | list | rm <n> | clear | auto on|off   (completion gates)
  /goal check   - run the completion gates and report pass/fail
  /keepgoing    - /keepgoing on|off|status
  /language     - UI language: /language (show) or /language Korean|ko|한국어
  /ab           - refusal edit: /ab (show) or /ab off|on|auto (auto = on only for the turn that accepts)
  /expand       - Print the last 3 tool outputs in full into the chat
  /expand N     - Same, but the last N outputs
  /export       - Save the conversation to a Markdown file (./codemason-export-<id>.md)
  /export PATH  - Same, to the given path
  /remote       - Phone/remote status: what's connected and what it can do
                  (also /remote-control, /rc)
  /help, /?     - Show this help message
  ! <command>   - Run a shell command directly (output shared with the model)
  Ctrl+O        - View ALL recent tool outputs in the pager (newest first)

Pager keys (Ctrl+O):
  Space   - page down
  b       - page up
  d / u   - half page down / up
  j / k   - one line down / up
  g / G   - jump to top / bottom
  /text   - search down
  ?text   - search up
  n / N   - next / previous match
  q       - quit

Available Tools:
{tool_names}
"
    );
}

// === Setup ===

/// Setup the CLI environment.
pub fn setup(cwd: &str, dangerously_skip_permissions: bool) {
    crate::utils::state::set_original_cwd(cwd);
    crate::utils::state::set_cwd(cwd);

    // Reflect the -dsp flag into the config so get_system_prompt() can raise
    // the autonomous execution layer. As a RUNTIME override, so it cannot be
    // written to disk by an unrelated save later in the run.
    crate::utils::config::set_runtime_override(
        "dangerously_skip_permissions",
        Value::Bool(dangerously_skip_permissions),
    );

    // Security check for --dangerously-skip-permissions.
    if dangerously_skip_permissions {
        #[cfg(unix)]
        {
            if unsafe { libc::getuid() } == 0 {
                eprintln!("Error: --dangerously-skip-permissions cannot be used with root/sudo privileges");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_key_bash_first_word() {
        let v = serde_json::json!({"command": "ls -la"});
        assert_eq!(permission_key("Bash", &v), Some(("Bash".into(), "ls".into())));
    }

    #[test]
    fn permission_key_compound_bash_is_none() {
        for cmd in ["ls && rm -rf /", "echo x > ~/.bashrc", "a | b", "x; y", "$(evil)"] {
            let v = serde_json::json!({ "command": cmd });
            assert_eq!(permission_key("Bash", &v), None, "{cmd}");
        }
    }

    #[test]
    fn strips_task_complete_sentinel() {
        assert_eq!(strip_completion_sentinel("done\nTASK_COMPLETE"), "done");
        assert_eq!(strip_completion_sentinel("no sentinel"), "no sentinel");
    }

    #[test]
    fn user_echo_elides_long_paste() {
        let text = (0..40).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let out = format_user_echo(&text);
        assert!(out.contains("lines elided"));
    }

    /// -dsp means skip permissions, with no exception; without it, a tool whose
    /// read leaves the machine must still be asked about.
    ///
    /// WebFetch and WebSearch change nothing locally, so they were auto-approved
    /// by the read-only branch — but the URL is model-controlled and Read has no
    /// path restriction, so reading a private key and posting it to an arbitrary
    /// host was two calls, neither of which prompted. The order of the two
    /// checks in has_permissions_to_use_tool is what keeps both rules true at
    /// once, and that order is what this pins.
    #[tokio::test]
    async fn dsp_skips_everything_and_web_tools_ask_otherwise() {
        let input = serde_json::json!({"url": "https://example.com", "prompt": "x"});

        let mut ctx = ToolUseContext::default();
        ctx.options.dangerously_skip_permissions = true;
        let d = has_permissions_to_use_tool("WebFetch", input.clone(), ctx).await;
        assert!(d.result, "-dsp must not prompt for anything");

        // Without -dsp the tool's own check is consulted and says ask, so the
        // read-only shortcut must NOT have returned allowed already.
        let tool = crate::tools::get_tool_by_name("WebFetch").expect("WebFetch");
        assert!(tool.is_read_only(), "still read-only, so plan mode is unchanged");
        let pre = tool.check_permissions(&input, &ToolUseContext::default()).await;
        assert_eq!(
            pre.behavior,
            crate::tools::base::PermissionBehavior::Ask,
            "a read-only tool that leaves the machine must ask"
        );

        let search = crate::tools::get_tool_by_name("WebSearch").expect("WebSearch");
        let pre = search
            .check_permissions(&serde_json::json!({"query": "x"}), &ToolUseContext::default())
            .await;
        assert_eq!(pre.behavior, crate::tools::base::PermissionBehavior::Ask);

        // An ordinary read-only tool is still waved through.
        let read = crate::tools::get_tool_by_name("Read").expect("Read");
        let pre = read
            .check_permissions(&serde_json::json!({"file_path": "/tmp/x"}), &ToolUseContext::default())
            .await;
        assert_eq!(pre.behavior, crate::tools::base::PermissionBehavior::Allow);
    }

    /// A session 'always' grant must never be broader than what the user was
    /// shown. None here means "no grant can cover this call", which is the
    /// strong answer.
    #[test]
    fn permission_key_scopes_what_a_grant_can_cover() {
        let cmd = |c: &str| serde_json::json!({ "command": c });

        // An interpreter gets no key at all: 'a' on `python build.py` must not
        // go on to auto-approve `python -c "<anything>"`, which contains no
        // character the compound guard looks for.
        for c in ["python build.py", "node x.js", "bash -c ls", "sudo apt update"] {
            assert_eq!(permission_key("Bash", &cmd(c)), None, "{c} must not be grantable");
        }
        // Including when spelled by absolute path.
        assert_eq!(permission_key("Bash", &cmd("/usr/bin/python3 a.py")), None);

        // Ordinary commands still key on the first word.
        assert_eq!(
            permission_key("Bash", &cmd("cat README.md")),
            Some(("Bash".into(), "cat".into()))
        );
        // PowerShell too — keying it on the tool name blanket-allowed every
        // later command of any content.
        assert_eq!(
            permission_key("PowerShell", &cmd("Get-ChildItem .")),
            Some(("PowerShell".into(), "Get-ChildItem".into()))
        );
        // Compound stays ungrantable.
        assert_eq!(permission_key("Bash", &cmd("ls && rm -rf /")), None);

        // File tools key on the path, so approving one file does not approve
        // every later one.
        let k1 = permission_key("Write", &serde_json::json!({"file_path": "/tmp/a.txt"}));
        let k2 = permission_key("Write", &serde_json::json!({"file_path": "/tmp/b.txt"}));
        assert!(k1.is_some() && k2.is_some());
        assert_ne!(k1, k2, "one Write grant must not cover a different path");
    }

    /// The early return is an AND, so text that is over the char cap but under
    /// the line cap falls into the eliding path — where the head/tail slices
    /// assume more than HEAD + TAIL lines exist. A single long line (a pasted
    /// Korean paragraph, a log line, a URL) has one, and this panicked on every
    /// submit before the length check went in.
    #[test]
    fn user_echo_survives_long_single_line() {
        let out = format_user_echo(&"가".repeat(2500));
        assert!(out.contains('가'));

        // Every line count from 1 up to and past the elision floor: the middle
        // of this range is where `lines.len() - HEAD - TAIL` used to underflow.
        for n in 1..=25 {
            let text = (0..n).map(|_| "가".repeat(200)).collect::<Vec<_>>().join("\n");
            let out = format_user_echo(&text);
            assert!(!out.is_empty(), "{n} lines produced nothing");
        }
    }

    /// The config-dir guard decides whether a write is really a request to run
    /// code, so every spelling of the same directory has to reach the same
    /// answer while a sibling stays out.
    #[test]
    fn config_path_guard_holds_across_spellings() {
        let sep = std::path::MAIN_SEPARATOR;
        let dir = crate::utils::config::CONFIG_DIR.to_string_lossy().into_owned();

        assert!(is_config_path(&dir));
        assert!(is_config_path(&format!("{dir}{sep}settings.json")));
        // A file that does not exist yet is the normal case for this guard:
        // goal-gates.txt and mcp.json are created on demand and executed after.
        assert!(is_config_path(&format!("{dir}{sep}zz_not_created_yet.txt")));
        assert!(is_config_path(&format!("{dir}{sep}hooks{sep}zz_new_hook.sh")));
        assert!(!is_config_path(""));

        // A sibling sharing the string prefix must stay outside.
        assert!(!is_config_path(&format!("{dir}-evil{sep}settings.json")));
    }

    /// An approval is for a FILE, so every way of naming that file has to
    /// reach the same key — otherwise 'always' on a path re-prompts the next
    /// time the model spells it differently, which on Windows was every time
    /// it switched between a relative and an absolute path.
    #[test]
    fn file_key_is_one_spelling_per_file() {
        let sep = std::path::MAIN_SEPARATOR;
        let cwd = crate::utils::state::get_cwd();
        let key = |p: &str| permission_key("Write", &serde_json::json!({ "file_path": p }));

        let relative = key("notes.txt");
        assert_eq!(relative, key(&format!("{cwd}{sep}notes.txt")), "relative vs absolute");
        assert_eq!(relative, key(&format!(".{sep}notes.txt")), "leading ./");
        assert_eq!(relative, key(&format!("sub{sep}..{sep}notes.txt")), "folded ..");
        assert_eq!(relative, key("notes.txt"), "stable across calls");

        // Different files still get different keys.
        assert_ne!(relative, key("other.txt"));
        assert_ne!(relative, key(&format!("sub{sep}notes.txt")));

        // Edit and NotebookEdit key the same way; the tool name keeps them apart.
        let e = permission_key("Edit", &serde_json::json!({"file_path": "notes.txt"}));
        assert_ne!(relative, e);
        assert_eq!(
            e.map(|(_, p)| p),
            relative.clone().map(|(_, p)| p),
            "same file, same path component"
        );
    }

    /// Case is not significant on Windows and must not change the verdict.
    /// The 8.3-alias half of this invariant is pinned in utils::state, where
    /// the resolution that sees through the alias lives.
    #[cfg(windows)]
    #[test]
    fn config_path_guard_ignores_case() {
        let sep = std::path::MAIN_SEPARATOR;
        let dir = crate::utils::config::CONFIG_DIR.to_string_lossy().into_owned();

        assert!(is_config_path(&dir.to_uppercase()));
        assert!(is_config_path(&format!("{}{sep}settings.json", dir.to_uppercase())));
        assert!(is_config_path(&format!("{}{sep}zz_not_created_yet.txt", dir.to_uppercase())));
        assert!(!is_config_path(&format!("{}-EVIL{sep}settings.json", dir.to_uppercase())));
    }

}
