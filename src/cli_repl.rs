//! The REPL half of the CLI — input router, turn loop, slash-command
//! dispatch, the two REPL modes and main().
//! Split out of cli.rs at the Python file's own section markers (`=== REPL
//! Loop ===` onward); everything above it lives in cli.rs.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::cli::{
    display_tool_use, get_token_warning, has_permissions_to_use_tool, pending_preview_lines,
    permission_callback, print_help, setup, user_echo_line, LOCAL_COMMAND_CAVEAT, PRODUCT_NAME,
    VERSION,
};
use crate::tools::base::Tool;
use crate::types::messages::{ContentBlock, ContentBlockParam};
use crate::types::query::{ContextOptions, InputRouter, Message, ToolUseContext};
use crate::utils::config::{clear_runtime_override, get_config, save_config, update_config};
use crate::utils::formatting::{
    colorize, colors, display_width, format_tokens_compact, plain_mode, render_markdown,
    safe_print, sanitize_terminal_text, summarize_tool_input, sym,
};
use crate::utils::i18n::{tr, trf, FmtArg};
use crate::utils::log::{debug_log, log_error};
use crate::utils::paste_store::{
    expand_pasted_refs, format_pasted_ref, pasted_num_lines, should_placeholder, PasteStore,
};
use crate::utils::prompt_engine::{EngineEvent, EngineHooks, Frag, PromptEngine};
use crate::utils::spinner::StreamingSpinner;
use crate::utils::surrogates::repair_surrogates;
use crate::{cmprint, cmprintln};

/// Segments in the input-line context gauge. Narrower than the 20-cell warning
/// bar: this one shares a line with what you are typing.
const CTX_BAR_WIDTH: usize = 8;

/// Cap the inline rewind selector height (CC MAX_VISIBLE_MESSAGES).
const REWIND_MAX_TARGETS: usize = 20;

/// What a queue item carries (the Python's (text, was_queued) tuple plus the
/// two sentinels).
enum QueueItem {
    /// A submitted line, and whether it was queued mid-turn (needs an echo when
    /// it is finally processed).
    Line(String, bool),
    /// A line was enqueued mid-turn for steering: wakes the boundary loop to
    /// reconcile a steering line the turn ended before it could inject.
    SteerWake,
    /// A background task finished while nothing was running: resume the turn
    /// without waiting for the user to press a key.
    TaskWake,
    /// Ctrl+D / /exit: quit.
    Exit,
}

/// Coordinates the interactive REPL's persistent prompt (the single owner of
/// stdin) with the turn loop: submitted-line queue, Esc/Ctrl+C interrupt
/// routing, tool-permission prompts, and prompt pausing for sub-UIs.
pub struct ReplInputRouter {
    tx: mpsc::UnboundedSender<QueueItem>,
    /// mid-turn typed lines waiting their turn — shown in the prompt block,
    /// NOT committed to scrollback, until processed.
    pending: Mutex<Vec<String>>,
    /// a turn is currently running
    processing: AtomicBool,
    /// the running turn was asked for from a remote client — its answer goes
    /// back there regardless of how quick it was
    turn_remote_origin: AtomicBool,
    /// line -> echo tag, for input that arrived from a remote client
    remote_lines: Mutex<BTreeMap<String, String>>,
    /// the CURRENT abort token
    abort: Mutex<CancellationToken>,
    /// live conversation list (shared with the turn loop); double-Esc rewind
    /// truncates it in place
    pub messages: Arc<Mutex<Vec<Message>>>,
    /// False until the first line is submitted (and again after /clear): the
    /// input rule is hidden then.
    conversation_started: AtomicBool,
    /// (sender, prompt_text) while a permission ask is pending
    permission: Mutex<Option<(tokio::sync::oneshot::Sender<String>, String)>>,
    paused: AtomicBool,
    /// Large/multi-line pastes are replaced by a [Pasted text #N] placeholder;
    /// the full text lives here and is spliced back in at submit time.
    pub paste_store: Mutex<PasteStore>,
    engine: Mutex<Option<Arc<PromptEngine>>>,
    /// The two-press bulk stop for background agents, shared by Ctrl+C (in
    /// hard_stop below) and the Ctrl+X Ctrl+K chord: the router is what both of
    /// them have in common.
    pub kill_agents: crate::utils::kill_agents_gesture::KillAgentsGesture,
}

impl ReplInputRouter {
    fn new(tx: mpsc::UnboundedSender<QueueItem>, messages: Arc<Mutex<Vec<Message>>>) -> Arc<Self> {
        Arc::new(ReplInputRouter {
            tx,
            pending: Mutex::new(Vec::new()),
            processing: AtomicBool::new(false),
            turn_remote_origin: AtomicBool::new(false),
            remote_lines: Mutex::new(BTreeMap::new()),
            abort: Mutex::new(CancellationToken::new()),
            messages,
            conversation_started: AtomicBool::new(false),
            permission: Mutex::new(None),
            paused: AtomicBool::new(false),
            paste_store: Mutex::new(PasteStore::new()),
            engine: Mutex::new(None),
            kill_agents: Default::default(),
        })
    }

    fn set_engine(&self, e: Arc<PromptEngine>) {
        *self.engine.lock() = Some(e);
    }

    fn engine(&self) -> Option<Arc<PromptEngine>> {
        self.engine.lock().clone()
    }

    pub fn set_abort(&self, token: CancellationToken) {
        *self.abort.lock() = token;
    }

    pub fn abort_token(&self) -> CancellationToken {
        self.abort.lock().clone()
    }

    pub fn is_processing(&self) -> bool {
        self.processing.load(Ordering::SeqCst)
    }

    fn set_processing(&self, v: bool) {
        self.processing.store(v, Ordering::SeqCst);
    }

    // --- Esc / Ctrl+C while a turn runs ---

    /// Interrupt the running turn (or deny a pending permission ask). Returns
    /// false when idle so the caller can treat the key normally.
    pub fn interrupt(&self) -> bool {
        let pending = self.permission.lock().take();
        if let Some((tx, _)) = pending {
            let _ = tx.send("n".into());
            return true;
        }
        if self.is_processing() {
            self.abort.lock().cancel();
            return true;
        }
        false
    }

    /// Ctrl+C: STOP — abort the turn AND drop everything queued behind it.
    ///
    /// Esc deliberately leaves the queue alone ("interject now"); Ctrl+C is the
    /// brake that must always land, so it discards the backlog and returns to
    /// an idle prompt.
    pub fn hard_stop(&self) -> bool {
        let stopped = self.interrupt();
        // The mpsc queue can't be drained without a receiver handle, so the
        // backlog lives in `pending` + the steering queue: clear both, and the
        // turn loop skips lines it no longer finds parked (see run_queue_item).
        let mut dropped = 0usize;
        let steer_dropped = crate::utils::steering_queue::drain_steering_messages().len();
        {
            let mut p = self.pending.lock();
            if !p.is_empty() {
                dropped = dropped.max(p.len()).max(steer_dropped);
                p.clear();
            } else if steer_dropped > 0 {
                dropped = dropped.max(steer_dropped);
            }
        }
        *DISCARD_BACKLOG.lock() = true;
        if dropped > 0 {
            cmprintln!(
                "{}",
                colorize(
                    &format!("  (discarded {dropped} queued message{})", if dropped != 1 { "s" } else { "" }),
                    colors::DIM
                )
            );
        }
        if stopped || dropped > 0 {
            // This press had work to do. Background agents are deliberately NOT
            // part of it: they are unlinked from the turn's abort scope on
            // purpose, and "killing agents from the main prompt stays a
            // deliberate gesture (chat:killAgents), not a side-effect of
            // cancelling a turn."
            return true;
        }

        // Nothing left to brake — and that is where the bulk stop lives, by the
        // same rule the source's cancel handler follows ("Priority 1: if
        // there's an active task running, cancel it first... Priority 2: pop
        // queue when Claude is idle"). Each press does the most urgent thing it
        // can and only falls through when there is nothing more urgent, which
        // is what makes mashing Ctrl+C to stop a turn unable to reach the
        // agents: the presses that brake never arm.
        //
        // Ctrl+C is the gesture's second entry point (the first is the source's
        // Ctrl+X Ctrl+K chord, which nobody will find in a REPL with no
        // shortcut footer) because it is the key someone reaches for when they
        // want everything to stop, and the alternative was closing the session.
        // It stays two presses: the first says what is still running, the
        // second ends it. This is also exactly where "(Use /exit or Ctrl+D to
        // quit)" used to be the only answer to a Ctrl+C that appeared to do
        // nothing.
        use crate::utils::kill_agents_gesture::Press;
        let (press, note) = self.kill_agents.press("Ctrl+C");
        if let Some(note) = note {
            let style = if press == Press::Armed { colors::DIM } else { colors::CYAN };
            cmprintln!("{}", colorize(&format!("  {note}"), style));
        }
        matches!(press, Press::Armed | Press::Killed)
    }

    // --- tool permission prompts ---

    pub fn permission_pending(&self) -> bool {
        self.permission.lock().is_some()
    }

    pub fn permission_prompt_text(&self) -> String {
        self.permission
            .lock()
            .as_ref()
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| "> ".into())
    }

    /// '(remote)', or '(remote · 3h ago)' when the message waited around.
    ///
    /// `source` names the transport the line came in on. Anything that is not
    /// the phone says so instead of borrowing its label: a line the local
    /// `inject` command queued shows as '(inject)', because presenting it as a
    /// phone message makes the user doubt a history they did not write.
    fn remote_tag(age: f64, source: &str) -> String {
        let what = if source.is_empty() || source == "telegram" { "remote" } else { source };
        if age < 60.0 {
            format!("  ({what})")
        } else if age < 3600.0 {
            format!("  ({what} · {}m ago)", (age / 60.0) as i64)
        } else if age < 86400.0 {
            format!("  ({what} · {}h ago)", (age / 3600.0) as i64)
        } else {
            format!("  ({what} · {}d ago)", (age / 86400.0) as i64)
        }
    }

    /// Queue a line that arrived from a remote client (phone).
    ///
    /// Same queue, same rule as a typed line. Echoing follows the local rules
    /// exactly: a message arriving mid-turn parks in `pending` and is echoed
    /// once at its boundary; an idle one is echoed straight away.
    pub fn submit_remote(&self, text: &str, age: f64, source: &str) {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        let tag = Self::remote_tag(age, source);
        self.remote_lines.lock().insert(text.clone(), tag.clone());
        // A line from the phone starts the conversation just as much as a typed one.
        self.conversation_started.store(true, Ordering::SeqCst);
        if self.is_processing() {
            // Mid-turn: STEER it into the running turn at its next model step.
            self.pending.lock().push(text.clone());
            self.submit_steering(&text);
        } else {
            cmprintln!();
            cmprintln!("{}", user_echo_line(&text));
            cmprintln!("{}", colorize(&tag, colors::DIM));
            let _ = self.tx.send(QueueItem::Line(text, false)); // already echoed
        }
        if let Some(e) = self.engine() {
            e.repaint();
        }
    }

    /// Ask the user a question through the persistent prompt: the prompt
    /// message switches to `prompt_text` and a single y/n/a keypress answers.
    pub async fn request_permission(&self, prompt_text: &str) -> String {
        let (tx, rx) = tokio::sync::oneshot::channel();
        // Text the user was composing for the NEXT message is stashed on entry
        // and restored afterwards — mid-turn typing must never leak into the ask.
        let stash = self.engine().map(|e| e.buffer_text()).unwrap_or_default();
        if let Some(e) = self.engine() {
            e.set_buffer_text("");
        }
        *self.permission.lock() = Some((tx, prompt_text.to_string()));
        if let Some(e) = self.engine() {
            e.repaint(); // show the question immediately
        }
        let answer = rx.await.unwrap_or_else(|_| "n".into());
        *self.permission.lock() = None;
        if let Some(e) = self.engine() {
            e.set_buffer_text(&stash);
            e.repaint();
        }
        answer
    }

    /// Route a submitted key to a pending permission ask. True if consumed.
    pub fn deliver(&self, line: &str) -> bool {
        let pending = self.permission.lock().take();
        match pending {
            Some((tx, _)) => {
                let _ = tx.send(line.to_string());
                true
            }
            None => false,
        }
    }

    // --- mid-turn steering (interject into the RUNNING turn) ---

    /// A line typed while a turn is running: enqueue it for injection into the
    /// current turn at its next model step. The wake marker makes the boundary
    /// loop reconcile the case where the turn ends before another step.
    pub fn submit_steering(&self, text: &str) {
        crate::utils::steering_queue::enqueue_steering_message(text);
        let _ = self.tx.send(QueueItem::SteerWake);
    }

    pub fn submit_line(&self, text: String, was_queued: bool) {
        let _ = self.tx.send(QueueItem::Line(text, was_queued));
    }

    pub fn submit_exit(&self) {
        let _ = self.tx.send(QueueItem::Exit);
    }

    // --- prompt pausing (a sub-UI needs the terminal) ---

    pub async fn pause_prompt(&self) {
        self.paused.store(true, Ordering::SeqCst);
        if let Some(e) = self.engine() {
            e.set_hidden(true);
            // Real handshake (cli.py:1908-1915): the engine loop RELEASES
            // terminal input and confirms before this returns, so the sub-UI's
            // input_events::acquire() is never contended by the engine.  The
            // old 20ms sleep only hoped the loop had noticed.
            e.suspend().await;
        }
    }

    pub fn resume_prompt(&self) {
        self.paused.store(false, Ordering::SeqCst);
        if let Some(e) = self.engine() {
            e.unsuspend();
            e.set_hidden(false);
        }
    }

    /// Pause, and hand back a guard that resumes when it drops.
    ///
    /// A `resume_prompt()` written at the end of the block only runs when the
    /// block is left by falling off the end. It is skipped by an early return,
    /// by a panic unwinding through, and by the future being cancelled — and
    /// the cost of skipping it is the worst failure this REPL has: the engine
    /// stays suspended, so the prompt is parked and every keystroke is
    /// discarded for the rest of the process, with the screen looking
    /// completely normal. That is indistinguishable from a hang.
    ///
    /// The sub-UIs this wraps (the rewind picker, the resume picker, the
    /// output pager, the question picker) are exactly the code most likely to
    /// panic: they index into option lists, measure widths and drive the
    /// terminal. Python fixed the same shape by moving the resume into a
    /// `finally`; Drop is this language's version of that, and unlike a
    /// trailing call it cannot be forgotten by the next caller.
    pub fn pause_prompt_scoped(self: &Arc<Self>) -> impl std::future::Future<Output = PromptPauseGuard> + '_ {
        let me = self.clone();
        async move {
            me.pause_prompt().await;
            PromptPauseGuard(me)
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }
}

/// Resumes the prompt on drop — see `pause_prompt_scoped`.
pub struct PromptPauseGuard(Arc<ReplInputRouter>);

impl Drop for PromptPauseGuard {
    fn drop(&mut self) {
        self.0.resume_prompt();
    }
}

impl InputRouter for ReplInputRouter {
    fn prompt_line(&self, _prompt: &str) -> Option<String> {
        // Permission prompts go through request_permission (async); this
        // synchronous entry point is unused in the Rust port.
        None
    }

    /// query() just injected a mid-turn steering line into the running turn.
    /// Drop it from the parked-input display and echo it to scrollback, so the
    /// transcript shows what steered the turn, and when.
    fn note_steer_injected(&self, text: &str) {
        {
            let mut p = self.pending.lock();
            if let Some(pos) = p.iter().position(|x| x == text) {
                p.remove(pos);
            }
        }
        cmprintln!();
        cmprintln!("{}", user_echo_line(text));
        // Take the value, THEN print. In `if let Some(x) = m.lock().remove(..)`
        // the guard is the scrutinee's temporary and lives for the whole block,
        // so the print ran holding `remote_lines` — and a print goes term_out ->
        // the prompt engine -> `paint` -> stdout. Printing under any lock makes
        // that lock an edge into the painting order; see with_out() in
        // prompt_engine for what that cost on 2026-09-29.
        let tag = self.remote_lines.lock().remove(text);
        if let Some(tag) = tag {
            cmprintln!("{}", colorize(&tag, colors::DIM));
        }
        if let Some(e) = self.engine() {
            e.repaint();
        }
    }
}

/// Set by hard_stop so the boundary loop drops queued lines that were
/// discarded (the mpsc queue itself can't be drained from the sender side).
static DISCARD_BACKLOG: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

/// Visible user turns the conversation can rewind to (oldest→newest).
fn enumerate_rewind_targets(messages: &[Message]) -> Vec<(usize, String, String)> {
    let mut targets: Vec<(usize, String, String)> = Vec::new();
    for (i, entry) in messages.iter().enumerate() {
        let Message::User(u) = entry else { continue };
        if u.is_meta {
            continue;
        }
        let mut full: Option<String> = None;
        for block in u.message.content.as_blocks() {
            let ContentBlockParam::Text { text, .. } = block else { continue };
            if text.is_empty() {
                continue;
            }
            static BASH_INPUT_RE: Lazy<regex::Regex> =
                Lazy::new(|| regex::Regex::new(r"(?s)^<bash-input>(.*?)</bash-input>").unwrap());
            if let Some(m) = BASH_INPUT_RE.captures(&text) {
                full = Some(format!("! {}", m[1].trim()));
                break;
            }
            // Mirrors extract_first_prompt's skip rules: control-XML payloads
            // are not user-selectable prompts.
            static SKIP_RE: Lazy<regex::Regex> = Lazy::new(|| {
                regex::Regex::new(r"^(?:\s*<[a-z][\w-]*[\s>]|\[Request interrupted by user[^\]]*\])").unwrap()
            });
            if SKIP_RE.is_match(&text) {
                continue;
            }
            full = Some(text.clone());
            break;
        }
        let Some(full) = full else { continue };
        let mut preview = full.replace('\n', " ").trim().to_string();
        if preview.chars().count() > 100 {
            preview = preview.chars().take(100).collect::<String>().trim().to_string() + "…";
        }
        targets.push((i, preview, full));
    }
    if targets.len() > REWIND_MAX_TARGETS {
        targets.drain(..targets.len() - REWIND_MAX_TARGETS);
    }
    targets
}

/// Double-Esc-on-empty → pick a previous user message and rewind the
/// conversation to it (CC's 'Restore conversation').
async fn run_message_selector(router: Arc<ReplInputRouter>, ctx: ToolUseContext) {
    let targets = {
        let messages = router.messages.lock();
        if messages.is_empty() {
            return;
        }
        enumerate_rewind_targets(&messages)
    };
    if targets.is_empty() {
        return;
    }
    let _pause = router.pause_prompt_scoped().await;
    let options: Vec<String> = targets.iter().map(|(_, p, _)| p.clone()).collect();
    let picked = crate::utils::arrow_picker::arrow_select(
        "Rewind conversation to a previous message:",
        &options,
        Default::default(),
    )
    .await;
    if let Some(picked) = picked {
        let (idx, preview, full) = targets[picked[0]].clone();
        // In-place truncation: drop the chosen user turn and everything after
        // it. Cutting at a user boundary never leaves a dangling assistant
        // tool_use without its tool_result.
        router.messages.lock().truncate(idx);
        ctx.read_file_state.lock().clear();
        if let Some(e) = router.engine() {
            e.set_buffer_text(&full);
        }
        cmprintln!("↩ Rewound conversation to: {preview}");
    }
}

/// `! cmd` bash passthrough — port of processUserInput/processBashCommand.
async fn process_bash_command(command: &str, messages: &Arc<Mutex<Vec<Message>>>, ctx: &ToolUseContext) {
    use crate::tools::base::{Tool as _, ToolResult};
    use crate::utils::messages::create_user_message_text;

    fn esc(s: &str) -> String {
        // escapeXml: & < > (quote=False)
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
    }

    let mut new_msgs: Vec<Message> = Vec::new();

    // Caveat once per conversation (createSyntheticUserCaveatMessage). The scan
    // is stateless, so /clear and resumed histories behave correctly.
    let has_caveat = messages.lock().iter().any(|m| {
        let Message::User(u) = m else { return false };
        u.message
            .content
            .as_blocks()
            .first()
            .map(|b| match b {
                ContentBlockParam::Text { text, .. } => text.contains("<local-command-caveat>"),
                _ => false,
            })
            .unwrap_or(false)
    });
    if !has_caveat {
        let mut cav = create_user_message_text(LOCAL_COMMAND_CAVEAT);
        cav.is_meta = true;
        new_msgs.push(Message::User(cav));
    }
    new_msgs.push(Message::User(create_user_message_text(&format!(
        "<bash-input>{command}</bash-input>"
    ))));

    // ESC-interruptible: mark processing so the Esc binding fires.
    let router = router_of(ctx);
    if let Some(r) = &router {
        r.set_processing(true);
    }

    let bash = crate::tools::bash_tool::BashTool;
    let mut stream = bash.call(
        serde_json::json!({"command": command}),
        ctx.clone(),
        Some(permission_callback()),
    );
    let mut data: Option<Value> = None;
    let mut failed: Option<String> = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(ToolResult::Result { data: d, .. }) => data = d,
            Ok(_) => {}
            Err(e) => failed = Some(e.message),
        }
    }

    match failed {
        Some(err) => {
            new_msgs.push(Message::User(create_user_message_text(&format!(
                "<bash-stderr>Command failed: {}</bash-stderr>",
                esc(&err)
            ))));
            cmprintln!("{}", colorize(&format!("Command failed: {err}"), colors::RED));
        }
        None => {
            let d = data.unwrap_or(Value::Null);
            let stdout = d.get("stdout").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let stderr = d.get("stderr").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let interrupted = d.get("interrupted").and_then(|v| v.as_bool()).unwrap_or(false);
            if interrupted {
                new_msgs.push(Message::User(create_user_message_text(
                    crate::utils::messages::INTERRUPT_MESSAGE,
                )));
                cmprintln!("{}", colorize("[Interrupted]", colors::DIM));
            } else {
                // Same persist mapping as model-initiated Bash.
                let mapped = crate::utils::tool_result_storage::process_tool_result_content(
                    &stdout,
                    "Bash",
                    &uuid::Uuid::new_v4().to_string(),
                    bash.max_result_size_chars(),
                );
                new_msgs.push(Message::User(create_user_message_text(&format!(
                    "<bash-stdout>{mapped}</bash-stdout><bash-stderr>{}</bash-stderr>",
                    esc(&stderr)
                ))));
                if !mapped.is_empty() && !is_tool_result_marker_local(&mapped) {
                    cmprintln!("{mapped}");
                }
                if !stderr.is_empty() {
                    cmprintln!("{}", colorize(&stderr, colors::RED));
                }
            }
        }
    }

    if let Some(r) = &router {
        r.set_processing(false);
    }

    {
        let mut m = messages.lock();
        m.extend(new_msgs);
        crate::utils::session_storage::record_transcript(&m, false);
    }
}

fn is_tool_result_marker_local(text: &str) -> bool {
    text.starts_with('(') && text.ends_with("completed with no output)")
}

/// The router bound to this context, if any.
pub(crate) fn router_of(ctx: &ToolUseContext) -> Option<Arc<ReplInputRouter>> {
    ctx.input_router.as_ref().and_then(|r| {
        // The router is stored as Arc<dyn InputRouter>; downcast is not
        // available for trait objects without Any, so the REPL keeps its own
        // handle in the module-level slot instead.
        let _ = r;
        ACTIVE_ROUTER.lock().clone()
    })
}

static ACTIVE_ROUTER: Lazy<Mutex<Option<Arc<ReplInputRouter>>>> = Lazy::new(|| Mutex::new(None));

/// `/remote` — is the phone connected, and what can it do.
async fn print_remote_status() {
    let cfg = get_config();
    cmprintln!();
    // Off wholesale (--plain) reads as "no credentials" otherwise, and the fix
    // for that — go set up a bot you already set up — is the wrong advice.
    let off = crate::remote::notify::off_reason();
    if !off.is_empty() {
        cmprintln!(
            "{}",
            colorize(
                &trf("Remote channel is off  ({reason})", &[("reason", FmtArg::Str(off))]),
                colors::BOLD
            )
        );
        cmprintln!(
            "{}",
            colorize(
                &tr("  This mode is for dumb consoles, so the phone channel is unused.\n  Your settings are intact; run without --plain/--ascii to turn it back on."),
                colors::DIM
            )
        );
        cmprintln!();
        return;
    }
    if !crate::remote::permissions::is_available() {
        // Asked where it is; answer with the file, not with a description of a
        // file. Creating it here too covers a session already running.
        let _ = crate::remote::adapters::telegram::ensure_credentials_scaffold();
        let path = crate::remote::adapters::telegram::credentials_path();
        cmprintln!("{}", colorize(&tr("No remote connection"), colors::BOLD));
        cmprintln!(
            "{}",
            colorize(
                &trf(
                    "  No Telegram bot token.\n  Fill in the slots in {path}:\n    TELEGRAM_BOT_TOKEN=...   (one line per bot; @BotFather issues them)\n    TELEGRAM_CHAT_ID=...     (your own numeric id; @userinfobot tells you)\n  (Environment variables work too. Restart the session afterwards.)",
                    &[("path", FmtArg::Str(path.to_string_lossy().into_owned()))]
                ),
                colors::DIM
            )
        );
        cmprintln!();
        return;
    }

    let sessions = crate::remote::inbox::live_sessions();
    let me = crate::utils::state::get_session_id();
    let mine = sessions.iter().find(|s| s.session_id == me);
    // Who actually holds the poll lock decides whether THIS session can be
    // reached — invisible otherwise, and it explains the one confusing case.
    let holder = crate::remote::adapters::telegram::poll_lock_holder();
    let listening = !holder.is_empty() && holder == std::process::id().to_string();

    // Which bot this instance claimed, named as the phone shows it.
    let bot = crate::remote::adapters::telegram::describe_claimed_bot().await;

    cmprintln!("{}", colorize(&tr("Remote connected  (Telegram)"), colors::BOLD));
    if !bot.is_empty() {
        cmprintln!(
            "{}",
            trf("  Bot          : {name}   (rename: /rc <name>)", &[("name", FmtArg::Str(bot))])
        );
    }
    cmprintln!(
        "{}",
        trf(
            "  This session : {label}{state}",
            &[
                ("label", FmtArg::Str(mine.map(|m| m.label.clone()).unwrap_or_else(|| tr("(unregistered)")))),
                ("state", FmtArg::Str(if listening { tr("  <- listening") } else { tr("  (another session is listening)") })),
            ]
        )
    );
    if sessions.len() > 1 {
        let others: Vec<String> = sessions
            .iter()
            .filter(|s| s.session_id != me)
            .map(|s| s.label.clone())
            .collect();
        cmprintln!("{}", trf("  Others       : {names}", &[("names", FmtArg::Str(others.join(", ")))]));
    }
    cmprintln!();
    cmprintln!("{}", colorize(&tr("What you can do from the phone"), colors::BOLD));
    cmprintln!("{}", tr("  Send a message   queued for this session (mid-turn ones wait their turn)"));
    cmprintln!("{}", tr("  Approve tools    risky tools raise buttons; first answer wins"));
    cmprintln!("{}", tr("  Answer questions choices arrive as buttons (free text stays local)"));
    cmprintln!("{}", tr("  /start           lists the running sessions"));
    cmprintln!("{}", tr("  Reply            replying to an alert routes to that session"));
    cmprintln!();
    cmprintln!("{}", colorize(&tr("Settings"), colors::BOLD));
    let after = cfg.remote_notify_after;
    cmprintln!(
        "{}",
        trf(
            "  Notify after     only turns longer than {secs:g}s{every}   CODEMASON_NOTIFY_AFTER",
            &[
                ("secs", FmtArg::Float(after)),
                ("every", FmtArg::Str(if after == 0.0 { tr(" (0 = every turn)") } else { String::new() })),
            ]
        )
    );
    cmprintln!(
        "{}",
        trf(
            "  Approval wait    re-asks every {mins:g} min   CODEMASON_ASK_TIMEOUT",
            &[("mins", FmtArg::Float(cfg.remote_ask_timeout / 60.0))]
        )
    );
    cmprintln!(
        "{}",
        colorize(&tr("  A turn started from the phone always gets its answer."), colors::DIM)
    );
    cmprintln!();
}

/// True if `name` is a model-invocable skill — so `/name` is the skill
/// shorthand, not an unknown command. Never raises.
fn is_skill_command(name: &str) -> bool {
    crate::utils::skills::get_skill(name).is_some()
}

/// What dispatch_builtin decided.
pub enum Dispatch {
    /// quit the REPL
    Exit,
    /// command consumed
    Handled,
    /// ordinary prompt — send it to the model unchanged
    Prompt,
}

/// Slash-command routing + built-in dispatch, shared by both REPL loops.
///
/// Routing (handlePromptSubmit.ts): built-ins are invoked ONLY with a leading
/// slash; bare command words are ordinary prompts. Only the FIRST token counts.
/// An unknown single-word /command is rejected instead of being sent to the
/// model (which hallucinated a plausible help page for /help).
pub async fn dispatch_builtin(
    user_input: &str,
    messages: &Arc<Mutex<Vec<Message>>>,
    ctx: &ToolUseContext,
    goal_kickoff: &Mutex<Option<String>>,
) -> Dispatch {
    // Local-operator affordances. `!` runs a shell command with NO permission
    // ask (it is user-initiated by definition) and `/dsp` changes the permission
    // mode itself and persists it. Neither is a decision the remote channel gets
    // to make: remote::permissions deliberately declines to offer even an
    // "always allow" button there, reasoning that a phone answer is a reflex tap
    // — and a remote `/dsp on` would hand out exactly the blanket permission
    // that module refused, for this session and every future one. Remote input
    // is still delivered to the model as an ordinary prompt; it just cannot
    // drive these two directly.
    //
    // Origin is read here rather than passed in, because a remote line reaches
    // the queue path as well as the steering path and missing one call site
    // would silently reopen this.
    let from_remote = ACTIVE_ROUTER
        .lock()
        .as_ref()
        .map(|r| r.remote_lines.lock().contains_key(user_input))
        .unwrap_or(false);
    if from_remote {
        let first = user_input.split_whitespace().next().unwrap_or("");
        if user_input.starts_with('!') || first.eq_ignore_ascii_case("/dsp") {
            cmprintln!("  Remote input cannot run ! commands or change permission mode. Send it as a normal message instead.");
            return Dispatch::Handled;
        }
    }

    // `! cmd` = bash passthrough (processUserInput mode === 'bash').
    if let Some(rest) = user_input.strip_prefix('!') {
        let cmd = rest.trim();
        if cmd.is_empty() {
            cmprintln!("Usage: ! <command>   (runs in the session shell; the model sees the output on your next prompt)");
            return Dispatch::Handled;
        }
        process_bash_command(cmd, messages, ctx).await;
        return Dispatch::Handled;
    }

    let mut user_input = user_input.to_string();
    if user_input == "/?" {
        user_input = "/help".into();
    } else if user_input == "/" {
        print_help(Some(&ctx.options.tools));
        return Dispatch::Handled;
    }

    let first = user_input.split_whitespace().next().unwrap_or("").to_string();
    static CMD_RE: Lazy<regex::Regex> =
        Lazy::new(|| regex::Regex::new(r"^/[A-Za-z][A-Za-z0-9_-]*$").unwrap());
    if !CMD_RE.is_match(&first) {
        return Dispatch::Prompt;
    }
    let bare = first[1..].to_lowercase();
    if !crate::cli::BUILTIN_COMMANDS.contains(&bare.as_str()) {
        // Not a built-in: if it names a skill, it's the `/skill-name` shorthand.
        if is_skill_command(&first[1..]) {
            // Runnable, but possibly invisible to the model: the skill listing
            // rides in the tool block, which is fixed for the session. Say so
            // rather than letting the model quietly act as if the skill did not
            // exist.
            let sname = &first[1..];
            if !crate::utils::skills::is_session_skill(sname) {
                cmprintln!(
                    "{}",
                    colorize(
                        &trf(
                            "  [!] Skill '{name}' appeared after this session started, so the model's skill list does not include it (running it by name still works).\n      Refreshing that list mid-session re-sends every tool definition, which makes the next reply re-read the entire conversation — minutes on a long one. Restart the session to have it listed.",
                            &[("name", FmtArg::Str(sname.to_string()))]
                        ),
                        colors::YELLOW
                    )
                );
            }
            return Dispatch::Prompt;
        }
        cmprintln!("Unknown command: {first} (see /help)");
        return Dispatch::Handled;
    }
    // Drop the leading '/', not the first BYTE. `first` comes from
    // split_whitespace, so leading whitespace still reaches here: "  /help"
    // passed the check above and then sliced the space off instead of the
    // slash, leaving cmd = "/help", which matches no arm — the command
    // silently did nothing. Worse, a multi-byte leading space (U+00A0, which
    // char::is_whitespace accepts and which arrives from a paste) made [1..]
    // land inside that character and panic.
    let body = user_input.trim_start().strip_prefix('/').unwrap_or("").to_string();
    let cmd = body.split_whitespace().next().unwrap_or("").to_lowercase();

    match cmd.as_str() {
        "exit" | "quit" => Dispatch::Exit,

        "clear" => {
            messages.lock().clear();
            // The repeated-call guard counts within a conversation.
            crate::query::reset_tool_repeat_history();
            // /clear starts a NEW session: the old transcript stays resumable.
            crate::utils::session_storage::switch_session(None);
            // Record a clear boundary so a later `-c` doesn't resume it.
            crate::utils::session_storage::record_clear_boundary();
            if let Some(r) = ACTIVE_ROUTER.lock().clone() {
                r.conversation_started.store(false, Ordering::SeqCst);
            }
            cmprintln!("Conversation cleared.");
            Dispatch::Handled
        }

        "resume" => {
            let router = ACTIVE_ROUTER.lock().clone();
            let _pause = match &router {
                Some(r) => Some(r.pause_prompt_scoped().await),
                None => None,
            };
            let picked = crate::utils::resume_picker::pick_session().await;
            if let Some(picked) = picked {
                match crate::utils::session_storage::load_conversation_for_resume(&picked.session_id) {
                    Ok(restored) => {
                        crate::utils::session_storage::adopt_resumed_session(&picked.session_id);
                        let typed = crate::utils::messages::messages_from_transcript(&restored);
                        {
                            let mut m = messages.lock();
                            m.clear();
                            m.extend(typed);
                        }
                        // A resume is a NEW process, so the UI-language evidence
                        // count starts empty while the answer is in the transcript.
                        crate::cli::observe_restored_turns_pub(&restored);
                        ctx.read_file_state.lock().clear();
                        // Rendered under the lock, printed after it drops:
                        // printing while holding `messages` deadlocks against
                        // the engine's repaint (see render_resume_banner).
                        let banner = {
                            let m = messages.lock();
                            crate::cli::render_resume_banner_pub(&m)
                        };
                        cmprint!("{banner}");
                    }
                    Err(e) => cmprintln!("Could not resume: {e}"),
                }
            }
            Dispatch::Handled
        }

        "ab" => {
            use crate::services::ab::{self, Mode};
            let want = body.splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim().to_string();
            if want.is_empty() {
                let served = match ab::query().await {
                    Some((scale, consistent)) => {
                        let state = if scale > 0.0 { "on (abliterated)" } else { "off (restored)" };
                        if consistent {
                            format!("server: {state}")
                        } else {
                            format!("server: {state} — WORKERS DISAGREE")
                        }
                    }
                    None => "server: no /ab endpoint".to_string(),
                };
                cmprintln!(
                    "/ab mode: {}\n{served}\n\n\
                     off   leave the server alone\n\
                     on    pin the refusal edit on — for measurement\n\
                     auto  on for the turn's first request, off as soon as it answers (default)\n\n\
                     The edit also costs the model its ability to say something does not exist\n\
                     (4/60 fabrications off, 18-22/60 on), so auto keeps it off for every tool\n\
                     result, link and diff — only the deciding turn gets it.",
                    ab::mode().as_str()
                );
                return Dispatch::Handled;
            }
            let Some(m) = Mode::parse(&want) else {
                cmprintln!("Unknown mode: {want}   (off | on | auto)");
                return Dispatch::Handled;
            };
            match ab::set_mode(m).await {
                Ok(()) => cmprintln!("/ab mode: {}", m.as_str()),
                Err(e) => cmprintln!("/ab mode: {} — server not updated: {e}", m.as_str()),
            }
            Dispatch::Handled
        }

        "language" | "lang" => {
            use crate::utils::i18n::{active, normalize, set_language, LANGUAGES};
            let want = body.splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim().to_string();
            let names: String = LANGUAGES
                .iter()
                .map(|(code, name, _)| format!("{name} ({code})"))
                .collect::<Vec<_>>()
                .join(", ");
            if want.is_empty() {
                let now = active();
                let label = now
                    .and_then(|c| LANGUAGES.iter().find(|(code, _, _)| *code == c).map(|(_, n, _)| *n))
                    .unwrap_or("English");
                cmprintln!("UI language: {label}\nAvailable: {names}\nUsage: /language <name>   (Korean, ko, ko_KR, 한국어 — all work)");
                return Dispatch::Handled;
            }
            let Some(code) = normalize(Some(&want)) else {
                cmprintln!("Unknown language: {want}\nAvailable: {names}");
                return Dispatch::Handled;
            };
            set_language(Some(code));
            update_config(|c| c.ui_language = Some(code.to_string()));
            let saved = save_config(None);
            let label = LANGUAGES
                .iter()
                .find(|(c, _, _)| *c == code)
                .map(|(_, n, _)| *n)
                .unwrap_or("English");
            if saved {
                cmprintln!("UI language: {label}");
            } else {
                cmprintln!("UI language: {label} ({})", crate::utils::config::saved_label(false));
            }
            Dispatch::Handled
        }

        "compact" => {
            let instr = body.splitn(2, char::is_whitespace).nth(1).map(|s| s.to_string());
            if messages.lock().is_empty() {
                cmprintln!("Nothing to compact.");
                return Dispatch::Handled;
            }
            let spin = StreamingSpinner::new(false);
            spin.start(Some("Compacting"));
            let snapshot = messages.lock().clone();
            let s1 = spin.clone();
            let s2 = spin.clone();
            let res = crate::services::compaction::compact_conversation(
                &snapshot,
                ctx,
                instr.as_deref(),
                false,
                Some(Arc::new(move |n| s1.update(n))),
                Some(Arc::new(move |n| s2.update_thinking(n))),
            )
            .await;
            spin.stop();
            match res {
                Ok(r) => {
                    {
                        let mut m = messages.lock();
                        m.clear();
                        m.extend(r.messages);
                    }
                    ctx.read_file_state.lock().clear(); // summary lacks file contents
                    cmprintln!("  ⎿  Compacted — conversation summarized.\n");
                }
                Err(e) => cmprintln!("Compaction failed: {e}"),
            }
            Dispatch::Handled
        }

        "expand" => {
            // Print the LAST FEW tool outputs in full, inline into the chat.
            if crate::cli::recent_tool_outputs_is_empty() {
                cmprintln!("No tool output to expand yet.");
                return Dispatch::Handled;
            }
            let parts: Vec<&str> = body.split_whitespace().collect();
            let mut n = crate::cli::EXPAND_DEFAULT_COUNT_PUB;
            if parts.len() > 1 {
                match parts[1].parse::<usize>() {
                    Ok(v) => n = v.max(1),
                    Err(_) => {
                        cmprintln!(
                            "Usage: /expand [N]   (print the last N tool outputs; default {})",
                            crate::cli::EXPAND_DEFAULT_COUNT_PUB
                        );
                        return Dispatch::Handled;
                    }
                }
            }
            crate::cli::expand_recent_outputs(n);
            Dispatch::Handled
        }

        "export" => {
            if messages.lock().is_empty() {
                cmprintln!("Nothing to export yet.");
                return Dispatch::Handled;
            }
            let arg = body.splitn(2, char::is_whitespace).nth(1).map(|s| s.trim().to_string());
            let path = match arg.filter(|a| !a.is_empty()) {
                Some(p) => {
                    if let Some(rest) = p.strip_prefix("~/") {
                        dirs::home_dir().map(|h| h.join(rest)).unwrap_or_else(|| p.clone().into())
                    } else {
                        std::path::PathBuf::from(p)
                    }
                }
                None => {
                    let sid: String = crate::utils::state::get_session_id().chars().take(8).collect();
                    std::path::PathBuf::from(crate::utils::state::get_cwd())
                        .join(format!("codemason-export-{sid}.md"))
                }
            };
            let md = crate::cli::export_conversation_md_pub(&messages.lock());
            match std::fs::write(&path, md) {
                Ok(_) => cmprintln!(
                    "  ⎿  Exported {} messages → {}",
                    messages.lock().len(),
                    path.display()
                ),
                Err(e) => cmprintln!("Export failed: {e}"),
            }
            Dispatch::Handled
        }

        "help" => {
            print_help(Some(&ctx.options.tools));
            Dispatch::Handled
        }

        "remote" | "remote-control" | "rc" => {
            dispatch_remote(&body).await;
            Dispatch::Handled
        }

        "dsp" => {
            // Approvals, switched mid-session. The permission gate re-reads the
            // value on every tool call, so this lands on the very next one.
            let parts: Vec<&str> = body.split_whitespace().collect();
            let arg = parts.get(1).map(|s| s.to_lowercase()).unwrap_or_else(|| "status".into());
            match arg.as_str() {
                "on" | "off" => {
                    let val = arg == "on";
                    set_session_dsp(val);
                    // Deliberate, and the message says it was saved — drop any
                    // shadow a startup flag left on this field.
                    clear_runtime_override("dangerously_skip_permissions");
                    update_config(|c| c.dangerously_skip_permissions = val);
                    let saved = crate::utils::config::saved_label(save_config(None));
                    cmprintln!(
                        "{}{} {}",
                        trf("Skip approvals {state} ({saved}). ", &[("state", FmtArg::Str(if val { "on".into() } else { "off".into() })), ("saved", FmtArg::Str(saved))]),
                        if val { tr("Tools run without asking.") } else { tr("Every tool call asks for approval.") },
                        tr("Plan approval (ExitPlanMode) and questions (AskUserQuestion) still ask either way.")
                    );
                }
                "status" => {
                    let eff = session_dsp();
                    cmprintln!(
                        "{}{}",
                        trf("Skip approvals is {state}.", &[("state", FmtArg::Str(if eff { "on".into() } else { "off".into() }))]),
                        if eff { tr("  Tools are auto-approved.") } else { tr("  Every tool call asks.") }
                    );
                }
                _ => cmprintln!("Usage: /dsp on | off | status"),
            }
            Dispatch::Handled
        }

        "goal" => {
            dispatch_goal(&body, goal_kickoff);
            Dispatch::Handled
        }

        "effort" | "reasoning" | "think" => {
            let parts: Vec<&str> = body.split_whitespace().collect();
            let arg = parts.get(1).map(|s| s.to_lowercase()).unwrap_or_else(|| "status".into());
            const LEVELS: &[&str] = &["low", "medium", "high", "max"]; // max: GLM-5.3 (low/high/max)
            if LEVELS.contains(&arg.as_str()) || matches!(arg.as_str(), "off" | "default" | "none") {
                let val = if LEVELS.contains(&arg.as_str()) { arg.clone() } else { String::new() };
                // Drop any env/flag shadow so the save writes the deliberate choice.
                clear_runtime_override("reasoning_effort");
                update_config(|c| c.reasoning_effort = val.clone());
                let saved = crate::utils::config::saved_label(save_config(None));
                let shown = if val.is_empty() { tr("default (server decides)") } else { val };
                cmprintln!(
                    "{}",
                    trf(
                        "Reasoning effort set to {level} ({saved}). Takes effect on your next message.",
                        &[("level", FmtArg::Str(shown)), ("saved", FmtArg::Str(saved))]
                    )
                );
            } else if arg == "status" {
                let cur = get_config().effective_reasoning_effort();
                let cur = if cur.is_empty() { tr("default (server decides)") } else { cur };
                cmprintln!("{}", trf("Reasoning effort is {level}.", &[("level", FmtArg::Str(cur))]));
            } else {
                cmprintln!("Usage: /effort low | medium | high | max | off | status");
            }
            Dispatch::Handled
        }

        "keepgoing" => {
            let parts: Vec<&str> = body.split_whitespace().collect();
            let arg = parts.get(1).map(|s| s.to_lowercase()).unwrap_or_else(|| "status".into());
            match arg.as_str() {
                "on" | "off" => {
                    std::env::set_var("CODEMASON_KEEP_GOING", if arg == "on" { "1" } else { "0" });
                    update_config(|c| c.keep_going = arg == "on");
                    let saved = crate::utils::config::saved_label(save_config(None));
                    if arg == "on" {
                        let acts = crate::utils::keep_going_installer::ensure_installed();
                        cmprintln!(
                            "keep-going on ({saved}).{}",
                            if acts.is_empty() { "" } else { " Installed Stop hook." }
                        );
                    } else {
                        cmprintln!("keep-going off ({saved}). Hook left installed but inert.");
                    }
                }
                "status" => {
                    let eff = keep_going_enabled_local();
                    let has = crate::utils::hooks::has_hook_for_event("Stop");
                    cmprintln!(
                        "keep-going is {}{}.",
                        if eff { "on" } else { "off" },
                        if has { "" } else { " — but no Stop hook is installed, so it has no effect" }
                    );
                }
                _ => cmprintln!("Usage: /keepgoing on | off | status"),
            }
            Dispatch::Handled
        }

        // matched BUILTIN_COMMANDS but no branch — never reached
        _ => Dispatch::Handled,
    }
}

fn keep_going_enabled_local() -> bool {
    match std::env::var("CODEMASON_KEEP_GOING") {
        Ok(v) => !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no" | "off" | ""),
        Err(_) => get_config().keep_going,
    }
}

/// The session's live dangerously_skip_permissions (the tool-use context's
/// options are rebuilt per turn from this, so /dsp lands on the next call).
static SESSION_DSP: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

pub fn set_session_dsp(v: bool) {
    *SESSION_DSP.lock() = v;
}

pub fn session_dsp() -> bool {
    *SESSION_DSP.lock()
}

async fn dispatch_remote(body: &str) {
    let parts: Vec<&str> = body.split_whitespace().collect();
    let arg = parts.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
    if arg == "on" || arg == "off" {
        if arg == "on" {
            // Refuse without credentials: "enabled" while nothing can be
            // delivered is the one state that lies to you.
            if !crate::remote::notify::adapters().iter().any(|a| a.is_configured()) {
                let _ = crate::remote::adapters::telegram::ensure_credentials_scaffold();
                let path = crate::remote::adapters::telegram::credentials_path();
                cmprintln!("{}", tr("Cannot turn the remote channel on — no credentials."));
                cmprintln!(
                    "{}",
                    trf("  Fill in the slots in {path} and try again:", &[("path", FmtArg::Str(path.to_string_lossy().into_owned()))])
                );
                cmprintln!("    TELEGRAM_BOT_TOKEN=...");
                cmprintln!("    TELEGRAM_CHAT_ID=...");
                return;
            }
            // If --no-remote switched this run off, lift that first.
            if matches!(
                std::env::var("CODEMASON_REMOTE").unwrap_or_default().trim().to_lowercase().as_str(),
                "0" | "off" | "false" | "no"
            ) {
                std::env::set_var("CODEMASON_REMOTE", "1");
            }
        }
        update_config(|c| c.remote_enabled = arg == "on");
        let saved = crate::utils::config::saved_label(save_config(None));
        // Outbound is decided per call; inbound is a poller already in its loop.
        cmprintln!("  {}", crate::remote::input::set_enabled(arg == "on").await);
        let why = crate::remote::notify::off_reason();
        if arg == "on" && !why.is_empty() {
            cmprintln!(
                "{}",
                trf("Remote channel on ({saved}) — but still off right now because of {reason}.", &[("reason", FmtArg::Str(why)), ("saved", FmtArg::Str(saved))])
            );
        } else if arg == "on" {
            cmprintln!("{}", trf("Remote channel on ({saved}). Alerts, approvals and questions go to the phone.", &[("saved", FmtArg::Str(saved))]));
        } else {
            cmprintln!("{}", trf("Remote channel off ({saved}). Alerts, spinner, approvals, questions and inbound messages all stop.", &[("saved", FmtArg::Str(saved))]));
        }
        return;
    }
    if arg == "clean" {
        // Explicit removal, ignoring the retention window.
        let n = crate::remote::inbox::sweep(None, true);
        cmprintln!("{}", trf("Removed {count} leftover file(s).", &[("count", FmtArg::Int(n as i64))]));
        cmprintln!(
            "{}",
            colorize(&tr("  Live sessions and anything still queued for them were kept."), colors::DIM)
        );
        return;
    }
    if !arg.is_empty() {
        // Anything else is a NAME for this session's bot.
        let name = body.splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim().to_string();
        if crate::remote::adapters::telegram::claimed_token().is_empty() {
            cmprintln!("{}", tr("No bot is claimed for this session — remote is off."));
            return;
        }
        let (renamed, why) = crate::remote::adapters::telegram::set_bot_name(&name).await;
        if renamed {
            cmprintln!(
                "{}",
                trf("Bot renamed to {name}. Its chat in Telegram is now this session.", &[("name", FmtArg::Str(name))])
            );
        } else {
            // Say what Telegram actually said (the real answer was a rename
            // cooldown measured in hours).
            cmprintln!("{}", trf("Could not rename the bot: {why}", &[("why", FmtArg::Str(why))]));
            cmprintln!(
                "{}",
                colorize(&tr("  The session is unaffected — the name is only a label in the chat list."), colors::DIM)
            );
        }
        return;
    }
    print_remote_status().await;
}

fn dispatch_goal(body: &str, goal_kickoff: &Mutex<Option<String>>) {
    use crate::utils::goal as g;
    let rest = body["goal".len()..].trim().to_string();
    let sub = rest.to_lowercase();

    // --- gate management: /goal gate add|list|rm|clear ---
    if sub == "gate" || sub.starts_with("gate ") {
        let gargs = rest["gate".len()..].trim().to_string();
        let mut it = gargs.splitn(2, char::is_whitespace);
        let gcmd = it.next().unwrap_or("").to_lowercase();
        let gval = it.next().unwrap_or("").to_string();
        match gcmd.as_str() {
            "add" if !gval.is_empty() => {
                g::add_gate(&gval);
                cmprintln!("{}", trf("Gate added: {cmd}", &[("cmd", FmtArg::Str(gval))]));
            }
            "list" | "ls" | "" => {
                let gates = g::get_gates();
                if !gates.is_empty() {
                    cmprintln!("{}", tr("Completion gates:"));
                    for (i, gg) in gates.iter().enumerate() {
                        cmprintln!("  {}. $ {gg}", i + 1);
                    }
                } else {
                    cmprintln!("{}", tr("No gates. Add one with: /goal gate add <command>"));
                }
            }
            "rm" | "remove" | "del" if gval.trim().parse::<usize>().is_ok() => {
                let n = gval.trim().parse::<usize>().unwrap();
                cmprintln!("{}", if g::remove_gate(n) { tr("Gate removed.") } else { tr("No gate at that number.") });
            }
            "clear" => {
                g::clear_gates();
                cmprintln!("{}", tr("Gates cleared."));
            }
            "auto" => {
                let av = gval.trim().to_lowercase();
                if av == "on" || av == "off" {
                    clear_runtime_override("goal_gate_autorun");
                    update_config(|c| c.goal_gate_autorun = av == "on");
                    let saved = crate::utils::config::saved_label(save_config(None));
                    cmprintln!(
                        "{}{}",
                        trf("Gate auto-run {state} ({saved}).", &[("state", FmtArg::Str(av.clone())), ("saved", FmtArg::Str(saved))]),
                        if av == "on" {
                            tr("  Gates run when a turn ends; a failure keeps the agent working.")
                        } else {
                            tr("  Gates only run on /goal check.")
                        }
                    );
                } else {
                    let cur = get_config().goal_gate_autorun;
                    cmprintln!(
                        "{}",
                        trf("Gate auto-run is {state}.", &[("state", FmtArg::Str(if cur { "on".into() } else { "off".into() }))])
                    );
                }
            }
            _ => cmprintln!("Usage: /goal gate add <command> | list | rm <n> | clear | auto on|off"),
        }
        return;
    }

    // --- run the gates now: /goal check ---
    if matches!(sub.as_str(), "check" | "verify" | "gates") {
        let gates = g::get_gates();
        if gates.is_empty() {
            cmprintln!("{}", tr("No gates to check. Add one with: /goal gate add <command>"));
            return;
        }
        cmprintln!("{}", trf("Running {n} gate(s)...", &[("n", FmtArg::Int(gates.len() as i64))]));
        let cwd = crate::utils::state::get_cwd();
        let results = g::run_gates(Some(&cwd));
        let mut allpass = true;
        for (c, passed, code, tail) in &results {
            let mark = if *passed { "PASS".to_string() } else { format!("FAIL (exit {code})") };
            cmprintln!("  [{mark}] $ {c}");
            if !passed {
                allpass = false;
                let t = tail.trim();
                if !t.is_empty() {
                    let lines: Vec<&str> = t.lines().collect();
                    for ln in lines.iter().rev().take(8).rev() {
                        cmprintln!("      {ln}");
                    }
                }
            }
        }
        cmprintln!(
            "{}",
            if allpass {
                tr("All gates pass — the goal's completion bar is met.")
            } else {
                tr("Some gates fail — the goal is not complete yet.")
            }
        );
        return;
    }

    if matches!(sub.as_str(), "" | "show" | "status") {
        let cur = g::get_goal();
        if !cur.is_empty() {
            cmprintln!("{}", trf("Active goal: {goal}", &[("goal", FmtArg::Str(cur))]));
            for (i, gg) in g::get_gates().iter().enumerate() {
                cmprintln!("  gate {}: $ {gg}", i + 1);
            }
        } else {
            cmprintln!("{}", tr("No active goal. Set one with: /goal <your objective>"));
        }
    } else if matches!(sub.as_str(), "clear" | "off" | "none" | "done") {
        g::clear_goal();
        cmprintln!("{}", tr("Goal cleared."));
    } else {
        g::set_goal(&rest);
        cmprintln!(
            "{}",
            trf(
                "Goal set — starting now (kept until /goal clear, injected each turn): {goal}",
                &[("goal", FmtArg::Str(rest.clone()))]
            )
        );
        // Kick off work toward the goal immediately.
        *goal_kickoff.lock() = Some(rest);
    }
}

// === The turn: process_prompt ===

/// Process a single prompt and display results.
///
/// `spinner` None (plain/sequential mode): a \r-rendering spinner is created
/// here. Given (interactive mode): the shared external-render spinner is used
/// and stdin stays with the persistent prompt engine.
#[allow(clippy::too_many_arguments)]
pub async fn process_prompt(
    user_input: &str,
    messages: &Arc<Mutex<Vec<Message>>>,
    system_prompt: &[String],
    context: &BTreeMap<String, String>,
    ctx: &ToolUseContext,
    verbose: bool,
    debug: bool,
    spinner: Option<Arc<StreamingSpinner>>,
) {
    // The gauge denominator, re-read where a server probe is allowed to block.
    refresh_ctx_limit();

    // The image section of the system prompt follows the server's limit; a
    // session started while the server could not be asked gets it here, on the
    // first turn the server answers (constants/prompts.rs main_prompt_for_turn).
    let system_prompt = crate::constants::prompts::main_prompt_for_turn(
        system_prompt,
        &ctx.options.tools,
        Some(&get_config().model_name),
    )
    .await;

    // What the user TYPED decides the UI language when nothing else has.
    // Before the paste expansion on purpose: a pasted English log is not the
    // user writing English.
    crate::utils::i18n::observe_user_text(user_input);

    // Expand [Pasted text #N] placeholders before the message goes to the model.
    let mut user_input = user_input.to_string();
    if let Some(r) = ACTIVE_ROUTER.lock().clone() {
        let store = r.paste_store.lock();
        user_input = expand_pasted_refs(&user_input, &store);
    }
    // Every path into the model funnels through here.
    let user_input = repair_surrogates(&user_input);

    // NEITHER cwd NOR date goes in `context` any more. `context` is folded into
    // the SYSTEM PROMPT (format_system_prompt_with_context), which is the cached
    // prefix: every byte of it has to be stable for the whole session or the
    // server re-prefills the entire conversation.
    //
    // The date used to be PINNED to the REPL-start value for exactly that
    // reason — accuracy traded away to protect the prefix — while cwd was
    // refreshed every turn on the reasoning that it "changes rarely". Measured
    // on a real session (2026-09-05): it changed 18 times while moving between
    // repos, and 4 of the 9 worst stalls immediately follow a change — 44%
    // against a 10.8% base rate. At ~200k tokens each stall was 138-191s.
    //
    // The trade was never necessary. Measured on this backend at 96k tokens:
    // appending to the end costs 3.4s, changing the front costs 72s. Same
    // information, 20x apart, decided purely by WHERE it sits. So both values
    // now ride on the user turn, where they are appended — the model gets the
    // CURRENT directory and the CURRENT date every turn, and the prefix never
    // moves. The date is accurate again as a side effect.
    // `context` is folded into the SYSTEM PROMPT, which is the cached prefix,
    // so NOTHING that can change during a session may be put in it. It is left
    // empty on purpose; the per-turn values below are APPENDED instead.
    let context = context.clone();
    let mut turn_text = format!(
        "<turn-env>\nCurrent Working Directory: {cwd}\nAll relative paths are based on this directory. \".\" means {cwd}.\nToday's date: {date}\n</turn-env>",
        cwd = crate::utils::state::get_cwd(),
        date = chrono::Local::now().format("%Y-%m-%d"),
    );

    // The Agent-prompt language rule. It used to be built into the Agent tool's
    // JSON schema, i.e. into the tool block at the very FRONT of the prefix,
    // where every flip of the detected language threw the entire conversation
    // out of the server's cache (measured 2026-09-06: ~123s of re-prefill on a
    // 121k session). Same information, appended instead of embedded.
    // Also the moment the SESSION language gets pinned (first prompt with a
    // script signal wins) — sub-agent prompts are built from that pin, never
    // from the per-request value, so their prefix cannot drift.
    crate::utils::language::pin_session_language(crate::utils::language::detect_language(&user_input));
    if let Some(language) = crate::utils::language::effective_language() {
        turn_text.push_str(&format!(
            "\n<agent-prompt-language>\n{}\n</agent-prompt-language>",
            crate::tools::agent_tool::prompt_language_rule(&language)
        ));
    }

    // Persistent goal: re-read every turn so a mid-session /goal change lands
    // on the next one. It used to go into `context` (i.e. the system prompt),
    // which meant changing the goal mid-session silently invalidated the whole
    // cached history; appended here it cannot.
    let goal = crate::utils::goal::render_goal_context();
    if !goal.is_empty() {
        turn_text.push_str(&format!("\n<active_goal>\n{goal}\n</active_goal>"));
    }

    let turn_env = crate::utils::messages::create_user_message_text(&turn_text);
    let turn_env = crate::utils::messages::as_meta(turn_env);
    // Fresh user turn: reset the per-turn goal-gate auto-run counter.
    *ctx.gate_recovery.lock() = 0;

    let user_msg = crate::utils::messages::create_user_message_text(&user_input);

    // Make good on a summary taken mid-turn. Same loss as the reduction below,
    // but this one cannot be replayed here: a summary is an API round trip, not
    // a function of the history, so microcompact_messages reproduces nothing.
    // The REPL summarizes its own history once, here.
    //
    // Deliberately NOT routed through auto_compact_if_needed. Its only
    // threshold-bypassing entrance is context_proven_fatal, which escalates to
    // microcompact and no further ("Not escalated to a full summary here") --
    // and by the time a mid-turn summary has happened microcompact is already
    // exhausted, so it would free 0, log "proven-fatal, exhausted", and leave
    // the full history in place. Its threshold gate is no use either: the
    // mid-turn requests went out summarized, so the usage anchor describes a
    // few thousand tokens while this history is still the whole conversation.
    //
    // One summary call, which an honest estimate would have made anyway. Its
    // output is capped at MAX_OUTPUT_TOKENS_FOR_SUMMARY (20k), so it fits where
    // the ordinary turn's 65,536-token reservation does not -- which is why
    // this reconciliation can still run on a history the server would refuse.
    //
    // `messages` is cloned and released before the await and taken again to
    // write: never held across the request, and never while the spinner can
    // reach `paint`. Same shape as the /compact handler above.
    let needs_summary = {
        let mut f = ctx.needs_summary.lock();
        std::mem::replace(&mut *f, false)
    };
    if needs_summary {
        let snapshot = messages.lock().clone();
        let spin = StreamingSpinner::new(false);
        spin.start(Some("Compacting"));
        let s1 = spin.clone();
        let s2 = spin.clone();
        let res = crate::services::compaction::compact_conversation(
            &snapshot,
            ctx,
            None,
            true,
            Some(Arc::new(move |n| s1.update(n))),
            Some(Arc::new(move |n| s2.update_thinking(n))),
        )
        .await;
        spin.stop();
        match res {
            Ok(r) => {
                {
                    let mut m = messages.lock();
                    m.clear();
                    m.extend(r.messages.clone());
                }
                // Same reason /compact does it: the summary carries no file
                // contents, so every path the model still believes it has read
                // must be re-read rather than assumed fresh.
                ctx.read_file_state.lock().clear();
                cmprintln!(
                    "  {} Auto compacted {} conversation summarized. Continuing...\n",
                    sym("connector"),
                    sym("dash")
                );
            }
            Err(e) => {
                // Left unreduced on purpose: the threshold path below still
                // gets its turn, and a wrong summary is worse than none.
                crate::utils::log::log_error(
                    &format!(
                        "a summary taken mid-turn could not be re-applied to the REPL history:                          {e} -- the full history is still in place"
                    ),
                    Some("compaction: summary readopt failed"),
                );
            }
        }
    }

    // Re-apply a reduction that was taken mid-turn. compact_before_recursion
    // computes a trimmed history, hands it to the recursion and then loses it —
    // that list belongs to the recursion, this one is authoritative — so it
    // raises needs_reduction and the reduction is redone here, where it sticks.
    //
    // This must run BEFORE the snapshot below. The mid-turn requests went out
    // trimmed, so the usage anchor describes the slim prompt while this history
    // is still the fat one: should_auto_compact would weigh the slim number,
    // decline, and send the fat history. On 2026-10-03 18:07 that was 458,753
    // input + 65,536 reserved output against a 524,288 window — a 400, with the
    // gauge reading 262k.
    //
    // microcompact_messages is idempotent (an already-cleared result is
    // skipped), so a history that needs nothing frees 0 and is left alone.
    // `messages` is cloned and released before the count, and taken again to
    // write: it is never held across anything that can reach `paint`.
    let needs_reduction = {
        let mut f = ctx.needs_reduction.lock();
        std::mem::replace(&mut *f, false)
    };
    if needs_reduction {
        let before = messages.lock().clone();
        let (cleared, freed) = crate::services::compaction::microcompact_messages(&before);
        if freed > 0 {
            let mut m = messages.lock();
            m.clear();
            m.extend(cleared);
        }
        crate::utils::log::debug_log(
            "REDUCTION_READOPTED",
            Some(&serde_json::json!({ "tokens_freed": freed })),
        );
    }

    // Auto-compact before sending the new message.
    let snapshot = messages.lock().clone();
    let ac_spin = if crate::services::compaction::should_auto_compact(
        &snapshot,
        &crate::services::compaction::oob_scope(ctx),
    ) {
        let s = StreamingSpinner::new(false);
        s.start(Some("Compacting"));
        Some(s)
    } else {
        None
    };
    // Pre-turn call passes a FRESH counter (Python cli.py:3205 omits the
    // tracking arg entirely): the accumulated ctx counter is the IN-turn
    // breaker (query.rs). Passing it here meant 3 transient failures — a
    // briefly unreachable server — disabled auto-compaction for the rest of
    // the session, which then died on prompt-too-long.
    // Same counter query.rs uses mid-turn, so the circuit breaker
    // (MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES) covers this call site too. A
    // literal 0 here meant a compaction that failed every time was retried
    // on every user message — bounded by the human, not by the breaker, and
    // in a remote -dsp session the human is a message queue.
    let tracking = ctx.autocompact_tracking.lock().consecutive_failures;
    let (cb1, cb2) = match &ac_spin {
        Some(s) => {
            let a = s.clone();
            let b = s.clone();
            (
                Some(Arc::new(move |n| a.update(n)) as Arc<dyn Fn(i64) + Send + Sync>),
                Some(Arc::new(move |n| b.update_thinking(n)) as Arc<dyn Fn(i64) + Send + Sync>),
            )
        }
        None => (None, None),
    };
    // context_proven_fatal: normally false — this is the pre-turn check, before
    // anything has been sent, so there is no collapse to have observed yet.
    //
    // True for exactly one turn after the repetition recovery's last resort
    // microcompacted a collapsed history and could not keep the result (see
    // ToolUseContext::proven_fatal). Taken here because this is the only
    // auto-compaction call site that owns `messages` and writes the reduced set
    // back; mid-turn it can only be handed to a recursion. Consumed, not read:
    // it applies to the turn that follows the rescue and then goes quiet, so a
    // one-off collapse cannot keep forcing reductions for the rest of the
    // session.
    let proven_fatal = {
        let mut flag = ctx.proven_fatal.lock();
        std::mem::replace(&mut *flag, false)
    };
    let compact = crate::services::compaction::auto_compact_if_needed(
        &snapshot, ctx, tracking, cb1, cb2, proven_fatal,
    )
    .await;
    if let Some(s) = &ac_spin {
        s.stop();
    }
    if let Some(cf) = compact.consecutive_failures {
        ctx.autocompact_tracking.lock().consecutive_failures = cf;
    }
    if compact.was_compacted {
        if let Some(cr) = &compact.compaction_result {
            let mut m = messages.lock();
            m.clear();
            m.extend(cr.messages.clone());
        }
        cmprintln!("  {} Auto compacted {} conversation summarized. Continuing...\n", sym("connector"), sym("dash"));
    } else if compact.was_microcompacted {
        // query.rs:743 already handles this shape; the turn-boundary caller did
        // not, and microcompaction returns a NEW list rather than mutating in
        // place, so the trimmed history was computed and then dropped. Two
        // effects compounded: the full untrimmed history was sent anyway, and
        // the auto-compact threshold subtracts tokens it believes were freed,
        // inflating it every turn — so the real summarizing compaction fired
        // later than designed and the request could cross the server's context
        // window before it ever triggered. Same fix as the Python original.
        if let Some(trimmed) = &compact.messages {
            let mut m = messages.lock();
            m.clear();
            m.extend(trimmed.clone());
        }
        cmprintln!("  {}  Microcompacted {} old tool results cleared, recent context kept.\n", sym("connector"), sym("dash"));
    } else if compact.consecutive_failures.is_some_and(|cf| cf > 0) {
        // Compaction was attempted and reduced NOTHING. This printed nothing at
        // all before: the spinner simply vanished and the turn carried on with
        // the full history, so "Compacting… (5.7s ↓ 0 tokens)" followed by
        // silence was indistinguishable from success. Worse, after
        // MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES the breaker disables compaction
        // for the rest of the SESSION — also silently — and only a successful
        // compaction resets the counter, which by then can no longer happen. So
        // say it, and say how close the breaker is.
        let cf = compact.consecutive_failures.unwrap_or(0);
        let max = crate::services::compaction::MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES;
        if cf >= max {
            cmprintln!(
                "  {}  Auto compaction is now DISABLED for this session after {cf} failures {} the context will keep growing. Use /clear, or /compact once the model server is healthy.\n",
                sym("connector"),
                sym("dash")
            );
        } else {
            cmprintln!(
                "  {}  Auto compaction failed ({cf}/{max}) {} context was NOT reduced. See error.log; the turn continues on the full history.\n",
                sym("connector"),
                sym("dash")
            );
        }
    }

    {
        let mut m = messages.lock();
        // Environment first, then what the user actually said. Both are
        // APPENDED, so neither disturbs the cached prefix.
        m.push(Message::User(turn_env));
        m.push(Message::User(user_msg));
        // Persist incrementally: full list in, only the unseen tail appended.
        crate::utils::session_storage::record_transcript(&m, false);
    }

    cmprintln!(); // Blank line before response

    // Create and start the spinner (owned here only in sequential mode).
    let sequential = spinner.is_none();
    let spinner = spinner.unwrap_or_else(|| StreamingSpinner::new(false));
    spinner.start(None);

    // Token callbacks. on_thinking_update only fires on turns that actually
    // produced reasoning, so the breakdown appears exactly then.
    let mut ctx = ctx.clone();
    {
        let s = spinner.clone();
        // resume() first: in sequential mode the spinner is stopped before each
        // printed block (below), and update() drops the count while inactive.
        // A tool call's ARGUMENTS carry no content deltas, so a big Write that
        // opens with a line of text would then freeze the counter for the whole
        // generation — measured 602s of it at 35 tok/s. resume() keeps the
        // count (unlike start()), so the number continues rather than restarts.
        ctx.on_token_update = Some(Arc::new(move |n| {
            s.resume();
            s.update(n);
        }));
        let s = spinner.clone();
        ctx.on_thinking_update = Some(Arc::new(move |n| s.update_thinking(n)));
        // pin_verb, not start(): a compaction in the middle of a turn must not
        // reset the turn's elapsed clock or its token counts.
        let s = spinner.clone();
        ctx.on_status = Some(Arc::new(move |v: Option<String>| s.pin_verb(v.as_deref())));
    }

    // -rt/--realtime: stream the reply live, MARKDOWN-rendered, BLOCK by block.
    let realtime = get_config().realtime;
    let rt_streamed = Arc::new(AtomicBool::new(false));
    let rt_printed = Arc::new(AtomicBool::new(false));
    let rt_md = Arc::new(Mutex::new(crate::utils::realtime_render::MarkdownBlockBuffer::new()));

    fn print_md_block(block: &str, rt_printed: &AtomicBool) {
        // Strip ANSI/control bytes from the MODEL's text before rendering.
        let block = sanitize_terminal_text(block);
        let rendered = render_markdown(&block);
        let rendered = rendered.trim_matches('\n');
        if rendered.is_empty() {
            return;
        }
        if rt_printed.load(Ordering::SeqCst) {
            cmprintln!();
        }
        rt_printed.store(true, Ordering::SeqCst);
        safe_print(rendered);
    }

    if realtime {
        let md = rt_md.clone();
        let streamed = rt_streamed.clone();
        let printed = rt_printed.clone();
        let spin = spinner.clone();
        ctx.on_content_delta = Some(Arc::new(move |text: &str| {
            for block in md.lock().feed(text) {
                if block.trim().is_empty() {
                    continue;
                }
                if sequential {
                    // Before EVERY block, not just the first: the \r-spinner
                    // would clobber the text, and on_token_update resumes it
                    // afterwards so generation that emits no further text
                    // (tool-call arguments) still shows a live count.
                    spin.stop();
                }
                streamed.store(true, Ordering::SeqCst);
                print_md_block(&block, &printed);
            }
        }));
    }

    // Track tool_use_id -> (tool_name, tool_input) for pretty display.
    let mut pending_tool_uses: BTreeMap<String, (String, Value)> = BTreeMap::new();
    let mut shown_tool_headers: std::collections::HashSet<String> = Default::default();

    let turn_started = std::time::Instant::now();
    let mut last_answer = String::new();
    // The '⎯ ↓ …' usage line, tracked across the whole turn (see the Python
    // for what each flag is for).
    let mut footer_shown = false;
    let mut footer_done = false;
    let mut footer_responded = false;
    let mut footer_usage: Option<crate::types::messages::BetaUsage> = None;
    let mut footer_duration_ms: i64 = 0;
    let mut tool_counts: std::collections::HashMap<String, i64> = Default::default();

    let remote_origin = ACTIVE_ROUTER
        .lock()
        .as_ref()
        .map(|r| r.turn_remote_origin.load(Ordering::SeqCst))
        .unwrap_or(false);
    let progress = match crate::remote::hooks::turn_report_delay(remote_origin) {
        Some(delay) => {
            let p = crate::remote::progress::RemoteProgress::new(
                &crate::utils::state::get_cwd(),
                Some(delay),
            );
            p.start();
            Some(p)
        }
        None => None,
    };
    let mut had_text; // text seen before a tool (for the one-blank rule)

    let snapshot = messages.lock().clone();
    let mut stream = crate::query::query(crate::query::QueryParams {
        messages: snapshot,
        system_prompt: system_prompt.to_vec(),
        context,
        can_use_tool: permission_callback(),
        tool_use_context: ctx.clone(),
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
                spinner.stop();
                {
                    let mut m = messages.lock();
                    m.push(message.clone());
                    crate::utils::session_storage::record_transcript(&m, false);
                }

                let usage = a.message.usage.clone();
                let mut duration_ms = a.duration_ms;
                // If no usage from the server, time the response off the spinner.
                if !crate::cli::reported_usage_pub(Some(&usage)) {
                    let (elapsed, _) = spinner.get_final_stats();
                    if elapsed > 0.0 {
                        duration_ms = (elapsed * 1000.0) as i64;
                    }
                }
                footer_done = false;
                footer_responded = true;
                footer_usage = Some(usage.clone());
                footer_duration_ms = duration_ms;

                // -rt: show a header before execution only for a SINGLE
                // tool_use (the slow-single-tool case). With MULTIPLE tools we
                // keep the paired header+result printing.
                let n_tool_uses = a
                    .message
                    .content
                    .iter()
                    .filter(|b| matches!(b, ContentBlock::ToolUse(_)))
                    .count();
                had_text = false;

                for block in &a.message.content {
                    match block {
                        ContentBlock::Text(t) => {
                            let text = crate::cli::strip_completion_sentinel_pub(&t.text);
                            if text.is_empty() {
                                continue;
                            }
                            had_text = true;
                            if realtime {
                                let rest = rt_md.lock().flush();
                                if rt_streamed.load(Ordering::SeqCst) {
                                    if !rest.trim().is_empty() {
                                        print_md_block(&rest, &rt_printed);
                                    }
                                } else {
                                    print_md_block(&text, &rt_printed);
                                }
                                // This message is done; the next one starts its
                                // own spacing.
                                rt_streamed.store(false, Ordering::SeqCst);
                                rt_printed.store(false, Ordering::SeqCst);
                            } else {
                                safe_print(&render_markdown(&text));
                            }
                            last_answer = text.clone(); // newest prose wins
                            if let Some(p) = &progress {
                                p.on_text(&text);
                            }
                        }
                        ContentBlock::ToolUse(tu) => {
                            // Prose in THIS response is finished — close it with
                            // the usage line now, above the tool.
                            if had_text && !footer_done {
                                footer_done = true;
                                let line = crate::cli::render_token_footer_pub(
                                    footer_usage.as_ref(),
                                    footer_duration_ms,
                                    &spinner,
                                    crate::cli::footer_fallback(footer_usage.as_ref(), messages),
                                    verbose,
                                );
                                if !line.is_empty() {
                                    cmprintln!("{line}");
                                    footer_shown = true;
                                }
                            }
                            if !tu.name.is_empty() {
                                *tool_counts.entry(tu.name.clone()).or_insert(0) += 1;
                                if let Some(p) = &progress {
                                    p.on_tool(&tu.name);
                                }
                            }
                            if !tu.id.is_empty() {
                                pending_tool_uses.insert(tu.id.clone(), (tu.name.clone(), tu.input.clone()));
                            }
                            if realtime && !tu.id.is_empty() && n_tool_uses == 1 {
                                let dn = crate::tools::get_tool_by_name(&tu.name)
                                    .map(|t| t.user_facing_name(Some(&tu.input)))
                                    .unwrap_or_else(|_| tu.name.clone());
                                // A header carries no leading blank; add one when
                                // this message streamed text before the tool.
                                if had_text {
                                    cmprintln!();
                                }
                                display_tool_use(&tu.name, &dn, &tu.input, verbose);
                                shown_tool_headers.insert(tu.id.clone());
                            }
                        }
                        ContentBlock::Thinking(_) => {}
                    }
                }

                // Prose that ran to the end of the response (no tool followed).
                if had_text && !footer_done {
                    footer_done = true;
                    let line = crate::cli::render_token_footer_pub(
                        footer_usage.as_ref(),
                        footer_duration_ms,
                        &spinner,
                        crate::cli::footer_fallback(footer_usage.as_ref(), messages),
                        verbose,
                    );
                    if !line.is_empty() {
                        cmprintln!("{line}");
                        footer_shown = true;
                    }
                }

                // Keep the spinner ALIVE during tool execution (Claude Code
                // parity): tools run inside query() between this message and
                // the tool_result, and a dark spinner reads as a freeze.
                //
                // Labelled with the tool. config.spinner_random_verb restores
                // upstream's decorative random verb for anyone who prefers it.
                // In a long run the "[Bash] $ ..." line above has scrolled away,
                // and a word picked at random is then the only thing on screen.
                // An in-progress todo still wins over both — that is what
                // activeForm is for.
                if n_tool_uses > 0 {
                    let label = if crate::utils::config::get_config().spinner_random_verb {
                        None
                    } else {
                        let names: Vec<&str> = a
                            .message
                            .content
                            .iter()
                            .filter_map(|b| match b {
                                ContentBlock::ToolUse(tu) => Some(tu.name.as_str()),
                                _ => None,
                            })
                            .collect();
                        match names.len() {
                            0 => None,
                            1 => Some(names[0].to_string()),
                            n => Some(format!("{} +{}", names[0], n - 1)),
                        }
                    };
                    spinner.start(label.as_deref());
                }
            }

            Message::User(u) => {
                // Stop the spinner before printing (the previous block ends
                // with start(), parking "Thinking…" at column 0).
                spinner.stop();
                {
                    let mut m = messages.lock();
                    m.push(message.clone());
                    crate::utils::session_storage::record_transcript(&m, false);
                }

                for block in u.message.content.as_blocks() {
                    let ContentBlockParam::ToolResult { tool_use_id, content, is_error, data, .. } = block
                    else {
                        continue;
                    };
                    let result_data = data.clone().unwrap_or(Value::Null);

                    // Print this tool's header PAIRED with its result. Show it
                    // for read-only / skip-mode / session-allowed tools; write
                    // tools in interactive mode were already shown by the
                    // permission handler.
                    if let Some((tname, tinput)) = pending_tool_uses.get(&tool_use_id).cloned() {
                        let (ro, dname) = match crate::tools::get_tool_by_name(&tname) {
                            Ok(t) => (t.is_read_only(), t.user_facing_name(Some(&tinput))),
                            Err(_) => (true, tname.clone()),
                        };
                        let skip = ctx.options.dangerously_skip_permissions;
                        let session_allowed = crate::cli::permission_key_pub(&tname, &tinput)
                            .map(|k| crate::cli::session_tool_allows_contains(&k))
                            .unwrap_or(false);
                        if (ro || skip || session_allowed) && !shown_tool_headers.contains(&tool_use_id) {
                            if had_text_flag(&mut pending_tool_uses) {
                                // consumed once so parallel results don't each
                                // add a blank (see the Python's _had_text reset)
                            }
                            display_tool_use(&tname, &dname, &tinput, verbose);
                        }
                    }

                    let mut result = content.as_text();
                    // Strip terminal-corrupting escape/control bytes before any
                    // display path — tool output is untrusted.
                    result = sanitize_terminal_text(&result);

                    // Keep the FULL text for Ctrl+O / /expand before truncation.
                    if !result.is_empty() && !is_error {
                        let (tn, ti) = pending_tool_uses
                            .get(&tool_use_id)
                            .cloned()
                            .unwrap_or_else(|| (String::new(), Value::Null));
                        crate::cli::record_tool_output_pub(&tn, &ti, &result);
                    }

                    // The tool's own result rendering (fed the structured
                    // Output). None = fall through to the raw text.
                    let rendered = match pending_tool_uses.get(&tool_use_id) {
                        Some((tn, _)) if !is_error => crate::tools::get_tool_by_name(tn)
                            .ok()
                            .and_then(|t| t.render_tool_result_message(&result_data, verbose)),
                        _ => None,
                    };

                    let prefix = format!("  {} ", sym("connector"));
                    let indent = " ".repeat(display_width(&prefix));
                    match rendered {
                        Some(r) => {
                            // '' = the tool renders nothing (surfaced elsewhere).
                            if !r.is_empty() {
                                cmprint!("{prefix}");
                                for (i, line) in r.split('\n').enumerate() {
                                    if i == 0 {
                                        safe_print(line);
                                    } else {
                                        safe_print(&format!("{indent}{line}"));
                                    }
                                }
                            }
                        }
                        None if !result.is_empty() => {
                            let lines: Vec<&str> = result.split('\n').collect();
                            if is_error {
                                let head: String = lines[0].chars().take(100).collect();
                                cmprintln!("  {} Error: {head}", sym("connector"));
                            } else {
                                let max_lines = if verbose { 20 } else { 5 };
                                cmprint!("{prefix}");
                                for (i, line) in lines.iter().take(max_lines).enumerate() {
                                    let cut: String = line.chars().take(100).collect();
                                    let ell = if line.chars().count() > 100 { "..." } else { "" };
                                    if i == 0 {
                                        safe_print(&format!("{cut}{ell}"));
                                    } else {
                                        safe_print(&format!("{indent}{cut}{ell}"));
                                    }
                                }
                                if lines.len() > max_lines {
                                    let hint = if sequential { "" } else { " · ctrl+o to expand" };
                                    cmprintln!(
                                        "{indent}{}",
                                        colorize(
                                            &format!("... ({} more lines{hint})", lines.len() - max_lines),
                                            colors::DIM
                                        )
                                    );
                                }
                            }
                        }
                        None => {}
                    }
                }
                cmprintln!(); // Blank line after tool result for readability
                spinner.start(None); // Restart for the next model response
            }

            Message::Progress(p) => {
                let result_text = &p.result_for_assistant;
                let data = &p.data;
                let text = if result_text.is_empty() {
                    data.get("result_for_assistant").and_then(|v| v.as_str()).unwrap_or("")
                } else {
                    result_text.as_str()
                };
                if text.is_empty() {
                    continue;
                }
                spinner.stop();
                let tool_name = data.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
                let tool_input = data.get("tool_input").cloned().unwrap_or(Value::Null);
                if !tool_name.is_empty() {
                    // One line per sub-agent tool call (renderToolActivity.tsx).
                    let preview = summarize_tool_input(&tool_input, crate::cli::AGENT_PROGRESS_COLS_PUB);
                    let label = colorize(&format!("Agent {} [{tool_name}]", sym("arrow")), colors::CYAN);
                    safe_print(&format!(
                        "  {} {label} {}",
                        sym("connector"),
                        colorize(&preview, colors::DIM)
                    ));
                } else if text.contains("Initializing") {
                    cmprintln!("  {} {}", sym("connector"), colorize("Agent starting...", colors::CYAN));
                } else {
                    cmprintln!("  {} {text}", sym("connector"));
                }
                spinner.resume();
            }
        }
    }

    spinner.stop();

    // A turn that printed no usage line still gets exactly one.
    if footer_responded && !footer_shown {
        let line = crate::cli::render_token_footer_pub(
            footer_usage.as_ref(),
            footer_duration_ms,
            &spinner,
            crate::cli::footer_fallback(footer_usage.as_ref(), messages),
            verbose,
        );
        if !line.is_empty() {
            cmprintln!("{line}");
        }
    }

    let cwd = crate::utils::state::get_cwd();
    let status = crate::utils::session_status::status_line(&messages.lock());
    crate::remote::hooks::notify_turn_end(
        turn_started.elapsed().as_millis() as i64,
        &last_answer,
        crate::remote::hooks::TurnEndInfo {
            cwd: Some(&cwd),
            tools: Some(tool_counts),
            remote_origin,
            progress,
            prompt: &user_input,
            status: &status,
            ..Default::default()
        },
    )
    .await;

    // Context warning, counted WITH estimation like the gauge and the footer.
    let total_tokens = crate::utils::tokens::token_count_with_estimation(&messages.lock());
    let warning = get_token_warning(total_tokens);
    if !warning.is_empty() {
        cmprintln!("\n{warning}");
    }
    let _ = debug;
}

/// Placeholder for the Python's per-message `_had_text` reset in the paired
/// tool-result path (kept as a named seam so the behavior is explicit: the
/// blank before a header is consumed once per message, not once per result).
fn had_text_flag(_pending: &mut BTreeMap<String, (String, Value)>) -> bool {
    false
}

// === REPL loops ===

/// Run one --plain turn with an interrupt path (Python cli.py:2759-2763):
/// Ctrl+C prints "[Interrupted]", cancels the abort token — the same token
/// ESC cancels in interactive mode, so the turn unwinds through its normal
/// abort paths — and the REPL continues. Without this, SIGINT took the
/// default action and killed the whole process mid-turn (signal_hook only
/// registers SIGTERM/SIGHUP), losing the queued transcript write.
async fn plain_turn_with_interrupt(
    turn: impl std::future::Future<Output = ()>,
    abort: &tokio_util::sync::CancellationToken,
) {
    tokio::pin!(turn);
    loop {
        tokio::select! {
            _ = &mut turn => break,
            r = tokio::signal::ctrl_c() => {
                if r.is_ok() {
                    cmprintln!("\n[Interrupted]");
                    abort.cancel();
                }
                // keep waiting for the turn to unwind through its abort paths
            }
        }
    }
}

/// Sequential prompt→turn→prompt loop for --plain/--ascii consoles.
/// Input is only possible between turns.
#[allow(clippy::too_many_arguments)]
async fn plain_repl(
    messages: Arc<Mutex<Vec<Message>>>,
    system_prompt: Vec<String>,
    context: BTreeMap<String, String>,
    mut ctx: ToolUseContext,
    verbose: bool,
    debug: bool,
    initial_prompt: String,
) {
    use std::io::{BufRead, Write};
    let goal_kickoff: Mutex<Option<String>> = Mutex::new(None);

    if !initial_prompt.is_empty() {
        // A user turn is starting: under `/ab auto` the model is about to
        // decide whether to accept, the one moment the refusal edit is wanted.
        crate::services::ab::on_user_turn().await;
        ctx.options.dangerously_skip_permissions = session_dsp(); // /dsp lands here too (Python mutates ctx.options directly, cli.py:2497)
        plain_turn_with_interrupt(
            process_prompt(&initial_prompt, &messages, &system_prompt, &context, &ctx, verbose, debug, None),
            &ctx.abort_controller.clone(),
        ).await;
        if ctx.abort_controller.is_cancelled() {
            ctx.abort_controller = CancellationToken::new();
        }
    }

    let stdin = std::io::stdin();
    loop {
        crate::cli::drain_bg_notifications_pub(&messages);
        cmprintln!();
        cmprint!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        // The StdinLock is the scrutinee's temporary, so it would live across
        // the arms — and both arms print. Read, release, then decide.
        let read = stdin.lock().read_line(&mut line);
        match read {
            Ok(0) => {
                cmprintln!("\nGoodbye!");
                break;
            }
            Ok(_) => {}
            Err(_) => {
                cmprintln!("Input encoding error. Please try again.");
                continue;
            }
        }
        let user_input = line.trim().to_string();
        if user_input.is_empty() {
            continue;
        }

        let action = dispatch_builtin(&user_input, &messages, &ctx, &goal_kickoff).await;
        let kick = goal_kickoff.lock().take();
        if let Some(k) = kick {
            // A user turn is starting: under `/ab auto` the model is about to
        // decide whether to accept, the one moment the refusal edit is wanted.
        crate::services::ab::on_user_turn().await;
        ctx.options.dangerously_skip_permissions = session_dsp(); // /dsp lands here too (Python mutates ctx.options directly, cli.py:2497)
        plain_turn_with_interrupt(
            process_prompt(&k, &messages, &system_prompt, &context, &ctx, verbose, debug, None),
            &ctx.abort_controller.clone(),
        ).await;
            if ctx.abort_controller.is_cancelled() {
                ctx.abort_controller = CancellationToken::new();
            }
            continue;
        }
        match action {
            Dispatch::Exit => {
                cmprintln!("Goodbye!");
                break;
            }
            Dispatch::Handled => continue,
            Dispatch::Prompt => {}
        }

        // A user turn is starting: under `/ab auto` the model is about to
        // decide whether to accept, the one moment the refusal edit is wanted.
        crate::services::ab::on_user_turn().await;
        ctx.options.dangerously_skip_permissions = session_dsp(); // /dsp lands here too (Python mutates ctx.options directly, cli.py:2497)
        plain_turn_with_interrupt(
            process_prompt(&user_input, &messages, &system_prompt, &context, &ctx, verbose, debug, None),
            &ctx.abort_controller.clone(),
        ).await;

        // Reset the abort token if ESC was pressed during processing.
        if ctx.abort_controller.is_cancelled() {
            ctx.abort_controller = CancellationToken::new();
        }
    }
}

/// Persistent-input REPL (default for real terminals).
///
/// ONE prompt engine runs for the whole app lifetime as the single owner of
/// stdin; all model/tool output is printed ABOVE the live prompt via
/// utils::term_out. A line submitted mid-turn is shown parked above the prompt
/// and INJECTED into the running turn at its next model step.
#[allow(clippy::too_many_arguments)]
async fn interactive_repl(
    messages: Arc<Mutex<Vec<Message>>>,
    system_prompt: Vec<String>,
    context: BTreeMap<String, String>,
    mut ctx: ToolUseContext,
    verbose: bool,
    debug: bool,
    initial_prompt: String,
) {
    let (tx, mut rx) = mpsc::unbounded_channel::<QueueItem>();
    let router = ReplInputRouter::new(tx, messages.clone());
    router.set_abort(ctx.abort_controller.clone());
    *ACTIVE_ROUTER.lock() = Some(router.clone());
    crate::cli::set_router_pub(Some(router.clone()));
    ctx.input_router = Some(router.clone() as Arc<dyn InputRouter>);

    // Remote input: a message from the phone goes on the very queue the prompt
    // already feeds, inheriting the mid-turn queueing behaviour.
    let r2 = router.clone();
    let remote_input = crate::remote::input::RemoteInput::new(
        &crate::utils::state::get_session_id(),
        &crate::utils::state::get_cwd(),
        Some(Arc::new(move |text: String, age: f64, source: String| {
            r2.submit_remote(&text, age, &source)
        })),
    );
    if remote_input.start().await {
        // One line at startup, naming the bot this instance claimed.
        let handle = crate::remote::adapters::telegram::claimed_handle().await;
        cmprintln!(
            "remote: telegram{} (details: /remote)",
            if handle.is_empty() { String::new() } else { format!(" @{handle}") }
        );
        let warn = crate::remote::adapters::telegram::credentials_file_warning();
        if !warn.is_empty() {
            cmprintln!("{}", colorize(&format!("  {} {warn}", sym("warn")), colors::YELLOW));
        }
    } else {
        // Not started, which on a fresh machine means "no credentials file".
        if let Some(made) = crate::remote::adapters::telegram::ensure_credentials_scaffold() {
            cmprintln!(
                "{}",
                colorize(
                    &trf(
                        "remote: off — no bot tokens yet. Wrote a starter file with 4 slots: {path}",
                        &[("path", FmtArg::Str(made.to_string_lossy().into_owned()))]
                    ),
                    colors::DIM
                )
            );
        }
    }

    // A resumed conversation or an argv initial prompt means there is already
    // output above the prompt — the input rule starts visible.
    router
        .conversation_started
        .store(!messages.lock().is_empty() || !initial_prompt.is_empty(), Ordering::SeqCst);

    let spinner = StreamingSpinner::new(true);

    // The engine's content hooks (the Python's _message()/rprompt()).
    //
    // The gauge's DENOMINATOR is read from a cell, not captured here. Captured,
    // it froze whatever the very first probe returned — and that probe runs
    // before the first request, so a model server still coming up (the normal
    // case when both are started together) gave 0 and the gauge stayed hidden
    // for the whole session, or gave the fallback default and every percentage
    // after it was measured against a window the server does not have. The
    // refresh cannot live in the repaint path: get_auto_compact_threshold ->
    // get_context_size -> fetch_server_info does blocking HTTP on a cache miss,
    // and repaint holds `paint`. It is refreshed at TURN START instead
    // (refresh_ctx_limit in process_prompt), where a probe is already expected.
    let ctx_limit = ctx_limit_cell();
    let esc_hint: Arc<Mutex<Option<(&'static str, std::time::Instant)>>> = Arc::new(Mutex::new(None));

    let hooks = {
        let router_m = router.clone();
        let spinner_m = spinner.clone();
        let messages_g = messages.clone();
        let esc_hint_r = esc_hint.clone();
        let router_rp = router.clone();
        let router_p = router.clone();
        let router_perm = router.clone();
        let router_proc = router.clone();
        let router_pause = router.clone();
        let router_paste = router.clone();
        let router_ev = router.clone();
        let ctx_ev = ctx.clone();

        EngineHooks {
            message_lines: Arc::new(move || {
                let mut parts: Vec<Frag> = Vec::new();
                let cols = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(80);
                // Live 'Thinking…' ticker directly above the input box. Hidden
                // while a permission ask is pending — the turn waits for the
                // USER then, and a ticking timer would be noise.
                if spinner_m.is_active() && !router_m.permission_pending() {
                    parts.push(Frag::styled("\x1b[36m", format!("{}\n", spinner_m.toolbar_text())));
                }
                if router_m.conversation_started.load(Ordering::SeqCst) {
                    // Rule between output and input, width cols-1.
                    let rule = "─".repeat(cols.saturating_sub(1).max(10));
                    parts.push(Frag::styled("\x1b[38;5;240m", format!("{rule}\n")));
                }
                // Queued-message block (mid-turn typing) directly ABOVE the
                // input line — NOT in scrollback.
                let pending = router_m.pending.lock().clone();
                for pl in pending_preview_lines(&pending, cols) {
                    parts.push(Frag::styled("\x1b[38;5;245m", format!("{pl}\n")));
                }
                parts
            }),
            rprompt: Arc::new(move || {
                // The bulk-stop confirmation outranks the Esc hints: it is
                // armed for 3s against their 700ms, and it is the one with
                // something destructive behind it.
                if let Some(hint) = router_rp.kill_agents.hint() {
                    return vec![Frag::plain(hint)];
                }
                if let Some((hint, at)) = *esc_hint_r.lock() {
                    if at.elapsed() < std::time::Duration::from_millis(700) {
                        return vec![Frag::plain(hint)];
                    }
                }
                context_gauge(ctx_limit.load(Ordering::Relaxed), &messages_g)
            }),
            prompt_marker: Arc::new(move || {
                if router_p.permission_pending() {
                    vec![Frag::styled("\x1b[1m", router_p.permission_prompt_text())]
                } else {
                    let theme = crate::utils::terminal_theme::active_theme_colors();
                    match theme.get("prompt") {
                        Some(p) => vec![Frag::styled(
                            format!("{}\x1b[1m", crate::utils::formatting::fg256_hex(p)),
                            "> ",
                        )],
                        None => vec![Frag::plain("> ")],
                    }
                }
            }),
            permission_pending: Arc::new(move || router_perm.permission_pending()),
            processing: Arc::new(move || router_proc.is_processing()),
            paused: Arc::new(move || router_pause.is_paused()),
            on_paste: Arc::new(move |data: &str| {
                // Large/multi-line paste → compact placeholder; the full text is
                // stashed and spliced back in at submit.
                if should_placeholder(data) {
                    let pid = router_paste.paste_store.lock().add(data);
                    let refstr = format_pasted_ref(pid, pasted_num_lines(data));
                    // Record the full paste so Ctrl+O can expand it.
                    crate::cli::record_tool_output_pub(
                        "Pasted text",
                        &Value::String(refstr.clone()),
                        data,
                    );
                    Some(refstr)
                } else {
                    None
                }
            }),
            completions: Arc::new(|before, tab| crate::utils::completion::get_completions(before, tab)),
            auto_suggest: Arc::new(|text| crate::utils::completion::get_suggestion(text, &[])),
            on_event: Arc::new(move |ev| {
                handle_engine_event(ev, &router_ev, &ctx_ev, &esc_hint);
            }),
        }
    };

    let engine = PromptEngine::new(hooks);
    router.set_engine(engine.clone());

    let mut engine_task = {
        let e = engine.clone();
        tokio::spawn(async move { e.run().await })
    };
    // Set once the handle has been consumed, so the join after the loop does
    // not poll a JoinHandle that already resolved.
    let mut engine_ended = false;
    // A session that ended because its input engine died did not finish
    // normally, and a wrapper that restarts or reports on codemason has to
    // be able to tell the difference from a clean quit.
    let mut engine_panicked = false;

    let goal_kickoff: Mutex<Option<String>> = Mutex::new(None);

    // The turn loop.
    if !initial_prompt.is_empty() {
        run_turn(&router, &initial_prompt, &messages, &system_prompt, &context, &mut ctx, verbose, debug, &spinner, false).await;
    }

    loop {
        // Watch the engine as well as the queue. The engine is the only thing
        // that feeds `rx`, and it runs as a spawned task, so a panic inside it
        // is caught by its JoinHandle instead of unwinding into main. Waiting
        // on `rx` alone then parks forever: the process stays alive with a
        // terminal the panic hook has politely restored and a prompt that will
        // never accept another key. That is worse than exiting, because it
        // looks like the app is merely thinking. Notice the engine is gone and
        // shut the session down instead.
        let item = tokio::select! {
            joined = &mut engine_task => {
                engine_ended = true;
                if joined.as_ref().err().is_some_and(|e| e.is_panic()) {
                    // The hook already restored the terminal and the default
                    // handler already printed the message and backtrace; say
                    // plainly that the session is over so this is not mistaken
                    // for a hang.
                    eprintln!("codemason: the input engine stopped after a panic — ending the session.");
                    engine_panicked = true;
                }
                break;
            }
            item = rx.recv() => match item {
                Some(i) => i,
                None => break,
            },
            // A background task finishing is a reason to carry on, not a
            // message that waits for the user to press a key.
            //
            // The notification was already being produced and queued; only the
            // last step was missing, and that gap is why the model would not
            // just end its turn and wait. It had no way to be woken, so it
            // reconstructed a sleep out of `sleep 600` in a shell plus polling
            // — burning a full request per cycle — and the alternative was an
            // unattended run that went quiet for as long as the task took.
            // print_runner already holds itself open for exactly this; the
            // interactive loop now does too.
            _ = async {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    if crate::utils::message_queue_manager::has_pending_notifications() {
                        return;
                    }
                }
            } => QueueItem::TaskWake,
        };
        match item {
            QueueItem::Exit => {
                cmprintln!("\nGoodbye!");
                break;
            }
            QueueItem::TaskWake => {
                // Fold the completions into the history and let the model act
                // on them. It asked for the work; it should not need a nudge
                // from the user to be handed the answer.
                crate::cli::drain_bg_notifications_pub(&messages);
                run_turn(
                    &router, TASK_WAKE_PROMPT, &messages, &system_prompt, &context,
                    &mut ctx, verbose, debug, &spinner, false,
                )
                .await;
            }
            QueueItem::SteerWake => {
                // If the turn's own gate already injected it, the steering
                // queue is empty now and this wake is a no-op. Otherwise the
                // turn ended before another model step — run the leftovers as
                // ordinary turns, in order.
                let mut exit = false;
                for leftover in crate::utils::steering_queue::drain_steering_messages() {
                    {
                        let mut p = router.pending.lock();
                        if let Some(pos) = p.iter().position(|x| *x == leftover) {
                            p.remove(pos);
                        }
                    }
                    cmprintln!();
                    cmprintln!("{}", user_echo_line(&leftover));
                    // Value out of the lock before printing — same reason as
                    // Router::echo_user_line above.
                    let tag = router.remote_lines.lock().remove(&leftover);
                    if let Some(tag) = tag {
                        cmprintln!("{}", colorize(&tag, colors::DIM));
                    }
                    crate::cli::drain_bg_notifications_pub(&messages);
                    match dispatch_builtin(&leftover, &messages, &ctx, &goal_kickoff).await {
                        Dispatch::Exit => {
                            cmprintln!("Goodbye!");
                            exit = true;
                            break;
                        }
                        Dispatch::Handled => continue,
                        Dispatch::Prompt => {}
                    }
                    run_turn(&router, &leftover, &messages, &system_prompt, &context, &mut ctx, verbose, debug, &spinner, false).await;
                }
                if exit {
                    break;
                }
            }
            QueueItem::Line(line, was_queued) => {
                // A Ctrl+C hard-stop discards the backlog: skip lines that are
                // no longer parked.
                if *DISCARD_BACKLOG.lock() {
                    if was_queued {
                        continue;
                    }
                    *DISCARD_BACKLOG.lock() = false;
                }
                // Consumed the moment the line leaves the queue, on EVERY path.
                let remote_tag = router.remote_lines.lock().remove(&line);
                crate::cli::drain_bg_notifications_pub(&messages);
                // A message typed mid-turn waited in pending; now it is actually
                // being processed — drop it from pending and echo it.
                if was_queued {
                    let mut p = router.pending.lock();
                    if let Some(pos) = p.iter().position(|x| *x == line) {
                        p.remove(pos);
                    }
                    drop(p);
                    cmprintln!();
                    cmprintln!("{}", user_echo_line(&line));
                    if let Some(tag) = &remote_tag {
                        cmprintln!("{}", colorize(tag, colors::DIM));
                    }
                }

                match dispatch_builtin(&line, &messages, &ctx, &goal_kickoff).await {
                    Dispatch::Exit => {
                        cmprintln!("Goodbye!");
                        break;
                    }
                    Dispatch::Handled => {
                        let kick = goal_kickoff.lock().take();
                        if let Some(k) = kick {
                            run_turn(&router, &k, &messages, &system_prompt, &context, &mut ctx, verbose, debug, &spinner, false).await;
                        }
                        continue;
                    }
                    Dispatch::Prompt => {}
                }

                run_turn(&router, &line, &messages, &system_prompt, &context, &mut ctx, verbose, debug, &spinner, remote_tag.is_some()).await;
            }
        }
    }

    engine.stop();
    if !engine_ended {
        let _ = engine_task.await;
    }
    // Stop advertising this session and hand the poll lock to whoever wants it
    // next. Anything still in the inbox is left alone.
    remote_input.stop().await;
    crate::remote::notify::drain_at_exit().await;
    crate::cli::set_router_pub(None);
    *ACTIVE_ROUTER.lock() = None;
    if engine_panicked {
        // Shutdown above has already run; exit non-zero so the failure is
        // visible to whatever launched this.
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_turn(
    router: &Arc<ReplInputRouter>,
    text: &str,
    messages: &Arc<Mutex<Vec<Message>>>,
    system_prompt: &[String],
    context: &BTreeMap<String, String>,
    ctx: &mut ToolUseContext,
    verbose: bool,
    debug: bool,
    spinner: &Arc<StreamingSpinner>,
    remote_origin: bool,
) {
    router.set_processing(true);
    router.turn_remote_origin.store(remote_origin, Ordering::SeqCst);
    // Same turn boundary as the REPL path: see on_user_turn.
    crate::services::ab::on_user_turn().await;
    // The permission gate re-reads this per call, so /dsp lands on the next one.
    ctx.options.dangerously_skip_permissions = session_dsp();
    process_prompt(text, messages, system_prompt, context, ctx, verbose, debug, Some(spinner.clone())).await;
    router.set_processing(false);
    if ctx.abort_controller.is_cancelled() {
        ctx.abort_controller = CancellationToken::new();
        router.set_abort(ctx.abort_controller.clone());
    }
}

/// Route one engine event to the router / turn loop (the Python's key
/// bindings in _make_prompt_kb).
fn handle_engine_event(
    ev: EngineEvent,
    router: &Arc<ReplInputRouter>,
    ctx: &ToolUseContext,
    esc_hint: &Arc<Mutex<Option<(&'static str, std::time::Instant)>>>,
) {
    match ev {
        EngineEvent::Submit(raw) => {
            // repair_surrogates AFTER the expansion; normalise CR here too (a
            // large paste is stashed verbatim and only spliced in at submit).
            let expanded = {
                let store = router.paste_store.lock();
                expand_pasted_refs(&raw, &store)
            };
            let full = repair_surrogates(&expanded.replace("\r\n", "\n").replace('\r', "\n"))
                .trim()
                .to_string();
            if full.is_empty() {
                return;
            }
            router.conversation_started.store(true, Ordering::SeqCst);
            let lower = full.to_lowercase();
            if lower == "/exit" || lower == "/quit" {
                cmprintln!("{}", colorize("  (exiting…)", colors::DIM));
                router.interrupt();
                router.submit_exit();
            } else if router.is_processing() {
                // mid-turn: INJECT into the running turn at its next model step.
                router.pending.lock().push(full.clone());
                router.submit_steering(&full);
            } else {
                // idle: echo the expanded prompt now, queue without a re-echo.
                cmprintln!();
                cmprintln!("{}", user_echo_line(&full));
                router.submit_line(full, false);
            }
        }
        EngineEvent::Interrupt => {
            router.interrupt();
        }
        EngineEvent::RewindRequested => {
            let r = router.clone();
            let c = ctx.clone();
            tokio::spawn(async move { run_message_selector(r, c).await });
        }
        EngineEvent::Permission(c) => {
            router.deliver(&c.to_string());
        }
        EngineEvent::OpenOutputs => {
            // Ctrl+O: ALL recent tool outputs, full, in one pager document.
            if router.permission_pending() || router.is_paused() {
                return;
            }
            let r = router.clone();
            tokio::spawn(async move {
                let _pause = r.pause_prompt_scoped().await;
                crate::cli::open_output_pager().await;
            });
        }
        EngineEvent::HardStop => {
            if !router.hard_stop() {
                cmprintln!("(Use /exit or Ctrl+D to quit)");
            }
        }
        EngineEvent::KillAgents => {
            // The source's own chord for the gesture Ctrl+C also reaches. Never
            // silent: a keypress that does nothing reads as broken.
            use crate::utils::kill_agents_gesture::Press;
            let (press, note) = router.kill_agents.press("Ctrl+X Ctrl+K");
            match press {
                Press::Idle => cmprintln!("(Ctrl+X Ctrl+K: no background agents running)"),
                _ => {
                    if let Some(note) = note {
                        let style = if press == Press::Armed { colors::DIM } else { colors::CYAN };
                        cmprintln!("{}", colorize(&format!("  {note}"), style));
                    }
                }
            }
        }
        EngineEvent::Eof => {
            router.submit_exit();
        }
    }
    let _ = esc_hint;
}

/// '███░░░░░ 38% · 88.4k/232.8k' — fragments, or [] when there's nothing to
/// say. Degrades by width instead of pushing the prompt around; hidden before
/// the first turn (a gauge reading 0% would lie about a context that already
/// holds the system prompt and every tool definition).
/// The context gauge's denominator, shared with the repaint hooks.
///
/// See the note at the `ctx_limit` binding: the repaint path must never probe,
/// so the value is refreshed from the turn path and only READ while painting.
fn ctx_limit_cell() -> Arc<std::sync::atomic::AtomicI64> {
    static CELL: Lazy<Arc<std::sync::atomic::AtomicI64>> =
        Lazy::new(|| Arc::new(std::sync::atomic::AtomicI64::new(0)));
    CELL.clone()
}

/// Re-read the auto-compact threshold for the gauge. Called at turn start.
fn refresh_ctx_limit() {
    let limit = crate::services::compaction::get_auto_compact_threshold().unwrap_or(0);
    ctx_limit_cell().store(limit, Ordering::Relaxed);
}

/// Gauge colour tiers, as a percentage of the auto-compact threshold.
///
/// Named because they were two bare numbers whose only documentation was "the
/// same escalation the footer warning uses, one step earlier" — which is not
/// true in either direction: the footer works in absolute token buffers
/// (WARNING_THRESHOLD_BUFFER_TOKENS / ERROR_THRESHOLD_BUFFER_TOKENS in
/// compaction.rs), so on a large window the footer's yellow can arrive well
/// after the gauge's red. Two schemes measuring the same thing is a deliberate
/// choice here (the gauge is an always-visible fill bar, the footer is a
/// "you must act now" line), and naming them is what makes it a choice rather
/// than a coincidence of two literals.
const CTX_GAUGE_RED_PCT: i64 = 80;
const CTX_GAUGE_YELLOW_PCT: i64 = 60;

fn context_gauge(ctx_limit: i64, messages: &Arc<Mutex<Vec<Message>>>) -> Vec<Frag> {
    if ctx_limit <= 0 {
        return Vec::new();
    }
    // token_count_with_estimation walks the whole history, so it must not run
    // per repaint: cache the count on (length, last message uuid).
    // (length, last uuid, tokens, image text — "이미지 1/8", empty without vision)
    static CACHE: Lazy<Mutex<(usize, String, i64, String)>> =
        Lazy::new(|| Mutex::new((0, String::new(), 0, String::new())));

    // This gauge runs inside the `rprompt` hook, i.e. inside repaint(), which
    // holds `paint`. It is the ONLY place the repaint path reaches for
    // `messages`, and blocking on it here is a lock-order INVERSION against the
    // turn/print path: that path holds `messages` and then prints —
    // cmprintln! -> term_out::emit -> write_above() -> `paint`. So one side
    // takes paint→messages and the other messages→paint, and they deadlock.
    // It fires under -rt, where a background RemoteProgress task drives repaint
    // concurrently with the turn thread's own streaming output (confirmed live:
    // two threads wedged in repaint, one holding paint waiting messages, the
    // other holding messages waiting paint).
    //
    // Fix the inversion at its root: NEVER wait on `messages` from inside a
    // repaint. Take it with try_lock, and when the turn thread holds it (it is
    // mid-mutation and about to print) reuse the last cached count instead of
    // blocking. A gauge that is one repaint stale is invisible; a hung prompt
    // is not.
    let used = match messages.try_lock() {
        Some(msgs) => {
            if msgs.is_empty() {
                return Vec::new();
            }
            let key_len = msgs.len();
            let key_uuid = msgs.last().map(|m| m.uuid().to_string()).unwrap_or_default();
            let mut c = CACHE.lock();
            if c.0 != key_len || c.1 != key_uuid {
                c.2 = crate::utils::tokens::token_count_with_estimation(&msgs);
                c.3 = crate::utils::session_status::image_text(&msgs);
                c.0 = key_len;
                c.1 = key_uuid;
            }
            c.2
        }
        None => CACHE.lock().2,
    };
    let images = {
        let c = CACHE.lock();
        if c.3.is_empty() { String::new() } else { format!("{} ", c.3) }
    };
    if used <= 0 {
        return Vec::new();
    }

    let pct = crate::utils::session_status::fill_percent(used, ctx_limit);
    let style = if pct >= CTX_GAUGE_RED_PCT {
        "\x1b[31;1m"
    } else if pct >= CTX_GAUGE_YELLOW_PCT {
        "\x1b[33m"
    } else {
        "\x1b[38;5;245m"
    };
    let cols = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(80);
    let counts = format!("{}/{}", format_tokens_compact(used), format_tokens_compact(ctx_limit));
    let text = if cols >= 80 {
        let filled = std::cmp::min(CTX_BAR_WIDTH, ((CTX_BAR_WIDTH as f64 * pct as f64 / 100.0).round()) as usize);
        let bar = format!(
            "{}{}",
            sym("bar_full").repeat(filled),
            sym("bar_empty").repeat(CTX_BAR_WIDTH - filled)
        );
        format!("{images}{bar} {pct}% {} {counts}", sym("dot"))
    } else if cols >= 60 {
        format!("{images}{pct}% {} {counts}", sym("dot"))
    } else {
        format!("{pct}%")
    };
    vec![Frag::styled(style, format!(" {text} "))]
}

/// Main REPL entry (repl_loop).
pub async fn repl_loop(
    tools: Vec<Arc<dyn Tool>>,
    initial_prompt: String,
    dangerously_skip_permissions: bool,
    verbose: bool,
    debug: bool,
    initial_messages: Option<Vec<Message>>,
) {
    let messages = Arc::new(Mutex::new(initial_messages.unwrap_or_default()));
    let model = get_config().model_name;
    let system_prompt = crate::constants::prompts::get_system_prompt(&tools, Some(&model), None).await;
    // Pin the skill listing to session start, matching the tool block that is
    // about to be built and then frozen. Without this the snapshot would be
    // taken at whatever moment something first asked, which is not a moment
    // anyone can reason about.
    crate::utils::skills::pin_session_skills();

    let mut context: BTreeMap<String, String> = BTreeMap::new();
    // cwd and date are NOT seeded here: they are per-turn values and the system
    // prompt is the cached prefix (see process_prompt). The session's starting
    // directory is still stated once in <env>; the live one rides the turn.

    set_session_dsp(dangerously_skip_permissions);
    let ctx = ToolUseContext {
        options: ContextOptions {
            tools: tools.clone(),
            dangerously_skip_permissions,
            slow_and_capable_model: None,
        },
        ..Default::default()
    };

    // Reconcile the configured model with what the server actually serves
    // BEFORE the first request, so a stale/renamed id self-heals.
    crate::utils::tokens::resolve_model();
    let ctx_size = crate::utils::tokens::get_context_size();

    // Silence the kernel echo before the banner: the prompt engine does not own
    // the terminal yet, and anything typed during startup would otherwise be
    // painted straight across these lines.
    crate::utils::tty_compat::suppress_echo();

    cmprintln!("{}", "-".repeat(63));
    cmprintln!("{PRODUCT_NAME} v{VERSION} ({})", crate::cli::BUILD_ID);
    cmprintln!("Working directory: {}", crate::utils::state::get_cwd());
    cmprintln!(
        "Context: {} tokens (auto-compact at {})",
        crate::utils::formatting::format_number(ctx_size),
        crate::utils::formatting::format_number(
            crate::services::compaction::get_auto_compact_threshold().unwrap_or(0)
        )
    );
    cmprintln!("Tools available: {}", tools.len());
    let bcfg = get_config();
    let beff = { let e = bcfg.effective_reasoning_effort(); if e.is_empty() { "default".to_string() } else { e } };
    cmprintln!(
        "Model: {} | Reasoning effort: {beff} (change: /effort)",
        if bcfg.model_name.is_empty() { "(server default)".to_string() } else { bcfg.model_name.clone() }
    );
    // Sampling actually in effect. Printed only when the family branch diverges
    // from what config.json says, because that divergence is deliberate and must
    // not be silent — a value quietly different from the file on disk is the
    // exact failure this session spent its day chasing.
    {
        let eff = bcfg.effective_sampling();
        let manual = bcfg.sampling_profile.trim().eq_ignore_ascii_case("manual");
        let mut parts = vec![format!("temperature {}", eff.temperature)];
        parts.push(format!("top_p {}", eff.top_p));
        if let Some(k) = eff.top_k {
            parts.push(format!("top_k {k}"));
        }
        // The VALUES are the point: they stop a silent divergence between the
        // sampling in effect and what config.json says. The paragraph that used
        // to follow them -- which profile to set, which value the file holds,
        // that the family default is "recommended" -- was scaffolding from the
        // investigation into the collapse and repetition symptoms. Those are
        // fixed, so it now explains a problem that no longer happens, once per
        // launch, in 156 columns against a 63-column frame: 93 of them outside
        // the rule it was printed under, which is where a banner stops being
        // read at all. How to change them belongs in /help.
        //
        // The one thing kept is the disagreement itself, and only when there is
        // one, because that is the part no other line on screen can tell you.
        let note = if manual || (bcfg.temperature - eff.temperature).abs() < 1e-9 {
            String::new()
        } else {
            format!(" (config.json: {})", bcfg.temperature)
        };
        cmprintln!(
            "{}",
            colorize(&format!("Sampling: {}{note}", parts.join(", ")), colors::DIM)
        );
    }

    // The served model's OWN tokenizer decides every token number in this
    // process: the auto-compact threshold, the context-window check, the usage
    // readout. Without it those come from a character heuristic — estimates,
    // not counts — and compaction can fire late enough to overrun the window.
    //
    // This used to be silent. tokenizer/ resolves next to the binary, exactly
    // like vendor/ripgrep, and setup did not install it, so every run since
    // fell back without a word while reporting the estimates as if exact.
    // ripgrep's absence has printed a WARNING since day one; this one printed
    // nothing. Say it, in red, once per run.
    if !crate::utils::tokens::has_model_tokenizer() {
        let red_bold = format!("{}{}", colors::BOLD, colors::RED);
        let model = if bcfg.model_name.is_empty() {
            "(server default)".to_string()
        } else {
            bcfg.model_name.clone()
        };
        cmprintln!();
        cmprintln!("{}", colorize(&"!".repeat(63), &red_bold));
        cmprintln!(
            "{}",
            colorize("  WARNING: the model tokenizer was NOT found.", &red_bold)
        );
        cmprintln!(
            "{}",
            colorize(
                &format!("  Token counts are ESTIMATES, not counts (model: {model})."),
                colors::RED
            )
        );
        cmprintln!(
            "{}",
            colorize(
                "  Auto-compact and context-window checks may be wrong as a result.",
                colors::RED
            )
        );
        cmprintln!(
            "{}",
            colorize(
                "  Fix: install tokenizer/<model>/tokenizer.json beside the binary",
                colors::RED
            )
        );
        cmprintln!(
            "{}",
            colorize("        (re-run setup.sh / setup.bat from the repo).", colors::RED)
        );
        cmprintln!("{}", colorize(&"!".repeat(63), &red_bold));
        cmprintln!();
    }

    let bg = crate::utils::goal::get_goal();
    if !bg.is_empty() {
        cmprintln!("Goal: {bg}  (change: /goal)");
    }
    cmprintln!("Type /exit to quit, /clear to clear history, /help for commands");
    // keep-going status line. ON is self-installing: make sure the Stop hook is
    // materialized on THIS machine.
    if crate::cli::keep_going_enabled_pub() {
        let actions = crate::utils::keep_going_installer::ensure_installed();
        cmprintln!(
            "keep-going: on{} (toggle: /keepgoing on|off)",
            if actions.is_empty() { "" } else { "  (installed Stop hook)" }
        );
    } else if crate::utils::hooks::has_hook_for_event("Stop") {
        cmprintln!("keep-going: off (toggle: /keepgoing on|off)");
    }
    cmprintln!("{}", "-".repeat(63));

    if !messages.lock().is_empty() {
        let banner = {
            let m = messages.lock();
            crate::cli::render_resume_banner_pub(&m)
        };
        cmprint!("{banner}");
    }

    if plain_mode() {
        // --plain/--ascii: those terminals can't do the cursor-repositioning
        // escapes the persistent-prompt UI needs, so they keep the sequential
        // loop. No input router here, and therefore no remote prompting —
        // deliberately (see the Python for the full reasoning).
        crate::cli::warn_remote_off_in_plain_pub();
        plain_repl(messages, system_prompt, context, ctx, verbose, debug, initial_prompt).await;
    } else {
        interactive_repl(messages, system_prompt, context, ctx, verbose, debug, initial_prompt).await;
    }
}

// === Main Entry Point ===

/// The parsed command line (the Python's argparse namespace).
#[derive(Default)]
struct Args {
    prompt: String,
    cwd: String,
    print_mode: bool,
    debug: bool,
    verbose: bool,
    output_format: String,
    input_format: String,
    dangerously_skip_permissions: bool,
    no_remote: bool,
    no_dsp: bool,
    resume: Option<String>,
    continue_session: bool,
    plain: bool,
    realtime: bool,
    no_realtime: bool,
    debug_log: bool,
    theme: Option<String>,
    keep_going: Option<bool>,
    /// Operator directive: free text pinned to the END of the system prompt.
    /// See constants::prompts::load_operator_directive.
    directive: Option<String>,
    /// Restrict which tools the model is offered (see tools::set_tool_filter).
    allowed_tools: String,
    disallowed_tools: String,
    /// -np/--no-plan: drop EnterPlanMode/ExitPlanMode for this run.
    no_plan: bool,
}

/// The turn a finished background task starts on its own.
///
/// Not empty: run_turn sends what it is given, and a blank line reads as the
/// user having pressed enter for no reason. The notification blocks are
/// already in the history by the time this runs, so this only has to say why
/// the model is being addressed again.
const TASK_WAKE_PROMPT: &str = "A background task you started has finished — its <task-notification> is above, with the path to its output. Pick the work back up from there.";

const USAGE: &str = concat!(
    "usage: codemason [-h] [-C CWD] [-p] [-d] [-v] [--output-format {text,json,stream-json}]\n",
    "                 [--input-format {text,stream-json}] [-dsp] [--no-remote] [--no-dsp]\n",
    "                 [--allowed-tools NAMES] [--disallowed-tools NAMES] [-np]\n",
    "                 [-r SESSION_ID] [-c] [--plain] [-rt] [--no-rt] [--theme NAME]\n",
    "                 [--keep-going] [--no-keep-going] [-dbg] [--version]\n",
    "                 [--directive TEXT | --directive-file PATH]\n",
    "                 [prompt]\n",
    "\n",
    "CodeMason - AI coding assistant\n",
    "\n",
    "subcommands:\n",
    "  mcp                   manage MCP servers\n",
    "  skill                 manage skills\n",
    "  whoami                this session's id, cwd and remote state\n",
    "  inject                wake a running session with a message\n",
    "  notify                push an alert to the phone\n",
    "\n",
    "positional arguments:\n",
    "  prompt                Initial prompt (optional)\n",
    "\n",
    "options:\n",
    "  -h, --help            show this help message and exit\n",
    "  -C CWD, --cwd CWD     Working directory (git-style -C; -c now means --continue)\n",
    "  -p, --print           Print response and exit (non-interactive)\n",
    "  -d, --debug           Enable debug mode\n",
    "  -dbg, --debug-log     Full logging to /tmp/codemason-dbg-<pid>/ (turns.log +\n",
    "                        debug.log, append-only, safe to tail while running)\n",
    "  -v, --verbose         Verbose output\n",
    "  --output-format {text,json,stream-json}\n",
    "                        Print-mode output format: text (default), json (single result\n",
    "                        object), or stream-json (NDJSON: init, each message, result).\n",
    "                        stream-json requires --verbose.\n",
    "  --input-format {text,stream-json}\n",
    "                        Print-mode input format: text (default) or stream-json (stdin is\n",
    "                        NDJSON of user messages).\n",
    "  -dsp, --dangerously-skip-permissions\n",
    "                        Skip permission checks (-dsp is shorthand)\n",
    "  --no-remote           Turn the remote (phone) channel off for this run only\n",
    "  --allowed-tools NAMES\n",
    "                        Offer only these tools (comma-separated, e.g.\n",
    "                        Bash,Grep,Glob,Read). Anything else is never shown to the\n",
    "                        model, so it cannot be called.\n",
    "  --disallowed-tools NAMES\n",
    "                        Remove these tools even if --allowed-tools lists them.\n",
    "  -np, --no-plan        Drop the plan-mode tools (EnterPlanMode/ExitPlanMode)\n",
    "                        from this run. Their descriptions are ~1,400 tokens of\n",
    "                        the fixed prefix resent every turn; a session that never\n",
    "                        plans pays it for nothing.\n",
    "  --no-dsp, --ask       Ask for tool approvals even if config enables skipping\n",
    "  -r SESSION_ID, --resume SESSION_ID\n",
    "                        Resume a stored session by id and continue it\n",
    "  -c, --continue        Continue the most recently modified session of this directory\n",
    "  --plain, --ascii      Pure-ASCII compatibility mode\n",
    "  -rt, --realtime       Stream the model reply as it generates (on by default)\n",
    "  --no-rt, --no-realtime\n",
    "                        Buffer the whole turn and print it at the end\n",
    "  --theme NAME          Repaint the terminal for this session via OSC (opt-in)\n",
    "  --directive TEXT      Operator directive pinned to the END of the system prompt, so\n",
    "                        it outranks the built-in sections. Set once at startup and\n",
    "                        kept for the session (it is part of the cached prefix).\n",
    "                        Also readable from CODEMASON_DIRECTIVE.\n",
    "  --directive-file PATH Read the directive from a file instead.\n",
    "  --keep-going          Keep the model working past natural turn-ends (default: on)\n",
    "  --no-keep-going       Disable keep-going for this session\n",
    "  --version             show program's version number and exit\n",
);

/// --disallowed-tools plus the plan tools when -np/--no-plan is given.
///
/// Its own function so the one rule that composes the two is testable without
/// standing up the whole REPL.
fn disallowed_with_plan(disallowed: &str, no_plan: bool) -> String {
    if !no_plan {
        return disallowed.to_string();
    }
    use crate::tools::plan_mode_tools::{ENTER_PLAN_MODE_TOOL_NAME, EXIT_PLAN_MODE_TOOL_NAME};
    let mut out = disallowed.to_string();
    if !out.is_empty() {
        out.push(',');
    }
    out.push_str(ENTER_PLAN_MODE_TOOL_NAME);
    out.push(',');
    out.push_str(EXIT_PLAN_MODE_TOOL_NAME);
    out
}

fn parse_args(argv: &[String]) -> Args {
    let mut args = Args {
        cwd: crate::utils::state::get_cwd(),
        output_format: "text".into(),
        input_format: "text".into(),
        ..Default::default()
    };
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut take_value = |i: &mut usize, name: &str| -> String {
            *i += 1;
            match argv.get(*i) {
                Some(v) => v.clone(),
                None => {
                    eprintln!("codemason: error: argument {name}: expected one argument");
                    std::process::exit(2);
                }
            }
        };
        match a {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--version" => {
                println!("{PRODUCT_NAME} {VERSION}");
                std::process::exit(0);
            }
            "-C" | "--cwd" => args.cwd = take_value(&mut i, "-C/--cwd"),
            "-p" | "--print" => args.print_mode = true,
            "-d" | "--debug" => args.debug = true,
            "-v" | "--verbose" => args.verbose = true,
            "--output-format" => {
                let v = take_value(&mut i, "--output-format");
                if !crate::print_runner::VALID_OUTPUT_FORMATS.contains(&v.as_str()) {
                    eprintln!(
                        "codemason: error: argument --output-format: invalid choice: '{v}' (choose from 'text', 'json', 'stream-json')"
                    );
                    std::process::exit(2);
                }
                args.output_format = v;
            }
            "--input-format" => {
                let v = take_value(&mut i, "--input-format");
                if !crate::print_runner::VALID_INPUT_FORMATS.contains(&v.as_str()) {
                    eprintln!(
                        "codemason: error: argument --input-format: invalid choice: '{v}' (choose from 'text', 'stream-json')"
                    );
                    std::process::exit(2);
                }
                args.input_format = v;
            }
            "-dsp" | "--dangerously-skip-permissions" => args.dangerously_skip_permissions = true,
            "--no-remote" => args.no_remote = true,
            "--allowed-tools" => args.allowed_tools = take_value(&mut i, "--allowed-tools"),
            "--disallowed-tools" => {
                args.disallowed_tools = take_value(&mut i, "--disallowed-tools")
            }
            "-np" | "--no-plan" => args.no_plan = true,
            "--no-dsp" | "--ask" => args.no_dsp = true,
            "-r" | "--resume" => args.resume = Some(take_value(&mut i, "-r/--resume")),
            "-c" | "--continue" => args.continue_session = true,
            "--plain" | "--ascii" => args.plain = true,
            "-rt" | "--realtime" => args.realtime = true,
            // -dbg: full logging, to a path someone else can watch.
            //
            // The two records that explain a slow turn — the per-request line
            // (prefill / think / output) and the debug trace — were gated
            // behind an env var that had to be set BEFORE the run that went
            // wrong. By the time a 500-second stall has happened, it is too
            // late to turn them on, and the evidence for it does not exist.
            // This makes turning them on a flag, and puts the files somewhere
            // append-only and readable so they can be tailed as the run goes
            // rather than reconstructed afterwards.
            "-dbg" | "--debug-log" => args.debug_log = true,
            "--no-rt" | "--no-realtime" => args.no_realtime = true,
            "--theme" => args.theme = Some(take_value(&mut i, "--theme")),
            "--directive" => args.directive = Some(take_value(&mut i, "--directive")),
            "--directive-file" => {
                let path = take_value(&mut i, "--directive-file");
                match std::fs::read_to_string(&path) {
                    Ok(text) => args.directive = Some(text),
                    Err(e) => {
                        eprintln!("codemason: error: --directive-file {path}: {e}");
                        std::process::exit(2);
                    }
                }
            }
            "--keep-going" => args.keep_going = Some(true),
            "--no-keep-going" => args.keep_going = Some(false),
            other if other.starts_with('-') && other.len() > 1 => {
                eprintln!("{USAGE}");
                eprintln!("codemason: error: unrecognized arguments: {other}");
                std::process::exit(2);
            }
            other => {
                if args.prompt.is_empty() {
                    args.prompt = other.to_string();
                } else {
                    eprintln!("{USAGE}");
                    eprintln!("codemason: error: unrecognized arguments: {other}");
                    std::process::exit(2);
                }
            }
        }
        i += 1;
    }
    args
}

/// Main entry point.
pub fn main() {
    // Before argument parsing: the help text contains an em dash, which a
    // legacy Windows codepage cannot encode once stdout is redirected.
    crate::utils::stdio_encoding::ensure_utf8_stdio();

    let argv: Vec<String> = std::env::args().skip(1).collect();

    // `codemason mcp ...` / `codemason skill ...` — management subcommands,
    // intercepted before argument parsing (which owns the positional prompt).
    if argv.first().map(|s| s.as_str()) == Some("mcp") {
        std::process::exit(crate::utils::mcp_cli::handle_mcp_cli(&argv[1..]));
    }
    if argv.first().map(|s| s.as_str()) == Some("skill") {
        std::process::exit(crate::utils::skill_cli::handle_skill_cli(&argv[1..]));
    }
    // The self-wake surface: who am I, wake a session, buzz the phone. Same
    // interception point and the same reason — these answer and exit, they
    // never open a session.
    if argv.first().map(|s| s.as_str()) == Some("whoami") {
        std::process::exit(crate::utils::wake_cli::handle_whoami_cli(&argv[1..]));
    }
    if argv.first().map(|s| s.as_str()) == Some("inject") {
        std::process::exit(crate::utils::wake_cli::handle_inject_cli(&argv[1..]));
    }
    if argv.first().map(|s| s.as_str()) == Some("notify") {
        std::process::exit(crate::utils::wake_cli::handle_notify_cli(&argv[1..]));
    }

    // Before any child spawn: give git-bash/MSYS2 a UTF-8 charset so it doesn't
    // mangle non-ASCII paths in tool output. No-op off Windows.
    crate::utils::windows_shell::ensure_utf8_locale();

    let mut args = parse_args(&argv);

    // Tool restriction. Installed before anything can build a tool list, so a
    // tool that is not permitted is never offered to the model — not shown and
    // then refused, simply absent.
    //
    // -np/--no-plan is sugar on top of it. The plan tools are removed through
    // the same channel rather than a second gate of their own — one place
    // that drops a tool is one place to audit.
    let disallowed_tools = disallowed_with_plan(&args.disallowed_tools, args.no_plan);
    if !args.allowed_tools.is_empty() || !disallowed_tools.is_empty() {
        crate::tools::set_tool_filter(&args.allowed_tools, &disallowed_tools);
    }

    // keep-going session override -> the env the Stop hook reads. Left unset
    // when neither flag is given, so the hook falls back to config.json.
    if let Some(kg) = args.keep_going {
        std::env::set_var("CODEMASON_KEEP_GOING", if kg { "1" } else { "0" });
    }

    // Operator directive -> the env the prompt assembly reads. Set before any
    // prompt is built. A flag beats an inherited CODEMASON_DIRECTIVE, so a bot
    // pool can set a default in the environment and a single run can override it.
    if let Some(d) = args.directive.as_deref() {
        std::env::set_var("CODEMASON_DIRECTIVE", d);
    }

    // Pure-ASCII mode: set the env the formatting layer reads.
    if args.plain {
        std::env::set_var("CODEMASON_PLAIN", "1");
    }

    // -rt/--realtime: streaming is the default, so the flag that actually
    // decides anything is --no-rt. Resolution mirrors the permission flags:
    // --no-rt > -rt > config. Runtime only — never written back.
    // -dbg: turn full logging on and put it where it can be watched.
    // Set BEFORE anything logs — LOG_DIR and the debug gate are read once,
    // lazily, on first use.
    if args.debug_log {
        // env::temp_dir(), not a literal "/tmp": on native Windows that literal
        // resolves to C:\tmp at the drive root, which is not the temp directory
        // and is not always writable. Same resolution the task output files
        // already use (utils::task::disk_output).
        let dir = std::env::temp_dir().join(format!("codemason-dbg-{}", std::process::id()));
        match std::fs::create_dir_all(&dir) {
            Ok(()) => {
                std::env::set_var("CODEMASON_DEBUG", "1");
                std::env::set_var("CODEMASON_LOG_DIR", &dir);
                cmprintln!("  {} debug logging -> {}", sym("connector"), dir.display());
                cmprintln!("     {}/turns.log   one line per request (prefill / think / output)", dir.display());
                cmprintln!("     {}/debug.log   full trace", dir.display());
            }
            // The directory was created with `let _ =` before, so a run that
            // could not make it announced three log files and then wrote none,
            // which is the one case -dbg exists to rule out.
            Err(e) => cmprintln!("  {} -dbg: cannot create {} ({e}) — logging stays off",
                                 sym("warn"), dir.display()),
        }
    }

    if args.no_realtime {
        crate::utils::config::set_runtime_override("realtime", Value::Bool(false));
    } else if args.realtime {
        crate::utils::config::set_runtime_override("realtime", Value::Bool(true));
    }

    // Optional terminal theme (OSC repaint, opt-in, restored on exit). Applied
    // after plain-mode is set so it correctly no-ops in pure-ASCII mode.
    let theme = args.theme.clone().or_else(|| std::env::var("CODEMASON_THEME").ok());
    if let Some(t) = theme.filter(|t| !t.is_empty()) {
        crate::utils::terminal_theme::apply_theme(&t);
    }

    // The effective value for skipping approvals: the FILE is the default and
    // the -dsp flag overrides it (a flag default of False must not overwrite
    // what the file said).
    let dsp = !args.no_dsp
        && (args.dangerously_skip_permissions || get_config().dangerously_skip_permissions);

    // --no-remote applies to this run only, so it goes in the environment.
    if args.no_remote {
        std::env::set_var("CODEMASON_REMOTE", "0");
    }

    setup(&args.cwd, dsp);

    // -c/--continue = resume the most recently modified session of this
    // directory. An explicit --resume <id> wins.
    if args.continue_session && args.resume.is_none() {
        let recent = crate::utils::session_storage::list_sessions(Some(1));
        if recent.is_empty() {
            eprintln!("Error: --continue: no stored sessions for this directory");
            std::process::exit(1);
        }
        // Honor a /clear: if the last real session predates a deliberate clear,
        // start fresh instead of silently resuming the pre-clear conversation.
        if crate::utils::session_storage::cleared_after_session(&recent[0].session_id) {
            println!("Starting a fresh session (the last one was cleared with /clear).");
        } else {
            args.resume = Some(recent[0].session_id.clone());
        }
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: could not start the async runtime: {e}");
            std::process::exit(1);
        }
    };

    runtime.block_on(async move {
        let tools = crate::tools::get_all_tools().await;

        if args.print_mode {
            // stream-json output requires --verbose (cli/print.ts:787-790).
            if args.output_format == "stream-json" && !args.verbose {
                eprintln!("Error: When using --print, --output-format=stream-json requires --verbose");
                std::process::exit(1);
            }
            let Some(prompt) = crate::print_runner::read_prompt_from_input(&args.prompt, &args.input_format)
            else {
                eprintln!("Error: Input must be provided either through stdin or as a prompt argument when using --print");
                std::process::exit(1);
            };
            let exit_code = crate::print_runner::run_print(
                &prompt,
                tools,
                &args.output_format,
                args.verbose,
                dsp,
                args.resume.as_deref(),
                permission_callback(),
            )
            .await;
            crate::remote::notify::drain_at_exit().await;
            crate::utils::terminal_theme::restore();
            std::process::exit(exit_code);
        }

        // Interactive REPL — --resume <id> restores the stored conversation and
        // continues appending to the same transcript file.
        let mut initial_messages: Option<Vec<Message>> = None;
        if let Some(rid) = &args.resume {
            match crate::utils::session_storage::load_conversation_for_resume(rid) {
                Ok(values) => {
                    crate::utils::session_storage::adopt_resumed_session(rid);
                    crate::cli::observe_restored_turns_pub(&values);
                    initial_messages = Some(crate::utils::messages::messages_from_transcript(&values));
                }
                Err(e) => {
                    eprintln!("Error: --resume {rid}: {e}");
                    std::process::exit(1);
                }
            }
        }
        repl_loop(tools, args.prompt.clone(), dsp, args.verbose, args.debug, initial_messages).await;
        crate::utils::terminal_theme::restore();
    });
}

#[cfg(test)]
mod tests {
    /// Mashing Ctrl+C to stop a turn must not be able to kill the agents.
    ///
    /// Each press does the most urgent thing it can and only falls through when
    /// there is nothing more urgent — the shape the source's cancel handler
    /// uses ("Priority 1: if there's an active task running, cancel it
    /// first..."). So a press that brakes never arms the bulk stop, and the
    /// gesture is reachable only from an idle prompt.
    #[tokio::test]
    async fn a_ctrl_c_that_brakes_never_arms_the_agent_stop() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();

        let agent_def = crate::tools::built_in_agents::get_agent_definition(None)
            .expect("the default agent definition");
        let mut tokens = Vec::new();
        for i in 0..2 {
            let token = tokio_util::sync::CancellationToken::new();
            crate::tasks::local_agent_task::register_async_agent(
                &format!("a{}{}", "0".repeat(15), i),
                &format!("lane {i}"),
                "p",
                agent_def.clone(),
                None,
                Some(token.clone()),
            );
            tokens.push(token);
        }

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let router = super::ReplInputRouter::new(tx, std::sync::Arc::new(parking_lot::Mutex::new(Vec::new())));

        // A turn is running: every press brakes, none of them arms.
        router.set_processing(true);
        for _ in 0..3 {
            assert!(router.hard_stop(), "a braking press reports it did something");
        }
        assert!(
            tokens.iter().all(|t| !t.is_cancelled()),
            "mashing Ctrl+C during a turn leaves background agents alone"
        );
        assert!(!router.kill_agents.is_armed(), "and never arms the gesture");

        // Idle: now the presses reach the gesture, and it still takes two.
        router.set_processing(false);
        assert!(router.hard_stop());
        assert!(router.kill_agents.is_armed());
        assert!(tokens.iter().all(|t| !t.is_cancelled()), "the first one only arms");
        assert!(router.hard_stop());
        assert!(tokens.iter().all(|t| t.is_cancelled()), "the second one stops them");
        assert_eq!(
            crate::utils::message_queue_manager::drain_pending_notifications().len(),
            1,
            "and the model is told once"
        );

        // Nothing left: back to the fall-through that prints the quit hint.
        assert!(!router.hard_stop());

        crate::state::app_state::reset_app_state();
        crate::utils::message_queue_manager::reset_pending_notifications();
    }

    /// A slash command must survive leading whitespace, including the
    /// multi-byte kind that arrives from a paste.
    ///
    /// `first` is taken with split_whitespace, so "  /help" reached the body
    /// extraction and the old `user_input[1..]` sliced the space off instead of
    /// the slash — cmd became "/help", matched no arm, and the command silently
    /// did nothing. With U+00A0 (which char::is_whitespace accepts) byte 1 lands
    /// inside the character and the slice panics outright.
    #[test]
    fn slash_command_body_survives_leading_whitespace() {
        fn body(s: &str) -> String {
            s.trim_start().strip_prefix('/').unwrap_or("").to_string()
        }
        fn cmd(s: &str) -> String {
            body(s).split_whitespace().next().unwrap_or("").to_lowercase()
        }
        assert_eq!(cmd("/help"), "help");
        assert_eq!(cmd("  /help"), "help", "leading spaces must not shift the cut");
        assert_eq!(cmd("\u{a0}/help"), "help", "a multi-byte space must not panic or shift");
        assert_eq!(cmd("\t/goal show"), "goal");
        assert_eq!(body("  /goal do a thing"), "goal do a thing");
        // Not a command at all: no slash to strip, and no panic.
        assert_eq!(cmd("  hello"), "");
    }

    /// Leaving a paused scope by unwinding must still resume the prompt.
    ///
    /// The sub-UIs that pause — the rewind and resume pickers, the output
    /// pager, the question picker — used to resume with a plain call at the
    /// end of the block, which a panic jumps straight over. The engine then
    /// stays suspended for the rest of the process: the screen looks normal
    /// and every keystroke is discarded, which reads as a hang rather than a
    /// crash.
    #[test]
    fn prompt_pause_guard_resumes_even_when_unwinding() {
        use super::*;

        let (tx, _rx) = mpsc::unbounded_channel();
        let router = ReplInputRouter::new(tx, Arc::new(Mutex::new(Vec::new())));
        let rt = tokio::runtime::Runtime::new().expect("runtime");

        assert!(!router.is_paused());

        // Ordinary exit.
        rt.block_on(async {
            let _pause = router.pause_prompt_scoped().await;
            assert!(router.is_paused(), "scope should be paused");
        });
        assert!(!router.is_paused(), "guard must resume when the scope ends");

        // Exit by panic — the case a trailing resume_prompt() never covered.
        rt.block_on(async {
            let guard = router.pause_prompt_scoped().await;
            assert!(router.is_paused());
            let inner = router.clone();
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _owned = guard; // dropped as the panic unwinds out
                assert!(inner.is_paused());
                panic!("a sub-UI blew up");
            }));
            assert!(caught.is_err(), "the panic should have been caught here");
        });
        assert!(
            !router.is_paused(),
            "a panic inside a paused scope must not leave the prompt parked"
        );
    }

    /// -np/--no-plan removes exactly the two plan tools, through the same
    /// --disallowed-tools channel, without eating an existing list.
    #[test]
    fn no_plan_appends_the_plan_tools_to_disallowed() {
        assert_eq!(super::disallowed_with_plan("", false), "");
        assert_eq!(super::disallowed_with_plan("Bash", false), "Bash");
        assert_eq!(
            super::disallowed_with_plan("", true),
            "EnterPlanMode,ExitPlanMode"
        );
        assert_eq!(
            super::disallowed_with_plan("Bash,WebFetch", true),
            "Bash,WebFetch,EnterPlanMode,ExitPlanMode"
        );
    }

    /// The short form is a real flag, not a prefix match of --no-plan.
    #[test]
    fn np_short_flag_parses() {
        let argv = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(super::parse_args(&argv(&["-np"])).no_plan);
        assert!(super::parse_args(&argv(&["--no-plan"])).no_plan);
        assert!(!super::parse_args(&argv(&[])).no_plan);
    }
}
