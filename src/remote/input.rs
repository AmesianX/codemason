//! Remote input — messages from the phone become session input.
//! Equivalent to codemason/remote/input.py — the poller files inbound messages
//! into per-session inboxes, and a session drains its own into the queue its
//! prompt already reads from. Routing is by session id and nothing else; an
//! ambiguous message is parked behind a picker, never guessed at. See the
//! Python module doc throughout.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::remote::inbox;
use crate::utils::i18n::{tr, trf, FmtArg};

/// How often a session checks its inbox. Cheap, and well under the threshold
/// where typing feels ignored.
pub const DRAIN_INTERVAL: f64 = 1.0;

/// How often a running session reclaims its own leftovers (claim_message
/// writes one file per outbound alert; a long-lived session must collect its
/// own aged records rather than wait for the NEXT session to start).
pub const SWEEP_INTERVAL: f64 = 900.0;

/// Claim a bot for this session, preferring the one this session slot used —
/// continuity, remembered per session slot (the label), stored as a
/// fingerprint (never the token) in project settings.
async fn claim_bot_for(cwd: &str, label: &str) -> bool {
    use crate::remote::adapters::telegram as tg;

    let saved = crate::utils::session_storage::read_project_settings(cwd);
    let slot_key = format!("remote_bot:{label}");
    let remembered = saved
        .get(&slot_key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            // Directories that only ever ran one session still carry the old
            // unsuffixed key; honour it for the first slot.
            if label == inbox::short_label(cwd) {
                saved.get("remote_bot").and_then(|v| v.as_str()).map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let pool = tg::bot_pool();
    let preferred = pool
        .iter()
        .find(|t| tg::token_tag(t) == remembered)
        .cloned()
        .unwrap_or_default();

    let Some(token) = tg::claim_bot(&preferred).await else {
        return false;
    };
    let _ = crate::utils::session_storage::write_project_setting(
        &slot_key,
        Value::String(tg::token_tag(&token)),
        cwd,
    );
    // A claimed bot keeps the name it already has — renaming is rate limited
    // per bot on an hours-long cooldown; `/rc <name>` renames, nothing else.
    true
}

/// Say why the phone is silent, naming the situation that actually occurred
/// (three causes, three different fixes). Printed rather than logged.
pub fn report_no_free_bot(pool_size: usize) {
    if pool_size == 0 {
        return; // unconfigured is not a shortfall
    }
    use crate::remote::adapters::telegram as tg;
    let verdicts = tg::last_claim_verdicts();
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for v in verdicts.values() {
        *counts.entry(v).or_insert(0) += 1;
    }
    let taken = counts.get(tg::TAKEN).copied().unwrap_or(0);
    let invalid = counts.get(tg::INVALID).copied().unwrap_or(0);
    let unreachable = counts.get(tg::UNREACHABLE).copied().unwrap_or(0);

    if unreachable > 0 && taken == 0 {
        crate::cmprintln!("{}", tr("remote: off — could not reach Telegram to claim a bot."));
        crate::cmprintln!(
            "{}",
            tr("  Nothing is wrong with your bots; the channel starts working again next time the network does.")
        );
        return;
    }

    if invalid > 0 && taken == 0 && unreachable == 0 {
        crate::cmprintln!(
            "{}",
            trf(
                "remote: off — Telegram rejected {n} of {total} token(s) in ~/.codemason/remote.env.",
                &[("n", FmtArg::Int(invalid as i64)), ("total", FmtArg::Int(pool_size as i64))]
            )
        );
        crate::cmprintln!(
            "{}",
            tr("  A rejected token was revoked, mistyped, or belongs to a deleted bot. Check it with @BotFather.")
        );
        return;
    }

    crate::cmprintln!(
        "{}",
        trf(
            "remote: no free bot — every usable one of the {n} in ~/.codemason/remote.env is already taken by another instance.",
            &[("n", FmtArg::Int(pool_size as i64))]
        )
    );
    crate::cmprintln!(
        "{}",
        tr("  Telegram delivers one bot to exactly one process, so each instance needs its own.")
    );
    crate::cmprintln!("{}", tr("  Create another with @BotFather and add a TELEGRAM_BOT_TOKEN line."));
    if invalid > 0 {
        crate::cmprintln!(
            "{}",
            trf(
                "  ({n} token(s) were also rejected — revoked or mistyped.)",
                &[("n", FmtArg::Int(invalid as i64))]
            )
        );
    }
    if unreachable > 0 {
        crate::cmprintln!(
            "{}",
            trf("  ({n} could not be checked right now.)", &[("n", FmtArg::Int(unreachable as i64))])
        );
    }
}

/// on_message callback: (text, age_seconds, source).
///
/// `source` rides along so the echo can say where a line came from. A message
/// the local `inject` command queued is not a message from the phone, and
/// showing it as one makes the user doubt their own history — they see a line
/// they never typed, attributed to a device they were not holding.
pub type OnInputMessage = Arc<dyn Fn(String, f64, String) + Send + Sync>;

struct RemoteInputState {
    session_id: String,
    label: String,
    started: bool,
    /// Whether this session holds a poller reference. The inbox runs either
    /// way, so stop() must not release a reference it never took.
    polling: bool,
    last_sweep: Option<std::time::Instant>,
    stop: CancellationToken,
}

/// Wires the poller and one session's inbox to that session's input queue.
pub struct RemoteInput {
    pub cwd: String,
    on_message: Option<OnInputMessage>,
    state: Mutex<RemoteInputState>,
}

/// This process's instance — the reference survives a stop, because /remote on
/// has to reach the same object (a fresh one would register the session twice).
static ACTIVE: Lazy<Mutex<Option<Arc<RemoteInput>>>> = Lazy::new(|| Mutex::new(None));

impl RemoteInput {
    pub fn new(session_id: &str, cwd: &str, on_message: Option<OnInputMessage>) -> Arc<RemoteInput> {
        Arc::new(RemoteInput {
            cwd: cwd.to_string(),
            on_message,
            state: Mutex::new(RemoteInputState {
                session_id: session_id.to_string(),
                label: String::new(),
                started: false,
                polling: false,
                last_sweep: None,
                stop: CancellationToken::new(),
            }),
        })
    }

    pub fn label(&self) -> String {
        self.state.lock().label.clone()
    }

    fn session_id(&self) -> String {
        self.state.lock().session_id.clone()
    }

    // --- lifecycle -----------------------------------------------------------

    /// Register the session, start the poller, begin draining. Returns false
    /// when remote input is unavailable (no credentials / no free bot). Never
    /// raises: a session must run identically with or without a phone.
    /// Arm the inbox, and the phone leg too when there is a bot for it.
    ///
    /// These are two different things and used to be one. A session's inbox is
    /// local: anything on this machine can drop a message in it and the drain
    /// loop hands it to the prompt a second later — `codemason inject`, a
    /// watcher script, a build hook, a cron line. Telegram is one way to reach
    /// that inbox, not what makes it exist. Tying them together meant a machine
    /// with no bot tokens had no way to wake its own session at all, so an
    /// agent could not arrange to be called back by work it had started, which
    /// is the plainer half of the feature and the half that needs no account
    /// anywhere.
    ///
    /// So registration and the drain loop run regardless; the claim and the
    /// poller are skipped when no bot is free. The return value still answers
    /// "is the phone leg up", because that is what the caller prints.
    pub async fn start(self: &Arc<Self>) -> bool {
        use crate::remote::adapters::telegram as tg;
        // Settle the name before claiming — the claim uses it twice (to pick
        // which bot to prefer, and as the slot key).
        let label = inbox::distinct_label(&inbox::short_label(&self.cwd), &self.session_id());
        self.state.lock().label = label.clone();

        let phone = if crate::remote::permissions::is_available() {
            let claimed = claim_bot_for(&self.cwd, &label).await;
            if !claimed {
                report_no_free_bot(tg::bot_pool().len());
            }
            claimed
        } else {
            false
        };

        inbox::sweep(None, false); // reclaim files left by crashed runs
        {
            let mut s = self.state.lock();
            s.last_sweep = Some(std::time::Instant::now());
        }
        inbox::register(&self.session_id(), &self.cwd, &label);

        if phone {
            let poller = crate::remote::poller::get();
            // File messages by session; the poller may be serving other
            // sessions too, and this callback runs on its task.
            poller.set_on_message(Arc::new(|text: String, meta: Value| {
                tokio::spawn(async move {
                    route(&text, &meta).await;
                });
            }));
            poller.acquire();
        }

        {
            let mut s = self.state.lock();
            s.started = true;
            s.stop = CancellationToken::new();
            s.polling = phone;
        }
        let me = self.clone();
        let stop = self.state.lock().stop.clone();
        tokio::spawn(async move {
            me.drain_loop(stop).await;
        });
        *ACTIVE.lock() = Some(self.clone());
        phone
    }

    pub async fn stop(self: &Arc<Self>) {
        {
            let mut s = self.state.lock();
            if !s.started {
                return;
            }
            s.started = false;
            s.stop.cancel();
        }
        // Only if this session actually took one; with no bot it never did.
        let was_polling = {
            let mut st = self.state.lock();
            let p = st.polling;
            st.polling = false;
            p
        };
        if was_polling {
            crate::remote::poller::get().release();
        }

        // Anything that arrived in the last moments is still the user's — say
        // so on the way out rather than leaving it to rot until sweep.
        let leftover: Vec<String> = inbox::take(&self.session_id())
            .into_iter()
            .filter(|m| !m.text.is_empty())
            .map(|m| m.text)
            .collect();
        if !leftover.is_empty() {
            let mut lines: Vec<String> = leftover
                .iter()
                .take(5)
                .map(|t| format!("• {}", t.chars().take(120).collect::<String>()))
                .collect();
            if leftover.len() > 5 {
                lines.push("…".to_string());
            }
            tell(
                &trf(
                    "{count} message(s) not delivered",
                    &[("count", FmtArg::Int(leftover.len() as i64))],
                ),
                &format!("{}\n{}", tr("The session ended before these could be handled:"), lines.join("\n")),
            )
            .await;
        }
        // The inbox file itself is left for `sweep`.
        inbox::unregister_registry_only(&self.session_id());

        // Say the phone has gone deaf, and whether anything else is still
        // listening — sent in BOTH cases (see the Python comment about the
        // notice that sat inside the else and never fired).
        let me = self.session_id();
        let others: Vec<_> =
            inbox::live_sessions().into_iter().filter(|s| s.session_id != me).collect();
        let body = if !others.is_empty() {
            let names = others
                .iter()
                .map(|s| {
                    if s.label.is_empty() {
                        s.session_id.chars().take(8).collect()
                    } else {
                        s.label.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            trf("Other sessions are still running: {names}", &[("names", FmtArg::Str(names))])
        } else {
            tr("No sessions are running. Messages will not be delivered until you start one.")
        };
        let label = {
            let l = self.label();
            if l.is_empty() {
                inbox::short_label(&self.cwd)
            } else {
                l
            }
        };
        tell(&trf("Session ended: {name}", &[("name", FmtArg::Str(label))]), &body).await;
    }

    /// Take over messages that arrived while no session was registered.
    ///
    /// Gated OFF by default — the caller in `drain_loop` runs this only under
    /// `remote_replay_backlog`; see the note there for what it did when it was
    /// unconditional. Moved rather than copied: two sessions starting together
    /// must not both run the same instruction.
    fn adopt_unrouted(&self) {
        let held = inbox::take_unrouted();
        for message in held {
            if message.text.is_empty() {
                continue;
            }
            if inbox::deliver(&self.session_id(), &message) {
                continue;
            }
            // Could not file it: put it back rather than lose it, and stop.
            let _ = inbox::deliver(inbox::UNROUTED, &message);
            return;
        }
    }

    // --- draining ------------------------------------------------------------

    /// Follow the session id if it changed under us (/clear mints a new one,
    /// /resume adopts the transcript's). Drain the old inbox first; move the
    /// registration rather than adding one.
    async fn resync_session_id(&self) {
        let current = crate::utils::state::get_session_id();
        let previous = self.session_id();
        if current.is_empty() || current == previous {
            return;
        }
        for message in inbox::take(&previous) {
            if !message.text.is_empty() {
                if let Some(cb) = &self.on_message {
                    cb(message.text.clone(), age_seconds(&message), message.source.clone());
                }
            }
        }
        inbox::unregister(&previous);
        // Carry the settled label across.
        let label = self.label();
        inbox::register(&current, &self.cwd, &label);
        self.state.lock().session_id = current;
    }

    /// Reclaim aged leftovers periodically, not only at startup.
    async fn sweep_if_due(&self) {
        let due = {
            let s = self.state.lock();
            s.last_sweep
                .map(|t| t.elapsed().as_secs_f64() >= SWEEP_INTERVAL)
                .unwrap_or(true)
        };
        if !due {
            return;
        }
        self.state.lock().last_sweep = Some(std::time::Instant::now());
        // Housekeeping must never end a session; sweep itself never panics.
        let _ = tokio::task::spawn_blocking(|| inbox::sweep(None, false)).await;
    }

    async fn drain_loop(self: &Arc<Self>, stop: CancellationToken) {
        // The filesystem work runs off the async executor's hot path via
        // spawn_blocking where it may hold a lock through a retry window (the
        // Python pushed the same calls to a worker thread for the same reason:
        // the terminal must not go dead for an invisible reason).
        let mut reported = false;
        loop {
            if stop.is_cancelled() || !self.state.lock().started {
                break;
            }
            let tick: Result<(), String> = async {
                self.resync_session_id().await;
                self.sweep_if_due().await;
                // Opt-in only, and OFF by default (`remote_replay_backlog`).
                //
                // A session must not inherit messages nobody asked it to run.
                // This call sits in the drain loop, so it fired EVERY tick, not
                // just at startup: anything in UNROUTED was pulled into
                // whichever session happened to be live and submitted as a user
                // turn, with no prompt, no confirmation and no size limit.
                // Measured 2026-10-01: a freshly started `cm2` whose only typed
                // input was one short line (18,144-token first request, 18:03:40)
                // was followed at 18:05:21 by a 497,039-character turn, and the
                // next request carried 262,342 prompt tokens — 59% of the window
                // in one step, from a 4-minute cold prefill. From the terminal
                // that is indistinguishable from the session resuming itself,
                // and it recreates the oversized-repetitive-context condition
                // that collapses the decoder (see services::ab).
                //
                // Held messages now stay held and the sender is told they were
                // NOT delivered (poller.rs), which is information they can act
                // on. The old behaviour is one config flag away for anyone who
                // wants it.
                if crate::utils::config::get_config().remote_replay_backlog {
                    let me = self.clone();
                    let _ = tokio::task::spawn_blocking(move || me.adopt_unrouted()).await;
                }
                let sid = self.session_id();
                let messages =
                    tokio::task::spawn_blocking(move || inbox::take(&sid)).await.map_err(|e| e.to_string())?;
                for message in messages {
                    if message.text.is_empty() {
                        continue;
                    }
                    if let Some(cb) = &self.on_message {
                        // Acknowledge before running it — until the turn has
                        // something to say, the phone shows nothing, which
                        // looks exactly like the message never arriving.
                        ack(&message, "👀").await;
                        cb(message.text.clone(), age_seconds(&message), message.source.clone());
                    }
                }
                Ok(())
            }
            .await;
            match tick {
                Ok(()) => {
                    reported = false; // drained cleanly; a later fault is news again
                }
                Err(e) => {
                    // Reported on the FIRST one, once per outbreak — a fault
                    // repeating every second must not notify every second.
                    if !reported {
                        reported = true;
                        crate::utils::log::debug_log(
                            "REMOTE_DRAIN_FAILED",
                            Some(&serde_json::json!({
                                "session": self.session_id(),
                                "error": e,
                            })),
                        );
                        // On screen too — the phone is the half that is broken.
                        crate::utils::formatting::safe_print(&format!(
                            "\n⚠️  {}",
                            trf(
                                "This session could not read its inbox: {error}",
                                &[("error", FmtArg::Str(e.clone()))]
                            )
                        ));
                        tell(
                            &tr("Messages from the phone are not being delivered"),
                            &trf(
                                "This session could not read its inbox: {error}",
                                &[("error", FmtArg::Str(e))],
                            ),
                        )
                        .await;
                    }
                }
            }
            tokio::select! {
                _ = stop.cancelled() => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs_f64(DRAIN_INTERVAL)) => {}
            }
        }
    }
}

/// Mark a received message so it visibly landed. Best-effort, never raises.
async fn ack(message: &inbox::InboundMessage, emoji: &str) {
    let mid = message.meta.get("message_id").and_then(|v| v.as_i64());
    let Some(mid) = mid else { return };
    for adapter in crate::remote::notify::adapters() {
        if adapter.is_configured() {
            adapter.react(mid, emoji).await;
        }
    }
}

/// How long ago the user actually sent this, or 0 when unknown — surfaced
/// rather than the message dropped (deciding it is stale is the user's call).
pub fn age_seconds(message: &inbox::InboundMessage) -> f64 {
    let sent = message.meta.get("sent_at").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if sent <= 0.0 {
        return 0.0;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    (now - sent).max(0.0)
}

/// File a message, and say so on the phone if it could not be filed — the one
/// thing that must not happen is silence.
async fn deliver_or_tell(session_id: &str, message: &inbox::InboundMessage) -> bool {
    if inbox::deliver(session_id, message) {
        return true;
    }
    tell(
        &tr("Message not delivered"),
        &tr("The session inbox is locked by another process and did not accept it. Nothing was written, so nothing is half-there — send it again."),
    )
    .await;
    false
}

/// This session's id, when it owns the bot the message came in through.
/// Empty when nothing is claimed — absence is not ambiguity to guess at.
pub fn owning_session_id() -> String {
    if crate::remote::adapters::telegram::claimed_token().is_empty() {
        return String::new();
    }
    ACTIVE.lock().as_ref().map(|a| a.session_id()).unwrap_or_default()
}

/// Poller-task callback: decide which session a message belongs to.
/// claimed → this session; reply → that session; only one → it; several →
/// ask; none → park and say so. Never guesses.
async fn route(text: &str, meta: &Value) {
    let message = inbox::InboundMessage::new(text, "telegram", meta.clone());

    // Typing an answer is at least as natural as tapping one.
    if answer_by_text(text, meta).await {
        return;
    }

    let owner = owning_session_id();
    if !owner.is_empty() {
        deliver_or_tell(&owner, &message).await;
        return;
    }

    if let Some(reply_to) = meta.get("reply_to").and_then(|v| v.as_i64()) {
        if let Some(owner) = inbox::message_owner(reply_to) {
            let live: std::collections::HashSet<String> =
                inbox::live_sessions().into_iter().map(|s| s.session_id).collect();
            if live.contains(&owner) {
                deliver_or_tell(&owner, &message).await;
                return;
            }
        }
    }

    let sessions = inbox::live_sessions();
    if sessions.len() == 1 {
        deliver_or_tell(&sessions[0].session_id, &message).await;
        return;
    }

    if sessions.is_empty() {
        if deliver_or_tell(inbox::UNROUTED, &message).await {
            tell(
                &tr("No sessions are running"),
                &tr("Your message is held and will be delivered when one starts."),
            )
            .await;
        }
        return;
    }

    ask_which_session(&message, &sessions).await;
}

/// Words that mean yes or no and nothing else. INPUT, not display: what the
/// user may type back — accumulated across languages, never translated. A word
/// only counts when the whole message is that word.
const YES: &[&str] = &[
    "y", "yes", "ok", "okay", "yep", "sure", "승인", "허용", "ㅇ", "ㅇㅇ", "はい", "是", "好",
    "ja", "oui", "si", "sí", "sim", "да",
];
const NO: &[&str] = &[
    "n", "no", "nope", "nah", "거부", "취소", "ㄴ", "ㄴㄴ", "いいえ", "否", "不", "nein", "non",
    "não", "nao", "нет",
];

/// Resolve a prompt from a typed yes/no — but only as a REPLY to it (the
/// message you replied to IS the target; a bare "y" is never guessed at).
async fn answer_by_text(text: &str, meta: &Value) -> bool {
    let word = text.trim().to_lowercase();
    let is_yes = YES.contains(&word.as_str());
    let is_no = NO.contains(&word.as_str());
    if !is_yes && !is_no {
        return false;
    }
    let Some(reply_to) = meta.get("reply_to").and_then(|v| v.as_i64()) else {
        return false;
    };
    if !inbox::pending_questions().contains_key(&reply_to) {
        return false; // not a live prompt — ordinary input
    }
    // "p:0"/"p:1" are POSITIONAL, and this queue carries both two-option
    // permission prompts and AskUserQuestion, whose options are arbitrary
    // model-authored labels. A typed "no" on a three-way question therefore
    // selected whatever option 1 happened to be — "Drop and recreate the DB"
    // reads as a denial to the user and as a choice to the agent — and the
    // confirmation below then reported it as "denied". Only take the shorthand
    // where the positions are known to mean allow/deny; anything else falls
    // through to ordinary input, and the buttons are still there to tap.
    if inbox::question_option_count(reply_to) != Some(2) {
        return false;
    }
    inbox::deliver_answer(reply_to, if is_yes { "p:0" } else { "p:1" });
    tell(
        &tr("Answer delivered"),
        &format!("\"{text}\" -> {}", if is_yes { tr("approved") } else { tr("denied") }),
    )
    .await;
    true
}

/// A note back to the phone about routing itself. INTERNAL (the body may quote
/// what you typed); blocking because the callers include shutdown paths.
async fn tell(title: &str, body: &str) {
    crate::remote::notify::notify(
        crate::remote::events::EventKind::Info,
        title,
        crate::remote::notify::NotifyOpts {
            body: body.to_string(),
            sensitivity: crate::remote::events::Sensitivity::Internal,
            blocking: true,
            ..Default::default()
        },
    )
    .await;
}

/// Post a picker listing the live sessions and park the message on it.
async fn ask_which_session(message: &inbox::InboundMessage, sessions: &[inbox::SessionEntry]) {
    for adapter in crate::remote::notify::adapters() {
        if !adapter.is_configured() {
            continue;
        }
        let labels: Vec<String> = sessions
            .iter()
            .map(|s| {
                if s.label.is_empty() {
                    s.session_id.chars().take(8).collect()
                } else {
                    s.label.clone()
                }
            })
            .collect();
        let preview: String = message.text.chars().take(200).collect();
        if let Some(picker_id) =
            adapter.pick(&tr("Which session should this go to?"), &preview, &labels).await
        {
            let choices: Vec<String> = sessions.iter().map(|s| s.session_id.clone()).collect();
            inbox::park_route(picker_id, &message.text, &message.meta, &choices);
            return;
        }
    }
    // No transport could ask — hold it rather than send it somewhere arbitrary.
    deliver_or_tell(inbox::UNROUTED, message).await;
}

/// Attach or detach this session's inbound remote input at runtime (outbound
/// is gated per call; inbound is a live loop that must be stopped/restarted).
/// Returns a short status for the caller to print.
pub async fn set_enabled(enabled: bool) -> String {
    let cur = ACTIVE.lock().clone();
    let Some(cur) = cur else {
        return tr("no inbound link (this session does not use remote input)");
    };
    if enabled {
        if cur.state.lock().started {
            return tr("inbound already attached");
        }
        if cur.start().await {
            tr("inbound attached")
        } else {
            tr("inbound failed to attach")
        }
    } else {
        if !cur.state.lock().started {
            return tr("inbound already detached");
        }
        cur.stop().await;
        tr("inbound detached (poller stopped, poll lock released)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yes_no_words_exact_match_only() {
        assert!(YES.contains(&"ㅇㅇ"));
        assert!(NO.contains(&"нет"));
        // Whole-message rule lives in answer_by_text: a sentence starting with
        // "no" must not be swallowed — verified by the trim+lookup shape.
        let word = "no, and also refactor".trim().to_lowercase();
        assert!(!NO.contains(&word.as_str()));
    }

    #[test]
    fn age_from_sent_at() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let m = inbox::InboundMessage::new("t", "telegram", serde_json::json!({"sent_at": now - 120.0}));
        let age = age_seconds(&m);
        assert!((115.0..130.0).contains(&age), "{age}");
        let m2 = inbox::InboundMessage::new("t", "telegram", serde_json::json!({}));
        assert_eq!(age_seconds(&m2), 0.0);
    }
}
