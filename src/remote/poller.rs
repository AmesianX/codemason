//! The single inbound loop for a process.
//! Equivalent to codemason/remote/poller.py — see that file's module doc and
//! constants for the full reasoning (exclusive getUpdates, 409 = positive
//! evidence of a live rival and never a timeout, lock released during failure
//! backoff, backlog classified by update id from a probe, MIN_CYCLE spin
//! guard).
//!
//! Substrate divergence (documented): the Python loop ran on a daemon thread
//! with blocking urllib; here it is a tokio task with async reqwest, stopped
//! through a CancellationToken (reset per start, since a token cannot be
//! un-cancelled). Question.wait's 0.5s slices become async sleeps.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::utils::i18n::{tr, trf, FmtArg};

/// One long-poll chunk. Also the worst-case delay before the loop notices it
/// should stop.
pub const POLL_CHUNK: u64 = 20;

/// Transient-failure backoff (retry forever; the lock is released while
/// failing so a broken session cannot starve a working one).
pub const RETRY_BASE: f64 = 2.0;
pub const RETRY_MAX: f64 = 60.0;

/// Two consecutive 409s = a live rival on this bot (never a clock).
pub const MAX_CONSECUTIVE_CONFLICTS: u32 = 2;

/// Floor on an empty cycle — spin guard for a proxy that ignores the long
/// poll's own pacing (measured: ~1M calls in 20s without it).
pub const MIN_CYCLE: f64 = 1.0;

/// How often a session that lost the lock checks whether it has been freed.
pub const LOCK_RETRY: f64 = 15.0;

/// A prompt awaiting a tap, keyed by the message its buttons live on.
pub struct Question {
    pub message_id: i64,
    pub options: Vec<String>,
    /// Takes a typed answer ("t:<text>") as well as a tap — see questions.rs.
    pub free_text: bool,
    answer: Mutex<Option<String>>,
    done: Arc<tokio::sync::Notify>,
    resolved: std::sync::atomic::AtomicBool,
}

impl Question {
    fn new(message_id: i64, options: Vec<String>, free_text: bool) -> Arc<Question> {
        Arc::new(Question {
            message_id,
            options,
            free_text,
            answer: Mutex::new(None),
            done: Arc::new(tokio::sync::Notify::new()),
            resolved: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn resolve(&self, answer: &str) {
        *self.answer.lock() = Some(answer.to_string());
        self.resolved.store(true, std::sync::atomic::Ordering::SeqCst);
        self.done.notify_waiters();
    }

    /// Block until answered, cancelled, or out of time. Each slice also checks
    /// for a forwarded answer (the tap may land at whichever process holds the
    /// poll lock — see inbox::deliver_answer).
    pub async fn wait(&self, timeout: f64, cancel: Option<CancellationToken>) -> Option<String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
        while std::time::Instant::now() < deadline {
            if let Some(c) = &cancel {
                if c.is_cancelled() {
                    return None;
                }
            }
            if self.resolved.load(std::sync::atomic::Ordering::SeqCst) {
                return self.answer.lock().clone();
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let slice = remaining.min(std::time::Duration::from_millis(500));
            tokio::select! {
                _ = self.done.notified() => {
                    if self.resolved.load(std::sync::atomic::Ordering::SeqCst) {
                        return self.answer.lock().clone();
                    }
                }
                _ = tokio::time::sleep(slice) => {}
            }
            if let Some(data) = crate::remote::inbox::take_answer(self.message_id) {
                if let Some(idx) = data.strip_prefix("p:") {
                    if let Ok(i) = idx.parse::<usize>() {
                        if let Some(opt) = self.options.get(i) {
                            return Some(opt.clone());
                        }
                    }
                }
                // A typed answer, filed by whoever read the message. Only a
                // prompt that offers "Other" takes one; anything else ignores
                // it rather than read a sentence as a choice.
                if let Some(t) = data.strip_prefix("t:") {
                    let t = t.trim();
                    if self.free_text && !t.is_empty() {
                        return Some(t.to_string());
                    }
                }
            }
        }
        None
    }
}

/// How many times a panicking poll loop is respawned before it stays down.
const MAX_POLL_RESTARTS: u32 = 5;

/// on_message callback: (text, meta) filed by whoever holds the poll lock.
pub type OnMessage = Arc<dyn Fn(String, Value) + Send + Sync>;

struct PollerState {
    users: usize,
    running: bool,
    stop: CancellationToken,
    questions: HashMap<i64, Arc<Question>>,
    conflicts: u32,
    /// Times the loop was respawned after a panic (see `start`).
    restarts: u32,
    /// An acquire arrived while the loop was shutting down (see `start`).
    relaunch: bool,
    backlog_ids: std::collections::HashSet<i64>,
    probe_ok: bool,
    first_fetch: bool,
    lock_file: Option<std::fs::File>,
    on_message: Option<OnMessage>,
    /// Messages carrying photos are handled on a worker task (handoff): a
    /// download can take a minute on a bad network, and the loop has to keep
    /// answering button taps meanwhile. Anything that arrives while the worker
    /// is busy queues behind it, so messages still reach the session in the
    /// order they were sent ("photo", then "what is this?").
    work_tx: Option<tokio::sync::mpsc::UnboundedSender<Work>>,
    /// Queued or running on the worker.
    work_pending: usize,
    /// Bumped per loop run (and bot change); stale work is dropped.
    gen: u64,
    offsets: OffsetBook,
}

/// One message (or album) for the worker.
struct Work {
    gen: u64,
    ids: Vec<i64>,
    msg: Value,
    extra: Vec<Value>,
}

/// Which updates are handled, and how far the offset may move.
///
/// The offset is what Telegram replays from, and it deletes everything below
/// it. So it may pass an update only once every update before it is handled:
/// while a photo message is on the worker, later updates are handled but the
/// offset waits for it, and when getUpdates hands them back they are skipped
/// (is_new).
#[derive(Default)]
struct OffsetBook {
    /// Handed to the worker, not done yet.
    inflight: std::collections::BTreeSet<i64>,
    /// Dispatched since the offset last moved past them.
    seen: std::collections::HashSet<i64>,
    done_max: Option<i64>,
    /// How far the offset has been allowed to move. Everything below this was
    /// handled — that is the whole rule the offset follows — so `seen` only has
    /// to remember the ids at or above it.
    ///
    /// Without this floor a photo was delivered twice, every time. `seen` is
    /// pruned the moment commit() moves the offset, but the fetch that is
    /// ALREADY in flight went out with the old offset, and Telegram answers it
    /// with the update again: back in the loop, `seen` no longer holds the id,
    /// so it looks new and the photo lands in the inbox a second time, with a
    /// second turn to answer it. Only the photo path has the window open —
    /// there the commit happens on the worker while the loop sits in a long
    /// poll, where the other paths commit between two fetches.
    passed: i64,
}

impl OffsetBook {
    fn is_new(&mut self, id: i64) -> bool {
        if id < self.passed {
            return false; // the offset is already past it: handled
        }
        self.seen.insert(id)
    }

    /// Mark `id` handled; the offset it allows.
    fn commit(&mut self, id: i64) -> i64 {
        self.inflight.remove(&id);
        let done = self.done_max.map_or(id, |d| d.max(id));
        self.done_max = Some(done);
        let target = self.inflight.first().copied().unwrap_or(done + 1);
        self.passed = self.passed.max(target);
        self.seen.retain(|u| *u >= target);
        target
    }
}

/// Owns the inbound connection for this process.
pub struct Poller {
    adapter: Arc<crate::remote::adapters::telegram::TelegramAdapter>,
    state: Mutex<PollerState>,
}

static INSTANCE: Lazy<Arc<Poller>> = Lazy::new(|| {
    Arc::new(Poller {
        adapter: Arc::new(crate::remote::adapters::telegram::TelegramAdapter::new()),
        state: Mutex::new(PollerState {
            users: 0,
            running: false,
            stop: CancellationToken::new(),
            questions: HashMap::new(),
            conflicts: 0,
            restarts: 0,
            relaunch: false,
            backlog_ids: std::collections::HashSet::new(),
            probe_ok: false,
            first_fetch: true,
            lock_file: None,
            on_message: None,
            work_tx: None,
            work_pending: 0,
            gen: 0,
            offsets: OffsetBook::default(),
        }),
    })
});

/// The process-wide poller, created on first use.
pub fn get() -> Arc<Poller> {
    INSTANCE.clone()
}

/// Stop the process-wide poller (the atexit hook's job in Python); called by
/// the CLI shutdown path.
pub fn shutdown() {
    let p = get();
    let mut s = p.state.lock();
    s.users = 0;
    s.stop.cancel();
}

impl Poller {
    pub fn set_on_message(&self, cb: OnMessage) {
        self.state.lock().on_message = Some(cb);
    }

    /// Register a consumer and start the loop if it is not already running.
    pub fn acquire(self: &Arc<Self>) {
        let mut s = self.state.lock();
        s.users += 1;
        self.start(s);
    }

    /// Spawn the loop if it is not already up. Takes the held guard so the
    /// check and the flag cannot race with a second acquire.
    fn start(self: &Arc<Self>, mut s: parking_lot::MutexGuard<'_, PollerState>) {
        if s.running {
            // RACE. release() cancels the token but the loop only notices at
            // its next await, so a release immediately followed by an acquire
            // (every /resume, every sub-agent that takes and gives back the
            // remote channel) lands here with running still true and a token
            // that is already cancelled. Returning early left a consumer
            // attached to a loop that was on its way out, and remote input
            // died silently until the next restart. Hand the relaunch to the
            // task that is leaving: it is the only one that knows when the old
            // loop is actually gone.
            if s.stop.is_cancelled() {
                s.relaunch = true;
            }
            return;
        }
        s.running = true;
        s.stop = CancellationToken::new();
        let me = self.clone();
        let stop = s.stop.clone();
        drop(s);
        tokio::spawn(async move {
            // The task is DETACHED: nothing joins it, so a panic inside run()
            // used to unwind straight past the `running = false` that came
            // after it. The flag stayed true for the life of the process,
            // every later acquire() saw "already running" and started
            // nothing, and remote input was dead with no note anywhere —
            // the session looks alive and simply stops answering Telegram.
            // The panic hook does put the crash in error.log (main.rs
            // install_panic_hook), so this is not about the record; it is
            // about the flag and the restart. The Python port survives the
            // same crash because Poller.acquire() restarts a dead thread
            // (cli.py:4223 + Poller.acquire), which is what this brings over.
            let crashed = futures::FutureExt::catch_unwind(
                std::panic::AssertUnwindSafe(me.run(stop)),
            )
            .await
            .is_err();
            let mut s = me.state.lock();
            s.running = false;
            if !crashed {
                // Asked to come back while it was shutting down (see start).
                if std::mem::take(&mut s.relaunch) && s.users > 0 {
                    me.start(s);
                }
                return;
            }
            s.relaunch = false;
            // Bounded, because a panic that reproduces on the first poll would
            // otherwise respawn forever and fill the log with it.
            if s.users == 0 || s.stop.is_cancelled() {
                return;
            }
            if s.restarts >= MAX_POLL_RESTARTS {
                drop(s);
                crate::utils::log::log_error(
                    &format!(
                        "remote poll loop panicked {MAX_POLL_RESTARTS} times — giving up; \
                         remote input is off for the rest of this session"
                    ),
                    Some("remote_poller: restart"),
                );
                return;
            }
            s.restarts += 1;
            let attempt = s.restarts;
            drop(s);
            crate::utils::log::log_error(
                &format!("remote poll loop panicked — restarting (attempt {attempt})"),
                Some("remote_poller: restart"),
            );
            me.start(me.state.lock());
        });
    }

    /// Drop a consumer; the loop stops once the last one is gone.
    pub fn release(&self) {
        let mut s = self.state.lock();
        s.users = s.users.saturating_sub(1);
        if s.users == 0 {
            s.stop.cancel();
        }
    }

    pub fn stop(&self) {
        let mut s = self.state.lock();
        s.users = 0;
        s.stop.cancel();
    }

    // --- questions -----------------------------------------------------------

    pub fn add_question(&self, message_id: i64, options: Vec<String>, free_text: bool) -> Arc<Question> {
        let q = Question::new(message_id, options, free_text);
        self.state.lock().questions.insert(message_id, q.clone());
        q
    }

    pub fn drop_question(&self, message_id: i64) {
        self.state.lock().questions.remove(&message_id);
    }

    // --- the loop ------------------------------------------------------------

    async fn run(self: &Arc<Self>, stop: CancellationToken) {
        let fd = self.await_lock(&stop).await;
        let Some(fd) = fd else {
            return; // asked to stop while waiting
        };
        self.state.lock().lock_file = Some(fd);
        self.reset_dispatch_state();
        crate::remote::adapters::telegram::prune_attachments(
            crate::remote::adapters::telegram::ATTACHMENT_MAX_AGE_DAYS,
        );
        // Ask what was already waiting before serving anything.
        self.probe_backlog().await;
        let mut failures: u32 = 0;
        self.state.lock().first_fetch = true;

        loop {
            if stop.is_cancelled() {
                break;
            }
            let cycle_started = std::time::Instant::now();
            let updates = self.fetch().await;
            match updates {
                None => {
                    let conflicts = self.state.lock().conflicts;
                    if conflicts >= MAX_CONSECUTIVE_CONFLICTS {
                        if !self.reclaim_bot(&stop).await {
                            break;
                        }
                        continue;
                    }
                    failures += 1;
                    // Hold the lock only while it is doing something.
                    if failures >= 2 {
                        // Not with a photo still on the worker: whoever polls
                        // next would be handed it again.
                        if self.state.lock().lock_file.is_some() {
                            self.drain_worker(&stop).await;
                        }
                        let fd = self.state.lock().lock_file.take();
                        if fd.is_some() {
                            self.adapter.release_poll_lock(fd);
                            crate::utils::log::debug_log(
                                "REMOTE_POLLER_YIELDED_LOCK",
                                Some(&serde_json::json!({"failures": failures})),
                            );
                        }
                    }
                    let delay = RETRY_MAX.min(RETRY_BASE * 2f64.powi((failures - 1).min(6) as i32));
                    tokio::select! {
                        _ = stop.cancelled() => break,
                        _ = tokio::time::sleep(std::time::Duration::from_secs_f64(delay)) => {}
                    }
                    // Take it back BEFORE fetching again — polling without the
                    // lock is never allowed.
                    if self.state.lock().lock_file.is_none() {
                        match self.await_lock(&stop).await {
                            Some(f) => self.state.lock().lock_file = Some(f),
                            None => break, // asked to stop while waiting
                        }
                    }
                    continue;
                }
                Some(updates) => {
                    if failures > 0 {
                        crate::utils::log::debug_log(
                            "REMOTE_POLLER_RECOVERED",
                            Some(&serde_json::json!({"after_failures": failures})),
                        );
                        failures = 0;
                    }
                    // Fail closed when the probe could not answer: anything in
                    // the FIRST fetch was queued before we listened.
                    {
                        let mut s = self.state.lock();
                        if s.first_fetch {
                            s.first_fetch = false;
                            if !s.probe_ok {
                                for upd in &updates {
                                    if let Some(id) = upd.get("update_id").and_then(|v| v.as_i64()) {
                                        s.backlog_ids.insert(id);
                                    }
                                }
                                if !s.backlog_ids.is_empty() {
                                    let n = s.backlog_ids.len();
                                    drop(s);
                                    crate::utils::log::debug_log(
                                        "REMOTE_BACKLOG_FROM_FIRST_FETCH",
                                        Some(&serde_json::json!({"count": n})),
                                    );
                                }
                            }
                        }
                    }
                    // While the offset waits for a photo on the worker,
                    // getUpdates hands back what was already dispatched; only
                    // the rest is new.
                    let fresh: Vec<Value> = updates.into_iter().filter(|u| self.is_new(u)).collect();
                    for group in group_albums(&fresh) {
                        // one bad update must not kill the loop
                        if group.len() == 1 {
                            self.dispatch(group[0]).await;
                        } else {
                            self.dispatch_album(&group).await;
                        }
                    }
                    // Only paced when nothing arrived — including the cycle that
                    // only returns updates already dispatched.
                    if fresh.is_empty() {
                        let elapsed = cycle_started.elapsed().as_secs_f64();
                        let slack = MIN_CYCLE - elapsed;
                        if slack > 0.0 {
                            tokio::select! {
                                _ = stop.cancelled() => break,
                                _ = tokio::time::sleep(std::time::Duration::from_secs_f64(slack)) => {}
                            }
                        }
                    }
                }
            }
        }
        // May already be None: a failing loop yields the lock.
        let fd = self.state.lock().lock_file.take();
        if fd.is_some() {
            self.adapter.release_poll_lock(fd);
        }
    }

    /// Another instance took our bot. Move to a free one; true if we did.
    async fn reclaim_bot(self: &Arc<Self>, stop: &CancellationToken) -> bool {
        use crate::remote::adapters::telegram as tg;
        self.drain_worker(stop).await;
        // The offset is per bot: nothing recorded against the old one applies.
        self.reset_dispatch_state();
        let fd = self.state.lock().lock_file.take();
        if fd.is_some() {
            self.adapter.release_poll_lock(fd);
        }
        tg::release_bot();
        {
            let conflicts = self.state.lock().conflicts;
            crate::utils::log::debug_log(
                "REMOTE_BOT_LOST",
                Some(&serde_json::json!({"conflicts": conflicts})),
            );
            // log_error too, not only debug_log: debug_log is a no-op unless
            // CODEMASON_DEBUG is set, and this is the moment remote input stops
            // working. Without a record, the morning after looks like "Telegram
            // just stopped" with nothing on disk to say the bot was handed over.
            crate::utils::log::log_error(
                &format!(
                    "gave up the Telegram bot after {conflicts} conflicts (409) \
                     — another poller is using the same token"
                ),
                Some("remote_poller: bot lost"),
            );
        }

        if tg::claim_bot("").await.is_none() {
            self.report_no_free_bot().await;
            return false;
        }
        {
            let mut s = self.state.lock();
            s.conflicts = 0;
            s.backlog_ids.clear(); // a different bot has a different queue
            s.probe_ok = false;
            s.first_fetch = true;
        }
        match self.await_lock(stop).await {
            Some(f) => self.state.lock().lock_file = Some(f),
            None => return false, // asked to stop while waiting
        }
        self.probe_backlog().await;
        true
    }

    /// Say that the pool is exhausted, and stand down. Printed, not
    /// debug-logged — something the user resolves in a minute with BotFather
    /// has to reach the user.
    async fn report_no_free_bot(&self) {
        use crate::remote::adapters::telegram as tg;
        let size = tg::bot_pool().len();
        crate::utils::log::log_error(
            &format!("no free bot in a pool of {size} — remote input is off"),
            Some("remote_poller: pool exhausted"),
        );
        crate::utils::log::debug_log(
            "REMOTE_POOL_EXHAUSTED",
            Some(&serde_json::json!({
                "pool": size,
                "verdicts": tg::last_claim_verdicts()
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
            })),
        );
        crate::cmprintln!();
        crate::remote::input::report_no_free_bot(size);
        crate::cmprintln!("{}", tr("  Local input, tools and output are unaffected."));
    }

    /// Take the poll lock, retrying until we get it or are told to stop.
    /// Waiting is a fault, not a queue, and is reported once — on screen and on
    /// the phone (see the Python docstring).
    async fn await_lock(&self, stop: &CancellationToken) -> Option<std::fs::File> {
        let mut waited = false;
        while !stop.is_cancelled() {
            if let Some(fd) = self.adapter.acquire_poll_lock() {
                if waited {
                    crate::utils::log::debug_log("REMOTE_POLLER_TOOK_OVER", Some(&serde_json::json!({})));
                }
                return Some(fd);
            }
            if !waited {
                waited = true;
                self.report_standby().await;
            }
            tokio::select! {
                _ = stop.cancelled() => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs_f64(LOCK_RETRY)) => {}
            }
        }
        None
    }

    /// Say that inbound is not starting, and why, exactly once.
    async fn report_standby(&self) {
        use crate::remote::adapters::telegram as tg;
        let holder = tg::poll_lock_holder();
        let path = tg::lock_path().to_string_lossy().into_owned();
        crate::utils::log::debug_log(
            "REMOTE_POLLER_STANDBY",
            Some(&serde_json::json!({"holder": holder, "lock": path})),
        );
        let detail = trf(
            "Waiting for the poll lock held by process {pid} ({path}). Messages sent from the phone will not arrive until this clears.",
            &[
                ("pid", FmtArg::Str(if holder.is_empty() { "?".into() } else { holder.clone() })),
                ("path", FmtArg::Str(if path.is_empty() { "?".into() } else { path.clone() })),
            ],
        );
        crate::utils::formatting::safe_print(&format!("\n⚠️  {detail}"));
        crate::remote::notify::notify_simple(
            crate::remote::events::EventKind::Info,
            &tr("Inbound messages are not being received"),
            &detail,
            crate::remote::events::Sensitivity::Internal,
        )
        .await;
    }

    /// Record which updates were already queued before we started listening —
    /// by update id, from Telegram itself (timeout=0 peek; offset untouched).
    async fn probe_backlog(&self) {
        let offset = self.adapter.read_offset();
        let resp = self
            .adapter
            .post(
                "getUpdates",
                &[
                    ("offset", offset.to_string()),
                    ("timeout", "0".to_string()),
                    ("allowed_updates", "[\"callback_query\", \"message\"]".to_string()),
                ],
                Some(15.0),
            )
            .await;
        let Some(resp) = resp else { return };
        if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            return; // probe_ok stays false → first fetch decides
        }
        let mut s = self.state.lock();
        s.probe_ok = true;
        if let Some(arr) = resp.get("result").and_then(|r| r.as_array()) {
            for upd in arr {
                if let Some(id) = upd.get("update_id").and_then(|v| v.as_i64()) {
                    s.backlog_ids.insert(id);
                }
            }
        }
        if !s.backlog_ids.is_empty() {
            let n = s.backlog_ids.len();
            drop(s);
            crate::utils::log::debug_log(
                "REMOTE_BACKLOG_PROBED",
                Some(&serde_json::json!({"count": n})),
            );
        }
    }

    async fn fetch(&self) -> Option<Vec<Value>> {
        let offset = self.adapter.read_offset();
        let resp = self
            .adapter
            .post(
                "getUpdates",
                &[
                    ("offset", offset.to_string()),
                    ("timeout", POLL_CHUNK.to_string()),
                    ("allowed_updates", "[\"callback_query\", \"message\"]".to_string()),
                ],
                Some((POLL_CHUNK + 10) as f64), // must outlast the long poll
            )
            .await;
        let ok = resp
            .as_ref()
            .map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        if !ok {
            let error = resp
                .as_ref()
                .and_then(|r| r.get("error"))
                .and_then(|e| e.as_str())
                .unwrap_or("no response")
                .to_string();
            // A 409 is diagnosed and stopped, not retried like a blip.
            if error.contains("HTTP 409") {
                self.state.lock().conflicts += 1;
            }
            crate::utils::log::debug_log(
                "REMOTE_POLL_FAILED",
                Some(&serde_json::json!({"error": error})),
            );
            return None;
        }
        self.state.lock().conflicts = 0;
        Some(
            resp.and_then(|r| r.get("result").and_then(|x| x.as_array()).cloned())
                .unwrap_or_default(),
        )
    }

    async fn dispatch(self: &Arc<Self>, upd: &Value) {
        // A missing/!i64 update_id used to fall back to 0, which wrote offset 1
        // and made the NEXT fetch replay everything Telegram still held (~24h)
        // as though it had just been typed — first_fetch is already false by
        // then, so none of it lands in backlog_ids either. Skip the update
        // instead: never move the offset somewhere we did not read from.
        let Some(update_id) = upd.get("update_id").and_then(|v| v.as_i64()) else {
            crate::utils::log::log_error(
                &"telegram update without a usable update_id — skipped",
                Some("remote_poller: dispatch"),
            );
            return;
        };
        // ORDER. The offset is what Telegram replays from, so writing it
        // first makes every update at-most-once: a panic or a kill between
        // that write and the inbox write drops the message for good, and the
        // user sees their instruction simply never arrive. Writing it AFTER
        // the handler makes the input path at-least-once instead, which is the
        // right trade here — the handler's job is to put the text in the inbox
        // file, so the only way to get a duplicate is a crash in the window
        // between that file write and this one, while the failure it removes
        // is silent loss of a command nobody knows was sent.
        //
        // The two INTERACTIVE paths keep the early write: both can sit on a
        // Telegram picker for as long as the user takes to tap, and an offset
        // that has not moved by then means getUpdates hands the same update
        // back on the next poll and opens a second picker for it.
        if let Some(cq) = upd.get("callback_query") {
            self.commit(update_id);
            self.on_callback(cq).await;
            return;
        }

        let Some(msg) = upd.get("message") else {
            self.commit(update_id);
            return;
        };
        // Anything that appears in the chat pushes an in-flight progress
        // message off the bottom — including what you typed yourself.
        if self.adapter.is_trusted_chat(msg, None) {
            crate::remote::adapters::telegram::note_chat_message(
                msg.get("message_id").and_then(|v| v.as_i64()),
            );
        }
        if self.needs_worker(&[msg]) {
            self.handoff(vec![update_id], msg.clone(), Vec::new());
            return;
        }
        self.on_message(msg, Some(update_id)).await;
        self.commit(update_id);
    }

    /// Several photos sent together arrive as one update each, sharing a
    /// media_group_id; usually only the first carries a caption. Delivered one
    /// by one they became one turn per photo, the later ones steered into a
    /// turn already answering the first. Handled as one message instead, offset
    /// moved past the whole group afterwards (same at-least-once order as
    /// dispatch).
    async fn dispatch_album(self: &Arc<Self>, group: &[&Value]) {
        let Some(first) = group[0].get("message") else { return };
        if self.adapter.is_trusted_chat(first, None) {
            for upd in group {
                crate::remote::adapters::telegram::note_chat_message(
                    upd.get("message").and_then(|m| m.get("message_id")).and_then(|v| v.as_i64()),
                );
            }
        }
        // Same rule as dispatch: never guess an offset.
        let Some(ids) = group
            .iter()
            .map(|u| u.get("update_id").and_then(|v| v.as_i64()))
            .collect::<Option<Vec<i64>>>()
        else {
            return;
        };
        let extra: Vec<&Value> = group[1..].iter().filter_map(|u| u.get("message")).collect();
        let mut all = vec![first];
        all.extend(extra.iter().copied());
        if self.needs_worker(&all) {
            self.handoff(ids, first.clone(), extra.into_iter().cloned().collect());
            return;
        }
        self.on_message_with(first, Some(ids[0]), &extra, None).await;
        for id in ids {
            self.commit(id);
        }
    }

    // --- offset and the photo worker -----------------------------------------

    fn reset_dispatch_state(&self) {
        let mut s = self.state.lock();
        s.gen += 1;
        s.offsets = OffsetBook::default();
    }

    /// False for an update already dispatched (and not yet behind the offset).
    fn is_new(&self, upd: &Value) -> bool {
        match upd.get("update_id").and_then(|v| v.as_i64()) {
            Some(id) => self.state.lock().offsets.is_new(id),
            None => true, // dispatch reports and skips it
        }
    }

    /// Record `id` as handled and move the offset as far as that allows
    /// (OffsetBook).
    fn commit(&self, id: i64) {
        let target = self.state.lock().offsets.commit(id);
        if target > self.adapter.read_offset() {
            self.adapter.write_offset(target);
        }
    }

    fn needs_worker(&self, msgs: &[&Value]) -> bool {
        msgs.iter().any(|m| self.adapter.is_trusted_chat(m, None) && !image_refs(m).is_empty())
            || self.state.lock().work_pending > 0 // keep the order behind a photo
    }

    /// Run the message on the worker, then commit `ids`. Work from an earlier
    /// loop run (stopped, bot changed) is not delivered — the next run is
    /// handed the same update again, since its offset was never committed.
    fn handoff(self: &Arc<Self>, ids: Vec<i64>, msg: Value, extra: Vec<Value>) {
        let mut s = self.state.lock();
        s.offsets.inflight.extend(ids.iter().copied());
        s.work_pending += 1;
        let gen = s.gen;
        if s.work_tx.as_ref().is_none_or(|t| t.is_closed()) {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            s.work_tx = Some(tx);
            let me = self.clone();
            tokio::spawn(async move { me.work_loop(rx).await });
        }
        let sent = s.work_tx.as_ref().map(|t| t.send(Work { gen, ids, msg, extra }).is_ok());
        if sent != Some(true) {
            s.work_pending -= 1;
        }
    }

    fn live(&self, gen: u64) -> bool {
        let s = self.state.lock();
        s.gen == gen && !s.stop.is_cancelled()
    }

    async fn work_loop(self: Arc<Self>, mut rx: tokio::sync::mpsc::UnboundedReceiver<Work>) {
        while let Some(w) = rx.recv().await {
            if self.live(w.gen) {
                let extra: Vec<&Value> = w.extra.iter().collect();
                // Same as the loop: one bad update is skipped, not fatal.
                let _ = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(
                    self.on_message_with(&w.msg, w.ids.first().copied(), &extra, Some(w.gen)),
                ))
                .await;
                if self.live(w.gen) {
                    for id in &w.ids {
                        self.commit(*id);
                    }
                }
            }
            let mut s = self.state.lock();
            s.work_pending = s.work_pending.saturating_sub(1);
        }
    }

    /// Wait (stop-aware) until nothing is left on the worker.
    async fn drain_worker(&self, stop: &CancellationToken) {
        while self.state.lock().work_pending > 0 {
            tokio::select! {
                _ = stop.cancelled() => return,
                _ = tokio::time::sleep(std::time::Duration::from_millis(200)) => {}
            }
        }
    }

    async fn on_callback(&self, cq: &Value) {
        let empty = Value::Object(Default::default());
        let msg = cq.get("message").unwrap_or(&empty);
        // The sender of a callback query is on the QUERY, not on the message it
        // is attached to — that message was posted by the bot, so authenticating
        // against it is always true.
        if !self.adapter.is_trusted_chat(msg, cq.get("from")) {
            return;
        }
        let mid = msg.get("message_id").and_then(|v| v.as_i64()).unwrap_or(0);
        let data = cq.get("data").and_then(|d| d.as_str()).unwrap_or("").to_string();
        let cq_id = cq.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
        if self.on_picker(&cq_id, mid, &data).await {
            return;
        }
        if data == OTHER_DATA {
            self.on_other(cq_id, mid).await;
            return;
        }
        if !data.starts_with("p:") {
            return;
        }
        let q = self.state.lock().questions.get(&mid).cloned();

        if let Some(q) = q {
            let answer = data[2..]
                .parse::<usize>()
                .ok()
                .and_then(|i| q.options.get(i).cloned());
            let Some(answer) = answer else { return };
            let _ = self
                .adapter
                .post(
                    "answerCallbackQuery",
                    &[("callback_query_id", cq_id), ("text", answer.clone())],
                    None,
                )
                .await;
            q.resolve(&answer);
            return;
        }

        // Not ours — probably another session's prompt (answers arrive at
        // whoever holds the poll lock). Forward the raw choice index.
        if let Some(_owner) = crate::remote::inbox::question_owner(mid) {
            let _ = self
                .adapter
                .post(
                    "answerCallbackQuery",
                    &[("callback_query_id", cq_id), ("text", tr("Forwarded"))],
                    None,
                )
                .await;
            crate::remote::inbox::deliver_answer(mid, &data);
            return;
        }

        // Genuinely unknown — an old button from a session long gone.
        let _ = self
            .adapter
            .post(
                "answerCallbackQuery",
                &[("callback_query_id", cq_id), ("text", tr("This request has expired"))],
                None,
            )
            .await;
    }

    /// The "Other" button: say how to answer, and leave the prompt open. A
    /// button cannot carry a sentence, so the tap answers nothing — the typed
    /// message that follows does (questions::answer_by_text). The alert is the
    /// whole job: without it the button looks broken.
    async fn on_other(&self, cq_id: String, mid: i64) {
        let live = self.state.lock().questions.contains_key(&mid);
        let (text, alert) = if live || crate::remote::inbox::question_owner(mid).is_some() {
            (tr("Type your answer and send it as a message"), "true")
        } else {
            (tr("This request has expired"), "false")
        };
        let _ = self
            .adapter
            .post(
                "answerCallbackQuery",
                &[("callback_query_id", cq_id), ("text", text), ("show_alert", alert.to_string())],
                None,
            )
            .await;
    }

    /// Answer a bot-owned command instead of feeding it to a session (/start,
    /// /help — the first thing you send from a phone).
    async fn on_bot_command(&self, _text: &str) {
        let sessions = crate::remote::inbox::live_sessions();
        let body = if !sessions.is_empty() {
            let lines: Vec<String> = sessions
                .iter()
                .map(|s| {
                    let label = if s.label.is_empty() {
                        s.session_id.chars().take(8).collect()
                    } else {
                        s.label.clone()
                    };
                    format!("• <code>{label}</code>")
                })
                .collect();
            let tail = if sessions.len() == 1 {
                tr("Just send a message and it goes there.")
            } else {
                tr("<b>Reply</b> to a session alert to route there.\nSend without replying and you will be asked which one.")
            };
            format!("{}\n{}\n\n{tail}", tr("Running sessions:"), lines.join("\n"))
        // Only promise delivery when something will actually deliver it:
        // adopting held messages is opt-in (see input.rs), so the old promise is
        // only true when that flag is on. Both strings are already in every
        // locale.
        } else if crate::utils::config::get_config().remote_replay_backlog {
            tr("No sessions are running.\nMessages are held and delivered once one starts.")
        } else {
            tr("Messages sent before a session started are not delivered")
        };
        let (token, chat) = crate::remote::adapters::telegram::credentials();
        if token.is_empty() || chat.is_empty() {
            return;
        }
        let resp = self
            .adapter
            .post(
                "sendMessage",
                &[
                    ("chat_id", chat),
                    ("text", format!("🤖 <b>codemason</b>\n\n{body}")),
                    ("parse_mode", "HTML".to_string()),
                    ("disable_notification", "true".to_string()),
                ],
                None,
            )
            .await;
        crate::remote::adapters::telegram::note_chat_message(
            resp.as_ref()
                .and_then(|r| r.get("result"))
                .and_then(|r| r.get("message_id"))
                .and_then(|m| m.as_i64()),
        );
    }

    /// Handle a tap on a "which session?" picker. True if it was one.
    async fn on_picker(&self, cq_id: &str, mid: i64, data: &str) -> bool {
        if !data.starts_with("r:") {
            return false;
        }
        let parked = crate::remote::inbox::take_route(mid);
        let _ = self
            .adapter
            .post(
                "answerCallbackQuery",
                &[("callback_query_id", cq_id.to_string())],
                None,
            )
            .await;
        let Some(parked) = parked else {
            return true; // already routed, or long expired
        };
        let target = data[2..]
            .parse::<usize>()
            .ok()
            .and_then(|i| parked.get("choices").and_then(|c| c.as_array()).and_then(|a| a.get(i).cloned()))
            .and_then(|v| v.as_str().map(|s| s.to_string()));
        let Some(target) = target else {
            return true;
        };
        if target.is_empty() {
            // The discard button on a held-over message.
            self.adapter.note_routed(mid, "").await;
            return true;
        }
        let text = parked.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let meta = parked.get("meta").cloned().unwrap_or_else(|| Value::Object(Default::default()));
        let message =
            crate::remote::inbox::InboundMessage::new(text.clone(), "telegram", meta.clone());
        // take_route already consumed the parked copy, so a failed delivery
        // loses the message outright — put it back and say so.
        if !crate::remote::inbox::deliver(&target, &message) {
            let choices: Vec<String> = parked
                .get("choices")
                .and_then(|c| c.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            crate::remote::inbox::park_route(mid, &text, &meta, &choices);
            crate::remote::notify::notify_simple(
                crate::remote::events::EventKind::Error,
                &tr("Message not delivered"),
                &tr("That session's inbox is locked by another process. Nothing was written — tap again."),
                crate::remote::events::Sensitivity::Public,
            )
            .await;
            return true;
        }
        self.adapter.note_routed(mid, &target).await;
        true
    }

    /// Was this update already queued before we started listening? Answered
    /// from the startup probe, by id — no clocks.
    fn is_backlog(&self, update_id: i64) -> bool {
        if crate::utils::config::get_config().remote_replay_backlog {
            return false;
        }
        self.state.lock().backlog_ids.contains(&update_id)
    }

    /// Hold a held-over message and ask, instead of dropping or running it.
    /// Scoped to the session that owns this bot (see the Python docstring).
    async fn offer_backlog(&self, text: &str, meta: &Value) {
        let owner = crate::remote::input::owning_session_id();
        let sessions: Vec<_> = crate::remote::inbox::live_sessions()
            .into_iter()
            .filter(|s| owner.is_empty() || s.session_id == owner)
            .collect();
        for adapter in crate::remote::notify::adapters() {
            if !adapter.is_configured() {
                continue;
            }
            let mut labels: Vec<String> = sessions
                .iter()
                .map(|s| {
                    if s.label.is_empty() {
                        s.session_id.chars().take(8).collect()
                    } else {
                        s.label.clone()
                    }
                })
                .collect();
            labels.push(tr("Discard"));
            let preview: String = text.chars().take(200).collect();
            if let Some(picker_id) = adapter
                .pick(&tr("Sent while nothing was listening — run it now?"), &preview, &labels)
                .await
            {
                // The discard choice is the one index past the sessions.
                let mut choices: Vec<String> =
                    sessions.iter().map(|s| s.session_id.clone()).collect();
                choices.push(String::new());
                crate::remote::inbox::park_route(picker_id, text, meta, &choices);
                return;
            }
        }
        // Nothing could ask. Holding it unread is the honest fallback.
        crate::utils::log::debug_log(
            "REMOTE_BACKLOG_UNASKED",
            Some(&serde_json::json!({"text": text.chars().take(80).collect::<String>()})),
        );
    }

    async fn on_message(&self, msg: &Value, update_id: Option<i64>) {
        self.on_message_with(msg, update_id, &[], None).await
    }

    /// `live`: the loop run this came from (worker only) — after the download,
    /// nothing is delivered if that run has ended.
    async fn on_message_with(&self, msg: &Value, update_id: Option<i64>, extra: &[&Value], live: Option<u64>) {
        if !self.adapter.is_trusted_chat(msg, None) {
            return;
        }
        // A photo's words are its caption; in an album each photo can have one.
        let mut parts: Vec<String> = Vec::new();
        for m in std::iter::once(msg).chain(extra.iter().copied()) {
            let t = m.get("text").or_else(|| m.get("caption")).and_then(|t| t.as_str()).unwrap_or("").trim();
            if !t.is_empty() && !parts.iter().any(|p| p == t) {
                parts.push(t.to_string());
            }
        }
        let mut text = parts.join("\n");
        let images: Vec<(String, String)> =
            std::iter::once(msg).chain(extra.iter().copied()).flat_map(image_refs).collect();
        if text.is_empty() && images.is_empty() {
            return; // stickers, joins — not input
        }
        if !images.is_empty() {
            // The session's input is a line of text (queue, mid-turn steering
            // and echo all carry strings), so a photo travels as the path it
            // was saved to and the model opens it with Read — the one way
            // images enter a conversation, which is also what resizes it,
            // numbers it, keeps the original in the session's image store and
            // counts it against the server's limit.
            // An album's photos download side by side, not one after another.
            let results = futures::future::join_all(
                images.iter().map(|(file_id, hint)| self.adapter.download_image(file_id, hint)),
            )
            .await;
            text = text_with_photos(&text, &results);
            if live.is_some_and(|g| !self.live(g)) {
                return; // this loop run ended during the download
            }
        }

        // /start and /help belong to the bot; everything else, including
        // codemason's own slash commands like /clear, is genuine input. A photo
        // is never a bot command, whatever its caption starts with.
        let interactive = self.state.lock().on_message.is_some();
        if images.is_empty() && is_bot_command(&text) {
            if interactive {
                self.on_bot_command(&text).await;
            }
            return;
        }

        let meta = serde_json::json!({
            "message_id": msg.get("message_id"),
            "reply_to": msg.get("reply_to_message").and_then(|r| r.get("message_id")),
            "thread_id": msg.get("message_thread_id"),
            // When the user actually sent it, not when we read it (Telegram
            // holds updates ~24h; carrying the send time lets the session show
            // the age instead of injecting it as though just typed).
            "sent_at": msg.get("date"),
        });
        if self.is_backlog(update_id.unwrap_or(0)) {
            if !interactive {
                return;
            }
            // The other interactive path (see dispatch): offer_backlog waits on
            // a picker, so move the offset before blocking on it.
            if let Some(id) = update_id {
                self.commit(id);
            }
            self.offer_backlog(&text, &meta).await;
            return;
        }

        // A question is waiting and you typed instead of tapping: that is the
        // answer. Checked here, ahead of the session's input, because the turn
        // is blocked on the question — filed as the next instruction, the text
        // sat unread until a button was tapped. Here rather than in the
        // session's router so a headless -p run, which has no router, gets it
        // too.
        if crate::remote::questions::answer_by_text(&text, &meta).await {
            return;
        }
        let cb = self.state.lock().on_message.clone();
        if let Some(cb) = cb {
            cb(text, meta);
        }
    }
}

/// (file_id, name hint) of each image a message carries. A photo comes in
/// several sizes, largest last; an image sent as a file keeps its original
/// bytes and name.
fn image_refs(msg: &Value) -> Vec<(String, String)> {
    let mut refs = Vec::new();
    if let Some(photo) = msg.get("photo").and_then(|p| p.as_array()) {
        let area = |p: &Value| {
            let g = |k: &str| p.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
            (g("width") * g("height"), g("file_size"))
        };
        if let Some(id) = photo.iter().max_by_key(|p| area(p)).and_then(|p| p.get("file_id")).and_then(|v| v.as_str()) {
            refs.push((id.to_string(), "photo.jpg".to_string()));
        }
    }
    if let Some(doc) = msg.get("document") {
        let mime = doc.get("mime_type").and_then(|m| m.as_str()).unwrap_or("");
        if let (Some(id), true) = (doc.get("file_id").and_then(|v| v.as_str()), mime.starts_with("image/")) {
            let name = doc.get("file_name").and_then(|v| v.as_str()).unwrap_or("");
            refs.push((id.to_string(), name.to_string()));
        }
    }
    refs
}

/// The message text with each photo named by its saved path. A download that
/// failed is said rather than dropped: the user sent something, and silence
/// reads as "it ignored my photo".
fn text_with_photos(text: &str, results: &[Result<String, String>]) -> String {
    let mut lines: Vec<String> = Vec::new();
    if !text.is_empty() {
        lines.push(text.to_string());
    }
    for r in results {
        lines.push(match r {
            Ok(path) => format!("[Photo from the phone: {path}]"),
            Err(e) => format!("[A photo was sent from the phone but could not be downloaded: {e}]"),
        });
    }
    lines.join("\n")
}

/// Consecutive updates of one media group as one list; everything else alone.
fn group_albums(updates: &[Value]) -> Vec<Vec<&Value>> {
    let gid = |u: &Value| {
        u.get("message").and_then(|m| m.get("media_group_id")).and_then(|g| g.as_str()).map(String::from)
    };
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    for upd in updates {
        let joins = match (gid(upd), groups.last().and_then(|g| g.last())) {
            (Some(g), Some(prev)) => gid(prev).as_deref() == Some(g.as_str()),
            _ => false,
        };
        if joins {
            groups.last_mut().unwrap().push(upd);
        } else {
            groups.push(vec![upd]);
        }
    }
    groups
}

/// Commands the BOT owns. Deliberately tiny — codemason's own slash commands
/// must still reach the session.
const BOT_COMMANDS: &[&str] = &["/start", "/help", "/status", "/sessions"];

/// Callback payload of the "Other" button on a question. Not "p:<n>": it
/// selects nothing, and a positional payload would be read as a choice.
pub const OTHER_DATA: &str = "o";

pub fn is_bot_command(text: &str) -> bool {
    let head = text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .split('@')
        .next()
        .unwrap_or("")
        .to_lowercase(); // '/help@mybot' → '/help'
    BOT_COMMANDS.contains(&head.as_str())
}

#[cfg(test)]
mod photo_tests {
    use super::*;
    use serde_json::json;

    /// Same cases as tests/test_remote_photo.py, for the pure parts.
    #[test]
    fn largest_photo_and_image_documents() {
        let m = json!({"photo": [
            {"file_id": "small", "width": 90, "height": 60},
            {"file_id": "big", "width": 1280, "height": 853},
            {"file_id": "mid", "width": 320, "height": 213}]});
        assert_eq!(image_refs(&m), vec![("big".to_string(), "photo.jpg".to_string())]);
        let d = json!({"document": {"file_id": "d1", "mime_type": "image/png", "file_name": "shot.png"}});
        assert_eq!(image_refs(&d), vec![("d1".to_string(), "shot.png".to_string())]);
        let pdf = json!({"document": {"file_id": "d2", "mime_type": "application/pdf"}});
        assert!(image_refs(&pdf).is_empty());
        assert!(image_refs(&json!({"sticker": {"file_id": "s"}})).is_empty());
    }

    #[test]
    fn text_names_paths_and_failures() {
        let ok: Vec<Result<String, String>> = vec![Ok("/att/big.jpg".into())];
        assert_eq!(text_with_photos("이거 뭐야?", &ok), "이거 뭐야?\n[Photo from the phone: /att/big.jpg]");
        let bad: Vec<Result<String, String>> = vec![Err("HTTP 400 file is too big".into())];
        assert_eq!(
            text_with_photos("", &bad),
            "[A photo was sent from the phone but could not be downloaded: HTTP 400 file is too big]"
        );
    }

    #[test]
    fn albums_grouped() {
        let mut ups: Vec<Value> = (0..3)
            .map(|i| json!({"update_id": 20 + i, "message": {"media_group_id": "g", "photo": []}}))
            .collect();
        ups.push(json!({"update_id": 23, "message": {"text": "next"}}));
        ups.push(json!({"update_id": 24, "message": {"media_group_id": "h", "photo": []}}));
        let sizes: Vec<usize> = group_albums(&ups).iter().map(|g| g.len()).collect();
        assert_eq!(sizes, vec![3, 1, 1]);
    }

    /// tests/test_remote_photo.py test_photo_on_worker_keeps_buttons_and_order,
    /// the offset half: a photo on the worker holds the offset, later updates
    /// are handled but skipped when handed back, and once the photo is done the
    /// offset passes everything.
    #[test]
    fn offset_waits_for_the_photo() {
        let mut b = OffsetBook::default();
        assert!(b.is_new(40) && b.is_new(41) && b.is_new(42));
        b.inflight.insert(40); // photo handed to the worker
        assert_eq!(b.commit(41), 40, "a button tap does not move it past the photo");
        b.inflight.insert(42); // text queued behind the photo
        assert!(!b.is_new(41) && !b.is_new(40), "handed back: not dispatched twice");
        assert_eq!(b.commit(40), 42);
        assert_eq!(b.commit(42), 43);
        assert!(b.seen.is_empty(), "nothing kept below the offset");
        assert!(b.is_new(43));
    }

    /// The fetch in flight when the offset moves answers with the update
    /// again — the photo must not be delivered a second time.
    #[test]
    fn an_update_the_offset_has_passed_is_not_new_again() {
        let mut b = OffsetBook::default();
        assert!(b.is_new(286));
        b.inflight.insert(286); // photo on the worker
        assert!(!b.is_new(286), "handed back while it downloads");
        assert_eq!(b.commit(286), 287); // worker done, offset moves
        assert!(
            !b.is_new(286),
            "the long poll that went out with offset 286 still answers with it"
        );
        assert!(b.is_new(287), "and the next update is still new");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A typed answer resolves a question that offers "Other", and only that:
    /// filed against a permission prompt it must read as nobody answering.
    #[tokio::test]
    async fn typed_answer_only_for_free_text_questions() {
        let mid = 876_543_210 + std::process::id() as i64;
        crate::remote::inbox::claim_question_with_options(mid, "s", 3, true);
        let q = Question::new(mid, vec!["A".into(), "B".into(), "C".into()], true);
        crate::remote::inbox::deliver_answer(mid, "t:  D로 가죠 ");
        assert_eq!(q.wait(2.0, None).await.as_deref(), Some("D로 가죠"));

        crate::remote::inbox::claim_question_with_options(mid, "s", 2, false);
        let grant = Question::new(mid, vec!["Allow".into(), "Deny".into()], false);
        crate::remote::inbox::deliver_answer(mid, "t:그냥 해");
        assert_eq!(grant.wait(0.7, None).await, None);
        crate::remote::inbox::release_question(mid);
    }

    #[test]
    fn bot_command_detection() {
        assert!(is_bot_command("/start"));
        assert!(is_bot_command("/help@mybot extra"));
        assert!(!is_bot_command("/clear"));
        assert!(!is_bot_command("hello /start"));
    }
}
