//! The notify API — the one thing the rest of the box calls.
//! Equivalent to codemason/remote/notify.py.
//!
//! Contract: never raises, never blocks the caller. A notification is not
//! worth a failed turn, and a wedged Telegram POST must not stall an agent
//! loop, so delivery runs on a background task and failures land in the debug
//! log.
//!
//! Substrate divergences (documented):
//! - Python's adapter registry was a duck-typed list (a Protocol); the only
//!   adapter is Telegram, and Rust keeps the registry as `Vec<Arc<TelegramAdapter>>`.
//!   A future adapter grows a trait here — the seam (adapters()/all_adapters())
//!   is unchanged.
//! - Python's daemon-thread + atexit drain becomes tokio tasks + an inflight
//!   counter; `drain_at_exit()` is called by the CLI/-p shutdown paths (Rust
//!   has no interpreter atexit for async work).

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::remote::adapters::telegram::TelegramAdapter;
use crate::remote::events::{Event, EventKind, Sensitivity};

/// Registry. A PWA/WebSocket adapter slots in later and both fire.
static ADAPTERS: Lazy<Vec<Arc<TelegramAdapter>>> =
    Lazy::new(|| vec![Arc::new(TelegramAdapter::new())]);

/// Why the channel is switched off wholesale, or "" when it is on.
///
/// One gate for the whole feature, checked where transports are handed out, so
/// "off" cannot mean off for some paths and on for others. Plain mode is the
/// pure-ASCII path for a dumb console — nobody drives one from a phone, so the
/// channel is not half-available there; it is not there.
pub fn off_reason() -> String {
    if let Ok(v) = std::env::var("CODEMASON_REMOTE") {
        if matches!(v.trim().to_lowercase().as_str(), "0" | "off" | "false" | "no") {
            return "--no-remote".to_string();
        }
    }
    if !crate::utils::config::get_config().remote_enabled {
        return "/remote off".to_string();
    }
    if crate::utils::formatting::plain_mode() {
        return "--plain/--ascii".to_string();
    }
    String::new()
}

/// The registry as configured, ignoring `off_reason`. Only for telling the
/// user what would be used — never for delivery.
pub fn all_adapters() -> Vec<Arc<TelegramAdapter>> {
    ADAPTERS.clone()
}

pub fn adapters() -> Vec<Arc<TelegramAdapter>> {
    if off_reason().is_empty() {
        ADAPTERS.clone()
    } else {
        Vec::new()
    }
}

/// True when at least one transport can actually deliver.
pub fn is_configured() -> bool {
    adapters().iter().any(|a| a.is_configured())
}

/// In-flight delivery tasks. Exit waits for them, briefly and boundedly —
/// a `-p` run finishes its turn and exits immediately, which would otherwise
/// silently drop every notification it tried to send.
static INFLIGHT: AtomicUsize = AtomicUsize::new(0);
const EXIT_GRACE: f64 = 5.0; // total seconds exit will wait for stragglers

/// Wait for in-flight notification deliveries (bounded). Call from the CLI /
/// print-runner shutdown paths — the Rust stand-in for the Python atexit hook.
pub async fn drain_at_exit() {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(EXIT_GRACE);
    while INFLIGHT.load(Ordering::SeqCst) > 0 && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

async fn deliver(event: Event) {
    for a in adapters() {
        if !a.is_configured() {
            continue;
        }
        // Redact per transport, not per event: the same event may go out in
        // full locally and stripped to Telegram.
        let redacted = event.redacted(a.max_sensitivity());
        if a.send(&redacted).await.is_none() {
            // send() logs its own failures; nothing else to do (defensive).
        }
    }
}

/// Options for `notify` beyond the required kind/title.
#[derive(Default)]
pub struct NotifyOpts {
    pub body: String,
    pub sensitivity: Sensitivity,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    /// blocking=true is for shutdown paths, where the process exits as soon as
    /// the caller returns and a background task would be killed mid-POST.
    pub blocking: bool,
    pub meta: std::collections::HashMap<String, serde_json::Value>,
}

/// Fire an event at every configured transport.
pub async fn notify(kind: EventKind, title: &str, opts: NotifyOpts) {
    let event = Event {
        kind,
        title: title.to_string(),
        body: opts.body,
        sensitivity: opts.sensitivity,
        session_id: opts.session_id,
        cwd: opts.cwd,
        meta: opts.meta,
    };
    if opts.blocking {
        deliver(event).await;
        return;
    }
    INFLIGHT.fetch_add(1, Ordering::SeqCst);
    tokio::spawn(async move {
        deliver(event).await;
        INFLIGHT.fetch_sub(1, Ordering::SeqCst);
    });
}

/// Convenience for plain fire-and-forget info-style calls.
pub async fn notify_simple(kind: EventKind, title: &str, body: &str, sensitivity: Sensitivity) {
    notify(
        kind,
        title,
        NotifyOpts { body: body.to_string(), sensitivity, ..Default::default() },
    )
    .await;
}
