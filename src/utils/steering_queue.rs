//! Mid-turn user steering queue — codemason substrate for interjecting a line
//! into a turn that is ALREADY running.
//! Equivalent to codemason/utils/steering_queue.py; see that file's module doc
//! for the full design (mirrors message_queue_manager; the REPL's Enter
//! handler enqueues while a turn runs, query()'s mid-turn gate drains).

use once_cell::sync::Lazy;
use parking_lot::Mutex;

static PENDING: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Append a mid-turn steering line to inject into the running turn.
pub fn enqueue_steering_message(text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    PENDING.lock().push(text.to_string());
}

/// Return and clear all queued steering lines, in FIFO order.
pub fn drain_steering_messages() -> Vec<String> {
    std::mem::take(&mut *PENDING.lock())
}

/// True if any steering line is queued (peek, without draining).
pub fn has_steering_messages() -> bool {
    !PENDING.lock().is_empty()
}

/// Clear the queue (Ctrl+C hard-stop / tests).
pub fn reset_steering_messages() {
    PENDING.lock().clear();
}
