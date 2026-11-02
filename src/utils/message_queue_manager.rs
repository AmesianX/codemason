//! Pending-notification queue — codemason substrate for Claude Code's
//! messageQueueManager (enqueuePendingNotification / drain).
//! Equivalent to codemason/utils/message_queue_manager.py: a process-global
//! FIFO of ready-to-inject notification strings. LocalShellTask appends
//! completion notices on finish; the REPL drains them between turns, and
//! query()'s mid-turn gate drains them mid-turn.

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// (value, mode) entries. `mode` mirrors enqueuePendingNotification's mode
/// field ('task-notification'); kept for fidelity even though codemason
/// injects all modes the same way.
struct Entry {
    value: String,
    #[allow(dead_code)]
    mode: String,
}

static PENDING: Lazy<Mutex<Vec<Entry>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Append a ready-to-inject notification (enqueuePendingNotification).
pub fn enqueue_pending_notification(value: &str, mode: Option<&str>) {
    PENDING.lock().push(Entry {
        value: value.to_string(),
        mode: mode.unwrap_or("task-notification").to_string(),
    });
}

/// Return and clear all queued notification strings, in FIFO order.
pub fn drain_pending_notifications() -> Vec<String> {
    let mut q = PENDING.lock();
    let drained = q.iter().map(|e| e.value.clone()).collect();
    q.clear();
    drained
}

/// True if any notification is queued (peek, without draining).
///
/// Used by the headless wait loop to close the drain-vs-check race: a task
/// that finishes between a drain and the running-task check still leaves a
/// queued notification, so the loop must keep going until the queue is empty.
pub fn has_pending_notifications() -> bool {
    !PENDING.lock().is_empty()
}

/// Clear the queue (tests / `clear`).
pub fn reset_pending_notifications() {
    PENDING.lock().clear();
}
