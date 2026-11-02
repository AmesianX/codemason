//! A spinner for the phone.
//! Equivalent to codemason/remote/progress.py — one message posted after a
//! hold-off, kept current by throttled edits, reposted at the bottom when
//! something lands below it, and finally rewritten into the answer. See the
//! Python module doc for the full reasoning.
//!
//! Substrate divergence (documented): the Python updater thread becomes a
//! tokio task; finish() takes ownership of the message id under the lock FIRST
//! (the same finished-flag protocol — correctness never rests on the join).

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::utils::i18n::tr;

/// Minimum gap between edits — Telegram throttles edits to roughly one per
/// second; this leaves headroom while still reading as motion.
pub const EDIT_INTERVAL: f64 = 2.5;

/// Below this, a turn finishes before a placeholder would even be useful.
pub const MIN_DURATION_TO_SHOW: f64 = 3.0;

const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Held while a prompt is waiting on an answer from the phone. A counter, not
/// a flag: two tools can be waiting at once, and the first to finish must not
/// un-pause the other.
static PAUSED: Lazy<Mutex<u32>> = Lazy::new(|| Mutex::new(0));

/// Stop updating any live progress message until `resume`.
pub fn pause() {
    *PAUSED.lock() += 1;
}

pub fn resume() {
    let mut p = PAUSED.lock();
    *p = p.saturating_sub(1);
}

/// Pause that un-pauses when it is dropped.
///
/// `pause()` / `resume()` as bare statements around an `.await` do not survive
/// the future being dropped, and these futures are dropped on ordinary paths:
/// the permission prompt races the phone against the terminal, so answering at
/// the terminal drops the remote arm mid-wait. `resume()` was then never
/// reached, and because PAUSED is a counter it stayed above zero for the rest of
/// the process — the phone's progress message froze after the first prompt
/// anyone answered locally, with nothing to unfreeze it.
///
/// Drop runs on cancellation as well as on completion, which is the whole point.
pub struct PauseGuard(());

impl PauseGuard {
    pub fn new() -> Self {
        pause();
        PauseGuard(())
    }
}

impl Drop for PauseGuard {
    fn drop(&mut self) {
        resume();
    }
}

fn is_paused() -> bool {
    *PAUSED.lock() > 0
}

struct ProgressState {
    text: String,
    tools: HashMap<String, u32>,
    message_id: Option<i64>,
    brk: bool, // a tool ran: the next text is a new block
    /// Set by finish(): the turn has taken the message id back and the updater
    /// no longer owns it.
    finished: bool,
    frame: usize,
}

/// Keeps one Telegram message current for the length of a turn.
pub struct RemoteProgress {
    /// How long to hold off before posting anything (never below the floor —
    /// that one is a rendering rule, not a filter).
    pub delay: f64,
    pub cwd: String,
    started: std::time::Instant,
    state: Arc<Mutex<ProgressState>>,
    stop: CancellationToken,
    adapter: Mutex<Option<Arc<crate::remote::adapters::telegram::TelegramAdapter>>>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl RemoteProgress {
    pub fn new(cwd: &str, delay: Option<f64>) -> Arc<RemoteProgress> {
        Arc::new(RemoteProgress {
            delay: MIN_DURATION_TO_SHOW.max(delay.unwrap_or(0.0)),
            cwd: cwd.to_string(),
            started: std::time::Instant::now(),
            state: Arc::new(Mutex::new(ProgressState {
                text: String::new(),
                tools: HashMap::new(),
                message_id: None,
                brk: false,
                finished: false,
                frame: 0,
            })),
            stop: CancellationToken::new(),
            adapter: Mutex::new(None),
            task: Mutex::new(None),
        })
    }

    // --- lifecycle -----------------------------------------------------------

    pub fn start(self: &Arc<Self>) {
        let adapters = crate::remote::notify::adapters();
        let adapter = adapters.into_iter().find(|a| a.is_configured());
        let Some(adapter) = adapter else {
            return;
        };
        *self.adapter.lock() = Some(adapter);
        let me = self.clone();
        let handle = tokio::spawn(async move {
            me.run().await;
        });
        *self.task.lock() = Some(handle);
    }

    /// Stop updating and hand back the message id to be rewritten as the
    /// answer, or None if nothing was ever posted — or when the placeholder is
    /// no longer the last message (rewriting a stranded placeholder would file
    /// the answer above the prompt you just tapped).
    pub async fn finish(self: &Arc<Self>) -> Option<i64> {
        // Take ownership of the message id FIRST, under the lock — the updater
        // may be inside a network call right now, and whoever else was
        // mid-flight must not write an id back after this point.
        self.stop.cancel();
        let mid = {
            let mut s = self.state.lock();
            s.finished = true;
            s.message_id
        };
        let task = self.task.lock().take();
        if let Some(t) = task {
            // Tidiness only now: nothing below depends on this completing.
            let timeout = EDIT_INTERVAL
                + self.adapter.lock().as_ref().map(|a| a.timeout).unwrap_or(10.0);
            let _ = tokio::time::timeout(std::time::Duration::from_secs_f64(timeout), t).await;
        }
        let adapter = self.adapter.lock().clone();
        let (Some(mid), Some(adapter)) = (mid, adapter) else {
            return mid;
        };
        if !adapter.progress_is_last(mid) {
            adapter.delete_message(mid).await;
            self.state.lock().message_id = None;
            return None;
        }
        Some(mid)
    }

    // --- turn events ---------------------------------------------------------

    /// Accumulate the answer prose, keeping the blocks apart (a tool call sat
    /// between them on screen; on the phone the break has to be put back).
    pub fn on_text(&self, delta: &str) {
        let mut s = self.state.lock();
        if s.brk && !delta.trim().is_empty() {
            s.text.push_str("\n\n");
            s.brk = false;
        }
        s.text.push_str(delta);
    }

    pub fn on_tool(&self, name: &str) {
        let mut s = self.state.lock();
        *s.tools.entry(name.to_string()).or_insert(0) += 1;
        // Whatever the model says next belongs after this call.
        s.brk = true;
    }

    // --- the updating loop ---------------------------------------------------

    async fn run(self: &Arc<Self>) {
        // Hold off at first: most turns are quick.
        tokio::select! {
            _ = self.stop.cancelled() => return,
            _ = tokio::time::sleep(std::time::Duration::from_secs_f64(self.delay)) => {}
        }
        loop {
            if self.stop.is_cancelled() {
                break;
            }
            if !is_paused() {
                self.push().await;
            }
            tokio::select! {
                _ = self.stop.cancelled() => break,
                _ = tokio::time::sleep(std::time::Duration::from_secs_f64(EDIT_INTERVAL)) => {}
            }
        }
    }

    fn snapshot(&self) -> String {
        let (text, tools, frame) = {
            let mut s = self.state.lock();
            s.frame = (s.frame + 1) % FRAMES.len();
            (s.text.clone(), s.tools.clone(), s.frame)
        };
        let elapsed = self.started.elapsed().as_secs_f64();

        let mut head = format!("{} {} ({elapsed:.0}s", FRAMES[frame], tr("working…"));
        if !tools.is_empty() {
            let mut entries: Vec<(&String, &u32)> = tools.iter().collect();
            entries.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let listed = entries
                .iter()
                .map(|(n, c)| if **c <= 1 { (*n).clone() } else { format!("{n} ×{c}") })
                .collect::<Vec<_>>()
                .join(" · ");
            head.push_str(&format!(" · {listed}"));
        }
        head.push(')');

        // Show the tail, not the head: the newest words tell you where it got to.
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let tail: String = {
                let chars: Vec<char> = trimmed.chars().collect();
                let start = chars.len().saturating_sub(500);
                chars[start..].iter().collect()
            };
            format!("{head}\n\n{tail}")
        } else {
            head
        }
    }

    async fn push(self: &Arc<Self>) {
        let adapter = self.adapter.lock().clone();
        let Some(adapter) = adapter else { return };
        let mid = self.state.lock().message_id;
        // The id can change under us: the adapter reposts at the bottom when
        // something has been posted below.
        let new_id = adapter.edit_progress(mid, &self.snapshot(), &self.cwd).await;
        let orphan = {
            let mut s = self.state.lock();
            if !s.finished {
                s.message_id = new_id;
                None
            } else {
                // finish() already handed the id to the turn while this call
                // was in flight. If the adapter reposted, the message it
                // created is ours to clean up.
                match new_id {
                    Some(n) if Some(n) != s.message_id => Some(n),
                    _ => None,
                }
            }
        };
        if let Some(orphan) = orphan {
            adapter.delete_message(orphan).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that move the process-global PAUSED counter.
    ///
    /// Both of these read it, change it, and assert on it. Run in parallel they
    /// see each other's pauses and the assertions compare unrelated numbers —
    /// measured as an intermittent failure of the cancellation test. Same reason
    /// as app_state::test_lock: a test that mutates process-global state has to
    /// be the only one doing it.
    fn pause_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pause must survive its future being CANCELLED, not just completed.
    /// A bare resume() after an .await does not, and these futures are dropped
    /// on the ordinary path where the terminal wins the permission race — so the
    /// counter stayed above zero and the phone's progress froze for the rest of
    /// the process.
    #[tokio::test]
    async fn a_cancelled_wait_still_un_pauses() {
        let _serial = pause_test_lock();
        let before = *PAUSED.lock();

        // A wait that never finishes, cancelled the way the select! arm is.
        let fut = async {
            let _g = PauseGuard::new();
            std::future::pending::<()>().await;
        };
        let timed = tokio::time::timeout(std::time::Duration::from_millis(20), fut).await;
        assert!(timed.is_err(), "the probe must have been cancelled, not completed");

        assert_eq!(
            *PAUSED.lock(), before,
            "the pause leaked: progress stays frozen for the rest of the session"
        );
    }

    /// Nesting still works — the counter exists because two tools can wait at
    /// once and the first to finish must not un-pause the other.
    #[test]
    fn nested_pauses_only_lift_at_the_last_one() {
        let _serial = pause_test_lock();
        let before = *PAUSED.lock();
        {
            let _a = PauseGuard::new();
            {
                let _b = PauseGuard::new();
                assert!(is_paused());
            }
            assert!(is_paused(), "the inner guard must not lift the outer pause");
        }
        assert_eq!(*PAUSED.lock(), before);
    }
}
