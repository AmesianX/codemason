//! Stopping every background agent at once — the two-press gesture.
//!
//! A background sub-agent is unlinked from the turn's abort scope on purpose,
//! so Esc cancels the turn and leaves it running. The source pairs that with a
//! deliberate way to end them in bulk, and says why it is deliberate: "killing
//! agents from the main prompt stays a deliberate gesture (chat:killAgents),
//! not a side-effect of cancelling a turn."
//!
//! Two presses, inside the source's own 3s window, because one keystroke must
//! not be able to throw away three agents' work. The first press arms and says
//! so; the second does it. A press with nothing running says nothing is
//! running, rather than looking broken.
//!
//! Two entry points share one gesture here:
//!
//!   * Ctrl+C — our brake. Esc is "interject now", Ctrl+C is the one key that
//!     brings the session to rest (it drops the queued backlog too), so it is
//!     also where someone reaches when they want everything to stop. Its hint
//!     only appears when agents are actually running, which is what makes the
//!     gesture discoverable at the moment it matters.
//!   * Ctrl+X Ctrl+K — the source's own chord, kept so muscle memory from
//!     Claude Code works here.
//!
//! The window is shared between them: what is armed is the gesture, not the
//! key.

use parking_lot::Mutex;
use std::time::{Duration, Instant};

/// KILL_AGENTS_CONFIRM_WINDOW_MS.
pub const CONFIRM_WINDOW: Duration = Duration::from_secs(3);

/// What a press did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Press {
    /// Nothing was running; nothing to confirm.
    Idle,
    /// Armed — the next press inside the window kills.
    Armed,
    /// Every running agent was stopped.
    Killed,
}

/// The arm-then-fire state for the bulk stop. One per REPL.
pub struct KillAgentsGesture {
    /// (armed at, the key that armed it) — the key so the hint names the one in
    /// the user's hand rather than whichever this module happens to prefer.
    armed: Mutex<Option<(Instant, &'static str)>>,
    window: Duration,
}

impl Default for KillAgentsGesture {
    fn default() -> Self {
        Self::new(CONFIRM_WINDOW)
    }
}

impl KillAgentsGesture {
    pub fn new(window: Duration) -> Self {
        Self { armed: Mutex::new(None), window }
    }

    /// Armed and still inside the window.
    pub fn armed_at(&self, now: Instant) -> bool {
        matches!(*self.armed.lock(), Some((t, _)) if now.duration_since(t) <= self.window)
    }

    pub fn is_armed(&self) -> bool {
        self.armed_at(Instant::now())
    }

    /// What to show while armed, or None. Painted by the rprompt.
    pub fn hint(&self) -> Option<String> {
        let now = Instant::now();
        match *self.armed.lock() {
            Some((t, key)) if now.duration_since(t) <= self.window => {
                Some(format!(" ({key} again to stop background agents) "))
            }
            _ => None,
        }
    }

    /// One press. Returns what it did and the line to show the user.
    ///
    /// The model is told separately, by kill_all_agents_and_notify, and only on
    /// the press that kills. The task table is read on every press rather than
    /// captured, because a gesture must act on what is running NOW — the source
    /// reads its store here for the same reason ("Reads tasks from the store
    /// directly to avoid stale closures"). So an arm is not a promise: if the
    /// agents finish in between, the second press finds nothing to kill.
    pub fn press(&self, key: &'static str) -> (Press, Option<String>) {
        self.press_at(key, Instant::now())
    }

    pub fn press_at(&self, key: &'static str, now: Instant) -> (Press, Option<String>) {
        let running = crate::state::app_state::get_app_state()
            .tasks
            .values()
            .filter(|t| {
                t.as_local_agent()
                    .map(|a| a.base.status == "running")
                    .unwrap_or(false)
            })
            .count();
        if running == 0 {
            *self.armed.lock() = None;
            return (Press::Idle, None);
        }

        if self.armed_at(now) {
            *self.armed.lock() = None;
            let summary = crate::tasks::local_agent_task::kill_all_agents_and_notify();
            return (Press::Killed, summary);
        }

        *self.armed.lock() = Some((now, key));
        let (plural, them) = if running == 1 { ("", "it") } else { ("s", "them") };
        (
            Press::Armed,
            Some(format!(
                "{running} background agent{plural} still running — press {key} again to stop {them}"
            )),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::app_state::{get_app_state, reset_app_state};
    use crate::tasks::local_agent_task::{complete_agent_task, register_async_agent};
    use crate::utils::message_queue_manager as mq;
    use tokio_util::sync::CancellationToken;

    fn fake_agent() -> crate::tools::built_in_agents::AgentDefinition {
        crate::tools::built_in_agents::get_agent_definition(None)
            .expect("the default agent definition")
    }

    /// n running background agents, each on its own unlinked token.
    fn spawn(n: usize) -> Vec<CancellationToken> {
        reset_app_state();
        mq::reset_pending_notifications();
        (0..n)
            .map(|i| {
                let token = CancellationToken::new();
                register_async_agent(
                    &format!("a{}", "0".repeat(16 - i.to_string().len()) + &i.to_string()),
                    &format!("lane {i}"),
                    "p",
                    fake_agent(),
                    None,
                    Some(token.clone()),
                );
                token
            })
            .collect()
    }

    #[test]
    fn a_press_with_nothing_running_says_so() {
        let _g = crate::state::app_state::test_lock();
        reset_app_state();
        mq::reset_pending_notifications();
        let gesture = KillAgentsGesture::default();
        assert_eq!(gesture.press("Ctrl+C"), (Press::Idle, None));
        assert!(!gesture.is_armed(), "nothing is armed");
        assert!(gesture.hint().is_none(), "and nothing is painted");
        assert!(mq::drain_pending_notifications().is_empty());
        reset_app_state();
    }

    #[test]
    fn one_press_arms_and_the_second_kills() {
        let _g = crate::state::app_state::test_lock();
        let tokens = spawn(3);
        let gesture = KillAgentsGesture::default();

        let (press, note) = gesture.press("Ctrl+C");
        assert_eq!(press, Press::Armed);
        assert_eq!(
            note.as_deref(),
            Some("3 background agents still running — press Ctrl+C again to stop them")
        );
        assert!(tokens.iter().all(|t| !t.is_cancelled()), "the first press kills nothing");
        assert!(mq::drain_pending_notifications().is_empty(), "and tells the model nothing");
        assert_eq!(
            gesture.hint().as_deref(),
            Some(" (Ctrl+C again to stop background agents) ")
        );

        let (press, summary) = gesture.press("Ctrl+C");
        assert_eq!(press, Press::Killed);
        assert!(tokens.iter().all(|t| t.is_cancelled()));
        let notes = mq::drain_pending_notifications();
        assert_eq!(notes.len(), 1, "told once: {notes:?}");
        assert_eq!(summary.as_deref(), Some(notes[0].as_str()));
        assert!(!gesture.is_armed(), "and it disarms after firing");
        reset_app_state();
        mq::reset_pending_notifications();
    }

    #[test]
    fn the_window_expires_and_the_two_keys_share_it() {
        let _g = crate::state::app_state::test_lock();
        let tokens = spawn(2);
        let gesture = KillAgentsGesture::default();
        let t0 = Instant::now();

        gesture.press_at("Ctrl+C", t0);
        assert!(gesture.armed_at(t0 + CONFIRM_WINDOW - Duration::from_millis(10)));
        assert!(!gesture.armed_at(t0 + CONFIRM_WINDOW + Duration::from_millis(10)));

        let (press, _) = gesture.press_at("Ctrl+C", t0 + CONFIRM_WINDOW + Duration::from_millis(500));
        assert_eq!(press, Press::Armed, "a late press arms again rather than killing");
        assert!(tokens.iter().all(|t| !t.is_cancelled()));

        // The chord completes a gesture Ctrl+C armed: what is armed is the
        // gesture, not the key.
        let (press, _) =
            gesture.press_at("Ctrl+X Ctrl+K", t0 + CONFIRM_WINDOW + Duration::from_millis(600));
        assert_eq!(press, Press::Killed);
        assert!(tokens.iter().all(|t| t.is_cancelled()));
        reset_app_state();
        mq::reset_pending_notifications();
    }

    #[test]
    fn an_arm_is_not_a_promise() {
        let _g = crate::state::app_state::test_lock();
        let ids: Vec<String> = {
            spawn(2);
            get_app_state().tasks.keys().cloned().collect()
        };
        let gesture = KillAgentsGesture::default();
        assert_eq!(gesture.press("Ctrl+C").0, Press::Armed);
        // They finish on their own between the presses.
        for id in &ids {
            complete_agent_task(id, serde_json::json!({"text": "done"}));
        }
        assert_eq!(
            gesture.press("Ctrl+C").0,
            Press::Idle,
            "the second press reads the table again, it does not kill from a captured list"
        );
        assert!(
            !mq::drain_pending_notifications()
                .iter()
                .any(|n| n.contains("stopped by the user")),
            "and says nothing to the model"
        );
        reset_app_state();
        mq::reset_pending_notifications();
    }
}
