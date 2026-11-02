//! AppState store — equivalent of codemason/state/app_state.py
//! (state/AppStateStore.ts).
//!
//! The one slice the task framework actually keys off is
//! `tasks: { [taskId]: TaskState }`, accessed via `get_app_state()` and
//! updated through `set_app_state(prev => next)`. Plan mode's permission
//! context and the TodoWrite slice live here too. See the Python reference
//! for the full design notes.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;

use crate::tasks::types::TaskState;

/// Ported slice of the source's toolPermissionContext (AppStateStore.ts).
/// Only the fields plan mode actually keys off are modeled: `mode`
/// ('default' | 'plan') and `pre_plan_mode` (the mode in effect when plan mode
/// was entered, restored by ExitPlanMode).
#[derive(Debug, Clone, Default)]
pub struct ToolPermissionContext {
    pub mode: String, // 'default' | 'plan' ('' treated as default)
    pub pre_plan_mode: Option<String>,
}

impl ToolPermissionContext {
    pub fn mode_or_default(&self) -> &str {
        if self.mode.is_empty() {
            "default"
        } else {
            &self.mode
        }
    }
}

/// The ported slice of the AppState.
///
/// `tasks` is the task registry — the single source of truth keyed by task id.
/// `todos` is keyed by agentId ?? sessionId (source AppState.todos).
#[derive(Debug, Clone, Default)]
pub struct AppState {
    pub tasks: HashMap<String, TaskState>,
    pub tool_permission_context: ToolPermissionContext,
    pub todos: HashMap<String, Vec<Value>>,
    /// name -> agentId for the sub-agents the model gave a name to
    /// (AppState.agentNameRegistry). This is what makes SendMessage({to: name})
    /// work: without it the model can only address an agent by the 16-hex id
    /// the spawn happened to mint, which it has to copy out of a tool result
    /// and carry by hand. Written only by a BACKGROUND spawn — a foreground
    /// agent is blocking its parent, so there is nobody left to send it
    /// anything.
    pub agent_name_registry: HashMap<String, String>,
}

/// Single maintained store instance (the AppStateStore singleton).
static APP_STATE: Lazy<Mutex<AppState>> = Lazy::new(|| Mutex::new(AppState::default()));

/// Read the current AppState (getAppState).
pub fn get_app_state() -> AppState {
    APP_STATE.lock().clone()
}

thread_local! {
    /// True while this thread is inside an `updater` closure.
    static IN_UPDATER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Apply an immutable update (setAppState). `updater` receives the previous
/// AppState and returns the next one, mirroring `setAppState(prev => next)`.
///
/// The closure runs while the store lock is HELD, which is what makes
/// read-modify-write atomic across tokio's worker threads — Python needs no lock
/// because it is single-threaded, and dropping the lock here to be safe would
/// trade a hang for a lost update between two threads.
///
/// The cost is that the lock is parking_lot's: not reentrant, and with no
/// deadlock detection. An updater that calls get_app_state() or set_app_state()
/// again would therefore not error — it would silently hang the whole session,
/// which is the least debuggable failure there is. No caller does this today
/// (every closure is a pure transform of `prev`), so the guard below exists to
/// keep it that way: re-entry is reported and skipped instead of hanging, and
/// panics in tests so it is caught at the point the mistake is made rather than
/// in production.
pub fn set_app_state<F: FnOnce(AppState) -> AppState>(updater: F) {
    if IN_UPDATER.with(|f| f.get()) {
        let msg = "set_app_state called from inside a set_app_state updater; \
                   the store lock is not reentrant. Compute the value first, \
                   then apply it in one update.";
        if cfg!(test) {
            panic!("{msg}");
        }
        crate::utils::log::log_error(&msg.to_string(), Some("app_state"));
        return;
    }
    let mut state = APP_STATE.lock();
    let prev = state.clone();
    IN_UPDATER.with(|f| f.set(true));
    // Restore the flag even if the updater panics, or one bad closure would
    // wedge every later update on this thread into the skip path above.
    let next = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| updater(prev)));
    IN_UPDATER.with(|f| f.set(false));
    match next {
        Ok(v) => *state = v,
        Err(e) => {
            drop(state);
            std::panic::resume_unwind(e);
        }
    }
}

/// Reset the store (used by tests / `clear`).
pub fn reset_app_state() {
    *APP_STATE.lock() = AppState::default();
}

/// Serializes tests that mutate the process-global AppState.
///
/// The store is a module singleton (as it is in the Python and in the source's
/// AppStateStore), so tests that reset it and then assert on their own task /
/// todo / plan-mode entries interfere when cargo runs them on parallel threads:
/// one test's `reset_app_state()` wipes the row another is about to read. Every
/// such test takes this lock first.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A poisoned lock only means some other test panicked while holding it —
    // the state is reset by each test anyway, so recover rather than cascade.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-entering an updater used to hang the whole session: parking_lot's
    /// mutex is not reentrant and does no deadlock detection, so the second
    /// lock would simply never return. It must be a named failure instead.
    ///
    /// Before the guard this test would HANG rather than fail, which is exactly
    /// why the guard is worth having — a hang tells you nothing about where.
    #[test]
    #[should_panic(expected = "not reentrant")]
    fn re_entering_an_updater_is_reported_not_hung() {
        let _guard = test_lock();
        set_app_state(|prev| {
            set_app_state(|inner| inner);
            prev
        });
    }

    /// A panicking updater must not leave the re-entry flag set — otherwise
    /// every later update on that thread silently does nothing.
    #[test]
    fn a_panicking_updater_does_not_wedge_the_thread() {
        let _guard = test_lock();
        reset_app_state();
        let outcome = std::panic::catch_unwind(|| {
            set_app_state(|_| panic!("boom"));
        });
        assert!(outcome.is_err(), "the panic must still propagate");

        // The store must still be usable from this same thread.
        set_app_state(|mut prev| {
            prev.todos.insert("probe".into(), vec![]);
            prev
        });
        assert!(
            get_app_state().todos.contains_key("probe"),
            "the flag stayed set and swallowed this update"
        );
        reset_app_state();
    }
}
