//! Todo store for CodeMason.
//! Equivalent to codemason/utils/todo_store.py.
//!
//! Faithful accessor over `AppState.todos` (the source keeps todo lists on
//! `appState.todos`, keyed by `agentId ?? sessionId`, read/written by
//! TodoWriteTool via getAppState/setAppState). Todos live on the real AppState
//! — NOT in a separate module-global map. These helpers are the thin get/set
//! layer over that slice.

use serde_json::Value;

use crate::state::app_state::{get_app_state, set_app_state};

/// Read the todo list for a key (appState.todos[key] ?? []).
pub fn get_todos(key: &str) -> Vec<Value> {
    get_app_state().todos.get(key).cloned().unwrap_or_default()
}

/// Write the todo list for a key (setAppState -> {...prev.todos, [key]: ...}).
pub fn set_todos(key: &str, todos: Vec<Value>) {
    let key = key.to_string();
    set_app_state(move |mut prev| {
        prev.todos.insert(key, todos);
        prev
    });
}

/// The key a conversation's todos live under: the sub-agent's id, or the
/// literal "session" for the main one.
///
/// This exists because the two sides disagreed. TodoWrite stored under
/// `agent_id ?? "session"` while the spinner read `get_session_id()` — a
/// UUID — so the lookup never matched and the in_progress todo's activeForm,
/// which is the whole reason that field exists, was dead code in both trees.
/// Both comments said "agentId ?? sessionId", which is why the mismatch
/// survived review. One function now, so there is nothing left to disagree.
pub fn todo_key(agent_id: Option<&str>) -> String {
    agent_id
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "session".to_string())
}

#[cfg(test)]
mod key_agreement_tests {
    use super::*;

    /// The bug this function exists to prevent: TodoWrite stored under
    /// `agent_id ?? "session"` while the spinner read `get_session_id()`, a
    /// UUID nothing ever wrote against. The lookup never matched, so an
    /// in_progress todo's activeForm — the entire reason that field exists —
    /// could not reach the screen. Both sides' comments said "agentId ??
    /// sessionId", which is why it survived review.
    #[test]
    fn what_todo_write_stores_is_what_the_spinner_reads() {
        let written = todo_key(None); // TodoWrite, main conversation
        let read = todo_key(None); // spinner
        assert_eq!(written, read);
        assert_eq!(written, "session", "not a session UUID");

        set_todos(&written, vec![serde_json::json!({
            "status": "in_progress",
            "activeForm": "Running the scan",
        })]);
        let seen = get_todos(&read);
        assert_eq!(seen.len(), 1, "the spinner must see what TodoWrite wrote");
        assert_eq!(seen[0]["activeForm"], "Running the scan");
        set_todos(&written, Vec::new());

        // A sub-agent keeps its own list, and does not collide with the parent.
        assert_eq!(todo_key(Some("a1b2c3")), "a1b2c3");
        assert_ne!(todo_key(Some("a1b2c3")), todo_key(None));
        // An empty id is not an id.
        assert_eq!(todo_key(Some("")), "session");
    }
}
