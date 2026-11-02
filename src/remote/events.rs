//! Backend-agnostic event model.
//! Equivalent to codemason/remote/events.py.
//!
//! An Event says *what happened*, never *how to render it* — that belongs to
//! the adapter, because the same event has to become a Telegram message today
//! and a WebSocket frame for a PWA later.
//!
//! Sensitivity is part of the model, not an afterthought. Telegram cloud chats
//! are not end-to-end encrypted and bots cannot use Secret Chats, so
//! everything handed to that adapter is stored in plaintext on a third party's
//! servers. This box holds research code, serving API keys and CTF material,
//! so each event declares how far it may travel and the adapter enforces it.

use serde_json::Value;
use std::collections::HashMap;

/// What happened. Kept small on purpose — a kind earns its place only when
/// something would act differently on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// assistant finished; user's move
    TurnDone,
    /// blocked, waiting on a human
    PermissionNeeded,
    /// blocked, but on a choice, not a grant
    QuestionAsked,
    /// turn died
    Error,
    /// background work finished
    TaskDone,
    /// anything else worth a buzz
    Info,
}

impl EventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventKind::TurnDone => "turn_done",
            EventKind::PermissionNeeded => "permission_needed",
            EventKind::QuestionAsked => "question_asked",
            EventKind::Error => "error",
            EventKind::TaskDone => "task_done",
            EventKind::Info => "info",
        }
    }

    pub fn from_str(s: &str) -> Option<EventKind> {
        match s {
            "turn_done" => Some(EventKind::TurnDone),
            "permission_needed" => Some(EventKind::PermissionNeeded),
            "question_asked" => Some(EventKind::QuestionAsked),
            "error" => Some(EventKind::Error),
            "task_done" => Some(EventKind::TaskDone),
            "info" => Some(EventKind::Info),
            _ => None,
        }
    }
}

/// How far this payload may travel.
///
/// PUBLIC   — metadata only: 'a turn finished', a tool name, a duration. Safe
///            for any transport.
/// INTERNAL — content: the assistant's answer text, and the body of a
///            permission prompt (which quotes a command line, a path, or the
///            first lines of a file about to be written). Opt-in per adapter.
/// SECRET   — reserved. Nothing constructs it today, and that is deliberate;
///            see below.
///
/// WHY SECRET IS UNUSED, and why leaving it that way is the decision rather
/// than an oversight (a security audit read the tier list and filed this as a
/// finding, which is what this note exists to prevent):
///
/// The one payload that would qualify is the permission prompt body. Tag it
/// SECRET and `redacted` replaces it with "(withheld — content above this
/// channel's clearance)" for every remote adapter, because none of them can
/// clear it. The phone then shows a prompt with Allow and Deny buttons and no
/// way to know what is being allowed. Approving blind is worse than the
/// exposure it would be avoiding, and it makes remote approval — the reason
/// this channel exists — pointless.
///
/// So the cost is stated plainly instead: with a remote adapter configured,
/// the command lines you approve from your phone, and the leading lines of
/// files being written, are stored in plaintext by that transport. For
/// Telegram specifically, bots cannot use Secret Chats, so that means a third
/// party's servers. Anyone who does not want that turns the remote channel off;
/// there is no setting that keeps both the approval and the secrecy, because
/// they are the same information.
///
/// The tier is kept rather than deleted because it is waiting for its
/// transport. Telegram is a TEST adapter, not the destination: the plan is a
/// dedicated remote-control app once this stabilises, and that is why the
/// adapter abstraction exists at all rather than the channel being written
/// against Telegram directly. An app the operator controls end to end can
/// declare max_sensitivity = SECRET honestly, and on that day the argument
/// above stops applying — the prompt body rides the channel with its real
/// classification, and `redacted` starts doing the job it was written for
/// against any transport that cannot clear it.
///
/// So do not read the unused tier as dead code to delete. It is the half of
/// the design that Telegram cannot satisfy, kept intact for the transport that
/// can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Sensitivity {
    #[default]
    Public = 0,
    Internal = 1,
    Secret = 2,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    /// one line, always PUBLIC
    pub title: String,
    /// detail, rated by `sensitivity`
    pub body: String,
    pub sensitivity: Sensitivity,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    /// Free-form extras an adapter may use (tool name, duration_ms, token
    /// counts). Treated at the event's own sensitivity level.
    pub meta: HashMap<String, Value>,
}

impl Event {
    pub fn new(kind: EventKind, title: impl Into<String>) -> Event {
        Event {
            kind,
            title: title.into(),
            body: String::new(),
            sensitivity: Sensitivity::Public,
            session_id: None,
            cwd: None,
            meta: HashMap::new(),
        }
    }

    /// A copy safe to hand to a transport cleared only up to `max_level`.
    ///
    /// The title always survives — the whole point of a notification is that
    /// it reaches you — but the body is dropped when it outranks the
    /// transport, and the drop is made visible rather than silent, so a
    /// truncated alert is never mistaken for the full picture.
    pub fn redacted(&self, max_level: Sensitivity) -> Event {
        if self.sensitivity <= max_level {
            return self.clone();
        }
        Event {
            kind: self.kind,
            title: self.title.clone(),
            body: "(withheld — content above this channel's clearance)".to_string(),
            sensitivity: max_level,
            session_id: self.session_id.clone(),
            cwd: self.cwd.clone(),
            // Kept even when the body is withheld: these are counts and tool
            // names, never paths or contents. They are the phone-sized answer
            // to "what did it actually do?" — the same signal Claude Code's app
            // gets from a collapsed tool list, without shipping what the tools
            // read. `status` is the context/image line ("Context 22% · images
            // 6/8"): numbers only.
            meta: self
                .meta
                .iter()
                .filter(|(k, _)| matches!(k.as_str(), "tool" | "duration_ms" | "tools" | "status"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redaction_drops_body_keeps_safe_meta() {
        let mut e = Event::new(EventKind::TurnDone, "t");
        e.body = "secret prose".into();
        e.sensitivity = Sensitivity::Internal;
        e.meta.insert("tools".into(), serde_json::json!({"Bash": 2}));
        e.meta.insert("path".into(), serde_json::json!("/etc/passwd"));
        let r = e.redacted(Sensitivity::Public);
        assert_eq!(r.body, "(withheld — content above this channel's clearance)");
        assert!(r.meta.contains_key("tools"));
        assert!(!r.meta.contains_key("path"));
        // At-or-below clearance passes through whole.
        let same = e.redacted(Sensitivity::Internal);
        assert_eq!(same.body, "secret prose");
    }
}
