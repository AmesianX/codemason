//! Where codemason itself hands events to the remote channel.
//! Equivalent to codemason/remote/hooks.py — shared by the interactive REPL
//! and the headless `-p` runner. Everything here is best-effort: whether a
//! phone is attached must not change what the turn does.

use crate::remote::events::{Event, EventKind, Sensitivity};
use crate::utils::i18n::{tr, trf, FmtArg};

/// Only errors are capped here. Answers are passed whole and each transport
/// truncates to its own limit (one cap, at the layer with the constraint, and
/// it announces itself).
const MAX_ERROR: usize = 500;

/// How much of the originating prompt is echoed above an answer — enough to
/// recognise which question this was, not enough to re-read it.
const MAX_PROMPT_ECHO: usize = 200;

/// Put the question back above the answer, for turns started elsewhere.
/// A remote-origin turn already has your message in the chat above it.
fn with_prompt(body: &str, prompt: &str, remote_origin: bool) -> String {
    if prompt.is_empty() || remote_origin {
        return body.to_string();
    }
    let mut head = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    if head.chars().count() > MAX_PROMPT_ECHO {
        let cut: String = head.chars().take(MAX_PROMPT_ECHO - 1).collect();
        head = format!("{}…", cut.trim_end());
    }
    let asked = tr("You asked");
    if body.is_empty() {
        format!("{asked}: {head}")
    } else {
        format!("{asked}: {head}\n\n{body}")
    }
}

/// When a live progress message may start posting, or None for never — the
/// single source of truth for "will the phone hear about this turn?".
pub fn turn_report_delay(remote_origin: bool) -> Option<f64> {
    if !crate::remote::notify::is_configured() {
        return None;
    }
    if remote_origin {
        return Some(0.0);
    }
    let after = threshold();
    if after < 0.0 {
        None
    } else {
        Some(after)
    }
}

/// Headline for a turn event — the bot *is* codemason, so its own name says
/// nothing; other sources (a build, a download) still identify themselves.
fn titled(source: &str, what: &str) -> String {
    if source == "codemason" {
        what.to_string()
    } else {
        format!("{source}: {what}")
    }
}

/// Seconds a local turn must exceed before it is announced. 0 = every turn.
/// The fallback comes from the Config default, never a literal repeated here.
fn threshold() -> f64 {
    crate::utils::config::get_config().remote_notify_after
}

/// Everything notify_turn_end needs beyond duration+answer.
pub struct TurnEndInfo<'a> {
    pub cwd: Option<&'a str>,
    pub error: &'a str,
    pub source: &'a str,
    pub tools: Option<std::collections::HashMap<String, i64>>,
    pub remote_origin: bool,
    pub progress: Option<std::sync::Arc<crate::remote::progress::RemoteProgress>>,
    pub prompt: &'a str,
    /// The context/image line (utils/session_status.rs). Numbers only, so it
    /// rides at PUBLIC like the duration.
    pub status: &'a str,
}

impl Default for TurnEndInfo<'_> {
    fn default() -> Self {
        TurnEndInfo {
            cwd: None,
            error: "",
            source: "codemason",
            tools: None,
            remote_origin: false,
            progress: None,
            prompt: "",
            status: "",
        }
    }
}

fn tools_meta(tools: &Option<std::collections::HashMap<String, i64>>) -> serde_json::Value {
    match tools {
        Some(t) => serde_json::to_value(t).unwrap_or_else(|_| serde_json::json!({})),
        None => serde_json::json!({}),
    }
}

/// Announce that a turn ended. Short successful local turns are skipped
/// (`remote_notify_after`); errors always go out; `remote_origin` overrides
/// the threshold (if you asked from the phone, the answer belongs there).
pub async fn notify_turn_end(duration_ms: i64, answer: &str, info: TurnEndInfo<'_>) {
    // The progress placeholder is finished FIRST (ownership of the message id
    // moves to this turn), whether or not anything is sent below.
    let placeholder = match &info.progress {
        Some(p) => p.finish().await,
        None => None,
    };

    let after = threshold();
    if after < 0.0 && !info.remote_origin {
        return;
    }
    let elapsed = duration_ms as f64 / 1000.0;
    if info.error.is_empty() && !info.remote_origin && elapsed < after {
        return;
    }

    if !crate::remote::notify::is_configured() {
        return;
    }

    let error_cut: String = info.error.chars().take(MAX_ERROR).collect();
    let body_src = if info.error.is_empty() { answer.trim().to_string() } else { error_cut };
    let body = with_prompt(&body_src, info.prompt, info.remote_origin);
    let title = titled(
        info.source,
        &if info.error.is_empty() { tr("Turn done") } else { tr("Turn failed") },
    );

    // A progress message is already in the chat saying "working…" — rewrite it
    // into the answer instead of posting a second message.
    if let Some(mid) = placeholder {
        let mut meta = std::collections::HashMap::new();
        meta.insert("duration_ms".to_string(), serde_json::json!(duration_ms));
        meta.insert("tools".to_string(), tools_meta(&info.tools));
        meta.insert("status".to_string(), serde_json::json!(info.status));
        let done = Event {
            kind: if info.error.is_empty() { EventKind::TurnDone } else { EventKind::Error },
            title: title.clone(),
            body: body.clone(),
            sensitivity: Sensitivity::Internal,
            session_id: None,
            cwd: info.cwd.map(|c| c.to_string()),
            meta,
        };
        for a in crate::remote::notify::adapters() {
            if a.is_configured() && a.replace(mid, &done.redacted(a.max_sensitivity())).await {
                return;
            }
        }
    }

    let mut meta = std::collections::HashMap::new();
    meta.insert("duration_ms".to_string(), serde_json::json!(duration_ms));
    // Names and counts only. Deliberately not the arguments: a Bash command
    // line or a file path is SECRET and never takes this path.
    meta.insert("tools".to_string(), tools_meta(&info.tools));
    meta.insert("status".to_string(), serde_json::json!(info.status));
    crate::remote::notify::notify(
        if info.error.is_empty() { EventKind::TurnDone } else { EventKind::Error },
        &title,
        crate::remote::notify::NotifyOpts {
            body,
            sensitivity: Sensitivity::Internal,
            cwd: info.cwd.map(|c| c.to_string()),
            meta,
            ..Default::default()
        },
    )
    .await;
}

/// Repeat-title helper exported for parity with the Python `_()` templates
/// (used by tests; the ask paths build their own).
#[allow(dead_code)]
pub fn reask_title(title: &str, mins: i64) -> String {
    trf(
        "{title} (re-asking · {mins} min so far)",
        &[("title", FmtArg::Str(title.to_string())), ("mins", FmtArg::Int(mins))],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_echo_rules() {
        // remote-origin turns never echo the prompt.
        assert_eq!(with_prompt("body", "question", true), "body");
        // local turns get the echo above the answer.
        let s = with_prompt("body", "  multi\n word   question ", false);
        assert!(s.starts_with("You asked: multi word question\n\nbody") || s.contains("multi word question"), "{s}");
        // long prompts are cut with an ellipsis.
        let long = "x".repeat(300);
        let s = with_prompt("", &long, false);
        assert!(s.chars().count() < 300);
        assert!(s.ends_with('…'));
    }

    #[test]
    fn titles_skip_codemason() {
        assert_eq!(titled("codemason", "Turn done"), "Turn done");
        assert_eq!(titled("build", "Turn done"), "build: Turn done");
    }
}
