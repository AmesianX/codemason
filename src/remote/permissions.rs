//! Remote tool approval.
//! Equivalent to codemason/remote/permissions.py — both prompts go up and the
//! first answer wins; None from an adapter means "nobody answered *here*",
//! never a decision. See the Python module doc for the ownership argument.

use crate::remote::events::{Event, EventKind, Sensitivity};
use crate::utils::i18n::{tr, trf, FmtArg};
use tokio_util::sync::CancellationToken;

/// Button labels, in the order they appear on the phone. 'always' (a) is
/// deliberately NOT offered remotely — a phone answer is a reflex tap, and a
/// blanket session grant deserves the terminal's context. Kept as canonical
/// English here and translated at call time, not at import.
const ALLOW: &str = "✅ Allow";
const DENY: &str = "⛔ Deny";

fn buttons() -> Vec<String> {
    vec![tr(ALLOW), tr(DENY)]
}

/// The tapped button back to the single key the local prompt speaks. Compared
/// against the list that was actually sent, not a fresh translation — the
/// language can change between posting and the tap landing.
fn to_key(answer: Option<&str>, options: &[String]) -> Option<String> {
    let answer = answer?;
    if !options.is_empty() && answer == options[0] {
        return Some("y".to_string());
    }
    if options.len() > 1 && answer == options[1] {
        return Some("n".to_string());
    }
    if answer == tr(ALLOW) || answer == ALLOW {
        return Some("y".to_string());
    }
    if answer == tr(DENY) || answer == DENY {
        return Some("n".to_string());
    }
    None
}

/// Fallback when config is unreadable; the real value is `remote_ask_timeout`.
/// NOT a display deadline — the buttons stay usable while the question is
/// live; this bounds the case where nothing else ever answers.
pub const DEFAULT_TIMEOUT: f64 = 1800.0;

/// Ceiling on the yield between re-ask rounds — a hang guard, not the
/// mechanism (the wait ends when the lock is observably gone).
pub const RELOCK_CEILING: f64 = 30.0;
pub const RELOCK_POLL: f64 = 0.25;

fn timeout_setting() -> f64 {
    crate::utils::config::get_config().remote_ask_timeout
}

/// A one-glance description of what is about to run — the tool's own renderer,
/// so the phone shows the same thing the terminal does. Never nothing.
fn summary(tool: &dyn crate::tools::base::Tool, input_data: &serde_json::Value) -> String {
    let rendered = tool.render_tool_use_message(input_data, false);
    if !rendered.trim().is_empty() {
        return rendered.trim().to_string();
    }
    let raw = serde_json::to_string(input_data).unwrap_or_else(|_| format!("{input_data:?}"));
    raw.chars().take(400).collect()
}

/// Run one adapter round-trip. Returns the option picked, verbatim, or None
/// when nobody answered here. Says nothing about what the options mean.
async fn ask_round(
    kind: EventKind,
    title: &str,
    body: &str,
    options: &[String],
    cwd: Option<&str>,
    timeout: f64,
    cancel: CancellationToken,
    meta: Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Option<String> {
    // The body can quote a command line or a file path, so it is INTERNAL.
    let event = Event {
        kind,
        title: title.to_string(),
        body: body.to_string(),
        sensitivity: Sensitivity::Internal,
        session_id: None,
        cwd: cwd.map(|c| c.to_string()),
        meta: meta.unwrap_or_default(),
    };
    for adapter in crate::remote::notify::adapters() {
        if !adapter.is_configured() {
            continue;
        }
        let redacted = event.redacted(adapter.max_sensitivity());
        if let Some(answer) = adapter.ask(&redacted, options, timeout, Some(cancel.clone())).await {
            return Some(answer);
        }
    }
    None
}

/// Put a choice on the phone and wait. Returns the chosen option, or None.
/// The waiting lives here once: cooperative cancellation, the poll lock
/// released between rounds, and re-asking rather than giving up while the
/// turn is still stopped.
pub async fn ask_options(
    kind: EventKind,
    title: &str,
    body: &str,
    options: &[String],
    cwd: Option<&str>,
    timeout: Option<f64>,
    meta: Option<std::collections::HashMap<String, serde_json::Value>>,
    repeat_title: Option<&(dyn Fn(i64) -> String + Send + Sync)>,
) -> Option<String> {
    if !is_available() {
        return None;
    }
    let timeout = timeout.unwrap_or_else(timeout_setting);
    if timeout <= 0.0 {
        return None; // remote prompting disabled
    }
    let cancel = CancellationToken::new();

    // A live question owns the bottom of the chat — the progress message holds
    // still until this resolves.
    // Guard, not a bare pause(): this future is dropped whenever the terminal
    // wins the race in cli.rs, and a plain resume() after the .await never runs
    // in that case.
    let _pause = crate::remote::progress::PauseGuard::new();
    let result = async {
        // Ask again rather than give up: the window closing does not mean the
        // question stopped mattering. Each round is a fresh push; between
        // rounds the poll lock is released so a long wait cannot starve
        // another session.
        let mut round_no: i64 = 0;
        loop {
            let shown = if round_no == 0 {
                title.to_string()
            } else {
                let mins = (timeout * round_no as f64 / 60.0) as i64;
                match repeat_title {
                    Some(f) => f(mins),
                    None => trf(
                        "{title} (re-asking · {mins} min so far)",
                        &[("title", FmtArg::Str(title.to_string())), ("mins", FmtArg::Int(mins))],
                    ),
                }
            };
            let answer = ask_round(
                kind,
                &shown,
                body,
                options,
                cwd,
                timeout,
                cancel.clone(),
                meta.clone(),
            )
            .await;
            if let Some(answer) = answer {
                return Some(answer);
            }
            if cancel.is_cancelled() {
                return None; // the terminal answered
            }
            round_no += 1;
            yield_poll_lock().await;
        }
    }
    .await;
    result
}

/// True while any configured adapter still owns the inbound lock here.
fn holds_poll_lock() -> bool {
    crate::remote::notify::adapters().iter().any(|a| a.holds_poll_lock())
}

/// Between re-ask rounds, actually let go before asking again — skipped when
/// there is nobody to yield to (a claimed bot has exactly one owner), and
/// otherwise a wait on the FACT of release, bounded only as a hang guard.
async fn yield_poll_lock() {
    if !crate::remote::adapters::telegram::claimed_token().is_empty() {
        return; // ours exclusively; no other claimant
    }
    // No claim: the bot may be shared, and then the session count is the only
    // signal there is.
    if crate::remote::inbox::live_sessions().len() <= 1 {
        return; // no other claimant; nothing to yield
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(RELOCK_CEILING);
    while holds_poll_lock() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_secs_f64(RELOCK_POLL)).await;
    }
}

/// True when at least one adapter could actually field a prompt.
pub fn is_available() -> bool {
    crate::remote::notify::adapters().iter().any(|a| a.is_configured())
}

/// Ask the phone. Returns 'y' | 'n', or None if nobody answered.
/// None is not a decision — callers fall back to the local prompt.
pub async fn ask(
    tool: &dyn crate::tools::base::Tool,
    input_data: &serde_json::Value,
    cwd: Option<&str>,
    timeout: Option<f64>,
) -> Option<String> {
    let tool_name = tool.name();
    let options = buttons();
    let mut meta = std::collections::HashMap::new();
    meta.insert("tool".to_string(), serde_json::Value::String(tool_name.to_string()));
    let title = trf("Permission needed: {tool}", &[("tool", FmtArg::Str(tool_name.to_string()))]);
    let tool_name_owned = tool_name.to_string();
    let repeat = move |mins: i64| {
        trf(
            "Permission needed (re-asking · {mins} min so far): {tool}",
            &[("mins", FmtArg::Int(mins)), ("tool", FmtArg::Str(tool_name_owned.clone()))],
        )
    };
    let answer = ask_options(
        EventKind::PermissionNeeded,
        &title,
        &summary(tool, input_data),
        &options,
        cwd,
        timeout,
        Some(meta),
        Some(&repeat),
    )
    .await;
    to_key(answer.as_deref(), &options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_maps_to_local_keys() {
        let opts = buttons();
        assert_eq!(to_key(Some(&opts[0]), &opts).as_deref(), Some("y"));
        assert_eq!(to_key(Some(&opts[1]), &opts).as_deref(), Some("n"));
        assert_eq!(to_key(Some("✅ Allow"), &[]).as_deref(), Some("y"));
        assert_eq!(to_key(Some("garbage"), &opts), None);
        assert_eq!(to_key(None, &opts), None);
    }
}
