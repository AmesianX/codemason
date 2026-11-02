//! Remote answering for AskUserQuestion.
//! Equivalent to codemason/remote/questions.py — same shape as `permissions`,
//! deliberately: both prompts go up, the first answer wins, and None means
//! "nobody answered *here*", never a choice. Multi-select stays on the
//! terminal; free text is carried through the chat itself — a message typed
//! while a question waits answers it (`answer_by_text`), and an "Other" button
//! says so. Before, the typed text was filed as the next instruction and sat
//! unread while the turn stayed blocked on the question. Permission prompts are
//! excluded: a sentence typed at one must never become a grant. (See the Python
//! module doc.)

use crate::remote::events::EventKind;
use crate::utils::i18n::{tr, trf, FmtArg};

/// Telegram's inline keyboard rejects a callback payload over 64 bytes, and
/// long labels wrap into unreadable buttons — the index is what travels, so
/// only the visible text is trimmed.
const MAX_LABEL: usize = 40;

fn short(label: &str) -> String {
    if label.chars().count() <= MAX_LABEL {
        label.to_string()
    } else {
        let head: String = label.chars().take(MAX_LABEL - 1).collect();
        format!("{head}…")
    }
}

pub fn is_available() -> bool {
    crate::remote::permissions::is_available()
}

/// Put a question on the phone. Returns the chosen label, or None.
///
/// The label is returned verbatim — trimming happens on the button, never on
/// the answer, so what comes back is exactly what the local picker would have
/// produced.
pub async fn ask(
    question: &str,
    header: &str,
    labels: &[String],
    descriptions: Option<&[String]>,
    cwd: Option<&str>,
    timeout: Option<f64>,
) -> Option<String> {
    if labels.is_empty() {
        return None;
    }

    // The options are already on the buttons; the body carries what the
    // terminal shows underneath each one.
    let mut body_lines = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let desc = descriptions.and_then(|d| d.get(i)).map(|s| s.as_str()).unwrap_or("");
        if desc.is_empty() {
            body_lines.push(format!("• {label}"));
        } else {
            body_lines.push(format!("• {label} — {desc}"));
        }
    }
    let body = if body_lines.is_empty() {
        question.to_string()
    } else {
        format!("{question}\n\n{}", body_lines.join("\n"))
    };

    let title = if header.is_empty() {
        tr("Question")
    } else {
        trf("Question: {header}", &[("header", FmtArg::Str(header.to_string()))])
    };
    let title_for_repeat = title.clone();
    let repeat = move |mins: i64| {
        trf(
            "{title} (re-asking · {mins} min so far)",
            &[("title", FmtArg::Str(title_for_repeat.clone())), ("mins", FmtArg::Int(mins))],
        )
    };
    let shorts: Vec<String> = labels.iter().map(|l| short(l)).collect();
    let chosen = crate::remote::permissions::ask_options(
        EventKind::QuestionAsked,
        &title,
        &body,
        &shorts,
        cwd,
        timeout,
        None,
        Some(&repeat),
    )
    .await?;

    // Map the button text back to the full label it was trimmed from.
    for label in labels {
        if short(label) == chosen {
            return Some(label.clone());
        }
    }
    Some(chosen)
}

/// File a typed message as the answer to a waiting question. True if it was.
///
/// Runs in whichever process holds the poll lock, which need not be the one
/// that asked — so the target is found through the on-disk claims, and the
/// answer goes through the same forwarding file a button tap uses.
///
/// reply → that question (a reply to anything else is not an answer here);
/// only one question taking text → it; several → ask you to reply to the one
/// you mean rather than guess, and do not file the text as an instruction
/// either. A slash command is never an answer.
pub async fn answer_by_text(text: &str, meta: &serde_json::Value) -> bool {
    let body = text.trim();
    if body.is_empty() || body.starts_with('/') {
        return false;
    }
    use crate::remote::inbox;
    let waiting: Vec<i64> = inbox::pending_questions()
        .into_keys()
        .filter(|mid| inbox::question_accepts_text(*mid))
        .collect();
    if waiting.is_empty() {
        return false;
    }
    let target = match meta.get("reply_to").filter(|v| !v.is_null()) {
        Some(v) => match v.as_i64() {
            Some(t) if waiting.contains(&t) => t,
            _ => return false,
        },
        None if waiting.len() == 1 => waiting[0],
        None => {
            crate::remote::notify::notify(
                EventKind::Info,
                &tr("Several questions are waiting"),
                crate::remote::notify::NotifyOpts {
                    body: tr("Reply to the question you are answering, then send it again."),
                    sensitivity: crate::remote::events::Sensitivity::Public,
                    blocking: true,
                    ..Default::default()
                },
            )
            .await;
            return true;
        }
    };
    inbox::deliver_answer(target, &format!("t:{body}"));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_trim_on_button_only() {
        let long = "x".repeat(60);
        let s = short(&long);
        assert_eq!(s.chars().count(), MAX_LABEL);
        assert!(s.ends_with('…'));
        assert_eq!(short("short"), "short");
    }
}
