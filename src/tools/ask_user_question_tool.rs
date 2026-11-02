//! AskUserQuestion Tool for CodeMason
//! Based on src/tools/AskUserQuestionTool/ from Claude Code
//! Equivalent to codemason/tools/ask_user_question_tool.py.
//!
//! WHAT (schema, description, prompt, semantics) is ported verbatim from the
//! original. The HOW (the multiple-choice UI) is implemented on codemason's
//! terminal interface instead of the original's Ink/React component — the
//! terminal interface is part of the preserved local substrate.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashSet;
use std::io::IsTerminal;

use tokio_util::sync::CancellationToken;

use crate::remote::questions as remote_questions;
use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::arrow_picker::{arrow_select, ArrowSelectOptions};
use crate::utils::formatting::{colorize, colors, safe_print};
use crate::utils::i18n::{tr, trf, FmtArg};

pub const ASK_USER_QUESTION_TOOL_NAME: &str = "AskUserQuestion";
#[allow(dead_code)]
pub const ASK_USER_QUESTION_TOOL_CHIP_WIDTH: usize = 12;

pub const DESCRIPTION: &str = "Asks the user multiple choice questions to gather information, clarify ambiguity, understand preferences, make decisions or offer them choices.";

/// ASK_USER_QUESTION_TOOL_PROMPT + PREVIEW_FEATURE_PROMPT_MARKDOWN, exactly as
/// the Python prompt() concatenates them (dumped verbatim from the running
/// Python module).
const PROMPT_WITH_PREVIEW: &str = include_str!("prompts/ask_user_question_prompt.txt");

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(ASK_USER_QUESTION_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

fn q_str<'a>(q: &'a Value, name: &str) -> &'a str {
    q.get(name).and_then(|v| v.as_str()).unwrap_or("")
}

fn q_options(q: &Value) -> Vec<Value> {
    q.get("options").and_then(|o| o.as_array()).cloned().unwrap_or_default()
}

/// Best-effort: restore cooked mode + echo so line reads are visible even when
/// something left the terminal in raw/cbreak mode. Substrate divergence: the
/// Python flipped termios ECHO|ICANON and restored the prior flags; crossterm
/// models the same thing as raw-mode on/off, and the plain path always wants
/// cooked mode, so a best-effort disable (idempotent) carries the intent.
fn enable_terminal_echo() {
    crate::utils::tty_compat::disable_raw_mode();
}

/// Read one line from stdin off the async runtime (the Python's blocking
/// input()). None on EOF/error — the callers treat it as "no answer".
async fn read_line_blocking() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        let mut raw = String::new();
        match std::io::stdin().read_line(&mut raw) {
            Ok(0) => None, // EOF
            Ok(_) => Some(raw.trim_end_matches(['\r', '\n']).to_string()),
            Err(_) => None,
        }
    })
    .await
    .ok()
    .flatten()
}

/// How long to wait before re-asking, and before deciding for ourselves.
///
/// A question with nobody in front of it used to wait forever, which in the mode
/// this is mostly used in (-dsp, started and left for hours) means the whole job
/// is parked on a prompt no one will see. Waiting is right; waiting forever is
/// not. So: ask, re-ask after 10 minutes, re-ask after 5 more, and after 3 more
/// take the recommended option and keep going.
///
/// Each step re-sends to the phone as well, so someone who picks their phone up
/// late still finds the question waiting rather than having missed it once.
use crate::utils::escalation::ASK_LADDER_SECONDS as ASK_ESCALATION_SECONDS;

/// The option to take when nobody answers: the one marked recommended, else the
/// first.
///
/// This is not a guess about which choice is "safest" in general — it is the
/// convention the tool's own description asks callers to follow ("If you
/// recommend a specific option, make that the first option in the list and add
/// (Recommended) at the end of the label"). Falling back to the first option
/// honours the same convention when the marker is absent, since the caller is
/// told to put its recommendation first either way.
fn recommended_option_index(labels: &[String]) -> usize {
    labels
        .iter()
        .position(|l| {
            let low = l.to_lowercase();
            low.contains("(recommended)") || l.contains("(추천)")
        })
        .unwrap_or(0)
}

/// Total minutes the ladder waits before deciding, for the message.
fn ask_escalation_total_minutes() -> u64 {
    crate::utils::escalation::ask_total_minutes()
}

/// What one question's local/remote race resolved to.
enum Sel {
    /// The phone answered; the answer is already recorded.
    Remote,
    /// The local picker returned (selected indices, or None = cancelled).
    Picked(Option<Vec<usize>>),
}

/// Run the local picker and (when configured) the phone at once (the Python
/// _race_local_and_remote).
///
/// Returns the local selection, or `Sel::Remote` once the phone has answered —
/// in which case `answers` already holds it and `remote_marks` notes which
/// question it was. A remote timeout is not an answer: it drops back to
/// waiting on the picker exactly as if the phone had never been asked.
#[allow(clippy::too_many_arguments)]
async fn race_local_and_remote(
    local: std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<usize>>> + Send + '_>>,
    qtext: &str,
    header: &str,
    labels: &[String],
    descs: &[String],
    remote_ok: bool,
    answers: &mut Vec<(String, String)>,
    closer: &CancellationToken,
    remote_marks: &mut HashSet<String>,
) -> Sel {
    let mut local = local;
    let cwd = std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).ok();

    // Ask, re-ask, re-ask, then decide. Each pass re-sends to the phone, so a
    // late reader still finds the question rather than having missed it once.
    for (step, wait) in ASK_ESCALATION_SECONDS.iter().enumerate() {
        let one = race_once(
            &mut local, qtext, header, labels, descs, remote_ok,
            cwd.as_deref(), answers, closer, remote_marks,
        );
        match tokio::time::timeout(std::time::Duration::from_secs(*wait), one).await {
            Ok(sel) => return sel,
            Err(_) => {
                if step + 1 < ASK_ESCALATION_SECONDS.len() {
                    crate::cmprintln!(
                        "  ⎿  {}",
                        trf(
                            "Still waiting on an answer — asking again ({left} min left before choosing).",
                            &[("left", FmtArg::Int(
                                crate::utils::escalation::ask_minutes_left_after(step) as i64
                            ))]
                        )
                    );
                }
            }
        }
    }

    // Nobody answered. Take the recommended option rather than parking the run
    // on a prompt no one is going to see.
    let idx = recommended_option_index(labels);
    let label = labels.get(idx).cloned().unwrap_or_default();
    closer.cancel();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(3), &mut local).await;
    crate::cmprintln!(
        "  ⎿  {}",
        trf(
            "No answer after {mins} min — going with the recommended option: {label}",
            &[
                ("mins", FmtArg::Int(ask_escalation_total_minutes() as i64)),
                ("label", FmtArg::Str(label.clone())),
            ]
        )
    );
    Sel::Picked(Some(vec![idx]))
}

/// One pass of the local/remote race, with no time limit of its own — the
/// caller bounds it. Returning here means somebody actually answered.
#[allow(clippy::too_many_arguments)]
async fn race_once(
    local: &mut std::pin::Pin<Box<dyn std::future::Future<Output = Option<Vec<usize>>> + Send + '_>>,
    qtext: &str,
    header: &str,
    labels: &[String],
    descs: &[String],
    remote_ok: bool,
    cwd: Option<&str>,
    answers: &mut Vec<(String, String)>,
    closer: &CancellationToken,
    remote_marks: &mut HashSet<String>,
) -> Sel {
    if !remote_ok {
        return Sel::Picked(local.await);
    }
    let remote = remote_questions::ask(qtext, header, labels, Some(descs), cwd, None);
    tokio::select! {
        sel = &mut *local => Sel::Picked(sel),
        chosen = remote => {
            let Some(chosen) = chosen else {
                // phone gave up; the picker still owns it
                return Sel::Picked(local.await);
            };
            answers.push((qtext.to_string(), chosen));
            // Close the picker through its own exit path rather than dropping
            // the future mid-render — a dropped picker never restores the
            // screen, and the next line printed ends up inside the prompt
            // instead of above it. Then say what happened: the picker
            // vanishing on its own is not an explanation.
            closer.cancel();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(3), &mut *local).await;
            remote_marks.insert(qtext.to_string());
            Sel::Remote
        }
    }
}

pub struct AskUserQuestionTool;

impl AskUserQuestionTool {
    /// Arrow-key (↑/↓/enter, space to toggle for multi) picker path for the
    /// interactive REPL. The Python pauses the persistent PromptSession for
    /// the duration so the picker owns the terminal, then restores it.
    ///
    /// WIRING NOTE (cluster CLI): the InputRouter trait does not yet expose
    /// pause_prompt/resume_prompt; when the persistent prompt engine lands,
    /// its owner must park the engine around this call (one raw-mode reader
    /// per terminal at a time), exactly as resume_picker documents.
    ///
    /// With a phone configured the same question also goes there and the
    /// first answer wins — the same race the permission prompt runs, for the
    /// same reason: a turn stopped on a question is stopped whether or not
    /// anyone is sitting at the terminal, and neither side should own the
    /// decision.
    async fn call_with_arrow_picker(
        questions: &[Value],
        router: Option<std::sync::Arc<crate::cli_repl::ReplInputRouter>>,
    ) -> (Vec<(String, String)>, Vec<String>) {
        let mut answers: Vec<(String, String)> = Vec::new();
        let mut remote_marks: HashSet<String> = HashSet::new();
        let other = tr("Other (type your own answer)");
        let remote_ok = remote_questions::is_available();

        // Park the resident prompt for the whole question loop (Python
        // ask_user_question_tool.py:327/371 — pause_prompt / finally:
        // resume_prompt): the picker must be the terminal's only input owner.
        // This was MISSING from the port; the picker used to fight the engine
        // for keystrokes.
        let _pause = match &router {
            Some(r) => Some(r.pause_prompt_scoped().await),
            None => None,
        };

        for q in questions {
            let qtext = q_str(q, "question").to_string();
            let header = q_str(q, "header").to_string();
            let options = q_options(q);
            let multi = q.get("multiSelect").and_then(|m| m.as_bool()).unwrap_or(false);

            let labels: Vec<String> =
                options.iter().map(|o| q_str(o, "label").to_string()).collect();
            let descs: Vec<String> =
                options.iter().map(|o| q_str(o, "description").to_string()).collect();
            let previews: Vec<Option<String>> = options
                .iter()
                .map(|o| o.get("preview").and_then(|p| p.as_str()).map(|s| s.to_string()))
                .collect();

            let mut disp = labels.clone();
            disp.push(other.clone());
            let mut disp_descs: Vec<Option<String>> =
                descs.iter().map(|d| Some(d.clone())).collect();
            disp_descs.push(None);
            let mut disp_prev = previews.clone();
            disp_prev.push(None);

            let closer = CancellationToken::new();
            let picker = Box::pin(arrow_select(
                &qtext,
                &disp,
                ArrowSelectOptions {
                    descriptions: Some(&disp_descs),
                    previews: Some(&disp_prev),
                    multi,
                    header: if header.is_empty() { None } else { Some(&header) },
                    close_when: Some(closer.clone()),
                },
            ));
            let sel = race_local_and_remote(
                picker,
                &qtext,
                &header,
                &labels,
                &descs,
                remote_ok,
                &mut answers,
                &closer,
                &mut remote_marks,
            )
            .await;
            let sel = match sel {
                Sel::Remote => continue,
                Sel::Picked(sel) => sel,
            };
            let Some(sel) = sel else {
                answers.push((qtext, "(no answer)".to_string()));
                continue;
            };
            if sel.is_empty() {
                answers.push((qtext, "(no answer)".to_string()));
                continue;
            }

            let mut chosen: Vec<String> = Vec::new();
            for idx in sel {
                if idx == labels.len() {
                    // the appended "Other" entry. The picker has exited, so
                    // the terminal is back in cooked mode and a plain line
                    // read is safe here.
                    crate::cmprint!("  {}", tr("Your answer: "));
                    let custom = read_line_blocking().await.unwrap_or_default();
                    let custom = custom.trim().to_string();
                    if !custom.is_empty() {
                        chosen.push(custom);
                    }
                } else if idx < labels.len() {
                    chosen.push(labels[idx].clone());
                }
            }
            let answer =
                if chosen.is_empty() { "(no answer)".to_string() } else { chosen.join(", ") };
            answers.push((qtext, answer));
        }

        let mut marks: Vec<String> = remote_marks.into_iter().collect();
        marks.sort();
        (answers, marks)
    }

    /// Numbered-prompt path (plain / headless mode; the Python's default
    /// call() body when no input_router is attached).
    async fn call_with_numbers(
        questions: &[Value],
        _context: &ToolUseContext,
    ) -> Vec<(String, String)> {
        let mut answers: Vec<(String, String)> = Vec::new();

        enable_terminal_echo();
        for q in questions {
            let qtext = q_str(q, "question").to_string();
            let header = q_str(q, "header").to_string();
            let options = q_options(q);
            let multi = q.get("multiSelect").and_then(|m| m.as_bool()).unwrap_or(false);

            crate::cmprintln!();
            if !header.is_empty() {
                crate::cmprintln!("{}", colorize(&format!("[{header}]"), colors::CYAN));
            }
            safe_print(&colorize(&qtext, colors::BOLD));

            let mut opt_labels: Vec<String> = Vec::new();
            for (i, o) in options.iter().enumerate() {
                let label = q_str(o, "label");
                let desc = q_str(o, "description");
                let preview = o.get("preview").and_then(|p| p.as_str());
                opt_labels.push(label.to_string());
                safe_print(&format!("  {}. {label}", colorize(&(i + 1).to_string(), colors::BOLD)));
                if !desc.is_empty() {
                    safe_print(&format!("     {}", colorize(desc, colors::DIM)));
                }
                if let Some(preview) = preview {
                    for line in preview.split('\n') {
                        crate::cmprintln!("       {}", colorize(line, colors::DIM));
                    }
                }
            }
            let other_idx = options.len() + 1;
            crate::cmprintln!(
                "  {}. {}",
                colorize(&other_idx.to_string(), colors::BOLD),
                tr("Other (type your own answer)")
            );

            // Headless (-p): there is no terminal to answer with, so the phone
            // is the only way this question gets answered at all. Asked before
            // the prompt below rather than racing it, because the local half
            // here is a blocking read that cannot be cancelled.
            if !std::io::stdin().is_terminal() {
                let descs: Vec<String> =
                    options.iter().map(|o| q_str(o, "description").to_string()).collect();
                let cwd =
                    std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).ok();
                let remote = remote_questions::ask(
                    &qtext,
                    &header,
                    &opt_labels,
                    Some(&descs),
                    cwd.as_deref(),
                    None,
                )
                .await;
                if let Some(remote) = remote {
                    crate::cmprintln!("  {remote}  {}", tr("(remote)"));
                    answers.push((qtext, remote));
                    continue;
                }
            }

            let hint =
                if multi { tr("numbers comma-separated") } else { tr("a number") };
            loop {
                crate::cmprint!(
                    "{}",
                    colorize(
                        &format!("  {}", trf("Select {hint}: ", &[("hint", FmtArg::Str(hint.clone()))])),
                        colors::CYAN
                    )
                );
                let Some(raw) = read_line_blocking().await else {
                    // Nothing is going to arrive on stdin — EOF is not a slow
                    // typist. Looping on it spun a headless run at 100% CPU
                    // forever; the question is simply unanswerable here.
                    answers.push((qtext.clone(), "(no answer)".to_string()));
                    break;
                };
                let raw = raw.trim().to_string();
                if raw.is_empty() {
                    continue;
                }
                let picks: Vec<String> = if multi {
                    raw.split(',').map(|p| p.trim().to_string()).collect()
                } else {
                    vec![raw]
                };
                let mut chosen: Vec<String> = Vec::new();
                let mut valid = true;
                for p in picks {
                    let Ok(n) = p.parse::<usize>() else {
                        valid = false;
                        break;
                    };
                    if n == other_idx {
                        crate::cmprint!(
                            "{}",
                            colorize(&format!("  {}", tr("Your answer: ")), colors::CYAN)
                        );
                        let custom =
                            read_line_blocking().await.unwrap_or_default().trim().to_string();
                        if !custom.is_empty() {
                            chosen.push(custom);
                        }
                    } else if n >= 1 && n <= opt_labels.len() {
                        chosen.push(opt_labels[n - 1].clone());
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid && !chosen.is_empty() {
                    answers.push((qtext.clone(), chosen.join(", ")));
                    break;
                }
            }
        }
        answers
    }
}

fn answers_to_map(answers: &[(String, String)]) -> Value {
    let mut map = serde_json::Map::new();
    for (q, a) in answers {
        map.insert(q.clone(), Value::String(a.clone()));
    }
    Value::Object(map)
}

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &'static str {
        ASK_USER_QUESTION_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    fn render_tool_use_message(&self, _input: &Value, _verbose: bool) -> String {
        // AskUserQuestionTool.tsx draws its own question/options UI in call();
        // echoing the questions array here would print it twice.
        String::new()
    }

    /// What the screen shows once the question is answered.
    ///
    /// The model-facing result is a sentence written for the model — "User has
    /// answered your questions: … You can now continue with the user's answers
    /// in mind." Printing it verbatim put a paragraph of English instructions
    /// in the middle of a Korean session, restating an answer the user had
    /// just given themselves. They know what they picked; what is worth one
    /// line is which answer landed against which question.
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        if !data.is_object() {
            return Some(String::new());
        }
        let answers = data.get("answers").and_then(|a| a.as_object());
        let Some(answers) = answers else {
            return Some(String::new());
        };
        if answers.is_empty() {
            return Some(String::new());
        }
        let remote: HashSet<&str> = data
            .get("remote_answers")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Say where the answer came from only when it was not this terminal —
        // you already know what you typed here.
        let mark = |q: &str| {
            if remote.contains(q) {
                format!("  {}", tr("(remote)"))
            } else {
                String::new()
            }
        };

        if answers.len() == 1 {
            let (q, a) = answers.iter().next()?;
            let a = a.as_str().unwrap_or("");
            return Some(format!("-> {a}{}", mark(q)));
        }
        Some(
            answers
                .iter()
                .map(|(q, a)| format!("-> {q}: {}{}", a.as_str().unwrap_or(""), mark(q)))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    async fn description(&self, _input: &Value) -> String {
        DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        PROMPT_WITH_PREVIEW.to_string()
    }

    /// The Python's `DESCRIPTION` class attribute — deliberately much
    /// shorter than this tool's system prompt (which the wire default would
    /// otherwise send). Oracle-verified byte-for-byte against
    /// AskUserQuestionTool.DESCRIPTION.
    async fn wire_description(&self) -> Option<String> {
        Some("Asks the user multiple choice questions to gather information, clarify ambiguity, understand preferences, make decisions or offer them choices.".to_string())
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        // The tool IS the user interaction; no separate approval prompt.
        false
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        // A friendly label for the tool-use header (the original's Ink UI
        // labels the question component; the terminal substrate shows this
        // instead of an empty `[]`). The interactive question body is rendered
        // by call().
        tr("Question")
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let questions = input.get("questions").and_then(|q| q.as_array()).cloned().unwrap_or_default();
        if questions.is_empty() {
            return ValidationResult::failure("At least one question is required");
        }
        // Uniqueness: question texts unique; option labels unique within a
        // question.
        let qtexts: Vec<&str> = questions.iter().map(|q| q_str(q, "question")).collect();
        let unique: HashSet<&str> = qtexts.iter().copied().collect();
        if qtexts.len() != unique.len() {
            return ValidationResult::failure("Question texts must be unique");
        }
        for q in &questions {
            let opts = q_options(q);
            let labels: Vec<&str> = opts.iter().map(|o| q_str(o, "label")).collect();
            let unique: HashSet<&str> = labels.iter().copied().collect();
            if labels.len() != unique.len() {
                return ValidationResult::failure(
                    "Option labels must be unique within each question",
                );
            }
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let questions =
                input.get("questions").and_then(|q| q.as_array()).cloned().unwrap_or_default();

            // Interactive REPL: use the arrow-key picker (↑/↓/enter) instead
            // of a raw line read that fights the persistent prompt for stdin.
            // input_router is None in -p / --plain mode, where we fall through
            // to the numbered prompt below.
            let (answers, remote_marks) = if context.input_router.is_some() {
                let router = crate::cli_repl::router_of(&context);
                AskUserQuestionTool::call_with_arrow_picker(&questions, router).await
            } else {
                (AskUserQuestionTool::call_with_numbers(&questions, &context).await, Vec::new())
            };

            // Result for the assistant — verbatim format of
            // mapToolResultToToolResultBlockParam (AskUserQuestionTool.tsx:
            // 228-241): `"q"="a", "q2"="a2"`. (Annotations preview/notes are an
            // SDK-side concept; the terminal UI here collects only the chosen
            // answer, so there are none to append.)
            let answers_text = answers
                .iter()
                .map(|(q, a)| format!("\"{q}\"=\"{a}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let result_text = format!(
                "User has answered your questions: {answers_text}. \
                 You can now continue with the user's answers in mind."
            );

            let mut data = serde_json::Map::new();
            data.insert("questions".into(), Value::Array(questions));
            data.insert("answers".into(), answers_to_map(&answers));
            if !remote_marks.is_empty() || context.input_router.is_some() {
                data.insert(
                    "remote_answers".into(),
                    Value::Array(remote_marks.into_iter().map(Value::String).collect()),
                );
            }
            yield Ok(ToolResult::Result {
                result_for_assistant: result_text,
                data: Some(Value::Object(data)),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With nobody in front of the terminal the question used to wait forever,
    /// parking a whole unattended run on a prompt no one would see. The ladder
    /// has to end in a decision, and the decision has to be the one the caller
    /// told us to prefer.
    #[test]
    fn the_recommended_option_is_what_gets_chosen() {
        let labels = vec![
            "Keep the current behaviour".to_string(),
            "Rewrite it (Recommended)".to_string(),
        ];
        assert_eq!(recommended_option_index(&labels), 1, "the marked option wins");

        // Case must not matter — the marker is written by a model, not a parser.
        let shouty = vec!["a".to_string(), "b (RECOMMENDED)".to_string()];
        assert_eq!(recommended_option_index(&shouty), 1);

        // Korean marker, since the questions are asked in the user's language.
        let ko = vec!["그대로 둔다".to_string(), "고친다 (추천)".to_string()];
        assert_eq!(recommended_option_index(&ko), 1);

        // No marker: the convention is that the recommendation comes first.
        let plain = vec!["first".to_string(), "second".to_string()];
        assert_eq!(recommended_option_index(&plain), 0);

        // Never panics on an empty list, and never picks out of range.
        assert_eq!(recommended_option_index(&[]), 0);
    }

    /// The schedule the user specified: 10 minutes, then 5, then 3, then decide.
    /// The message quotes the total, so it must stay derived from the schedule
    /// rather than hardcoded — a changed constant with a stale message is how
    /// a prompt starts lying about what it is about to do.
    #[test]
    fn the_escalation_schedule_is_what_was_asked_for() {
        assert_eq!(ASK_ESCALATION_SECONDS, [600, 300, 180]);
        assert_eq!(ask_escalation_total_minutes(), 18);
        // Each step must be a real wait; a zero would decide instantly.
        assert!(ASK_ESCALATION_SECONDS.iter().all(|s| *s > 0));
    }

    #[tokio::test]
    async fn uniqueness_validation() {
        let ctx = ToolUseContext::default();
        let dup_q = serde_json::json!({"questions": [
            {"question": "Q?", "header": "H", "options": [
                {"label": "a", "description": "d"}, {"label": "b", "description": "d"}]},
            {"question": "Q?", "header": "H2", "options": [
                {"label": "a", "description": "d"}, {"label": "b", "description": "d"}]},
        ]});
        let v = AskUserQuestionTool.validate_input(&dup_q, &ctx).await;
        assert_eq!(v.message, "Question texts must be unique");

        let dup_label = serde_json::json!({"questions": [
            {"question": "Q?", "header": "H", "options": [
                {"label": "same", "description": "d"}, {"label": "same", "description": "d"}]},
        ]});
        let v2 = AskUserQuestionTool.validate_input(&dup_label, &ctx).await;
        assert_eq!(v2.message, "Option labels must be unique within each question");

        let empty = serde_json::json!({"questions": []});
        let v3 = AskUserQuestionTool.validate_input(&empty, &ctx).await;
        assert_eq!(v3.message, "At least one question is required");
    }

    #[test]
    fn result_render_single_and_multi() {
        let single = serde_json::json!({"answers": {"Q?": "yes"}, "remote_answers": []});
        assert_eq!(
            AskUserQuestionTool.render_tool_result_message(&single, false),
            Some("-> yes".to_string())
        );
        let multi = serde_json::json!({
            "answers": {"Q1?": "a", "Q2?": "b"},
            "remote_answers": ["Q2?"],
        });
        let out = AskUserQuestionTool.render_tool_result_message(&multi, false).unwrap();
        assert!(out.contains("-> Q1?: a"));
        assert!(out.contains("-> Q2?: b  "));
    }

    #[test]
    fn answers_text_format() {
        let answers = vec![("Q?".to_string(), "a, custom".to_string())];
        let text = answers
            .iter()
            .map(|(q, a)| format!("\"{q}\"=\"{a}\""))
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(text, "\"Q?\"=\"a, custom\"");
    }
}
