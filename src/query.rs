//! Query Loop for CodeMason
//! Equivalent to codemason/query.py (src/query.ts).
//!
//! The recursive agent loop: query the model, yield the assistant message,
//! run any tool calls (concurrently when all are concurrency-safe, else
//! serially), pair each tool_use with its tool_result, auto-compact, drain
//! mid-turn notifications/steering, and recurse. All the guards from the
//! Python are ported verbatim — max-output-token recovery, goal gates, Stop
//! hooks, the repeated-tool-call cap and the (args,result)-repeat brake.

use std::collections::{BTreeMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;

use futures::stream::{Stream, StreamExt};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::services::backend::query_model;
use crate::services::compaction::auto_compact_if_needed;
use crate::services::openai_common::{QueryModelParams, QueryOptions};
use crate::tools::base::{
    CanUseToolFn, PermissionBehavior, Tool, ToolError, ToolResult,
};
use crate::types::messages::{
    ApiMessage, BetaUsage, ContentBlock, ContentBlockParam, MessageContent, TextBlock,
    ToolResultContent,
    ToolUseBlock,
};
use crate::types::query::{AssistantMessage, Message, ToolUseContext};
use crate::utils::generators::{abortable, all_streams};
use crate::utils::log::debug_log;
use crate::utils::messages::{
    create_user_message_blocks, create_user_message_text, INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
};

pub const MAX_TOOL_USE_CONCURRENCY: usize = 10;

/// query.ts:164. When a turn ends because the model hit the output token
/// limit (finish_reason=length -> stop_reason=max_tokens) rather than stopping
/// naturally, the response was cut mid-thought. We inject a resume nudge and
/// re-query, up to this many times per stall.
const MAX_OUTPUT_TOKENS_RECOVERY_LIMIT: i64 = 3;

/// Cap on how many times, within ONE user turn, a failing goal gate may push
/// the turn to continue.
const GOAL_GATE_RECOVERY_LIMIT: i64 = 5;

const GATE_CONTINUATION_TEMPLATE: &str = "[Continuing toward your standing goal — a completion gate failed]\nGoal: {goal}\n\nThis goal is complete only when its gate commands pass, and {n} just failed (auto-check {attempt}/{limit}):\n{failures}\nFix the underlying problem so the gate passes, then it will be re-checked. Do not declare the goal done while any gate fails. If a gate itself is wrong or genuinely cannot pass, say so clearly and stop.";

fn format_gate_failures(goal: &str, failed: &[(String, i32, String)], attempt: i64, limit: i64) -> String {
    let mut blocks = Vec::new();
    for (cmd, code, tail) in failed {
        let tail = tail.trim();
        let mut body = format!("  $ {cmd}\n  exit code: {code}");
        if !tail.is_empty() {
            body += &format!("\n  output (tail):\n  ```\n{tail}\n  ```");
        }
        blocks.push(body);
    }
    GATE_CONTINUATION_TEMPLATE
        .replace("{goal}", goal)
        .replace("{n}", &failed.len().to_string())
        .replace("{attempt}", &attempt.to_string())
        .replace("{limit}", &limit.to_string())
        .replace("{failures}", &blocks.join("\n"))
}

/// query.ts:1209 — the verbatim recovery prompt (isMeta user message).
/// The cut turn's reasoning is now carried into this request — recovery
/// requests hold clear_thinking OFF, see openai_common. So the instruction is
/// to CONTINUE from that thinking, not to start over without it.
///
/// It used to say "no recap of what you were doing", written when the thinking
/// was in fact gone: the recovery injects a user message, the server template
/// treats everything before a user message as a prior turn, and clear_thinking
/// dropped it. Traced 2026-09-13 — a turn spent its whole 65,536-token budget
/// on the reasoning channel, and the next request's prompt FELL by 16,405
/// tokens instead of rising. The model was told to pick up mid-thought with the
/// thought deleted, and redid the same work for hours.
const MAX_OUTPUT_RECOVERY_PROMPT: &str = "Output token limit hit mid-thought. Your reasoning from before the cut is still in this conversation — read it and continue from where it left off rather than starting the analysis again. No apology. Write your conclusions down as you reach them instead of holding them in reasoning, so the next cut cannot cost them. Break the remaining work into smaller pieces.";

/// Layer-2 repetition recovery (Rust+Python enhancement, not in upstream):
/// vLLM's detector cuts a degenerate n-gram loop mid-generation. Rather than
/// end the turn and hand a "[Generation stopped]" note to the USER, the loop
/// gets up to this many chances to self-correct — each time the degenerate
/// emission's tool calls are discarded and a nudge is injected so the model
/// realizes it looped and changes approach WITHIN the same turn. Only after
/// the budget is spent does the turn end with a visible note.
/// Attempts before the turn is allowed to end. Was 2, which under -dsp meant
/// an unattended run stopped with work half-done — the exact failure that mode
/// exists to avoid. The budget is now spent on ESCALATING instructions rather
/// than repeating one, because repeating the same nudge to a model that is
/// already stuck is itself a loop.
///
/// READ THE NUMBER CAREFULLY: this is the limit, not the attempt count. The
/// three call sites test `count <= LIMIT`, so the model gets LIMIT + 1 = 7
/// attempts, and the doc above ("up to this many chances") read as 6 for long
/// enough to be worth spelling out. The extra attempt is deliberate — it is the
/// one that tells the model to ABANDON this line of work — and
/// `the_attempt_count_is_one_more_than_the_limit` pins the relationship so the
/// two cannot drift apart again.
///
/// WAS 6, giving 8 nudge attempts and a 9th for the reduction. Cut to 0 — ONE
/// nudge attempt, then the reduction — on 176 measured collapse runs from
/// turns.log, 2026-09-22 to 2026-10-02 (2,483 requests):
///
/// ```text
///  attempt   runs entering   escaped by a nudge
///        2             176          167  (94.9%)
///        3               7            0
///        4               7            2
///        5               4            0
///        6               4            1   <- the most that ever worked
///        7               3            0
///        8               3            0
/// ```
///
/// One plain retry cures 94.9% of collapses. Past that the ladder is close to
/// worthless: attempts 7 and 8 have never once saved a turn, and the two
/// collapses that reached them (2026-10-02 16:03 and 16:09, at 289,350 and
/// 290,984 prompt tokens) were cured by the reduction on its first try, both
/// times. The 16-failure run on 2026-10-01 was the `/ab` abliteration bug, not
/// length, and is excluded.
///
/// So the nudges were costing 6 requests and ~90 seconds per collapse to buy a
/// measured 0, and the step that does work was held back behind them. The
/// budget now buys exactly what the data supports: retry once, then reduce.
///
/// Also measured and NOT acted on: prompt size separates the two populations
/// sharply (every nudge-only recovery was at <= 162,688 tokens; both collapses
/// above 250,000 needed the reduction). Gating on size was rejected because the
/// gate would have to sit in the 163k-289k range, where there are zero
/// observations — a predicted threshold, which is the thing this file keeps
/// learning not to trust. Two failed attempts is a measurement.
const REPETITION_RECOVERY_LIMIT: i64 = 0;

/// One attempt PAST the nudge ladder, spent on cutting the context down instead
/// of on another sentence about it. Taken only when every nudge has failed and
/// the sole remaining alternative is ending the turn.
///
/// Why last and not first. Talking to the model is free; microcompact is not —
/// it replaces the body of every compactable tool result but the newest
/// MICROCOMPACT_KEEP_RECENT, cannot be undone (it is idempotent, so a second
/// pass frees nothing), and on the 2026-10-01 session that was 403 of 408
/// results and 391,114 characters. Spending that on the FIRST collapse would
/// destroy the history in the ~50% of cases a plain retry already recovers
/// (measured: recovery calls "came back with think=26 and still collapsed, 50%
/// of the time"), and in every collapse whose cause was never length at all —
/// a poisoned tool result, a pathological repetition in the input.
///
/// Here the cost is close to zero, because the measured alternative is
/// REPETITION_GIVEUP_NOTE and a half-finished unattended run. Nudges first,
/// reduction only when they are exhausted.
const REPETITION_LAST_RESORT_ATTEMPT: i64 = REPETITION_RECOVERY_LIMIT + 1;
const REPETITION_RECOVERY_PROMPT: &str = "You started repeating the same output or tool call and the server cut the generation. Stop — the current approach has stopped making progress. Do something DIFFERENT this step: a different tool, a different command or input, or state your conclusion from what you already know. Do not repeat the last action.";

/// Later stages of the same nudge. Repeating stage 1 at a model that ignored
/// stage 1 is the same mistake the model is making, so each stage asks for
/// something categorically different: first vary the action, then abandon the
/// goal of this step, then abandon the line of work entirely and carry on with
/// the rest of the task. The last one is what keeps an unattended run moving:
/// the turn does not end, the work does.
const REPETITION_ESCALATION_PROMPTS: [&str; 3] = [
    "You have been cut short repeatedly now. Changing the command is not enough — the GOAL of this step is what is not working. Do not attempt it again in another form. Write down plainly what you have established so far and what remains unknown, then pick a different part of the task and continue.",
    "This line of work has failed several times and is not going to succeed as approached. Abandon it now. State your conclusion from what you already know — \"this could not be verified\" is a valid and useful conclusion — and move on to the next item of work. Do not return to this one.",
    "Stop this line of work completely. Do not run another tool against it. Summarise what you found, mark it unresolved, and continue with whatever else the task requires. If nothing else remains, say so and finish.",
];

/// The nudge for `attempt` (0-based), escalating and then holding at the last.
///
/// The last-resort attempt gets the FINAL rung, not the rung its index would
/// pick. It is now the last attempt of the turn as well as the reduced one, and
/// the final rung is the only one that hands an unattended run somewhere to go
/// ("mark it unresolved, and continue with whatever else the task requires").
/// It is also the text that was measured working: both rescues on 2026-10-02
/// ran on it (by clamping, at attempt 8) and came back with think=503 / 1,276
/// output and a tool call, from eight consecutive 200-character runs of '!'.
///
/// With REPETITION_RECOVERY_LIMIT at 0 the two middle rungs are unreachable.
/// They are kept, not deleted: they are the ladder the budget would climb again
/// if the measurement that cut it ever stops holding, and the constant's doc
/// comment says what to re-measure first.
fn repetition_stage_prompt(attempt: i64) -> &'static str {
    if attempt <= 0 {
        return REPETITION_RECOVERY_PROMPT;
    }
    if attempt >= REPETITION_LAST_RESORT_ATTEMPT {
        return REPETITION_ESCALATION_PROMPTS[REPETITION_ESCALATION_PROMPTS.len() - 1];
    }
    let i = (attempt as usize - 1).min(REPETITION_ESCALATION_PROMPTS.len() - 1);
    REPETITION_ESCALATION_PROMPTS[i]
}
const REPETITION_GIVEUP_NOTE: &str = "\n\n[Stopped: the model kept repeating itself across several attempts and could not break out of the loop. The output above is incomplete — this line of work needs a different approach or more information.]";

/// Header for the failing tool results appended to the repetition nudge.
///
/// The nudge alone says "you repeated yourself, do something different", which
/// names the symptom and leaves the model to rediscover the cause. The cause is
/// almost always in the results it just got: a command that errors, or answers
/// something other than what it was asked, returns the model to the identical
/// state every round, so it re-emits the identical plan. (The repeated-call cap
/// note above records the same finding from the other direction.)
///
/// Observed working: given the nudge alone, one recovery turn produced 3611
/// characters of reasoning that opened by naming the loop itself — the model
/// could see it once told to look. Handing it the failures directly starts it
/// from there instead of hoping it gets there.
const REPETITION_FAILURES_HEADER: &str = "\n\nThese are the results the repeated attempts were getting back. The loop is most likely here — an error, or output that does not answer what the call was for, leaves you in the same position each round:\n";

/// How many recent failing tool results to quote, newest last.
const REPETITION_FAILURES_MAX: usize = 3;
/// Characters kept from each quoted result.
const REPETITION_FAILURE_CHARS: usize = 300;

/// Recent tool results that failed or came back empty, oldest first.
///
/// Walks back from the tail over user messages only, so it sees this turn's
/// results and stops at the previous assistant text. Empty output counts: a
/// command that prints nothing answers nothing, and that is the same dead end
/// as an error for a model deciding what to do next.
fn recent_tool_failures(messages: &[Message]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for msg in messages.iter().rev() {
        let Message::User(u) = msg else { continue };
        let MessageContent::Blocks(blocks) = &u.message.content else { continue };
        for b in blocks.iter().rev() {
            let ContentBlockParam::ToolResult { content, is_error, .. } = b else { continue };
            let text = match content {
                ToolResultContent::Text(s) => s.clone(),
                ToolResultContent::Blocks(v) => v
                    .iter()
                    .filter_map(|x| x.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            let trimmed = text.trim();
            if !is_error && !trimmed.is_empty() {
                continue;
            }
            let quoted = if trimmed.is_empty() {
                "(no output)".to_string()
            } else {
                truncate_chars(trimmed, REPETITION_FAILURE_CHARS)
            };
            out.push(quoted);
            if out.len() >= REPETITION_FAILURES_MAX {
                out.reverse();
                return out;
            }
        }
    }
    out.reverse();
    out
}

/// First `max` chars of `s`, marked when it was cut. Char-based: byte slicing
/// splits a CJK codepoint and panics, and these results are frequently Korean.
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head} […]")
}

/// The repetition nudge, with the failing results appended when there are any.
fn repetition_recovery_prompt(messages: &[Message], attempt: i64) -> String {
    let base = repetition_stage_prompt(attempt);
    let failures = recent_tool_failures(messages);
    if failures.is_empty() {
        return base.to_string();
    }
    let body = failures
        .iter()
        .map(|f| format!("- {f}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{base}{REPETITION_FAILURES_HEADER}{body}")
}

/// Layer-2 lost-tool-call recovery (Rust+Python enhancement, not in upstream):
/// the model invoked a tool and NONE of the invocation survived assembly —
/// openai_common surfaces stop_reason "tool_parse_failed" when it had tool
/// calls to build and produced no ToolUse block from any of them. Every
/// upstream cause (a stream cut mid 'arguments', a non-string 'arguments', a
/// malformed fragment) lands on that one signal, so the recovery does not need
/// to know which one it was.
///
/// Re-asking is safe here in a way it is not for a PARTIAL loss: with no
/// ToolUse block, nothing from this turn ran, so there is nothing to run twice.
/// A partial loss deliberately does NOT come through this path — it is logged
/// and the surviving calls proceed, because re-asking would re-execute them.
///
/// The budget exists because the trigger can be deterministic: a server that
/// always serialises 'arguments' as an object would fail identically forever,
/// and an unattended run must stop with a stated reason rather than spin.
const TOOL_PARSE_RECOVERY_LIMIT: i64 = 2;

/// Empty-turn recovery: attempts before the turn is allowed to end.
///
/// One, not the two or six the other layers get. A turn that produced nothing
/// at all is rarer than a lost tool call or a repetition loop, and asking twice
/// buys nothing: if the model had no answer the first time it does not acquire
/// one by being asked the same question again. One nudge is the difference
/// between "it stopped for no reason" and "it was asked and had nothing", which
/// is the distinction worth a request.
const EMPTY_TURN_RECOVERY_LIMIT: i64 = 1;
const EMPTY_TURN_RECOVERY_PROMPT: &str = "Your last turn was empty — no text, no tool call, nothing. Say what you found, and what you are doing next. If the last thing you tried was a dead end, say so plainly and take a different approach; if the work is finished, say that instead. Do not answer with nothing again.";
const EMPTY_TURN_GIVEUP_NOTE: &str = "\n\n[Stopped: the model returned an empty turn — no text, no tool call — and did so again when asked. Nothing was done. Whatever it was last looking at is above.]";
const TOOL_PARSE_RECOVERY_PROMPT: &str = "Your last tool call did not arrive intact — the arguments could not be read, so nothing ran. Nothing from that step took effect. Issue the call again now, as a normal tool call; if it carried a large argument, send a smaller one or split the work into more than one call.";
const TOOL_PARSE_GIVEUP_NOTE: &str = "\n\n[Stopped: the model called a tool but the call could not be read — the arguments did not parse — and it did not come through on retry. Nothing from that step ran. See the error log for the raw failure.]";
/// The max_tokens layer had no give-up note, alone among the four. An exhausted
/// budget therefore ended the turn on a sentence cut mid-word, with nothing
/// anywhere saying the answer was incomplete — under -dsp indistinguishable
/// from a finished one.
const MAX_OUTPUT_GIVEUP_NOTE: &str = "\n\n[Stopped: the model ran into the output limit on every attempt, so the text above is cut off mid-answer and is incomplete. Ask for the rest, or for it in smaller pieces.]";

// --- Repeated-tool-call cap (see the Python module doc for the full rationale).
const REPEATED_TOOL_CALL_NUDGE: usize = 4; // not run; the result slot carries a warning
const REPEATED_TOOL_CALL_LIMIT: usize = 6; // not run; the turn ends

const REPEATED_TOOL_CALL_NUDGE_PROMPT: &str = "This exact call, with these exact arguments, has now been made {n} times in a row, and it was not run again. Repeating it will not return anything different.\nIf the output you needed was missing from the earlier results, the command itself is what has to change — a different query, or one that does not discard the part you are looking for. If the answer is simply not reachable this way, say what is blocked and answer in text instead. Do not issue this call again.";

/// Injected after a mid-turn interjection in the MAIN conversation.
///
/// The steering text used to go in bare. To the model that is indistinguishable
/// from a fresh question, so it answers and ends the turn — and the work it was
/// running is silently abandoned. Observed: the user asked a check-in question
/// during a research run, got a full answer, and the run stopped there with
/// nothing said about it. Under -dsp nobody is watching, so that is a halt.
///
/// The subagent path already had the opposite instruction (AGENT_STEER_WRAPUP,
/// "wrap up now"), because a subagent's caller IS waiting. The main path had
/// neither, so the model was left to guess, and guessing "this is a new turn"
/// is the reasonable reading of a bare message.
///
/// Deliberately not an unconditional "keep going": an interjection is often a
/// redirect or a stop, and overriding that would be worse than halting. The
/// instruction is to resume UNLESS the message itself said otherwise.
const MAIN_STEER_RESUME: &str = "The message above arrived while you were working. The task you were running is NOT finished and no one has cancelled it. Answer the message briefly, then GO BACK to what you were doing and carry on in the same turn — do not stop to summarise, and do not re-plan. The exception is the message itself: if it told you to stop, or to do something else instead, follow it and ignore this note.";

const AGENT_STEER_WRAPUP: &str = "The user has just interjected in the main conversation and is waiting on it. Stop starting new work now: do NOT call any more tools. Immediately write up what you have found SO FAR as your final answer — partial results are fine and expected — then end your turn.";

// === Helper Functions ===

/// Create a SYNTHETIC assistant message (used only for interrupt notices).
/// model MUST be '<synthetic>' (not the real model): these carry zero usage,
/// so if they aren't flagged synthetic the token-usage anchor walk latches
/// onto their 0 and the running token count collapses.
fn create_assistant_message_obj(text: &str) -> AssistantMessage {
    AssistantMessage {
        uuid: uuid::Uuid::new_v4().to_string(),
        message: ApiMessage {
            id: uuid::Uuid::new_v4().to_string(),
            message_type: "message".into(),
            role: "assistant".into(),
            content: vec![ContentBlock::text(text)],
            model: "<synthetic>".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: BetaUsage::default(),
        },
        cost_usd: 0.0,
        duration_ms: 0,
        is_api_error_message: false,
    }
}

/// Format system prompt with context.
pub fn format_system_prompt_with_context(system_prompt: &[String], context: &BTreeMap<String, String>) -> String {
    let mut parts: Vec<String> = system_prompt.to_vec();
    if !context.is_empty() {
        let mut context_parts = Vec::new();
        for (key, value) in context {
            if !value.is_empty() {
                context_parts.push(format!("<{key}>{value}</{key}>"));
            }
        }
        if !context_parts.is_empty() {
            parts.push(context_parts.join("\n"));
        }
    }
    parts.join("\n\n")
}

/// Canonical signature of an assistant turn's tool calls, or None ("not a
/// tool-calling assistant turn" — neutral, skipped by the caller).
fn tool_batch_signature(msg: &Message) -> Option<String> {
    let a = msg.as_assistant()?;
    let mut calls: Vec<Value> = Vec::new();
    for block in &a.message.content {
        if let ContentBlock::ToolUse(tu) = block {
            let args = canonical_json(&tu.input);
            calls.push(json!([tu.name, args]));
        }
    }
    if calls.is_empty() {
        return None;
    }
    Some(serde_json::to_string(&calls).unwrap_or_default())
}

/// How many assistant turns at the tail of `messages` issued this exact batch.
/// Walks backward, skipping non-tool-calling assistant turns, stops at the
/// first DIFFERING batch. Contiguity at the tail is the only thing that ends a
/// run — a user turn does NOT reset it.
fn count_repeated_tool_batches(messages: &[Message], signature: &str) -> usize {
    let mut run = 0;
    for msg in messages.iter().rev() {
        match tool_batch_signature(msg) {
            None => continue,
            Some(sig) => {
                if sig != signature {
                    break;
                }
                run += 1;
            }
        }
    }
    run
}

/// json.dumps(..., sort_keys=True, ensure_ascii=False) equivalent for tool args.
fn canonical_json(v: &Value) -> String {
    // serde_json with sorted keys: re-serialize through a BTreeMap-backed value.
    fn sort_value(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut sorted: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
                for (k, val) in m {
                    sorted.insert(k.clone(), sort_value(val));
                }
                serde_json::to_value(sorted).unwrap_or(Value::Null)
            }
            Value::Array(a) => Value::Array(a.iter().map(sort_value).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort_value(v)).unwrap_or_default()
}

// === Repeated-tool-call guard state (module-level, like the Python dicts) ===

struct RepeatEntry {
    digest: String,
    count: usize,
}

static TOOL_REPEAT_HISTORY: Lazy<Mutex<BTreeMap<String, RepeatEntry>>> =
    Lazy::new(|| Mutex::new(BTreeMap::new()));

const TOOL_REPEAT_LIMIT: usize = 2;

/// Forget every recorded call. Called when the conversation resets.
pub fn reset_tool_repeat_history() {
    TOOL_REPEAT_HISTORY.lock().clear();
}

/// Fields that label a call for the human but do not change what it does.
const TOOL_KEY_IGNORED_FIELDS: &[&str] = &["description"];

fn tool_call_key(tool_name: &str, input: &Value) -> String {
    let filtered = match input {
        Value::Object(m) => {
            let mut out = serde_json::Map::new();
            for (k, v) in m {
                if !TOOL_KEY_IGNORED_FIELDS.contains(&k.as_str()) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    };
    format!("{tool_name}:{}", canonical_json(&filtered))
}

/// Digest of a tool_result payload, or None if this message isn't one.
fn tool_result_digest(message: &Message) -> Option<String> {
    let u = message.as_user()?;
    for block in u.message.content.as_blocks() {
        if let ContentBlockParam::ToolResult { content, .. } = block {
            use sha1_digest as sha1;
            let payload = content.as_text();
            return Some(sha1(&payload));
        }
    }
    None
}

/// SHA-1 hex of a UTF-8 string (hashlib.sha1(...).hexdigest()).
fn sha1_digest(s: &str) -> String {
    // sha2 crate is in deps, but we need SHA-1; implement a tiny SHA-1 to avoid
    // a new dependency (the digest is only used to detect result-equality).
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = s.as_bytes().to_vec();
    let bitlen = (msg.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// Format an error for display (format_error).
pub fn format_error(error: &ToolError) -> String {
    let mut parts = vec![error.message.clone()];
    if let Some(s) = &error.stderr {
        if !s.is_empty() {
            parts.push(s.clone());
        }
    }
    if let Some(s) = &error.stdout {
        if !s.is_empty() {
            parts.push(s.clone());
        }
    }
    let full: String = parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n");
    if full.chars().count() <= 10000 {
        return full;
    }
    let chars: Vec<char> = full.chars().collect();
    let half = 5000;
    let start: String = chars[..half].iter().collect();
    let end: String = chars[chars.len() - half..].iter().collect();
    format!("{start}\n\n... [{} characters truncated] ...\n\n{end}", chars.len() - 10000)
}

/// Whether a tool call may run concurrently (tool.isConcurrencySafe).
fn is_tool_concurrency_safe(tool_name: &str, tools: &[Arc<dyn Tool>], input: Option<&Value>) -> bool {
    for tool in tools {
        if tool.matches_name(tool_name) {
            return tool.is_concurrency_safe(input);
        }
    }
    // Unknown/unresolved tool: fail closed.
    false
}

type MsgStream = Pin<Box<dyn Stream<Item = Message> + Send>>;

/// The engine-owned recursion parameters, bundled so the recursive call is
/// readable.
pub struct QueryParams {
    pub messages: Vec<Message>,
    pub system_prompt: Vec<String>,
    pub context: BTreeMap<String, String>,
    pub can_use_tool: CanUseToolFn,
    pub tool_use_context: ToolUseContext,
    pub max_output_recovery_count: i64,
    /// Layer-2 repetition recoveries spent this turn (see REPETITION_RECOVERY_LIMIT).
    pub repetition_recovery_count: i64,
    /// Layer-2 lost-tool-call recoveries spent this turn (see TOOL_PARSE_RECOVERY_LIMIT).
    pub tool_parse_recovery_count: i64,
    /// Empty-turn recoveries spent this turn (see EMPTY_TURN_RECOVERY_LIMIT).
    pub empty_turn_recovery_count: i64,
    /// Omit reasoning_effort for THIS request only.
    ///
    /// Set by the repetition branch when it retries a collapsed turn, and
    /// false on every other recursion, so reasoning returns the moment the
    /// model is producing content again. The recovery COUNTER cannot stand in
    /// for this: it stays spent for the rest of the turn, which would leave
    /// the model working blind through every tool call after one hiccup.
    ///
    /// Measured 2026-09-13: eight consecutive repetition retries on one turn
    /// each produced 2000 characters of a single repeated token on the
    /// reasoning channel and ZERO characters of content, ~125s apiece. The
    /// nudge between attempts is a user message and cannot reach the channel
    /// that is stuck, so the final attempt — the one meant to make the model
    /// abandon this line and carry on — produced no text either and the turn
    /// ended. Dropping reasoning_effort removes the channel the collapse
    /// lives on, so the model has to answer in content.
    pub suppress_reasoning: bool,
    pub stop_hook_active: bool,
    pub turn_count: i64,
}

/// Main query loop that handles tool calls recursively (query()).
pub fn query(params: QueryParams) -> MsgStream {
    Box::pin(async_stream::stream! {
        let QueryParams {
            repetition_recovery_count,
            suppress_reasoning,
            tool_parse_recovery_count,
            empty_turn_recovery_count,
            messages,
            system_prompt,
            context,
            can_use_tool,
            tool_use_context,
            max_output_recovery_count,
            stop_hook_active,
            turn_count,
        } = params;

        // Snapshot the input list (the REPL mutates the same list while
        // consuming this generator; our history is threaded via next_messages).
        let messages = messages;

        // maxTurns runaway guard. 0 (or None) = no cap. Catches a loop whose
        // arguments change every turn, which the two repetition guards cannot
        // see because no call repeats exactly.
        let max_turns = crate::utils::config::get_config().max_turns;
        if let Some(mt) = max_turns {
            if mt > 0 && turn_count >= mt {
                yield Message::Assistant(create_assistant_message_obj(&format!(
                    "[Reached the maximum of {mt} turns for this request. Stopping.]"
                )));
                return;
            }
        }

        let full_system_prompt = format_system_prompt_with_context(&system_prompt, &context);

        if tool_use_context.is_aborted() {
            yield Message::Assistant(create_assistant_message_obj(INTERRUPT_MESSAGE));
            return;
        }

        // Query the model.
        let tools = tool_use_context.options.tools.clone();
        // `/ab auto`: the turn's first request is the one deciding whether to
        // accept, and the only one that wants the refusal edit. On here, off as
        // soon as it answers (below) — the two are paired inside this request,
        // so no outcome can leave the switch on. See services::ab.
        let ab_edit = crate::services::ab::begin_request().await;
        let mut assistant_message = query_model(QueryModelParams {
            messages: messages.clone(),
            system_prompt: vec![full_system_prompt],
            tools: tools.clone(),
            signal: Some(tool_use_context.abort_controller.clone()),
            options: QueryOptions {
                dangerously_skip_permissions: tool_use_context.options.dangerously_skip_permissions,
                model: tool_use_context.options.slow_and_capable_model.clone(),
                // A sub-agent (agent_id set by the Agent tool / forked skills)
                // may run on a smaller output budget than the parent — see
                // config.subagent_max_tokens. 0 or unset leaves cfg.max_tokens.
                max_tokens: match (
                    tool_use_context.agent_id.as_ref(),
                    crate::utils::config::get_config().subagent_max_tokens,
                ) {
                    (Some(_), n) if n > 0 => Some(n),
                    _ => None,
                },
                // NOTE (2026-09-28): every recovery recursion now passes
                // suppress_reasoning: false, so both of these are false for
                // conversation requests. Compaction sets its own no_reasoning
                // directly and is unaffected.
                //
                // Why: these two are the only things that differ between a
                // normal request and a recovery request, and BOTH change how
                // the server renders the prompt — keep_thinking through
                // chat_template_kwargs {clear_thinking, drop_thinking}, and
                // no_reasoning through reasoning_effort, which this deployment
                // takes as a template kwarg too (see the note at the
                // reasoning_effort assignment). Either one re-renders the whole
                // conversation, so the token sequence diverges near the HEAD of
                // the prompt and the server's prefix cache misses completely.
                // Measured on this install: recovery requests logged cached=0
                // and "cold prefill: 164109 of 164109 prompt tokens must be
                // recomputed (0.0% served from cache) — about 123s before the
                // first token", twice per repetition event (the flags flip back
                // on the next normal recursion). ~1.3M prefill tokens to answer
                // one nudge, and two minutes of a blank screen that is
                // indistinguishable from a hang.
                //
                // What it cost to stop: the thinking this comment used to
                // preserve. For the repetition path there is nothing left to
                // preserve — the collapsed generation is now cleared where the
                // server raises it, and that reasoning channel WAS the garbage.
                // The low effort did not help either: those recovery calls came
                // back with think=26 and still collapsed, 50% of the time.
                // If a future path genuinely needs the cut turn's reasoning,
                // inline it as text in the appended nudge (the steering path
                // does exactly this, "Appended, so the cached prefix is
                // intact") rather than flipping a flag that re-renders 160k
                // tokens. To revert: set these five recursions back to
                // suppress_reasoning: true.
                no_reasoning: suppress_reasoning,
                keep_thinking: suppress_reasoning,
            },
            on_token_update: tool_use_context.on_token_update.clone(),
            on_content_delta: tool_use_context.on_content_delta.clone(),
            on_thinking_update: tool_use_context.on_thinking_update.clone(),
        })
        .await;
        // Off, whatever came back: a tool call, a sentence, an empty turn, an
        // abort, or a generation the degenerate guard cut. Every recursion below
        // — repetition retries included — runs on the restored weights.
        crate::services::ab::end_request(ab_edit).await;

        if tool_use_context.is_aborted() {
            yield Message::Assistant(create_assistant_message_obj(INTERRUPT_MESSAGE));
            return;
        }

        // Layer-2 repetition recovery. openai_common surfaces stop_reason
        // "repetition" (the raw vLLM signal) instead of baking in end_turn; the
        // POLICY lives here. The degenerate emission's tool calls are discarded
        // (executing a call the model made while looping just extends the loop)
        // and, while the budget lasts, a nudge is injected so the model breaks
        // out on its own — the USER never sees a "stopped" note. stop_reason is
        // normalized to end_turn so the transcript stays standard, and the
        // dangling tool_use blocks are stripped so history stays balanced.
        let is_repetition =
            assistant_message.message.stop_reason.as_deref() == Some("repetition");
        if is_repetition {
            assistant_message
                .message
                .content
                .retain(|b| !matches!(b, ContentBlock::ToolUse(_)));
            assistant_message.message.stop_reason = Some("end_turn".into());
            if repetition_recovery_count > REPETITION_LAST_RESORT_ATTEMPT {
                // Nudges AND the last-resort reduction are both spent: surface
                // the note in the yielded message and end.
                assistant_message.message.content.push(ContentBlock::Text(TextBlock {
                    text: REPETITION_GIVEUP_NOTE.into(),
                }));
            }
        }

        // Layer-2 lost-tool-call recovery, shaped exactly like the repetition
        // path above: openai_common surfaces the raw signal, the POLICY lives
        // here. There are no ToolUse blocks to strip — their absence IS the
        // signal — so this only normalizes stop_reason so the transcript stays
        // standard, and appends the give-up note once the budget is spent. The
        // note matters more here than for repetition: a lost call leaves text
        // that reads like work about to happen, so silence is indistinguishable
        // from success to anyone reading it hours later.
        let is_empty_turn =
            assistant_message.message.stop_reason.as_deref() == Some("empty_turn");
        if is_empty_turn {
            assistant_message.message.stop_reason = Some("end_turn".into());
            if empty_turn_recovery_count > EMPTY_TURN_RECOVERY_LIMIT {
                assistant_message.message.content.push(ContentBlock::Text(TextBlock {
                    text: EMPTY_TURN_GIVEUP_NOTE.into(),
                }));
            }
        }

        let is_tool_parse_failure =
            assistant_message.message.stop_reason.as_deref() == Some("tool_parse_failed");
        if is_tool_parse_failure {
            assistant_message.message.stop_reason = Some("end_turn".into());
            if tool_parse_recovery_count > TOOL_PARSE_RECOVERY_LIMIT {
                assistant_message.message.content.push(ContentBlock::Text(TextBlock {
                    text: TOOL_PARSE_GIVEUP_NOTE.into(),
                }));
            }
        }

        // A stall guard cut the stream (openai_common: stalled stream). Shaped
        // like the three above — openai_common surfaces the raw signal, the
        // POLICY lives here — but the policy is deliberately different: a stall
        // is NOT handed to Layer-2 recovery. Nothing about a server that
        // stopped sending suggests the model is looping, and nudging it would
        // spend the recovery budget on a transport problem. The note naming the
        // cause is already in the message (openai_common wrote it there, where
        // the elapsed seconds are known), so this only normalizes stop_reason
        // to keep the transcript standard.
        //
        // Reaching here at all is the point: a stall used to cancel the turn's
        // shared abort token, so it came out of is_aborted() above as
        // "[Request interrupted by user]" and never got this far.
        if assistant_message.message.stop_reason.as_deref() == Some("stalled") {
            assistant_message.message.stop_reason = Some("end_turn".into());
        }

        // Repeated-tool-call cap, decided BEFORE the yield below. The yielded
        // message is what the REPL appends to history and persists, so ending
        // the turn while it still carries tool_use blocks leaves them with no
        // matching tool_result: convert_to_chat_messages has no pass that drops
        // an unanswered tool_calls message (openai_common only handles the
        // head-of-conversation orphan the other way round), and
        // merge_consecutive_assistants folds the note in while KEEPING
        // tool_calls, so every later request in the session is rejected. Strip
        // them and fold the note into this same message, exactly as the
        // repetition give-up path above does.
        // DIVERGENCE FROM THE PYTHON ORIGINAL, deliberate and kept.
        //
        // This decides the cap BEFORE the yield below, so the tool_use blocks can
        // be stripped and the note folded into the same message. Python cannot:
        // it yields the assistant message ~170 lines earlier, so the blocks are
        // already in history and on disk and the only correct move left there is
        // to answer each one with a tool_result.
        //
        // The two therefore write different transcripts for the same event. Both
        // are self-consistent, each loads fine in the other on --resume, and the
        // model-visible ending text is byte-identical (verified). Aligning them
        // would mean restructuring one hot path for a cosmetic result, risking
        // the orphaned tool_use that poisons a whole session.
        let repeated_cap: Option<(String, usize)> = (!is_repetition)
            .then(|| tool_batch_signature(&Message::Assistant(assistant_message.clone())))
            .flatten()
            .and_then(|sig| {
                let repeat_count = count_repeated_tool_batches(&messages, &sig) + 1;
                (repeat_count >= REPEATED_TOOL_CALL_LIMIT).then(|| {
                    let mut names: Vec<String> = assistant_message
                        .message
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::ToolUse(tu) => Some(tu.name.clone()),
                            _ => None,
                        })
                        .collect();
                    names.sort();
                    names.dedup();
                    (names.join(", "), repeat_count)
                })
            });
        if let Some((names, repeat_count)) = &repeated_cap {
            assistant_message
                .message
                .content
                .retain(|b| !matches!(b, ContentBlock::ToolUse(_)));
            assistant_message.message.stop_reason = Some("end_turn".into());
            assistant_message.message.content.push(ContentBlock::Text(TextBlock {
                text: format!(
                    "[Stopped: the same {names} call was repeated {repeat_count} times in a row, \
                     so this turn was ended without running it again. Tell me how to proceed, \
                     or give me a different angle to try.]"
                ),
            }));
        }

        // Only when nothing will be dispatched: a truncated turn that still
        // carries a readable tool call continues normally and is not "stopped".
        let max_tokens_spent = assistant_message.message.stop_reason.as_deref()
            == Some("max_tokens")
            && max_output_recovery_count >= MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
            && !assistant_message
                .message
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse(_)));
        if max_tokens_spent {
            assistant_message.message.content.push(ContentBlock::Text(TextBlock {
                text: MAX_OUTPUT_GIVEUP_NOTE.into(),
            }));
        }

        yield Message::Assistant(assistant_message.clone());

        if is_repetition {
            // <= , not < : the last attempt is the one that tells the model to
            // ABANDON this line of work and carry on with the rest of the task.
            // Ending the turn instead is the failure an unattended run cannot
            // survive — the session goes quiet with hours of work left, and
            // nobody is watching to restart it.
            //
            // Against REPETITION_LAST_RESORT_ATTEMPT, not the raw limit, so the
            // ladder gets its full run of nudges AND one further attempt on a
            // reduced context before the turn is allowed to end. Ending it is
            // the worst available outcome, so it is the last thing tried, never
            // the second.
            if repetition_recovery_count <= REPETITION_LAST_RESORT_ATTEMPT {
                debug_log("REPETITION_RECOVERY", Some(&json!({
                    "attempt": repetition_recovery_count + 1,
                    "limit": REPETITION_RECOVERY_LIMIT,
                })));
                let mut nudge = create_user_message_text(&repetition_recovery_prompt(&messages, repetition_recovery_count));
                nudge.is_meta = true;
                yield Message::User(nudge.clone());
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                // context_proven_fatal: only on the LAST attempt, and nowhere
                // else in this file.
                //
                // True here because this history has been DEMONSTRATED to kill
                // the decoder and every nudge has already failed on it. False
                // for the attempts before it, so the ~50% of collapses a plain
                // retry recovers keep their tool results; false at the other
                // recovery call sites because nothing has been demonstrated
                // there at all — an empty turn, a truncated tool argument, a
                // repeated call, a Stop hook or a failing gate each say the
                // model did something wrong, not that this context cannot be
                // decoded.
                //
                // That it has to be passed at all is the bug this fixes. The
                // call asked should_auto_compact whether there was room, was
                // told 264,397 of 445,752, and faithfully re-sent the history
                // that had just collapsed — with 46 tokens of advice on the
                // end. Seven times, 94.5 seconds, then the give-up note. The
                // reduction step was never reached: microcompact sits BEHIND
                // that threshold, so 112,000 tokens of dead tool results stayed
                // in the prompt because a number derived from max_model_len said
                // the prompt was fine.
                let next = compact_before_recursion(
                    head,
                    vec![Message::User(nudge)],
                    &tool_use_context,
                    repetition_recovery_count == REPETITION_LAST_RESORT_ATTEMPT,
                ).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count,
                    repetition_recovery_count: repetition_recovery_count + 1,
                    suppress_reasoning: false,
                    tool_parse_recovery_count,
                    empty_turn_recovery_count,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
            }
            // budget spent (note already shown) OR recovery done: end the turn.
            return;
        }

        if is_empty_turn {
            // <= for the same reason as the paths below: spend the last attempt
            // asking, not on ending the run in silence.
            if empty_turn_recovery_count <= EMPTY_TURN_RECOVERY_LIMIT {
                debug_log("EMPTY_TURN_RECOVERY", Some(&json!({
                    "attempt": empty_turn_recovery_count + 1,
                    "limit": EMPTY_TURN_RECOVERY_LIMIT,
                })));
                let mut nudge = create_user_message_text(EMPTY_TURN_RECOVERY_PROMPT);
                nudge.is_meta = true;
                yield Message::User(nudge.clone());
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                let next = compact_before_recursion(
                    head, vec![Message::User(nudge)], &tool_use_context, false,
                ).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count,
                    repetition_recovery_count,
                    // 빈 턴 회복: 추론만 하고 내용을 못 낸 실패. 붕괴와 같은 계열이다.
                    // 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                    // 같은 자리에서 무너진다 (2026-09-13 실측).
                    suppress_reasoning: false,
                    tool_parse_recovery_count,
                    empty_turn_recovery_count: empty_turn_recovery_count + 1,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
            }
            // budget spent (note already shown) OR recovery done: end the turn.
            return;
        }

        if is_tool_parse_failure {
            // <= for the same reason as the repetition path above: spend the
            // last attempt telling the model to move on, not on ending the run.
            if tool_parse_recovery_count <= TOOL_PARSE_RECOVERY_LIMIT {
                debug_log("TOOL_PARSE_RECOVERY", Some(&json!({
                    "attempt": tool_parse_recovery_count + 1,
                    "limit": TOOL_PARSE_RECOVERY_LIMIT,
                })));
                let mut nudge = create_user_message_text(TOOL_PARSE_RECOVERY_PROMPT);
                nudge.is_meta = true;
                yield Message::User(nudge.clone());
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                let next = compact_before_recursion(
                    head, vec![Message::User(nudge)], &tool_use_context, false,
                ).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count,
                    repetition_recovery_count,
                    // 도구 인자 파싱 실패 회복: 인자가 잘린 턴을 다시 시킨다.
                    // 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                    // 같은 자리에서 무너진다 (2026-09-13 실측).
                    suppress_reasoning: false,
                    tool_parse_recovery_count: tool_parse_recovery_count + 1,
                    empty_turn_recovery_count: 0,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
            }
            // budget spent (note already shown) OR recovery done: end the turn.
            return;
        }

        if let Some((names, repeat_count)) = repeated_cap {
            debug_log("REPEATED_TOOL_CALL_CAPPED", Some(&json!({
                "tools": names, "consecutive": repeat_count, "limit": REPEATED_TOOL_CALL_LIMIT,
            })));
            crate::cmprintln!("  ⎿  Blocked — the same {names} call was repeated {repeat_count} times. Not running it again.\n");
            // Blocking the call is right (identical arguments cannot produce a
            // new answer), but ENDING THE TURN here is not: an unattended run
            // dies mid-task over one stuck step it could have walked around.
            // Share the repetition budget — it is the same class of failure —
            // and hand the model a directive instead of a full stop.
            if repetition_recovery_count <= REPETITION_RECOVERY_LIMIT {
                let mut nudge = create_user_message_text(&format!(
                    "The same {names} call was issued {repeat_count} times in a row with identical \
                     arguments, so it was blocked rather than run again — an identical call cannot \
                     return a different answer. Do not reissue it. Either change the arguments \
                     substantively, use a different tool, or state your conclusion from what you \
                     already have (\"this could not be determined\" is a valid conclusion) and \
                     continue with the rest of the task."
                ));
                nudge.is_meta = true;
                yield Message::User(nudge.clone());
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                let next = compact_before_recursion(
                    head, vec![Message::User(nudge)], &tool_use_context, false,
                ).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count,
                    repetition_recovery_count: repetition_recovery_count + 1,
                    // 동일 도구 반복 회복. 여기서는 모델이 정상적으로 내용을
                    // 쓰고 형식도 맞는 호출을 하므로 추론이 막힌 것은 아니다.
                    // 그래도 low 로 보내는 이유는, 이 분기가 repetition 예산을
                    // 공유하기 때문이다. 여기서 붕괴가 나면 토큰 붕괴 쪽이 쓸
                    // 예산까지 같이 태운다.
                    suppress_reasoning: false,
                    tool_parse_recovery_count,
                    empty_turn_recovery_count,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
            }
            return;
        }

        // Check for tool use.
        let tool_use_blocks: Vec<ToolUseBlock> = assistant_message
            .message
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(tu) => Some(tu.clone()),
                _ => None,
            })
            .collect();

        if tool_use_blocks.is_empty() {
            // max_output_tokens recovery.
            let stop_reason = assistant_message.message.stop_reason.as_deref();
            if stop_reason == Some("max_tokens")
                && max_output_recovery_count < MAX_OUTPUT_TOKENS_RECOVERY_LIMIT
            {
                debug_log("MAX_OUTPUT_RECOVERY", Some(&json!({
                    "attempt": max_output_recovery_count + 1,
                    "limit": MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
                })));
                let mut recovery = create_user_message_text(MAX_OUTPUT_RECOVERY_PROMPT);
                recovery.is_meta = true;
                yield Message::User(recovery.clone());
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                let next = compact_before_recursion(
                    head, vec![Message::User(recovery)], &tool_use_context, false,
                ).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count: max_output_recovery_count + 1,
                    repetition_recovery_count,
                    // 출력 한도 회복: 추론이 예산을 먹고 한도에 닿은 경우가 있다.
                    // 회복 요청은 low 로 — 추론이 막힌 상태면 회복 자체가
                    // 같은 자리에서 무너진다 (2026-09-13 실측).
                    suppress_reasoning: false,
                    tool_parse_recovery_count,
                    empty_turn_recovery_count,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
                return;
            }

            // API-error guard: don't run Stop hooks if the model call errored.
            if assistant_message.is_api_error_message {
                return;
            }

            // Goal completion gates.
            if let Some(cont) = maybe_run_goal_gates(
                &messages, &assistant_message, &system_prompt, &context,
                &can_use_tool, &tool_use_context, stop_hook_active, turn_count,
                RecoveryBudgets {
                    max_output: max_output_recovery_count,
                    repetition: repetition_recovery_count,
                    tool_parse: tool_parse_recovery_count,
                    empty_turn: empty_turn_recovery_count,
                },
            ).await {
                let (gmsg, mut next) = cont;
                yield Message::User(gmsg);
                let mut inner = query(next.take().unwrap());
                while let Some(m) = inner.next().await { yield m; }
                return;
            }

            // Stop hooks.
            if crate::utils::hooks::has_hook_for_event("Stop") {
                let sr = crate::utils::hooks::handle_stop_hooks(
                    &messages, &assistant_message, &tool_use_context, stop_hook_active,
                ).await;
                if sr.prevent_continuation {
                    debug_log("STOP_HOOK_PREVENTED", Some(&json!({})));
                    return;
                }
                if !sr.blocking_errors.is_empty() {
                    debug_log("STOP_HOOK_BLOCKING", Some(&json!({"count": sr.blocking_errors.len()})));
                    let mut head = messages.clone();
                    head.push(Message::Assistant(assistant_message.clone()));
                    let mut tail = Vec::new();
                    for bm in &sr.blocking_errors {
                        yield Message::User(bm.clone());
                        tail.push(Message::User(bm.clone()));
                    }
                    let next =
                        compact_before_recursion(head, tail, &tool_use_context, false).await;
                    let mut inner = query(QueryParams {
                        messages: next,
                        system_prompt: system_prompt.clone(),
                        context: context.clone(),
                        can_use_tool: can_use_tool.clone(),
                        tool_use_context: tool_use_context.clone(),
                        max_output_recovery_count,
                        repetition_recovery_count,
                        suppress_reasoning: false,
                        tool_parse_recovery_count,
                        empty_turn_recovery_count,
                        stop_hook_active: true,
                        turn_count: turn_count + 1,
                    });
                    while let Some(m) = inner.next().await { yield m; }
                }
            }
            return;
        }

        // Repeated-tool-call cap.
        if let Some(batch_sig) = tool_batch_signature(&Message::Assistant(assistant_message.clone())) {
            let repeat_count = count_repeated_tool_batches(&messages, &batch_sig) + 1;
            // The LIMIT branch is handled before the yield above, where the
            // tool_use blocks can still be stripped from the message that goes
            // into history. Only the nudge remains here: it injects a synthetic
            // tool_result for every call, so history stays balanced.
            if repeat_count >= REPEATED_TOOL_CALL_NUDGE {
                let mut names: Vec<String> = tool_use_blocks.iter().map(|t| t.name.clone()).collect();
                names.sort();
                names.dedup();
                let names = names.join(", ");
                debug_log("REPEATED_TOOL_CALL_NUDGED", Some(&json!({
                    "tools": names, "consecutive": repeat_count,
                    "nudge": REPEATED_TOOL_CALL_NUDGE, "limit": REPEATED_TOOL_CALL_LIMIT,
                })));
                crate::cmprintln!("  ⎿  Skipped — the same {names} call, {repeat_count} times in a row. Asked for a different approach.\n");
                let mut head = messages.clone();
                head.push(Message::Assistant(assistant_message.clone()));
                let mut nudges = Vec::new();
                for tu in &tool_use_blocks {
                    let m = create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                        tool_use_id: tu.id.clone(),
                        content: ToolResultContent::Text(
                            REPEATED_TOOL_CALL_NUDGE_PROMPT.replace("{n}", &repeat_count.to_string()),
                        ),
                        is_error: true,
                        cache_control: None,
                        data: None,
                    }]);
                    nudges.push(m);
                }
                let mut tail = Vec::new();
                for nm in &nudges {
                    yield Message::User(nm.clone());
                    tail.push(Message::User(nm.clone()));
                }
                let next =
                    compact_before_recursion(head, tail, &tool_use_context, false).await;
                let mut inner = query(QueryParams {
                    messages: next,
                    system_prompt: system_prompt.clone(),
                    context: context.clone(),
                    can_use_tool: can_use_tool.clone(),
                    tool_use_context: tool_use_context.clone(),
                    max_output_recovery_count,
                    repetition_recovery_count,
                    suppress_reasoning: false,
                    tool_parse_recovery_count: 0,
                    empty_turn_recovery_count: 0,
                    stop_hook_active,
                    turn_count: turn_count + 1,
                });
                while let Some(m) = inner.next().await { yield m; }
                return;
            }
        }

        // Run tools (concurrent iff ALL are concurrency-safe).
        let all_safe = tool_use_blocks.iter().all(|b| {
            is_tool_concurrency_safe(&b.name, &tools, Some(&b.input))
        });

        let mut tool_results: Vec<Message> = Vec::new();
        let mut exec = if all_safe {
            run_tools_concurrently(tool_use_blocks.clone(), assistant_message.clone(), can_use_tool.clone(), tool_use_context.clone())
        } else {
            run_tools_serially(tool_use_blocks.clone(), assistant_message.clone(), can_use_tool.clone(), tool_use_context.clone())
        };
        while let Some(message) = exec.next().await {
            let is_user = matches!(message, Message::User(_));
            yield message.clone();
            if is_user {
                tool_results.push(message);
            }
        }

        // Guarantee tool_use <-> tool_result pairing.
        let aborted = tool_use_context.is_aborted();
        let mut answered_ids: HashSet<String> = HashSet::new();
        for r in &tool_results {
            if let Some(u) = r.as_user() {
                for block in u.message.content.as_blocks() {
                    if let ContentBlockParam::ToolResult { tool_use_id, .. } = block {
                        answered_ids.insert(tool_use_id);
                    }
                }
            }
        }
        for tu in &tool_use_blocks {
            if !answered_ids.contains(&tu.id) {
                let missing_error = if aborted {
                    "Interrupted by user".to_string()
                } else {
                    debug_log("TOOL_RESULT_ORPHAN", Some(&json!({
                        "tool_use_id": tu.id, "tool_name": tu.name,
                        "note": "tool_use had no matching tool_result outside abort/error path",
                    })));
                    "Tool did not return a result.".to_string()
                };
                let missing = create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                    tool_use_id: tu.id.clone(),
                    content: ToolResultContent::Text(missing_error),
                    is_error: true,
                    cache_control: None,
                    data: None,
                }]);
                yield Message::User(missing.clone());
                tool_results.push(Message::User(missing));
            }
        }

        if aborted {
            yield Message::Assistant(create_assistant_message_obj(INTERRUPT_MESSAGE_FOR_TOOL_USE));
            return;
        }

        // Sort tool results to match the order of tool_use blocks.
        let order: BTreeMap<String, usize> = tool_use_blocks
            .iter()
            .enumerate()
            .map(|(i, tu)| (tu.id.clone(), i))
            .collect();
        tool_results.sort_by_key(|r| {
            let tid = r.as_user()
                .and_then(|u| u.message.content.as_blocks().into_iter().find_map(|b| match b {
                    ContentBlockParam::ToolResult { tool_use_id, .. } => Some(tool_use_id),
                    _ => None,
                }))
                .unwrap_or_default();
            *order.get(&tid).unwrap_or(&tool_use_blocks.len())
        });

        // Auto-compact before recursive query.
        let mut next_messages = messages.clone();
        next_messages.push(Message::Assistant(assistant_message.clone()));
        next_messages.extend(tool_results.into_iter());

        next_messages =
            compact_before_recursion(next_messages, Vec::new(), &tool_use_context, false).await;

        // Mid-turn notification drain (main thread only).
        if tool_use_context.agent_id.is_none() {
            for block in crate::utils::message_queue_manager::drain_pending_notifications() {
                let notif = create_user_message_text(&block);
                next_messages.push(Message::User(notif.clone()));
                yield Message::User(notif);
            }
        }

        // === Mid-turn message to a sub-agent (SendMessage) ===
        // The mirror of the block above, and the exact opposite gate: ONLY a
        // sub-agent drains here, and only its own queue. The parent queued this
        // with SendMessage while the agent was already mid-turn, so it could not
        // be handed to the agent's message list directly — the loop running the
        // agent owns that list. This is the tool-round boundary the queue was
        // waiting for.
        //
        // Nothing can be stolen the way an untagged notification queue can be:
        // the queue lives on the agent's own task, so the only reader is the
        // agent the message was addressed to.
        //
        // The source drains it inside its per-turn attachment builder, which
        // runs before every turn including a sub-agent's; this is codemason's
        // equivalent per-turn injection point. The framing is wrapCommandText's
        // 'coordinator' case verbatim — it has to say not just what arrived but
        // when to deal with it, or a mid-task agent treats a new instruction as
        // idle context.
        if let Some(agent_id) = tool_use_context.agent_id.as_deref() {
            for queued in crate::tasks::local_agent_task::drain_pending_messages(agent_id) {
                let framed = crate::utils::messages::wrap_command_text(&queued, Some("coordinator"));
                let mut sent = create_user_message_text(&framed);
                sent = crate::utils::messages::wrap_messages_in_system_reminder(vec![sent])
                    .into_iter()
                    .next()
                    .expect("one message in, one message out");
                sent.is_meta = true;
                next_messages.push(Message::User(sent.clone()));
                yield Message::User(sent);
            }
        }

        // Mid-turn user steering.
        if tool_use_context.agent_id.is_none() {
            let mut steered = false;
            for steer_text in crate::utils::steering_queue::drain_steering_messages() {
                let steer_msg = create_user_message_text(&steer_text);
                next_messages.push(Message::User(steer_msg.clone()));
                yield Message::User(steer_msg);
                steered = true;
                if let Some(router) = &tool_use_context.input_router {
                    router.note_steer_injected(&steer_text);
                }
            }
            // One note per drain, after the interjections, so the model knows
            // the turn is still live. Appended, so the cached prefix is intact.
            if steered {
                let mut resume = create_user_message_text(MAIN_STEER_RESUME);
                resume.is_meta = true;
                next_messages.push(Message::User(resume.clone()));
                yield Message::User(resume);
            }
        } else if crate::utils::steering_queue::has_steering_messages() {
            let already = *tool_use_context.steer_wrapup.lock();
            if !already {
                *tool_use_context.steer_wrapup.lock() = true;
                let mut wrap = create_user_message_text(AGENT_STEER_WRAPUP);
                wrap.is_meta = true;
                next_messages.push(Message::User(wrap.clone()));
                yield Message::User(wrap);
            } else {
                // Already asked to wrap up, yet it called tools again — force-stop.
                return;
            }
        }

        // Recursive query with tool results.
        let mut inner = query(QueryParams {
            messages: next_messages,
            system_prompt,
            context,
            can_use_tool,
            tool_use_context,
            max_output_recovery_count,
            // Reset, like the two below it. The budget is there to end a turn
            // the model cannot escape, not to count the times it escaped: a
            // tool-using turn means it produced content and a well-formed
            // call, so whatever it was stuck on is behind it.
            //
            // Carrying the count forward instead made a working recovery fail
            // by succeeding. Observed 2026-09-13: a session collapsed and
            // recovered eight separate times, each on the first retry, and
            // then stopped with "the model kept repeating itself across
            // several attempts and could not break out of the loop" — the one
            // outcome an unattended run cannot survive, reached by a guard
            // that had worked every single time.
            repetition_recovery_count: 0,
            suppress_reasoning: false,
            tool_parse_recovery_count: 0,
            empty_turn_recovery_count: 0,
            stop_hook_active,
            turn_count: turn_count + 1,
        });
        while let Some(m) = inner.next().await { yield m; }
    })
}

/// The four Layer-2 recovery budgets, carried across a continuation.
///
/// Grouped only so a continuation cannot silently drop one by passing a literal
/// 0 for it — which is what the goal-gate path did for all four.
#[derive(Clone, Copy)]
struct RecoveryBudgets {
    max_output: i64,
    repetition: i64,
    tool_parse: i64,
    empty_turn: i64,
}

/// Compact before a recursion, exactly as the tool-result path does.
///
/// Every recursion in this file appends to the history and calls back into
/// `query`, but only the tool-result path ever asked whether the history had
/// outgrown the window first. The recovery paths — repetition, empty turn,
/// tool-parse failure, the repeated-call cap, max_tokens, Stop-hook blocking —
/// recursed straight past that check, and each of them appends TWO messages
/// (the failed assistant turn plus a nudge) to a history that was already at
/// the threshold. A turn that collapsed at 158k therefore made its next attempt
/// at 160k, then 162k, with the collapse getting more likely at every step.
/// Measured on 2026-09-28: 158k → 186k across one morning, no compaction at any
/// point, every turn ending in a degenerate emission.
///
/// `head` is summarized when compaction runs; `tail` is appended AFTERWARDS and
/// survives. The nudge belongs in the tail — a full compaction replaces the
/// whole conversation with a single summary message, so a nudge inside `head`
/// is a recursion with no instruction left in it, which is worse than not
/// compacting at all.
///
/// `context_proven_fatal` is passed straight through to auto_compact_if_needed,
/// which documents it. True on exactly one of the nine call sites: the
/// repetition recovery, the only one that has watched this history collapse.
/// That 2026-09-28 measurement is also why the check cannot simply be "are we
/// over the threshold" — it was under it, by 181,355 tokens, while dying.
async fn compact_before_recursion(
    head: Vec<Message>,
    tail: Vec<Message>,
    tool_use_context: &ToolUseContext,
    context_proven_fatal: bool,
) -> Vec<Message> {
    let mut next = head;
    let tracking = tool_use_context.autocompact_tracking.lock().consecutive_failures;
    let compact = auto_compact_if_needed(
        &next, tool_use_context, tracking,
        tool_use_context.on_token_update.clone(),
        tool_use_context.on_thinking_update.clone(),
        context_proven_fatal,
    ).await;
    if let Some(cf) = compact.consecutive_failures {
        tool_use_context.autocompact_tracking.lock().consecutive_failures = cf;
    }
    if compact.was_compacted {
        if let Some(cr) = &compact.compaction_result {
            next = cr.messages.clone();
        }
        // The summary goes into the recursion and dies with it, exactly like a
        // microcompaction -- but this one the REPL cannot replay, because a
        // summary is an API round trip and not a function of the history. So
        // record that a summary is owed and let the turn boundary make it.
        //
        // Nothing was raised here at all before. The REPL then carried the FULL
        // history into the next turn while the usage anchor described the
        // summarized prompt -- a few thousand tokens against four hundred
        // thousand -- so the gauge read near-empty while every request was
        // refused for length: the same defect as the microcompaction one, with
        // the divergence an order of magnitude wider.
        *tool_use_context.needs_summary.lock() = true;
        crate::cmprintln!("  ⎿  Auto compacted — conversation summarized. Continuing...\n");
    } else if compact.was_microcompacted {
        if let Some(m) = compact.messages {
            next = m;
        }
        // This set goes into the recursion and is then dropped: the history is
        // the REPL's, and nothing here can replace it. So tell the REPL to redo
        // the reduction where it does stick — for EVERY microcompaction.
        //
        // This was once gated on context_proven_fatal, reasoning that the other
        // eight call sites reduce a HEALTHY history and "the turn-boundary check
        // will reach it on its own terms". It cannot. The mid-turn requests go
        // out TRIMMED, so the usage anchor describes the slim prompt while this
        // history is still the fat one; should_auto_compact then weighs the slim
        // number and declines. Worse, the only writer of proven_fatal is the
        // repetition recovery, so once collapses stopped the gate never opened
        // again and ordinary reductions were dropped every time.
        //
        // Measured 2026-10-03 18:07: message 7 returned at its original 17,958
        // bytes — the same hash it carried before the 17:55 reduction cleared it
        // to 114 — the gauge read 262k, and the server was handed 458,753 input
        // + 65,536 reserved output. 524,289 against a 524,288 window: 400.
        *tool_use_context.needs_reduction.lock() = true;
        // proven_fatal keeps its own, narrower meaning: the decoder died here.
        if context_proven_fatal {
            *tool_use_context.proven_fatal.lock() = true;
        }
        crate::cmprintln!("  ⎿  Microcompacted — old tool results cleared, recent context kept.\n");
    }
    next.extend(tail);
    next
}

/// The goal-gate continuation, or None. Returns the meta gate message and the
/// fully-built recursive params (so the caller can yield the message then
/// stream the recursion).
async fn maybe_run_goal_gates(
    messages: &[Message],
    assistant_message: &AssistantMessage,
    system_prompt: &[String],
    context: &BTreeMap<String, String>,
    can_use_tool: &CanUseToolFn,
    tool_use_context: &ToolUseContext,
    stop_hook_active: bool,
    turn_count: i64,
    budgets: RecoveryBudgets,
) -> Option<(crate::types::query::UserMessage, Option<QueryParams>)> {
    use crate::utils::goal::{get_gates, get_goal, run_gates};
    let goal = get_goal();
    if goal.is_empty() || get_gates().is_empty() || !crate::utils::config::get_config().goal_gate_autorun {
        // All gates pass / no gates — clear the counter for the next turn.
        *tool_use_context.gate_recovery.lock() = 0;
        return None;
    }
    let gr = *tool_use_context.gate_recovery.lock();
    if gr >= GOAL_GATE_RECOVERY_LIMIT {
        return None;
    }
    let cwd = crate::utils::state::get_cwd();
    let results = tokio::task::spawn_blocking(move || run_gates(Some(&cwd)))
        .await
        .unwrap_or_default();
    let failed: Vec<(String, i32, String)> = results
        .into_iter()
        .filter(|(_c, ok, _code, _tail)| !ok)
        .map(|(c, _ok, code, tail)| (c, code, tail))
        .collect();
    if failed.is_empty() {
        *tool_use_context.gate_recovery.lock() = 0;
        return None;
    }
    *tool_use_context.gate_recovery.lock() = gr + 1;
    debug_log("GOAL_GATE_FAILED", Some(&json!({
        "attempt": gr + 1, "limit": GOAL_GATE_RECOVERY_LIMIT, "failed": failed.len(),
    })));
    let mut gmsg = create_user_message_text(&format_gate_failures(&goal, &failed, gr + 1, GOAL_GATE_RECOVERY_LIMIT));
    gmsg.is_meta = true;
    let mut head = messages.to_vec();
    head.push(Message::Assistant(assistant_message.clone()));
    let next = compact_before_recursion(
        head, vec![Message::User(gmsg.clone())], tool_use_context, false,
    ).await;
    let params = QueryParams {
        messages: next,
        system_prompt: system_prompt.to_vec(),
        context: context.clone(),
        can_use_tool: can_use_tool.clone(),
        tool_use_context: tool_use_context.clone(),
        // CARRIED, not reset. Resetting all four here handed every gate attempt
        // a full fresh recovery budget, so a model collapsing on the same gate
        // could spend REPETITION_RECOVERY_LIMIT nudges per attempt, times
        // GOAL_GATE_RECOVERY_LIMIT attempts, on one stuck turn. The budgets
        // exist to end a turn the model cannot escape; a failing gate is
        // evidence it has not escaped, not evidence it has.
        max_output_recovery_count: budgets.max_output,
        repetition_recovery_count: budgets.repetition,
        suppress_reasoning: false,
        tool_parse_recovery_count: budgets.tool_parse,
        empty_turn_recovery_count: budgets.empty_turn,
        stop_hook_active,
        turn_count: turn_count + 1,
    };
    Some((gmsg, Some(params)))
}

// === Tool Execution ===

/// Run multiple tools concurrently (runToolsConcurrently).
fn run_tools_concurrently(
    tool_use_blocks: Vec<ToolUseBlock>,
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> MsgStream {
    Box::pin(async_stream::stream! {
        let sibling_ids: HashSet<String> = tool_use_blocks.iter().map(|t| t.id.clone()).collect();

        // Two identical calls in ONE turn: keep the first, answer the rest here.
        let mut unique = Vec::new();
        let mut duplicates = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tu in &tool_use_blocks {
            let key = tool_call_key(&tu.name, &tu.input);
            if seen.contains(&key) {
                duplicates.push(tu.clone());
            } else {
                seen.insert(key);
                unique.push(tu.clone());
            }
        }
        for tu in duplicates {
            debug_log("TOOL_REPEAT_BLOCKED", Some(&json!({"tool": tu.name, "reason": "same_batch"})));
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tu.id.clone(),
                content: ToolResultContent::Text(format!(
                    "Blocked: an identical {} call with these exact arguments was already issued in this same turn. Duplicating it cannot return anything different — use the other call's result.", tu.name
                )),
                is_error: true, cache_control: None, data: None,
            }]));
        }

        let generators: Vec<MsgStream> = unique
            .into_iter()
            .map(|tu| run_tool_use(tu, sibling_ids.clone(), assistant_message.clone(), can_use_tool.clone(), tool_use_context.clone()))
            .collect();

        let mut merged = all_streams(generators, MAX_TOOL_USE_CONCURRENCY);
        while let Some(m) = merged.next().await { yield m; }
    })
}

/// Run tools one at a time (runToolsSerially).
fn run_tools_serially(
    tool_use_blocks: Vec<ToolUseBlock>,
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> MsgStream {
    Box::pin(async_stream::stream! {
        let sibling_ids: HashSet<String> = tool_use_blocks.iter().map(|t| t.id.clone()).collect();
        for tu in tool_use_blocks {
            let mut inner = run_tool_use(tu, sibling_ids.clone(), assistant_message.clone(), can_use_tool.clone(), tool_use_context.clone());
            while let Some(m) = inner.next().await { yield m; }
        }
    })
}

/// Execute a single tool use (runToolUse).
fn run_tool_use(
    tool_use: ToolUseBlock,
    sibling_ids: HashSet<String>,
    assistant_message: AssistantMessage,
    can_use_tool: CanUseToolFn,
    tool_use_context: ToolUseContext,
) -> MsgStream {
    Box::pin(async_stream::stream! {
        let tool_name = tool_use.name.clone();
        let tools = tool_use_context.options.tools.clone();
        let tool = tools.iter().find(|t| t.matches_name(&tool_name)).cloned();

        let Some(tool) = tool else {
            debug_log("TOOL_USE_ERROR", Some(&json!({"error": format!("No such tool available: {tool_name}")})));
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use.id.clone(),
                content: ToolResultContent::Text(format!("Error: No such tool available: {tool_name}")),
                is_error: true, cache_control: None, data: None,
            }]));
            return;
        };

        let tool_input = tool_use.input.clone();

        // Refuse a call already made with these exact args returning the same
        // thing every time. Decide (and mutate the history) under the guard,
        // then yield OUTSIDE it — a parking_lot guard is not Send and must not
        // straddle the yield await.
        let repeat_key = tool_call_key(&tool_name, &tool_input);
        let blocked_count: Option<usize> = {
            let mut hist = TOOL_REPEAT_HISTORY.lock();
            match hist.get(&repeat_key) {
                Some(prior) if prior.count >= TOOL_REPEAT_LIMIT => {
                    let count = prior.count;
                    hist.remove(&repeat_key);
                    Some(count)
                }
                _ => None,
            }
        };
        if let Some(count) = blocked_count {
            debug_log("TOOL_REPEAT_BLOCKED", Some(&json!({"tool": tool_name, "count": count})));
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use.id.clone(),
                content: ToolResultContent::Text(format!(
                    "Blocked: {tool_name} was already called {count} times with these exact arguments, and every call returned the same result. Running it again cannot produce anything new. This approach is not working — change it: use different arguments, a different tool, or state what is blocking you."
                )),
                is_error: true, cache_control: None, data: None,
            }]));
            return;
        }

        if tool_use_context.is_aborted() {
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use.id.clone(),
                content: ToolResultContent::Text("Tool execution was cancelled".into()),
                is_error: true, cache_control: None, data: None,
            }]));
            return;
        }

        let mut inner = check_permissions_and_call_tool(
            tool, tool_use.id.clone(), sibling_ids, tool_input, tool_use_context.clone(),
            can_use_tool, assistant_message,
        );
        while let Some(message) = inner.next().await {
            if let Some(digest) = tool_result_digest(&message) {
                // Scope the lock guard so it is dropped before the yield await —
                // a parking_lot guard is not Send and would poison the stream's
                // Send bound if held across the suspension point.
                {
                    let mut hist = TOOL_REPEAT_HISTORY.lock();
                    match hist.get_mut(&repeat_key) {
                        Some(seen) if seen.digest == digest => seen.count += 1,
                        _ => { hist.insert(repeat_key.clone(), RepeatEntry { digest, count: 1 }); }
                    }
                }
            }
            yield message;
        }
    })
}

/// Check permissions and call the tool (checkPermissionsAndCallTool).
fn check_permissions_and_call_tool(
    tool: Arc<dyn Tool>,
    tool_use_id: String,
    _sibling_ids: HashSet<String>,
    input: Value,
    context: ToolUseContext,
    can_use_tool: CanUseToolFn,
    _assistant_message: AssistantMessage,
) -> MsgStream {
    Box::pin(async_stream::stream! {
        // Plan mode: block file-modifying tools while in plan mode.
        const PLAN_MODE_ALLOWED: &[&str] = &["EnterPlanMode", "ExitPlanMode"];
        let plan_bypass = context.options.dangerously_skip_permissions;
        if crate::utils::plan_mode::is_plan_mode_active()
            && !tool.is_read_only()
            && !PLAN_MODE_ALLOWED.contains(&tool.name())
            && !plan_bypass
        {
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: ToolResultContent::Text(format!(
                    "{} is blocked in plan mode — no file modifications are allowed. Finish exploring and designing, then call ExitPlanMode to present your plan for approval before coding.", tool.name()
                )),
                is_error: true, cache_control: None, data: None,
            }]));
            return;
        }

        // Validate input.
        let validation = tool.validate_input(&input, &context).await;
        if !validation.result {
            debug_log("TOOL_USE_ERROR", Some(&json!({"error": validation.message})));
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: ToolResultContent::Text(validation.message),
                is_error: true, cache_control: None, data: None,
            }]));
            return;
        }

        // Check permissions (bypassed under dangerously_skip_permissions unless
        // the tool always_ask()).
        let should_skip = context.options.dangerously_skip_permissions;
        if (!should_skip || tool.always_ask()) && tool.needs_permissions() {
            let perm = tool.check_permissions(&input, &context).await;
            match perm.behavior {
                PermissionBehavior::Deny => {
                    let msg = if perm.message.is_empty() { "Permission denied".to_string() } else { perm.message };
                    yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: ToolResultContent::Text(msg),
                        is_error: true, cache_control: None, data: None,
                    }]));
                    return;
                }
                PermissionBehavior::Ask => {
                    if !perm.message.is_empty() {
                        crate::cmprintln!("  {}", perm.message);
                    }
                    let decision = (can_use_tool)(tool.name(), input.clone(), context.clone()).await;
                    if !decision.result {
                        let msg = if decision.message.is_empty() { "Permission denied".to_string() } else { decision.message };
                        yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: ToolResultContent::Text(msg),
                            is_error: true, cache_control: None, data: None,
                        }]));
                        return;
                    }
                }
                PermissionBehavior::Allow => {}
            }
        }

        // Call the tool with a per-call context carrying this tool_use id,
        // wrapped so ESC cancels it mid-await (abortable).
        let call_context = context.with_tool_use_id(&tool_use_id);
        let signal = Some(call_context.abort_controller.clone());
        let raw = tool.call(input, call_context, Some(can_use_tool.clone()));
        let mut stream = abortable(raw, signal, None);

        let mut produced_result = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(ToolResult::Result { result_for_assistant, data }) => {
                    let content_str = crate::utils::tool_result_storage::process_tool_result_content(
                        &result_for_assistant,
                        tool.name(),
                        &tool_use_id,
                        tool.max_result_size_chars(),
                    );
                    yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: ToolResultContent::Text(content_str),
                        is_error: false, cache_control: None,
                        data,
                    }]));
                    produced_result = true;
                    return;
                }
                Ok(ToolResult::ResultWithImages { result_for_assistant, images, data }) => {
                    // Image blocks stay blocks next to the processed text; never
                    // flatten them (the base64 payload would become text and trip
                    // the persist-to-disk cap).
                    let content_str = crate::utils::tool_result_storage::process_tool_result_content(
                        &result_for_assistant,
                        tool.name(),
                        &tool_use_id,
                        tool.max_result_size_chars(),
                    );
                    let mut blocks = vec![json!({"type": "text", "text": content_str})];
                    blocks.extend(images);
                    yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: ToolResultContent::Blocks(blocks),
                        is_error: false, cache_control: None,
                        data,
                    }]));
                    produced_result = true;
                    return;
                }
                Ok(ToolResult::Progress { content, .. }) => {
                    yield Message::Progress(crate::types::query::ProgressMessage {
                        result_for_assistant: String::new(),
                        data: content.unwrap_or(Value::Null),
                    });
                }
                Err(e) => {
                    debug_log("TOOL_USE_ERROR", Some(&json!({"error": e.message})));
                    yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: ToolResultContent::Text(format_error(&e)),
                        is_error: true, cache_control: None, data: None,
                    }]));
                    return;
                }
            }
        }

        if !produced_result {
            // The tool stream finished without yielding a result — an anomaly.
            yield Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: ToolResultContent::Text(format!("Error: {} returned no result.", tool.name())),
                is_error: true, cache_control: None, data: None,
            }]));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_known_vector() {
        // hashlib.sha1(b"abc").hexdigest()
        assert_eq!(sha1_digest("abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(sha1_digest(""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let v = json!({"b": 1, "a": 2});
        assert_eq!(canonical_json(&v), r#"{"a":2,"b":1}"#);
    }

    #[test]
    fn tool_call_key_ignores_description() {
        let a = json!({"command": "ls", "description": "one"});
        let b = json!({"command": "ls", "description": "two"});
        assert_eq!(tool_call_key("Bash", &a), tool_call_key("Bash", &b));
    }

    #[test]
    fn batch_signature_none_for_text_only() {
        let m = Message::Assistant(create_assistant_message_obj("hello"));
        assert!(tool_batch_signature(&m).is_none());
    }
}

#[cfg(test)]
mod cached_prefix_tests {
    use super::*;

    /// The system prompt is the cached prefix: the server re-prefills the WHOLE
    /// conversation when any byte of it moves. At 96k tokens on this project's
    /// backend that is 72s against 3.4s for the same information appended
    /// instead, and a real session paid 138-191s stalls for it at ~200k.
    ///
    /// So nothing turn-scoped may be folded into it. This asserts the rule
    /// positively — the prefix is byte-identical across turns — rather than
    /// naming the two keys that got it wrong, because the next offender will be
    /// a third key nobody thought of.
    #[test]
    fn the_cached_prefix_is_byte_identical_across_turns() {
        let system_prompt = vec!["You are CodeMason.".to_string(), "<env>…</env>".to_string()];

        // What a turn legitimately carries: a stable session-scoped context.
        let mut ctx = BTreeMap::new();
        ctx.insert("active_goal".to_string(), "ship the port".to_string());

        let turn_one = format_system_prompt_with_context(&system_prompt, &ctx);

        // Second turn, later, in a different directory. Both are things the
        // model must know and neither may reach the prefix to say it.
        let turn_two = format_system_prompt_with_context(&system_prompt, &ctx);

        assert_eq!(
            turn_one, turn_two,
            "the cached prefix moved between turns; whatever was just added to \
             the system prompt is turn-scoped and belongs on the user turn"
        );
    }

    /// The guard above only bites if turn-scoped values are actually kept out,
    /// so pin that too: a context carrying one costs the whole prefix.
    #[test]
    fn a_turn_scoped_value_in_the_context_would_move_the_prefix() {
        let sp = vec!["You are CodeMason.".to_string()];
        let mut a = BTreeMap::new();
        a.insert("cwd".to_string(), "/home/user/work/repo-a".to_string());
        let mut b = BTreeMap::new();
        b.insert("cwd".to_string(), "/home/user/work/repo-b".to_string());
        assert_ne!(
            format_system_prompt_with_context(&sp, &a),
            format_system_prompt_with_context(&sp, &b),
            "if this ever passes, the fold stopped including context and the \
             test above has stopped proving anything"
        );
    }
}

#[cfg(test)]
mod repetition_nudge_tests {
    use super::*;
    use crate::utils::messages::create_user_message_blocks;

    fn result(text: &str, is_error: bool) -> Message {
        Message::User(create_user_message_blocks(vec![ContentBlockParam::ToolResult {
            tool_use_id: "t".into(),
            content: ToolResultContent::Text(text.into()),
            is_error,
            cache_control: None,
            data: None,
        }]))
    }

    /// The whole point of the change: the nudge must carry WHAT came back, not
    /// just "you repeated yourself". Asserting the failure text is present, not
    /// merely that the prompt grew.
    #[test]
    fn the_nudge_quotes_what_the_failing_calls_returned() {
        let msgs = vec![
            result("Command failed with exit code -1", true),
            result("Error: execution reverted", true),
        ];
        let p = repetition_recovery_prompt(&msgs, 0);
        assert!(p.starts_with(REPETITION_RECOVERY_PROMPT), "base nudge must survive");
        assert!(p.contains("execution reverted"), "missing the actual failure: {p}");
        assert!(p.contains("exit code -1"), "missing the earlier failure: {p}");
    }

    /// Empty output is a dead end too — it answers nothing, so the model is
    /// returned to the same state exactly as an error does.
    #[test]
    fn a_silent_result_counts_as_a_dead_end() {
        let p = repetition_recovery_prompt(&[result("   ", false)], 0);
        assert!(p.contains("(no output)"), "{p}");
    }

    /// Succeeding calls are not the loop, and quoting them would bury the ones
    /// that are.
    #[test]
    fn successful_results_are_not_quoted() {
        let p = repetition_recovery_prompt(&[result("Solc finished. Compiler run successful!", false)], 0);
        assert_eq!(p, REPETITION_RECOVERY_PROMPT, "nothing failed, so nothing to add");
    }

    /// Bounded: a loop can run for many turns, and the nudge must not become
    /// the largest thing in the request.
    #[test]
    fn the_quote_is_bounded_in_count_and_length() {
        let msgs: Vec<Message> = (0..10).map(|i| result(&format!("failure {i}"), true)).collect();
        let p = repetition_recovery_prompt(&msgs, 0);
        assert_eq!(p.matches("\n- ").count(), REPETITION_FAILURES_MAX);
        assert!(p.contains("failure 9"), "the newest failure must be there: {p}");
        assert!(!p.contains("failure 6"), "older ones must be dropped: {p}");

        let long = "가".repeat(REPETITION_FAILURE_CHARS + 500);
        let p2 = repetition_recovery_prompt(&[result(&long, true)], 0);
        assert!(p2.contains("[…]"), "an over-long result must be marked as cut");
        assert!(
            p2.chars().count() < REPETITION_RECOVERY_PROMPT.chars().count()
                + REPETITION_FAILURES_HEADER.chars().count()
                + REPETITION_FAILURE_CHARS + 20,
            "the cut must actually bound the length"
        );
    }

    /// Korean tool output is the norm on this project; byte slicing here would
    /// panic mid-codepoint rather than truncate.
    #[test]
    fn truncation_does_not_split_a_multibyte_character() {
        let s = "실행이 되돌려졌습니다".repeat(200);
        let out = truncate_chars(&s, 300);
        assert_eq!(out.chars().count(), 300 + 4, "300 chars plus the ' […]' marker");
    }
}


#[cfg(test)]
mod repetition_escalation_tests {
    use super::*;

    /// The turn gets THREE requests: the one that collapsed, one plain retry,
    /// and one on a reduced context. Pinned as a total, because that is the
    /// number the measurement is about — 94.9% of 176 collapse runs escaped on
    /// the retry, nothing escaped by nudge after the 6th attempt, and the
    /// reduction cured both collapses that got that far on its first try.
    ///
    /// This replaces a `LIMIT >= 4` assertion that encoded the opposite belief
    /// ("the budget has to be big enough to talk it out of the loop"). It was
    /// reasonable and it was wrong: the talking never worked, and holding the
    /// reduction behind it cost ~90 seconds and 6 requests per collapse. Raising
    /// the budget again needs new numbers in turns.log, not a new argument.
    #[test]
    fn the_turn_gets_exactly_three_requests() {
        let mut requests = 1; // the one that collapsed
        let mut count = 0;
        while count <= REPETITION_LAST_RESORT_ATTEMPT {
            requests += 1;
            count += 1;
        }
        assert_eq!(
            requests, 3,
            "collapse, one retry, one reduced retry — see REPETITION_RECOVERY_LIMIT"
        );
    }

    /// The single nudge attempt must still be a nudge, with the history intact.
    /// If the reduction crept onto it, every collapse would pay the
    /// irreversible step — including the 94.9% a plain retry already cures.
    #[test]
    fn the_first_retry_does_not_reduce_the_context() {
        assert!(
            REPETITION_RECOVERY_LIMIT < REPETITION_LAST_RESORT_ATTEMPT,
            "there must be at least one plain retry before the reduction"
        );
        assert_ne!(
            0, REPETITION_LAST_RESORT_ATTEMPT,
            "attempt 0 is the collapse itself and must never be the reducing one"
        );
    }

    /// `<=` at the call sites means the budget buys one more attempt than the
    /// constant says. Pinned because the doc comment and the code disagreed
    /// about it (6 vs 7) until someone counted, and because a later change to
    /// `<` would silently remove the abandon-the-work attempt that keeps an
    /// unattended run going.
    #[test]
    fn the_attempt_count_is_one_more_than_the_limit() {
        let mut attempts = 0;
        let mut count = 0;
        while count <= REPETITION_LAST_RESORT_ATTEMPT {
            attempts += 1;
            count += 1;
        }
        assert_eq!(
            attempts,
            REPETITION_RECOVERY_LIMIT + 2,
            "LIMIT + 1 nudge attempts, walked with <=, plus the last-resort one"
        );
    }

    /// The reduction is the LAST thing tried, never an early one.
    ///
    /// Pinned because the ordering is the whole design. microcompact cannot be
    /// undone and costs 403 of 408 tool results on a session like the one that
    /// motivated this; spending it before the nudges are exhausted throws that
    /// away in the ~50% of collapses a plain retry recovers. Spending it at the
    /// end costs nothing, because the only remaining alternative is ending the
    /// turn — which under -dsp is the worst outcome available.
    #[test]
    fn the_context_is_only_reduced_on_the_final_attempt() {
        let forced = |count: i64| count == REPETITION_LAST_RESORT_ATTEMPT;

        // Every nudge attempt leaves the history alone.
        for count in 0..=REPETITION_RECOVERY_LIMIT {
            assert!(!forced(count), "attempt {count} must not reduce the context");
        }
        // The one past the ladder reduces it.
        assert!(forced(REPETITION_LAST_RESORT_ATTEMPT), "the last resort must reduce");
        // And it is reached: the recursion gate admits it.
        assert!(
            REPETITION_LAST_RESORT_ATTEMPT <= REPETITION_LAST_RESORT_ATTEMPT,
            "the gate must admit the attempt that does the reducing",
        );
        // Exactly one attempt reduces, so the irreversible step is spent once.
        let reducing = (0..=REPETITION_LAST_RESORT_ATTEMPT).filter(|c| forced(*c)).count();
        assert_eq!(reducing, 1, "the irreversible step is spent exactly once");
    }

    /// Repeating one instruction at a model that already ignored it is the same
    /// mistake the model is making. Each stage must ask for something
    /// categorically different.
    #[test]
    fn every_stage_asks_for_something_different() {
        // 0..=LIMIT, not 0..LIMIT: the last attempt is an attempt too, and
        // leaving it out of this check is what let the ladder be measured
        // against the wrong number of tries.
        // 0..=LAST_RESORT, not 0..=LIMIT: the reducing attempt carries a nudge
        // too, and it is the one the turn most depends on, so it belongs in
        // this check. Every REACHABLE stage is counted and they must all differ.
        let stages: Vec<&str> = (0..=REPETITION_LAST_RESORT_ATTEMPT)
            .map(repetition_stage_prompt)
            .collect();
        let distinct: std::collections::HashSet<&&str> = stages.iter().collect();
        assert_eq!(
            distinct.len(),
            stages.len(),
            "two reachable attempts must not ask for the same thing twice: {stages:?}"
        );
        assert_eq!(stages[0], REPETITION_RECOVERY_PROMPT, "stage 1 stays the mild one");
    }

    /// The last stage must tell it to move ON, not to try harder — that is what
    /// keeps an unattended run going instead of grinding.
    #[test]
    fn the_last_stage_releases_the_work_instead_of_retrying_it() {
        // The attempt that is actually last, not LIMIT - 1: with the budget cut
        // to one nudge that expression went negative and quietly tested the
        // MILD stage instead of the final one.
        let last = repetition_stage_prompt(REPETITION_LAST_RESORT_ATTEMPT);
        assert!(
            last.contains("continue with") || last.contains("move on"),
            "the final stage must hand the run somewhere to go: {last}"
        );
        assert!(
            last.contains("Do not run another tool"),
            "and must stop it re-entering the same loop: {last}"
        );
    }

    /// The failure quotes ride along at every stage, not only the first.
    #[test]
    fn the_failures_are_quoted_at_every_stage() {
        let msgs = vec![Message::User(create_user_message_blocks(vec![
            ContentBlockParam::ToolResult {
                tool_use_id: "t".into(),
                content: ToolResultContent::Text("Error: execution reverted".into()),
                is_error: true,
                cache_control: None,
                data: None,
            },
        ]))];
        // 0..=LAST_RESORT: `0..LIMIT` is an EMPTY range now that the budget is
        // one nudge, so this test asserted nothing at all.
        for attempt in 0..=REPETITION_LAST_RESORT_ATTEMPT {
            let p = repetition_recovery_prompt(&msgs, attempt);
            assert!(
                p.contains("execution reverted"),
                "stage {attempt} dropped the evidence"
            );
        }
    }
}

#[cfg(test)]
mod steer_resume_tests {
    use super::*;

    /// The two steering paths must pull in OPPOSITE directions, and it is easy
    /// to copy one into the other by accident. A subagent's caller is blocked
    /// waiting, so it wraps up; the main conversation has nobody waiting on it,
    /// so it carries on. If these ever agree, one of them has lost its reason.
    #[test]
    fn the_main_conversation_resumes_where_a_subagent_wraps_up() {
        assert!(
            AGENT_STEER_WRAPUP.contains("do NOT call any more tools"),
            "the subagent path must still stop starting work"
        );
        assert!(
            MAIN_STEER_RESUME.contains("GO BACK to what you were doing"),
            "the main path must send the model back to the work"
        );
        assert!(
            !MAIN_STEER_RESUME.contains("end your turn"),
            "the main path must not tell it to finish: {MAIN_STEER_RESUME}"
        );
    }

    /// An interjection is often a redirect or a stop. Resuming unconditionally
    /// would override the user, which is worse than halting.
    #[test]
    fn resuming_defers_to_an_interjection_that_says_otherwise() {
        assert!(
            MAIN_STEER_RESUME.contains("if it told you to stop"),
            "the note must yield to the message it follows: {MAIN_STEER_RESUME}"
        );
    }
}
