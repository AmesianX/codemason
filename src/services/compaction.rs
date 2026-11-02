//! Conversation compaction for CodeMason.
//! Equivalent to codemason/services/compaction.py — based on
//! src/services/compact/ (autoCompact.ts + prompt.ts + the core of compact.ts
//! compactConversation) from Claude Code.
//!
//! WHAT (thresholds, summary prompt, summary formatting, continuation message,
//! circuit breaker) is ported verbatim. The token-counting it triggers on uses
//! codemason's local substrate (utils::tokens), and the summarization query
//! goes through the model service — the original used the Anthropic API + its
//! own token accounting, neither of which exists locally.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::json;

use crate::types::messages::{ContentBlockParam, MessageContent, ToolResultContent};
use crate::types::query::{AssistantMessage, Message, ReadFileState, ToolUseContext, UserMessage};
use crate::utils::config::get_config;
use crate::utils::log::debug_log;
use crate::utils::messages::create_user_message_text;
use crate::utils::tokens::{
    count_text_tokens, get_context_size, rough_token_count_estimation,
    rough_token_count_estimation_for_messages,
};

// === Threshold constants (autoCompact.ts) ===

pub const MAX_OUTPUT_TOKENS_FOR_SUMMARY: i64 = 20_000;
pub const AUTOCOMPACT_BUFFER_TOKENS: i64 = 13_000;
/// Auto-compact fires when the estimated prompt reaches this share of the
/// context window — a percentage so the trigger scales with the model instead
/// of a fixed absolute reserve. Clamped by the overflow-safe point (see
/// auto_compact_threshold_for) so a large max_tokens can't let a turn overflow.
pub const AUTOCOMPACT_FRACTION: f64 = 0.88;
/// Share of the PARENT's threshold a sub-agent gets when the operator has not
/// set subagent_context_ceiling_tokens. Derived, not absolute, so it scales
/// with the model instead of hard-coding one deployment's KV pool into the
/// client.
///
/// Blocks, not tokens, are the currency, and the client cannot see the two
/// server numbers that set the exchange rate (pool size and
/// VLLM_PREFIX_CACHE_RETENTION_INTERVAL), so the default has to be a fraction
/// that survives an untuned server.
///
/// Measured 2026-09-13 against this deployment (vLLM, 215-block pool, 4608-token
/// attention pages, retention 18432, parent parked at 441,165 tokens; each trial
/// prefilled a separate conversation of size S and then re-queried the parent,
/// reading the prefix-cache hit rate):
///
///   S <= 270,000  -> 99.2%, nothing lost. 270,000 ran 10/10 clean; smaller
///                    sizes 9 more trials over 8 sizes from 60k up. (The 0.8%
///                    is the unfillable last partial block, not a loss.)
///   S  = 280,000  -> 98.2% in 9 of 10 trials, 96.1% in the tenth: the parent
///                    gives up one 4608-token attention block, ~8s to recompute
///   S  = 290,000  -> survived once, evicted twice (0.0% and 25.1%)
///   S >= 296,945  -> 0.0%, ~360s full re-prefill
///
/// Two things about the shape. The boundary is PROBABILISTIC, not a threshold:
/// the same 290,000 survived one trial and killed the parent in the next two,
/// so a size is only usable when it survives repeatedly, never on one green
/// run. And the failure is all-or-nothing: a Mamba hit needs the one state
/// block ending on the boundary, so losing a single checkpoint zeroes the hit
/// rather than trimming its tail, costing a cold re-prefill of the whole
/// parent. Both are why the default sits far below the cliff, not near it.
///
/// The block cost that produces this: a conversation of L tokens holds
/// L/4608 attention blocks plus L/retention Mamba checkpoints. Mamba state
/// during the run does not accumulate — single_type_kv_cache_manager's
/// remove_skipped_blocks frees each superseded block under mamba_cache_mode
/// "align" — so only the checkpoints scale with length. At 441,165 the parent
/// holds ~119 of the 215 blocks and leaves ~96, and 0.30 of the parent
/// threshold (133,725) needs ~37 of them: roughly a third of the measured room,
/// which is the margin the all-or-nothing failure mode deserves.
///
/// That whole derivation assumed ONE sub-agent. This line runs up to
/// max_concurrent_subagents of them at once, so the footprint this fraction
/// describes is multiplied by the slot count — and nothing used to notice.
/// Hence the total below.
pub const SUBAGENT_THRESHOLD_FRACTION: f64 = 0.30;
/// Ceiling on the COMBINED footprint of every sub-agent that may be in flight,
/// as a share of the parent's threshold. The per-agent fraction above is a
/// measurement of what one extra tenant costs; this is a measurement of what
/// all of them together may cost, and auto_compact_threshold takes whichever
/// is smaller.
///
/// Why both numbers are needed. The per-agent fraction alone is silent about
/// concurrency: raise max_concurrent_subagents to 6 and the client cheerfully
/// admits 6 x 133,725 = 802,350 tokens of sub-agent on top of a 445,752-token
/// parent, which is 1,248,102 before Mamba checkpoints against a pool that
/// holds 1,356,053 — a wipe with nothing reporting it. A total alone is no good
/// either: it would hand a single sub-agent the whole budget, far past the
/// 270,000 the September table shows one tenant can cost.
///
/// 0.90 is three sub-agents at the per-agent cap — this line's actual
/// configuration — and that is the case it was measured in. Measured 2026-10-10
/// against this deployment (KV 12GB, fp8_ds_mla, 344 blocks of 4,608 tokens =
/// 1,356,053, read from vLLM's own cache_config_info; the September table above
/// was taken on a 215-block pool, which is why three tenants fit now where the
/// old arithmetic left room for one):
///
///   parent parked at its FULL threshold (444,860-446,967 tokens over the three
///   trials), then three concurrent conversations of ~133,000 each prefilled to
///   completion, then the parent re-read alone and vLLM's prefix-cache counters
///   read across that one request:
///
///     trial 1 -> 99.4% hit, 0 blocks lost, re-read 3.6s
///     trial 2 -> 99.4% hit, 0 blocks lost, re-read 3.7s
///     trial 3 -> 99.0% hit, 0 blocks lost, re-read 5.5s
///
///   The parent held 442,368 tokens — 96 x 4,608, every attention block it has
///   — in all three. The residue is the unfillable last partial block, not a
///   loss, which is why the verdict is taken against the warm baseline and not
///   against 100%. Three trials because the note above says a size is only
///   usable when it survives repeatedly; the cost it is protecting against is
///   the 374s cold prefill those same runs measured.
///
///   Block accounting: 96 attention blocks for the parent plus 29 for each
///   sub-agent is 183 of 344, a little over half the pool, before the Mamba
///   checkpoints — this measurement did not re-read
///   VLLM_PREFIX_CACHE_RETENTION_INTERVAL, so their count is not re-derived
///   here.
///
/// A smaller pool is where the total has to come down, exactly as the per-agent
/// fraction does; subagent_gate's docs say how to read the slot count.
pub const SUBAGENT_TOTAL_FRACTION: f64 = 0.90;
pub const WARNING_THRESHOLD_BUFFER_TOKENS: i64 = 20_000;
pub const ERROR_THRESHOLD_BUFFER_TOKENS: i64 = 20_000;
pub const MANUAL_COMPACT_BUFFER_TOKENS: i64 = 3_000;
pub const MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES: i64 = 3;

// PTL (prompt-too-long) retry: if the compaction summary REQUEST itself
// overflows the context, drop the oldest rounds and retry, so the session
// isn't stuck permanently over-limit (compact.ts truncateHeadForPTLRetry /
// MAX_PTL_RETRIES).
pub const MAX_PTL_RETRIES: i64 = 3;
pub const PTL_RETRY_MARKER: &str = "[earlier conversation truncated for compaction retry]";
/// Safety margin added to the parsed prompt-too-long gap so a single retry
/// actually fits: covers the re-added marker, vLLM's "at least" overflow
/// wording, and rough-estimate vs server-count slack.
const PTL_GAP_MARGIN: i64 = 1024;

fn msg_first_text(m: &Message) -> String {
    match m {
        Message::User(u) => match &u.message.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => match blocks.first() {
                Some(ContentBlockParam::Text { text, .. }) => text.clone(),
                _ => String::new(),
            },
        },
        Message::Assistant(a) => match a.message.content.first() {
            Some(crate::types::messages::ContentBlock::Text(t)) => t.text.clone(),
            _ => String::new(),
        },
        Message::Progress(_) => String::new(),
    }
}

fn looks_like_prompt_too_long(text: &str) -> bool {
    let s = text.to_lowercase();
    [
        "context length",
        "maximum context",
        "context_length_exceeded",
        "longer than the maximum",
        "reduce the length",
        "max_model_len",
        "exceeds the maximum",
        "too long",
    ]
    .iter()
    .any(|k| s.contains(k))
}

pub const POST_COMPACT_MAX_FILES_TO_RESTORE: usize = 5;
pub const POST_COMPACT_TOKEN_BUDGET: i64 = 50_000;
pub const POST_COMPACT_MAX_CHARS_PER_FILE: usize = 20_000; // ~5k tokens

/// Re-read the most-recently-read files fresh and return
/// (path, full_content, meta_user_message) tuples, bounded to 5 files /
/// ~5k tokens each / 50k-token budget. Port of createPostCompactFileAttachments:
/// lets the model keep recent file context after compaction instead of
/// re-reading.
/// Messages telling the model about its agents, for after a compaction
/// (createAsyncAgentAttachmentsIfNeeded).
///
/// The summary does not carry the spawn, so without this the model comes out of
/// a compaction having forgotten that it has agents running — and spawns them
/// again. It covers both the ones still working (so no duplicate) and the ones
/// that finished but whose result nobody has read.
///
/// Three are skipped: a task already retrieved, one that has not started, and
/// the agent's own task — an agent must not be told about itself.
pub fn create_async_agent_attachments_if_needed(
    own_agent_id: Option<&str>,
) -> Vec<crate::types::query::UserMessage> {
    use crate::utils::messages::render_task_status_attachment;
    use crate::utils::task::disk_output::get_task_output_path;

    let state = crate::state::app_state::get_app_state();
    let mut out = Vec::new();
    for task in state.tasks.values() {
        let Some(agent) = task.as_local_agent() else {
            continue;
        };
        if agent.retrieved
            || agent.base.status == "pending"
            || Some(agent.agent_id.as_str()) == own_agent_id
        {
            continue;
        }
        // Running: the prose summary, if the summarisation path wrote one.
        // Terminal: the error, which is the one thing the status alone does not
        // say.
        let delta_summary = if agent.base.status == "running" {
            agent.progress.as_ref().and_then(|p| p.summary.clone())
        } else {
            agent.error.clone()
        };
        out.extend(render_task_status_attachment(&serde_json::json!({
            "type": "task_status",
            "taskId": agent.agent_id,
            "taskType": "local_agent",
            "description": agent.base.description,
            "status": agent.base.status,
            "deltaSummary": delta_summary,
            "outputFilePath": get_task_output_path(&agent.agent_id).to_string_lossy(),
        })));
    }
    out
}

fn post_compact_file_attachments(
    read_file_state: &std::collections::HashMap<String, ReadFileState>,
) -> Vec<(String, String, UserMessage)> {
    let mut entries: Vec<(String, f64)> = read_file_state
        .iter()
        .map(|(path, st)| (path.clone(), st.timestamp))
        .collect();
    // most recently read first
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::new();
    let mut used: i64 = 0;
    for (path, _ts) in entries.into_iter().take(POST_COMPACT_MAX_FILES_TO_RESTORE) {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let full = String::from_utf8_lossy(&bytes).into_owned();
        let shown = if full.chars().count() <= POST_COMPACT_MAX_CHARS_PER_FILE {
            full.clone()
        } else {
            let head: String = full.chars().take(POST_COMPACT_MAX_CHARS_PER_FILE).collect();
            format!("{head}\n... [truncated for post-compact restore] ...")
        };
        let block = format!(
            "<system-reminder>\nRestored after compaction — recently read file {path}:\n\n{shown}\n</system-reminder>"
        );
        // exact (model tokenizer when available), not raw chars/4
        let tokens = count_text_tokens(&block);
        if used > 0 && used + tokens > POST_COMPACT_TOKEN_BUDGET {
            break;
        }
        used += tokens;
        let mut m = create_user_message_text(&block);
        m.is_meta = true;
        out.push((path, full, m));
    }
    out
}

static RE_PTL_LIMIT_A: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"maximum context length is\s+(\d+)").unwrap());
static RE_PTL_LIMIT_B: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(\d+)\s+maximum").unwrap());
static RE_PTL_TOTAL: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"total of(?:\s+at least)?\s+(\d+)").unwrap());
static RE_PTL_INPUT: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(\d+)\s+input tokens").unwrap());
static RE_PTL_OUTPUT: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"(\d+)\s+output tokens").unwrap());

/// Parse how many tokens over the limit a prompt-too-long error is
/// (getPromptTooLongTokenGap). Handles the vLLM/OpenAI form ("maximum context
/// length is L tokens ... for a total of at least A tokens") and the
/// input+output form, returning A-L when > 0, else None so the caller falls
/// back to the 20% heuristic on unparseable (e.g. Vertex/Bedrock) errors.
fn parse_ptl_token_gap(text: Option<&str>) -> Option<i64> {
    let text = text?;
    if text.is_empty() {
        return None;
    }
    let s = text.to_lowercase();
    let limit = RE_PTL_LIMIT_A
        .captures(&s)
        .or_else(|| RE_PTL_LIMIT_B.captures(&s))?;
    let limit_n: i64 = limit[1].parse().ok()?;
    if let Some(total) = RE_PTL_TOTAL.captures(&s) {
        let gap = total[1].parse::<i64>().ok()? - limit_n;
        return if gap > 0 { Some(gap) } else { None };
    }
    // input + requested-output form: actual = input + output.
    if let Some(inp) = RE_PTL_INPUT.captures(&s) {
        let mut actual: i64 = inp[1].parse().ok()?;
        if let Some(out) = RE_PTL_OUTPUT.captures(&s) {
            actual += out[1].parse::<i64>().unwrap_or(0);
        }
        let gap = actual - limit_n;
        return if gap > 0 { Some(gap) } else { None };
    }
    None
}

/// groupMessagesByApiRound: group 0 is the leading preamble (everything before
/// the first assistant); each later group starts at an assistant message and
/// runs until the next one, keeping tool_use/tool_result pairs together so
/// dropping whole groups never orphans a tool result.
fn group_messages_by_api_round(messages: &[Message]) -> Vec<Vec<Message>> {
    let mut groups: Vec<Vec<Message>> = Vec::new();
    let mut cur: Vec<Message> = Vec::new();
    for m in messages {
        if matches!(m, Message::Assistant(_)) && !cur.is_empty() {
            groups.push(std::mem::take(&mut cur));
        }
        cur.push(m.clone());
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// Drop the OLDEST api-round groups until the prompt-too-long token gap is
/// covered — surgical, not a flat 20% — and prepend a marker. Faithful port of
/// compact.ts truncateHeadForPTLRetry: parse the overflow from the error and
/// peel only enough context to fit, keeping >=1 group. Falls back to 20% of
/// groups when the gap is unparseable.
fn truncate_head_for_ptl(messages: &[Message], ptl_text: Option<&str>) -> Option<Vec<Message>> {
    // Strip our own marker from a prior retry so grouping/progress isn't stalled.
    let inp: &[Message] = match messages.first() {
        Some(m0 @ Message::User(_)) if msg_first_text(m0) == PTL_RETRY_MARKER => &messages[1..],
        _ => messages,
    };

    let groups = group_messages_by_api_round(inp);
    if groups.len() < 2 {
        return None;
    }

    let gap = parse_ptl_token_gap(ptl_text);
    let mut drop_count: usize;
    if let Some(gap) = gap {
        // Deviation from the strict port (acc >= gap): require a margin on top.
        // (1) we re-add PTL_RETRY_MARKER below, so freeing exactly `gap` nets
        // to zero progress when the oldest group is tiny; (2) vLLM reports "a
        // total of AT LEAST N tokens", so the real overflow may exceed the
        // parsed gap; (3) the rough estimate differs from the server's real
        // count. The margin is still tiny next to one round, so this stays
        // surgical (far from 20%).
        let need = gap + PTL_GAP_MARGIN;
        let mut acc: i64 = 0;
        drop_count = 0;
        for g in &groups {
            acc += rough_token_count_estimation_for_messages(g);
            drop_count += 1;
            if acc >= need {
                break;
            }
        }
    } else {
        drop_count = std::cmp::max(1, groups.len() / 5);
    }

    // Keep at least one group so there's something to summarize.
    drop_count = std::cmp::min(drop_count, groups.len() - 1);
    if drop_count < 1 {
        return None;
    }

    let sliced: Vec<Message> = groups[drop_count..].iter().flatten().cloned().collect();
    // Dropping the preamble can leave an assistant-first sequence (the API
    // requires a user-role first message); prepend the marker to fix it.
    if matches!(sliced.first(), Some(Message::Assistant(_))) {
        let mut marker = create_user_message_text(PTL_RETRY_MARKER);
        marker.is_meta = true;
        let mut out = vec![Message::User(marker)];
        out.extend(sliced);
        return Some(out);
    }
    Some(sliced)
}

/// getEffectiveContextWindowSize: context window minus reserved output.
///
/// Every turn asks for cfg.max_tokens of output (65536 by default), so the
/// effective window must keep at least that much free — otherwise a large
/// input slips under the compaction threshold while still overflowing the
/// model (input + max_tokens > context, the 458753+65536 > 524288 error). The
/// original port reserved only MAX_OUTPUT_TOKENS_FOR_SUMMARY, which is correct
/// only while max_tokens <= that; keep it as the floor so compaction always
/// has room to write its summary too.
pub fn get_effective_context_window_size() -> i64 {
    let reserved = std::cmp::max(get_config().max_tokens, MAX_OUTPUT_TOKENS_FOR_SUMMARY);
    get_context_size() - reserved
}

/// The smallest prompt at which this deployment's decoder has actually
/// collapsed this session, or None until one is seen.
///
/// Reported, never acted on. Every input to the auto-compact threshold
/// describes the window the SERVER advertises; this describes a history the
/// MODEL failed on, which is a different kind of fact and a weaker one — one
/// sample, no control, and no evidence about the largest prompt that WOULD have
/// survived. It is kept because it is the number nobody had, not because it is
/// authority to shrink the window.
static OBSERVED_COLLAPSE_TOKENS: Lazy<Mutex<Option<i64>>> = Lazy::new(|| Mutex::new(None));

/// Collapses smaller than this are recorded but not written to error.log.
/// Decoders also collapse for reasons that have nothing to do with length (a
/// poisoned tool result, a pathological repetition in the input), and a line
/// pointing at the context window would misdirect whoever reads it.
const MIN_COLLAPSE_REPORT_TOKENS: i64 = 65_536;

/// Record that a request of `prompt_tokens` came back collapsed, and say so.
///
/// Called from the one place that sees both numbers: the stop reason and the
/// prompt size of the request that earned it (services::openai_common, next to
/// the turns.log line that reports the same figure).
///
/// Keeps the LOWEST collapse seen. A later, higher one says nothing new. This
/// only decides what gets reported — no threshold reads it.
pub fn note_decoder_collapse(prompt_tokens: i64) {
    if prompt_tokens <= 0 {
        return;
    }
    let mut slot = OBSERVED_COLLAPSE_TOKENS.lock();
    let is_new_low = slot.map_or(true, |seen| prompt_tokens < seen);
    if !is_new_low {
        return;
    }
    *slot = Some(prompt_tokens);
    drop(slot);
    let window = get_context_size();
    let threshold = auto_compact_threshold_for(false);
    debug_log(
        "DECODER_COLLAPSE_OBSERVED",
        Some(&json!({
            "prompt_tokens": prompt_tokens,
            "advertised_window": window,
            "threshold": threshold,
            "reported": prompt_tokens >= MIN_COLLAPSE_REPORT_TOKENS,
        })),
    );
    // Logged unconditionally, not just under debug. The collapse point of the
    // served weights is not reported by anything else, and without it the only
    // visible symptom is a turn that stops with the status bar reading 59%.
    //
    // Three numbers, no advice. Nothing here changes when compaction fires,
    // and nothing here proposes changing it: the threshold is already a cut
    // down from the window (85% of it, output reserve and buffer), and a single
    // collapse is not grounds to cut it further. If an operator decides
    // otherwise, auto_compact_threshold_tokens is theirs.
    if prompt_tokens >= MIN_COLLAPSE_REPORT_TOKENS {
        crate::utils::log::log_error(
            &format!(
                "decoder collapsed at {prompt_tokens} prompt tokens \
                 (window {window}, auto-compaction fires at {threshold})"
            ),
            Some("compaction: collapse observed"),
        );
    }
}

fn auto_compact_threshold() -> i64 {
    auto_compact_threshold_for(false)
}

/// The threshold for one tenant of the server's KV pool.
///
/// `is_subagent` matters because the two tenants are not interchangeable. The
/// parent is the conversation the user is having and must be allowed the whole
/// window; the sub-agent is transient and only has to fit in the space the
/// parent leaves. Capping the parent to make room would be paying for the
/// sub-agent with the thing the user actually wanted.
fn auto_compact_threshold_for(is_subagent: bool) -> i64 {
    // Primarily a PERCENTAGE of the context window (AUTOCOMPACT_FRACTION), so
    // it scales with the model's size instead of a fixed absolute reserve. But
    // never past the point where the prompt plus a full max_tokens response
    // would overflow the model — otherwise a large input slips under a high
    // percentage threshold while still 400-ing the server
    // (458753 + 65536 > 524288). So take the smaller of the two.
    let percent_threshold = (get_context_size() as f64 * AUTOCOMPACT_FRACTION) as i64;
    let overflow_safe = get_effective_context_window_size() - AUTOCOMPACT_BUFFER_TOKENS;
    let derived = percent_threshold.min(overflow_safe);
    // An observed collapse does NOT move this number, deliberately.
    //
    // It briefly did, and that was wrong. On 2026-10-01 the decoder collapsed
    // at 264,197 on a 524,288 window; deriving a ceiling from that one
    // observation took the threshold from 445,752 (85% of the window — output
    // reserve and buffer, which is all it should ever be) down to 211,357, or
    // 40%. That is 234,395 tokens of a window the operator is paying for,
    // discarded on the strength of a single sample, with compaction then firing
    // more than twice as often — summarising healthy conversations away and
    // spending a model call each time. A collapse is evidence about one
    // history, not a measurement of the window.
    //
    // note_decoder_collapse still records and logs the collapse, with the
    // ceiling it WOULD imply, because that is the number nobody had. Acting on
    // it belongs to the operator, through auto_compact_threshold_tokens
    // directly below — which is what that knob is for.
    let cfg = get_config();
    // Operator ceiling over EVERY tenant (rarely needed; 0 = off). "Every"
    // used to mean two; on this line it is the parent plus
    // max_concurrent_subagents.
    let mut threshold = if cfg.auto_compact_threshold_tokens > 0 {
        derived.min(cfg.auto_compact_threshold_tokens)
    } else {
        derived
    };
    if is_subagent {
        // Explicit operator value wins; 0 (the default) derives one from the
        // parent's threshold AFTER the operator ceiling, so the sub-agent is a
        // share of what the parent actually gets, not of the raw window.
        let sub_cap = if cfg.subagent_context_ceiling_tokens > 0 {
            cfg.subagent_context_ceiling_tokens
        } else {
            // Two measurements, whichever binds first: what one extra tenant
            // may cost, and what all the concurrent ones may cost together.
            // The second is what makes raising the slot count safe — without
            // it the per-agent share is simply multiplied by the slots and
            // nothing says so.
            let slots = cfg.max_concurrent_subagents.max(1) as f64;
            let share = SUBAGENT_THRESHOLD_FRACTION.min(SUBAGENT_TOTAL_FRACTION / slots);
            (threshold as f64 * share) as i64
        };
        threshold = threshold.min(sub_cap);
    }
    threshold
}

/// getAutoCompactThreshold: effective window minus the autocompact buffer.
/// Always Some in this substrate; the Option shape is kept for the display
/// fallback path (formatting falls back to get_context_size on None, the
/// Python try/except equivalent).
pub fn get_auto_compact_threshold() -> Option<i64> {
    Some(auto_compact_threshold())
}

/// Port of calculateTokenWarningState (auto-compact enabled path).
#[derive(Debug, Clone)]
pub struct TokenWarningState {
    pub percent_left: i64,
    pub is_above_warning_threshold: bool,
    pub is_above_error_threshold: bool,
    pub is_above_auto_compact_threshold: bool,
    pub is_at_blocking_limit: bool,
}

pub fn calculate_token_warning_state(token_usage: i64) -> TokenWarningState {
    let threshold = auto_compact_threshold();
    let percent_left = if threshold > 0 {
        std::cmp::max(
            0,
            (((threshold - token_usage) as f64 / threshold as f64) * 100.0).round() as i64,
        )
    } else {
        0
    };
    let warning_threshold = threshold - WARNING_THRESHOLD_BUFFER_TOKENS;
    let error_threshold = threshold - ERROR_THRESHOLD_BUFFER_TOKENS;
    let effective = get_effective_context_window_size();
    let blocking_limit = effective - MANUAL_COMPACT_BUFFER_TOKENS;
    TokenWarningState {
        percent_left,
        is_above_warning_threshold: token_usage >= warning_threshold,
        is_above_error_threshold: token_usage >= error_threshold,
        is_above_auto_compact_threshold: token_usage >= threshold,
        is_at_blocking_limit: token_usage >= blocking_limit,
    }
}

pub fn is_auto_compact_enabled() -> bool {
    get_config().auto_compact
}

/// Port of shouldAutoCompact. Counts via the unified count_text_tokens
/// substrate — the served model's own tokenizer when available (exact), else
/// the model tokenizer (or its CJK-aware heuristic) — never raw chars/4.
pub fn should_auto_compact(messages: &[Message], scope: &str) -> bool {
    should_auto_compact_for(messages, false, scope)
}

/// The out-of-band scope for a context's tool set (see tokens::oob_scope_key).
pub fn oob_scope(tool_use_context: &ToolUseContext) -> String {
    crate::utils::tokens::oob_scope_key(
        tool_use_context.options.tools.iter().map(|t| t.name()),
    )
}

/// `should_auto_compact` for one tenant — see auto_compact_threshold_for.
pub fn should_auto_compact_for(messages: &[Message], is_subagent: bool, scope: &str) -> bool {
    if !is_auto_compact_enabled() {
        return false;
    }
    // tokenCountWithEstimation: last API usage + estimate of everything after
    // it, so the count grows mid-tool-loop and compaction still fires.
    // estimate_input_tokens, NOT token_count_with_estimation: this number is
    // compared against a threshold derived from the model's context window, so
    // it must be the size of the REQUEST (history + system prompt + tool
    // block), not of the history alone. The narrower count lets a large input
    // sit under the threshold and then be rejected by the server, and the
    // output cap downstream cannot save it once the INPUT alone is over.
    let token_count = crate::utils::tokens::estimate_input_tokens(messages, scope);
    let threshold = auto_compact_threshold_for(is_subagent);
    debug_log(
        "AUTOCOMPACT_CHECK",
        Some(&json!({
            "tokens": token_count,
            "threshold": threshold,
            "effective_window": get_effective_context_window_size(),
        })),
    );
    token_count >= threshold
}

// === Microcompact: content-clear old tool results (microCompact.ts core) ===
//
// Faithful port of maybeTimeBasedMicrocompact's CLEARING mechanism: keep the
// last MICROCOMPACT_KEEP_RECENT compactable tool results, and replace the
// CONTENT of older ones with a marker (structure/pairing preserved). This
// relieves context pressure WITHOUT a destructive full summary.
//
// One deliberate deviation (flagged): upstream TRIGGERS this on an idle
// time-gap (server cache cold, measured from message timestamps). codemason
// messages carry no timestamp, so the caller runs this under TOKEN pressure
// instead (before the full autocompact). The ant-only cached-microcompact and
// the HISTORY_SNIP path are NOT ported.

pub const MICROCOMPACT_KEEP_RECENT: usize = 5;
pub const TIME_BASED_MC_CLEARED_MESSAGE: &str = "[Old tool result content cleared]";
/// COMPACTABLE_TOOLS (microCompact.ts): output-heavy read-only tools whose
/// results are safe to drop once old.
const COMPACTABLE_TOOLS: &[&str] =
    &["Read", "Bash", "Grep", "Glob", "WebSearch", "WebFetch", "Edit", "Write"];

/// collectCompactableToolIds: tool_use ids of compactable tools, in order.
fn collect_compactable_tool_ids(messages: &[Message]) -> Vec<String> {
    let mut ids = Vec::new();
    for m in messages {
        let Message::Assistant(a) = m else {
            continue;
        };
        for b in &a.message.content {
            if let crate::types::messages::ContentBlock::ToolUse(tu) = b {
                if COMPACTABLE_TOOLS.contains(&tu.name.as_str()) && !tu.id.is_empty() {
                    ids.push(tu.id.clone());
                }
            }
        }
    }
    ids
}

/// calculateToolResultTokens: rough token estimate of a tool_result block.
fn tool_result_tokens(content: &ToolResultContent) -> i64 {
    match content {
        ToolResultContent::Text(s) => {
            if s.is_empty() {
                0
            } else {
                rough_token_count_estimation(s, 4)
            }
        }
        ToolResultContent::Blocks(items) => {
            let mut total = 0;
            for it in items {
                match it.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        total += rough_token_count_estimation(
                            it.get("text").and_then(|t| t.as_str()).unwrap_or(""),
                            4,
                        )
                    }
                    Some("image") => {
                        total += it
                            .get("source")
                            .map(crate::utils::image_prep::image_source_tokens)
                            .unwrap_or(crate::utils::image_prep::MAX_TOKENS_PER_IMAGE)
                            as i64
                    }
                    Some("document") => total += 2000,
                    _ => {}
                }
            }
            total
        }
    }
}

fn is_image_item(v: &serde_json::Value) -> bool {
    v.get("type").and_then(|t| t.as_str()) == Some("image")
}

/// A tool result microcompact already cleared: the marker alone, or the marker
/// followed by the images it kept.
fn is_cleared(content: &ToolResultContent) -> bool {
    match content {
        ToolResultContent::Text(s) => s == TIME_BASED_MC_CLEARED_MESSAGE,
        ToolResultContent::Blocks(items) => {
            items.first().and_then(|f| f.get("text")).and_then(|t| t.as_str())
                == Some(TIME_BASED_MC_CLEARED_MESSAGE)
                && items.first().and_then(|f| f.get("type")).and_then(|t| t.as_str()) == Some("text")
                && items[1..].iter().all(is_image_item)
        }
    }
}

/// Clear the content of old compactable tool results (keep the last
/// MICROCOMPACT_KEEP_RECENT). Returns (new_messages, tokens_freed);
/// (messages, 0) when nothing is cleared.
pub fn microcompact_messages(messages: &[Message]) -> (Vec<Message>, i64) {
    let compactable = collect_compactable_tool_ids(messages);
    let keep_recent = std::cmp::max(1, MICROCOMPACT_KEEP_RECENT);
    let keep: std::collections::HashSet<&String> =
        compactable.iter().rev().take(keep_recent).collect();
    let clear: std::collections::HashSet<&String> =
        compactable.iter().filter(|i| !keep.contains(i)).collect();
    if clear.is_empty() {
        return (messages.to_vec(), 0);
    }

    let mut tokens_freed: i64 = 0;
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for m in messages {
        let Message::User(u) = m else {
            out.push(m.clone());
            continue;
        };
        let MessageContent::Blocks(blocks) = &u.message.content else {
            out.push(m.clone());
            continue;
        };
        let mut touched = false;
        let mut new_content: Vec<ContentBlockParam> = Vec::with_capacity(blocks.len());
        for b in blocks {
            match b {
                ContentBlockParam::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    cache_control,
                    data,
                } if clear.contains(tool_use_id) && !is_cleared(content) => {
                    // [ours] Image blocks stay. What they cost is already bounded
                    // by the image limit (the oldest go out as one-line labels —
                    // openai_common::images_to_evict), and taking them out here
                    // would renumber every later [Image N] the model has notes
                    // on and drop the saved PATH their labels give. So only the
                    // text is cleared, and only the text counts as freed.
                    let images: Vec<serde_json::Value> = match content {
                        ToolResultContent::Blocks(items) => items.iter().filter(|x| is_image_item(x)).cloned().collect(),
                        ToolResultContent::Text(_) => Vec::new(),
                    };
                    let image_tokens: i64 = images
                        .iter()
                        .map(|i| {
                            i.get("source")
                                .map(crate::utils::image_prep::image_source_tokens)
                                .unwrap_or(crate::utils::image_prep::MAX_TOKENS_PER_IMAGE) as i64
                        })
                        .sum();
                    tokens_freed += tool_result_tokens(content) - image_tokens;
                    touched = true;
                    let cleared = if images.is_empty() {
                        ToolResultContent::Text(TIME_BASED_MC_CLEARED_MESSAGE.into())
                    } else {
                        let mut items = vec![json!({"type": "text", "text": TIME_BASED_MC_CLEARED_MESSAGE})];
                        items.extend(images);
                        ToolResultContent::Blocks(items)
                    };
                    new_content.push(ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: cleared,
                        is_error: *is_error,
                        cache_control: cache_control.clone(),
                        data: data.clone(),
                    });
                }
                other => new_content.push(other.clone()),
            }
        }
        if !touched {
            out.push(m.clone());
            continue;
        }
        let mut nu = u.clone();
        nu.message.content = MessageContent::Blocks(new_content);
        out.push(Message::User(nu));
    }

    if tokens_freed == 0 {
        return (messages.to_vec(), 0);
    }
    (out, tokens_freed)
}

// === Summary prompt (prompt.ts — verbatim) ===

pub const NO_TOOLS_PREAMBLE: &str = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.\n- You already have all the context you need in the conversation above.\n- Tool calls will be REJECTED and will waste your only turn — you will fail the task.\n- Your entire response must be plain text: an <analysis> block followed by a <summary> block.\n\n";

pub const NO_TOOLS_TRAILER: &str = "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task.";

const DETAILED_ANALYSIS_INSTRUCTION_BASE: &str = "Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:\n\n1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:\n   - The user's explicit requests and intents\n   - Your approach to addressing the user's requests\n   - Key decisions, technical concepts and code patterns\n   - Specific details like:\n     - file names\n     - full code snippets\n     - function signatures\n     - file edits\n   - Errors that you ran into and how you fixed them\n   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.";

pub static BASE_COMPACT_PROMPT: Lazy<String> = Lazy::new(|| {
    format!(
        "Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.\nThis summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.\n\n{DETAILED_ANALYSIS_INSTRUCTION_BASE}\n\nYour summary should include the following sections:\n\n1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail\n2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.\n3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.\n4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.\n6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.\n7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.\n8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.\n9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.\n                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.\n\nHere's an example of how your output should be structured:\n\n<example>\n<analysis>\n[Your thought process, ensuring all points are covered thoroughly and accurately]\n</analysis>\n\n<summary>\n1. Primary Request and Intent:\n   [Detailed description]\n\n2. Key Technical Concepts:\n   - [Concept 1]\n   - [Concept 2]\n   - [...]\n\n3. Files and Code Sections:\n   - [File Name 1]\n      - [Summary of why this file is important]\n      - [Summary of the changes made to this file, if any]\n      - [Important Code Snippet]\n   - [File Name 2]\n      - [Important Code Snippet]\n   - [...]\n\n4. Errors and fixes:\n    - [Detailed description of error 1]:\n      - [How you fixed the error]\n      - [User feedback on the error if any]\n    - [...]\n\n5. Problem Solving:\n   [Description of solved problems and ongoing troubleshooting]\n\n6. All user messages:\n    - [Detailed non tool use user message]\n    - [...]\n\n7. Pending Tasks:\n   - [Task 1]\n   - [Task 2]\n   - [...]\n\n8. Current Work:\n   [Precise description of current work]\n\n9. Optional Next Step:\n   [Optional Next step to take]\n\n</summary>\n</example>\n\nPlease provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.\n\nThere may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:\n<example>\n## Compact Instructions\nWhen summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.\n</example>\n\n<example>\n# Summary instructions\nWhen you are using compact - please focus on test output and code changes. Include file reads verbatim.\n</example>\n"
    )
});

/// The language the user has been writing in, from the messages themselves.
///
/// Read here rather than from the request-language set during streaming: by
/// the time compaction runs, the last thing detected may be a tool result or
/// an English summary, and after the FIRST compaction the only user message
/// left is the previous summary. Scanning back over the real user turns is
/// what survives repeated compaction.
fn conversation_language(messages: &[Message]) -> Option<String> {
    for m in messages.iter().rev() {
        let Message::User(u) = m else {
            continue;
        };
        let text = match &u.message.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => {
                // Tool results are the tool's words, not the user's — a page of
                // English stdout must not decide what language the user speaks.
                blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlockParam::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .concat()
            }
        };
        if let Some(lang) = crate::utils::language::detect_language(&text) {
            return Some(lang.to_string());
        }
    }
    // Latin script carries no signal, so fall back to the configured language —
    // the same precedence effective_language() uses (conversation first, config
    // as the fallback for "always answer in X").
    get_config().language.filter(|l| !l.is_empty())
}

pub fn get_compact_prompt(custom_instructions: Option<&str>, language: Option<&str>) -> String {
    let mut prompt = format!("{NO_TOOLS_PREAMBLE}{}", &*BASE_COMPACT_PROMPT);
    // The summary does not describe the conversation, it REPLACES it. Written
    // in English it makes the whole history English, and the model goes on in
    // English for the rest of the session — with nothing left to signal
    // otherwise, since the user's own words are exactly what was just
    // discarded. Every later compaction summarises that English, so it never
    // comes back.
    if let Some(language) = language {
        prompt.push_str(&format!(
            "\n\nWrite the summary in {language}. The summary replaces the conversation itself, and the conversation is in {language}; an English summary would silently switch the language of everything that follows. Keep code, identifiers, paths, commands and error strings verbatim."
        ));
    }
    if let Some(ci) = custom_instructions {
        if !ci.trim().is_empty() {
            prompt.push_str(&format!("\n\nAdditional Instructions:\n{ci}"));
        }
    }
    prompt.push_str(NO_TOOLS_TRAILER);
    prompt
}

static RE_ANALYSIS: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"<analysis>[\s\S]*?</analysis>").unwrap());
static RE_SUMMARY: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"<summary>([\s\S]*?)</summary>").unwrap());
static RE_MULTI_NL: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\n\n+").unwrap());

/// Strip the <analysis> scratchpad and turn <summary> into a header.
pub fn format_compact_summary(summary: &str) -> String {
    let formatted = RE_ANALYSIS.replace_all(summary, "").into_owned();
    let formatted = if let Some(m) = RE_SUMMARY.captures(&formatted) {
        let content = m.get(1).map(|g| g.as_str()).unwrap_or("").trim().to_string();
        // NoExpand: `content` is model-authored text and may hold $-sequences
        // the replacer would otherwise parse (the Python bad-escape trap).
        // count=1 matches JS .replace(str,str).
        RE_SUMMARY
            .replacen(&formatted, 1, regex::NoExpand(&format!("Summary:\n{content}")))
            .into_owned()
    } else {
        formatted
    };
    RE_MULTI_NL.replace_all(&formatted, "\n\n").trim().to_string()
}

/// Opening sentence of every compaction summary.
///
/// Also what `session_storage::trim_to_last_compaction` looks for: a resume
/// must start where the last compaction did, and the transcript records no
/// flag for it. Shared rather than copied, so the two cannot drift apart —
/// a changed wording here would otherwise make every later session replay
/// history the compaction had already thrown away.
pub const COMPACT_CONTINUATION_PREFIX: &str =
    "This session is being continued from a previous conversation that ran out of context.";

pub fn get_compact_user_summary_message(summary: &str, suppress_follow_up_questions: bool) -> String {
    let formatted = format_compact_summary(summary);
    let base = format!(
        "{COMPACT_CONTINUATION_PREFIX} The summary below covers the earlier portion of the conversation.\n\n{formatted}"
    );
    if suppress_follow_up_questions {
        format!(
            "{base}\nContinue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with \"I'll continue\" or similar. Pick up the last task as if the break never happened."
        )
    } else {
        base
    }
}

// === Compaction execution (core of compact.ts compactConversation) ===

pub const COMPACT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant tasked with summarizing conversations.";

pub const API_ERROR_MESSAGE_PREFIX: &str = "Error";

/// Floor for a summary we are willing to REPLACE the conversation with.
///
/// Deliberately low: its only job is to catch text that formats to nothing
/// (whitespace-only is not `is_empty()`, and format_compact_summary trims it to
/// ""). The substantive rejections are made on stop_reason, because the strings
/// that must not be accepted — the stall note, a partial abort — are long enough
/// to clear any length bar worth setting.
const MIN_USABLE_SUMMARY_CHARS: usize = 40;

#[derive(Clone)]
pub struct CompactionResult {
    pub summary: String,
    /// post-compact messages (replace the conversation)
    pub messages: Vec<Message>,
    pub summary_response: Option<AssistantMessage>,
}

/// The first TEXT block, wherever it sits. Looking only at `first()` made a
/// summary that arrived behind a thinking or tool_use block read as "no
/// summary" and fail the compaction outright.
fn extract_text(response: &AssistantMessage) -> Option<String> {
    response.message.content.iter().find_map(|b| match b {
        crate::types::messages::ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    })
}

/// Port of stripImagesFromMessages: the summary request carries no images
/// (services/compaction.py strip_images_for_summary).
///
/// The summarizer writes text, the images are already in the store, and with
/// them the largest request a session sends would also carry up to the
/// server's image limit in pixels (each up to VISION_TOKENS_PER_IMAGE_MAX
/// tokens). Each image becomes "[Image N: saved at PATH]", numbered in the
/// order convert_to_chat_messages numbers them, so the summary can keep the
/// path and the model can Read the image again after compaction. Returns the
/// request copy; `messages` is not modified.
pub fn strip_images_for_summary(messages: &[Message]) -> Vec<Message> {
    let mut n = 0usize;
    let mut note = |source: Option<&serde_json::Value>, numbered: bool| -> String {
        let place = source
            .and_then(|s| crate::utils::image_store::stored_path(s, None))
            .map(|p| format!(": saved at {}", p.display()))
            .unwrap_or_default();
        if !numbered {
            return format!("[Image{place}]");
        }
        n += 1;
        format!("[Image {n}{place}]")
    };
    let mut seen_results: std::collections::HashSet<String> = Default::default();
    messages
        .iter()
        .map(|msg| {
            let Message::User(u) = msg else { return msg.clone() };
            let MessageContent::Blocks(blocks) = &u.message.content else { return msg.clone() };
            // Numbered in convert_to_chat_messages' order: in a message with
            // tool results, the results' images first (a repeated result is
            // skipped there, so it gets no number here), then images beside
            // them. Notes are keyed by (block index, item index).
            let has_results = blocks.iter().any(|b| matches!(b, ContentBlockParam::ToolResult { .. }));
            let mut order: Vec<usize> = (0..blocks.len()).collect();
            if has_results {
                order.sort_by_key(|&i| !matches!(blocks[i], ContentBlockParam::ToolResult { .. }));
            }
            let mut notes: std::collections::HashMap<(usize, usize), String> = Default::default();
            for i in order {
                match &blocks[i] {
                    ContentBlockParam::Image { source, .. } => {
                        notes.insert((i, 0), note(Some(source), true));
                    }
                    ContentBlockParam::ToolResult { tool_use_id, content, .. } => {
                        let first = seen_results.insert(tool_use_id.clone());
                        if let ToolResultContent::Blocks(items) = content {
                            for (j, x) in items.iter().enumerate() {
                                if is_image_item(x) {
                                    notes.insert((i, j), note(x.get("source"), first));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if notes.is_empty() {
                return msg.clone();
            }
            let blocks = blocks
                .iter()
                .enumerate()
                .map(|(i, b)| match b {
                    ContentBlockParam::Image { .. } => ContentBlockParam::text(notes[&(i, 0)].clone()),
                    ContentBlockParam::ToolResult {
                        tool_use_id,
                        content: ToolResultContent::Blocks(items),
                        is_error,
                        cache_control,
                        data,
                    } if items.iter().any(is_image_item) => ContentBlockParam::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: ToolResultContent::Blocks(
                            items
                                .iter()
                                .enumerate()
                                .map(|(j, x)| match notes.get(&(i, j)) {
                                    Some(t) => json!({"type": "text", "text": t}),
                                    None => x.clone(),
                                })
                                .collect(),
                        ),
                        is_error: *is_error,
                        cache_control: cache_control.clone(),
                        data: data.clone(),
                    },
                    other => other.clone(),
                })
                .collect();
            let mut u = u.clone();
            u.message.content = MessageContent::Blocks(blocks);
            Message::User(u)
        })
        .collect()
}

/// Query the model for a summary and build the post-compact message list.
///
/// The callbacks drive the caller's spinner. Without them this is the longest
/// silent wait in a session — the whole conversation is re-prefilled and
/// summarised with nothing on screen, which is indistinguishable from a hang,
/// and the bigger the context the longer it looks frozen.
pub async fn compact_conversation(
    messages: &[Message],
    tool_use_context: &ToolUseContext,
    custom_instructions: Option<&str>,
    is_auto_compact: bool,
    on_token_update: Option<std::sync::Arc<dyn Fn(i64) + Send + Sync>>,
    on_thinking_update: Option<std::sync::Arc<dyn Fn(i64) + Send + Sync>>,
) -> Result<CompactionResult, String> {
    use crate::services::backend::query_model;
    use crate::services::openai_common::{QueryModelParams, QueryOptions};

    let lang = conversation_language(messages);
    let summary_request = Message::User(create_user_message_text(&get_compact_prompt(
        custom_instructions,
        lang.as_deref(),
    )));

    let signal = tool_use_context.abort_controller.clone();
    let model = tool_use_context.options.slow_and_capable_model.clone();

    // Cap the summary output at the reserved budget
    // (get_effective_context_window_size already subtracts
    // MAX_OUTPUT_TOKENS_FOR_SUMMARY on the assumption the summary won't exceed
    // it — enforce that here).
    let summary_output_cap = std::cmp::min(MAX_OUTPUT_TOKENS_FOR_SUMMARY, get_config().max_tokens);

    // PTL retry loop: if the summary request overflows context, truncate the
    // oldest rounds and retry (up to MAX_PTL_RETRIES) instead of failing
    // straight into the circuit breaker with the session stuck over-limit.
    let mut conv: Vec<Message> = strip_images_for_summary(messages);
    let mut attempt: i64 = 0;
    loop {
        let mut request_messages = conv.clone();
        request_messages.push(summary_request.clone());
        let response = query_model(QueryModelParams {
            messages: request_messages,
            system_prompt: vec![COMPACT_SYSTEM_PROMPT.to_string()],
            tools: vec![], // NO tools during summarization
            signal: Some(signal.clone()),
            // no_reasoning: summarising is not a problem to deliberate over,
            // and the reasoning pass runs before a single summary token
            // appears — doubling the one wait the user is already staring at.
            // The effort level is a setting for the work, not for the
            // bookkeeping around it.
            //
            // keep_thinking stays FALSE here, unlike the recovery paths that
            // also set no_reasoning: this request's whole purpose is to make
            // the history smaller, so it must not carry every earlier turn's
            // reasoning into itself.
            options: QueryOptions {
                dangerously_skip_permissions: false,
                model: model.clone(),
                max_tokens: Some(summary_output_cap),
                no_reasoning: true,
                keep_thinking: false,
            },
            on_token_update: on_token_update.clone(),
            on_content_delta: None,
            on_thinking_update: on_thinking_update.clone(),
        })
        .await;

        let summary = extract_text(&response);
        // Accepting a summary REPLACES the entire conversation, so a wrong
        // accept is unrecoverable. The old gate was "non-empty and does not
        // start with Error", which also accepts:
        //   - the stall guard's "\n\n[Stopped: ...]" text (stop_reason
        //     "stalled"). Very reachable here: this is the largest request the
        //     session ever sends, and it runs on a cold prefix.
        //   - ESC during "Compacting…" — it shares the turn's abort token and
        //     yields whatever partial text arrived as a clean end_turn.
        //   - a summary cut at the output cap ("max_tokens"), which has no
        //     </analysis> and no <summary>, so the raw SCRATCHPAD would become
        //     the new history.
        //   - whitespace-only text: not is_empty(), formats to "".
        // A rejected summary falls through to the PTL retry (which must keep
        // seeing the raw text) and then to Err — the pre-existing failure path.
        let stop = response.message.stop_reason.as_deref();
        let usable_stop = !matches!(
            stop,
            Some("stalled") | Some("max_tokens") | Some("empty_turn") | Some("repetition")
        ) && !response.is_api_error_message;
        if let Some(s) = &summary {
            if usable_stop
                && !s.starts_with(API_ERROR_MESSAGE_PREFIX)
                && format_compact_summary(s).chars().count() >= MIN_USABLE_SUMMARY_CHARS
            {
                let user_summary = get_compact_user_summary_message(s, is_auto_compact);
                let post_compact_messages =
                    vec![Message::User(create_user_message_text(&user_summary))];
                return Ok(CompactionResult {
                    summary: s.clone(),
                    messages: post_compact_messages,
                    summary_response: Some(response),
                });
            }
        }
        if attempt < MAX_PTL_RETRIES {
            if let Some(s) = &summary {
                if !s.is_empty() && looks_like_prompt_too_long(s) {
                    if let Some(truncated) = truncate_head_for_ptl(&conv, Some(s)) {
                        conv = truncated;
                        attempt += 1;
                        debug_log(
                            "COMPACT_PTL_RETRY",
                            Some(&json!({"attempt": attempt, "kept_messages": conv.len()})),
                        );
                        continue;
                    }
                }
            }
        }
        // Name the reason, and put it on disk. Compaction failure used to be
        // entirely silent (the caller prints nothing when was_compacted is
        // false), so three of these in a row tripped the circuit breaker with
        // no record anywhere of what went wrong.
        let why = format!(
            "compaction rejected the summary (stop_reason {}, {} chars){}",
            stop.unwrap_or("none"),
            summary.as_deref().map(|s| s.chars().count()).unwrap_or(0),
            if response.is_api_error_message { ", server error" } else { "" }
        );
        crate::utils::log::log_error(&why, Some("compaction: no usable summary"));
        return Err(format!("Compaction failed: {why}"));
    }
}

/// Result of auto_compact_if_needed (the Python return dict, typed).
#[derive(Default)]
pub struct AutoCompactOutcome {
    pub was_compacted: bool,
    pub was_microcompacted: bool,
    /// The microcompacted message set (when was_microcompacted).
    pub messages: Option<Vec<Message>>,
    pub compaction_result: Option<CompactionResult>,
    /// Present when the attempt ran: 0 on success, prev+1 on failure.
    pub consecutive_failures: Option<i64>,
}

/// Port of autoCompactIfNeeded with the consecutive-failure circuit breaker.
///
/// Callbacks are forwarded to compact_conversation for the caller's spinner —
/// auto-compaction needs them more than the manual command does, since it
/// happens without being asked and its silence has no obvious cause at all.
/// `context_proven_fatal` — the caller has just watched the decoder collapse on
/// this exact history (services::query's repetition recovery, and nothing else).
///
/// It exists because the threshold is the wrong authority in that one case. The
/// threshold predicts whether a history is too big; a collapse is the
/// observation that it was, and a prediction does not get to overrule the
/// measurement it was predicting. Without this the recovery path asked
/// should_auto_compact for permission to shrink a context that had just killed
/// the turn, was told there was room, and re-sent it unchanged — seven times,
/// 94.5 seconds, prompt moving 264,197 → 264,506 (0.12%), every attempt
/// collapsing in the same place (think=26, out=200 '!'). Measured 2026-10-01.
///
/// Under it ONLY microcompact runs — the free, local, non-destructive one. The
/// full summary is deliberately not reachable this way: its own request carries
/// the same history that was just proven fatal, so it would likely collapse
/// too, and three failed summaries trip a one-way breaker that ends compaction
/// for the rest of the session. Reducing the context must not be able to
/// destroy the ability to reduce the context. When microcompact is exhausted,
/// the measured ceiling (note_decoder_collapse) is what lowers the threshold so
/// the ordinary path — full summary included — runs BELOW the collapse line
/// instead of on top of it.
pub async fn auto_compact_if_needed(
    messages: &[Message],
    tool_use_context: &ToolUseContext,
    tracking_consecutive_failures: i64,
    on_token_update: Option<std::sync::Arc<dyn Fn(i64) + Send + Sync>>,
    on_thinking_update: Option<std::sync::Arc<dyn Fn(i64) + Send + Sync>>,
    context_proven_fatal: bool,
) -> AutoCompactOutcome {
    let consecutive = tracking_consecutive_failures;
    if consecutive >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES {
        // The breaker is one-way: only a SUCCESSFUL compaction resets the
        // counter, and once it is tripped no compaction is attempted, so the
        // reset can never happen. Every later turn then sends the full,
        // still-growing history until the server rejects it. Nothing recorded
        // that this had happened, which is why an over-context session looked
        // like a model problem. Log it once per turn it suppresses, at the one
        // place that knows — the caller only sees an empty outcome.
        crate::utils::log::log_error(
            &format!(
                "auto-compaction suppressed: {consecutive} consecutive failures \
                 (limit {MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES}) — the history is no longer \
                 being reduced and only /clear or a successful /compact recovers"
            ),
            Some("compaction: breaker open"),
        );
        return AutoCompactOutcome::default();
    }

    // A sub-agent (agent_id set by the Agent tool / forked skills; None for the
    // user's own conversation) gets the tighter ceiling.
    let is_subagent = tool_use_context.agent_id.is_some();
    let scope = oob_scope(tool_use_context);
    if !should_auto_compact_for(messages, is_subagent, &scope) {
        if !context_proven_fatal {
            return AutoCompactOutcome::default();
        }
        // Proven fatal but under the threshold. Microcompact unconditionally —
        // see the note on context_proven_fatal for why this is the only step
        // taken here, and why it is taken without asking.
        //
        // Not gated on the threshold, but still gated on there being something
        // to clear: microcompact is idempotent (an already-cleared result is
        // skipped), so a history whose old tool results are all cleared frees
        // 0, and returning "microcompacted" for a set that did not change would
        // report progress the caller then recurses on.
        let (cleared, freed) = microcompact_messages(messages);
        debug_log(
            "MICROCOMPACT_PROVEN_FATAL",
            Some(&json!({
                "tokens_freed": freed,
                "threshold": auto_compact_threshold_for(is_subagent),
                "exhausted": freed == 0,
            })),
        );
        if freed == 0 {
            // Nothing left to clear locally. Not escalated to a full summary
            // here (see the note above); the lowered threshold reached via
            // note_decoder_collapse is what routes this history into the
            // ordinary path on the next turn.
            crate::utils::log::log_error(
                &"the decoder collapsed on this history and microcompact had nothing left to \
                  clear — context was not reduced before the retry",
                Some("compaction: proven-fatal, exhausted"),
            );
            return AutoCompactOutcome::default();
        }
        return AutoCompactOutcome {
            was_compacted: false,
            was_microcompacted: true,
            messages: Some(cleared),
            ..Default::default()
        };
    }

    // Name the wait, for BOTH callers. The turn-boundary path (cli_repl.rs)
    // pins "Compacting" on its own spinner; the mid-turn path
    // (query.rs compact_before_recursion) has no spinner of its own, so a
    // reduction in the middle of a tool loop was reported with whatever label
    // the turn was already wearing — "analysing servd" while the conversation
    // was being reduced. Setting it HERE, past the threshold check, is what
    // makes both paths honest without either of them knowing about a spinner.
    //
    // WAIT_NOTE is a leaf lock: set_wait_note assigns and releases,
    // wait_note() clones and releases, and neither holds it while acquiring
    // anything else — so it cannot be one side of an inversion, which is the
    // only reason this is done with a note instead of a second spinner. The
    // guard clears it on every exit below, including the early returns.
    crate::utils::tokens::set_wait_note(Some("compacting conversation".into()));
    let _wait_note = crate::utils::tokens::WaitNoteGuard;

    // Microcompact first: content-clear old tool results (keep the recent
    // tail) and see if that alone drops us under the threshold. If so, skip
    // the destructive full summary and keep granular context (microCompact.ts
    // intent: "if collapse gets us under the autocompact threshold,
    // autocompact is a no-op"). Old results are pre-anchor, so the usage-
    // anchored count can't see the saving — subtract tokens_freed explicitly
    // (upstream's snipTokensFreed plumbing).
    let mut messages_ref: Vec<Message> = messages.to_vec();
    let (cleared, freed) = microcompact_messages(&messages_ref);
    if freed > 0 {
        let threshold = auto_compact_threshold_for(is_subagent);
        // Same count the trigger used (estimate_input_tokens, the request
        // size), so "relieved" is judged with the ruler that said "over".
        let effective =
            crate::utils::tokens::estimate_input_tokens(&messages_ref, &scope) - freed;
        debug_log(
            "MICROCOMPACT",
            Some(&json!({
                "tokens_freed": freed, "effective": effective, "threshold": threshold,
                "relieved": effective < threshold,
            })),
        );
        if effective < threshold {
            return AutoCompactOutcome {
                was_compacted: false,
                was_microcompacted: true,
                messages: Some(cleared),
                ..Default::default()
            };
        }
        messages_ref = cleared; // still over -> full-summarize the smaller, cleared set
    }

    // Only around the SUMMARY. Microcompaction returns above in microseconds,
    // so pinning a label for it would flash something nobody can read; the wait
    // note set earlier already covers that case. A summary is an API round trip
    // on the whole conversation -- the longest wait in a session -- and that is
    // the one worth renaming the line for.
    if let Some(on_status) = &tool_use_context.on_status {
        on_status(Some("Compacting".into()));
    }
    let summary = compact_conversation(
        &messages_ref,
        tool_use_context,
        None,
        true,
        on_token_update,
        on_thinking_update,
    )
    .await;
    // Hand the label back before branching, so every arm below -- including the
    // failure arms that return early -- leaves the turn wearing its own name.
    if let Some(on_status) = &tool_use_context.on_status {
        on_status(None);
    }
    match summary {
        Ok(mut result) => {
            // Post-compact file reconstruction (compact.ts): the summary does
            // NOT carry full file contents, so (1) CLEAR readFileState —
            // otherwise Edit/Write's read-before-edit gate would pass on a
            // file the model no longer has, a stale write — then (2) re-inject
            // the most recent files fresh and re-mark them read, so work
            // continues without a re-read round-trip.
            {
                let mut rfs = tool_use_context.read_file_state.lock();
                let pre = rfs.clone();
                rfs.clear();
                for (path, full, msg) in post_compact_file_attachments(&pre) {
                    result.messages.push(Message::User(msg));
                    if let Ok(meta) = std::fs::metadata(&path) {
                        let mtime = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0);
                        rfs.insert(
                            path,
                            ReadFileState {
                                timestamp: mtime,
                                content: full,
                                offset: None,
                                limit: None,
                                is_partial_view: false,
                            },
                        );
                    }
                }
            }
            // The summary does not carry the spawn either: without this the
            // model comes out of the compaction having forgotten its agents and
            // spawns them again. Appended after the file restoration, as the
            // source builds both attachment sets for the same post-compact
            // message list.
            for msg in create_async_agent_attachments_if_needed(tool_use_context.agent_id.as_deref()) {
                result.messages.push(Message::User(msg));
            }
            AutoCompactOutcome {
                was_compacted: true,
                compaction_result: Some(result),
                consecutive_failures: Some(0),
                ..Default::default()
            }
        }
        Err(e) => {
            debug_log("AUTOCOMPACT_FAILED", Some(&json!({"error": e})));
            AutoCompactOutcome {
                was_compacted: false,
                consecutive_failures: Some(consecutive + 1),
                ..Default::default()
            }
        }
    }
}

#[cfg(test)]
mod strip_images_tests {
    use super::*;
    use crate::types::messages::MessageParam;
    use serde_json::Value;

    const B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

    fn img() -> Value {
        json!({"type": "base64", "media_type": "image/png", "data": B64})
    }

    fn user(blocks: Vec<ContentBlockParam>) -> Message {
        Message::User(UserMessage {
            uuid: "u".into(),
            message: MessageParam { role: "user".into(), content: MessageContent::Blocks(blocks) },
            is_meta: false,
            tool_use_result: None,
        })
    }

    /// Same guards as tests/test_compaction_strip.py.
    #[test]
    fn summary_request_carries_no_images() {
        let msgs = vec![
            user(vec![ContentBlockParam::text("see"), ContentBlockParam::Image { source: img(), cache_control: None }]),
            user(vec![ContentBlockParam::ToolResult {
                tool_use_id: "r1".into(),
                content: ToolResultContent::Blocks(vec![
                    json!({"type": "text", "text": "Read /x.png"}),
                    json!({"type": "image", "source": img()}),
                ]),
                is_error: false,
                cache_control: None,
                data: None,
            }]),
            user(vec![ContentBlockParam::text("plain")]),
        ];
        let before = serde_json::to_string(&msgs).unwrap();
        let out = strip_images_for_summary(&msgs);
        let s = serde_json::to_string(&out).unwrap();
        assert!(!s.contains(B64), "{s}");
        assert_eq!(serde_json::to_string(&msgs).unwrap(), before, "caller messages untouched");
        let v: Vec<Value> = serde_json::from_str(&s).unwrap();
        let first = v[0]["message"]["content"][1]["text"].as_str().unwrap();
        let second = v[1]["message"]["content"][0]["content"][1]["text"].as_str().unwrap();
        assert!(first.starts_with("[Image 1: saved at ") && first.ends_with(".png]"), "{first}");
        assert!(second.starts_with("[Image 2: saved at "), "{second}");
        let path = &first["[Image 1: saved at ".len()..first.len() - 1];
        assert!(std::path::Path::new(path).is_file(), "{path}");
        assert_eq!(serde_json::to_string(&out[2]).unwrap(), serde_json::to_string(&msgs[2]).unwrap());
    }

    fn read_call(id: &str) -> Message {
        Message::Assistant(crate::types::query::AssistantMessage {
            uuid: String::new(),
            message: crate::types::messages::ApiMessage {
                content: vec![crate::types::messages::ContentBlock::ToolUse(
                    crate::types::messages::ToolUseBlock { id: id.into(), name: "Read".into(), input: json!({}) },
                )],
                model: "m".into(),
                ..Default::default()
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    fn result(id: &str, items: Vec<Value>) -> ContentBlockParam {
        ContentBlockParam::ToolResult {
            tool_use_id: id.into(),
            content: ToolResultContent::Blocks(items),
            is_error: false,
            cache_control: None,
            data: None,
        }
    }

    /// tests/test_vision_followups.py test_microcompact_keeps_images.
    #[test]
    fn microcompact_keeps_images_and_numbers() {
        let mut msgs = Vec::new();
        for i in 0..MICROCOMPACT_KEEP_RECENT + 2 {
            msgs.push(read_call(&format!("r{i}")));
            msgs.push(user(vec![result(&format!("r{i}"), vec![
                json!({"type": "text", "text": "x".repeat(4000)}),
                json!({"type": "image", "source": img()}),
            ])]));
        }
        let (out, freed) = microcompact_messages(&msgs);
        let v: Vec<Value> = serde_json::from_value(serde_json::to_value(&out).unwrap()).unwrap();
        let first = &v[1]["message"]["content"][0]["content"];
        assert_eq!(first[0]["text"], TIME_BASED_MC_CLEARED_MESSAGE, "{first}");
        assert_eq!(first[1]["type"], "image");
        assert!(freed > 0 && freed < 2 * 4000, "{freed}");
        let count = |m: &[Message]| crate::services::openai_common::attached_image_count(m, 64);
        assert_eq!(count(&out), count(&msgs), "same images, same numbers");
        let (_, again) = microcompact_messages(&out);
        assert_eq!(again, 0, "idempotent");
    }

    /// tests/test_vision_followups.py test_strip_numbers_like_convert.
    #[test]
    fn strip_numbers_like_convert() {
        let msgs = vec![
            user(vec![ContentBlockParam::text("x")]),
            read_call("t"),
            user(vec![
                ContentBlockParam::Image { source: img(), cache_control: None },
                result("t", vec![json!({"type": "image", "source": img()})]),
            ]),
            user(vec![result("t", vec![json!({"type": "image", "source": img()})])]),
            user(vec![ContentBlockParam::Image { source: img(), cache_control: None }]),
        ];
        let v: Vec<Value> = serde_json::from_value(serde_json::to_value(strip_images_for_summary(&msgs)).unwrap()).unwrap();
        let c = &v[2]["message"]["content"];
        assert!(c[1]["content"][0]["text"].as_str().unwrap().starts_with("[Image 1"), "{c}");
        assert!(c[0]["text"].as_str().unwrap().starts_with("[Image 2"), "{c}");
        let dup = v[3]["message"]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert!(dup.starts_with("[Image:") || dup == "[Image]", "{dup}");
        assert!(v[4]["message"]["content"][0]["text"].as_str().unwrap().starts_with("[Image 3"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived sub-agent default must actually lower the threshold, and the
    /// parent must never be touched by it. Guards the untuned-install case: a
    /// 0 ceiling used to mean "same as the parent", which let a sub-agent grow
    /// until the allocator evicted the parent.
    #[test]
    fn subagent_default_is_derived_and_never_caps_the_parent() {
        let parent = 445_752i64;
        // ceiling = 0 -> derived
        let derived = (parent as f64 * SUBAGENT_THRESHOLD_FRACTION) as i64;
        assert!(derived < parent, "derived default did not lower the threshold");
        assert_eq!(derived, 133_725);
        // an explicit ceiling wins over the derivation
        let explicit = 180_000i64;
        assert_eq!(parent.min(explicit), explicit);
        // and an explicit ceiling ABOVE the parent still cannot raise it
        assert_eq!(parent.min(999_999), parent);
    }

    /// Raising the slot count must not multiply the sub-agent footprint.
    ///
    /// The per-agent fraction was measured with ONE extra tenant in the pool.
    /// Taken on its own it says nothing about concurrency, so a slot count of 6
    /// would admit six conversations of 133,725 on top of a 445,752 parent and
    /// wipe the cache with nothing reporting it. The total fraction is what
    /// holds the combined footprint fixed instead.
    #[test]
    fn the_subagent_share_shrinks_as_the_slot_count_grows() {
        let parent = 445_752i64;
        let share = |slots: i64| {
            SUBAGENT_THRESHOLD_FRACTION.min(SUBAGENT_TOTAL_FRACTION / slots.max(1) as f64)
        };
        let cap = |slots: i64| (parent as f64 * share(slots)) as i64;

        // The measured counts are untouched: 1 and 2 are the September
        // single-tenant table's regime, 3 is what this deployment runs.
        assert_eq!(cap(1), 133_725, "one slot keeps the measured per-agent cap");
        assert_eq!(cap(2), 133_725);
        // 0.90 is not a round number chosen for comfort: it is three sub-agents
        // at the per-agent cap, which is this line's actual configuration and
        // the one the 2026-10-10 trials measured (3/3 survived, 0 blocks lost,
        // parent at its full threshold).
        assert_eq!(cap(3), 133_725, "3 x 0.30 is exactly the total, so nothing binds yet");
        assert!(
            (SUBAGENT_TOTAL_FRACTION - 3.0 * SUBAGENT_THRESHOLD_FRACTION).abs() < 1e-9,
            "the total is three sub-agents at the measured per-agent cap"
        );

        // Above them the total binds, and the combined footprint stops growing.
        assert!(cap(4) < cap(3), "a fourth slot has to cost each agent something");
        for slots in 4..=12i64 {
            let total = cap(slots) * slots;
            assert!(
                total <= (parent as f64 * SUBAGENT_TOTAL_FRACTION) as i64 + slots,
                "{slots} slots budget {total}, past the measured total"
            );
        }
        // A nonsense slot count must not divide by zero or hand out more.
        assert_eq!(cap(0), 133_725);
    }

    /// Report Q2 claims a small context window makes percent_left exceed 100,
    /// so `100 - percent_left` underflows usize in the progress bar and aborts.
    /// Check the arithmetic across the whole range rather than assuming.
    #[test]
    fn the_token_bar_width_can_never_underflow() {
        for usage in [0i64, 1, 1_000, 32_000, 100_000, 10_000_000] {
            let st = calculate_token_warning_state(usage);
            assert!(
                (0..=100).contains(&st.percent_left),
                "percent_left {} out of range for usage {usage}", st.percent_left
            );
            let used_pct = 100 - st.percent_left;
            let filled = (20i64 * used_pct / 100) as usize;
            assert!(filled <= 20, "bar filled {filled} for usage {usage}");
            // The subtraction the report says aborts.
            let _ = 20usize - filled;
        }
    }

    #[test]
    fn ptl_gap_vllm_form() {
        let e = "This model's maximum context length is 262144 tokens. However, you requested a total of at least 270000 tokens";
        assert_eq!(parse_ptl_token_gap(Some(e)), Some(270000 - 262144));
    }

    #[test]
    fn ptl_gap_input_output_form() {
        let e = "maximum context length is 1000 tokens, got 900 input tokens and 200 output tokens";
        assert_eq!(parse_ptl_token_gap(Some(e)), Some(100));
    }

    #[test]
    fn ptl_gap_unparseable() {
        assert_eq!(parse_ptl_token_gap(Some("prompt is too long")), None);
        assert_eq!(parse_ptl_token_gap(None), None);
    }

    #[test]
    fn format_summary_strips_analysis() {
        let s = "<analysis>thinking</analysis>\n\n<summary>\nthe facts $1 \\g<0>\n</summary>";
        let out = format_compact_summary(s);
        assert!(out.starts_with("Summary:\n"));
        assert!(out.contains("the facts $1 \\g<0>"));
        assert!(!out.contains("analysis"));
    }

    /// A summary that arrives behind a thinking or tool_use block is still
    /// the summary. `first()` used to fail the whole compaction on it.
    #[test]
    fn summary_text_is_found_behind_a_leading_non_text_block() {
        use crate::types::messages::{ApiMessage, BetaUsage, ContentBlock, ThinkingBlock};
        let with = |content: Vec<ContentBlock>| AssistantMessage {
            uuid: "u".into(),
            message: ApiMessage {
                id: "m".into(),
                message_type: "message".into(),
                role: "assistant".into(),
                content,
                model: "<synthetic>".into(),
                stop_reason: Some("end_turn".into()),
                stop_sequence: None,
                usage: BetaUsage::default(),
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        };
        let thinking = || ContentBlock::Thinking(ThinkingBlock { thinking: "hm".into() });
        let behind = with(vec![thinking(), ContentBlock::text("<summary>s</summary>")]);
        assert_eq!(extract_text(&behind).as_deref(), Some("<summary>s</summary>"));
        assert_eq!(extract_text(&with(vec![thinking()])), None);
        assert_eq!(extract_text(&with(vec![])), None);
    }

    #[test]
    fn looks_like_ptl() {
        assert!(looks_like_prompt_too_long("Error: maximum context length exceeded"));
        assert!(!looks_like_prompt_too_long("connection refused"));
    }

    #[test]
    fn group_rounds() {
        use crate::utils::messages::create_user_message_text;
        let u = |t: &str| Message::User(create_user_message_text(t));
        let a = || {
            Message::Assistant(AssistantMessage {
                uuid: uuid::Uuid::new_v4().to_string(),
                message: Default::default(),
                cost_usd: 0.0,
                duration_ms: 0,
                is_api_error_message: false,
            })
        };
        let msgs = vec![u("hi"), a(), u("tool result"), a(), u("more")];
        let groups = group_messages_by_api_round(&msgs);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].len(), 1); // preamble
        assert_eq!(groups[1].len(), 2);
        assert_eq!(groups[2].len(), 2);
    }

    /// microcompact frees nothing on a second pass.
    ///
    /// The proven-fatal branch treats `freed == 0` as "exhausted" and reports no
    /// reduction rather than a changed message set. That is only correct if
    /// clearing is idempotent — otherwise an exhausted history would look like
    /// progress and the recovery would recurse on an unchanged context, which is
    /// the bug it exists to fix.
    #[test]
    fn microcompact_is_idempotent() {
        use crate::types::messages::{
            ContentBlock, ContentBlockParam, MessageContent, ToolResultContent, ToolUseBlock,
        };
        use crate::utils::messages::create_user_message_text;

        // A result is only compactable when a COMPACTABLE_TOOLS tool_use with
        // the same id precedes it, so the pairs have to be built in full.
        let call = |id: &str| {
            let mut a = AssistantMessage {
                uuid: uuid::Uuid::new_v4().to_string(),
                message: Default::default(),
                cost_usd: 0.0,
                duration_ms: 0,
                is_api_error_message: false,
            };
            a.message.content = vec![ContentBlock::ToolUse(ToolUseBlock {
                id: id.into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "echo hi"}),
            })];
            Message::Assistant(a)
        };
        let result = |id: &str, body: &str| {
            let mut u = create_user_message_text("");
            u.message.content = MessageContent::Blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: id.into(),
                content: ToolResultContent::Text(body.into()),
                is_error: false,
                cache_control: None,
                data: None,
            }]);
            Message::User(u)
        };
        // MICROCOMPACT_KEEP_RECENT survive, so make more pairs than that.
        let big = "x".repeat(4096);
        let mut msgs: Vec<Message> = Vec::new();
        for i in 0..MICROCOMPACT_KEEP_RECENT + 4 {
            let id = format!("t{i}");
            msgs.push(call(&id));
            msgs.push(result(&id, &big));
        }

        let (cleared, freed) = microcompact_messages(&msgs);
        assert!(freed > 0, "the first pass must clear the old results");

        let (_again, freed_again) = microcompact_messages(&cleared);
        assert_eq!(freed_again, 0, "a cleared history has nothing left to free");
    }

    /// A collapse is REPORTED, and changes nothing.
    ///
    /// The threshold assertion is the point of the test. An earlier version of
    /// this code derived a ceiling from the collapse and took the threshold from
    /// 445,752 to 211,357 on one 524,288-token window — 40% of a window the
    /// operator pays for, off a single sample. Lowering it is the operator's
    /// call (auto_compact_threshold_tokens), so the invariant worth pinning is
    /// that observing a collapse does NOT move it.
    ///
    /// One test, not four: cargo runs tests in parallel threads of one process,
    /// they would share OBSERVED_COLLAPSE_TOKENS, and each reset would clobber
    /// whichever other was mid-assertion. Split in two, this failed that way.
    #[test]
    fn a_collapse_is_reported_and_changes_nothing() {
        *OBSERVED_COLLAPSE_TOKENS.lock() = None;
        let before = auto_compact_threshold_for(false);

        // The 2026-10-01 measurement: collapsed at 264,197 on a 524,288 window.
        note_decoder_collapse(264_197);
        assert_eq!(
            auto_compact_threshold_for(false),
            before,
            "an observed collapse must not move the auto-compact threshold",
        );
        assert_eq!(*OBSERVED_COLLAPSE_TOKENS.lock(), Some(264_197), "but it IS recorded");

        // A later, higher collapse says nothing new.
        note_decoder_collapse(400_000);
        assert_eq!(*OBSERVED_COLLAPSE_TOKENS.lock(), Some(264_197), "keeps the lowest");

        // A lower one is the better estimate of where the decoder goes.
        note_decoder_collapse(200_000);
        assert_eq!(*OBSERVED_COLLAPSE_TOKENS.lock(), Some(200_000), "a new low replaces it");

        // Still no effect on the threshold, however many are seen.
        assert_eq!(
            auto_compact_threshold_for(false),
            before,
            "three collapses, threshold still untouched",
        );

        *OBSERVED_COLLAPSE_TOKENS.lock() = None;
    }
}


#[cfg(test)]
mod agent_attachment_tests {
    //! What the model is told about its agents after a compaction. Without
    //! this it comes out of the compaction having forgotten them, and spawns
    //! them again.
    use super::create_async_agent_attachments_if_needed;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::task::create_task_state_base;
    use crate::tasks::local_agent_task::{
        complete_agent_task, fail_agent_task, kill_async_agent, register_agent_foreground,
        update_agent_summary,
    };
    use crate::tasks::local_shell_task::LocalShellTaskState;
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::task::disk_output::get_task_output_path;
    use crate::utils::task::framework::{register_task, update_task_state};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn fake_agent() -> AgentDefinition {
        AgentDefinition {
            agent_type: "Explore".into(),
            when_to_use: String::new(),
            get_system_prompt: Arc::new(|| String::new()),
            tools: None,
            disallowed_tools: Vec::new(),
            source: "built-in".into(),
        }
    }

    fn spawn(description: &str) -> String {
        let id = create_agent_id(None);
        register_agent_foreground(
            &id, description, "find the thing", fake_agent(), None,
            Some(CancellationToken::new()), None,
        );
        id
    }

    fn text_of(messages: &[crate::types::query::UserMessage]) -> String {
        messages
            .iter()
            .map(|m| match &m.message.content {
                crate::types::messages::MessageContent::Text(t) => t.clone(),
                crate::types::messages::MessageContent::Blocks(b) => b
                    .iter()
                    .filter_map(|x| serde_json::to_value(x).ok())
                    .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect::<Vec<_>>()
                    .join(""),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn every_agent_the_model_should_know_about_is_reported() {
        let _guard = test_lock();
        reset_app_state();
        let running = spawn("survey the lane");
        update_agent_summary(&running, "reading the config");
        let done = spawn("count the tokens");
        complete_agent_task(&done, serde_json::json!({"text": "counted"}));
        let broke = spawn("try the kpool");
        fail_agent_task(&broke, "kpool refused");
        let stopped = spawn("the one the user stopped");
        kill_async_agent(&stopped);

        let messages = create_async_agent_attachments_if_needed(None);
        assert_eq!(messages.len(), 4);
        assert!(messages.iter().all(|m| m.is_meta), "all are meta messages");
        let all = text_of(&messages);
        assert!(all.matches("<system-reminder>").count() == 4);

        assert!(all.contains("Background agent \"survey the lane\""));
        assert!(all.contains("Do NOT spawn a duplicate."));
        assert!(all.contains("Progress: reading the config"));
        assert!(all.contains(&format!(
            "You can read partial output at {}",
            get_task_output_path(&running).to_string_lossy()
        )));
        // The model has just lost the record of the spawn, so the reminder has
        // to tell it the agent is still reachable — not only readable.
        assert!(
            all.contains("or send it a message with SendMessage"),
            "and that it can still send the agent a message: {all}"
        );

        assert!(all.contains(&format!(
            "Task {done} (type: local_agent) (status: completed) (description: count the tokens)"
        )), "{all}");
        assert!(all.contains("Delta: kpool refused"));
        assert!(all.contains(&format!(
            "Task \"the one the user stopped\" ({stopped}) was stopped by the user."
        )));
        reset_app_state();
    }

    #[test]
    fn three_kinds_of_task_are_skipped() {
        let _guard = test_lock();
        reset_app_state();
        let retrieved = spawn("already read");
        update_task_state(&retrieved, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.retrieved = true;
            Some(next.into())
        });
        assert!(
            create_async_agent_attachments_if_needed(None).is_empty(),
            "a retrieved agent is not reported again"
        );

        reset_app_state();
        let pending = spawn("not started");
        update_task_state(&pending, |t| {
            let mut next = t.as_local_agent()?.clone();
            next.base.status = "pending".into();
            Some(next.into())
        });
        assert!(
            create_async_agent_attachments_if_needed(None).is_empty(),
            "a pending agent is not reported"
        );

        reset_app_state();
        let mine = spawn("my own work");
        assert!(
            create_async_agent_attachments_if_needed(Some(&mine)).is_empty(),
            "an agent is not told about itself"
        );

        reset_app_state();
        let mut base = create_task_state_base("b11111111", "local_bash", "run tests", None);
        base.status = "running".into();
        register_task(
            LocalShellTaskState { base, command: "pytest".into(), kind: "bash".into(), ..Default::default() }
                .into(),
        );
        assert!(
            create_async_agent_attachments_if_needed(None).is_empty(),
            "a shell task is not an agent attachment"
        );
        reset_app_state();
    }
}
