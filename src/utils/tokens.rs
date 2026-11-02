//! Token counting utilities for CodeMason
//! Equivalent to codemason/utils/tokens.py (src/utils/tokens.ts)

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::types::messages::{ContentBlock, ContentBlockParam, MessageContent, ToolResultContent};
use crate::types::query::Message;

use super::log::debug_log;

// === Token Limits ===

// Default fallback values
const DEFAULT_CONTEXT_SIZE: i64 = 131072;
#[allow(dead_code)]
const DEFAULT_REQUEST_MAX_TOKENS: i64 = 16384; // max_tokens sent per request (matches server -n)

#[derive(Debug, Clone, Copy)]
struct ServerInfo {
    n_ctx: i64,
    #[allow(dead_code)]
    total_slots: i64,
}

/// Cached values (set once on first successful probe).
static SERVER_INFO: Lazy<Mutex<Option<ServerInfo>>> = Lazy::new(|| Mutex::new(None));

/// When the last failed probe finished. A failure is still not cached as an
/// ANSWER — get_context_size must not be pinned to the fallback default for the
/// session, which is what the note on fetch_server_info is protecting — but it
/// is rate-limited, because the probe costs a 5s HTTP timeout on a blocked
/// worker thread and the footer asks for it on every render. Without this, a
/// backend that is down turned each rendered line into another five seconds.
static LAST_PROBE_FAILURE: Lazy<Mutex<Option<std::time::Instant>>> =
    Lazy::new(|| Mutex::new(None));

/// How long to wait before re-probing after a failure. Short enough that a
/// server coming up is picked up within seconds, long enough that a server that
/// is down does not cost a timeout per rendered footer.
const PROBE_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

/// Get base URL and API key from Config (single source of truth).
fn get_server_credentials() -> (String, String) {
    let cfg = crate::utils::config::get_config();
    (cfg.model_base_url, cfg.model_api_key)
}

/// Run a small async HTTP probe from a sync context. The Python module used
/// blocking httpx calls; reqwest here is async-only, so this bridges: inside a
/// running multi-thread tokio runtime it block_in_place's, otherwise it spins
/// a temporary current-thread runtime. (Substrate divergence, sync API kept.)
fn block_on_http<F, T>(fut: F) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => Some(tokio::task::block_in_place(|| handle.block_on(fut))),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()
            .map(|rt| rt.block_on(fut)),
    }
}

async fn http_get_json(url: &str, api_key: &str) -> Option<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let mut req = client.get(url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {api_key}"));
    }
    let resp = req.send().await.ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    resp.json::<serde_json::Value>().await.ok()
}

/// Resolve-once guard: model discovery runs a single /v1/models probe per process.
static MODEL_RESOLVED: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

/// Make the model id used this run agree with what the server actually serves.
///
/// Why: hard-coding the model id means one stale name makes every request fail
/// late, in-stream, with a cryptic "model does not exist". So discover:
///
///   - config value present AND served        -> keep it (no change).
///   - config value missing / not served      -> if the server advertises
///     exactly ONE model, adopt it FOR THIS PROCESS and say so on stderr. If
///     SEVERAL, leave it alone and print the available list.
///   - /v1/models unreachable / errors         -> no-op; fall back to config.
///
/// config.json is never written from here — which id a server advertises is a
/// fact about the endpoint this run is pointed at, not a correction to what
/// the user meant. Runs at most once per process. Returns the resolved model
/// id (or the unchanged configured value on any fallback path).
pub fn resolve_model() -> String {
    {
        let mut resolved = MODEL_RESOLVED.lock();
        if *resolved {
            return crate::utils::config::get_config().model_name;
        }
        *resolved = true;
    }

    let cfg = crate::utils::config::get_config();
    let current = cfg.model_name.trim().to_string();

    // --- 1. Reconcile the model id against what the server serves. ---
    let (base_url, api_key) = get_server_credentials();
    let ids: Option<Vec<String>> = block_on_http(async {
        let v = http_get_json(&format!("{base_url}/v1/models"), &api_key).await?;
        let data = v.get("data")?.as_array()?;
        Some(
            data.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(|s| s.to_string()))
                .collect::<Vec<String>>(),
        )
    })
    .flatten()
    .filter(|v: &Vec<String>| !v.is_empty());

    if let Some(ids) = ids {
        if !current.is_empty() && ids.contains(&current) {
            // already valid
        } else if ids.len() == 1 {
            crate::utils::config::update_config(|c| c.model_name = ids[0].clone());
            if !current.is_empty() {
                eprintln!(
                    "[codemason] Model '{current}' is not served; auto-switched to '{}'.",
                    ids[0]
                );
            } else {
                eprintln!("[codemason] Model auto-detected: '{}'.", ids[0]);
            }
        } else {
            // Several served, configured one invalid/empty: don't guess.
            eprintln!(
                "[codemason] Configured model '{}' is not served. Available: {}. Set model_name in config.json.",
                if current.is_empty() { "(none)" } else { &current },
                ids.join(", ")
            );
        }
    }

    // model_dir is left exactly as written; nothing here writes to config.json
    // (see the Python reference for the full history of why).

    crate::utils::config::get_config().model_name
}

/// Detect the served context size, SERVER-AGNOSTICALLY.
///
/// The backend may be vLLM or llama.cpp, so probe both:
///   - vLLM / OpenAI servers: GET /v1/models -> data[].max_model_len
///   - llama.cpp:             GET /slots     -> [0].n_ctx
///
/// Cached ONLY on success. A failed probe must NOT be cached: an early call
/// may happen before the server is up, and a cached miss would pin
/// get_context_size() to the fallback default for the whole session —
/// mis-sizing the auto-compact threshold.
fn fetch_server_info() -> Option<ServerInfo> {
    if let Some(info) = *SERVER_INFO.lock() {
        return Some(info);
    }
    if let Some(at) = *LAST_PROBE_FAILURE.lock() {
        if at.elapsed() < PROBE_RETRY_AFTER {
            return None;
        }
    }

    let (base_url, api_key) = get_server_credentials();

    // vLLM / OpenAI-compatible: /v1/models advertises max_model_len.
    //
    // Match the CONFIGURED model id FIRST. `data[0]` is whichever model the
    // server happened to list first, which on an endpoint serving several is
    // not the one this session talks to — and every number downstream (the
    // auto-compact threshold, the output cap, the status bar) is derived from
    // it, then cached permanently on the first success. Wrong in either
    // direction hurts: a smaller window compacts a healthy session to death, a
    // larger one lets the request 400. resolve_model() already reconciles
    // cfg.model_name against this same list, so by the time anything can send
    // a request the configured id IS the served one.
    let model_name = crate::utils::config::get_config().model_name.trim().to_string();
    let from_models: Option<ServerInfo> = block_on_http(async {
        let v = http_get_json(&format!("{base_url}/v1/models"), &api_key).await?;
        let data = v.get("data")?.as_array()?;
        let max_len = |m: &serde_json::Value| {
            m.get("max_model_len").and_then(|x| x.as_i64()).filter(|n| *n > 0)
        };
        if !model_name.is_empty() {
            for m in data {
                if m.get("id").and_then(|i| i.as_str()) == Some(model_name.as_str()) {
                    if let Some(mlen) = max_len(m) {
                        return Some(ServerInfo { n_ctx: mlen, total_slots: 1 });
                    }
                }
            }
        }
        // No id matched (model_name empty, an alias the server renames, or a
        // matching entry that omits max_model_len): fall back to the first
        // entry that advertises a window — the old behaviour. Some served
        // window still beats DEFAULT_CONTEXT_SIZE, which is a guess about a
        // server nobody probed.
        for m in data {
            if let Some(mlen) = max_len(m) {
                return Some(ServerInfo { n_ctx: mlen, total_slots: 1 });
            }
        }
        None
    })
    .flatten();
    if let Some(info) = from_models {
        *SERVER_INFO.lock() = Some(info);
        return Some(info);
    }

    // llama.cpp: /slots reports per-slot n_ctx.
    let from_slots: Option<ServerInfo> = block_on_http(async {
        let v = http_get_json(&format!("{base_url}/slots"), &api_key).await?;
        let slots = v.as_array()?;
        if slots.is_empty() {
            return None;
        }
        Some(ServerInfo {
            n_ctx: slots[0].get("n_ctx").and_then(|x| x.as_i64()).unwrap_or(DEFAULT_CONTEXT_SIZE),
            total_slots: slots.len() as i64,
        })
    })
    .flatten();
    if let Some(info) = from_slots {
        *SERVER_INFO.lock() = Some(info);
        return Some(info);
    }

    // All probes failed. Still NOT cached as an answer: the next call after the
    // retry window will probe again, so a server that comes up later is picked
    // up. Only the retry RATE is bounded, so a server that stays down costs one
    // timeout per window instead of one per rendered footer.
    *LAST_PROBE_FAILURE.lock() = Some(std::time::Instant::now());
    None
}

/// Per-slot context size. The SERVER is the authority; env only fills in.
///
/// An env override ABOVE the served size is not a preference, it is wrong (the
/// server rejects such requests, so auto-compact could never fire in time) —
/// the served size wins. Below the served size it is a real preference
/// (deliberately running small), so it is honoured. When no probe succeeds,
/// the override is all we have.
pub fn get_context_size() -> i64 {
    let env_val = std::env::var("CODEMASON_MODEL_CONTEXT_SIZE")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var("LLAMACPP_CONTEXT_SIZE").ok().filter(|v| !v.is_empty()));
    let served = fetch_server_info().map(|i| i.n_ctx);

    let Some(env_val) = env_val else {
        return served.unwrap_or(DEFAULT_CONTEXT_SIZE);
    };

    let requested = match env_val.parse::<i64>() {
        Ok(v) => v,
        Err(_) => {
            debug_log("CONTEXT_SIZE_ENV_INVALID", Some(&serde_json::json!({"value": env_val})));
            return served.unwrap_or(DEFAULT_CONTEXT_SIZE);
        }
    };

    if let Some(served) = served {
        if requested > served {
            debug_log(
                "CONTEXT_SIZE_ENV_CLAMPED",
                Some(&serde_json::json!({
                    "requested": requested,
                    "served": served,
                    "note": "env override exceeds what the server was started with",
                })),
            );
            return served;
        }
    }
    requested
}

/// Get output token reserve = config.max_tokens (env overridable).
pub fn get_output_reserve() -> i64 {
    let env_val = std::env::var("CODEMASON_MODEL_OUTPUT_RESERVE")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var("LLAMACPP_OUTPUT_RESERVE").ok().filter(|v| !v.is_empty()));
    if let Some(v) = env_val {
        if let Ok(n) = v.parse::<i64>() {
            return n;
        }
    }
    crate::utils::config::get_config().max_tokens
}

/// Get maximum usable input tokens = context_size - output_reserve.
///
/// NOTE: deliberately NO module-level constant for this — computing it eagerly
/// would run a blocking server probe on startup (see the Python reference).
pub fn get_max_tokens() -> i64 {
    get_context_size() - get_output_reserve()
}

// === Model tokenizer (exact, matches the server) ===
//
// A generic vocab (the old tiktoken cl100k fallback) belongs to a different
// model, NOT the served one — it
// mis-counts DSV4 (Korean +42%, code -13%). When a tokenizer.json is found we
// load the model's OWN tokenizer, which matches the server's /tokenize
// exactly. Its presence also flips context handling to the upstream-faithful
// path (the non-canonical _manage_context pre-send trim switches OFF).
//
// Cache: (key, value) where key is the model name the cache was built for;
// a miss (None value) is cached too, so we don't re-stat every count. Rebuilt
// when config.model_name changes (an early count before resolve_model() adopts
// the served id must not pin the fallback for the whole process).
type TokenizerCache = Option<(String, Option<Arc<tokenizers::Tokenizer>>)>;
static MODEL_TOKENIZER: Lazy<Mutex<TokenizerCache>> = Lazy::new(|| Mutex::new(None));

/// Candidate roots for the bundled tokenizer tree. Python resolved relative to
/// the package __file__; the binary equivalent is the executable's directory
/// (and its ancestors, which covers a dev-tree target/debug layout).
/// Substrate divergence, documented: drop tokenizer/<model>/tokenizer.json
/// next to the binary (or at the repo root in a source tree).
fn bundled_roots() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
            if let Some(p1) = dir.parent() {
                roots.push(p1.to_path_buf());
                if let Some(p2) = p1.parent() {
                    roots.push(p2.to_path_buf());
                }
            }
        }
    }
    roots
}

/// Load the served model's own tokenizer for the currently-configured model,
/// or None if unavailable. Keyed by config.model_name (no hardcoded name).
/// First existing candidate wins:
///   1./2. <bundled root>/tokenizer/<model>/tokenizer.json
///   3.    <model_dir>/<model>/tokenizer.json
///   4.    <model_dir>/tokenizer.json  (model_dir = an HF checkout)
/// Returns None (=> CJK-aware heuristic) if none found or the file is bad.
pub fn get_model_tokenizer() -> Option<Arc<tokenizers::Tokenizer>> {
    let cfg = crate::utils::config::get_config();
    let model = cfg.model_name.trim().to_string();

    {
        let cache = MODEL_TOKENIZER.lock();
        if let Some((key, value)) = cache.as_ref() {
            if *key == model {
                return value.clone();
            }
        }
    }

    let model_dir = cfg.model_dir.trim().to_string();

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if !model.is_empty() {
        for root in bundled_roots() {
            candidates.push(root.join("tokenizer").join(&model).join("tokenizer.json"));
        }
        if !model_dir.is_empty() {
            candidates.push(
                std::path::Path::new(&model_dir).join(&model).join("tokenizer.json"),
            );
        }
    }
    if !model_dir.is_empty() {
        candidates.push(std::path::Path::new(&model_dir).join("tokenizer.json"));
    }

    let mut loaded: Option<Arc<tokenizers::Tokenizer>> = None;
    for tok_path in candidates {
        if tok_path.is_file() {
            // Bad file -> stay on the heuristic fallback path.
            loaded = tokenizers::Tokenizer::from_file(&tok_path).ok().map(Arc::new);
            break;
        }
    }
    *MODEL_TOKENIZER.lock() = Some((model, loaded.clone()));
    loaded
}

/// True when a tokenizer for `model` ships alongside the binary (candidates
/// 1-2 of get_model_tokenizer). When this holds, config.model_dir is redundant
/// — the bundle always wins.
pub fn bundled_tokenizer_exists(model: &str) -> bool {
    let model = model.trim();
    if model.is_empty() {
        return false;
    }
    bundled_roots()
        .iter()
        .any(|d| d.join("tokenizer").join(model).join("tokenizer.json").is_file())
}

/// True when the exact model tokenizer is available (accurate-counting mode).
/// Gates OFF the non-canonical _manage_context trim in the backends.
pub fn has_model_tokenizer() -> bool {
    get_model_tokenizer().is_some()
}

/// Script-aware character heuristic, used whenever the served model's own
/// tokenizer is unavailable. Latin-ish text at len/bytes_per_token
/// (~chars/4), the dense scripts at ~1 token per char, symbols/emoji at 2
/// (see DENSE_RANGES / SYMBOL_RANGES).
///
/// This replaced a tiktoken cl100k fallback. cl100k is ANOTHER model's vocab —
/// it mis-counts the served one (Korean +42%, code -13%), so it never gave a
/// count worth trusting, and it reported that wrong number as if exact. It
/// also made the process hang: tiktoken's byte_pair_merge rescans the whole
/// `parts` vector per merge (documented as O(mn), "n is often very small"),
/// and its pretokenizer emits an UNBOUNDED piece for a run of punctuation. A
/// model that fell into repetition emitted 26,655 consecutive '!' — one piece,
/// ~7e8 comparisons plus ~5.7GB of memmove, re-run every time the growing
/// buffer was recounted. One core pinned for 30+ minutes, the response socket
/// never drained. An estimate that is honest about being an estimate is
/// strictly better than an exact-looking count from the wrong vocabulary.
fn heuristic_token_count(content: &str, bytes_per_token: i64) -> i64 {
    let t = tally_scripts(content);
    ((t.latinish as f64) / bytes_per_token as f64
        + t.dense as f64 * DENSE_TOKENS_PER_CHAR
        + t.symbol as f64 * SYMBOL_TOKENS_PER_CHAR)
        .round() as i64
}

/// Exact token count for a string when the model's own tokenizer is available
/// (matches the server); otherwise the CJK-aware heuristic. This is the single
/// substrate both backends' count_tokens_precise should call.
pub fn count_text_tokens(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    if let Some(tok) = get_model_tokenizer() {
        // add_special_tokens=true matches the HF Python encode() default the
        // reference used.
        if let Ok(enc) = tok.encode(text, true) {
            return enc.get_ids().len() as i64;
        }
    }
    heuristic_token_count(text, 4)
}

// === Token Counting Functions ===

/// Extract text content from a message for token estimation.
pub fn extract_text_content(message: &Message) -> String {
    match message {
        Message::User(u) => match &u.message.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(blocks) => {
                let mut parts: Vec<String> = Vec::new();
                for block in blocks {
                    match block {
                        ContentBlockParam::Text { text, .. } => parts.push(text.clone()),
                        ContentBlockParam::ToolUse { input, .. } => {
                            parts.push(serde_json::to_string(input).unwrap_or_default())
                        }
                        ContentBlockParam::ToolResult { content, .. } => match content {
                            ToolResultContent::Text(s) => parts.push(s.clone()),
                            ToolResultContent::Blocks(list) => {
                                for c in list {
                                    if let Some(s) = c.as_str() {
                                        parts.push(s.to_string());
                                    } else if c.get("type").and_then(|t| t.as_str())
                                        == Some("text")
                                    {
                                        parts.push(
                                            c.get("text")
                                                .and_then(|t| t.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                        );
                                    }
                                }
                            }
                        },
                        _ => {}
                    }
                }
                parts.join("\n")
            }
        },
        Message::Assistant(a) => {
            let mut parts: Vec<String> = Vec::new();
            for block in &a.message.content {
                match block {
                    ContentBlock::Text(t) => parts.push(t.text.clone()),
                    ContentBlock::ToolUse(tu) => {
                        parts.push(serde_json::to_string(&tu.input).unwrap_or_default())
                    }
                    _ => {}
                }
            }
            parts.join("\n")
        }
        Message::Progress(_) => String::new(),
    }
}

/// Synthetic message texts to skip when counting tokens. MUST match the
/// strings actually produced (utils::messages INTERRUPT_MESSAGE*).
pub const SYNTHETIC_ASSISTANT_MESSAGES: &[&str] = &[
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
    "No response requested.",
];

fn first_text(content: &[ContentBlock]) -> &str {
    match content.first() {
        Some(ContentBlock::Text(t)) => &t.text,
        _ => "",
    }
}

fn usage_total(usage: &crate::types::messages::BetaUsage) -> i64 {
    usage.input_tokens
        + usage.cache_creation_input_tokens
        + usage.cache_read_input_tokens
        + usage.output_tokens
}

/// Count tokens from the most recent assistant message's usage info.
/// Falls back (returns 0) if no usage info available — matches the TS original.
/// A zero usage total does not anchor (server reported no usage, not a free
/// response); keep walking back.
pub fn count_tokens(messages: &[Message]) -> i64 {
    for message in messages.iter().rev() {
        if let Message::Assistant(a) = message {
            // Skip synthetic messages.
            if !a.message.content.is_empty()
                && SYNTHETIC_ASSISTANT_MESSAGES.contains(&first_text(&a.message.content))
            {
                continue;
            }
            let total = usage_total(&a.message.usage);
            if total > 0 {
                return total;
            }
        }
    }
    0
}

// === Rough block-based estimation (services/tokenEstimation.ts) ===

/// Unicode blocks the model's BPE charges ~1+ tokens EACH, where the plain
/// chars/4 rule under-counts by 4x or more.
///
/// Two families, same charge. CJK proper (Hangul, Kana, ideographs, fullwidth
/// forms) because the vocabulary holds those characters whole. And the
/// non-Latin ALPHABETS — Cyrillic, Greek, Hebrew, Arabic, Thai, the Indic
/// blocks — because a byte-level BPE spends 2-3 UTF-8 bytes on each of them
/// and has very few merges to spend them on: the real cost sits between 1 and
/// 2 tokens/char, so 1 is the FLOOR of the true range, not a padded
/// over-estimate. They were previously counted at chars/4, i.e. 4-8x low.
///
/// This path is live whenever the served model's tokenizer.json is absent, and
/// an under-count is the dangerous direction: the threshold fires late and the
/// server rejects the whole request. Compaction firing a little early costs
/// nothing by comparison.
const DENSE_RANGES: &[(u32, u32)] = &[
    (0x0370, 0x03FF),   // Greek and Coptic
    (0x0400, 0x04FF),   // Cyrillic
    (0x0500, 0x052F),   // Cyrillic Supplement
    (0x0590, 0x05FF),   // Hebrew
    (0x0600, 0x06FF),   // Arabic
    (0x0750, 0x077F),   // Arabic Supplement
    (0x0900, 0x0DFF),   // Devanagari .. Sinhala (the 3-byte Indic blocks)
    (0x0E00, 0x0E7F),   // Thai
    (0x1100, 0x11FF),   // Hangul Jamo
    (0x1F00, 0x1FFF),   // Greek Extended
    (0x2E80, 0x2FDF),   // CJK + Kangxi Radicals
    // 0x3000-0x303F is CJK Symbols and Punctuation — 。、「」 and the
    // ideographic space. The old range started at 0x3040 and charged all of it
    // at chars/4, and it is not an edge case: it is the punctuation of every
    // Japanese sentence.
    (0x3000, 0x30FF),   // CJK punctuation + Hiragana + Katakana
    (0x3130, 0x318F),   // Hangul Compatibility Jamo
    (0x31F0, 0x31FF),   // Katakana Phonetic Extensions
    (0x3400, 0x4DBF),   // CJK Ext A
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0xA8E0, 0xA8FF),   // Devanagari Extended
    (0xAC00, 0xD7A3),   // Hangul Syllables
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0xFB50, 0xFDFF),   // Arabic Presentation Forms-A
    (0xFE30, 0xFE4F),   // CJK Compatibility Forms
    (0xFE70, 0xFEFF),   // Arabic Presentation Forms-B
    (0xFF00, 0xFFEF),   // Halfwidth / Fullwidth Forms
    // The old list stopped at 0xFFEF, so every rarer ideograph — 4 UTF-8 bytes
    // each and far less likely to be a single merge than Ext A — was counted
    // at a quarter of a token.
    (0x20000, 0x323AF), // CJK Ext B..H + Compatibility Ideographs Supplement
];

/// Symbols and pictographs, charged at 2 tokens each: astral 4-byte
/// characters that a byte-level BPE normally splits, and a rendered emoji
/// carries a variation selector (and often a ZWJ) on top of its base
/// character. 2 is the low end of what they cost, chars/4 was not on the
/// scale.
const SYMBOL_RANGES: &[(u32, u32)] = &[
    (0x2600, 0x27BF),   // Misc Symbols + Dingbats
    (0xFE00, 0xFE0F),   // Variation Selectors (U+FE0F rides most emoji)
    (0x1F000, 0x1F2FF), // Mahjong / Domino / Cards / Enclosed
    (0x1F300, 0x1FAFF), // Pictographs, Emoticons, Transport, Ext-A
];

const DENSE_TOKENS_PER_CHAR: f64 = 1.0;
const SYMBOL_TOKENS_PER_CHAR: f64 = 2.0;

/// Everything below this is ASCII / Latin-1 / Latin Extended-A+B and is always
/// charged at chars/bytes_per_token. One comparison per character keeps the
/// scan as cheap as the old eight-range one for code and English, which is
/// what this counts nearly all of the time.
const LATINISH_CEILING: u32 = 0x0370;

#[derive(Default)]
struct ScriptTally {
    /// charged at 1/bytes_per_token each
    latinish: i64,
    /// charged at DENSE_TOKENS_PER_CHAR each
    dense: i64,
    /// charged at SYMBOL_TOKENS_PER_CHAR each
    symbol: i64,
}

fn in_ranges(o: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|(lo, hi)| *lo <= o && o <= *hi)
}

fn tally_scripts(content: &str) -> ScriptTally {
    let mut t = ScriptTally::default();
    for ch in content.chars() {
        let o = ch as u32;
        if o < LATINISH_CEILING {
            t.latinish += 1;
        } else if in_ranges(o, DENSE_RANGES) {
            t.dense += 1;
        } else if in_ranges(o, SYMBOL_RANGES) {
            t.symbol += 1;
        } else {
            // Unlisted scripts (Latin Extended Additional, Armenian, …) keep
            // the chars/4 rule: still a guess, but not a 4x one.
            t.latinish += 1;
        }
    }
    t
}

/// roughTokenCountEstimation. When the served model's OWN tokenizer is
/// available this returns the EXACT count (matches the server) so compaction /
/// microcompact size on real tokens. Otherwise the script-aware char
/// heuristic: Latin-ish text at len/bytes_per_token (~chars/4), the dense
/// scripts at ~1 token each, symbols/emoji at 2.
pub fn rough_token_count_estimation(content: &str, bytes_per_token: i64) -> i64 {
    if content.is_empty() {
        return 0;
    }
    if has_model_tokenizer() {
        return count_text_tokens(content);
    }
    heuristic_token_count(content, bytes_per_token)
}

/// roughTokenCountEstimationForBlock — per content-block type (assistant side).
fn rough_block_out(block: &ContentBlock) -> i64 {
    match block {
        ContentBlock::Text(t) => rough_token_count_estimation(&t.text, 4),
        ContentBlock::ToolUse(tu) => rough_token_count_estimation(
            &format!("{}{}", tu.name, serde_json::to_string(&tu.input).unwrap_or_default()),
            4,
        ),
        ContentBlock::Thinking(t) => rough_token_count_estimation(&t.thinking, 4),
    }
}

/// Per content-block estimate (user/param side). Images are charged what the
/// server spends on them (header-only read, utils/image_prep.rs); documents the
/// original's conservative flat 2000 — NEVER a base64 char count.
fn rough_block_param(block: &ContentBlockParam) -> i64 {
    match block {
        ContentBlockParam::Text { text, .. } => rough_token_count_estimation(text, 4),
        ContentBlockParam::Image { source, .. } => {
            crate::utils::image_prep::image_source_tokens(source) as i64
        }
        ContentBlockParam::Document { .. } => 2000,
        ContentBlockParam::ToolResult { content, .. } => match content {
            ToolResultContent::Text(s) => rough_token_count_estimation(s, 4),
            ToolResultContent::Blocks(list) => list
                .iter()
                .map(|b| {
                    if let Some(s) = b.as_str() {
                        rough_token_count_estimation(s, 4)
                    } else if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                        rough_token_count_estimation(
                            b.get("text").and_then(|t| t.as_str()).unwrap_or(""),
                            4,
                        )
                    } else if b.get("type").and_then(|t| t.as_str()) == Some("image") {
                        b.get("source")
                            .map(crate::utils::image_prep::image_source_tokens)
                            .unwrap_or(crate::utils::image_prep::MAX_TOKENS_PER_IMAGE)
                            as i64
                    } else if b.get("type").and_then(|t| t.as_str()) == Some("document") {
                        2000
                    } else {
                        rough_token_count_estimation(&b.to_string(), 4)
                    }
                })
                .sum(),
        },
        ContentBlockParam::ToolUse { name, input, .. } => rough_token_count_estimation(
            &format!("{}{}", name, serde_json::to_string(input).unwrap_or_default()),
            4,
        ),
        ContentBlockParam::Thinking { thinking, .. } => {
            rough_token_count_estimation(thinking, 4)
        }
    }
}

/// roughTokenCountEstimationForMessages — sum of per-message content estimates.
pub fn rough_token_count_estimation_for_messages(messages: &[Message]) -> i64 {
    let mut total = 0i64;
    for message in messages {
        match message {
            Message::Assistant(a) => {
                total += a.message.content.iter().map(rough_block_out).sum::<i64>();
            }
            Message::User(u) => match &u.message.content {
                MessageContent::Text(s) => total += rough_token_count_estimation(s, 4),
                MessageContent::Blocks(blocks) => {
                    total += blocks.iter().map(rough_block_param).sum::<i64>()
                }
            },
            Message::Progress(_) => {}
        }
    }
    total
}

/// Synthetic assistant messages whose usage must NOT anchor the token count.
const SYNTHETIC_MODEL: &str = "<synthetic>";

/// getTokenUsage: usage only for a real (non-synthetic) assistant message.
fn get_token_usage(message: &Message) -> Option<&crate::types::messages::BetaUsage> {
    let Message::Assistant(a) = message else {
        return None;
    };
    if a.message.model == SYNTHETIC_MODEL {
        return None;
    }
    if !a.message.content.is_empty()
        && SYNTHETIC_ASSISTANT_MESSAGES.contains(&first_text(&a.message.content))
    {
        return None;
    }
    Some(&a.message.usage)
}

fn get_assistant_message_id(message: &Message) -> Option<&str> {
    let Message::Assistant(a) = message else {
        return None;
    };
    if a.message.model == SYNTHETIC_MODEL {
        return None;
    }
    if a.message.id.is_empty() {
        return None;
    }
    Some(&a.message.id)
}

/// tokenCountWithEstimation.
///
/// Anchors on the most recent assistant message with real (non-synthetic) API
/// usage, then rewinds past sibling records split from the SAME API response
/// (parallel tool calls share message.id) so interleaved tool_results between
/// splits are counted, and adds a per-block char/4 estimation of everything
/// after the anchor. A usage record whose counts sum to ZERO does NOT anchor —
/// a real API response always spends at least its input tokens, so a zero
/// total means the server reported no usage; anchoring on it collapsed the
/// whole count to zero and silently switched auto-compact off. Falls back to a
/// pure estimate when no message has usage.
/// Tokens the estimate cannot see, measured once per process.
///
/// The system prompt and the tool block go out with every request but live
/// outside `messages`, so a walk over the history never counts them. An
/// ANCHORED estimate does not need them — it starts from the server's own count
/// of the previous request, which already included both — but the unanchored
/// fallback is blind to roughly 30k tokens here (~25k of tool schemas plus the
/// system prompt). That gap is what lets a large input sit under the compaction
/// threshold and then be rejected by the server: measured on the Python port,
/// 458,753 input + 65,536 requested output = 524,289 against a 524,288 window,
/// one token over, on a one-line request.
///
/// Cached because both callers (should_auto_compact and the output cap in
/// services/openai_common) ask every turn and the answer is fixed for the
/// process: the tool block is frozen per tool-set and the system prompt is
/// built once at session start.
/// PER SCOPE, not per process. A single cell was written by whichever request
/// went out last, and the main conversation and its sub-agents do not have the
/// same out-of-band size: a sub-agent carries its own system prompt and a subset
/// of the tools (often three or four against the main agent's full block). They
/// run concurrently, so the main agent's compaction check could read a
/// sub-agent's small number and under-count its own request by ~25k — the exact
/// gap this value exists to close — or a sub-agent could be measured against the
/// main block and compacted while its history was still short. The scope key is
/// the tool set (see `oob_scope_key`), which is what differs between them and is
/// available on both the writing and the reading side.
static OUT_OF_BAND_TOKENS: Lazy<Mutex<std::collections::HashMap<String, i64>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// The scope a request's out-of-band size is filed under: its tool set.
///
/// Built in one place so the writer (openai_common, before the request) and the
/// reader (the compaction check, a turn later) cannot drift apart — they did not
/// share a key at all before, which is how one could read the other's number.
pub fn oob_scope_key<'a>(tool_names: impl Iterator<Item = &'a str>) -> String {
    tool_names.collect::<Vec<_>>().join("\u{1f}")
}

/// Record the system-prompt + tool-block size for `estimate_input_tokens`.
pub fn set_out_of_band_tokens(scope: &str, n: i64) {
    OUT_OF_BAND_TOKENS.lock().insert(scope.to_string(), n.max(0));
}

/// The out-of-band size for `scope`.
///
/// An unregistered scope falls back to the LARGEST one recorded, not to zero.
/// Zero is what the old unanchored estimate already did, and it is the direction
/// that ends in a server rejection: the request goes out believing it is ~25k
/// smaller than it is, and once the INPUT alone is over the window the output cap
/// downstream cannot save it. Over-counting only compacts a little early.
fn out_of_band_for(scope: &str) -> i64 {
    let m = OUT_OF_BAND_TOKENS.lock();
    if let Some(n) = m.get(scope) {
        return *n;
    }
    m.values().copied().max().unwrap_or(0)
}

/// True when `token_count_with_estimation` will anchor on a REAL usage record.
///
/// Callers need this to know what the estimate already contains. The condition
/// must stay identical to the anchor test in token_count_with_estimation: both
/// go through get_token_usage (which skips synthetic messages) and the >0 total
/// rule (a zero total means the server reported nothing, not a free turn).
pub fn has_usage_anchor(messages: &[Message]) -> bool {
    messages.iter().rev().any(|m| match get_token_usage(m) {
        Some(u) => usage_total(u) > 0,
        None => false,
    })
}

/// What the SERVER will see: the history plus the out-of-band prompt parts.
///
/// Use this, not `token_count_with_estimation`, wherever the number is compared
/// against the model's context window — the compaction threshold and the output
/// cap both are. token_count_with_estimation answers a narrower question ("how
/// big is this history"); using it as a stand-in for the request size is what
/// makes compaction miss and the request overflow.
///
/// The out-of-band tokens are added only when the estimate is UNANCHORED.
/// Adding them to an anchored count double-counts the tool block on every turn
/// of an ongoing conversation — not an overflow, but a silent ~25k tax that
/// shortens answers and drags compaction forward.
pub fn estimate_input_tokens(messages: &[Message], scope: &str) -> i64 {
    let est = token_count_with_estimation(messages);
    let oob = out_of_band_for(scope);
    if oob > 0 && !has_usage_anchor(messages) {
        return est + oob;
    }
    est
}

pub fn token_count_with_estimation(messages: &[Message]) -> i64 {
    let mut i = messages.len() as i64 - 1;
    while i >= 0 {
        let message = &messages[i as usize];
        if let Some(usage) = get_token_usage(message) {
            let total = usage_total(usage);
            if total > 0 {
                // Rewind to the FIRST sibling sharing this response id so
                // interleaved tool_results between parallel-tool splits are
                // included.
                if let Some(response_id) = get_assistant_message_id(message) {
                    let response_id = response_id.to_string();
                    let mut j = i - 1;
                    while j >= 0 {
                        match get_assistant_message_id(&messages[j as usize]) {
                            Some(prior_id) if prior_id == response_id => i = j,
                            Some(_) => break,
                            None => {}
                        }
                        j -= 1;
                    }
                }
                return total
                    + rough_token_count_estimation_for_messages(&messages[(i as usize + 1)..]);
            }
        }
        i -= 1;
    }

    rough_token_count_estimation_for_messages(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_scripts_are_counted_one_token_per_char() {
        // 10 Hangul syllables ~ 10 tokens (1/char), not 2.5.
        let korean = "안녕하세요반갑습니다";
        let t = tally_scripts(korean);
        assert_eq!((t.latinish, t.dense, t.symbol), (0, 10, 0));
        // no model tokenizer in tests -> heuristic path
        if !has_model_tokenizer() {
            assert_eq!(rough_token_count_estimation(korean, 4), 10);
            assert_eq!(rough_token_count_estimation("abcdefgh", 4), 2);
        }
    }

    /// The blocks the old CJK-only list charged at chars/4. Each of these was
    /// under-counted 4x or more, which is the direction that overflows the
    /// server's window rather than compacting early.
    #[test]
    fn the_non_latin_alphabets_are_dense_too() {
        for (name, sample, expect) in [
            ("Cyrillic", "привет", 6),
            ("Greek", "καλημέρα", 8),
            ("Arabic", "مرحبا", 5),
            ("Hebrew", "שלום", 4),
            ("Thai", "สวัสดี", 6),
            ("Devanagari", "नमस्ते", 6),
            // CJK punctuation: the old range began at 0x3040 and left these at
            // chars/4, though they end every Japanese sentence.
            ("JP punctuation", "。、「」", 4),
            // Astral ideographs: the old list stopped at 0xFFEF.
            ("CJK Ext B", "𠀀𠀁", 2),
        ] {
            let t = tally_scripts(sample);
            assert_eq!(t.dense, expect, "{name}: {sample}");
            assert_eq!(t.latinish, 0, "{name} must not fall through to chars/4");
        }
    }

    /// A rendered emoji is a base character plus U+FE0F, and byte-level BPE
    /// splits astral pictographs; 2 tokens each is the floor, chars/4 was not
    /// on the scale.
    #[test]
    fn symbols_and_emoji_are_charged_two() {
        let t = tally_scripts("\u{1F600}\u{2714}\u{FE0F}");
        assert_eq!((t.latinish, t.dense, t.symbol), (0, 0, 3));
        if !has_model_tokenizer() {
            assert_eq!(rough_token_count_estimation("\u{1F600}", 4), 2);
        }
    }

    /// ASCII and accented Latin stay on the chars/4 rule — this is the hot
    /// path (code and English) and must not have moved.
    #[test]
    fn latinish_text_keeps_the_chars_per_token_rule() {
        let t = tally_scripts("fn main() { café }");
        assert_eq!((t.dense, t.symbol), (0, 0));
        assert_eq!(t.latinish, "fn main() { café }".chars().count() as i64);
    }

}

#[cfg(test)]
mod out_of_band_estimate_tests {
    use super::*;
    use crate::types::messages::{ApiMessage, BetaUsage};
    use crate::types::query::AssistantMessage;

    fn user_msg(text: &str) -> Message {
        Message::User(crate::utils::messages::create_user_message_text(text))
    }

    fn assistant_with_usage(input_tokens: i64, output_tokens: i64) -> Message {
        Message::Assistant(AssistantMessage {
            uuid: "u".into(),
            message: ApiMessage {
                id: "a1".into(),
                message_type: "message".into(),
                role: "assistant".into(),
                content: vec![],
                model: "test-model".into(),
                stop_reason: Some("end_turn".into()),
                stop_sequence: None,
                usage: BetaUsage {
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            },
            cost_usd: 0.0,
            duration_ms: 0,
            is_api_error_message: false,
        })
    }

    /// One test, not three: the cases share one scope's registration, and cargo
    /// runs tests on parallel threads, so separate cases would race each
    /// other's setup. Sequencing them here keeps the assertions honest. The
    /// scope name is unique to this test for the same reason.
    #[test]
    fn out_of_band_is_added_only_when_unanchored() {
        const SCOPE: &str = "test:out_of_band_is_added_only_when_unanchored";
        // Anchored: the server already counted the system prompt and the tool
        // block when it produced this usage record, so adding them again would
        // tax every ordinary turn (~25k here) - shorter answers, earlier
        // compaction, no overflow to show for it.
        set_out_of_band_tokens(SCOPE, 30_000);
        let anchored = vec![assistant_with_usage(400_000, 10), user_msg("hi")];
        assert!(has_usage_anchor(&anchored), "a real usage record must anchor");
        assert_eq!(
            estimate_input_tokens(&anchored, SCOPE),
            token_count_with_estimation(&anchored),
            "anchored: out-of-band tokens must NOT be added again"
        );

        // Unanchored: a walk over the history cannot see either part, so the
        // request really is bigger than the history by that much. This is the
        // gap that let a full context sit under the compaction threshold and
        // then be rejected by the server.
        let unanchored = vec![user_msg("hi")];
        assert!(!has_usage_anchor(&unanchored), "no usage record means no anchor");
        assert_eq!(
            estimate_input_tokens(&unanchored, SCOPE),
            token_count_with_estimation(&unanchored) + 30_000,
            "unanchored: the out-of-band parts must be added"
        );

        // A usage record whose counts sum to zero is the server reporting
        // nothing (no final include_usage chunk, or an interrupted stream), not
        // a turn that cost nothing. It must not anchor - same rule the
        // estimator itself applies.
        let zero = vec![assistant_with_usage(0, 0), user_msg("hi")];
        assert!(!has_usage_anchor(&zero), "a zero total is not an anchor");

        set_out_of_band_tokens(SCOPE, 0);
    }
}

// === Prefill estimate (what the server must re-read before it can answer) ===
//
// The cached prefix makes most requests cost only their new tail. When it
// breaks, the same request costs the WHOLE conversation — 1,335 tok/s here, so
// 73k is 55 seconds and 233k is 175 — and the screen looks identical either
// way: a spinner, and seconds. Publishing the number turns "it froze" into
// "it is re-reading 73k tokens", which is the difference between a bug report
// and a wait the user can judge.
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering as AOrd};

static PENDING_PREFILL: AtomicI64 = AtomicI64::new(0);
static PENDING_CACHED: AtomicI64 = AtomicI64::new(0);
/// True once the number came from the server rather than our own estimate.
static PREFILL_MEASURED: AtomicBool = AtomicBool::new(false);
static OUTPUT_STARTED: AtomicBool = AtomicBool::new(false);

/// Our own estimate, from comparing this request against the previous one.
/// Available immediately; superseded by the server's number when it arrives.
pub fn set_pending_prefill(tokens: i64) {
    if PREFILL_MEASURED.load(AOrd::Relaxed) {
        return; // never walk a measured value back to a guess
    }
    PENDING_PREFILL.store(tokens.max(0), AOrd::Relaxed);
}

/// vLLM's own accounting: `compute` tokens must be computed, `cached` were
/// served from the prefix cache (services::prefill_probe).
pub fn set_pending_prefill_measured(compute: i64, cached: i64) {
    PENDING_PREFILL.store(compute.max(0), AOrd::Relaxed);
    PENDING_CACHED.store(cached.max(0), AOrd::Relaxed);
    PREFILL_MEASURED.store(true, AOrd::Relaxed);
}

/// Call at the start of every request: the previous turn's numbers must not
/// leak into this one's display.
/// Serialises the tests that drive the spinner's process-global readout state
/// (prefill counters, the wait note, CODEMASON_DEBUG).
///
/// Those are one set of globals shared by every test in the binary, and
/// `cargo test` runs them on many threads: one test clearing the wait note
/// while another asserts on it produced a failure that only appeared as the
/// suite grew. Same shape as the sampling-env guard — recover from a poisoned
/// lock, because a poisoned lock only means some other test panicked while
/// holding it, and each of these resets the state it touches anyway.
#[cfg(test)]
pub fn spinner_state_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn reset_prefill_state() {
    PENDING_PREFILL.store(0, AOrd::Relaxed);
    PENDING_CACHED.store(0, AOrd::Relaxed);
    PREFILL_MEASURED.store(false, AOrd::Relaxed);
    OUTPUT_STARTED.store(false, AOrd::Relaxed);
}

pub fn mark_output_started() {
    OUTPUT_STARTED.store(true, AOrd::Relaxed);
}

pub fn output_started() -> bool {
    OUTPUT_STARTED.load(AOrd::Relaxed)
}

pub fn pending_prefill() -> i64 {
    PENDING_PREFILL.load(AOrd::Relaxed)
}

pub fn pending_cached() -> i64 {
    PENDING_CACHED.load(AOrd::Relaxed)
}

pub fn prefill_is_measured() -> bool {
    PREFILL_MEASURED.load(AOrd::Relaxed)
}

// === What a tool is currently waiting on ===
//
// A tool that waits shows nothing about WHY. web_fetch gives a slow host 60
// seconds, web_search retries three times with backoff (51s worst case), and
// bash runs 120 seconds in the foreground before moving to the background —
// all of it behind an unchanging spinner. The tool label answers "which tool";
// this answers "waiting for what", which is the part that decides whether the
// wait is worth sitting through.
//
// A single slot, not a stack: tools that can run concurrently are the
// read-only ones that finish fast, and the last writer is the one still going.
static WAIT_NOTE: Lazy<parking_lot::Mutex<Option<String>>> =
    Lazy::new(|| parking_lot::Mutex::new(None));

/// Say what this tool is waiting on; None clears it. Keep it short — it shares
/// one terminal line with the elapsed time and token counts.
pub fn set_wait_note(note: Option<String>) {
    *WAIT_NOTE.lock() = note;
}

pub fn wait_note() -> Option<String> {
    WAIT_NOTE.lock().clone()
}

/// Clears the wait note on drop, so an early return or a `?` cannot leave a
/// stale "fetching ..." on screen after the tool has finished.
pub struct WaitNoteGuard;

impl Drop for WaitNoteGuard {
    fn drop(&mut self) {
        set_wait_note(None);
    }
}
