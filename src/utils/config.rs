//! Configuration management for CodeMason
//! Equivalent to codemason/utils/config.py (itself from src/utils/config.ts)

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────
// SINGLE SOURCE OF TRUTH for every codemason config/state location.
// One env var (CODEMASON_CONFIG_DIR) relocates the whole tree; default ~/.codemason.
// All modules (env, log, hooks, keep_going_installer) import these — never
// redefine a path elsewhere.
// ─────────────────────────────────────────────────────────────────────────
pub static CONFIG_DIR: Lazy<PathBuf> = Lazy::new(|| {
    // A test run must never write to the operational config dir — the same rule
    // LOG_DIR has, for the same reason, and now a sharper one: `cargo test`
    // reached get_config() (through the sub-agent gate, which asks the config
    // which endpoint's pool to use) and so could rewrite the user's real
    // config.json, drop lock files in their locks/ and leave a goal in a project
    // directory. It is also what makes tests deterministic: a Lazy resolves once,
    // so whichever test set CODEMASON_CONFIG_DIR first silently decided the dir
    // for all the others. The Python suite isolates the same way, per test
    // process (CODEMASON_CONFIG_DIR into a throwaway dir).
    if cfg!(test) {
        return std::env::temp_dir().join(format!("codemason-test-config-{}", std::process::id()));
    }
    match std::env::var("CODEMASON_CONFIG_DIR") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".codemason"),
    }
});
pub static CONFIG_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("config.json"));
pub static SETTINGS_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("settings.json"));
pub static MEMORY_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("memory"));
/// Where the logs go. CODEMASON_LOG_DIR relocates them; `-dbg` points it at a
/// world-readable /tmp directory so a second pair of eyes can tail the run
/// live instead of asking the user to copy lines out of a terminal.
pub static LOG_DIR: Lazy<PathBuf> = Lazy::new(|| {
    // A test run must never write to the operational log. `cargo test` used to
    // append straight into ~/.codemason/logs/error.log, and its fixtures are
    // indistinguishable from real failures: 91 "probe-panic", 86
    // "worker-panic", and 36 Write calls truncated at column 9 — the fixture
    // string `{"file_pa` — all of which were read back on 2026-09-07 as
    // evidence of a live truncation bug that did not exist, and sent this
    // investigation down a day-long detour. Isolate them by construction.
    if cfg!(test) {
        return std::env::temp_dir().join(format!("codemason-test-logs-{}", std::process::id()));
    }
    match std::env::var("CODEMASON_LOG_DIR") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => CONFIG_DIR.join("logs"),
    }
});
pub static HOOKS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("hooks"));
/// Every inter-process lock in the system lives here, named after what it
/// protects, so there is one directory to audit and no .lock files scattered
/// through the tree the user reads. The files are empty and are never deleted —
/// see utils::file::FileLock for why that is the safe shape and not residue.
pub static LOCKS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("locks"));

/// Short fingerprint of a model endpoint, for keying anything whose real owner
/// is the SERVER rather than the user or the session.
///
/// Hashed rather than used raw because a URL is not a filename, and normalized
/// first (trim, drop a trailing '/') so the same server written two ways is one
/// tag. The Python client computes this identically — anything keyed by it is
/// shared between the builds, so a one-byte difference would split the namespace
/// in half without any error.
///
/// Case is folded over the scheme and host, which are case-insensitive, so
/// "http://LocalHost:8000" and "http://localhost:8000" are one pool rather than
/// two sessions rationed against one GPU as if it were two. The PATH keeps its
/// case: a path is case-sensitive, and merging two endpoints that differ there
/// would make each wait on a cache the other never touches.
///
/// Spellings that differ by more than case (localhost vs 127.0.0.1, a proxy vs
/// the server behind it) read as two endpoints no matter what; point both
/// sessions at one string if they share a server.
pub fn endpoint_tag(base_url: Option<&str>) -> String {
    let owned = match base_url {
        Some(u) => u.to_string(),
        None => get_config().model_base_url.clone(),
    };
    let url = fold_endpoint(owned.trim().trim_end_matches('/'));
    if url.is_empty() {
        return "none".to_string();
    }
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(url.as_bytes());
    hex::encode(h.finalize())[..12].to_string()
}

/// `url` with its scheme and host lowercased and its path left alone.
///
/// Hand-rolled rather than handed to a URL parser so the Python build can land
/// on the same bytes from the same few lines: find the authority, fold that,
/// copy the rest. The separators are ASCII, so both builds slice at the same
/// index.
///
/// Folding is ASCII-only on purpose. The two builds must agree byte for byte,
/// and full Unicode lowercasing has cases (the Turkish dotted I) where Rust's
/// to_lowercase and Python's str.lower need not — exactly the one-byte
/// difference endpoint_tag warns about. Every byte either build folds is ASCII.
fn fold_endpoint(url: &str) -> String {
    let start = match url.find("://") {
        Some(p) => p + 3,
        None => 0,
    };
    let cut = match url[start..].find('/') {
        Some(q) => start + q,
        None => url.len(),
    };
    format!("{}{}", url[..cut].to_ascii_lowercase(), &url[cut..])
}

/// The lock sidecar for a shared file. `name` is what it protects
/// ("config.json"); the Python client derives the same path from the same name,
/// so the two builds exclude each other and not just their own kind.
pub fn lock_path(name: &str) -> PathBuf {
    LOCKS_DIR.join(format!("{name}.lock"))
}

/// The lock the config writer takes. Its own function so the test pins the path
/// the writer actually uses, rather than one spelled the same way twice.
pub fn config_lock_path() -> PathBuf {
    lock_path(
        &CONFIG_FILE
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "config.json".to_string()),
    )
}

/// Legacy config.json keys -> their neutral replacements (the llamacpp_* ->
/// model_* rename). Consumed by `Config::from_value_checked` so an older on-disk config
/// still loads.
const LEGACY_CONFIG_KEYS: &[(&str, &str)] = &[
    ("llamacpp_base_url", "model_base_url"),
    ("llamacpp_model", "model_name"),
    ("llamacpp_api_key", "model_api_key"),
    ("llamacpp_aes_key", "model_aes_key"),
];

/// Return the first environment variable that is set among `names` (new
/// canonical name first, legacy alias last), else `default`. Lets the renamed
/// CODEMASON_MODEL_* vars take over while the old LLAMACPP_* still resolve, so
/// an existing .env / launch script keeps working unchanged.
fn env_first(default: &str, names: &[&str]) -> String {
    for n in names {
        if let Ok(v) = std::env::var(n) {
            return v;
        }
    }
    default.to_string()
}

/// The on-disk schema version this build writes. Bump it when a default changes
/// in a way an existing config.json must pick up, and add the step to
/// MIGRATIONS. A file without the key predates versioning and counts as 1.
/// Identical to CONFIG_VERSION in codemason/utils/config.py.
pub const CONFIG_VERSION: i64 = 3;

/// Defaults that were later found to be defects. A config written by an older
/// version holds them as if the user had chosen them — first run materializes
/// every default into the file — so a fixed default would never reach that
/// user. Each step replaces a value ONLY when it still equals what the old
/// version wrote; anything else is the user's choice and is left alone. Runs
/// once per file: the version is stamped forward afterwards.
/// (version, key, old values as JSON, new value as JSON)
const MIGRATIONS: &[(i64, &str, &[&str], &str)] = &[
    // Truncated long reasoning turns mid-answer (default until 2026-08-13).
    (2, "max_tokens", &["16384"], "65536"),
    // A turn cap that silently ended -dsp runs at turn 200 (default for 2h40m
    // on 2026-09-04).
    (2, "max_turns", &["200"], "null"),
    // "Send no effort" (default until 2026-08-06); policy is high, always.
    (2, "reasoning_effort", &["\"\""], "\"high\""),
    // This build's old name for auto (default until 2026-10-05).
    (2, "sampling_profile", &["\"vendor\""], "\"auto\""),
    // The serial line's slot count. 1 was its default and 2 its only other
    // sensible value there; on this line the tools are concurrency-safe, so a
    // file carried over from main would let the code dispatch three spawns and
    // the gate still admit one — parallel on paper, serial in fact, with nothing
    // reporting it. Anything at 3 or above is a deliberate choice, left alone.
    (3, "max_concurrent_subagents", &["1", "2"], "3"),
];

/// Fields whose default is null: the JSON kind a non-null value must have.
const OPTIONAL_FIELD_KINDS: &[(&str, &str)] =
    &[("max_turns", "int"), ("language", "str"), ("ui_language", "str")];

/// A config.json value read as the kind `default` has, or None if it cannot be.
///
/// Someone who hand-edits config.json without knowing the schema writes
/// "false" for false and "16384" for 16384. Those spellings are read as what
/// they mean (and agree with the Python build, where "false" was a truthy
/// string that turned permission prompts OFF); anything else is reported and
/// left to the default.
fn coerce_field(name: &str, value: &Value, default: &Value) -> Option<Value> {
    let kind = match OPTIONAL_FIELD_KINDS.iter().find(|(k, _)| *k == name) {
        Some((_, kind)) => {
            if value.is_null() {
                return Some(Value::Null);
            }
            *kind
        }
        None => match default {
            Value::Bool(_) => "bool",
            // usize on the struct: a negative count would fail to deserialize.
            Value::Number(_) if name == "max_concurrent_subagents" => "uint",
            Value::Number(n) if n.is_i64() || n.is_u64() => "int",
            Value::Number(_) => "float",
            Value::String(_) => "str",
            Value::Object(_) => "obj",
            _ => return Some(value.clone()),
        },
    };
    let as_f64 = |v: &Value| -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }
    };
    match kind {
        "bool" => match value {
            Value::Bool(b) => Some(Value::Bool(*b)),
            Value::Number(n) if n.as_f64() == Some(0.0) || n.as_f64() == Some(1.0) => {
                Some(Value::Bool(n.as_f64() == Some(1.0)))
            }
            Value::String(s) => match s.trim().to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(Value::Bool(true)),
                "false" | "0" | "no" | "off" => Some(Value::Bool(false)),
                _ => None,
            },
            _ => None,
        },
        "int" | "uint" => {
            if value.is_boolean() {
                return None;
            }
            let f = as_f64(value)?;
            if f.fract() != 0.0 || !f.is_finite() || (kind == "uint" && f < 0.0) {
                return None;
            }
            Some(Value::from(f as i64))
        }
        "float" => {
            if value.is_boolean() {
                return None;
            }
            as_f64(value).and_then(serde_json::Number::from_f64).map(Value::Number)
        }
        "str" => match value {
            Value::String(_) => Some(value.clone()),
            Value::Number(n) => Some(Value::String(n.to_string())),
            _ => None,
        },
        "obj" => value.is_object().then(|| value.clone()),
        _ => Some(value.clone()),
    }
}

/// A value the file held that could not be read: (key, file value, default).
pub type BadValue = (String, Value, Value);

/// Move a file written by an older version past defaults that were fixed.
/// Returns (key, old, new) for every value changed and stamps the version
/// (never lowered, so a newer build's file keeps its number).
fn apply_migrations(cfg: &mut Config, data: &Value) -> Vec<(String, Value, Value)> {
    let Value::Object(on_disk) = data else {
        return Vec::new();
    };
    let file_version = on_disk
        .get("config_version")
        .and_then(|v| if v.is_boolean() { None } else { v.as_i64() })
        .unwrap_or(1);
    let mut notes = Vec::new();
    let mut map = match cfg.to_value() {
        Value::Object(m) => m,
        _ => return notes,
    };
    for (version, key, olds, new) in MIGRATIONS {
        if file_version >= *version || !on_disk.contains_key(*key) {
            continue;
        }
        let current = map.get(*key).cloned().unwrap_or(Value::Null);
        let is_old = olds
            .iter()
            .any(|o| serde_json::from_str::<Value>(o).map_or(false, |o| values_equal(&o, &current)));
        if is_old {
            let new_v: Value = serde_json::from_str(new).unwrap_or(Value::Null);
            notes.push((key.to_string(), current, new_v.clone()));
            map.insert(key.to_string(), new_v);
        }
    }
    map.insert("config_version".into(), Value::from(file_version.max(CONFIG_VERSION)));
    if let Ok(parsed) = serde_json::from_value::<Config>(Value::Object(map)) {
        *cfg = parsed;
    }
    notes
}

/// JSON equality that treats 16384 and 16384.0 as the same number.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

/// CodeMason configuration.
///
/// Field-by-field commentary lives in the Python reference
/// (codemason/utils/config.py); the values and semantics here are identical.
/// Sampling knobs for one request. `None` = omit the key entirely, so the
/// server's own default applies rather than a value we invented.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    // Schema version of this file (see CONFIG_VERSION / MIGRATIONS). Not a
    // setting: it records which defaults the file has already been moved past.
    pub config_version: i64,

    // Model-server settings (backend-agnostic — llama.cpp or vLLM, selected by
    // `backend` below). Defaults are the live GLM-5.3-Flash endpoint so a fresh
    // checkout just works with zero setup.
    pub model_base_url: String,
    pub model_name: String,
    pub model_api_key: String,

    // Optional symmetric key (base64 or hex, 32 bytes) for end-to-end BODY
    // encryption with the vLLM endpoint (see utils/body_crypto.rs).
    pub model_aes_key: String,

    // Model server backend: 'vllm' | 'llamacpp' | 'auto'.
    pub backend: String,

    // Directory of the served model (tokenizer.json => exact token counting).
    pub model_dir: String,

    // LLM request parameters. See the Python reference for the measured
    // rationale behind every value; do not "neutralise" them.
    pub max_tokens: i64,
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    /// 0, and not by preference: DeepSeek-V4's own guidance (Unsloth's "how to
    /// run locally" page for the V4 family) recommends temperature 1.0, top_p
    /// 1.0 and min_p 0.01. The last is not reachable here — any min_p > 0 is
    /// rejected outright (400) under MTP speculative decoding, in vLLM's
    /// _validate_spec_decode — and MTP is worth more than the recommendation.
    /// A thing given up knowingly; recorded because the reason lived only in
    /// somebody's memory for months. (Turning MTP off would make 0.01 available.)
    pub min_p: f64,
    pub repetition_penalty: f64, // 1.0 = off
    pub presence_penalty: f64,   // 0.0 = off
    pub frequency_penalty: f64,  // 0.0 = off

    // "auto"  = per-family vendor defaults win (see effective_sampling).
    // "manual" = send the stored values above verbatim, whatever the model is.
    // Anything else is treated as "auto".
    pub sampling_profile: String,

    // Degenerate-run guard: abort a generation once the last N characters of
    // its output collapse to almost no distinct characters. 0 disables it.
    // See openai_common::StreamState::note_output for why this exists and why
    // it cannot fire on real work.
    pub degenerate_run_chars: i64,

    // Stall guards, in seconds. 0 disables either one.
    //
    // first_token: how long a request may produce NOTHING before it is cut.
    // Generous, because prefill legitimately scales with the uncached prompt —
    // backend.rs measured a real 1M-context compaction at ~410s before its
    // first token — so this is 1.5x that, not a tight bound.
    //
    // token_gap: how long a stream that HAS started may then go silent. Once
    // tokens are flowing, a two-minute hole is pathological, not slow.
    //
    // Observed 2026-09-10: a request whose prefill completed normally (1,541
    // new + 84,480 cached, measured) then produced zero tokens for 21 minutes.
    // Nothing noticed. The HTTP read_timeout is 1800s and only fires when no
    // BYTES arrive at all, so an SSE keep-alive would have held it open past
    // that too; the turn ended when the operator pressed ESC.
    pub first_token_timeout_secs: f64,
    pub token_gap_timeout_secs: f64,

    // Server-side loop detection (vLLM RepetitionDetectionParams).
    pub repetition_min_pattern: i64,
    pub repetition_max_pattern: i64,
    pub repetition_min_count: i64,
    // Per-model on/off for the loop detector, keyed by served model id.
    // Missing key => the family default in repetition_detection_enabled().
    pub repetition_detection_per_model: std::collections::HashMap<String, bool>,

    // Thinking (reasoning) effort carried on every request. Empty = "don't
    // send reasoning_effort" (defer to server default).
    pub reasoning_effort: String,
    // Per-model override of reasoning_effort, keyed by served model id
    // (config.model_name). Policy 2026-09-03: DSV4-Flash = high always;
    // glm-5.3-flash = high or max. Missing key => global reasoning_effort.
    pub reasoning_effort_per_model: std::collections::HashMap<String, String>,
    // Ask the server's chat template to DROP reasoning from earlier user turns
    // (clear_thinking). In-turn (tool-loop) reasoning is always kept — that is
    // the round-trip that makes the model smart. Prior-turn reasoning is what
    // made GLM-5.3 cling to old conclusions ("변심 없음, 확신" on every answer).
    // DeepSeek-V4's template drops it by default and ignores this key.
    // Policy 2026-09-03: default TRUE.
    pub clear_thinking: bool,

    // Remote notifications (codemason.remote).
    pub remote_notify_after: f64,
    pub remote_retention_days: f64,
    pub remote_ask_timeout: f64,
    pub remote_replay_backlog: bool,
    pub input_highlight: bool,
    pub input_highlight_bg: String,
    pub input_highlight_fg: String,
    pub remote_file_threshold: i64,
    pub remote_collapse_over: i64,
    pub remote_file_ext: String,
    pub remote_excerpt: i64,
    /// Read another product's config directories (~/.claude, <cwd>/.claude) for
    /// skills and sub-agent definitions. OFF by default: this is our program,
    /// and another tool's layout is an opt-in convenience for someone migrating,
    /// not something to inherit silently. Env: CODEMASON_IMPORT_CLAUDE_DIRS.
    pub import_claude_dirs: bool,
    pub remote_enabled: bool,
    pub remote_table_width: i64,

    // UI settings
    pub theme: String,
    pub verbose: bool,
    pub realtime: bool,

    // Permission settings
    pub dangerously_skip_permissions: bool,

    // Feature flags
    pub auto_compact: bool,
    // Spinner label while a tool runs. Default: the tool's name ("Bash…").
    //
    // Upstream picks a decorative verb at random per turn ("Bloviating",
    // "Boondoggling") on the reasoning that the tool is already named on the
    // line above. In a long autonomous run that line has scrolled away, and
    // then the only thing on screen is a word chosen at random — which reads
    // as the model doing something arbitrary rather than a tool waiting.
    // Naming the tool costs nothing and answers the question being asked.
    // Set true for upstream's decorative behaviour.
    // Env: CODEMASON_SPINNER_RANDOM_VERB.
    pub spinner_random_verb: bool,
    // Absolute ceiling for auto-compaction, in tokens; 0 = window-derived only.
    //
    // The window-derived threshold lets ONE conversation grow to nearly the
    // whole model window. Against a server whose KV pool holds ~1.6 windows
    // (measured 2026-09-07: 847,533 tokens for a 524,288 window) that leaves
    // no room for a sub-agent: the moment one starts working, the parent's
    // blocks are evicted, and when the sub-agent returns the parent re-prefills
    // from zero — 200s at 267k. Two tenants can only coexist if
    // 2 * threshold < pool, and only the operator knows the pool.
    pub auto_compact_threshold_tokens: i64,
    // How many sub-agents (Agent tool, forked skills) may run at once, across
    // every codemason process on this machine — each is a separate cache
    // tenant competing with the parent for the same KV pool.
    //
    // 3 on this line, where the tools are concurrency-safe and a spawn can
    // actually overlap. Measured 2026-10-10 against the 2x-Spark lane at KV 12GB
    // (1,356,053 tokens): a 400,708-token parent still answered from cache at
    // 98.9% after three 133k sub-agents ran to completion, and three agents
    // finished the same work in 76.0s where they took 121.5s one at a time.
    // Against a smaller pool this is the number to lower — subagent_gate's
    // doc comment says how to read it.
    pub max_concurrent_subagents: usize,
    // Context ceiling for SUB-AGENTS only, in tokens. 0 (the default) does NOT
    // mean unlimited: it derives one from the parent's own threshold
    // (SUBAGENT_THRESHOLD_FRACTION in services::compaction), so an untuned
    // install still keeps the sub-agent small enough not to evict the parent.
    // Set a positive value to override that derivation — do that only from a
    // measurement on the server you actually serve against, never from
    // arithmetic over its advertised numbers (see below).
    //
    // The parent must be free to use the whole window — a 512K conversation is
    // the point of a 512K model. The sub-agent is what has to fit in what is
    // left. Blocks, not tokens, are the currency: a conversation of L tokens
    // holds L/4608 attention blocks plus L/VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    // Mamba checkpoints, out of a 215-block pool on this deployment. Mamba
    // state does not accumulate while a request runs; each superseded block is
    // freed as the state advances (mamba_cache_mode "align"), so only the
    // retained checkpoints scale with length.
    //
    // Measured 2026-09-13 at retention 18432 with the parent parked at 441,165
    // tokens: a concurrent conversation of 270,000 left the parent fully
    // cached in 10 of 10 trials; 280,000 cost it one attention block every
    // time; 290,000 survived once and wiped it twice; 296,945 and up wiped it
    // outright with a ~360s re-prefill. Eviction near the cliff is
    // all-or-nothing — a Mamba hit needs the single state block ending on the
    // boundary, so one lost checkpoint zeroes the whole hit — and the boundary
    // is probabilistic, so a value is only safe when it survives repeatedly,
    // never on one green run.
    //
    // Note what this ceiling has to cover: a sub-agent's peak footprint is this
    // value PLUS the output that turn generates, so budget
    // ceiling + subagent_max_tokens against the measured limit, not the ceiling
    // alone. This deployment runs 230,000 + 32,768 = 262,768 against 270,000.
    //
    // Do NOT derive a value from the server's "GPU KV cache size: N tokens"
    // log line. That N is int(max_concurrency * max_model_len) where
    // max_concurrency is pool_blocks / blocks_for_one_max_len_request; it
    // describes one request's worst case, not how much conversation can sit in
    // the cache at once, and reading it as the latter is what produced the
    // wrong numbers this comment replaces.
    pub subagent_context_ceiling_tokens: i64,
    // Output budget for SUB-AGENT requests, in tokens; 0 (the default) = same
    // as the parent's max_tokens.
    //
    // Worth setting below max_tokens because the two reserves are paid for
    // differently. A sub-agent's peak KV footprint is its compaction ceiling
    // PLUS whatever that turn generates, so the full max_tokens has to be held
    // back from subagent_context_ceiling_tokens even though it is almost never
    // spent: measured over this deployment's turn log the median output is 81
    // tokens and the largest 1,860. What justifies reserving it at all is the
    // tail, not the mean — 2026-09-10 the model collapsed onto one token and
    // rode it to max_tokens twice, 53,859 consecutive '!' for 63 minutes a turn.
    //
    // So cap the sub-agent instead of hoping: a smaller budget shrinks the
    // reserve one-for-one and buys the same number of tokens back as working
    // context, which is what a sub-agent that reads whole sources actually
    // needs. Nothing is truncated by the cap — hitting it yields
    // stop_reason=max_tokens and the recovery path resumes the generation — but
    // leave headroom for reasoning, which is billed as output and runs long at
    // high effort.
    pub subagent_max_tokens: i64,

    // Legacy encoding to run a SECOND ripgrep pass in, or empty to skip it.
    //
    // ripgrep's automatic detection only looks at a BOM, so a CP949 file is
    // searched as if it were UTF-8 and Hangul in it matches nothing. Passing
    // -E is not a fix on its own either — measured, `-E euc-kr` finds the CP949
    // file and MISSES the UTF-8 one, so it trades one blind spot for another.
    // Running both passes and merging is what actually covers a mixed tree, and
    // it is nearly free: measured 4ms -> 8ms over this repository, against a
    // model turn that costs seconds.
    //
    // On by default for that reason — a setting nobody knows about would leave
    // the feature unreachable where users meet it. Empty turns the second pass
    // off. Env: CODEMASON_GREP_LEGACY_ENCODING.
    pub grep_legacy_encoding: String,

    // Wall-clock cap on ONE background shell task, in seconds. 0 disables it.
    //
    // A background task has no other brake. The foreground Bash tool caps at 10
    // minutes and then ADOPTS the still-running process as a background task, so
    // the command the model expected to be quick is exactly the one that ends up
    // here unbounded. The disk cap (MAX_BG_OUTPUT_BYTES) already stops it filling
    // the disk; this stops it holding a core forever.
    //
    // Six hours, not something tight: this is a runaway guard, not a scheduling
    // policy. A real build, test suite or dev server has to finish comfortably
    // inside it, because killing legitimate work is a worse failure than the
    // runaway it prevents. Raise it, or set 0, for a job that genuinely runs
    // longer. Env: CODEMASON_BACKGROUND_TASK_TIMEOUT.
    pub background_task_timeout: f64,

    // Training cutoff of the served model, e.g. "March 2025". Empty (default)
    // leaves the "Assistant knowledge cutoff is ..." line out of the environment
    // block entirely, which is what happened before this existed: the cutoff was
    // looked up in a hardcoded table of another product's model ids, so it never
    // matched anything we serve and the line was never emitted. Fill it in for
    // the model you actually run. Env: CODEMASON_MODEL_KNOWLEDGE_CUTOFF.
    pub model_knowledge_cutoff: String,

    // Engine-side runaway guard (QueryParams.maxTurns): cap on the number of
    // recursive tool turns in a single user prompt. Env: CODEMASON_MAX_TURNS.
    //
    // This is a BACKSTOP, not the main defence. Two cheaper guards run first and
    // catch the common loops: an identical (tool, args) call is refused after
    // REPEATED_TOOL_CALL_LIMIT, and Layer-2 repetition recovery rewrites a turn
    // that keeps repeating itself. What neither sees is a loop whose arguments
    // change every time — reading one more file, forever — and that is what this
    // stops.
    //
    // OFF by default, and that is the considered answer, not an oversight.
    //
    // I set it to 200 during the audit, reasoning that -dsp plus keep-going left
    // nothing between a session and an unbounded loop. That reasoning ignored
    // what -dsp is FOR here: the user starts a long job, walks away for hours,
    // and comes back to the result. A cap does not prompt — it silently ENDS the
    // turn mid-task. So the failure it creates is exactly the one that hurts
    // most in the only mode where it would ever fire: hours later, the work is
    // half done, and the transcript just says it stopped.
    //
    // The loops it was meant to catch are already caught, and more cheaply: an
    // identical (tool, args) call is refused after REPEATED_TOOL_CALL_LIMIT, and
    // Layer-2 repetition recovery rewrites a turn that keeps repeating itself.
    // What a turn cap adds over those is a loop whose arguments change every
    // time — and paying for that with "your six-hour run stopped at turn 200" is
    // not a trade worth making.
    //
    // Set it to a number if you want the cap for a specific run. A null on disk
    // means unset, which is off.
    pub max_turns: Option<i64>,

    // "keep-going" autonomy (Stop hook nudge). Persistent default; a session
    // can override via --keep-going / --no-keep-going / `keepgoing on|off`.
    pub keep_going: bool,

    // Auto-run the active goal's completion gates when a turn ends.
    pub goal_gate_autorun: bool,

    // Preferred response language (settings.language). None = model default.
    pub language: Option<String>,
    // The program's own language — menus, status, alerts. Empty follows
    // `language`, then the OS locale, then English.
    pub ui_language: Option<String>,

    // WebSearch: self-hosted SearXNG as the 0-priority backend.
    pub searxng_url: String,
    pub searxng_timeout: f64,
    pub searxng_lang: String,
    pub searxng_api_key: String,

    // WebSearch rate-limit posture (ddgs fallback path).
    pub websearch_backends: String,
    pub websearch_max_expansions: i64,
    pub websearch_throttle_min: f64,
    pub websearch_throttle_max: f64,
    pub websearch_retry_attempts: i64,
    pub websearch_backoff_base: f64,
    pub websearch_backoff_cap: f64,
    pub websearch_cache_ttl: i64,
    pub websearch_cache_dir: String,
}

impl Config {
    /// reasoning_effort to send for the current model: per-model override
    /// (reasoning_effort_per_model[model_name]) else the global value.
    pub fn effective_reasoning_effort(&self) -> String {
        let m = self.model_name.trim();
        if !m.is_empty() {
            if let Some(v) = self.reasoning_effort_per_model.get(m) {
                return v.trim().to_lowercase();
            }
        }
        self.reasoning_effort.clone()
    }

    /// The sampling knobs actually sent for one request.
    ///
    /// `None` means "omit the key", which hands that decision to the server.
    ///
    /// THREE SOURCES, IN THIS ORDER OF AUTHORITY
    ///
    /// 1. The environment wins outright. A knob exported for a run is a
    ///    deliberate, per-run instruction and outranks everything below.
    /// 2. `sampling_profile = "manual"` means the operator has taken the wheel:
    ///    config.json is sent verbatim, whatever the served model is.
    /// 3. `sampling_profile = "auto"` (the default, and what any unrecognised
    ///    value falls back to) means the client decides, per family. config.json
    ///    is NOT consulted — that is the point of auto. Someone who just wants
    ///    the thing to work should not have to learn which number belongs to
    ///    which model, and a config file that fills itself in cannot tell a
    ///    value someone chose from one it wrote itself.
    ///
    /// THE AUTO TABLE
    ///
    /// GLM: the vendor's published numbers — z.ai's "Migrate to GLM-5.3" guide
    /// and the zai-org/GLM-5.3-Flash card both give temperature 1.0 with top_p
    /// 0.95, and the guide adds "it is not recommended to adjust both
    /// simultaneously". No top_k (40 narrows the candidate set on top of top_p,
    /// which is exactly the "adjust both" the vendor warns against), no
    /// penalties (the vendor recommends none).
    ///
    /// Everything else, DeepSeek-V4 above all: temperature 0.6 with top_k 40 and
    /// top_p 0.95. Not a vendor number — it is what this deployment has run on
    /// for months and what gets handed to anyone setting the client up against
    /// the same model. A recommendation earns its place by having been run, not
    /// by being published.
    ///
    /// min_p stays 0 in both, and that is a concession, not a preference:
    /// DeepSeek-V4's own guidance (Unsloth's run-locally page for the V4 family)
    /// asks for temperature 1.0, top_p 1.0 and min_p 0.01. min_p > 0 is rejected
    /// outright (400) under MTP speculative decoding, in vLLM's
    /// _validate_spec_decode, and MTP is worth more than the recommendation.
    /// Turning MTP off is what would make 0.01 available. (The DeepSeek
    /// temperature here is 0.6 rather than the recommended 1.0 for the reason
    /// above it: measured beats published on the deployment it is about.)
    ///
    /// The penalties stay off in both: a multiplicative repetition penalty
    /// suppresses the structurally-identical CLOSING tags of tool calls and
    /// leaves them unclosed. Reach for them through "manual".
    ///
    /// (Measured 2026-09-14 on the live GLM deployment, counting answers
    /// containing U+FFFD: temperature 1.0 broke 7 of 8, 0.6 + top_k 40 broke
    /// 1 of 8. That is a symptom rate, not a cause — the actual defect was the
    /// sampler letting a Latin token follow the lead byte of a Korean syllable,
    /// fixed server-side — so it is not a reason to hold GLM off its vendor
    /// number. The trace is in the lane's UTF8-GUARD.md.)
    pub fn effective_sampling(&self) -> Sampling {
        let mut s = if self.sampling_profile.trim().eq_ignore_ascii_case("manual") {
            Sampling {
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: Some(self.top_k),
                min_p: Some(self.min_p),
                repetition_penalty: Some(self.repetition_penalty),
                presence_penalty: Some(self.presence_penalty),
                frequency_penalty: Some(self.frequency_penalty),
            }
        } else {
            Self::recommended_sampling(&self.model_name)
        };
        Self::apply_sampling_env(&mut s);
        s
    }

    /// The client's own recommendation for a served model — the auto table.
    pub fn recommended_sampling(model_name: &str) -> Sampling {
        if model_name.trim().to_lowercase().contains("glm") {
            Sampling {
                temperature: 1.0,
                top_p: 0.95,
                top_k: None,
                min_p: Some(0.0),
                repetition_penalty: None,
                presence_penalty: None,
                frequency_penalty: None,
            }
        } else {
            Sampling {
                temperature: 0.6,
                top_p: 0.95,
                top_k: Some(40),
                min_p: Some(0.0),
                repetition_penalty: None,
                presence_penalty: None,
                frequency_penalty: None,
            }
        }
    }

    /// The environment has the last word, in every profile.
    ///
    /// Both the bare name and a CODEMASON_-prefixed one are read, the prefixed
    /// one first: TEMPERATURE is conventional enough to collide with something
    /// else in a shell, and someone who needs to be unambiguous should have a
    /// way to say so.
    fn apply_sampling_env(s: &mut Sampling) {
        fn f64_env(name: &str) -> Option<f64> {
            for key in [format!("CODEMASON_{name}"), name.to_string()] {
                if let Ok(v) = std::env::var(&key) {
                    let v = v.trim().to_string();
                    if v.is_empty() {
                        continue;
                    }
                    if let Ok(parsed) = v.parse::<f64>() {
                        return Some(parsed);
                    }
                }
            }
            None
        }
        if let Some(v) = f64_env("TEMPERATURE") {
            s.temperature = v;
        }
        if let Some(v) = f64_env("TOP_P") {
            s.top_p = v;
        }
        if let Some(v) = f64_env("TOP_K") {
            // A negative or zero top_k is how a shell says "do not send it".
            s.top_k = if v >= 1.0 { Some(v as i64) } else { None };
        }
        if let Some(v) = f64_env("MIN_P") {
            s.min_p = Some(v);
        }
        if let Some(v) = f64_env("REPETITION_PENALTY") {
            s.repetition_penalty = Some(v);
        }
        if let Some(v) = f64_env("PRESENCE_PENALTY") {
            s.presence_penalty = Some(v);
        }
        if let Some(v) = f64_env("FREQUENCY_PENALTY") {
            s.frequency_penalty = Some(v);
        }
    }

    /// Whether to ask the server for loop detection on the CURRENT model.
    ///
    /// The detector was measured into place on 2026-08-06 against the DeepSeek
    /// substrate, where a forced 40x repetition was cut at ~140 tokens. Its
    /// stated premise is that "legitimate repetition is not n-gram-identical".
    ///
    /// GLM-5.3-Flash breaks that premise in ordinary work. Enumerating an API
    /// means holding a command skeleton fixed and varying one field, so six
    /// consecutive tool calls are byte-identical for a hundred-odd tokens
    /// apiece while the ANSWERS advance. Observed 2026-09-06: a live probe of a
    /// router's ubus endpoints was cut mid-enumeration having just established
    /// that three of them accept unauthenticated calls — real progress, judged
    /// a loop on shape alone. The recovery nudges then told a model that had
    /// done nothing wrong to stop repeating itself, six times, and the turn
    /// ended.
    ///
    /// So the default is per family: on for DeepSeek, off for GLM. An explicit
    /// entry in repetition_detection_per_model overrides either way.
    pub fn repetition_detection_enabled(&self) -> bool {
        let m = self.model_name.trim().to_lowercase();
        if let Some(v) = self.repetition_detection_per_model.get(&m) {
            return *v;
        }
        // Also honour an entry written with the server's original casing.
        if let Some(v) = self.repetition_detection_per_model.get(self.model_name.trim()) {
            return *v;
        }
        !m.contains("glm")
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            config_version: CONFIG_VERSION,
            model_base_url: "https://llm.systemboan.net".into(),
            model_name: "glm-5.3-flash".into(),
            // DIVERGENCE (security) from config.py:63: no baked-in credential.
            // A compiled-in bearer token ships in source AND in every binary,
            // and rotating it means recompiling every deployment. Set the key
            // in ~/.codemason/config.json (existing installs already have it;
            // this default only affects fresh checkouts).
            model_api_key: String::new(),
            model_aes_key: String::new(),
            backend: "vllm".into(),
            model_dir: String::new(),
            max_tokens: 65536,
            // The auto profile (recommended_sampling) is what actually runs;
            // these are the starting point someone switching to "manual"
            // inherits. GLM's vendor numbers -- temperature 1.0, top_p 0.95,
            // no top_k -- are the RECOMMENDED values and live in
            // recommended_sampling, not here. This field is the DEFAULT: what
            // a fresh config.json is stamped with, and what this deployment is
            // actually run on. A default nobody runs is a trap, because any
            // path that rewrites the file drops the session onto it.
            temperature: 0.6,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            sampling_profile: "auto".into(),
            degenerate_run_chars: 2000,
            first_token_timeout_secs: 600.0,
            token_gap_timeout_secs: 120.0,
            repetition_min_pattern: 8,
            repetition_max_pattern: 512,
            repetition_min_count: 4,
            repetition_detection_per_model: std::collections::HashMap::new(),
            auto_compact_threshold_tokens: 0,
            max_concurrent_subagents: 3,
            subagent_context_ceiling_tokens: 0,
            subagent_max_tokens: 0,
            reasoning_effort: "high".into(),
            reasoning_effort_per_model: std::collections::HashMap::new(),
            clear_thinking: true,
            remote_notify_after: 0.0,
            remote_retention_days: 30.0,
            remote_ask_timeout: 1800.0,
            remote_replay_backlog: false,
            input_highlight: true,
            input_highlight_bg: String::new(),
            input_highlight_fg: String::new(),
            remote_file_threshold: 3000,
            remote_collapse_over: 1200,
            remote_file_ext: ".md".into(),
            remote_excerpt: 3000,
            import_claude_dirs: false,
            remote_enabled: true,
            remote_table_width: 36,
            theme: "dark".into(),
            verbose: false,
            realtime: true,
            dangerously_skip_permissions: false,
            auto_compact: true,
            spinner_random_verb: false,
            model_knowledge_cutoff: String::new(),
            grep_legacy_encoding: "euc-kr".to_string(),
            background_task_timeout: 21600.0,
            max_turns: None,
            keep_going: false,
            goal_gate_autorun: true,
            language: None,
            ui_language: None,
            searxng_url: "https://searxng.systemboan.net".into(),
            searxng_timeout: 8.0,
            searxng_lang: "auto".into(),
            // DIVERGENCE (security) from config.py:365 — same reason as
            // model_api_key above.
            searxng_api_key: String::new(),
            websearch_backends: "duckduckgo, brave, mojeek, google".into(),
            websearch_max_expansions: 1,
            websearch_throttle_min: 0.4,
            websearch_throttle_max: 1.2,
            websearch_retry_attempts: 3,
            websearch_backoff_base: 0.8,
            websearch_backoff_cap: 8.0,
            websearch_cache_ttl: 6 * 60 * 60,
            websearch_cache_dir: String::new(),
        }
    }
}

impl Config {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Migrate legacy config.json keys, then deserialize with unknown keys
    /// ignored and absent keys defaulted (Config.from_dict), reading each value
    /// as its field's type. Also returns every value that could not be read.
    pub fn from_value_checked(data: &Value) -> (Config, Vec<BadValue>) {
        let mut bad: Vec<BadValue> = Vec::new();
        let mut migrated = match data {
            Value::Object(m) => m.clone(),
            _ => serde_json::Map::new(),
        };
        for (old, new) in LEGACY_CONFIG_KEYS {
            if migrated.contains_key(*old) && !migrated.contains_key(*new) {
                if let Some(v) = migrated.remove(*old) {
                    migrated.insert((*new).to_string(), v);
                }
            }
        }
        // Field-tolerant load: a value of the wrong JSON type for one field must
        // not discard the rest (Python filtered by name and let the constructor
        // take what it was given). Deserialize field-by-field onto defaults.
        let mut cfg = Config::default();
        let base = serde_json::to_value(&cfg).unwrap();
        if let Value::Object(base_map) = base {
            let mut merged = base_map;
            for (k, v) in migrated {
                let Some(default) = merged.get(&k) else {
                    continue;
                };
                match coerce_field(&k, &v, default) {
                    Some(value) => {
                        merged.insert(k, value);
                    }
                    None => bad.push((k.clone(), v, default.clone())),
                }
            }
            if let Ok(parsed) = serde_json::from_value::<Config>(Value::Object(merged.clone())) {
                cfg = parsed;
            } else {
                // Per-field fallback: apply each key individually, skipping the
                // ones that fail to parse.
                for (k, v) in merged {
                    let mut single = serde_json::to_value(&cfg).unwrap();
                    if let Value::Object(ref mut m) = single {
                        m.insert(k, v);
                    }
                    if let Ok(parsed) = serde_json::from_value::<Config>(single) {
                        cfg = parsed;
                    }
                }
            }
        }
        (cfg, bad)
    }
}

struct ConfigStore {
    config: Option<Config>,
    /// Fields the environment overrode, holding the value the file had before
    /// it — what save_config() writes back so env stays runtime-only.
    env_shadow: HashMap<String, Value>,
    /// The config as it was ON DISK when this process loaded it. Base of the
    /// 3-way merge in write_config_file — see the comment there.
    file_baseline: Map<String, Value>,
    /// config.json exists but could not be parsed. This session runs on
    /// defaults and must not write: the next save would replace the user's
    /// file — API key, base URL, language — with those defaults. Cleared only
    /// by reset_config().
    load_failed: bool,
    load_failed_reported: bool,
}

static STORE: Lazy<Mutex<ConfigStore>> = Lazy::new(|| {
    Mutex::new(ConfigStore {
        config: None,
        env_shadow: HashMap::new(),
        file_baseline: Map::new(),
        load_failed: false,
        load_failed_reported: false,
    })
});

/// Ensure config directory exists.
pub fn ensure_config_dir() {
    let _ = std::fs::create_dir_all(&*CONFIG_DIR);
}

fn parse_bool_env(v: &str) -> bool {
    matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// Get the current configuration (singleton; loads + env-overlays on first use).
pub fn get_config() -> Config {
    let mut store = STORE.lock();
    if let Some(cfg) = &store.config {
        return cfg.clone();
    }

    ensure_config_dir();

    let mut needs_write = false;
    let mut load_failed = false;
    let mut bad_values: Vec<BadValue> = Vec::new();
    let mut migrated_notes: Vec<(String, Value, Value)> = Vec::new();
    // Printed after the store lock is released: tr() resolves the UI language
    // through get_config(), which would deadlock on this lock.
    let mut say: Vec<(&'static str, Vec<(&'static str, String)>)> = Vec::new();
    let mut config = if CONFIG_FILE.exists() {
        match std::fs::read_to_string(&*CONFIG_FILE)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string()))
        {
            Ok(data) => {
                let (mut cfg, bad) = Config::from_value_checked(&data);
                bad_values = bad;
                migrated_notes = apply_migrations(&mut cfg, &data);
                if !migrated_notes.is_empty()
                    || data.get("config_version") != Some(&Value::from(cfg.config_version))
                {
                    needs_write = true;
                }
                // Backfill: if the on-disk file predates any config field,
                // write the full set back so missing keys are materialized with
                // their defaults. Idempotent.
                let known: Vec<String> = match serde_json::to_value(&cfg).unwrap() {
                    Value::Object(m) => m.keys().cloned().collect(),
                    _ => vec![],
                };
                match &data {
                    Value::Object(m) => {
                        if known.iter().any(|k| !m.contains_key(k)) {
                            needs_write = true;
                        }
                    }
                    _ => needs_write = true,
                }
                cfg
            }
            Err(e) => {
                // Don't silently reset a corrupt config. It used to go only to
                // error.log, so a stray comma meant running on defaults — no
                // API key, someone else's server — with nothing on screen, and
                // the next /dsp or /effort then saved those defaults over the
                // user's file.
                crate::utils::log::log_error(&e, Some("Failed to load config; using defaults"));
                load_failed = true;
                say.push((
                    "[codemason] Could not read {path}: {error}",
                    vec![("path", CONFIG_FILE.to_string_lossy().into_owned()), ("error", e.clone())],
                ));
                say.push((
                    "[codemason] Running on defaults. Settings changed in this session will not be saved until the file is fixed (it is left untouched).",
                    vec![],
                ));
                Config::default()
            }
        }
    } else {
        // First run: materialize config.json so the user has a real file to
        // edit instead of silently running on in-memory defaults.
        needs_write = true;
        Config::default()
    };

    if needs_write {
        // Persist BEFORE env overrides so env stays runtime-only (never baked
        // into the file).
        // Empty baseline: this is the first materialization (or a backfill of
        // keys a newer version added), so every key counts as ours and the full
        // set is written. Unknown keys already on disk are still preserved.
        if let Err(e) = write_config_file(&config, &HashMap::new(), &Map::new()) {
            crate::utils::log::log_error(&e, Some("Failed to write config"));
        }
    }

    // What the file said, captured before the env overlay goes on. save_config()
    // writes these back over any field env changed, so a variable set for one
    // run stays runtime-only instead of being baked in by the next save.
    let pre_env = config.to_value();

    // Override with environment variables
    config.model_base_url = env_first(
        &config.model_base_url,
        &["CODEMASON_MODEL_BASE_URL", "LLAMACPP_BASE_URL"],
    );
    warn_plaintext_endpoint(&config.model_base_url, "model_base_url");
    config.model_name = env_first(&config.model_name, &["CODEMASON_MODEL_NAME", "LLAMACPP_MODEL"]);
    config.model_api_key = env_first(
        &config.model_api_key,
        &["CODEMASON_MODEL_API_KEY", "LLAMACPP_API_KEY"],
    );
    config.model_aes_key = env_first(
        &config.model_aes_key,
        &["CODEMASON_MODEL_AES_KEY", "LLAMACPP_AES_KEY"],
    );
    if let Ok(v) = std::env::var("CODEMASON_IMPORT_CLAUDE_DIRS") {
        let v = v.trim().to_lowercase();
        if !v.is_empty() {
            config.import_claude_dirs = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_BACKEND") {
        config.backend = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_MODEL_DIR") {
        config.model_dir = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_MAX_TURNS") {
        if !v.is_empty() {
            if let Ok(n) = v.parse::<i64>() {
                config.max_turns = Some(n);
            }
        }
    }
    if let Ok(v) = std::env::var("MAX_TOKENS") {
        if !v.is_empty() {
            if let Ok(n) = v.parse::<i64>() {
                config.max_tokens = n;
            }
        }
    }
    if let Ok(v) = std::env::var("TEMPERATURE") {
        if !v.is_empty() {
            if let Ok(n) = v.parse::<f64>() {
                config.temperature = n;
            }
        }
    }
    if let Ok(v) = std::env::var("REASONING_EFFORT") {
        let re = v.trim().to_lowercase();
        // '', 'off', 'none', 'false' all disable; otherwise must be a valid level.
        config.reasoning_effort = if matches!(re.as_str(), "low" | "medium" | "high" | "max") {
            re
        } else {
            String::new()
        };
    }
    if let Ok(v) = std::env::var("CODEMASON_REPLAY_BACKLOG") {
        config.remote_replay_backlog = parse_bool_env(&v);
    }
    if let Ok(v) = std::env::var("CODEMASON_ASK_TIMEOUT") {
        if let Ok(n) = v.parse::<f64>() {
            config.remote_ask_timeout = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_NOTIFY_AFTER") {
        if let Ok(n) = v.parse::<f64>() {
            config.remote_notify_after = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_GREP_LEGACY_ENCODING") {
        config.grep_legacy_encoding = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_MODEL_KNOWLEDGE_CUTOFF") {
        config.model_knowledge_cutoff = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_BACKGROUND_TASK_TIMEOUT") {
        if let Ok(n) = v.parse::<f64>() {
            config.background_task_timeout = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_REMOTE_RETENTION_DAYS") {
        if let Ok(n) = v.parse::<f64>() {
            config.remote_retention_days = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_TABLE_WIDTH") {
        if let Ok(n) = v.parse::<i64>() {
            config.remote_table_width = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_SEARXNG_URL") {
        config.searxng_url = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_SEARXNG_LANG") {
        config.searxng_lang = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_SEARXNG_API_KEY") {
        config.searxng_api_key = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_SEARXNG_TIMEOUT") {
        if let Ok(n) = v.parse::<f64>() {
            config.searxng_timeout = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_BACKENDS") {
        config.websearch_backends = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_CACHE_DIR") {
        config.websearch_cache_dir = v;
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_MAX_EXPANSIONS") {
        if let Ok(n) = v.parse::<i64>() {
            config.websearch_max_expansions = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_THROTTLE_MIN") {
        if let Ok(n) = v.parse::<f64>() {
            config.websearch_throttle_min = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_THROTTLE_MAX") {
        if let Ok(n) = v.parse::<f64>() {
            config.websearch_throttle_max = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_RETRY_ATTEMPTS") {
        if let Ok(n) = v.parse::<i64>() {
            config.websearch_retry_attempts = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_BACKOFF_BASE") {
        if let Ok(n) = v.parse::<f64>() {
            config.websearch_backoff_base = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_BACKOFF_CAP") {
        if let Ok(n) = v.parse::<f64>() {
            config.websearch_backoff_cap = n;
        }
    }
    if let Ok(v) = std::env::var("CODEMASON_WEBSEARCH_CACHE_TTL") {
        if let Ok(n) = v.parse::<i64>() {
            config.websearch_cache_ttl = n;
        }
    }

    // Record which fields env changed and what the file said for them.
    let post_env = config.to_value();
    let mut env_shadow = HashMap::new();
    if let (Value::Object(pre), Value::Object(post)) = (&pre_env, &post_env) {
        for (k, pre_v) in pre {
            if post.get(k) != Some(pre_v) {
                env_shadow.insert(k.clone(), pre_v.clone());
            }
        }
    }

    store.config = Some(config.clone());
    store.env_shadow = env_shadow;
    store.load_failed = load_failed;
    if let Value::Object(m) = &pre_env {
        store.file_baseline = m.clone();
    }
    drop(store);

    for (key, value, default) in &bad_values {
        say.push((
            "[codemason] config.json: \"{key}\" = {value} is not a valid value for this setting; using the default ({default}).",
            vec![("key", key.clone()), ("value", value.to_string()), ("default", default.to_string())],
        ));
    }
    for (key, old, new) in &migrated_notes {
        say.push((
            "[codemason] config.json updated: \"{key}\" {old} -> {new} (an old default that has since been fixed).",
            vec![("key", key.clone()), ("old", old.to_string()), ("new", new.to_string())],
        ));
    }
    for (text, args) in say {
        let args: Vec<(&str, crate::utils::i18n::FmtArg)> =
            args.into_iter().map(|(k, v)| (k, crate::utils::i18n::FmtArg::Str(v))).collect();
        eprintln!("{}", crate::utils::i18n::trf(text, &args));
    }
    config
}

/// Overlay only the keys this process changed onto the current file contents.
///
/// `baseline` is what the file said when we loaded it. A key whose value still
/// equals the baseline was not touched here, so the on-disk value (possibly
/// newer, from another session) wins. A key that differs is our edit and is
/// written. Unknown keys already on disk are kept.
///
/// Returns None when the file cannot be read or parsed, in which case the
/// caller writes its own view — the old behaviour, and no worse.
fn merge_over_disk(
    mine: &Map<String, Value>,
    baseline: &Map<String, Value>,
) -> Option<Map<String, Value>> {
    let text = std::fs::read_to_string(&*CONFIG_FILE).ok()?;
    let Value::Object(on_disk) = serde_json::from_str::<Value>(&text).ok()? else {
        return None;
    };
    Some(merge_maps(mine, baseline, on_disk))
}

/// The merge itself, split from the file read so it is testable: CONFIG_FILE is
/// resolved once at static init, so a test cannot point the reader elsewhere,
/// and a test that re-implemented this would only assert my understanding of it.
fn merge_maps(
    mine: &Map<String, Value>,
    baseline: &Map<String, Value>,
    on_disk: Map<String, Value>,
) -> Map<String, Value> {
    let mut merged = on_disk;
    for (key, value) in mine {
        let ours = baseline.get(key) != Some(value);
        if ours || !merged.contains_key(key) {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

fn write_config_file(
    config: &Config,
    env_shadow: &HashMap<String, Value>,
    file_baseline: &Map<String, Value>,
) -> Result<(), String> {
    // Put back what the file said for anything the environment overrode, so an
    // env var set for one run never becomes a stored setting.
    let mut data = config.to_value();
    if let Value::Object(ref mut m) = data {
        for (k, v) in env_shadow {
            m.insert(k.clone(), v.clone());
        }
    }
    // 3-way merge against the file as it is RIGHT NOW, with the copy we loaded
    // at startup as the base. Writing our whole view is a lost update: this
    // process's config was read at startup, so another session that changed a
    // DIFFERENT setting meanwhile had its change silently reverted to whatever
    // we happened to load. With several sessions open at once (normal here)
    // settings quietly rolled back for no visible reason.
    //
    // Only the keys THIS process actually changed are ours to write; every other
    // key takes the on-disk value, and keys we have never heard of (a newer
    // version's) are preserved rather than dropped.
    //
    // The merge reads the file and the write below replaces it, and those two
    // steps straddle another process's identical pair unless something makes
    // them one: measured 150/150 edits lost with two processes saving one
    // setting each at the same moment. This lock is that something, and it is
    // shared with the Python build (same sidecar path). Not getting it
    // (timeout) degrades to exactly the old behaviour — see utils::file.
    // Held from here to the rename and no further.
    let _lock = crate::utils::file::FileLock::acquire(&config_lock_path());
    if let Value::Object(mine) = &data {
        if let Some(merged) = merge_over_disk(mine, file_baseline) {
            data = Value::Object(merged);
        }
    }
    let text = serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?;
    let existed = CONFIG_FILE.exists();
    // Atomic: temp file in the same directory + rename. A plain write truncates
    // first, so a crash or a Ctrl+C between the truncate and the write left an
    // unparseable config — and the next start then silently fell back to
    // defaults, resetting model, base URL, language and permission mode with no
    // message. atomic_write_bytes already existed here and is what the file
    // tools use; this bypassed it. A newly created config also lands 0600
    // instead of 0644, which matters because this file holds the API key and
    // the body AES key.
    crate::utils::file::atomic_write_bytes(
        &CONFIG_FILE.to_string_lossy(),
        text.as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    // atomic_write_bytes keeps an existing file's mode but creates a new one
    // with the process umask (0644/0664) — readable by other users, in the
    // file that holds the API key and the body AES key. The Python build gets
    // 0600 from mkstemp; match it for a file this write created.
    #[cfg(unix)]
    if !existed {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&*CONFIG_FILE, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Save configuration to file. `config: Some(..)` replaces the singleton first.
/// Returns whether the file was written — false when config.json could not be
/// read at startup (see load_failed) or the write failed — so a command can say
/// "saved" only when it was.
pub fn save_config(config: Option<Config>) -> bool {
    let mut store = STORE.lock();
    if let Some(cfg) = config {
        store.config = Some(cfg);
    }
    let Some(cfg) = store.config.clone() else {
        return false;
    };
    if store.load_failed {
        // See load_failed: writing now would replace the user's unreadable
        // file with defaults. Say so once instead of failing silently.
        let first = !store.load_failed_reported;
        store.load_failed_reported = true;
        drop(store);
        if first {
            eprintln!(
                "{}",
                crate::utils::i18n::trf(
                    "[codemason] Not saved: {path} could not be read when this session started. Fix the file and restart; your change applies to this session only.",
                    &[("path", crate::utils::i18n::FmtArg::Str(CONFIG_FILE.to_string_lossy().into_owned()))],
                )
            );
        }
        return false;
    }
    ensure_config_dir();
    let written = match write_config_file(&cfg, &store.env_shadow, &store.file_baseline) {
        Ok(()) => true,
        Err(e) => {
            crate::utils::log::log_error(&e, Some("Failed to save config"));
            false
        }
    };
    drop(store);
    // The UI language is resolved once and cached; a saved config that changes
    // it would otherwise not take effect until the next run.
    crate::utils::i18n::reset();
    written
}

/// Mutate the live singleton in place (the Python idiom `get_config().field = x`).
/// Callers that change a real preference follow with `save_config(None)`.
pub fn update_config<F: FnOnce(&mut Config)>(f: F) {
    // Materialize the singleton first (outside the lock; get_config locks).
    let _ = get_config();
    let mut store = STORE.lock();
    if let Some(cfg) = store.config.as_mut() {
        f(cfg);
    }
}

/// Serializes every test that reads or writes `grep_legacy_encoding`.
///
/// The setting lives in the process-wide singleton and cargo runs tests as
/// threads in one process, so a_mixed_tree_is_searched_in_both_encodings
/// turning it off ("" = second pass off) is visible to any test reading it at
/// that moment. utf8_and_cp949_in_one_stream_both_survive read it inside that
/// window under load and decoded the CP949 half to U+FFFD; the same window
/// would cut an_ascii_term_is_not_counted_twice_per_file down to one pass.
/// Pinning also takes the developer's own config.json out of the result: ""
/// is a documented setting, and with it in the file the decode test failed
/// every run on that machine.
///
/// Same shape as SAMPLING_ENV: hold the guard for the whole test body; the
/// previous value is put back on drop, so a panicking test cannot leak "" into
/// the next one; poison is stepped over.
#[cfg(test)]
static LEGACY_ENCODING: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) struct LegacyEncodingGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: String,
}

#[cfg(test)]
impl Drop for LegacyEncodingGuard {
    fn drop(&mut self) {
        let prev = std::mem::take(&mut self.prev);
        update_config(|c| c.grep_legacy_encoding = prev);
    }
}

#[cfg(test)]
pub(crate) fn pin_legacy_encoding(label: &str) -> LegacyEncodingGuard {
    let lock = LEGACY_ENCODING.lock().unwrap_or_else(|e| e.into_inner());
    let prev = get_config().grep_legacy_encoding;
    update_config(|c| c.grep_legacy_encoding = label.to_string());
    LegacyEncodingGuard { _lock: lock, prev }
}

/// Apply a setting for this run only, without letting it reach the file
/// (set_runtime_override). `value` is the JSON value for the field.
pub fn set_runtime_override(field: &str, value: Value) {
    let cfg = get_config();
    let cur = cfg.to_value();
    let cur_map = match cur {
        Value::Object(m) => m,
        _ => return,
    };
    if !cur_map.contains_key(field) {
        return;
    }
    let mut store = STORE.lock();
    let prev = cur_map.get(field).cloned().unwrap_or(Value::Null);
    store.env_shadow.entry(field.to_string()).or_insert(prev);
    // Apply the new value onto the singleton via JSON round-trip.
    if let Some(live) = store.config.clone() {
        let mut v = live.to_value();
        if let Value::Object(ref mut m) = v {
            m.insert(field.to_string(), value);
        }
        if let Ok(parsed) = serde_json::from_value::<Config>(v) {
            store.config = Some(parsed);
        }
    }
}

/// Let a field be persisted again, for a value the user deliberately set.
pub fn clear_runtime_override(field: &str) {
    STORE.lock().env_shadow.remove(field);
}

/// The word a command puts after a setting it just changed: "saved", or why
/// not. Pairs with save_config()'s return value.
pub fn saved_label(saved: bool) -> String {
    if saved {
        crate::utils::i18n::tr("saved")
    } else {
        crate::utils::i18n::tr("not saved: config.json could not be read")
    }
}

/// Reset configuration to defaults.
pub fn reset_config() {
    {
        let mut store = STORE.lock();
        store.env_shadow.clear();
        store.config = Some(Config::default());
        // An explicit reset is the one write that may replace an unreadable file.
        store.load_failed = false;
    }
    let _ = save_config(None);
}

/// Warn once when a bearer token would cross a network in the clear.
///
/// The Authorization header is attached OUTSIDE the body-encryption branch, so
/// an http:// endpoint sends the API key — and, without body encryption, the
/// whole conversation, file contents and generated code — as plaintext. No
/// scheme check existed anywhere for this value.
///
/// Loopback is exempt and silent: a local vLLM on 127.0.0.1 is a normal setup
/// and nothing leaves the machine. This warns rather than refuses, because a
/// gateway reachable only over a trusted link is a deliberate deployment and
/// breaking a working one is not this function's call.
fn warn_plaintext_endpoint(url: &str, label: &str) {
    static WARNED: Lazy<Mutex<HashSet<String>>> = Lazy::new(|| Mutex::new(HashSet::new()));
    let Ok(parsed) = url::Url::parse(url.trim()) else {
        return;
    };
    if parsed.scheme() != "http" {
        return;
    }
    let host = parsed.host_str().unwrap_or("").to_lowercase();
    if host == "localhost" || host == "::1" || host.starts_with("127.") {
        return;
    }
    if !WARNED.lock().insert(label.to_string()) {
        return;
    }
    let msg = format!(
        "  Warning: {label} is http:// ({host}). The API key is sent as a header, \
         which body encryption does not cover, so it crosses the network in the \
         clear. Use https:// unless the link is trusted."
    );
    // A diagnostic, not output: stdout is the answer channel, and in
    // `-p --output-format json` a line here makes the JSON unparseable. Above a
    // live REPL prompt it still goes through the router so the prompt repaints.
    if crate::utils::term_out::sink_active() {
        crate::cmprintln!("{msg}");
    } else {
        eprintln!("{msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Another product's directories are not read unless asked for.
    ///
    /// Until this became a setting, ~/.claude/skills and ~/.claude/agents were
    /// always searched — so a skill or sub-agent the user put there for a
    /// different tool appeared in this one's list without them choosing that,
    /// and sub-agents had no native directory at all. Interop is a convenience
    /// for someone migrating; the default has to be off, or it is not interop,
    /// it is inheritance.
    /// A `"max_turns": null` on disk is the old default written back, not a
    /// user choice, and honouring it would leave the guard off for every
    /// existing install — the fix would change nothing for anyone.
    /// Two sessions changing DIFFERENT settings must both survive. Writing our
    /// whole in-memory view reverts the other one's change to whatever we
    /// happened to load at startup, which is how settings silently rolled back.
    #[test]
    fn a_concurrent_edit_to_another_key_is_not_clobbered() {
        // What the file said when we started.
        let mut baseline = Map::new();
        baseline.insert("theme".into(), Value::from("dark"));
        baseline.insert("max_turns".into(), Value::from(200));

        // Another session has since changed max_turns; we changed theme.
        let mut on_disk = baseline.clone();
        on_disk.insert("max_turns".into(), Value::from(77));
        on_disk.insert("some_future_key".into(), Value::from("keep me"));

        let mut mine = baseline.clone();
        mine.insert("theme".into(), Value::from("light"));

        let merged = merge_maps(&mine, &baseline, on_disk);

        assert_eq!(merged.get("theme"), Some(&Value::from("light")), "our edit must land");
        assert_eq!(
            merged.get("max_turns"),
            Some(&Value::from(77)),
            "the other session's edit must survive — this is the lost update"
        );
        assert_eq!(
            merged.get("some_future_key"),
            Some(&Value::from("keep me")),
            "a key we do not know must not be dropped"
        );
    }

    /// The turn cap must stay OFF unless someone asks for it. It does not
    /// prompt — it ends the turn — so in the one mode where it would fire (a
    /// long unattended -dsp run) it silently truncates the work the user walked
    /// away from. Defaulting it on was my mistake; this pins it.
    /// Anything keyed by the endpoint is shared with the Python build, so the
    /// tag has to be the same STRING there — a different hash or a different
    /// normalization splits the namespace silently. The literal values are
    /// pinned on both sides (tests/test_parity_with_rust.py).
    #[test]
    fn the_endpoint_tag_matches_the_python_build() {
        assert_eq!(endpoint_tag(Some("https://llm.systemboan.net")), "4214d21b0e48");
        assert_eq!(endpoint_tag(Some("http://127.0.0.1:8000/v1")), "87459d11adee");
        // Normalized first: the same server written two ways is one pool.
        assert_eq!(
            endpoint_tag(Some("  https://llm.systemboan.net/  ")),
            endpoint_tag(Some("https://llm.systemboan.net"))
        );
        // Scheme and host are case-insensitive, so folding them keeps one server
        // one pool. Before this, a mixed-case spelling handed out a slot too
        // many — two sessions rationed against one GPU as if it were two.
        assert_eq!(endpoint_tag(Some("https://LLM.SystemBoan.NET")), "4214d21b0e48");
        assert_eq!(
            endpoint_tag(Some("http://LocalHost:8000")),
            endpoint_tag(Some("http://localhost:8000"))
        );
        // The path is NOT folded: it is case-sensitive, and two endpoints
        // differing only there do not share a cache.
        assert_ne!(
            endpoint_tag(Some("http://h:8000/v1")),
            endpoint_tag(Some("http://h:8000/V1"))
        );
        assert_eq!(endpoint_tag(Some("")), "none");
    }

    /// The lock path is a cross-build CONTRACT, not an implementation detail: a
    /// name that differs by one byte from the Python client's leaves the two
    /// builds locking different files and excluding nobody — and nothing else
    /// in either test suite would notice. Both sides pin the literal path.
    #[test]
    fn the_config_lock_path_is_the_one_the_python_build_uses() {
        assert_eq!(lock_path("config.json"), CONFIG_DIR.join("locks").join("config.json.lock"));
        // The path the WRITER takes, not just one spelled the same way here.
        assert_eq!(config_lock_path(), CONFIG_DIR.join("locks").join("config.json.lock"));
        assert_eq!(*LOCKS_DIR, CONFIG_DIR.join("locks"));
    }

    #[test]
    fn the_turn_cap_is_off_unless_asked_for() {
        assert_eq!(Config::default().max_turns, None, "a default cap truncates unattended runs");
    }


    #[test]
    fn foreign_config_dirs_are_off_by_default() {
        assert!(!Config::default().import_claude_dirs);
    }

    /// auto is the default and means "the client decides": the file is not
    /// consulted, so a config.json that filled itself in cannot leak values
    /// nobody chose into a request.
    /// Serializes every test that reads the sampling environment.
    ///
    /// `std::env::set_var` is process-wide and cargo runs these tests as threads
    /// in one process, so `the_environment_has_the_last_word` setting TEMPERATURE
    /// is visible to any test reading it at that moment: the auto table's 1.0 came
    /// back as 0.6 in 3 of 16 tests, every run, depending on nothing the test
    /// itself did. Taking this lock makes the window exclusive; the guard clears
    /// the vars on drop so a panicking test cannot leak them into the next one.
    static SAMPLING_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct SamplingEnv(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl Drop for SamplingEnv {
        fn drop(&mut self) {
            for k in ["TEMPERATURE", "TOP_K", "CODEMASON_TEMPERATURE", "CODEMASON_TOP_K"] {
                std::env::remove_var(k);
            }
        }
    }

    /// Hold for the whole test body. Poison is stepped over: one failing test
    /// must not turn every later one into a lock error.
    fn sampling_env() -> SamplingEnv {
        let g = SAMPLING_ENV.lock().unwrap_or_else(|e| e.into_inner());
        for k in ["TEMPERATURE", "TOP_K", "CODEMASON_TEMPERATURE", "CODEMASON_TOP_K"] {
            std::env::remove_var(k);
        }
        SamplingEnv(g)
    }

    #[test]
    fn auto_ignores_the_file_and_uses_the_family_table() {
        let _env = sampling_env();
        for (name, temp, top_k) in [
            ("glm-5.3-flash", 1.0, None),
            ("GLM-5.3", 1.0, None),
            ("DSV4-Flash", 0.6, Some(40)),
            ("deepseek-v4", 0.6, Some(40)),
            ("", 0.6, Some(40)),
        ] {
            let mut c = Config::default();
            c.model_name = name.into();
            c.temperature = 0.123;
            c.top_k = 7;
            c.top_p = 0.111;
            c.frequency_penalty = 0.9;
            let s = c.effective_sampling();
            assert_eq!(s.temperature, temp, "{name}");
            assert_eq!(s.top_k, top_k, "{name}");
            assert_eq!(s.top_p, 0.95, "{name}");
            assert_eq!(s.min_p, Some(0.0), "{name}: min_p is pinned at 0 under MTP");
            assert_eq!(s.frequency_penalty, None, "{name}: auto sends no penalties");
        }
    }

    /// An unrecognised profile is not a third mode: it falls back to auto, which
    /// is what someone who has not read this file wants.
    #[test]
    fn an_unknown_profile_falls_back_to_auto() {
        let _env = sampling_env();
        for profile in ["", "auto", "AUTO", "vendor", "typo"] {
            let mut c = Config::default();
            c.model_name = "glm-5.3-flash".into();
            c.sampling_profile = profile.into();
            c.temperature = 0.123;
            assert_eq!(c.effective_sampling().temperature, 1.0, "profile {profile:?}");
        }
    }

    /// manual hands the wheel back: the file is sent verbatim, every knob.
    #[test]
    fn manual_sends_the_file_verbatim() {
        let _env = sampling_env();
        let mut c = Config::default();
        c.model_name = "glm-5.3-flash".into();
        c.sampling_profile = "manual".into();
        c.temperature = 0.42;
        c.top_k = 7;
        c.frequency_penalty = 0.3;
        let s = c.effective_sampling();
        assert_eq!(s.temperature, 0.42);
        assert_eq!(s.top_k, Some(7));
        assert_eq!(s.frequency_penalty, Some(0.3));
    }

    /// The environment outranks both profiles — it is the per-run instruction.
    #[test]
    fn the_environment_has_the_last_word() {
        let _env = sampling_env();
        let mut c = Config::default();
        c.model_name = "glm-5.3-flash".into();
        std::env::set_var("TEMPERATURE", "0.6");
        std::env::set_var("TOP_K", "40");
        let auto = c.effective_sampling();
        c.sampling_profile = "manual".into();
        c.temperature = 0.42;
        let manual = c.effective_sampling();
        std::env::remove_var("TEMPERATURE");
        std::env::remove_var("TOP_K");
        assert_eq!(auto.temperature, 0.6, "env beats the auto table");
        assert_eq!(auto.top_k, Some(40), "env can add a knob auto omits");
        assert_eq!(manual.temperature, 0.6, "env beats the file too");
    }

    /// The guard is on by default and family-independent.
    #[test]
    fn the_degenerate_guard_is_on_by_default() {
        assert_eq!(Config::default().degenerate_run_chars, 2000);
    }

    /// Hand-typed spellings mean what they say; "false" never enables a flag.
    #[test]
    fn hand_typed_values_are_read_as_meant() {
        let data = serde_json::json!({
            "dangerously_skip_permissions": "false",
            "realtime": "no",
            "max_turns": "50",
            "max_tokens": "32000",
            "keep_going": "maybe",
            "model_base_url": null,
            "remote_ask_timeout": 900,
            "max_concurrent_subagents": -1,
        });
        let (c, bad) = Config::from_value_checked(&data);
        assert!(!c.dangerously_skip_permissions);
        assert!(!c.realtime);
        assert_eq!(c.max_turns, Some(50));
        assert_eq!(c.max_tokens, 32000);
        assert_eq!(c.remote_ask_timeout, 900.0, "an integer is a valid float");
        assert!(!c.keep_going, "unreadable -> default");
        assert_eq!(c.model_base_url, Config::default().model_base_url);
        assert_eq!(c.max_concurrent_subagents, 3, "a negative count is refused");
        let keys: Vec<&str> = bad.iter().map(|(k, _, _)| k.as_str()).collect();
        for k in ["keep_going", "model_base_url", "max_concurrent_subagents"] {
            assert!(keys.contains(&k), "{k} must be reported, got {keys:?}");
        }
        assert_eq!(bad.len(), 3, "only the unreadable ones are reported: {keys:?}");
    }

    /// Defaults that were defects move forward once; user choices stay.
    #[test]
    fn old_defaults_are_migrated_and_choices_kept() {
        let data = serde_json::json!({
            "max_tokens": 16384, "max_turns": 200, "reasoning_effort": "",
            "sampling_profile": "vendor", "realtime": false, "keep_going": true,
        });
        let (mut c, _) = Config::from_value_checked(&data);
        let notes = apply_migrations(&mut c, &data);
        assert_eq!(notes.len(), 4);
        assert_eq!(c.max_tokens, 65536);
        assert_eq!(c.max_turns, None);
        assert_eq!(c.reasoning_effort, "high");
        assert_eq!(c.sampling_profile, "auto");
        assert!(!c.realtime && c.keep_going, "preferences are not migrated");
        assert_eq!(c.config_version, CONFIG_VERSION);

        let chosen = serde_json::json!({"max_tokens": 32000, "reasoning_effort": "low"});
        let (mut c, _) = Config::from_value_checked(&chosen);
        assert!(apply_migrations(&mut c, &chosen).is_empty());
        assert_eq!((c.max_tokens, c.reasoning_effort.as_str()), (32000, "low"));

        let current = serde_json::json!({"config_version": CONFIG_VERSION, "max_tokens": 16384});
        let (mut c, _) = Config::from_value_checked(&current);
        assert!(apply_migrations(&mut c, &current).is_empty(), "runs once per file");
        assert_eq!(c.max_tokens, 16384);

        let newer = serde_json::json!({"config_version": 99, "max_tokens": 16384});
        let (mut c, _) = Config::from_value_checked(&newer);
        apply_migrations(&mut c, &newer);
        assert_eq!(c.config_version, 99, "never lowered");
    }

    /// The fresh-install default agrees with the Python build.
    #[test]
    fn the_sampling_profile_default_is_auto() {
        assert_eq!(Config::default().sampling_profile, "auto");
        assert_eq!(Config::default().config_version, CONFIG_VERSION);
    }
}
