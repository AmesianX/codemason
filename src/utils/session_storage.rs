//! Session transcript persistence — faithful subset of
//! utils/sessionStorage.ts + utils/sessionStoragePortable.ts + types/logs.ts
//! from Claude Code. Equivalent to codemason/utils/session_storage.py.
//!
//! Storage layout (identical structure to the original, rooted at codemason's
//! own config home instead of ~/.claude):
//!
//!     <CONFIG_DIR>/projects/<sanitized-cwd>/<sessionId>.jsonl
//!
//! One session = one JSONL file; one line per transcript entry. Entry shape is
//! the original's TranscriptMessage core:
//!
//!     { parentUuid, isSidechain, userType, cwd, sessionId, version, gitBranch?,
//!       type, message, uuid, timestamp, ...passthrough (isMeta, ...) }
//!
//! PORTED (the subset codemason needs now):
//!   - sanitize_path / projects dir / transcript path resolution
//!   - append-side: record_transcript (uuid-dedup + linear parentUuid chain),
//!     appendEntryToFile semantics (0600 file, 0700 dir, append-only)
//!   - read-side: load_transcript_file (tolerant JSONL parse, 50MB cap),
//!     leaf detection, buildConversationChain (cycle-guarded walk root←leaf),
//!     first-prompt extraction, visible-message counting
//!   - resume: load_conversation_for_resume + adopt_resumed_session
//!     (adoptResumedSessionFile: same id keeps appending to the same file;
//!     dedup seeding prevents re-writing loaded messages)
//!   - listing: list_sessions (LogOption subset, sortLogs order)
//!
//! NOT ported (subsystems codemason doesn't have — each is an isolated
//! extension point, the entry format already reserves their shapes):
//!   - sidechain/agent transcripts (record_transcript(is_sidechain) is
//!     plumbed; getAgentTranscriptPath and AgentTool wiring are not)
//!   - metadata entry types (summary / custom-title / tag / pr-link / ...):
//!     the parser skips unknown entry types instead of crashing
//!   - compact-boundary truncation on load (readTranscriptForLoad): a resumed
//!     post-compaction session restores the FULL pre-compact chain
//!   - parallel tool_result orphan recovery: codemason's write chain is
//!     strictly linear write-order, so those DAG shapes cannot occur
//!   - lite/tail reads (getSessionFilesLite): list_sessions does full reads
//!
//! JSON key-order divergence (Rust vs Python): Python json.dumps writes dict
//! insertion order; serde_json's Map is BTree-ordered, so entry keys land
//! alphabetically. JSON semantics are unchanged — every reader (Python and
//! Rust alike) accesses keys by name, never by position — so transcripts
//! written by either implementation are mutually readable.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::config::CONFIG_DIR;
use super::state::{get_cwd, get_original_cwd, get_session_id, set_session_id};

/// 50 MB — sessionStorage.ts MAX_TRANSCRIPT_READ_BYTES. Callers reading a raw
/// transcript bail above this to avoid OOM.
pub const MAX_TRANSCRIPT_READ_BYTES: u64 = 50 * 1024 * 1024;

/// sessionStoragePortable.ts MAX_SANITIZED_LENGTH.
pub const MAX_SANITIZED_LENGTH: usize = 200;

/// sessionStorage.ts:125 — non-meaningful first prompts (XML-ish control
/// payloads like <task-notification>/<local-command-stdout>, interrupts).
static SKIP_FIRST_PROMPT_PATTERN: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(r"^(?:\s*<[a-z][\w-]*[\s>]|\[Request interrupted by user[^\]]*\])").unwrap()
});

/// Entry types that participate in the message chain (isTranscriptMessage).
pub const TRANSCRIPT_TYPES: &[&str] = &["user", "assistant", "attachment", "system"];

fn djb2_hash(s: &str) -> i64 {
    let mut h: u32 = 5381;
    for ch in s.chars() {
        h = h.wrapping_mul(33) ^ (ch as u32);
    }
    // JS djb2Hash returns a signed 32-bit int; mirror that before abs().
    h as i32 as i64
}

fn to_base36(mut n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut out = Vec::new();
    while n > 0 {
        out.push(digits[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// sessionStoragePortable.ts sanitizePath: every non-alphanumeric char
/// becomes '-'; over-long results are truncated with a hash suffix.
pub fn sanitize_path(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    if sanitized.chars().count() <= MAX_SANITIZED_LENGTH {
        return sanitized;
    }
    let hash_suffix = to_base36(djb2_hash(name).unsigned_abs());
    let head: String = sanitized.chars().take(MAX_SANITIZED_LENGTH).collect();
    format!("{head}-{hash_suffix}")
}

pub fn get_projects_dir() -> PathBuf {
    CONFIG_DIR.join("projects")
}

pub fn get_project_dir(project_path: &str) -> PathBuf {
    get_projects_dir().join(sanitize_path(project_path))
}

/// Delete a session: its transcript <id>.jsonl and its own folder <id>/ (the
/// image store, utils/image_store.rs), which nothing else refers to. False when
/// the transcript could not be removed.
pub fn delete_session(transcript: &std::path::Path) -> bool {
    if std::fs::remove_file(transcript).is_err() {
        return false;
    }
    let side = transcript.with_extension("");
    if side.is_dir() && !side.is_symlink() {
        let _ = std::fs::remove_dir_all(&side);
    }
    true
}

#[cfg(test)]
mod delete_session_tests {
    #[test]
    fn takes_its_folder_and_nothing_else() {
        let d = std::env::temp_dir().join(format!("cm-sess-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join("abc").join("images")).unwrap();
        std::fs::create_dir_all(d.join("other")).unwrap();
        std::fs::write(d.join("abc.jsonl"), "{}\n").unwrap();
        std::fs::write(d.join("abc").join("images").join("x.png"), "1").unwrap();
        assert!(super::delete_session(&d.join("abc.jsonl")));
        assert!(!d.join("abc.jsonl").exists() && !d.join("abc").exists() && d.join("other").exists());
        assert!(!super::delete_session(&d.join("abc.jsonl")));
        let _ = std::fs::remove_dir_all(&d);
    }
}

// Settings that belong to a working directory rather than to the machine.
//
// config.json is one file per home, so it cannot answer a question two
// instances on the same machine answer differently — "which bot does this
// window use?" is exactly that shape. The project directory already exists
// (it holds the transcripts), and it is keyed by the cwd, which is the thing
// that actually distinguishes the two windows. Kept as its own file so it is
// never confused with a transcript: those are *.jsonl, this is settings.json.
const PROJECT_SETTINGS: &str = "settings.json";

fn project_settings_path(cwd: &str) -> PathBuf {
    let cwd = if cwd.is_empty() { get_original_cwd() } else { cwd.to_string() };
    get_project_dir(&cwd).join(PROJECT_SETTINGS)
}

/// The lock sidecar for one of this directory's shared documents.
///
/// Project state is shared by every session in the same directory — two `cm2`
/// windows in one repo, or a `cm2` and a `codemason` — so its documents need the
/// same exclusion as the config dir's, keyed by the thing that distinguishes
/// them: the cwd. Every lock in the system still lives in one directory
/// (config::LOCKS_DIR); the project is part of the NAME, so a repo's lock can
/// never be mistaken for another's. Same path as the Python build's
/// session_storage.project_lock.
pub fn project_lock(name: &str, cwd: &str) -> PathBuf {
    let cwd = if cwd.is_empty() { get_original_cwd() } else { cwd.to_string() };
    crate::utils::config::lock_path(&format!("{}.{name}", sanitize_path(&cwd)))
}

/// This directory's saved settings, or {} — never raises.
pub fn read_project_settings(cwd: &str) -> Map<String, Value> {
    (|| -> Option<Map<String, Value>> {
        let text = std::fs::read_to_string(project_settings_path(cwd)).ok()?;
        match serde_json::from_str::<Value>(&text).ok()? {
            Value::Object(m) => Some(m),
            _ => None,
        }
    })()
    .unwrap_or_default()
}

/// Merge one setting into this directory's file. True when it landed.
///
/// The read and the write are one step. A temp file + rename alone keeps a
/// reader from seeing half a document, which is what this used to do, but it does
/// not stop the other session from having read the SAME document first and then
/// writing its own copy over this one — the key it set is simply gone. Same
/// shape, same fix, as config.json.
pub fn write_project_setting(key: &str, value: Value, cwd: &str) -> bool {
    let path = project_settings_path(cwd);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let key = key.to_string();
    crate::utils::file::update_json_document(
        &path,
        &project_lock(PROJECT_SETTINGS, cwd),
        move |current| {
            let mut data = match current {
                Some(Value::Object(m)) => m,
                _ => Map::new(),
            };
            data.insert(key, value);
            Some(Value::Object(data))
        },
        false,
    )
}

/// Path of the CURRENT session's transcript.
pub fn get_transcript_path() -> PathBuf {
    get_transcript_path_for_session(&get_session_id())
}

pub fn get_transcript_path_for_session(session_id: &str) -> PathBuf {
    get_project_dir(&get_original_cwd()).join(format!("{session_id}.jsonl"))
}

fn clear_boundary_path() -> PathBuf {
    get_project_dir(&get_original_cwd()).join(".cleared")
}

/// Mark that the user just ran /clear in this project.
///
/// /clear switches to a fresh, empty session, but an empty session is never
/// written to disk and list_sessions() skips it — so a later `--continue`
/// would resume the PRE-clear session and the clear looks ignored. Recording
/// a boundary (a file whose mtime is the clear time) lets `--continue` tell
/// that the last real session predates a deliberate clear and start fresh
/// instead. Best-effort — a failure just falls back to the old behavior.
pub fn record_clear_boundary() {
    let p = clear_boundary_path();
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&p, ""); // fresh write updates mtime
}

/// True if a /clear happened AT OR AFTER the last write to session_id — i.e.
/// resuming it would undo a deliberate clear. Used by --continue.
pub fn cleared_after_session(session_id: &str) -> bool {
    let b = clear_boundary_path();
    let Ok(b_meta) = std::fs::metadata(&b) else {
        return false;
    };
    if !b_meta.is_file() {
        return false;
    }
    let s = get_transcript_path_for_session(session_id);
    let Ok(s_meta) = std::fs::metadata(&s) else {
        return true;
    };
    match (b_meta.modified(), s_meta.modified()) {
        (Ok(bm), Ok(sm)) => bm >= sm,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Write side
// ---------------------------------------------------------------------------

/// Per-process session-file state (the original's Project instance subset):
/// uuids already on disk (dedup) and the last chain participant's uuid (the
/// next entry's parentUuid). Reset on switch_session; seeded on resume-adopt.
struct WriteState {
    seen_uuids: HashSet<String>,
    last_chain_uuid: Option<String>,
}

static WRITE_STATE: Lazy<Mutex<WriteState>> = Lazy::new(|| {
    Mutex::new(WriteState { seen_uuids: HashSet::new(), last_chain_uuid: None })
});

static GIT_BRANCH: Lazy<Mutex<(bool, Option<String>)>> = Lazy::new(|| Mutex::new((false, None)));

/// Current git branch for the entry's gitBranch field (cached; None outside a
/// repo). Resolved once per process — matches the original, which snapshots
/// it at session scope. 2s timeout, best-effort.
fn get_git_branch() -> Option<String> {
    let mut cache = GIT_BRANCH.lock();
    if cache.0 {
        return cache.1.clone();
    }
    cache.0 = true;
    cache.1 = (|| -> Option<String> {
        let mut child = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(get_cwd())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null())
            .spawn()
            .ok()?;
        // Poll with a 2s deadline (std has no wait-with-timeout).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return None;
                    }
                    let mut out = String::new();
                    use std::io::Read;
                    child.stdout.take()?.read_to_string(&mut out).ok()?;
                    let branch = out.trim().to_string();
                    return if branch.is_empty() { None } else { Some(branch) };
                }
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(_) => return None,
            }
        }
    })();
    cache.1.clone()
}

fn get_version() -> String {
    // The Python lazily imported cli.VERSION to dodge a circular import; Rust
    // has no such cycle, but the value is the same constant.
    crate::cli::VERSION.to_string()
}

/// ISO-8601 UTC with milliseconds and 'Z' — the original's timestamp form
/// (new Date().toISOString()).
fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// appendEntryToFile: append one JSONL line; 0600 file, 0700 parent dir
/// created on demand.
pub fn append_entry_to_file(full_path: &Path, entry: &Value) -> std::io::Result<()> {
    let line = format!("{}\n", serde_json::to_string(entry)?);
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(full_path)?;
    f.write_all(line.as_bytes())
}

fn msg_field<'a>(m: &'a Value, key: &str) -> Option<&'a Value> {
    m.get(key)
}

/// Build the TranscriptMessage-core JSONL entry for one in-memory message.
/// Messages pass their extra top-level keys through (isMeta, isCompactSummary,
/// ...) so restore round-trips them.
pub fn serialize_entry(message: &Value, parent_uuid: Option<&str>, is_sidechain: bool) -> Value {
    let mut entry = Map::new();
    if let Value::Object(m) = message {
        for (k, v) in m {
            if !matches!(k.as_str(), "type" | "uuid" | "message") {
                entry.insert(k.clone(), v.clone());
            }
        }
    }
    entry.insert(
        "parentUuid".into(),
        parent_uuid.map(|s| Value::String(s.to_string())).unwrap_or(Value::Null),
    );
    entry.insert("isSidechain".into(), Value::Bool(is_sidechain));
    entry.insert("userType".into(), Value::String("external".into()));
    entry.insert("cwd".into(), Value::String(get_cwd()));
    entry.insert("sessionId".into(), Value::String(get_session_id()));
    entry.insert("version".into(), Value::String(get_version()));
    entry.insert(
        "type".into(),
        msg_field(message, "type").cloned().unwrap_or(Value::Null),
    );
    // Image payloads go to <session>/images/, the entry keeps a reference
    // (utils/image_store.rs); the in-memory message is not touched.
    entry.insert(
        "message".into(),
        crate::utils::image_store::dehydrate(
            msg_field(message, "message").unwrap_or(&Value::Null),
            None,
        ),
    );
    let uuid_str = msg_field(message, "uuid")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    entry.insert("uuid".into(), Value::String(uuid_str));
    entry.insert("timestamp".into(), Value::String(now_iso()));
    if let Some(branch) = get_git_branch() {
        entry.insert("gitBranch".into(), Value::String(branch));
    }
    Value::Object(entry)
}

/// recordTranscript subset: called with the FULL in-memory history after
/// every mutation (the useLogMessages pattern). Messages whose uuid was
/// already written are skipped, so incremental calls append only the new
/// tail; each new entry's parentUuid is the previously recorded entry
/// (linear write-order chain).
///
/// Persistence must never break the conversation — failures are logged and
/// swallowed.
pub fn record_transcript_values(messages: &[Value], is_sidechain: bool) {
    let path = get_transcript_path();
    let mut state = WRITE_STATE.lock();
    for message in messages {
        let mtype = message.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if !TRANSCRIPT_TYPES.contains(&mtype) {
            continue; // progress etc. — not persisted (post-#24099 shape)
        }
        let uuid_str = message
            .get("uuid")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if uuid_str.is_empty() || state.seen_uuids.contains(&uuid_str) {
            continue;
        }
        let entry = serialize_entry(message, state.last_chain_uuid.as_deref(), is_sidechain);
        if let Err(e) = append_entry_to_file(&path, &entry) {
            crate::utils::log::log_error(&e, None);
            // Say it ONCE on screen too. Persistence must never break the
            // conversation, so swallowing the error is right — but writing it
            // only to a log file meant a full disk or a read-only $HOME left
            // the session looking normal while nothing was recorded, and the
            // user found out at --resume time with the transcript short or
            // missing. Once per process: a failing disk fails on every message,
            // and a warning per message would bury the conversation itself.
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            let warn = !WARNED.swap(true, std::sync::atomic::Ordering::SeqCst);
            // Print AFTER the write lock is gone: a print runs term_out -> the
            // prompt engine -> `paint` -> stdout, so printing here would make
            // WRITE_STATE an edge into the painting order for the sake of one
            // warning line.
            drop(state);
            if warn {
                crate::cmprintln!(
                    "  Warning: this session is no longer being recorded ({e}). \
                     --resume will not have it."
                );
            }
            return;
        }
        state.seen_uuids.insert(uuid_str.clone());
        state.last_chain_uuid = Some(uuid_str);
    }
}

/// Typed-message convenience over record_transcript_values.
pub fn record_transcript(messages: &[crate::types::query::Message], is_sidechain: bool) {
    let values: Vec<Value> = messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
        .collect();
    record_transcript_values(&values, is_sidechain);
}

/// Switch to a new (or given) session id and reset the file-pointer state
/// (switchSession + resetSessionFilePointer). The new transcript file is
/// created lazily by the next record_transcript.
pub fn switch_session(session_id: Option<&str>) -> String {
    let new_id = session_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    set_session_id(&new_id);
    let mut state = WRITE_STATE.lock();
    state.seen_uuids.clear();
    state.last_chain_uuid = None;
    new_id
}

// ---------------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------------

/// loadTranscriptFile subset: parse a session JSONL into
/// (messages_by_uuid, write_order_uuids). Unknown entry types and malformed
/// lines are skipped (metadata entry types are a future extension, not an
/// error). Errors on missing/oversized files.
pub fn load_transcript_file(
    file_path: &Path,
) -> Result<(HashMap<String, Value>, Vec<String>), String> {
    let size = std::fs::metadata(file_path).map_err(|e| e.to_string())?.len();
    if size > MAX_TRANSCRIPT_READ_BYTES {
        return Err(format!(
            "Transcript exceeds {MAX_TRANSCRIPT_READ_BYTES} bytes ({size}); refusing to load"
        ));
    }
    let bytes = std::fs::read(file_path).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&bytes);
    let mut messages: HashMap<String, Value> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if !entry.is_object() {
            continue;
        }
        let etype = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let uuid_val = entry.get("uuid");
        let has_uuid = matches!(uuid_val, Some(v) if !v.is_null()
            && !(v.as_str().map(|s| s.is_empty()).unwrap_or(false)));
        if TRANSCRIPT_TYPES.contains(&etype) && has_uuid {
            let u = value_as_key(uuid_val.unwrap());
            if !messages.contains_key(&u) {
                order.push(u.clone());
            }
            messages.insert(u, entry);
        }
    }
    Ok((messages, order))
}

/// Python str() on a JSON scalar (uuid/parentUuid keys are strings in
/// practice; be tolerant like `str(entry['uuid'])` was).
fn value_as_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Latest leaf = newest message that no other message names as parent
/// (findLatestMessage over pre-computed leafUuids). Timestamp-latest wins;
/// write order breaks ties.
pub fn find_latest_leaf<'a>(
    messages: &'a HashMap<String, Value>,
    order: &[String],
) -> Option<&'a Value> {
    let referenced: HashSet<String> = messages
        .values()
        .filter_map(|m| {
            let p = m.get("parentUuid")?;
            if p.is_null() {
                None
            } else {
                Some(value_as_key(p))
            }
        })
        .collect();
    let mut leaf: Option<&Value> = None;
    let mut leaf_ts = String::new();
    for u in order {
        let m = &messages[u];
        if referenced.contains(u) {
            continue;
        }
        let ts = m.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        if leaf.is_none() || ts >= leaf_ts.as_str() {
            leaf = Some(m);
            leaf_ts = ts.to_string();
        }
    }
    leaf
}

/// buildConversationChain: walk parentUuid leaf→root (cycle-guarded), return
/// root→leaf. Orphaned-sibling recovery is unnecessary for files we write
/// (linear chain — see module docs).
pub fn build_conversation_chain(messages: &HashMap<String, Value>, leaf: &Value) -> Vec<Value> {
    let mut chain: Vec<Value> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut current: Option<&Value> = Some(leaf);
    while let Some(cur) = current {
        let u = cur.get("uuid").map(value_as_key).unwrap_or_default();
        if seen.contains(&u) {
            crate::utils::log::log_error(
                &format!(
                    "Cycle detected in parentUuid chain at message {u}. Returning partial transcript."
                ),
                None,
            );
            break;
        }
        seen.insert(u);
        chain.push(cur.clone());
        current = match cur.get("parentUuid") {
            Some(p) if !p.is_null() => messages.get(&value_as_key(p)),
            _ => None,
        };
    }
    chain.reverse();
    chain
}

/// removeExtraFields + restore: convert transcript entries back into
/// in-memory message values. Storage-only fields are stripped; passthrough
/// extras (isMeta, isCompactSummary, ...) survive the round trip.
pub fn restore_messages(chain: &[Value]) -> Vec<Value> {
    chain
        .iter()
        .map(|entry| {
            let mut msg = Map::new();
            if let Value::Object(m) = entry {
                for (k, v) in m {
                    if !matches!(
                        k.as_str(),
                        "parentUuid"
                            | "isSidechain"
                            | "userType"
                            | "cwd"
                            | "sessionId"
                            | "version"
                            | "gitBranch"
                            | "timestamp"
                    ) {
                        msg.insert(k.clone(), v.clone());
                    }
                }
                if let Some(m) = msg.get("message").cloned() {
                    // Image references back to base64, from the session that wrote them.
                    let sid = entry.get("sessionId").and_then(|s| s.as_str());
                    msg.insert("message".into(), crate::utils::image_store::hydrate(&m, sid));
                }
            }
            Value::Object(msg)
        })
        .collect()
}

/// Load a session's conversation chain as in-memory messages
/// (loadConversationForResume subset). Err("No session ...") for an unknown
/// id; Err(...) for an empty/unreadable transcript.
pub fn load_conversation_for_resume(session_id: &str) -> Result<Vec<Value>, String> {
    let path = get_transcript_path_for_session(session_id);
    if !path.is_file() {
        return Err(format!(
            "No session {session_id} in {}",
            path.parent().map(|p| p.display().to_string()).unwrap_or_default()
        ));
    }
    let (messages, order) = load_transcript_file(&path)?;
    if messages.is_empty() {
        return Err(format!("No messages found in {}", path.display()));
    }
    let leaf = find_latest_leaf(&messages, &order)
        .ok_or_else(|| format!("No valid conversation chain found in {}", path.display()))?;
    let chain = build_conversation_chain(&messages, leaf);
    Ok(restore_messages(trim_to_last_compaction(&chain)))
}

use crate::services::compaction::COMPACT_CONTINUATION_PREFIX;

/// Drop everything a compaction already replaced.
///
/// A compaction rewrites the in-memory history: the turns it summarised are
/// gone and the summary stands in for them. The transcript is append-only and
/// knows nothing about that, so the chain from leaf to root still holds every
/// one of those turns. Replaying it undoes the compaction — the session comes
/// back at the size it had BEFORE, which is how a conversation that closed at
/// 207k reopened as a 458k request the server refused. One compaction is
/// enough to make a session unopenable, and the bigger the session the surer.
///
/// So resume starts where the last compaction did. Matching on the summary's
/// own opening sentence rather than on a flag means the sessions already on
/// disk are readable too — they were written before anything marked them.
/// `isCompactSummary` is honoured first, for transcripts that carry it.
fn trim_to_last_compaction(chain: &[Value]) -> &[Value] {
    let is_boundary = |e: &Value| -> bool {
        if e.get("isCompactSummary").and_then(|v| v.as_bool()).unwrap_or(false) {
            return true;
        }
        if e.get("type").and_then(|v| v.as_str()) != Some("user") {
            return false;
        }
        let content = e.get("message").and_then(|m| m.get("content"));
        let starts = |t: &str| t.trim_start().starts_with(COMPACT_CONTINUATION_PREFIX);
        match content {
            Some(Value::String(t)) => starts(t),
            Some(Value::Array(blocks)) => blocks.iter().any(|b| {
                b.get("type").and_then(|v| v.as_str()) == Some("text")
                    && b.get("text").and_then(|v| v.as_str()).map(starts).unwrap_or(false)
            }),
            _ => false,
        }
    };
    match chain.iter().rposition(is_boundary) {
        Some(i) => &chain[i..],
        None => chain,
    }
}

/// adoptResumedSessionFile: make session_id the active session and keep
/// appending to its EXISTING file. Seeds the dedup set with every uuid
/// already on disk and the chain pointer with the latest leaf, so the
/// restored history is not re-written and new entries chain onto it.
pub fn adopt_resumed_session(session_id: &str) {
    switch_session(Some(session_id));
    let path = get_transcript_path_for_session(session_id);
    if !path.is_file() {
        return;
    }
    let Ok((messages, order)) = load_transcript_file(&path) else {
        return;
    };
    let leaf_uuid = find_latest_leaf(&messages, &order)
        .and_then(|l| l.get("uuid").map(value_as_key));
    let mut state = WRITE_STATE.lock();
    state.seen_uuids = messages.keys().cloned().collect();
    state.last_chain_uuid = leaf_uuid;
}

// ---------------------------------------------------------------------------
// Listing (for /resume)
// ---------------------------------------------------------------------------

/// extractFirstPrompt + getFirstMeaningfulUserMessageTextContent subset:
/// first non-meta, non-compact-summary user text that isn't a control
/// payload; flattened and capped for display.
pub fn extract_first_prompt(chain: &[Value]) -> String {
    static BASH_INPUT_RE: Lazy<regex::Regex> =
        Lazy::new(|| regex::Regex::new(r"(?s)^<bash-input>(.*?)</bash-input>").unwrap());
    for entry in chain {
        if entry.get("type").and_then(|v| v.as_str()) != Some("user") {
            continue;
        }
        if entry.get("isMeta").and_then(|v| v.as_bool()).unwrap_or(false) {
            continue;
        }
        if entry
            .get("isCompactSummary")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            continue;
        }
        let content = entry.get("message").and_then(|m| m.get("content"));
        let mut texts: Vec<String> = Vec::new();
        match content {
            Some(Value::String(s)) => texts.push(s.clone()),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            if !t.is_empty() {
                                texts.push(t.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        for text in texts {
            if text.is_empty() {
                continue;
            }
            // Bash-mode input renders as the user typed it (original: `! cmd`)
            // — checked before the generic XML skip. (Python re.match anchors
            // at the start; the Rust pattern carries the ^ explicitly.)
            if let Some(m) = BASH_INPUT_RE.captures(&text) {
                let s = format!("! {}", m[1].trim());
                return s.chars().take(200).collect();
            }
            if SKIP_FIRST_PROMPT_PATTERN.is_match(&text) {
                continue;
            }
            let result = text.replace('\n', " ").trim().to_string();
            if result.chars().count() > 200 {
                let head: String = result.chars().take(200).collect();
                return format!("{}…", head.trim_end());
            }
            return result;
        }
    }
    "No prompt".to_string()
}

/// countVisibleMessages: user messages with visible (non-tool_result,
/// non-meta) content + assistant messages with text content.
fn count_visible_messages(chain: &[Value]) -> usize {
    let mut count = 0;
    for entry in chain {
        let etype = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let content = entry.get("message").and_then(|m| m.get("content"));
        match etype {
            "user" => {
                if entry.get("isMeta").and_then(|v| v.as_bool()).unwrap_or(false) {
                    continue;
                }
                match content {
                    Some(Value::String(s)) if !s.trim().is_empty() => count += 1,
                    Some(Value::Array(blocks)) => {
                        if blocks.iter().any(|b| {
                            matches!(
                                b.get("type").and_then(|v| v.as_str()),
                                Some("text") | Some("image")
                            )
                        }) {
                            count += 1;
                        }
                    }
                    _ => {}
                }
            }
            "assistant" => {
                if let Some(Value::Array(blocks)) = content {
                    if blocks.iter().any(|b| {
                        b.get("type").and_then(|v| v.as_str()) == Some("text")
                            && !b
                                .get("text")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .trim()
                                .is_empty()
                    }) {
                        count += 1;
                    }
                }
            }
            _ => {}
        }
    }
    count
}

/// LogOption-lite record for one session (fetchLogs subset).
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub full_path: String,
    pub first_prompt: String,
    pub message_count: usize,
    pub created: String,
    pub modified: String,
    pub git_branch: Option<String>,
}

/// fetchLogs subset: LogOption-lite records for every session of the current
/// project, newest-modified first (sortLogs order). Unreadable/empty files
/// are skipped, not fatal.
pub fn list_sessions(limit: Option<usize>) -> Vec<SessionInfo> {
    let project_dir = get_project_dir(&get_original_cwd());
    if !project_dir.is_dir() {
        return Vec::new();
    }
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&project_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
                .filter_map(|p| {
                    let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?;
                    Some((mtime, p))
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    if let Some(limit) = limit {
        files.truncate(limit);
    }
    let mut sessions = Vec::new();
    for (_, path) in files {
        let Ok((messages, order)) = load_transcript_file(&path) else {
            continue;
        };
        if messages.is_empty() {
            continue;
        }
        let Some(leaf) = find_latest_leaf(&messages, &order) else {
            continue;
        };
        let chain = build_conversation_chain(&messages, leaf);
        let Some(first) = chain.first() else {
            continue;
        };
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let last = chain.last().cloned().unwrap_or(Value::Null);
        sessions.push(SessionInfo {
            session_id: leaf
                .get("sessionId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or(stem),
            full_path: path.display().to_string(),
            first_prompt: extract_first_prompt(&chain),
            message_count: count_visible_messages(&chain),
            created: first
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            modified: last
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            git_branch: last
                .get("gitBranch")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        });
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::compaction::get_compact_user_summary_message;

    fn turn(text: &str) -> Value {
        serde_json::json!({
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        })
    }

    fn summary_turn() -> Value {
        turn(&get_compact_user_summary_message("요약", false))
    }

    /// A resume must come back the size the session closed at.
    ///
    /// Measured on a real transcript (3,444 entries, two compactions): the
    /// session closed at 207,490 tokens and reopened as a 458,753-token
    /// request the server refused outright, so the conversation could not be
    /// opened at all. The chain from leaf to root still held every turn the
    /// compactions had thrown away.
    #[test]
    fn a_resume_starts_where_the_last_compaction_did() {
        let old: Vec<Value> = (0..50).map(|i| turn(&format!("before {i}"))).collect();
        let after: Vec<Value> = (0..3).map(|i| turn(&format!("after {i}"))).collect();

        let mut chain = old.clone();
        chain.push(summary_turn());
        chain.extend(after.clone());
        let kept = trim_to_last_compaction(&chain);
        assert_eq!(kept.len(), 4, "resume must start at the compaction, not the root");

        let text_of = |c: &[Value]| -> String { serde_json::to_string(c).unwrap() };
        assert!(!text_of(kept).contains("before"), "nothing the compaction replaced comes back");
        assert!(text_of(kept).contains("after 2"), "and everything since it does");

        // Several compactions: the LAST one wins. Cutting at the first would
        // bring back everything the later ones threw away.
        let mut chain2 = old.clone();
        chain2.push(summary_turn());
        chain2.extend((0..10).map(|_| turn("mid")));
        chain2.push(summary_turn());
        chain2.extend(after.clone());
        let kept2 = trim_to_last_compaction(&chain2);
        assert_eq!(kept2.len(), 4);
        assert!(!text_of(kept2).contains("mid"), "turns between two compactions stay dropped");

        // A flag alone is enough, whatever the text says — and a session that
        // never compacted must come back whole.
        let mut flagged = turn("nothing recognisable");
        flagged["isCompactSummary"] = Value::Bool(true);
        let mut chain3 = old.clone();
        chain3.push(flagged);
        chain3.extend(after.clone());
        assert_eq!(trim_to_last_compaction(&chain3).len(), 4);

        let plain: Vec<Value> = old.iter().chain(after.iter()).cloned().collect();
        assert_eq!(trim_to_last_compaction(&plain).len(), plain.len());
    }

    use super::*;

    /// A per-project lock is only a lock if the OTHER build computes the same
    /// path for the same directory. Pinned as a literal here and in the Python
    /// suite (test_parity_with_rust.py), not derived from sanitize_path on both
    /// sides — that would only assert that each build agrees with itself.
    #[test]
    fn the_project_lock_path_is_the_one_the_python_build_uses() {
        assert_eq!(
            project_lock("goal-gates.txt", "/tmp/x"),
            crate::utils::config::LOCKS_DIR.join("-tmp-x.goal-gates.txt.lock")
        );
        assert_eq!(
            project_lock("settings.json", "/home/user/work"),
            crate::utils::config::LOCKS_DIR.join("-home-user-work.settings.json.lock")
        );
    }

    #[test]
    fn sanitize_matches_python() {
        assert_eq!(sanitize_path("/home/user/work"), "-home-user-work");
        assert_eq!(sanitize_path("abC123"), "abC123");
        // Over-long: truncated + base36 djb2 suffix, deterministic.
        let long = "x".repeat(300);
        let s = sanitize_path(&long);
        assert!(s.starts_with(&"x".repeat(200)));
        assert!(s.chars().nth(200) == Some('-'));
        assert!(s.len() > 201);
    }

    #[test]
    fn sanitize_long_gets_hash() {
        // (kept from the U3 stub-extension's test set)
        let long = "a/".repeat(300);
        let s = sanitize_path(&long);
        assert!(s.len() > MAX_SANITIZED_LENGTH); // head + '-' + hash
        assert!(s.starts_with(&"a-".repeat(100)));
        assert!(s[MAX_SANITIZED_LENGTH..].starts_with('-'));
    }

    #[test]
    fn djb2_signed_like_js() {
        // djb2("") = 5381; must be stable and within i32 range.
        assert_eq!(djb2_hash(""), 5381);
        let h = djb2_hash("/home/user/some/very/long/path");
        assert!(h >= i32::MIN as i64 && h <= i32::MAX as i64);
    }

    #[test]
    fn to_base36_values() {
        assert_eq!(to_base36(0), "0");
        assert_eq!(to_base36(35), "z");
        assert_eq!(to_base36(36), "10");
    }

    #[test]
    fn first_prompt_extraction() {
        let chain = vec![
            serde_json::json!({"type":"user","isMeta":true,
                "message":{"content":"meta line"}}),
            serde_json::json!({"type":"user",
                "message":{"content":"<task-notification> x"}}),
            serde_json::json!({"type":"user",
                "message":{"content":[{"type":"text","text":"real question\nline2"}]}}),
        ];
        assert_eq!(extract_first_prompt(&chain), "real question line2");
    }

    #[test]
    fn first_prompt_bash_input() {
        let chain = vec![serde_json::json!({"type":"user",
            "message":{"content":"<bash-input>ls -la</bash-input>"}})];
        assert_eq!(extract_first_prompt(&chain), "! ls -la");
    }

    #[test]
    fn chain_walk_and_restore() {
        let mut messages = HashMap::new();
        let a = serde_json::json!({"uuid":"a","parentUuid":null,"type":"user",
            "timestamp":"2026-01-01T00:00:00.000Z","message":{"content":"hi"},
            "isSidechain":false,"userType":"external","cwd":"/x","sessionId":"s",
            "version":"2.0.0"});
        let b = serde_json::json!({"uuid":"b","parentUuid":"a","type":"assistant",
            "timestamp":"2026-01-01T00:00:01.000Z","message":{"content":[]},
            "isSidechain":false,"userType":"external","cwd":"/x","sessionId":"s",
            "version":"2.0.0"});
        messages.insert("a".to_string(), a);
        messages.insert("b".to_string(), b.clone());
        let order = vec!["a".to_string(), "b".to_string()];
        let leaf = find_latest_leaf(&messages, &order).unwrap();
        assert_eq!(leaf.get("uuid").unwrap(), "b");
        let chain = build_conversation_chain(&messages, leaf);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].get("uuid").unwrap(), "a");
        let restored = restore_messages(&chain);
        assert!(restored[0].get("cwd").is_none());
        assert!(restored[0].get("uuid").is_some());
    }
}
