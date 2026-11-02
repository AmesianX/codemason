//! Telegram adapter.
//! Equivalent to codemason/remote/adapters/telegram.py — see that file for the
//! full design narrative (why Telegram first, why max_sensitivity is INTERNAL,
//! why the poll lock and offset are keyed by bot token, why probing holds its
//! poll open, why setMyName reads before writing).
//!
//! Substrate divergences (documented):
//! - Python used stdlib urllib so the notification path never changed
//!   dependency resolution; Rust uses the crate's existing async reqwest.
//!   Network methods are therefore `async fn`.
//! - The poll lock is no longer Python's exclusive-create file. O_CREAT|O_EXCL
//!   plus a PID-checked stale reclaim needs three syscalls to take a lock, and
//!   two processes interleaving them both end up "holding" it — one deletes the
//!   other's fresh lock while reclaiming what it still thinks is a corpse, and
//!   then both poll the same bot into 409s. On unix it is now a real flock on
//!   that same path, held for the poller's lifetime: one atomic operation, and
//!   the kernel releases it for us when a holder dies, so there is no stale
//!   reclaim to race over. The pid is still written into the body, but only as
//!   a label for the standby message. Non-unix has no flock in std, so it keeps
//!   the PID file with the reclaim tightened (see acquire_poll_lock).

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::remote::events::{Event, EventKind, Sensitivity};
use crate::utils::i18n::{tr, trf, FmtArg};

pub const API: &str = "https://api.telegram.org";

/// Telegram hard-caps a message at 4096 characters. Leave room for the title,
/// markup and the truncation notice so a long body degrades visibly.
const MAX_TEXT: usize = 3800;

/// 429 handling. Telegram states exactly how long to wait, so these are only a
/// ceiling for a malformed or missing retry_after.
const RATE_LIMIT_RETRIES: usize = 3;
const RATE_LIMIT_MAX_WAIT: f64 = 30.0;

fn icon(kind: EventKind) -> &'static str {
    match kind {
        EventKind::TurnDone => "✅",
        EventKind::PermissionNeeded => "🔐",
        EventKind::QuestionAsked => "❓",
        EventKind::Error => "🔴",
        EventKind::TaskDone => "📦",
        EventKind::Info => "ℹ️",
    }
}

/// Events that should buzz the phone. Everything else arrives silently so
/// routine progress does not train you to ignore the channel.
fn is_loud(kind: EventKind) -> bool {
    matches!(kind, EventKind::PermissionNeeded | EventKind::QuestionAsked | EventKind::Error)
}

// getUpdates is exclusive — see the Python module doc for why both the lock
// and the offset are keyed by the BOT token, not just the home directory.
static REMOTE_DIR: Lazy<PathBuf> =
    Lazy::new(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".codemason").join("remote"));

/// Pre-token-scoping paths. Read once for the offset so an upgrade does not
/// re-deliver a day of history, never written.
static LEGACY_LOCK_PATH: Lazy<PathBuf> = Lazy::new(|| REMOTE_DIR.join("poll.lock"));
static LEGACY_OFFSET_PATH: Lazy<PathBuf> = Lazy::new(|| REMOTE_DIR.join("offset"));

/// The legacy lock is checked once per process, not on every acquire.
static LEGACY_LOCK_CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bot API getFile serves files up to 20 MB.
const MAX_DOWNLOAD: usize = 20 * 1024 * 1024;
/// Tries for one file download (download_image).
const DOWNLOAD_ATTEMPTS: usize = 3;
/// Attachments older than this are deleted when the poller starts. A session
/// that Read one keeps its own copy (utils/image_store.rs), so this folder is
/// only the landing place between the phone and that Read.
pub const ATTACHMENT_MAX_AGE_DAYS: f64 = 30.0;
/// Hex digits of the content hash an attachment is named by. The path reaches
/// the model inside a line of text and comes back typed into a Read, and all 64
/// came back once with a digit dropped: ENOENT, then the identical Read
/// succeeded. A name is only as good as the trip it survives, and 16 digits
/// still name the bytes. Truncation is a bet the full hash does not make, so
/// `attachment_path` does not take it.
pub const ATTACHMENT_NAME_DIGITS: usize = 16;
/// Extensions an image from the chat is saved under; anything else gets .jpg.
const IMAGE_EXTS: &[&str] = &[".jpg", ".jpeg", ".png", ".gif", ".webp"];

pub fn attachments_dir() -> PathBuf {
    REMOTE_DIR.join("attachments")
}

/// Delete attachments older than `max_age_days`; returns how many.
pub fn prune_attachments(max_age_days: f64) -> usize {
    let Ok(entries) = std::fs::read_dir(attachments_dir()) else { return 0 };
    let max_age = std::time::Duration::from_secs_f64(max_age_days * 86400.0);
    let mut removed = 0;
    for e in entries.flatten() {
        let Ok(meta) = e.path().symlink_metadata() else { continue };
        let old = meta.modified().ok().and_then(|m| m.elapsed().ok()).is_some_and(|age| age > max_age);
        if meta.is_file() && old && std::fs::remove_file(e.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Extension to save an image under: the sender's file name, else Telegram's
/// path, else .jpg — and only ever one of IMAGE_EXTS, since it lands in a file
/// name and in the text the model reads.
fn image_ext(name_hint: &str, remote: &str) -> String {
    for name in [name_hint, remote] {
        if let Some(e) = std::path::Path::new(name).extension().and_then(|e| e.to_str()) {
            let e = format!(".{}", e.to_lowercase());
            if IMAGE_EXTS.contains(&e.as_str()) {
                return e;
            }
        }
    }
    ".jpg".into()
}

/// Where these bytes are saved, and the name the model will have to type back.
///
/// Short by default, and still the content address, so the same photo twice is
/// still one file. Two different photos under one short name is ~1e-13 by
/// chance but cheap to force (a 2**32 birthday search), and the cost would be a
/// Read quietly answering with the other photo -- so the bet is not taken: the
/// bytes are compared, and the rare clash wears the full name, which cannot
/// clash. A short name that holds these same bytes is simply reused.
fn attachment_path(folder: &std::path::Path, data: &[u8], ext: &str) -> PathBuf {
    use sha2::{Digest, Sha256};
    let sha = hex::encode(Sha256::digest(data));
    let short = folder.join(format!("{}{ext}", &sha[..ATTACHMENT_NAME_DIGITS]));
    match std::fs::read(&short) {
        Ok(existing) if existing != data => folder.join(format!("{sha}{ext}")),
        _ => short,
    }
}

/// GET `url`, retrying a failed attempt with a short pause.
///
/// A GET is safe to repeat, and the first failure seen in use was a bare
/// "error sending request" on a WSL2 desktop (getFile, a moment earlier, had
/// worked): a pooled keep-alive connection that NAT had dropped, or a route that
/// failed once, look the same at that level. The error keeps its whole source
/// chain — the top-level Display alone said nothing about which it was. A 4xx
/// is final (the file is gone or the path is wrong) except 429, which says to
/// come back later. At most `cap` + 1 bytes are read: more than `cap` is an
/// answer in itself, and nothing larger is held in memory.
async fn get_bytes_retrying(url: &str, timeout_s: f64, cap: usize) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..DOWNLOAD_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;
        }
        let resp = match HTTP
            .get(url)
            .timeout(std::time::Duration::from_secs_f64(timeout_s))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last = error_chain(e);
                continue;
            }
        };
        let status = resp.status();
        if !status.is_success() {
            last = format!("HTTP {}", status.as_u16());
            if status.is_client_error() && status.as_u16() != 429 {
                break;
            }
            continue;
        }
        let mut resp = resp;
        let mut body: Vec<u8> = Vec::new();
        let read = loop {
            match resp.chunk().await {
                Ok(Some(c)) => {
                    body.extend_from_slice(&c[..c.len().min(cap + 1 - body.len())]);
                    if body.len() > cap {
                        break Ok(());
                    }
                }
                Ok(None) => break Ok(()),
                Err(e) => break Err(error_chain(e)),
            }
        };
        match read {
            Ok(()) => return Ok(body),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// True when a failed Bot API call is worth sending again.
///
/// post() reports a transport failure as the error chain and an HTTP answer as
/// "HTTP <code> <desc>", so a 4xx is Telegram itself saying the request was
/// wrong — the file id is unknown, the file is gone — and repeating it changes
/// nothing. A 429 is already waited out inside post(). Everything else (no
/// status at all, or a 5xx) is the connection, and a read-only call can go out
/// again.
fn worth_retrying(err: &str) -> bool {
    !err.starts_with("HTTP 4")
}

/// Call a read-only Bot API method until it answers, pausing between tries.
///
/// Only for methods that are safe to repeat. post() does not retry a transport
/// failure, which is right for sendMessage — a request that timed out may well
/// have arrived, and the phone does not need the message twice — but getFile
/// only reads, and the failure that brought this here was on that call: a photo
/// from the phone came back as "[could not be downloaded: error sending
/// request]" while the file GET beside it (get_bytes_retrying, which does retry)
/// never got its turn. On this desktop IPv6 reaches Telegram over the 5G link
/// and loses roughly one connection in twenty; IPv4 over the other adapter
/// loses none. One lost connection should cost a pause, not the photo.
async fn retry_read<'a>(
    mut call: impl FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Value> + Send + 'a>>,
) -> Value {
    let mut last = Value::Null;
    for attempt in 0..DOWNLOAD_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;
        }
        let resp = call().await;
        if resp.get("ok").and_then(|o| o.as_bool()) == Some(true) {
            return resp;
        }
        let again = resp
            .get("error")
            .or_else(|| resp.get("description"))
            .and_then(|e| e.as_str())
            .is_none_or(worth_retrying);
        last = resp;
        if !again {
            break;
        }
    }
    last
}

/// A reqwest error with its whole source chain ("error sending request: …:
/// connection reset"), URL removed first — the bot token is in the URL.
fn error_chain(e: reqwest::Error) -> String {
    let e = e.without_url();
    let mut out = e.to_string();
    let mut src = std::error::Error::source(&e);
    while let Some(s) = src {
        out.push_str(": ");
        out.push_str(&s.to_string());
        src = s.source();
    }
    out
}

/// Create ~/.codemason/remote, owner-only, and return it. Centralised so the
/// directory mode does not depend on which caller's mkdir wins the race.
pub fn ensure_remote_dir() -> PathBuf {
    let _ = std::fs::create_dir_all(&*REMOTE_DIR);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&*REMOTE_DIR, std::fs::Permissions::from_mode(0o700));
    }
    REMOTE_DIR.clone()
}

/// Short fingerprint of a bot token, for use in a filename or a record.
/// Hashed, not truncated: a bot token is a credential.
pub fn token_tag(token: &str) -> String {
    if token.is_empty() {
        return "none".to_string();
    }
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    hex::encode(h.finalize())[..12].to_string()
}

/// Fingerprint of the bot THIS instance is using.
fn own_token_tag() -> String {
    let (token, _) = credentials();
    token_tag(&token)
}

pub fn lock_path() -> PathBuf {
    REMOTE_DIR.join(format!("poll.{}.lock", own_token_tag()))
}

/// Newest-message record, per bot — see the Python docstring for why the chat
/// id alone cannot key it.
fn last_message_path() -> PathBuf {
    REMOTE_DIR.join(format!("last_message.{}", own_token_tag()))
}

/// PID currently holding this bot's poll lock, as a string, or "".
pub fn poll_lock_holder() -> String {
    std::fs::read_to_string(lock_path()).map(|s| s.trim().to_string()).unwrap_or_default()
}

fn offset_path() -> PathBuf {
    REMOTE_DIR.join(format!("offset.{}", own_token_tag()))
}

/// Record a message that now sits below anything posted before it.
///
/// Progress messages are deliberately *not* recorded — they relocate
/// themselves, and recording them would have two sessions each bump below the
/// other's spinner forever.
pub fn note_chat_message(message_id: Option<i64>) {
    let Some(mid) = message_id else { return };
    if mid <= 0 {
        return;
    }
    let (_, chat) = credentials();
    if chat.is_empty() {
        return;
    }
    if mid <= newest_chat_message() {
        return;
    }
    // Written through a temp file so a reader never sees a half-written id,
    // and per-pid so two processes recording at once cannot share one.
    let path = last_message_path();
    let tmp = path.with_file_name(format!(
        "{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    ensure_remote_dir();
    if std::fs::write(&tmp, format!("{chat} {mid}")).is_ok() {
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The newest recorded message id for the configured chat, or 0.
pub fn newest_chat_message() -> i64 {
    let (_, chat) = credentials();
    let Ok(raw) = std::fs::read_to_string(last_message_path()) else {
        return 0;
    };
    let raw = raw.trim();
    let (stored_chat, mid) = match raw.split_once(' ') {
        Some((a, b)) => (a, b),
        None => (raw, ""),
    };
    if chat.is_empty() || stored_chat != chat {
        return 0; // a different chat's numbering
    }
    mid.parse().unwrap_or(0)
}

/// Where credentials may live, most specific first. CONFIG_DIR so
/// CODEMASON_CONFIG_DIR relocates the credentials with the rest of the tree.
fn config_paths() -> Vec<PathBuf> {
    vec![crate::utils::config::CONFIG_DIR.join("remote.env")]
}

/// Four, because four is the number of windows somebody actually opens at once
/// and because a lone slot reads as "one bot is the design". Blank slots cost
/// nothing: as_list drops them.
const SCAFFOLD_SLOTS: usize = 4;

/// The file the user is meant to edit. Named once, so messages agree.
pub fn credentials_path() -> PathBuf {
    config_paths().remove(0)
}

/// Write a commented, empty remote.env if none exists. Returns it, or None.
/// Never overwrites; failure is silent (remote is optional).
pub fn ensure_credentials_scaffold() -> Option<PathBuf> {
    let paths = config_paths();
    if paths.iter().any(|p| p.exists()) {
        return None;
    }
    let path = paths[0].clone();
    let mut body = tr(
        "# codemason remote channel — drive this session from Telegram.\n#\n# 1. In Telegram, talk to @BotFather and send /newbot. It replies with\n#    a token. Paste one per slot below. Blank slots are ignored.\n# 2. Every codemason you start claims one FREE bot by itself — no flag,\n#    no name to remember. Four slots means four windows at the same\n#    time, anywhere: this machine, a VM, another desktop. Want more at\n#    once? Add more lines.\n# 3. Send /start to each bot once, so it is allowed to message you.\n# 4. TELEGRAM_CHAT_ID is your own numeric id — ask @userinfobot for it.\n#\n# This file holds your bot tokens. Anyone who can read it can read your\n# alerts and type into your sessions, so keep it mode 600.\n",
    );
    body.push('\n');
    for _ in 0..SCAFFOLD_SLOTS {
        body.push_str("TELEGRAM_BOT_TOKEN=\n");
    }
    body.push_str("\nTELEGRAM_CHAT_ID=\n");

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Created 600 rather than written and chmod-ed: the user fills the slots in
    // place, and a file that was ever world-readable is one umask away from
    // staying that way.
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    match opts.open(&path) {
        Ok(mut f) => {
            use std::io::Write;
            if f.write_all(body.as_bytes()).is_err() {
                return None;
            }
            Some(path)
        }
        Err(_) => None,
    }
}

/// Warn if the credentials file is readable by other accounts, or "".
/// Reported, not corrected. POSIX only.
pub fn credentials_file_warning() -> String {
    #[cfg(not(unix))]
    {
        return String::new();
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        for path in config_paths() {
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            let mode = meta.mode();
            if mode & 0o077 != 0 {
                return trf(
                    "{path} is readable by other users on this machine (mode {mode:o}). It holds your bot token — chmod 600 it.",
                    &[
                        ("path", FmtArg::Str(path.to_string_lossy().into_owned())),
                        ("mode", FmtArg::Str(format!("{:o}", mode & 0o777))),
                    ],
                );
            }
        }
        String::new()
    }
}

/// Parse KEY=VALUE lines. Repeated keys collect into a list — the file is a
/// POOL of bots, one line each, and the shape has to survive parsing.
fn load_env_file(path: &std::path::Path) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let (k, v) = line.split_once('=').unwrap();
        let k = k.trim().to_string();
        let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
        out.entry(k).or_default().push(v);
    }
    out
}

fn as_list(values: Option<&Vec<String>>) -> Vec<String> {
    values
        .map(|v| v.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default()
}

/// Every bot token available to this machine, in the order written.
/// Environment wins, so a single-token shell override still works.
pub fn bot_pool() -> Vec<String> {
    if let Ok(env) = std::env::var("TELEGRAM_BOT_TOKEN") {
        let env = env.trim().to_string();
        if !env.is_empty() {
            return vec![env];
        }
    }
    for path in config_paths() {
        if !path.exists() {
            continue;
        }
        let tokens = as_list(load_env_file(&path).get("TELEGRAM_BOT_TOKEN"));
        if !tokens.is_empty() {
            return tokens;
        }
    }
    Vec::new()
}

/// The chat every bot in the pool talks to. One account, so one id.
pub fn chat_id() -> String {
    if let Ok(env) = std::env::var("TELEGRAM_CHAT_ID") {
        let env = env.trim().to_string();
        if !env.is_empty() {
            return env;
        }
    }
    for path in config_paths() {
        if path.exists() {
            let found = as_list(load_env_file(&path).get("TELEGRAM_CHAT_ID"));
            if let Some(first) = found.into_iter().next() {
                return first;
            }
        }
    }
    String::new()
}

/// Display width in terminal cells — CJK glyphs occupy two.
fn w(s: &str) -> usize {
    crate::utils::formatting::display_width(s)
}

fn get_cfg_collapse_over() -> i64 {
    crate::utils::config::get_config().remote_collapse_over
}

/// Should this body arrive in a quote box? Boxed, not folded: Telegram draws a
/// blockquote in a smaller face, which is room enough, while an *expandable*
/// one closed again on its own and cost a tap on every look. (See the Python
/// `_collapse`.)
fn collapse(event: &Event, body: &str) -> bool {
    if event.meta.get("collapse").map(|v| v.as_bool().unwrap_or(false)).unwrap_or(false) {
        return true;
    }
    let over = get_cfg_collapse_over();
    over > 0 && body.chars().count() as i64 > over
}

/// Widest grid still worth drawing, in cells. See `remote_table_width`.
fn table_width() -> usize {
    let w = crate::utils::config::get_config().remote_table_width;
    if w > 0 {
        w as usize
    } else {
        1_000_000 // 0 or negative: always a grid
    }
}

static RE_INLINE_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"`([^`\n]+)`").unwrap());
static RE_BOLD: Lazy<Regex> = Lazy::new(|| Regex::new(r"\*\*([^*\n]+)\*\*").unwrap());
static RE_HEADING: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^#{1,6}\s+(.+)$").unwrap());
static RE_LINK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"\[([^\]\n]+)\]\((https?://[^)\s"']+)\)"#).unwrap());
static RE_TABLE_SEP: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\|[\s:|-]+\|$").unwrap());
static RE_TAG: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());

/// html.escape (quote=True default in Python's html.escape; the adapter uses
/// the default for titles/cells).
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

/// Italic: Python `(?<![\w*])\*([^*\n]+)\*(?![\w*])` — the regex crate has no
/// look-around, so scan match positions and check the neighbour chars by hand
/// (identical semantics; the char class is \w or '*').
fn apply_italic(s: &str) -> String {
    static RE_IT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\*([^*\n]+)\*").unwrap());
    let mut out = String::with_capacity(s.len());
    let mut last = 0usize;
    for m in RE_IT.captures_iter(s) {
        let whole = m.get(0).unwrap();
        let before_ok = s[..whole.start()]
            .chars()
            .next_back()
            .map(|c| !(c.is_alphanumeric() || c == '_' || c == '*'))
            .unwrap_or(true);
        let after_ok = s[whole.end()..]
            .chars()
            .next()
            .map(|c| !(c.is_alphanumeric() || c == '_' || c == '*'))
            .unwrap_or(true);
        if whole.start() < last {
            continue;
        }
        if before_ok && after_ok {
            out.push_str(&s[last..whole.start()]);
            out.push_str("<i>");
            out.push_str(&m[1]);
            out.push_str("</i>");
            last = whole.end();
        }
    }
    out.push_str(&s[last..]);
    out
}

/// The inline Markdown rules, applied to one already-escaped cell.
fn inline(s: &str) -> String {
    let s = RE_INLINE_CODE.replace_all(s, "<code>$1</code>");
    let s = RE_BOLD.replace_all(&s, "<b>$1</b>");
    apply_italic(&s)
}

/// Drop emphasis markers that would show as literal characters in `<pre>`.
fn unmark(s: &str) -> String {
    let s = RE_BOLD.replace_all(s, "$1");
    RE_INLINE_CODE.replace_all(&s, "$1").into_owned()
}

/// Which column names a row, rather than merely describes it — the leftmost
/// column whose values are (mostly) distinct; 0 as the fallback.
fn title_column(body: &[Vec<String>]) -> usize {
    let ncols = body.first().map(|r| r.len()).unwrap_or(0);
    for i in 0..ncols {
        let filled: Vec<&str> =
            body.iter().filter_map(|r| Some(r.get(i)?.trim()).filter(|s| !s.is_empty())).collect();
        if filled.len() < body.len() {
            continue; // gaps: not an identifier
        }
        let unique: std::collections::HashSet<&str> = filled.iter().copied().collect();
        if unique.len() >= std::cmp::max(2, (filled.len() as f64 * 0.8) as usize) {
            return i;
        }
    }
    0
}

/// A table too wide for the screen, re-laid one row at a time. See the Python
/// docstrings for the full rationale.
fn reflow_table(cells: &[Vec<String>]) -> String {
    let (header, body): (Vec<String>, &[Vec<String>]) = if cells.len() > 1 {
        (cells[0].iter().map(|c| c.trim().to_string()).collect(), &cells[1..])
    } else {
        (Vec::new(), cells) // header-only: nothing to key by
    };
    let key = title_column(body);

    let mut out: Vec<String> = Vec::new();
    for row in body {
        let first = row.get(key).map(|c| inline(c.trim())).unwrap_or_default();
        // Bold it only when it is plain text — Telegram rejects the whole
        // message on markup it cannot parse.
        let title = if !first.is_empty() && !first.contains('<') {
            format!("<b>{first}</b>")
        } else {
            first
        };
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (i, cell) in row.iter().enumerate() {
            let value = cell.trim();
            if i == key || value.is_empty() {
                continue; // an empty cell is not worth a line
            }
            let name = header.get(i).map(|h| inline(h)).unwrap_or_default();
            pairs.push((name, inline(value)));
        }
        // One line while one line will hold it.
        if !title.is_empty() && !pairs.is_empty() {
            let joined = format!(
                "{title} — {}",
                pairs.iter().map(|(_, v)| v.as_str()).collect::<Vec<_>>().join(" · ")
            );
            let visible = RE_TAG.replace_all(&joined, "");
            if pairs.len() == 1 || w(&visible) <= table_width() {
                out.push(joined);
                continue;
            }
        }
        let mut lines: Vec<String> = if title.is_empty() { Vec::new() } else { vec![title] };
        for (n, v) in &pairs {
            lines.push(if n.is_empty() { v.clone() } else { format!("{n}: {v}") });
        }
        if !lines.is_empty() {
            out.push(lines.join("\n"));
        }
    }
    if out.is_empty() {
        return String::new();
    }
    // Spacing decided at each boundary, not once for the table.
    let mut text = out[0].clone();
    for pair in out.windows(2) {
        let (prev, row) = (&pair[0], &pair[1]);
        text.push_str(if prev.contains('\n') || row.contains('\n') { "\n\n" } else { "\n" });
        text.push_str(row);
    }
    format!("<blockquote>{text}</blockquote>")
}

/// Render the model's Markdown as the small HTML subset Telegram accepts.
/// Escape first, then translate on the escaped text; the caller retries
/// unformatted if Telegram refuses the markup anyway.
fn md_to_html(text: &str) -> String {
    let mut out = html_escape(text);
    let mut blocks: Vec<String> = Vec::new();

    // Fenced blocks first: contents must not be touched by the inline rules.
    static RE_FENCE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```(\w*)\n?(.*?)```").unwrap());
    out = RE_FENCE
        .replace_all(&out, |m: &regex::Captures| {
            let lang = m.get(1).map(|g| g.as_str()).unwrap_or("");
            let code = m.get(2).map(|g| g.as_str()).unwrap_or("");
            blocks.push(if lang.is_empty() {
                format!("<pre><code>{code}</code></pre>")
            } else {
                format!("<pre><code class=\"language-{lang}\">{code}</code></pre>")
            });
            format!("\u{0}{}\u{0}", blocks.len() - 1)
        })
        .into_owned();

    // A fence left open by truncation.
    if out.matches("```").count() % 2 == 1 {
        if let Some(pos) = out.find("```") {
            let head = out[..pos].to_string();
            let rest_full = &out[pos + 3..];
            let rest = rest_full.split_once('\n').map(|(_, r)| r).unwrap_or("");
            blocks.push(format!("<pre><code>{rest}</code></pre>"));
            out = format!("{head}\u{0}{}\u{0}", blocks.len() - 1);
        }
    }

    // Markdown tables — aligned monospace when they fit, re-laid otherwise.
    static RE_TABLE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?m)((?:^\|.*\|[ \t]*$\n?){2,})").unwrap());
    out = RE_TABLE
        .replace_all(&out, |m: &regex::Captures| {
            let raw = m.get(1).unwrap().as_str();
            let rows: Vec<&str> = raw.trim().split('\n').map(|r| r.trim()).filter(|r| !r.is_empty()).collect();
            let rows: Vec<&str> = rows.into_iter().filter(|r| !RE_TABLE_SEP.is_match(r)).collect();
            let mut cells: Vec<Vec<String>> = rows
                .iter()
                .map(|r| {
                    r.trim_matches('|').split('|').map(|c| c.trim().to_string()).collect::<Vec<_>>()
                })
                .collect();
            if cells.is_empty() {
                return raw.to_string();
            }
            let ncols = cells.iter().map(|r| r.len()).max().unwrap_or(0);
            for r in cells.iter_mut() {
                while r.len() < ncols {
                    r.push(String::new());
                }
            }
            // Emphasis markers have no meaning inside <pre>.
            let grid: Vec<Vec<String>> =
                cells.iter().map(|r| r.iter().map(|c| unmark(c)).collect()).collect();
            let cols: Vec<usize> = (0..ncols)
                .map(|i| grid.iter().map(|r| w(&r[i])).max().unwrap_or(0))
                .collect();
            if cols.iter().sum::<usize>() + 2 * (ncols.saturating_sub(1)) <= table_width() {
                let lines: Vec<String> = grid
                    .iter()
                    .map(|r| {
                        r.iter()
                            .enumerate()
                            .map(|(i, c)| format!("{}{}", c, " ".repeat(cols[i].saturating_sub(w(c)))))
                            .collect::<Vec<_>>()
                            .join("  ")
                            .trim_end()
                            .to_string()
                    })
                    .collect();
                blocks.push(format!("<pre>{}</pre>", lines.join("\n")));
            } else {
                blocks.push(reflow_table(&cells));
            }
            // The match swallowed the newline that ended the last row.
            format!("\u{0}{}\u{0}\n", blocks.len() - 1)
        })
        .into_owned();

    out = RE_INLINE_CODE.replace_all(&out, "<code>$1</code>").into_owned();
    out = RE_BOLD.replace_all(&out, "<b>$1</b>").into_owned();
    out = apply_italic(&out);
    // [text](url) — the url is already escaped; anything with a quote is left
    // as plain text (the escape turned quotes into entities already).
    out = RE_LINK.replace_all(&out, "<a href=\"$2\">$1</a>").into_owned();
    // Headings have no equivalent; bold reads the same on a phone.
    out = RE_HEADING.replace_all(&out, "<b>$1</b>").into_owned();

    for (i, block) in blocks.iter().enumerate() {
        out = out.replace(&format!("\u{0}{i}\u{0}"), block);
    }
    out
}

fn session_id() -> String {
    crate::utils::state::get_session_id()
}

/// The bot this process claimed, once it has one.
static CLAIMED_TOKEN: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub fn claimed_token() -> String {
    CLAIMED_TOKEN.lock().clone().unwrap_or_default()
}

/// (token, chat_id) for THIS instance. Before a claim, the first bot in the
/// pool — enough for is_configured() and for outbound, which is not exclusive.
pub fn credentials() -> (String, String) {
    let pool = bot_pool();
    let token = CLAIMED_TOKEN
        .lock()
        .clone()
        .unwrap_or_else(|| pool.first().cloned().unwrap_or_default());
    (token, chat_id())
}

/// How long a probe holds its poll open while deciding whether a bot is taken.
/// Derived, not picked — must outlast one incumbent backoff plus a round trip.
const PROBE_SECONDS: u64 = 6;

// Probe verdicts — four outcomes, because "cannot use this bot" has different
// causes needing different fixes (see the Python constants' comment).
pub const FREE: &str = "free";
pub const TAKEN: &str = "taken";
pub const INVALID: &str = "invalid"; // Telegram rejects the token itself
pub const UNREACHABLE: &str = "unreachable"; // we could not ask

/// One client for every Bot API call, so connections are pooled.
///
/// Idle connections are dropped after 30 s instead of reqwest's 90. The user's
/// desktop reaches Telegram through a KT 5G router, and a mobile carrier's NAT
/// forgets an idle TCP mapping well before 90 s; reusing one of those sends
/// into nothing and fails as a bare "error sending request" — what the first
/// photo download from that desktop did, a moment after getFile had worked on
/// a live connection. urllib (the Python port) opens a fresh connection per
/// call and never had this.
static HTTP: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
});

/// FREE / TAKEN / INVALID / UNREACHABLE for one bot.
pub async fn probe_bot(token: &str) -> &'static str {
    let params = [
        ("offset", "0".to_string()), // peek only; the offset file is untouched
        ("timeout", PROBE_SECONDS.to_string()),
        ("allowed_updates", "[]".to_string()),
    ];
    let resp = HTTP
        .post(format!("{API}/bot{token}/getUpdates"))
        .form(&params)
        .timeout(std::time::Duration::from_secs(PROBE_SECONDS + 15))
        .send()
        .await;
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let _ = r.bytes().await;
            match status {
                200..=299 => FREE,
                409 => TAKEN,        // an instance is alive on this bot
                401 | 404 => INVALID, // revoked, mistyped, or never existed
                _ => UNREACHABLE,     // 429, 5xx — true later, maybe, not now
            }
        }
        Err(_) => UNREACHABLE,
    }
}

/// Why the last claim failed, per bot (token_tag -> verdict).
static LAST_CLAIM_VERDICTS: Lazy<Mutex<HashMap<String, &'static str>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn last_claim_verdicts() -> HashMap<String, &'static str> {
    LAST_CLAIM_VERDICTS.lock().clone()
}

/// Take a free bot from the pool, or None when every one is in use.
/// `preferred` is tried first — the bot this directory used last time.
pub async fn claim_bot(preferred: &str) -> Option<String> {
    let pool = bot_pool();
    LAST_CLAIM_VERDICTS.lock().clear();
    if pool.is_empty() {
        return None;
    }
    let order: Vec<String> = if !preferred.is_empty() && pool.contains(&preferred.to_string()) {
        let mut o = vec![preferred.to_string()];
        o.extend(pool.iter().filter(|t| t.as_str() != preferred).cloned());
        o
    } else {
        pool
    };
    for token in order {
        let verdict = probe_bot(&token).await;
        LAST_CLAIM_VERDICTS.lock().insert(token_tag(&token), verdict);
        if verdict == FREE {
            *CLAIMED_TOKEN.lock() = Some(token.clone());
            return Some(token);
        }
    }
    None
}

pub fn release_bot() {
    *CLAIMED_TOKEN.lock() = None;
}

static HANDLE_CACHE: Lazy<Mutex<HashMap<String, String>>> = Lazy::new(|| Mutex::new(HashMap::new()));

async fn get_me(token: &str) -> Option<Value> {
    let resp = HTTP
        .post(format!("{API}/bot{token}/getMe"))
        .form(&[("", "")])
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .ok()?;
    resp.json::<Value>().await.ok()
}

/// @username of the claimed bot, cached. "" when unknown.
pub async fn claimed_handle() -> String {
    let (token, _) = credentials();
    if token.is_empty() {
        return String::new();
    }
    let tag = token_tag(&token);
    if let Some(h) = HANDLE_CACHE.lock().get(&tag) {
        return h.clone();
    }
    let handle = match get_me(&token).await {
        Some(me) if me.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) => me
            .get("result")
            .and_then(|r| r.get("username"))
            .and_then(|u| u.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    };
    HANDLE_CACHE.lock().insert(tag, handle.clone());
    handle
}

/// 'Name (@handle)' for the claimed bot, or "" — for /remote status.
pub async fn describe_claimed_bot() -> String {
    let (token, _) = credentials();
    if token.is_empty() {
        return String::new();
    }
    let Some(me) = get_me(&token).await else {
        return String::new();
    };
    if !me.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        return String::new();
    }
    let res = me.get("result").cloned().unwrap_or(Value::Null);
    let name = res.get("first_name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let handle = res.get("username").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if handle.is_empty() {
        name
    } else {
        format!("{name} (@{handle})")
    }
}

/// The claimed bot's current display name, or "".
pub async fn get_bot_name() -> String {
    let (token, _) = credentials();
    if token.is_empty() {
        return String::new();
    }
    let resp = HTTP
        .post(format!("{API}/bot{token}/getMyName"))
        .form(&[("", "")])
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await;
    let Ok(r) = resp else {
        return String::new();
    };
    let Ok(body) = r.json::<Value>().await else {
        return String::new();
    };
    if body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        body.get("result")
            .and_then(|r| r.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    }
}

/// Rename the claimed bot. Returns (ok, reason) — reason is "" on success.
/// Reads before writing: setMyName is rate-limited per BOT on a cooldown
/// measured in hours, so the common case must cost nothing.
pub async fn set_bot_name(name: &str) -> (bool, String) {
    let (token, _chat) = credentials();
    if token.is_empty() {
        return (false, tr("no bot claimed"));
    }
    let wanted: String = name.chars().take(64).collect();
    if get_bot_name().await == wanted {
        return (true, String::new()); // already right; spend nothing
    }
    let resp = HTTP
        .post(format!("{API}/bot{token}/setMyName"))
        .form(&[("name", wanted.as_str())])
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await;
    let r = match resp {
        Ok(r) => r,
        // without_url(): this request's URL carries the bot token as a path
        // segment, and reqwest's Display appends it. See the note at the
        // send-with-retry error arm.
        Err(e) => return (false, e.without_url().to_string()),
    };
    let status = r.status().as_u16();
    let body: Value = r.json().await.unwrap_or(Value::Null);
    if body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        return (true, String::new());
    }
    if status == 429 {
        let wait = body
            .get("parameters")
            .and_then(|p| p.get("retry_after"))
            .and_then(|v| v.as_f64());
        if let Some(wait) = wait {
            let hours = (wait / 3600.0).max(1.0);
            return (
                false,
                trf(
                    "Telegram limits how often a bot may be renamed; this one can be renamed again in about {hours:.0f}h.",
                    &[("hours", FmtArg::Str(format!("{hours:.0}")))],
                ),
            );
        }
        return (false, tr("renamed too often recently"));
    }
    let desc = body
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("HTTP {status}"));
    (false, desc)
}

/// Bound on stale-lock reclaim attempts (a race between two sessions both
/// finding the same dead holder must stay finite). Only the non-unix path
/// reclaims by PID; unix takes a real flock, which the kernel reclaims for us
/// when a holder dies.
#[cfg(not(unix))]
const LOCK_RECLAIM_TRIES: usize = 3;

pub struct TelegramAdapter {
    pub timeout: f64,
}

impl Default for TelegramAdapter {
    fn default() -> Self {
        TelegramAdapter { timeout: 10.0 }
    }
}

impl TelegramAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(&self) -> &'static str {
        "telegram"
    }

    /// INTERNAL, not SECRET — see module docstring.
    pub fn max_sensitivity(&self) -> Sensitivity {
        Sensitivity::Internal
    }

    pub fn is_configured(&self) -> bool {
        let (token, chat) = credentials();
        !token.is_empty() && !chat.is_empty()
    }

    // --- rendering -----------------------------------------------------------

    pub fn format(&self, event: &Event) -> String {
        let mut parts = vec![format!("{} <b>{}</b>", icon(event.kind), html_escape(&event.title))];

        // Skip the directory line when the title already names it.
        if let Some(cwd) = &event.cwd {
            let where_ = crate::remote::inbox::short_label(cwd);
            if !where_.is_empty() && !event.title.to_lowercase().contains(&where_.to_lowercase()) {
                parts.push(format!("<code>{}</code>", html_escape(&where_)));
            }
        }

        if !event.body.is_empty() {
            let mut body = event.body.clone();
            let nchars = body.chars().count();
            if nchars > MAX_TEXT {
                // Say so rather than silently cutting.
                let cut: String = body.chars().take(MAX_TEXT).collect();
                body = format!("{cut}\n… (+{} chars truncated)", nchars - MAX_TEXT);
            }
            parts.push(String::new());
            let mut rendered = md_to_html(&body);
            // Box after the conversion, never in the body.
            if collapse(event, &body) {
                rendered = format!("<blockquote>{rendered}</blockquote>");
            }
            parts.push(rendered);
        }

        // The collapsed-tool-list equivalent: what ran, how often, nothing more.
        if let Some(Value::Object(tools)) = event.meta.get("tools") {
            if !tools.is_empty() {
                let mut entries: Vec<(String, i64)> = tools
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(0)))
                    .collect();
                entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let summary = entries
                    .iter()
                    .map(|(name, n)| {
                        if *n <= 1 {
                            html_escape(name)
                        } else {
                            format!("{} ×{n}", html_escape(name))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" · ");
                parts.push(String::new());
                parts.push(summary);
            }
        }

        if let Some(tool) = event.meta.get("tool").and_then(|v| v.as_str()) {
            parts.push(format!("tool: <code>{}</code>", html_escape(tool)));
        }
        // Context fill and images ride on the same line as the duration.
        let status = event.meta.get("status").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
        let dur = event.meta.get("duration_ms").and_then(|v| v.as_f64()).filter(|d| *d > 0.0);
        match (dur, status) {
            (Some(d), Some(st)) => parts.push(format!("{:.1}s · {}", d / 1000.0, html_escape(st))),
            (Some(d), None) => parts.push(format!("{:.1}s", d / 1000.0)),
            (None, Some(st)) => parts.push(html_escape(st)),
            (None, None) => {}
        }

        parts.join("\n")
    }

    // --- transport -----------------------------------------------------------

    /// `http_timeout` overrides the socket deadline for this call
    /// (long-polling needs it — a deadline shorter than the poll's own
    /// `timeout` aborts the read every time).
    pub async fn post(
        &self,
        method: &str,
        payload: &[(&str, String)],
        http_timeout: Option<f64>,
    ) -> Option<Value> {
        let (token, _) = credentials();
        if token.is_empty() {
            return None;
        }
        // Telegram throttles a single chat to roughly one message a second; a
        // 429 comes with the exact wait, and honouring it costs a pause.
        for attempt in 0..=RATE_LIMIT_RETRIES {
            let req = HTTP
                .post(format!("{API}/bot{token}/{method}"))
                .form(payload)
                .timeout(std::time::Duration::from_secs_f64(http_timeout.unwrap_or(self.timeout)));
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if (200..300).contains(&status) {
                        return resp.json::<Value>().await.ok();
                    }
                    let body: Value = resp.json().await.unwrap_or(Value::Null);
                    if status == 429 && attempt < RATE_LIMIT_RETRIES {
                        let wait = body
                            .get("parameters")
                            .and_then(|p| p.get("retry_after"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or_else(|| (2u64.pow(attempt as u32)) as f64);
                        tokio::time::sleep(std::time::Duration::from_secs_f64(
                            wait.min(RATE_LIMIT_MAX_WAIT),
                        ))
                        .await;
                        continue;
                    }
                    let desc = body
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    return Some(serde_json::json!({
                        "ok": false,
                        "error": format!("HTTP {status} {desc}").trim(),
                    }));
                }
                Err(e) => {
                    // without_url() BEFORE stringifying. The bot token is a path
                    // segment of the request URL, and reqwest's Display appends
                    // " for url ({url})" (error.rs:268) — so one DNS blip or
                    // connection reset put the whole credential into
                    // ~/.codemason/logs/debug.log, which is created with no mode
                    // and lands 0644. Anyone who reads it can steal the poll
                    // with getUpdates, silently killing the user's remote
                    // channel, and read every permission prompt: each one
                    // carries a full bash command line or a file-write preview.
                    //
                    // Port-only: the Python original uses urllib, whose
                    // URLError and HTTPError Display forms render no URL, so
                    // this leak arrived with the HTTP library swap. The rest of
                    // this module is careful about the token — token_tag()
                    // hashes it for filenames precisely so it never appears in
                    // a path — and this was the one hole.
                    return Some(
                        serde_json::json!({"ok": false, "error": error_chain(e)}),
                    );
                }
            }
        }
        Some(serde_json::json!({"ok": false, "error": "rate limited"}))
    }

    /// Post text as HTML, falling back to unformatted on a parse rejection —
    /// losing the alert is the one outcome worth avoiding; ugly is fine.
    async fn send_text(&self, chat: &str, text: &str, silent: bool, plain_fallback: &str) -> Option<Value> {
        let resp = self
            .post(
                "sendMessage",
                &[
                    ("chat_id", chat.to_string()),
                    ("text", text.to_string()),
                    ("parse_mode", "HTML".to_string()),
                    ("disable_web_page_preview", "true".to_string()),
                    ("disable_notification", if silent { "true" } else { "false" }.to_string()),
                ],
                None,
            )
            .await;
        if let Some(r) = &resp {
            if r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
                return resp;
            }
            let err = r.get("error").and_then(|e| e.as_str()).unwrap_or("").to_lowercase();
            if err.contains("parse") {
                crate::utils::log::debug_log(
                    "REMOTE_HTML_REJECTED",
                    Some(&serde_json::json!({"error": r.get("error")})),
                );
                let fallback = if plain_fallback.is_empty() {
                    RE_TAG.replace_all(text, "").into_owned()
                } else {
                    plain_fallback.to_string()
                };
                return self
                    .post(
                        "sendMessage",
                        &[
                            ("chat_id", chat.to_string()),
                            ("text", fallback),
                            ("disable_web_page_preview", "true".to_string()),
                            ("disable_notification", if silent { "true" } else { "false" }.to_string()),
                        ],
                        None,
                    )
                    .await;
            }
        }
        resp
    }

    pub async fn send(&self, event: &Event) -> Option<String> {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return None;
        }
        let split = self.split_for_file(event);
        let event_ref: &Event = split.as_ref().map(|(e, _, _)| e).unwrap_or(event);
        let resp = self
            .send_text(&chat, &self.format(event_ref), !is_loud(event_ref.kind), "")
            .await;
        let ok = resp
            .as_ref()
            .map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        if !ok {
            crate::utils::log::debug_log(
                "REMOTE_NOTIFY_FAILED",
                Some(&serde_json::json!({
                    "adapter": self.name(),
                    "error": resp.as_ref().and_then(|r| r.get("error")).cloned()
                        .unwrap_or(Value::String("no response".into())),
                })),
            );
            return None;
        }
        let mid = resp
            .as_ref()
            .and_then(|r| r.get("result"))
            .and_then(|r| r.get("message_id"))
            .and_then(|m| m.as_i64());
        note_chat_message(mid);
        if let Some((_, full, name)) = &split {
            self.send_document(full, name, "").await;
        }
        // Remember who posted it, so replying to this alert routes back here.
        if let Some(mid) = mid {
            let sid = event.session_id.clone().unwrap_or_else(session_id);
            crate::remote::inbox::claim_message(mid, &sid);
        }
        mid.map(|m| m.to_string())
    }

    /// True while nothing else has been posted below the progress message.
    pub fn progress_is_last(&self, message_id: i64) -> bool {
        let newest = newest_chat_message();
        newest == 0 || newest < message_id
    }

    pub async fn delete_message(&self, message_id: i64) -> bool {
        let resp = self
            .post(
                "deleteMessage",
                &[
                    ("chat_id", credentials().1),
                    ("message_id", message_id.to_string()),
                ],
                None,
            )
            .await;
        resp.map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)).unwrap_or(false)
    }

    /// Create or update the single in-progress message for a turn. Plain text,
    /// not HTML (partial output can end mid-tag). Deletes and reposts at the
    /// bottom when something has landed below it.
    pub async fn edit_progress(&self, message_id: Option<i64>, body: &str, cwd: &str) -> Option<i64> {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return None;
        }
        let where_ = if cwd.is_empty() {
            String::new()
        } else {
            format!("{}\n", crate::remote::inbox::short_label(cwd))
        };
        let text = format!("{where_}{body}");
        let mut message_id = message_id;
        if let Some(mid) = message_id {
            if !self.progress_is_last(mid) {
                self.delete_message(mid).await;
                message_id = None; // repost below whatever arrived
            }
        }
        let resp = match message_id {
            None => {
                self.post(
                    "sendMessage",
                    &[
                        ("chat_id", chat),
                        ("text", text),
                        ("disable_notification", "true".to_string()), // motion, not an alert
                        ("disable_web_page_preview", "true".to_string()),
                    ],
                    None,
                )
                .await
            }
            Some(mid) => {
                self.post(
                    "editMessageText",
                    &[
                        ("chat_id", chat),
                        ("message_id", mid.to_string()),
                        ("text", text),
                        ("disable_web_page_preview", "true".to_string()),
                    ],
                    None,
                )
                .await
            }
        };
        let ok = resp
            .as_ref()
            .map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        if !ok {
            return message_id;
        }
        resp.and_then(|r| {
            r.get("result").and_then(|res| res.get("message_id")).and_then(|m| m.as_i64())
        })
        .or(message_id)
    }

    /// Fetch an image the chat sent; Ok(path) or Err(reason).
    ///
    /// getFile resolves the id to a path on Telegram's file server, valid for
    /// an hour; the bytes then come from /file/bot<token>/<path>. Saved under
    /// ~/.codemason/remote/attachments (owner-only, like the rest of remote/)
    /// by a short content hash (`attachment_path`), so the same photo twice is one
/// file. The session that
    /// receives the message Reads it from there, and that Read is what puts a
    /// copy in the session's own image store. Bot API downloads stop at 20 MB.
    /// Errors go through without_url(), like post(): the token is in the URL.
    pub async fn download_image(&self, file_id: &str, name_hint: &str) -> Result<String, String> {
        use sha2::{Digest, Sha256};
        let (token, _) = credentials();
        if token.is_empty() {
            return Err("no bot token".into());
        }
        let resp = retry_read(|| {
            Box::pin(async {
                self.post("getFile", &[("file_id", file_id.to_string())], None)
                    .await
                    .unwrap_or(Value::Null)
            })
        })
        .await;
        let remote = resp
            .get("result")
            .and_then(|r| r.get("file_path"))
            .and_then(|p| p.as_str())
            .filter(|_| resp.get("ok").and_then(|o| o.as_bool()) == Some(true));
        let Some(remote) = remote else {
            let why = resp
                .get("error")
                .or_else(|| resp.get("description"))
                .and_then(|e| e.as_str())
                .unwrap_or("getFile failed");
            crate::utils::log::log_error(
                &format!("telegram getFile failed: {why}"),
                Some("remote: download_image"),
            );
            return Err(why.to_string());
        };
        const PATH: &percent_encoding::AsciiSet =
            &percent_encoding::CONTROLS.add(b' ').add(b'"').add(b'#').add(b'?').add(b'%');
        let data = get_bytes_retrying(
            &format!("{API}/file/bot{token}/{}", percent_encoding::utf8_percent_encode(remote, PATH)),
            self.timeout.max(60.0),
            MAX_DOWNLOAD,
        )
        .await
        .map_err(|e| {
            crate::utils::log::log_error(
                &format!("telegram photo download failed: {e}"),
                Some("remote: download_image"),
            );
            e
        })?;
        if data.len() > MAX_DOWNLOAD {
            return Err(format!("larger than {} MB", MAX_DOWNLOAD / (1024 * 1024)));
        }
        let ext = image_ext(name_hint, remote);
        ensure_remote_dir();
        let folder = attachments_dir();
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o700));
        }
        let path = attachment_path(&folder, &data, &ext);
        if !path.exists() {
            let tmp = path.with_extension(format!(
                "{}.{}.tmp",
                ext.trim_start_matches('.'),
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::write(&tmp, &data).map_err(|e| e.to_string())?;
            if let Err(e) = std::fs::rename(&tmp, &path) {
                let _ = std::fs::remove_file(&tmp);
                if !path.exists() {
                    return Err(e.to_string());
                }
            }
        }
        Ok(path.display().to_string())
    }

    /// Upload text as a file. Returns the message id, or None. BOM prepended
    /// so phone viewers do not guess a legacy codepage.
    pub async fn send_document(&self, content: &str, filename: &str, caption: &str) -> Option<i64> {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return None;
        }
        let mut data = vec![0xEFu8, 0xBB, 0xBF];
        data.extend_from_slice(content.as_bytes());
        let mut form = reqwest::multipart::Form::new().text("chat_id", chat);
        if !caption.is_empty() {
            // Telegram caps a caption at 1024 chars; the file carries the rest.
            let cap: String = caption.chars().take(1000).collect();
            form = form.text("caption", cap).text("parse_mode", "HTML");
        }
        let part = reqwest::multipart::Part::bytes(data)
            .file_name(filename.to_string())
            .mime_str("text/plain; charset=utf-8")
            .ok()?;
        form = form.part("document", part);
        let resp = HTTP
            .post(format!("{API}/bot{token}/sendDocument"))
            .multipart(form)
            .timeout(std::time::Duration::from_secs_f64(self.timeout.max(60.0)))
            .send()
            .await;
        let body: Value = match resp {
            Ok(r) => r.json().await.unwrap_or(Value::Null),
            Err(e) => {
                crate::utils::log::debug_log(
                    "REMOTE_DOCUMENT_FAILED",
                    // without_url(): the token is a path segment of this URL and
                    // debug.log is world-readable.
                    Some(&serde_json::json!({"error": e.without_url().to_string()})),
                );
                return None;
            }
        };
        if !body.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            crate::utils::log::debug_log(
                "REMOTE_DOCUMENT_FAILED",
                Some(&serde_json::json!({"error": body.get("description")})),
            );
            return None;
        }
        let mid = body.get("result").and_then(|r| r.get("message_id")).and_then(|m| m.as_i64());
        note_chat_message(mid);
        mid
    }

    fn file_threshold(&self) -> i64 {
        crate::utils::config::get_config().remote_file_threshold
    }

    /// Long body -> (chat stand-in, full text, filename), else None.
    /// Deliberately does NOT upload (the caller may fall back and re-post).
    pub fn split_for_file(&self, event: &Event) -> Option<(Event, String, String)> {
        let limit = self.file_threshold();
        let body = event.body.clone();
        if limit <= 0 || body.chars().count() as i64 <= limit {
            return None;
        }
        let cfg = crate::utils::config::get_config();
        let mut ext = if cfg.remote_file_ext.is_empty() { ".md".to_string() } else { cfg.remote_file_ext };
        let excerpt = cfg.remote_excerpt;
        if !ext.starts_with('.') {
            ext = format!(".{ext}");
        }
        let name = format!("codemason-{}{ext}", chrono::Local::now().format("%Y%m%d-%H%M%S"));
        let head: String = body.chars().take(std::cmp::max(200, excerpt) as usize).collect();
        let head = head.trim_end().to_string();
        let mut meta = event.meta.clone();
        meta.insert("collapse".into(), Value::Bool(true));
        let stand_in = Event {
            kind: event.kind,
            title: event.title.clone(),
            body: format!(
                "{head}\n\n{}",
                trf(
                    "… the full {chars:,} characters were sent as `{name}`.",
                    &[
                        ("chars", FmtArg::Int(body.chars().count() as i64)),
                        ("name", FmtArg::Str(name.clone())),
                    ],
                )
            ),
            sensitivity: event.sensitivity,
            session_id: event.session_id.clone(),
            cwd: event.cwd.clone(),
            meta,
        };
        Some((stand_in, body, name))
    }

    /// Turn the progress message into the finished alert. From here on it is
    /// an ordinary message.
    pub async fn replace(&self, message_id: i64, event: &Event) -> bool {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return false;
        }
        let split = self.split_for_file(event);
        let event_ref: &Event = split.as_ref().map(|(e, _, _)| e).unwrap_or(event);
        let formatted = self.format(event_ref);
        let resp = self
            .post(
                "editMessageText",
                &[
                    ("chat_id", chat.clone()),
                    ("message_id", message_id.to_string()),
                    ("text", formatted.clone()),
                    ("parse_mode", "HTML".to_string()),
                    ("disable_web_page_preview", "true".to_string()),
                ],
                None,
            )
            .await;
        let ok = resp
            .map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        if ok {
            note_chat_message(Some(message_id));
            if let Some((_, full, name)) = &split {
                self.send_document(full, name, "").await;
            }
            return true;
        }
        // Formatting failed on the finished text — keep the content, drop the
        // markup, rather than leaving a stale "working…" as the last thing said.
        let resp = self
            .post(
                "editMessageText",
                &[
                    ("chat_id", chat),
                    ("message_id", message_id.to_string()),
                    ("text", RE_TAG.replace_all(&formatted, "").into_owned()),
                    ("disable_web_page_preview", "true".to_string()),
                ],
                None,
            )
            .await;
        let ok = resp
            .map(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .unwrap_or(false);
        if ok {
            note_chat_message(Some(message_id));
            if let Some((_, full, name)) = &split {
                self.send_document(full, name, "").await;
            }
            return true;
        }
        false
    }

    /// Put a reaction on a message the user sent — acknowledgement without a
    /// second chat line. Best-effort.
    pub async fn react(&self, message_id: i64, emoji: &str) {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() || message_id == 0 {
            return;
        }
        let _ = self
            .post(
                "setMessageReaction",
                &[
                    ("chat_id", chat),
                    ("message_id", message_id.to_string()),
                    (
                        "reaction",
                        serde_json::json!([{"type": "emoji", "emoji": emoji}]).to_string(),
                    ),
                ],
                None,
            )
            .await;
    }

    // --- session picker ------------------------------------------------------

    /// Ask which session a message is for. Returns the picker's message id.
    /// One button per session, stacked one per row.
    pub async fn pick(&self, title: &str, preview: &str, labels: &[String]) -> Option<i64> {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return None;
        }
        let keyboard: Vec<Vec<Value>> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| vec![serde_json::json!({"text": l, "callback_data": format!("r:{i}")})])
            .collect();
        let mut text = format!("🔀 <b>{}</b>", html_escape(title));
        if !preview.is_empty() {
            text.push_str(&format!("\n\n<pre>{}</pre>", html_escape(preview)));
        }
        let resp = self
            .post(
                "sendMessage",
                &[
                    ("chat_id", chat),
                    ("text", text),
                    ("parse_mode", "HTML".to_string()),
                    (
                        "reply_markup",
                        serde_json::json!({"inline_keyboard": keyboard}).to_string(),
                    ),
                ],
                None,
            )
            .await?;
        if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            return None;
        }
        let mid = resp.get("result").and_then(|r| r.get("message_id")).and_then(|m| m.as_i64());
        note_chat_message(mid);
        mid
    }

    /// Collapse the picker once it has been answered. An empty session id is
    /// the Discard choice — it must read differently from a route.
    pub async fn note_routed(&self, picker_id: i64, session_id: &str) {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return;
        }
        if session_id.is_empty() {
            let _ = self
                .post(
                    "editMessageText",
                    &[
                        ("chat_id", chat),
                        ("message_id", picker_id.to_string()),
                        ("text", format!("🗑 <b>{}</b>", tr("Discarded"))),
                        ("parse_mode", "HTML".to_string()),
                    ],
                    None,
                )
                .await;
            return;
        }
        let label = crate::remote::inbox::live_sessions()
            .into_iter()
            .find(|s| s.session_id == session_id)
            .map(|s| if s.label.is_empty() { session_id.chars().take(8).collect() } else { s.label })
            .unwrap_or_else(|| session_id.chars().take(8).collect());
        let _ = self
            .post(
                "editMessageText",
                &[
                    ("chat_id", chat),
                    ("message_id", picker_id.to_string()),
                    (
                        "text",
                        format!("🔀 <b>{}</b> -> <code>{}</code>", tr("Routed"), html_escape(&label)),
                    ),
                    ("parse_mode", "HTML".to_string()),
                ],
                None,
            )
            .await;
    }

    // --- poll lock -----------------------------------------------------------

    /// Remove the pre-token-scoping lock file once its holder is gone (checked
    /// once per process).
    fn retire_legacy_lock(&self) {
        if LEGACY_LOCK_CHECKED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let Ok(raw) = std::fs::read_to_string(&*LEGACY_LOCK_PATH) else {
            return;
        };
        let holder: i64 = raw.trim().parse().unwrap_or(0);
        if holder != 0 && crate::utils::platform::pid_alive(holder) {
            return;
        }
        let _ = std::fs::remove_file(&*LEGACY_LOCK_PATH);
    }

    /// Take the inbound poll lock, or None if someone else is polling.
    ///
    /// On unix this is a real flock held for as long as the returned File
    /// lives; the PID written into the body is a label for the standby message
    /// and for owns_poll_lock, never the exclusion mechanism.
    ///
    /// It used to be exclusive-create plus a PID body, and that admitted two
    /// pollers on the same bot — the 409-conflict condition this lock exists to
    /// prevent. The sequence: A opens, reads a dead PID, decides to reclaim;
    /// B does the same; B unlinks and create_news its own lock; A's unlink then
    /// deletes B's *live* lock and A create_news on top. Both now believe they
    /// hold it, both call getUpdates, and Telegram starts 409ing whichever
    /// asked second — which the poller reads as "a live rival" and answers by
    /// giving the bot away. Every step of that is a separate syscall with a
    /// window after it, so no amount of re-reading closes it; the decision has
    /// to be one atomic operation. flock is that operation, and it also deletes
    /// the whole stale-reclaim problem: the kernel drops a dead process's flock
    /// whether it exited, crashed or was killed, so there is nothing to
    /// heuristically reclaim and no pid-reuse hazard to get wrong.
    pub fn acquire_poll_lock(&self) -> Option<std::fs::File> {
        self.retire_legacy_lock();
        let path = lock_path();
        ensure_remote_dir();

        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            use std::os::unix::io::AsRawFd;
            // create, never create_new: the file persists across holders and is
            // only ever a carrier for the lock. It is deliberately NOT unlinked
            // on release — unlinking a flocked path is how you get two holders
            // (a waiter locks the orphaned inode while the next arrival creates
            // and locks a fresh one), so the name stays and the body is cleared
            // instead.
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .mode(0o600)
                .open(&path)
                .ok()?;
            // Non-blocking: this is called from await_lock, which does its own
            // paced retry and reports standby. Blocking here would swallow the
            // stop token.
            if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return None; // held by a live poller, or a fs that cannot lock
            }
            // Ours exclusively from here. Truncate first: a shorter pid written
            // over a longer one leaves the previous holder's trailing digits
            // and poll_lock_holder then names a process that never held it.
            let _ = f.set_len(0);
            let _ = f.write_all(std::process::id().to_string().as_bytes());
            let _ = f.flush();
            Some(f)
        }
        #[cfg(not(unix))]
        {
            // No flock here, so the PID file stays — but hardened. Windows does
            // most of the work for us: std opens the lock without
            // FILE_SHARE_DELETE, so remove_file on a file a LIVE holder still
            // has open simply fails, and the A-deletes-B's-lock step above
            // cannot happen while B is alive. What remains is the narrow window
            // where B has created its lock but not yet written its pid, so we
            // re-read immediately before unlinking and only reclaim a body that
            // has not changed since we judged it dead, then re-validate that the
            // lock we took still names us.
            for _attempt in 0..LOCK_RECLAIM_TRIES {
                match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                    Ok(mut f) => {
                        use std::io::Write;
                        let _ = f.write_all(std::process::id().to_string().as_bytes());
                        let _ = f.flush();
                        if !self.owns_poll_lock() {
                            // Somebody reclaimed and rewrote it under us. Not
                            // ours — drop the handle and stand by rather than
                            // poll alongside them.
                            drop(f);
                            return None;
                        }
                        return Some(f);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        let read_holder = || -> i64 {
                            std::fs::read_to_string(&path)
                                .ok()
                                .and_then(|s| s.trim().parse().ok())
                                .unwrap_or(0)
                        };
                        let holder = read_holder();
                        if holder != 0 && crate::utils::platform::pid_alive(holder) {
                            return None; // genuinely held
                        }
                        // The body must still say what it said when we decided,
                        // or somebody is mid-acquire and we would be deleting a
                        // live lock.
                        if read_holder() != holder {
                            return None;
                        }
                        if std::fs::remove_file(&path).is_err() {
                            return None;
                        }
                        continue; // reclaimed; try to take it ourselves
                    }
                    Err(_) => return None,
                }
            }
            None
        }
    }

    /// Public form of owns_poll_lock, for callers that must wait it out.
    pub fn holds_poll_lock(&self) -> bool {
        self.owns_poll_lock()
    }

    /// True when the lock file on disk still names THIS process.
    fn owns_poll_lock(&self) -> bool {
        std::fs::read_to_string(lock_path())
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|pid| pid == std::process::id())
            .unwrap_or(false)
    }

    /// Give the lock back — but only the part that is still ours
    /// (an unconditional unlink can delete the NEW holder's lock after a
    /// reclaim, which manufactures the two-poller failure the lock prevents).
    pub fn release_poll_lock(&self, fd: Option<std::fs::File>) {
        let Some(f) = fd else {
            return; // nothing was held
        };
        #[cfg(unix)]
        {
            // Clear the body while we still hold the flock, so nobody can
            // observe a stale pid, then drop — which is what actually releases
            // the lock. The file itself stays: see acquire_poll_lock for why
            // unlinking a flocked path re-creates the two-holder bug. An empty
            // body is what makes owns_poll_lock/holds_poll_lock go false again,
            // which permissions::yield_poll_lock waits on.
            if self.owns_poll_lock() {
                let _ = f.set_len(0);
            }
            drop(f);
        }
        #[cfg(not(unix))]
        {
            drop(f);
            if !self.owns_poll_lock() {
                return;
            }
            let _ = std::fs::remove_file(lock_path());
        }
    }

    /// True only for the chat we are configured to talk to, from a sender we
    /// accept. A stranger's message must never be treated as your input, nor
    /// their tap as approval.
    ///
    /// The chat id alone is not enough. If the configured chat is a GROUP the
    /// id check is a tautology — the bot posted the prompt there — and the
    /// sender was never examined anywhere, so every current and future member
    /// could approve tool use and inject input. The whole boundary rested on a
    /// deployment choice (use a personal chat) that nothing enforced.
    ///
    /// Default with no configuration: accept only a sender whose id equals the
    /// configured chat id, which Telegram makes true in a 1:1 chat — so the
    /// intended setup keeps working untouched and a group must say so via
    /// TELEGRAM_ALLOWED_USER_IDS (comma-separated ids, replaces this rule).
    pub fn is_trusted_chat(&self, msg: &Value, sender: Option<&Value>) -> bool {
        let (_, chat) = credentials();
        if chat.is_empty() {
            return false;
        }
        let id_of = |v: Option<&Value>| -> String {
            v.map(|v| match v {
                Value::Number(n) => n.to_string(),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default()
        };
        if id_of(msg.get("chat").and_then(|c| c.get("id"))) != chat {
            return false;
        }

        let from = sender.or_else(|| msg.get("from"));
        let sender_id = id_of(from.and_then(|f| f.get("id")));

        let allowed_raw = std::env::var("TELEGRAM_ALLOWED_USER_IDS").unwrap_or_default();
        let allowed: Vec<&str> =
            allowed_raw.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        if !allowed.is_empty() {
            return allowed.iter().any(|a| *a == sender_id);
        }
        if sender_id.is_empty() {
            // No sender on the update (channel post, service message). Nothing
            // to authenticate, so it is not input from the operator.
            return false;
        }
        sender_id == chat
    }

    pub fn read_offset(&self) -> i64 {
        if let Ok(raw) = std::fs::read_to_string(offset_path()) {
            if let Ok(v) = raw.trim().parse() {
                return v;
            }
        }
        // First run after the offset became token-scoped: read the legacy file
        // once so an upgrade does not replay a day of held updates.
        std::fs::read_to_string(&*LEGACY_OFFSET_PATH)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Written through a temp file and renamed (atomic on Windows too); per-pid
    /// temp name so two sessions writing at once cannot share one.
    pub fn write_offset(&self, value: i64) {
        let path = offset_path();
        let tmp = path.with_file_name(format!(
            "{}.{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        ensure_remote_dir();
        if std::fs::write(&tmp, value.to_string()).is_ok() {
            if std::fs::rename(&tmp, &path).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// Post `event` with one button per option and wait for a tap.
    ///
    /// Returns the chosen option, or None on cancel / timeout / not configured.
    /// None always means "nobody answered here". The buttons stay live for the
    /// whole timeout and are collapsed only once the question is over.
    pub async fn ask(
        &self,
        event: &Event,
        options: &[String],
        timeout: f64,
        cancel: Option<tokio_util::sync::CancellationToken>,
    ) -> Option<String> {
        let (token, chat) = credentials();
        if token.is_empty() || chat.is_empty() {
            return None;
        }

        // Two short options (a grant: Allow/Deny) sit side by side; anything
        // else goes one per row (a question's options are sentences).
        let side_by_side = options.len() == 2 && options.iter().all(|o| o.chars().count() <= 12);
        let keyboard: Vec<Vec<Value>> = if side_by_side {
            vec![options
                .iter()
                .enumerate()
                .map(|(i, o)| serde_json::json!({"text": o, "callback_data": format!("p:{i}")}))
                .collect()]
        } else {
            options
                .iter()
                .enumerate()
                .map(|(i, o)| vec![serde_json::json!({"text": o, "callback_data": format!("p:{i}")})])
                .collect()
        };
        // A question also takes an answer of your own, as the terminal
        // picker's "Other" does. The button only explains how (a button cannot
        // carry a sentence); the typed message is the answer. Never on a
        // permission prompt — a sentence there must not become a grant.
        let free_text = event.kind == crate::remote::events::EventKind::QuestionAsked;
        let mut keyboard = keyboard;
        if free_text {
            keyboard.push(vec![serde_json::json!({
                "text": tr("✏️ Other (type your answer)"),
                "callback_data": crate::remote::poller::OTHER_DATA,
            })]);
        }
        let sent = self
            .post(
                "sendMessage",
                &[
                    ("chat_id", chat.clone()),
                    ("text", self.format(event)),
                    ("parse_mode", "HTML".to_string()),
                    ("disable_web_page_preview", "true".to_string()),
                    (
                        "reply_markup",
                        serde_json::json!({"inline_keyboard": keyboard}).to_string(),
                    ),
                ],
                None,
            )
            .await?;
        if !sent.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            return None;
        }
        let msg_id = sent.get("result").and_then(|r| r.get("message_id")).and_then(|m| m.as_i64())?;
        note_chat_message(Some(msg_id));

        let poller = crate::remote::poller::get();
        let question = poller.add_question(msg_id, options.to_vec(), free_text);
        // Publish who is waiting on this prompt — the tap may land at another
        // process holding the poll lock, which forwards it by this claim.
        crate::remote::inbox::claim_question_with_options(
            msg_id,
            &session_id(),
            options.len(),
            free_text,
        );
        // Everything acquired above is released by a guard, not by the three
        // statements that used to follow the .await. This future is dropped on
        // ordinary paths — the terminal winning the permission race, and every
        // AskUserQuestion escalation boundary — and none of that cleanup ran
        // then. poller.acquire() increments a user count that release() must
        // bring back to zero; left high, the getUpdates loop never stops and
        // this process holds the shared poll lock until it exits, so any other
        // session on the same bot waits out the relock ceiling on every round,
        // forever. The question registration and the inbox claim leaked with it.
        struct WaitGuard {
            poller: std::sync::Arc<crate::remote::poller::Poller>,
            msg_id: i64,
        }
        impl Drop for WaitGuard {
            fn drop(&mut self) {
                self.poller.drop_question(self.msg_id);
                crate::remote::inbox::release_question(self.msg_id);
                self.poller.release();
            }
        }

        poller.acquire();
        let _wait_guard = WaitGuard { poller: poller.clone(), msg_id };
        let chosen = question.wait(timeout, cancel.clone()).await;

        let cancelled =
            chosen.is_none() && cancel.map(|c| c.is_cancelled()).unwrap_or(false);
        let outcome = match &chosen {
            Some(c) => format!("\n\n→ <b>{}</b>", html_escape(c)),
            None if cancelled => format!("\n\n-> <i>{}</i>", tr("answered at the terminal")),
            // Not "expired" — the turn is still blocked and a fresh prompt is
            // on its way.
            None => format!("\n\n-> <i>{}</i>", tr("waiting… (a fresh prompt is on its way)")),
        };
        let _ = self
            .post(
                "editMessageText",
                &[
                    ("chat_id", chat),
                    ("message_id", msg_id.to_string()),
                    ("text", format!("{}{outcome}", self.format(event))),
                    ("parse_mode", "HTML".to_string()),
                ],
                None,
            )
            .await;
        chosen
    }
}

#[cfg(test)]
mod download_retry_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A read-only call goes out again after a transport failure, and stops at
    /// a 4xx: Telegram answered, and the answer will not change. The pauses are
    /// real (tokio's test-util clock is not a dependency here), so the whole
    /// test costs the two it waits out.
    #[tokio::test]
    async fn retry_read_repeats_a_transport_failure_not_a_4xx() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let script = |answers: Vec<Value>| async move {
            let calls = AtomicUsize::new(0);
            let got = retry_read(|| {
                let n = calls.fetch_add(1, Ordering::SeqCst);
                let answer = answers.get(n).cloned().unwrap_or(Value::Null);
                Box::pin(async move { answer })
            })
            .await;
            (got, calls.load(Ordering::SeqCst))
        };
        let dead = serde_json::json!({"ok": false, "error": "error sending request: connection reset"});
        let ok = serde_json::json!({"ok": true, "result": {"file_path": "photos/f.jpg"}});
        let (got, calls) = script(vec![dead.clone(), ok.clone()]).await;
        assert_eq!(got, ok);
        assert_eq!(calls, 2);
        let gone = serde_json::json!({"ok": false, "error": "HTTP 400 Bad Request: file not found"});
        let (got, calls) = script(vec![gone.clone(), ok.clone()]).await;
        assert_eq!(got, gone, "a 4xx is the answer, not a reason to ask again");
        assert_eq!(calls, 1);
        let (_, calls) = script(vec![dead.clone(), dead.clone(), dead]).await;
        assert_eq!(calls, DOWNLOAD_ATTEMPTS, "and it does not try forever");
    }

    /// First connection dropped without a response (a dead keep-alive or a
    /// failed route looks like this), second answered: the download succeeds.
    /// A 404 is not retried, and the reason survives in the error.
    #[tokio::test]
    async fn retries_a_dropped_connection_not_a_404() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            drop(sock);
            loop {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let resp: &[u8] = if req.starts_with("GET /missing") {
                    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                } else {
                    b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\nconnection: close\r\n\r\nhello"
                };
                let _ = sock.write_all(resp).await;
            }
        });
        let got = get_bytes_retrying(&format!("http://{addr}/file"), 5.0, 100).await;
        assert_eq!(got.as_deref(), Ok(&b"hello"[..]));
        let missing = get_bytes_retrying(&format!("http://{addr}/missing"), 5.0, 100).await;
        assert_eq!(missing, Err("HTTP 404".to_string()));
        // Over the cap: cap + 1 bytes read, not the whole body.
        let capped = get_bytes_retrying(&format!("http://{addr}/file"), 5.0, 3).await.unwrap();
        assert_eq!(capped, b"hell");
    }

    /// 429 means come back later, unlike the other 4xx.
    #[tokio::test]
    async fn retries_429() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for i in 0.. {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf).await;
                let resp: &[u8] = if i == 0 {
                    b"HTTP/1.1 429 Too Many Requests\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                } else {
                    b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok"
                };
                let _ = sock.write_all(resp).await;
            }
        });
        let got = get_bytes_retrying(&format!("http://{addr}/f"), 5.0, 100).await;
        assert_eq!(got.as_deref(), Ok(&b"ok"[..]));
    }

    #[test]
    fn extension_only_from_images() {
        assert_eq!(image_ext("a.PNG", "x"), ".png");
        assert_eq!(image_ext("evil.sh", "photos/f.webp"), ".webp");
        assert_eq!(image_ext("x.exe;rm", ""), ".jpg");
        assert_eq!(image_ext("photo.jpg", "photos/file_1.jpg"), ".jpg");
    }
}

#[cfg(test)]
mod status_line_tests {
    use super::*;

    /// The turn message carries the context/image line beside the duration
    /// (tests/test_session_status.py test_telegram_line).
    #[test]
    fn status_beside_duration() {
        let mut meta = HashMap::new();
        meta.insert("duration_ms".to_string(), serde_json::json!(3200));
        meta.insert("status".to_string(), serde_json::json!("Context 22% · images 6/8"));
        let ev = Event {
            kind: EventKind::TurnDone,
            title: "Turn done".into(),
            body: "hi".into(),
            sensitivity: Sensitivity::Public,
            session_id: None,
            cwd: None,
            meta,
        };
        let t = TelegramAdapter::new().format(&ev);
        assert!(t.trim_end().ends_with("3.2s · Context 22% · images 6/8"), "{t}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attachment_is_named_by_a_short_content_hash() {
        use sha2::{Digest, Sha256};
        let dir = std::env::temp_dir().join(format!("cm-tg-attach-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let data = b"\xff\xd8jpegbytes";
        let sha = hex::encode(Sha256::digest(data));

        let p = attachment_path(&dir, data, ".jpg");
        let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
        assert_eq!(stem.len(), ATTACHMENT_NAME_DIGITS, "{stem}");
        assert_eq!(stem.as_str(), &sha[..ATTACHMENT_NAME_DIGITS]);

        std::fs::write(&p, data).unwrap();
        assert_eq!(attachment_path(&dir, data, ".jpg"), p, "the same photo twice is one file");

        std::fs::write(&p, b"different bytes entirely").unwrap();
        let other = attachment_path(&dir, data, ".jpg");
        assert_eq!(other.file_stem().unwrap().to_str().unwrap(), sha,
                   "a short name other bytes own falls back to the full hash");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn env_file_pool_parsing() {
        let dir = std::env::temp_dir().join(format!("cm-tg-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("remote.env");
        std::fs::write(
            &p,
            "# comment\nTELEGRAM_BOT_TOKEN=aaa\nTELEGRAM_BOT_TOKEN=\nTELEGRAM_BOT_TOKEN='bbb'\n\nTELEGRAM_CHAT_ID=\"123\"\n",
        )
        .unwrap();
        let parsed = load_env_file(&p);
        assert_eq!(as_list(parsed.get("TELEGRAM_BOT_TOKEN")), vec!["aaa", "bbb"]);
        assert_eq!(as_list(parsed.get("TELEGRAM_CHAT_ID")), vec!["123"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn token_tag_is_hash_not_prefix() {
        let tag = token_tag("123456:ABC-secret");
        assert_eq!(tag.len(), 12);
        assert!(!"123456:ABC-secret".contains(&tag));
        assert_eq!(token_tag(""), "none");
    }

    #[test]
    fn md_tables_fit_and_reflow() {
        // Narrow table -> <pre> grid.
        let narrow = "|a|b|\n|-|-|\n|1|2|\n";
        let html = md_to_html(narrow);
        assert!(html.contains("<pre>"), "{html}");
        // Wide table -> reflowed blockquote.
        let wide = format!(
            "|name|description|\n|-|-|\n|{}|{}|\n|{}|{}|\n",
            "x".repeat(30),
            "y".repeat(30),
            "z".repeat(30),
            "w".repeat(30)
        );
        let html = md_to_html(&wide);
        assert!(html.contains("<blockquote>"), "{html}");
    }

    #[test]
    fn md_inline_and_escape() {
        let html = md_to_html("**bold** & `code` <tag> *it*");
        assert!(html.contains("<b>bold</b>"));
        assert!(html.contains("<code>code</code>"));
        assert!(html.contains("&amp;"));
        assert!(html.contains("&lt;tag&gt;"));
        assert!(html.contains("<i>it</i>"));
    }

    #[test]
    fn unterminated_fence_recovers() {
        let html = md_to_html("before\n```\ncode tail");
        assert!(html.contains("<pre><code>code tail</code></pre>"), "{html}");
    }

    #[test]
    fn scaffold_key_matches_ko_catalog() {
        // The scaffold body goes through tr(); its English key must byte-match
        // the ko catalog's key or the translation silently misses.
        let key = "# codemason remote channel — drive this session from Telegram.\n#\n# 1. In Telegram, talk to @BotFather and send /newbot. It replies with\n#    a token. Paste one per slot below. Blank slots are ignored.\n# 2. Every codemason you start claims one FREE bot by itself — no flag,\n#    no name to remember. Four slots means four windows at the same\n#    time, anywhere: this machine, a VM, another desktop. Want more at\n#    once? Add more lines.\n# 3. Send /start to each bot once, so it is allowed to message you.\n# 4. TELEGRAM_CHAT_ID is your own numeric id — ask @userinfobot for it.\n#\n# This file holds your bot tokens. Anyone who can read it can read your\n# alerts and type into your sessions, so keep it mode 600.\n";
        assert!(crate::locales::ko::MESSAGES.contains_key(key));
    }
}
