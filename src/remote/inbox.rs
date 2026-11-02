//! Per-session message inboxes.
//! Equivalent to codemason/remote/inbox.py — see that file's module doc for
//! the design (append-only JSONL + byte cursor, coarse flock over the whole
//! read-or-append, oldest-first capped drop with a visible marker, sweep as a
//! backstop for crashed runs, question/owner/answer/route side-stores).
//!
//! Disk formats are byte-compatible with the Python implementation (a mixed
//! Python/Rust machine shares these files): JSON payloads are written with
//! Python's default separators (", " / ": ") and ensure_ascii=False via
//! `py_json_dumps`.

use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub static ROOT: Lazy<PathBuf> =
    Lazy::new(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".codemason").join("remote"));
pub static INBOX_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("inbox"));
pub static SESSION_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("sessions"));

/// Messages with no identifiable target.
pub const UNROUTED: &str = "_unrouted";

/// An inbox nobody drains must not grow without bound.
pub const MAX_MESSAGES: usize = 200;
pub const MAX_BYTES: usize = 256 * 1024;

pub const RETENTION_DAYS_DEFAULT: f64 = 30.0;
pub const STALE_QUESTION: f64 = 3600.0;

/// How long content-bearing files are kept. Config: remote_retention_days.
pub fn retention_seconds() -> f64 {
    let days = crate::utils::config::get_config().remote_retention_days;
    days.max(0.0) * 24.0 * 3600.0
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Python json.dumps(..., ensure_ascii=False) with default separators
/// (", ", ": ") — used so files written here are byte-identical to the Python
/// implementation's.
pub fn py_json_dumps(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let inner: Vec<String> =
                m.iter().map(|(k, val)| format!("{}: {}", py_json_dumps(&Value::String(k.clone())), py_json_dumps(val))).collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Array(a) => {
            let inner: Vec<String> = a.iter().map(py_json_dumps).collect();
            format!("[{}]", inner.join(", "))
        }
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

/// One message from a remote client, on its way into a session.
#[derive(Debug, Clone, Default)]
pub struct InboundMessage {
    pub text: String,
    pub source: String,
    pub ts: f64,
    /// transport extras (message id, reply_to, sent_at…)
    pub meta: Value,
}

impl InboundMessage {
    pub fn new(text: impl Into<String>, source: impl Into<String>, meta: Value) -> Self {
        InboundMessage { text: text.into(), source: source.into(), ts: 0.0, meta }
    }

    pub fn to_json(&self) -> String {
        let mut obj = serde_json::Map::new();
        obj.insert("text".into(), Value::String(self.text.clone()));
        obj.insert("source".into(), Value::String(self.source.clone()));
        let ts = if self.ts != 0.0 { self.ts } else { now() };
        obj.insert(
            "ts".into(),
            serde_json::Number::from_f64(ts).map(Value::Number).unwrap_or(Value::Null),
        );
        obj.insert(
            "meta".into(),
            if self.meta.is_null() { Value::Object(Default::default()) } else { self.meta.clone() },
        );
        py_json_dumps(&Value::Object(obj))
    }

    pub fn from_obj(obj: &Value) -> InboundMessage {
        InboundMessage {
            text: obj.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            source: obj.get("source").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
            ts: obj.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0),
            meta: obj.get("meta").cloned().unwrap_or_else(|| Value::Object(Default::default())),
        }
    }
}

fn ensure_dirs() {
    let _ = std::fs::create_dir_all(&*INBOX_DIR);
    let _ = std::fs::create_dir_all(&*SESSION_DIR);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&*ROOT, std::fs::Permissions::from_mode(0o700));
    }
}

fn inbox_path(session_id: &str) -> PathBuf {
    INBOX_DIR.join(format!("{session_id}.jsonl"))
}

fn cursor_path(session_id: &str) -> PathBuf {
    INBOX_DIR.join(format!("{session_id}.cursor"))
}

/// The inbox could not be locked, so the caller must not touch it.
#[derive(Debug)]
pub struct LockUnavailable;

/// How long acquiring may be retried before giving up on this attempt.
/// Both platforms honour it. It used to be the Windows sidecar's deadline only,
/// because "flock waits" was treated as good enough — it is not: a bare
/// LOCK_EX waits in the kernel with no timeout and no way to notice a stop
/// token, and `Locked::open` is reached from the poll loop (poller::on_picker
/// -> inbox::deliver). One process holding an inbox lock — including a hung or
/// stopped one, since flock survives SIGSTOP — therefore parked the whole
/// inbound loop indefinitely, with nothing logged and nothing to see but
/// "Telegram stopped working". A bounded wait turns that into a visible,
/// recoverable LockUnavailable on both platforms.
const LOCK_RETRY_SECONDS: f64 = 10.0;
const LOCK_RETRY_SLEEP_MS: u64 = 50;

/// flock a file for the whole read-or-append. Always binary (the cursor is a
/// byte offset). The lock is coarse on purpose; failing to lock returns Err —
/// carrying on unlocked is the one option that loses data silently.
struct Locked {
    file: std::fs::File,
    #[allow(dead_code)]
    side: Option<std::fs::File>,
    side_path: Option<PathBuf>,
}

/// Does the lock sidecar exist with nobody holding it?
///
/// This is the whole reclaim decision, and getting it wrong in the permissive
/// direction is worse than not reclaiming at all: taking a lock from a live
/// owner reintroduces the two writers the lock exists to keep apart. So the
/// question is asked of the OS rather than inferred. An exclusive open
/// (share_mode 0) can only succeed when no other handle is open on the file,
/// and the holder keeps one open for as long as it holds the lock. A live lock
/// therefore answers false, every time, without any pid bookkeeping — and
/// without the pid-reuse hazard that bookkeeping would carry.
#[cfg(windows)]
fn sidecar_is_unheld(side_path: &std::path::Path) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).share_mode(0).open(side_path).is_ok()
}

impl Locked {
    fn open(path: &std::path::Path, append: bool) -> Result<Locked, LockUnavailable> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(false)
            .open(path)
            .map_err(|_| LockUnavailable)?;
        let _ = append; // seek-to-end is done by callers, as in 'ab+'
        #[cfg(unix)]
        {
            // LOCK_NB in a deadline loop rather than a blocking LOCK_EX. The
            // lock semantics on success are identical (same flock, same fd,
            // released the same way in Drop); what changes is the failure mode.
            // A blocking LOCK_EX is uninterruptible from Rust: the caller sits
            // in the kernel forever, so a peer that holds this inbox and never
            // lets go took the poll loop down with it and left no trace. Now it
            // gives up after LOCK_RETRY_SECONDS with the same LockUnavailable
            // the Windows path has always returned on timeout, and every caller
            // already treats that as "do not touch the inbox" rather than
            // "carry on unlocked".
            use std::os::unix::io::AsRawFd;
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_secs_f64(LOCK_RETRY_SECONDS);
            loop {
                let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                if rc == 0 {
                    return Ok(Locked { file, side: None, side_path: None });
                }
                match std::io::Error::last_os_error().raw_os_error() {
                    // Somebody else holds it. That is the one case worth
                    // waiting out.
                    Some(e) if e == libc::EWOULDBLOCK => {}
                    // A signal landed mid-call. It says nothing about the lock,
                    // so ask again at once instead of spending the budget.
                    Some(e) if e == libc::EINTR => continue,
                    // EBADF, or a filesystem with no flock at all. Retrying
                    // cannot change the answer — fail exactly as the old
                    // `rc != 0` arm did.
                    _ => return Err(LockUnavailable),
                }
                if std::time::Instant::now() >= deadline {
                    return Err(LockUnavailable);
                }
                std::thread::sleep(std::time::Duration::from_millis(LOCK_RETRY_SLEEP_MS));
            }
        }
        #[cfg(not(unix))]
        {
            // Windows sidecar: exclusive-create retry loop with the same 10s
            // deadline the Python msvcrt path used. Divergence (documented):
            // std exposes no LockFile binding, so the sidecar's EXISTENCE is
            // the lock rather than a byte-range lock on it.
            //
            // The sidecar is opened FILE_FLAG_DELETE_ON_CLOSE, which is what
            // makes that divergence safe. flock on unix is released by the
            // kernel when the holder dies; a plain file is not. Without this
            // flag a taskkill, a crash or a power cut left the sidecar behind
            // and every later acquire waited out the full deadline and then
            // failed. For a per-session inbox that only strands the dead
            // session, but `_unrouted` is shared and sweep() deliberately skips
            // it, so a lock stranded there stops all inbound remote messages
            // until somebody deletes the file by hand. With the flag the OS
            // closes the handle however the process ends and removes the name
            // with it. Measured: the file is gone the moment a hard kill lands,
            // and the next acquire succeeds.
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
            let side_path = PathBuf::from(format!("{}.winlock", path.display()));
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_secs_f64(LOCK_RETRY_SECONDS);
            let mut reclaim_tried = false;
            loop {
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .custom_flags(FILE_FLAG_DELETE_ON_CLOSE)
                    .open(&side_path)
                {
                    Ok(side) => {
                        return Ok(Locked { file, side: Some(side), side_path: Some(side_path) })
                    }
                    Err(_) => {
                        // A sidecar written by a build from before the flag, or
                        // on a filesystem that ignored it, has no handle behind
                        // it and would block forever. Ask the OS which it is:
                        // an exclusive open (share_mode 0) can only succeed when
                        // nobody holds the file. A live holder keeps it open, so
                        // this refuses and we go on waiting — the lock is never
                        // taken from someone still using it, which would be the
                        // two-writer bug this whole mechanism exists to prevent.
                        // Once, so a live holder that closes mid-loop is not
                        // raced for.
                        if !reclaim_tried {
                            reclaim_tried = true;
                            if sidecar_is_unheld(&side_path) {
                                let _ = std::fs::remove_file(&side_path);
                                continue;
                            }
                        }
                        if std::time::Instant::now() >= deadline {
                            return Err(LockUnavailable);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(LOCK_RETRY_SLEEP_MS));
                    }
                }
            }
        }
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
            }
        }
        // Closing the sidecar handle is what releases the lock: it is opened
        // FILE_FLAG_DELETE_ON_CLOSE, so the OS unlinks the name as the handle
        // goes away — on this path and equally on a path we never reach,
        // because a killed process closes its handles too. Take it explicitly
        // so the close happens here rather than after drop() returns, and only
        // then fall back to an unlink for a filesystem that ignored the flag.
        // On such a volume the name can linger delete-pending until the handle
        // closes, and a waiter's exclusive create is refused with ACCESS_DENIED
        // rather than finding the name free — measured NOT to happen here,
        // where std::fs::remove_file asks for POSIX semantics, but the order
        // costs one line and the acquire path's reclaim covers the rest.
        //
        // Python has no counterpart: its Windows lock is a byte-range lock on a
        // sidecar it deliberately never unlinks ("deleting a lock another
        // process may be holding is how you get two writers"), so both the
        // ordering and the stale-file question exist only in this port's
        // divergent design.
        let side = self.side.take();
        drop(side);
        if let Some(p) = &self.side_path {
            let _ = std::fs::remove_file(p);
        }
    }
}

// --- session registry --------------------------------------------------------

/// A human-sized name for a working directory: home is `~`, one level under it
/// keeps the parent for context, anything deeper is just its own name; outside
/// home the anchor IS the name for a bare root.
pub fn short_label(cwd: &str) -> String {
    let cwd_path = if cwd.is_empty() {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    } else {
        PathBuf::from(cwd)
    };
    let p = cwd_path.canonicalize().unwrap_or(cwd_path.clone());
    if let Some(home) = dirs::home_dir().and_then(|h| h.canonicalize().ok()) {
        if p == home {
            return "~".to_string();
        }
        if let Ok(rel) = p.strip_prefix(&home) {
            let parts: Vec<_> = rel.components().collect();
            if parts.len() == 1 {
                return format!("~/{}", rel.to_string_lossy());
            }
            if let Some(last) = parts.last() {
                return last.as_os_str().to_string_lossy().into_owned();
            }
        }
    }
    // Outside home: a root has no basename — the anchor is the name there.
    let name = cwd_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !name.is_empty() {
        return name;
    }
    let anchor = cwd_path.to_string_lossy().into_owned();
    if anchor.is_empty() {
        "?".to_string()
    } else {
        anchor
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionEntry {
    pub session_id: String,
    pub pid: i64,
    pub cwd: String,
    pub label: String,
    pub started: f64,
}

/// Announce a live session so the poller can route to it.
///
/// Also exports the id to this process's environment, and so to every child
/// it spawns: a watcher script the session writes can then address the
/// session it belongs to without being told which one that is. The variable
/// exists exactly when a wake can land — registration is also what makes the
/// inbox drain — so a script can test it rather than guess.
pub fn register(session_id: &str, cwd: &str, label: &str) {
    ensure_dirs();
    crate::utils::wake_cli::export_self(session_id);
    let path = SESSION_DIR.join(format!("{session_id}.json"));
    let cwd_val = if cwd.is_empty() {
        std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
    } else {
        cwd.to_string()
    };
    let label_val = if label.is_empty() {
        distinct_label(&short_label(cwd), session_id)
    } else {
        label.to_string()
    };
    let mut obj = serde_json::Map::new();
    obj.insert("session_id".into(), Value::String(session_id.to_string()));
    obj.insert("pid".into(), Value::Number((std::process::id() as i64).into()));
    obj.insert("cwd".into(), Value::String(cwd_val));
    obj.insert("label".into(), Value::String(label_val));
    obj.insert(
        "started".into(),
        serde_json::Number::from_f64(now()).map(Value::Number).unwrap_or(Value::Null),
    );
    let tmp = SESSION_DIR.join(format!("{session_id}.json.tmp"));
    if std::fs::write(&tmp, py_json_dumps(&Value::Object(obj))).is_ok() {
        let _ = std::fs::rename(&tmp, &path); // atomic — no partial entry
    }
}

pub fn unregister(session_id: &str) {
    if std::env::var(crate::utils::wake_cli::SESSION_ENV).as_deref() == Ok(session_id) {
        crate::utils::wake_cli::unexport_self();
    }
    for p in [
        SESSION_DIR.join(format!("{session_id}.json")),
        inbox_path(session_id),
        cursor_path(session_id),
    ] {
        let _ = std::fs::remove_file(p);
    }
}

/// Stop advertising the session but keep whatever is still in its inbox
/// (`sweep` reclaims it later once the process is provably gone).
pub fn unregister_registry_only(session_id: &str) {
    if std::env::var(crate::utils::wake_cli::SESSION_ENV).as_deref() == Ok(session_id) {
        crate::utils::wake_cli::unexport_self();
    }
    let _ = std::fs::remove_file(SESSION_DIR.join(format!("{session_id}.json")));
}

fn alive(pid: i64) -> bool {
    crate::utils::platform::pid_alive(pid)
}

/// Registered sessions whose process is still running, newest first. Duplicate
/// labels get a session-id-slice suffix as a backstop (see distinct_label).
pub fn live_sessions() -> Vec<SessionEntry> {
    ensure_dirs();
    let mut out: Vec<SessionEntry> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&*SESSION_DIR) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e != "json").unwrap_or(true) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&text) else { continue };
            let e = SessionEntry {
                session_id: v.get("session_id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                pid: v.get("pid").and_then(|x| x.as_i64()).unwrap_or(-1),
                cwd: v.get("cwd").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                label: v.get("label").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                started: v.get("started").and_then(|x| x.as_f64()).unwrap_or(0.0),
            };
            if alive(e.pid) {
                out.push(e);
            }
        }
    }
    out.sort_by(|a, b| b.started.partial_cmp(&a.started).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: HashMap<String, usize> = HashMap::new();
    for e in &out {
        *seen.entry(e.label.clone()).or_insert(0) += 1;
    }
    for e in out.iter_mut() {
        if seen.get(&e.label).copied().unwrap_or(0) > 1 {
            let slice: String = e.session_id.chars().take(6).collect();
            e.label = format!("{} ({slice})", e.label);
        }
    }
    out
}

/// `base`, or the first free `base (2)`, `base (3)`, … among live sessions.
/// Settled once, at registration.
pub fn distinct_label(base: &str, session_id: &str) -> String {
    let taken: std::collections::HashSet<String> = live_sessions()
        .into_iter()
        .filter(|s| s.session_id != session_id)
        .map(|s| s.label)
        .collect();
    if !taken.contains(base) {
        return base.to_string();
    }
    let mut n = 2;
    while taken.contains(&format!("{base} ({n})")) {
        n += 1;
    }
    format!("{base} ({n})")
}

pub static QUESTION_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("questions"));
pub static OWNER_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("owners"));
pub static ANSWER_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("answers"));
pub static ROUTE_DIR: Lazy<PathBuf> = Lazy::new(|| ROOT.join("routes"));

/// Reclaim everything left behind by sessions that are gone. A backstop for
/// crashed runs, not tidying; `force=true` ignores the window (/remote clean).
pub fn sweep(stale_after: Option<f64>, force: bool) -> usize {
    ensure_dirs();
    let mut stale_after = stale_after.unwrap_or_else(retention_seconds);
    if force {
        stale_after = 0.0;
    }
    let mut removed = 0usize;
    let now_ts = now();
    let live: std::collections::HashSet<String> =
        live_sessions().into_iter().map(|s| s.session_id).collect();

    // 1. Registry entries whose process is gone.
    if let Ok(rd) = std::fs::read_dir(&*SESSION_DIR) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e != "json").unwrap_or(true) {
                continue;
            }
            let parsed: Value = std::fs::read_to_string(&p)
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or(Value::Null);
            let pid = parsed.get("pid").and_then(|v| v.as_i64()).unwrap_or(-1);
            if alive(pid) {
                continue;
            }
            let started = parsed.get("started").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let mtime = p
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            if now_ts - started.max(mtime) < stale_after {
                continue;
            }
            let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            unregister(&stem);
            removed += 1;
        }
    }

    // 2. Inboxes with no live owner — including ones a clean shutdown left
    //    behind. `.lock`/`.winlock` sidecars retire here with the inbox.
    if let Ok(rd) = std::fs::read_dir(&*INBOX_DIR) {
        for entry in rd.flatten() {
            let p = entry.path();
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let is_ours = name.ends_with(".jsonl") || name.ends_with(".cursor") || name.ends_with(".lock") || name.ends_with(".winlock");
            if !is_ours {
                continue;
            }
            let sid = name.split('.').next().unwrap_or("").to_string();
            if live.contains(&sid) || sid == UNROUTED {
                continue;
            }
            let mtime = p
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(f64::MAX);
            if now_ts - mtime < stale_after {
                continue;
            }
            if std::fs::remove_file(&p).is_ok() {
                removed += 1;
            }
        }
    }

    // 3. In-flight coordination (question claims, forwarded answers): consumed
    //    within one prompt round; an hour is generous.
    for d in [&*QUESTION_DIR, &*ANSWER_DIR] {
        if !d.exists() {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            for entry in rd.flatten() {
                let p = entry.path();
                let mtime = p
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(f64::MAX);
                if !force && now_ts - mtime < STALE_QUESTION {
                    continue;
                }
                if std::fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
        }
    }

    // 4. Content and routing, on the long window (parked messages hold the
    //    user's own words; message-owner records are useful while the alert is
    //    visible, far longer than an hour).
    for d in [&*ROUTE_DIR, &*OWNER_DIR] {
        if !d.exists() {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(d) {
            for entry in rd.flatten() {
                let p = entry.path();
                let mtime = p
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(f64::MAX);
                if now_ts - mtime < stale_after {
                    continue;
                }
                if std::fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
        }
    }

    removed
}

// --- delivery ----------------------------------------------------------------

/// File one message for `session_id`. False when the inbox could not be locked
/// and the message was NOT filed — never writes unlocked as a fallback.
pub fn deliver(session_id: &str, message: &InboundMessage) -> bool {
    ensure_dirs();
    let path = inbox_path(session_id);
    let Ok(locked) = Locked::open(&path, true) else {
        return false;
    };
    let mut fh = &locked.file;
    if fh.seek(SeekFrom::End(0)).is_err() {
        return false;
    }
    let line = format!("{}\n", message.to_json());
    if fh.write_all(line.as_bytes()).is_err() {
        return false;
    }
    let _ = fh.flush();
    enforce_caps(&locked.file, session_id);
    true
}

/// Trim an undrained inbox, oldest first, and say so — dropping the newest or
/// dropping silently would both be worse.
fn enforce_caps(mut fh: &std::fs::File, session_id: &str) {
    if fh.seek(SeekFrom::Start(0)).is_err() {
        return;
    }
    let mut blob = Vec::new();
    if fh.read_to_end(&mut blob).is_err() {
        return;
    }
    let mut lines: Vec<Vec<u8>> = blob.split(|b| *b == b'\n').map(|s| s.to_vec()).collect();
    if lines.last().map(|l| l.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    let mut size: usize = lines.iter().map(|l| l.len() + 1).sum();
    if lines.len() <= MAX_MESSAGES && size <= MAX_BYTES {
        return;
    }
    let mut dropped = 0usize;
    while !lines.is_empty() && (lines.len() > MAX_MESSAGES || size > MAX_BYTES) {
        let gone = lines.remove(0);
        size -= gone.len() + 1;
        dropped += 1;
    }
    if dropped > 0 {
        let notice = InboundMessage {
            text: format!("(inbox overflow — {dropped} older message(s) dropped)"),
            source: "system".into(),
            ts: now(),
            meta: Value::Object(Default::default()),
        }
        .to_json()
        .into_bytes();
        lines.insert(0, notice);
    }
    if fh.seek(SeekFrom::Start(0)).is_err() {
        return;
    }
    let _ = fh.set_len(0);
    if !lines.is_empty() {
        let mut joined: Vec<u8> = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            if i > 0 {
                joined.push(b'\n');
            }
            joined.extend_from_slice(l);
        }
        joined.push(b'\n');
        let _ = fh.write_all(&joined);
    }
    let _ = fh.flush();
    // The cursor now points at nothing meaningful — reset it (re-reading a
    // trimmed inbox is correct, skipping past live messages is not).
    let _ = std::fs::remove_file(cursor_path(session_id));
}

/// Return messages queued since the last call, and advance the cursor. The
/// cursor moves only after the messages are in the caller's hands.
pub fn take(session_id: &str) -> Vec<InboundMessage> {
    let path = inbox_path(session_id);
    if !path.exists() {
        return Vec::new();
    }
    let cursor_file = cursor_path(session_id);
    match take_locked(&path, &cursor_file) {
        Ok(out) => out,
        // Nothing is lost by coming back: the cursor has not moved.
        Err(_) => Vec::new(),
    }
}

fn take_locked(path: &std::path::Path, cursor_file: &std::path::Path) -> Result<Vec<InboundMessage>, LockUnavailable> {
    let mut out = Vec::new();
    let locked = Locked::open(path, false)?;
    let mut fh = &locked.file;
    let mut start: u64 = std::fs::read_to_string(cursor_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
    if start > size {
        start = 0; // file was trimmed or replaced
    }
    if fh.seek(SeekFrom::Start(start)).is_err() {
        return Ok(out);
    }
    let mut blob = Vec::new();
    let _ = fh.read_to_end(&mut blob);
    let end = start + blob.len() as u64;
    for raw in blob.split(|b| *b == b'\n') {
        let raw = String::from_utf8_lossy(raw);
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(raw) {
            Ok(v) => out.push(InboundMessage::from_obj(&v)),
            Err(_) => continue, // a corrupt line must not stall the inbox
        }
    }
    if !out.is_empty() || !blob.is_empty() {
        let _ = std::fs::write(cursor_file, end.to_string());
    }
    // Fully drained and already large: reclaim the space.
    if end >= size && size as usize > MAX_BYTES / 2 {
        if fh.seek(SeekFrom::Start(0)).is_ok() {
            let _ = locked.file.set_len(0);
            let _ = std::fs::write(cursor_file, "0");
        }
    }
    Ok(out)
}

// --- question ownership ------------------------------------------------------

/// Record that `session_id` is the one waiting on this prompt.

/// Fingerprint of the bot this process polls through.
///
/// Question claims and forwarded answers were keyed on the bare Telegram
/// message id. That assumed ids are globally unique, but the adapter runs a
/// POOL of bots, each with its own chat and therefore its own id sequence
/// starting at 1 — so two concurrent sessions routinely hold prompts with the
/// same id, and a tap in session A's chat could be consumed by session B. After
/// a bot rotation an old answer file could resolve a later prompt with nobody
/// having tapped anything. telegram.rs already solved exactly this for
/// `last_message` by keying per token; questions, answers and owners did not.
fn bot_tag() -> String {
    let (token, _) = crate::remote::adapters::telegram::credentials();
    crate::remote::adapters::telegram::token_tag(&token)
}

/// Filename for a per-question file, scoped to the bot that asked.
fn scoped(message_id: i64) -> String {
    format!("{}-{message_id}", bot_tag())
}

/// Number of options the prompt behind `message_id` offered, if recorded.
///
/// A yes/no word typed in the chat is turned into a POSITIONAL answer ("p:0" /
/// "p:1"), and permission prompts and AskUserQuestion share this machinery — so
/// without the count, "no" on a three-way model-authored question silently
/// selected whatever option 1 happened to be. Absent for a claim written before
/// this was recorded, in which case the caller must not assume two.
pub fn question_option_count(message_id: i64) -> Option<usize> {
    let s = std::fs::read_to_string(QUESTION_DIR.join(scoped(message_id))).ok()?;
    s.lines().nth(1)?.trim().parse().ok()
}

/// Write one side-store record atomically.
///
/// These files are read by OTHER processes while they are being written — a
/// question claim is read by whichever session polls — so the publish has to be
/// the rename and not the write. A plain write truncates first, so the reader
/// could see an empty or half-written claim and discard the tap as stale, which
/// is the failure the side-store exists to prevent. Mirrors the Python build's
/// inbox._write_private.
fn write_private(path: &std::path::Path, text: &str) {
    let _ = crate::utils::file::atomic_write_bytes(&path.to_string_lossy(), text.as_bytes());
}

/// Record that `session_id` is waiting on this prompt. `free_text` marks one
/// that takes a typed answer of its own (an AskUserQuestion, whose picker always
/// offers "Other") — recorded rather than inferred, because whichever process
/// holds the poll lock reads the typed message, and it may not be the one that
/// asked.
pub fn claim_question_with_options(
    message_id: i64,
    session_id: &str,
    options: usize,
    free_text: bool,
) {
    let _ = std::fs::create_dir_all(&*QUESTION_DIR);
    write_private(
        &QUESTION_DIR.join(scoped(message_id)),
        &format!("{session_id}\n{options}\n{}", if free_text { 1 } else { 0 }),
    );
}

/// True when the prompt behind `message_id` takes a typed answer. False for a
/// claim without the flag — a permission prompt, or one written before the
/// flag existed — so a typed sentence can never become a grant.
pub fn question_accepts_text(message_id: i64) -> bool {
    std::fs::read_to_string(QUESTION_DIR.join(scoped(message_id)))
        .ok()
        .and_then(|s| s.lines().nth(2).map(|l| l.trim() == "1"))
        .unwrap_or(false)
}

pub fn question_owner(message_id: i64) -> Option<String> {
    let raw = std::fs::read_to_string(QUESTION_DIR.join(scoped(message_id))).ok()?;
    // First line only: the claim also records the prompt's option count.
    let s = raw.lines().next().unwrap_or("").trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Outstanding prompts, message id -> owning session.
pub fn pending_questions() -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let Ok(rd) = std::fs::read_dir(&*QUESTION_DIR) else {
        return out;
    };
    let prefix = format!("{}-", bot_tag());
    for entry in rd.flatten() {
        let p = entry.path();
        let Some(name) = p.file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
        // Only this bot's claims. Another bot's ids are a different sequence,
        // and answering one of those here is the collision scoping prevents.
        let Some(rest) = name.strip_prefix(&prefix) else { continue };
        let Ok(mid) = rest.parse::<i64>() else { continue };
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        out.insert(mid, text.trim().to_string());
    }
    out
}

/// Record which session posted an outbound message — makes replying a routing
/// signal. Its own directory (not the question store) so a typed "y" cannot
/// resolve against an hour-old alert.
pub fn claim_message(message_id: i64, session_id: &str) {
    if message_id == 0 || session_id.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all(&*OWNER_DIR);
    write_private(&OWNER_DIR.join(message_id.to_string()), session_id);
}

pub fn message_owner(message_id: i64) -> Option<String> {
    let s = std::fs::read_to_string(OWNER_DIR.join(message_id.to_string())).ok()?;
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub fn release_question(message_id: i64) {
    let _ = std::fs::remove_file(QUESTION_DIR.join(scoped(message_id)));
}

/// Leave a button tap where the session that asked will find it (one file per
/// question — no second inbox drainer, no lock).
pub fn deliver_answer(message_id: i64, data: &str) {
    let _ = std::fs::create_dir_all(&*ANSWER_DIR);
    write_private(&ANSWER_DIR.join(scoped(message_id)), data);
}

/// Hold a message whose target session is being chosen — parking beats
/// guessing.
pub fn park_route(picker_id: i64, text: &str, meta: &Value, choices: &[String]) {
    let _ = std::fs::create_dir_all(&*ROUTE_DIR);
    let mut obj = serde_json::Map::new();
    obj.insert("text".into(), Value::String(text.to_string()));
    obj.insert("meta".into(), meta.clone());
    obj.insert(
        "choices".into(),
        Value::Array(choices.iter().map(|c| Value::String(c.clone())).collect()),
    );
    write_private(&ROUTE_DIR.join(picker_id.to_string()), &py_json_dumps(&Value::Object(obj)));
}

pub fn take_route(picker_id: i64) -> Option<Value> {
    let path = ROUTE_DIR.join(picker_id.to_string());
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
    let _ = std::fs::remove_file(&path);
    Some(data)
}

/// Consume a forwarded answer for this question, if one arrived.
pub fn take_answer(message_id: i64) -> Option<String> {
    let path = ANSWER_DIR.join(scoped(message_id));
    let data = std::fs::read_to_string(&path).ok()?.trim().to_string();
    // An answer must POSTDATE the question it answers. Bot scoping already
    // stops ids colliding across the pool, but a leftover answer from a session
    // that died on this same bot could still resolve a later prompt reusing the
    // id, with nobody having tapped anything. Consumption had no time check at
    // all; the only bound was the hour-long sweep. A missing claim (already
    // released, or never written) falls through rather than discarding a real
    // answer.
    let claim = QUESTION_DIR.join(scoped(message_id));
    if let (Ok(a), Ok(q)) = (std::fs::metadata(&path), std::fs::metadata(&claim)) {
        if let (Ok(at), Ok(qt)) = (a.modified(), q.modified()) {
            if at < qt {
                let _ = std::fs::remove_file(&path);
                return None;
            }
        }
    }
    let _ = std::fs::remove_file(&path);
    if data.is_empty() {
        None
    } else {
        Some(data)
    }
}

/// File a message that did not name a session — owner bot first, then the only
/// live session, else parked. Returns the session id it was routed to, or
/// None if parked. (Kept correct though currently unreferenced, as in Python.)
#[allow(dead_code)]
pub fn deliver_unrouted(message: &InboundMessage) -> Option<String> {
    let owner = crate::remote::input::owning_session_id();
    if !owner.is_empty() {
        deliver(&owner, message);
        return Some(owner);
    }
    let sessions = live_sessions();
    if sessions.len() == 1 {
        let sid = sessions[0].session_id.clone();
        deliver(&sid, message);
        return Some(sid);
    }
    deliver(UNROUTED, message);
    None
}

pub fn take_unrouted() -> Vec<InboundMessage> {
    take(UNROUTED)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A forwarded answer must reach only the question it answers.
    ///
    /// The filename carries the bot fingerprint because the adapter runs a pool
    /// and each bot's message ids start at 1, so two sessions holding id 1 at
    /// once is ordinary. Keyed on the bare id, one session consumed the other's
    /// tap. The time check is the second half: a leftover answer on the same bot
    /// must not resolve a later prompt that reuses the id.
    ///
    /// Exercised through the file layer directly, since bot_tag() reads real
    /// credentials — what matters is that the two halves of the naming and the
    /// mtime ordering behave as claimed.
    #[test]
    fn answers_are_scoped_and_must_postdate_their_question() {
        let root = std::env::temp_dir().join(format!("cm_inbox_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temp root");

        // Two bots, same message id: distinct files, so neither can read the
        // other's.
        let a = root.join(format!("{}-{}", "botAAA", 1));
        let b = root.join(format!("{}-{}", "botBBB", 1));
        std::fs::write(&a, "session-A").unwrap();
        std::fs::write(&b, "session-B").unwrap();
        assert_ne!(a, b, "same id on two bots must not collide");
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "session-A");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "session-B");

        // The prefix filter in pending_questions only accepts its own bot.
        let prefix = "botAAA-";
        assert!(a.file_name().unwrap().to_string_lossy().strip_prefix(prefix).is_some());
        assert!(b.file_name().unwrap().to_string_lossy().strip_prefix(prefix).is_none());

        // An answer written BEFORE its claim is older, which is what take_answer
        // rejects on.
        let ans = root.join("botCCC-7.answer");
        std::fs::write(&ans, "p:1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let claim = root.join("botCCC-7.claim");
        std::fs::write(&claim, "session-C").unwrap();
        let at = std::fs::metadata(&ans).unwrap().modified().unwrap();
        let qt = std::fs::metadata(&claim).unwrap().modified().unwrap();
        assert!(at < qt, "a stale answer must compare older than the claim");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn py_dumps_matches_python_separators() {
        let v = serde_json::json!({"text": "안녕", "ts": 1.5, "meta": {"a": 1, "b": [1, 2]}});
        let s = py_json_dumps(&v);
        assert_eq!(s, r#"{"text": "안녕", "ts": 1.5, "meta": {"a": 1, "b": [1, 2]}}"#);
    }

    #[test]
    fn deliver_take_cursor_roundtrip() {
        std::env::set_var("HOME", std::env::temp_dir().join(format!("cm-inbox-{}", std::process::id())).to_str().unwrap());
        // NOTE: ROOT is Lazy and may already be resolved by another test; use a
        // synthetic session id to isolate instead.
        let sid = format!("test-{}", uuid::Uuid::new_v4());
        let m1 = InboundMessage::new("first", "telegram", Value::Object(Default::default()));
        let m2 = InboundMessage::new("second", "telegram", Value::Object(Default::default()));
        assert!(deliver(&sid, &m1));
        assert!(deliver(&sid, &m2));
        let got = take(&sid);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text, "first");
        assert_eq!(got[1].text, "second");
        // Cursor advanced: nothing more to take.
        assert!(take(&sid).is_empty());
        // A third message arrives after the drain.
        assert!(deliver(&sid, &InboundMessage::new("third", "telegram", Value::Null)));
        let got = take(&sid);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "third");
        unregister(&sid);
    }

    #[test]
    fn caps_drop_oldest_with_marker() {
        let sid = format!("test-cap-{}", uuid::Uuid::new_v4());
        for i in 0..(MAX_MESSAGES + 10) {
            assert!(deliver(&sid, &InboundMessage::new(format!("m{i}"), "t", Value::Null)));
        }
        let got = take(&sid);
        assert!(got.len() <= MAX_MESSAGES + 1);
        assert!(got[0].text.contains("inbox overflow"), "{}", got[0].text);
        assert_eq!(got.last().unwrap().text, format!("m{}", MAX_MESSAGES + 9));
        unregister(&sid);
    }

    #[test]
    fn question_answer_route_stores() {
        let mid = 987_654_321 + std::process::id() as i64;
        claim_question_with_options(mid, "sess-a", 2, false);
        assert_eq!(question_owner(mid).as_deref(), Some("sess-a"));
        assert!(!question_accepts_text(mid)); // a grant never takes a sentence
        claim_question_with_options(mid, "sess-a", 3, true);
        assert!(question_accepts_text(mid));
        assert_eq!(question_option_count(mid), Some(3));
        assert!(pending_questions().contains_key(&mid));
        deliver_answer(mid, "p:1");
        assert_eq!(take_answer(mid).as_deref(), Some("p:1"));
        assert!(take_answer(mid).is_none()); // consumed
        release_question(mid);
        assert!(question_owner(mid).is_none());

        park_route(mid, "hello", &serde_json::json!({"k": 1}), &["a".into(), "b".into()]);
        let parked = take_route(mid).unwrap();
        assert_eq!(parked.get("text").unwrap().as_str().unwrap(), "hello");
        assert!(take_route(mid).is_none()); // consumed
    }

    /// A sidecar left by a build from before FILE_FLAG_DELETE_ON_CLOSE has no
    /// handle behind it, so it would block every acquire until somebody deleted
    /// it by hand — for `_unrouted`, which sweep() skips on purpose, that means
    /// inbound remote messages stop for good. Acquiring must reclaim it.
    ///
    /// The far more important half is the one below it: a lock somebody is
    /// still holding must NOT be reclaimed. Taking a live lock reintroduces the
    /// two-writer corruption the lock exists to prevent, so an over-eager
    /// reclaim is worse than none.
    #[cfg(windows)]
    #[test]
    fn stale_sidecar_is_reclaimed_but_a_held_one_is_not() {
        let dir = std::env::temp_dir().join("codemason_stale_lock_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let target = dir.join("inbox.jsonl");
        let side = PathBuf::from(format!("{}.winlock", target.display()));

        // A plain file with nobody holding it: exactly what an older build
        // left behind after a crash.
        std::fs::write(&side, b"").expect("write stale sidecar");
        assert!(side.exists());
        let lock = Locked::open(&target, false).expect("stale sidecar must be reclaimed");
        assert!(side.exists(), "the reclaiming holder owns a sidecar of its own");

        // While that lock is held, reclaim must refuse. Asked of the predicate
        // rather than by calling Locked::open again: a second acquire is
        // SUPPOSED to wait out LOCK_RETRY_SECONDS before giving up, so driving
        // it here would add ten seconds to every run of the suite to observe a
        // decision the predicate makes instantly. This is the decision — the
        // loop reclaims if and only if it answers true.
        assert!(
            !sidecar_is_unheld(&side),
            "a lock somebody is holding must never be reclaimed"
        );

        // Releasing removes the sidecar, and the next acquire succeeds.
        drop(lock);
        assert!(!side.exists(), "closing the handle must remove the sidecar");
        let again = Locked::open(&target, false).expect("free after release");
        drop(again);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
