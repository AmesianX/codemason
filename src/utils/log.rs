//! Logging utilities for CodeMason
//! Equivalent to codemason/utils/log.py (src/utils/log.ts)

use once_cell::sync::Lazy;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::config::LOG_DIR;

pub static ERROR_LOG: Lazy<PathBuf> = Lazy::new(|| LOG_DIR.join("error.log"));
pub static DEBUG_LOG: Lazy<PathBuf> = Lazy::new(|| LOG_DIR.join("debug.log"));
/// One line per model request, ALWAYS on. See log_turn.
pub static TURN_LOG: Lazy<PathBuf> = Lazy::new(|| LOG_DIR.join("turns.log"));

/// Per-file size cap before rotation (one previous generation is kept as .1).
/// Without this the append-only logs grow without bound over a long autonomous
/// session. 0 disables rotation.
const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;

/// debug_log is OFF unless CODEMASON_DEBUG is set to a truthy value.
///
/// Rationale: debug_log serializes its payload and does a blocking disk write
/// on EVERY call, and hot-path callers pass the full request body each turn.
/// Gate it so the default run pays nothing; CODEMASON_DEBUG=1 restores it.
pub fn debug_enabled() -> bool {
    match std::env::var("CODEMASON_DEBUG") {
        Ok(v) => !matches!(
            v.trim().to_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        ),
        Err(_) => false,
    }
}

/// Rotate `path` to `path.1` (overwriting a prior .1) once it exceeds the cap,
/// so a single log file can't fill the disk. Best-effort; never raises.
///
/// Every codemason process on this machine appends to the same log, so the
/// rotation is the one part that needs coordinating: two processes that both saw
/// the file over the cap both renamed, and the second rename landed on the .1 the
/// first had just written — 16MB of log gone — or on the fresh file the first had
/// just started. The size check stays outside the lock, so the normal path (under
/// the cap) costs one metadata call, as before. Only the rare over-the-cap branch
/// asks for the lock, and it asks WITHOUT waiting: if another process is
/// rotating, this one has nothing left to do. The re-check inside is what makes
/// that correct — by then the rotation may already have happened.
fn rotate_if_needed(path: &Path) {
    if MAX_LOG_BYTES == 0 {
        return;
    }
    let over_cap = |p: &Path| std::fs::metadata(p).map(|m| m.len() > MAX_LOG_BYTES).unwrap_or(false);
    if !over_cap(path) {
        return;
    }
    let lock = crate::utils::file::FileLock::try_acquire(&crate::utils::config::lock_path(
        "log-rotate",
    ));
    if !lock.held {
        return; // another process is rotating it
    }
    if over_cap(path) {
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(".1");
        let dest = path.with_file_name(name);
        let _ = std::fs::rename(path, dest);
    }
}

/// Ensure log directory exists.
pub fn ensure_log_dir() {
    let _ = std::fs::create_dir_all(&*LOG_DIR);
    // 0700 to match the files inside it. A world-readable log directory makes
    // the per-file mode moot for anything created before this landed.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&*LOG_DIR, std::fs::Permissions::from_mode(0o700));
    }
}

pub fn now_iso() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    // 0600 at creation. These logs carry request bodies, tool results and — on
    // a transport failure — whatever an error string happened to contain, and
    // with no mode they land 0644 (0664 under a group-writable umask) in a
    // directory anyone can read. Only affects files created from here on; an
    // existing log keeps the mode it already had.
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(line.as_bytes())
}

/// Log an error to the error log file.
/// Record a panic where it can still be read tomorrow morning.
///
/// The default panic handler prints to stderr, which in the run this matters
/// most for — started at night, looked at the next day — is scrollback that may
/// be gone, scrolled past, or in a closed terminal. And it prints no stack at
/// all unless RUST_BACKTRACE happens to be set, so what survives is one line
/// with no location.
///
/// The point of a crash here is not recovery, it is DIAGNOSIS: the run is over
/// either way, and the only thing worth salvaging is why. So the backtrace is
/// force-captured rather than left to an environment variable, and it is
/// written to the same error log everything else uses, alongside the session id
/// so it can be lined up with the transcript that ends at the same moment.
///
/// Never panics itself: a panic inside a panic hook aborts the process and
/// takes the record with it.
pub fn log_panic(message: &str, location: Option<String>, session_id: Option<String>) {
    ensure_log_dir();
    // force_capture, not capture: the latter is a no-op unless RUST_BACKTRACE is
    // set, which is exactly the case we cannot rely on at 3am.
    let backtrace = std::backtrace::Backtrace::force_capture().to_string();
    let entry = json!({
        "timestamp": now_iso(),
        "type": "Panic",
        "message": message,
        "context": location,
        "session_id": session_id,
        "stack": backtrace,
    });
    rotate_if_needed(&ERROR_LOG);
    let _ = append_line(&ERROR_LOG, &format!("{}\n", entry));
}

pub fn log_error(error: &dyn std::fmt::Display, context: Option<&str>) {
    ensure_log_dir();
    let entry = json!({
        "timestamp": now_iso(),
        "type": "Unknown",
        "message": error.to_string(),
        "context": context,
        "stack": "",
    });
    rotate_if_needed(&ERROR_LOG);
    if let Err(e) = append_line(&ERROR_LOG, &format!("{}\n", entry)) {
        eprintln!("Failed to write error log: {e}");
    }
}

/// One line per model request: what was sent, what came back, how long.
///
/// Unconditional, unlike debug_log, and separate from error.log, unlike the
/// cold-prefill warning. Both of those were the wrong shape for the question
/// that keeps being asked of a slow turn — "was that prefill, or thinking, or
/// just a long answer?" — because a threshold only records the bad cases and
/// a debug flag records nothing unless it was set BEFORE the run that went
/// wrong. By then the run is over. A turn that took 500 seconds has to leave
/// its own numbers behind whether or not anyone expected it to.
///
/// One line, so it stays greppable across a long unattended session; rotated
/// like the others so it cannot grow without bound.
pub fn log_turn(fields: &serde_json::Value) {
    ensure_log_dir();
    rotate_if_needed(&TURN_LOG);
    let mut line = format!("[{}]", now_iso());
    if let Some(map) = fields.as_object() {
        for (k, v) in map {
            let v = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            line.push_str(&format!(" {k}={v}"));
        }
    }
    if let Err(e) = append_line(&TURN_LOG, &format!("{line}\n")) {
        eprintln!("Failed to write turn log: {e}");
    }
}

/// Write debug log entry (no-op unless CODEMASON_DEBUG is set).
pub fn debug_log(title: &str, data: Option<&serde_json::Value>) {
    if !debug_enabled() {
        return;
    }
    ensure_log_dir();
    rotate_if_needed(&DEBUG_LOG);
    let mut line = format!("[{}] {}", now_iso(), title);
    if let Some(d) = data {
        match serde_json::to_string_pretty(d) {
            Ok(s) => line.push_str(&format!(": {s}")),
            Err(_) => line.push_str(&format!(": {d}")),
        }
    }
    line.push('\n');
    let _ = append_line(&DEBUG_LOG, &line);
}

/// Log an analytics event (placeholder for statsig).
pub fn log_event(event_name: &str, properties: Option<&serde_json::Value>) {
    debug_log(&format!("EVENT: {event_name}"), properties);
}

/// Clear all log files.
pub fn clear_logs() {
    ensure_log_dir();
    for f in [&*ERROR_LOG, &*DEBUG_LOG] {
        let _ = std::fs::remove_file(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panic on a WORKER thread must be recorded too. Rust's panic hook is
    /// process-wide so this should hold by construction, but "should by
    /// construction" is what the asyncio equivalent also looked like right up
    /// until it was measured and found to record nothing.
    #[test]
    fn a_panic_on_another_thread_is_recorded_too() {
        let _serial = hook_test_lock();
        let marker = format!("worker-panic-{}", std::process::id());
        let previous = std::sync::Arc::new(std::panic::take_hook());
        let previous_hook = previous.clone();
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_default();
            log_panic(&message, info.location().map(|l| l.to_string()), None);
            // Chain to the handler we replaced, exactly as install_panic_hook
            // does in main.rs. A hook that only logs SWALLOWS the message: the
            // Windows session hit this when the lock was removed and a panic
            // landed on the sibling's hook — the test failed with no reason in
            // the output at all, and the assertion text was sitting in the log
            // file instead. Failing loudly matters more here than tidiness,
            // because a silent failure is one nobody can act on.
            previous_hook(info);
        }));

        let m = marker.clone();
        let handle = std::thread::spawn(move || panic!("{m}"));
        let joined = handle.join();
        // Restore. Not literally the original box — the Arc is wrapped in a
        // fresh one — but calling it reaches the same handler, which is what
        // matters. The Arc exists only so the hook closure above could chain to
        // it while this scope still owns it.
        std::panic::set_hook(Box::new(move |i| previous(i)));
        assert!(joined.is_err(), "the worker must actually have panicked");

        let text = std::fs::read_to_string(&*ERROR_LOG).unwrap_or_default();
        assert!(
            text.contains(&marker),
            "a worker-thread panic left nothing on disk"
        );
    }

    /// The hook path end to end: a REAL panic on a thread, caught so the test
    /// survives, must still have left the record. The unit test above proves
    /// log_panic writes; this proves the hook actually calls it, which is the
    /// half that can silently not be wired.
    #[test]
    fn a_real_panic_goes_through_the_hook_to_disk() {
        let _serial = hook_test_lock();
        let marker = format!("probe-panic-{}", std::process::id());
        // The binary installs its hook in main(), which tests do not run, so
        // install the same one here.
        let previous = std::sync::Arc::new(std::panic::take_hook());
        let previous_hook = previous.clone();
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_default();
            let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
            log_panic(&message, location, None);
            // Chain, for the same reason as the sibling above.
            previous_hook(info);
        }));

        let m = marker.clone();
        let outcome = std::panic::catch_unwind(move || panic!("{m}"));
        // Restore. Not literally the original box — the Arc is wrapped in a
        // fresh one — but calling it reaches the same handler, which is what
        // matters. The Arc exists only so the hook closure above could chain to
        // it while this scope still owns it.
        std::panic::set_hook(Box::new(move |i| previous(i)));
        assert!(outcome.is_err(), "the probe must actually have panicked");

        let text = std::fs::read_to_string(&*ERROR_LOG).unwrap_or_default();
        assert!(
            text.contains(&marker),
            "a real panic left nothing on disk — the hook is not wired to log_panic"
        );
        let line = text.lines().rev().find(|l| l.contains(&marker)).unwrap();
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or(json!({}));
        assert!(
            v["context"].as_str().unwrap_or("").contains("log.rs"),
            "the location must point at where it blew up: {line}"
        );
    }

    /// Serializes the tests that swap the PROCESS-WIDE panic hook.
    ///
    /// take_hook/set_hook are global. Two tests doing take-set-restore in
    /// parallel interleave into a swap: the second takes the FIRST's hook as its
    /// "previous", the first restores the default, and the second then restores
    /// the first's hook permanently — while the first's assertions run against a
    /// hook that is no longer its own. Measured as a 1-in-3 failure of
    /// a_panic_on_another_thread_is_recorded_too on Linux.
    ///
    /// Same shape as app_state::test_lock, and the same reason: a test that
    /// mutates process-global state has to be the only one doing it.
    fn hook_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A crash has to leave something readable the next morning. stderr is
    /// scrollback; this is a file. The record is useless without a location and
    /// a stack, so those are what this checks — not merely that a line was
    /// written.
    #[test]
    fn a_panic_leaves_its_cause_on_disk() {
        // Find THIS test's record, not simply the last Panic in the file. The
        // sibling above writes one too, tests run in parallel, and the error
        // log is shared — so "the newest Panic record" belongs to whichever
        // test wrote last. That raced: measured on Windows, the suite failed
        // roughly one run in four, always with the sibling's record quoted back
        // as the mismatch. A unique tag makes the search name what it wants,
        // the way the sibling already does with its own marker.
        let tag = format!("sess-{}", std::process::id());
        log_panic(
            "index out of bounds: the len is 3 but the index is 7",
            Some("src/tools/thing.rs:42:9".to_string()),
            Some(tag.clone()),
        );

        let text = std::fs::read_to_string(&*ERROR_LOG).unwrap_or_default();
        let line = text.lines().rev().find(|l| l.contains(&tag)).unwrap_or("");
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or(json!({}));

        assert_eq!(v["type"], "Panic", "no panic record was written: {line}");
        assert!(
            v["message"].as_str().unwrap_or("").contains("index out of bounds"),
            "the panic message must survive: {line}"
        );
        assert_eq!(
            v["context"], "src/tools/thing.rs:42:9",
            "without the location the record cannot be acted on"
        );
        assert_eq!(
            v["session_id"], tag,
            "needed to line the crash up with the transcript that ends at it"
        );
        let stack = v["stack"].as_str().unwrap_or("");
        assert!(
            stack.len() > 40,
            "a FORCED backtrace must be present even with RUST_BACKTRACE unset — \
             that is the whole point, since nobody sets it before going to bed: {stack:?}"
        );
        assert!(v["timestamp"].as_str().unwrap_or("").len() >= 19, "needs a timestamp");
    }
}
