//! Disk-backed task output — adapted from src/utils/task/diskOutput.ts.
//! Equivalent to codemason/utils/task/disk_output.py; see that file for the
//! substrate notes (fixed temp dir, write-side cap in LocalShellTask).

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

/// Read at most this many bytes when tailing a task's output file, so a
/// multi-GB log never gets pulled fully into memory (DEFAULT_MAX_READ_BYTES).
pub const DEFAULT_MAX_READ_BYTES: u64 = 8 * 1024 * 1024; // 8MB

/// Directory holding per-task output files (getTaskOutputDir).
pub fn get_task_output_dir() -> PathBuf {
    std::env::temp_dir().join("codemason").join("tasks")
}

/// Output file path for a task (getTaskOutputPath).
pub fn get_task_output_path(task_id: &str) -> PathBuf {
    get_task_output_dir().join(format!("{task_id}.output"))
}

/// Ensure the task output directory exists (ensureOutputDir).
pub fn ensure_output_dir() {
    // 0700, and the parent too. These files hold every background command's
    // full stdout and stderr — build logs, environment dumps, credentials that
    // appear in a tool's error output — and a bare create_dir_all lands 0755 in
    // a world-readable /tmp. tool_result_storage already chmods its own
    // directory 0700 with that reasoning; this one was missed. Leaving the
    // parent (/tmp/codemason) at 0755 in a world-writable /tmp also lets a
    // squatter who pre-creates it own everything written underneath.
    let dir = get_task_output_dir();
    let _ = std::fs::create_dir_all(&dir);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = std::path::Path::new(&dir);
        for p in [d.parent(), Some(d)].into_iter().flatten() {
            // Not ours to tighten (pre-existing, another owner) is fine: the
            // write itself still succeeds or fails on its own merits.
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700));
        }
    }
}

/// Read a task's output, tailing the file (getTaskOutput).
///
/// Caps at `max_bytes`; when the file is larger, returns the last `max_bytes`
/// prefixed with a one-line note recording how much earlier output was
/// omitted. Missing file → empty string.
pub fn get_task_output(task_id: &str, max_bytes: u64) -> String {
    let path = get_task_output_path(task_id);
    let Ok(meta) = std::fs::metadata(&path) else {
        return String::new();
    };
    let bytes_total = meta.len();
    let Ok(mut f) = std::fs::File::open(&path) else {
        return String::new();
    };
    if bytes_total > max_bytes && f.seek(SeekFrom::Start(bytes_total - max_bytes)).is_err() {
        return String::new();
    }
    let mut data = Vec::new();
    if f.read_to_end(&mut data).is_err() {
        return String::new();
    }
    let content = String::from_utf8_lossy(&data);
    let bytes_read = data.len() as u64;
    if bytes_total > bytes_read {
        let omitted_kb = ((bytes_total - bytes_read) as f64 / 1024.0).round() as u64;
        format!("[{omitted_kb}KB of earlier output omitted]\n{content}")
    } else {
        content.into_owned()
    }
}

/// Delete a task's output file (cleanupTaskOutput).
pub fn cleanup_task_output(task_id: &str) {
    let _ = std::fs::remove_file(get_task_output_path(task_id));
}

/// Create a task's output file, empty (initTaskOutput).
///
/// The source has two initializers: this one, and `initTaskOutputAsSymlink`,
/// which points the output path at the agent's own transcript JSONL so
/// TaskOutput reads the real conversation. codemason does not write per-agent
/// transcripts yet, so there is no target to link to and this is the right
/// equivalent — it is also what the source falls back to when the symlink
/// fails. Creating the file at spawn is what lets a tool report `output_file`
/// before the task has produced anything.
///
/// Best-effort: the output file is a convenience for TaskOutput, not a
/// precondition for running the task, so a read-only or full disk must not fail
/// the spawn.
pub fn init_task_output(task_id: &str) -> PathBuf {
    let path = get_task_output_path(task_id);
    ensure_output_dir();
    if !path.exists() {
        let _ = std::fs::write(&path, b"");
    }
    path
}

/// New content since `from_offset` (getTaskOutputDelta).
///
/// Seeks rather than reading the whole file: a long-running task's output grows
/// without bound and the only part anyone wants is what has arrived since the
/// last pass. A missing file is an empty delta, not an error — the file is
/// created at spawn but a task may be evicted from under us.
pub fn get_task_output_delta(task_id: &str, from_offset: i64) -> (String, i64) {
    use std::io::{Read, Seek, SeekFrom};
    let path = get_task_output_path(task_id);
    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return (String::new(), from_offset),
    };
    if file.seek(SeekFrom::Start(from_offset.max(0) as u64)).is_err() {
        return (String::new(), from_offset);
    }
    let mut buf = vec![0u8; DEFAULT_MAX_READ_BYTES as usize];
    let read = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return (String::new(), from_offset),
    };
    buf.truncate(read);
    (
        String::from_utf8_lossy(&buf).into_owned(),
        from_offset + read as i64,
    )
}

/// Current size (byte offset) of a task's output file (getTaskOutputSize).
pub fn get_task_output_size(task_id: &str) -> i64 {
    std::fs::metadata(get_task_output_path(task_id))
        .map(|m| m.len() as i64)
        .unwrap_or(0)
}
