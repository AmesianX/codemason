//! Persist large tool results to disk — adapted from utils/toolResultStorage.ts
//! (the per-tool persistence layer). Equivalent to
//! codemason/utils/tool_result_storage.py.
//!
//! When a single tool result exceeds its effective threshold, the full content
//! is written to a session file and the model receives a `<persisted-output>`
//! preview + file path instead of the full text. Also injects a marker for
//! empty results.
//!
//! Effective threshold = get_persistence_threshold(tool_name, declared): a
//! declared max_result_size_chars of Infinity opts out entirely (Read),
//! otherwise it is clamped by the system-wide DEFAULT_MAX_RESULT_SIZE_CHARS
//! (50k). See the Python reference for the substrate notes on what is NOT
//! ported (per-message aggregate budget, GrowthBook override, session dirs).

use once_cell::sync::Lazy;
use std::path::PathBuf;

/// System-wide cap regardless of what a tool declares.
pub const DEFAULT_MAX_RESULT_SIZE_CHARS: f64 = 50_000.0;

pub const TOOL_RESULTS_SUBDIR: &str = "tool-results";
pub const PERSISTED_OUTPUT_TAG: &str = "<persisted-output>";
pub const PERSISTED_OUTPUT_CLOSING_TAG: &str = "</persisted-output>";
/// Preview size in bytes for the reference message.
pub const PREVIEW_SIZE_BYTES: usize = 2000;

/// `value.toFixed(1).replace(/\.0$/, '')` — one decimal, strip a trailing .0.
fn fixed1_drop_zero(value: f64) -> String {
    let s = format!("{value:.1}");
    s.strip_suffix(".0").map(|p| p.to_string()).unwrap_or(s)
}

/// Mirror of utils/format.formatFileSize (KB/MB/GB, drop trailing .0).
fn format_file_size(size_in_bytes: usize) -> String {
    let kb = size_in_bytes as f64 / 1024.0;
    if kb < 1.0 {
        return format!("{size_in_bytes} bytes");
    }
    if kb < 1024.0 {
        return format!("{}KB", fixed1_drop_zero(kb));
    }
    let mb = kb / 1024.0;
    if mb < 1024.0 {
        return format!("{}MB", fixed1_drop_zero(mb));
    }
    let gb = mb / 1024.0;
    format!("{}GB", fixed1_drop_zero(gb))
}

/// Per-session/process token so concurrent sessions don't share a namespace
/// and a stale tool_use_id from a prior run can't collide with our
/// exclusive-create writes. Original keyed results under projectDir/sessionId.
static SESSION_TOKEN: Lazy<String> = Lazy::new(|| {
    std::env::var("CODEMASON_SESSION_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            let mut buf = [0u8; 4];
            use rand::RngCore;
            rand::rngs::OsRng.fill_bytes(&mut buf);
            format!("p{}-{}", std::process::id(), hex::encode(buf))
        })
});

fn session_root() -> PathBuf {
    std::env::temp_dir().join("codemason").join(&*SESSION_TOKEN)
}

pub fn get_tool_results_dir() -> PathBuf {
    session_root().join(TOOL_RESULTS_SUBDIR)
}

pub fn get_tool_result_path(id: &str, is_json: bool) -> PathBuf {
    let ext = if is_json { "json" } else { "txt" };
    get_tool_results_dir().join(format!("{id}.{ext}"))
}

pub fn ensure_tool_results_dir() {
    let d = get_tool_results_dir();
    let _ = std::fs::create_dir_all(&d);
    // Owner-only: tool outputs may hold sensitive data and /tmp is
    // world-readable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700));
        let _ = std::fs::set_permissions(session_root(), std::fs::Permissions::from_mode(0o700));
    }
}

/// Remove this session's results tree on exit (the Python atexit hook).
/// Registered with the C runtime's atexit on first use of this module's
/// persistence path, so normal exits clean up like the reference did.
pub fn cleanup_tool_results_session_dir() {
    let _ = std::fs::remove_dir_all(session_root());
}

extern "C" fn cleanup_at_exit() {
    cleanup_tool_results_session_dir();
}

static ATEXIT_REGISTERED: Lazy<()> = Lazy::new(|| {
    unsafe {
        libc::atexit(cleanup_at_exit);
    }
});

/// Resolve the effective persistence threshold (getPersistenceThreshold).
///
/// Infinity is a hard opt-out (checked first). Otherwise clamp the declared
/// cap by the system-wide default.
pub fn get_persistence_threshold(_tool_name: &str, declared_max_result_size_chars: f64) -> f64 {
    if !declared_max_result_size_chars.is_finite() {
        return declared_max_result_size_chars;
    }
    declared_max_result_size_chars.min(DEFAULT_MAX_RESULT_SIZE_CHARS)
}

/// Truncate at a newline boundary when possible (generatePreview).
/// Returns (preview, has_more). Sizes are measured in CHARS, exactly as the
/// Python reference sliced (despite the "bytes" name).
pub fn generate_preview(content: &str, max_bytes: usize) -> (String, bool) {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= max_bytes {
        return (content.to_string(), false);
    }
    let truncated: String = chars[..max_bytes].iter().collect();
    let last_newline = truncated.rfind('\n');
    let cut_point = match last_newline {
        Some(pos) => {
            // Compare in char terms, matching the Python index arithmetic.
            let char_pos = truncated[..pos].chars().count();
            if (char_pos as f64) > max_bytes as f64 * 0.5 {
                char_pos
            } else {
                max_bytes
            }
        }
        None => max_bytes,
    };
    (chars[..cut_point].iter().collect(), true)
}

/// True when content is empty or whitespace-only (isToolResultContentEmpty).
pub fn is_tool_result_content_empty(content: &str) -> bool {
    content.trim().is_empty()
}

/// Persistence info for a stored result.
pub struct PersistedResult {
    pub filepath: String,
    pub original_size: usize,
    pub is_json: bool,
    pub preview: String,
    pub has_more: bool,
}

/// Write the content to disk and return persistence info (persistToolResult).
///
/// Uses exclusive create so a result already persisted on a prior turn (same
/// deterministic tool_use_id) is not rewritten. Returns Err(message) on
/// failure so the caller can fall back to the original content.
pub fn persist_tool_result(
    content_str: &str,
    tool_use_id: &str,
    is_json: bool,
) -> Result<PersistedResult, String> {
    Lazy::force(&ATEXIT_REGISTERED);
    ensure_tool_results_dir();
    let filepath = get_tool_result_path(tool_use_id, is_json);
    match std::fs::OpenOptions::new().write(true).create_new(true).open(&filepath) {
        Ok(mut f) => {
            use std::io::Write;
            if let Err(e) = f.write_all(content_str.as_bytes()) {
                return Err(e.to_string());
            }
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Already persisted on a prior turn — fall through to preview.
        }
        Err(e) => return Err(e.to_string()),
    }

    let (preview, has_more) = generate_preview(content_str, PREVIEW_SIZE_BYTES);
    Ok(PersistedResult {
        filepath: filepath.to_string_lossy().into_owned(),
        original_size: content_str.chars().count(),
        is_json,
        preview,
        has_more,
    })
}

/// Build the <persisted-output> preview message (buildLargeToolResultMessage).
pub fn build_large_tool_result_message(result: &PersistedResult) -> String {
    let mut message = format!("{PERSISTED_OUTPUT_TAG}\n");
    message.push_str(&format!(
        "Output too large ({}). Full output saved to: {}\n\n",
        format_file_size(result.original_size),
        result.filepath
    ));
    message.push_str(&format!(
        "Preview (first {}):\n",
        format_file_size(PREVIEW_SIZE_BYTES)
    ));
    message.push_str(&result.preview);
    message.push_str(if result.has_more { "\n...\n" } else { "\n" });
    message.push_str(PERSISTED_OUTPUT_CLOSING_TAG);
    message
}

/// Map a tool's result string for the API, persisting it when too large.
///
/// codemason equivalent of processToolResultBlock → maybePersistLargeToolResult:
///   - empty content → a short marker so the model always has something to
///     react to (inc-4586)
///   - content over the effective threshold → persisted to disk, replaced with
///     the <persisted-output> preview message
///   - otherwise → unchanged
/// On persist failure the original content is returned (never silently lost).
pub fn process_tool_result_content(
    content: &str,
    tool_name: &str,
    tool_use_id: &str,
    max_result_size_chars: f64,
) -> String {
    if is_tool_result_content_empty(content) {
        return format!("({tool_name} completed with no output)");
    }

    let threshold = get_persistence_threshold(tool_name, max_result_size_chars);
    if (content.chars().count() as f64) <= threshold {
        return content.to_string();
    }

    match persist_tool_result(content, tool_use_id, false) {
        Ok(result) => build_large_tool_result_message(&result),
        // Persistence failed — return original content unchanged.
        Err(_) => content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_size_format() {
        assert_eq!(format_file_size(500), "500 bytes");
        assert_eq!(format_file_size(2048), "2KB");
        assert_eq!(format_file_size(2000), "2KB"); // 1.953 -> 2.0 -> "2"
        assert_eq!(format_file_size(1536), "1.5KB");
    }

    #[test]
    fn empty_marker() {
        let out = process_tool_result_content("  \n ", "Bash", "t1", 100.0);
        assert_eq!(out, "(Bash completed with no output)");
    }

    #[test]
    fn preview_newline_boundary() {
        let content = format!("{}\n{}", "a".repeat(1500), "b".repeat(1500));
        let (preview, has_more) = generate_preview(&content, 2000);
        assert!(has_more);
        assert_eq!(preview.chars().count(), 1500); // cut at newline (>50% of max)
    }

    #[test]
    fn infinity_opts_out() {
        let threshold = get_persistence_threshold("Read", f64::INFINITY);
        assert!(!threshold.is_finite());
    }
}
