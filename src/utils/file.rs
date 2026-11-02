//! File Utilities for CodeMason
//! Equivalent to codemason/utils/file.py (src/utils/file.ts)

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::log::log_error;
use super::state::get_cwd;

// === Types ===

/// 'CRLF' | 'LF'
pub type LineEndingType = &'static str;

// === Glob ===

/// Find files matching a glob pattern (joined under `cwd`, `**` recursive).
///
/// Returns (matching files sorted by mtime desc, whether results were
/// truncated past offset+limit). Errors are logged and yield ([], false),
/// matching the Python except path.
///
/// Substrate note: Python globlib skips dot-files unless a pattern component
/// itself starts with '.'; that rule is reproduced coarsely here — hidden
/// path components are skipped unless the pattern mentions a '.'-prefixed
/// component.
pub fn glob(
    file_pattern: &str,
    cwd: &str,
    limit: usize,
    offset: usize,
) -> (Vec<String>, bool) {
    match glob_inner(file_pattern, cwd) {
        Ok(mut file_paths) => {
            // Sort by modification time, newest first.
            file_paths.sort_by(|a, b| {
                let ma = std::fs::metadata(&a.0).and_then(|m| m.modified()).ok();
                let mb = std::fs::metadata(&b.0).and_then(|m| m.modified()).ok();
                mb.cmp(&ma)
            });
            let paths: Vec<String> = file_paths.into_iter().map(|(p, _)| p).collect();
            let truncated = paths.len() > offset + limit;
            let result = paths
                .into_iter()
                .skip(offset)
                .take(limit)
                .collect();
            (result, truncated)
        }
        Err(e) => {
            log_error(&format!("Error globbing pattern {file_pattern}: {e}"), None);
            (Vec::new(), false)
        }
    }
}

fn glob_inner(file_pattern: &str, cwd: &str) -> Result<Vec<(String, ())>, String> {
    let joined = Path::new(cwd).join(file_pattern);
    let pattern_str = joined.to_string_lossy().into_owned();

    // Base dir = longest prefix with no glob metacharacters.
    let mut base = PathBuf::new();
    let mut saw_magic = false;
    for comp in joined.components() {
        let s = comp.as_os_str().to_string_lossy();
        if !saw_magic && !s.contains('*') && !s.contains('?') && !s.contains('[') {
            base.push(comp.as_os_str());
        } else {
            saw_magic = true;
        }
    }
    if !saw_magic {
        // No glob magic: the pattern is a literal path.
        return if joined.is_file() {
            Ok(vec![(pattern_str, ())])
        } else {
            Ok(Vec::new())
        };
    }

    let glob = globset::GlobBuilder::new(&pattern_str)
        .literal_separator(true)
        .build()
        .map_err(|e| e.to_string())?
        .compile_matcher();

    // Python glob hides dot-entries unless the pattern names one explicitly.
    let pattern_mentions_hidden = file_pattern
        .split(['/', '\\'])
        .any(|c| c.starts_with('.'));

    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&base).follow_links(false).into_iter().filter_entry(|e| {
        pattern_mentions_hidden
            || e.depth() == 0
            || !e.file_name().to_string_lossy().starts_with('.')
    }) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        if glob.is_match(entry.path()) {
            out.push((entry.path().to_string_lossy().into_owned(), ()));
        }
    }
    Ok(out)
}

// === File Reading ===

/// Read file content safely, returning None on error (utf-8, errors replaced).
pub fn read_file_safe(filepath: &str) -> Option<String> {
    match std::fs::read(filepath) {
        Ok(bytes) => Some(String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) => {
            log_error(&format!("Error reading file {filepath}: {e}"), None);
            None
        }
    }
}

/// Decode raw bytes per a detected encoding name, replacing errors — the Rust
/// side of Python's open(encoding=..., errors='replace').
fn decode_with_encoding(bytes: &[u8], encoding: &str) -> String {
    match encoding {
        "utf-16-le" => {
            let mut data = bytes;
            // Strip the BOM the detector matched on.
            if data.starts_with(&[0xFF, 0xFE]) {
                data = &data[2..];
            }
            let units: Vec<u16> = data
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        "utf-8-sig" => {
            let data = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(bytes);
            String::from_utf8_lossy(data).into_owned()
        }
        "latin-1" => bytes.iter().map(|&b| b as char).collect(),
        // A legacy label reaches here only because verified_legacy_encoding
        // PROVED it round-trips, so this decode is exact. Missing this arm made
        // the label fall through to lossy UTF-8 — the same shape of gap that
        // Read and Grep each had. It is currently unreachable (nothing calls
        // read_text_content), and that is exactly why it is worth fixing now:
        // a decoder that is wrong while dead becomes a decoder that is wrong
        // when someone gives it a caller.
        other => match encoding_rs::Encoding::for_label(other.as_bytes()) {
            Some(enc) if !other.eq_ignore_ascii_case("utf-8") => enc.decode(bytes).0.into_owned(),
            _ => String::from_utf8_lossy(bytes).into_owned(),
        },
    }
}

/// Read text content from a file with optional line offset and limit.
///
/// Returns (content, line_count, total_lines). offset is 0-indexed.
pub fn read_text_content(
    file_path: &str,
    offset: usize,
    max_lines: Option<usize>,
) -> (String, usize, usize) {
    let encoding = detect_file_encoding(file_path);
    let content = match std::fs::read(file_path) {
        Ok(bytes) => decode_with_encoding(&bytes, &encoding),
        Err(e) => {
            log_error(&format!("Error reading file {file_path}: {e}"), None);
            return (String::new(), 0, 0);
        }
    };

    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    let to_return: Vec<&str> = match max_lines {
        Some(max) if total_lines.saturating_sub(offset) > max => {
            lines.iter().skip(offset).take(max).copied().collect()
        }
        _ => lines.iter().skip(offset).copied().collect(),
    };

    (to_return.join("\n"), to_return.len(), total_lines)
}

/// Write bytes atomically: a temp file in the same directory + rename, with
/// fsync and preserved mode. Prevents a truncated/corrupt file if the process
/// dies mid-write (the original's writeFileSyncAndFlush). Follows a symlink to
/// its target so we don't clobber the link with a plain file.
pub fn atomic_write_bytes(file_path: &str, data: &[u8]) -> std::io::Result<()> {
    let real = std::fs::canonicalize(file_path)
        .unwrap_or_else(|_| PathBuf::from(file_path));
    let directory = real.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));

    #[cfg(unix)]
    let mode = std::fs::metadata(&real)
        .map(|m| std::os::unix::fs::PermissionsExt::mode(&m.permissions()))
        .ok();

    // mkstemp equivalent: exclusive-create with a random suffix.
    let mut tmp = None;
    for _ in 0..16 {
        use rand::Rng;
        let suffix: String = {
            let mut rng = rand::rngs::OsRng;
            (0..8).map(|_| {
                let c = rng.gen_range(0..36u32);
                char::from_digit(c, 36).unwrap_or('0')
            }).collect()
        };
        let cand = directory.join(format!(".cm-tmp-{suffix}"));
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        // 0600 at creation, as the Python build's mkstemp already did. Without a
        // mode the temp file lands 0644 (0664 under a group-writable umask), and
        // a destination that did not exist has no mode to restore — so 0644 is
        // what the rename publishes. That reaches config.json, which holds the
        // API key, and the cut dumps, which hold request material.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&cand) {
            Ok(f) => {
                tmp = Some((cand, f));
                break;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let Some((tmp_path, mut fh)) = tmp else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create temp file",
        ));
    };

    let result = (|| -> std::io::Result<()> {
        fh.write_all(data)?;
        fh.flush()?;
        fh.sync_all()?;
        drop(fh);
        #[cfg(unix)]
        if let Some(mode) = mode {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(mode));
        }
        // Atomic within the same filesystem (os.replace), on Windows too.
        //
        // This used to delete the destination first, on the belief that a
        // Windows rename cannot land on an existing file. That is true of the
        // raw Win32 MoveFile and of Python's os.rename — which is exactly why
        // the Python side spells this os.replace — but NOT of std::fs::rename,
        // which goes through MoveFileExW with MOVEFILE_REPLACE_EXISTING and
        // overwrites (verified on this platform: rename onto an existing file
        // succeeds and the destination holds the new content).
        //
        // The delete was not merely redundant, it removed the guarantee this
        // function exists to provide: between the unlink and the rename the
        // file did not exist at all, so a crash or a concurrent reader in that
        // window saw config.json MISSING rather than stale. Windows now gets
        // the same single-step replace as everywhere else.
        std::fs::rename(&tmp_path, &real)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

/// Write text content to a file ('CRLF' converts newlines).
pub fn write_text_content(
    file_path: &str,
    content: &str,
    _encoding: &str,
    endings: &str,
) -> std::io::Result<()> {
    let to_write = if endings == "CRLF" {
        content.replace('\n', "\r\n")
    } else {
        content.to_string()
    };
    std::fs::write(file_path, to_write.as_bytes())
}

// === Encoding Detection ===

static ENCODING_CACHE: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static LINE_ENDINGS_CACHE: Lazy<Mutex<HashMap<String, &'static str>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn read_header(file_path: &str, n: usize) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(file_path)?;
    let mut buf = vec![0u8; n];
    let mut read_total = 0usize;
    loop {
        match f.read(&mut buf[read_total..]) {
            Ok(0) => break,
            Ok(n) => read_total += n,
            Err(e) => return Err(e),
        }
        if read_total == buf.len() {
            break;
        }
    }
    buf.truncate(read_total);
    Ok(buf)
}

/// Legacy encodings we are willing to name, most likely first.
///
/// Korean first because that is the environment this runs in; the rest are the
/// other single/double-byte families a source file on a CJK machine turns up in.
const LEGACY_CANDIDATES: &[&str] = &["euc-kr", "shift_jis", "gbk", "big5"];

/// Can these bytes be decoded as UTF-16 and re-encoded to exactly themselves?
///
/// Rust's UTF-16 decode cannot fail, which is the problem: `from_utf16_lossy`
/// turns an unpaired surrogate into U+FFFD, and chunking by 2 PADS a trailing
/// odd byte with 0x00. Both then get written back, so an edit to an unrelated
/// line silently rewrote a surrogate and made the file one byte longer.
///
/// Python never had this because its `utf-16-le` decode is strict and raises,
/// which drops Edit to byte-preserving latin-1. This is that strictness,
/// expressed as the same round-trip proof used for the legacy encodings.
pub fn utf16_round_trips(bytes: &[u8], little_endian: bool) -> bool {
    if bytes.len() % 2 != 0 {
        return false; // a trailing odd byte cannot survive; it would be padded
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| {
            if little_endian {
                u16::from_le_bytes([c[0], c[1]])
            } else {
                u16::from_be_bytes([c[0], c[1]])
            }
        })
        .collect();
    // from_utf16 (not _lossy) fails on an unpaired surrogate, which is exactly
    // the case that was being silently replaced.
    let Ok(text) = String::from_utf16(&units) else { return false };
    let back: Vec<u8> = text
        .encode_utf16()
        .flat_map(|u| if little_endian { u.to_le_bytes() } else { u.to_be_bytes() })
        .collect();
    back == bytes
}

/// Is this one of the legacy labels the detector can return?
pub fn is_legacy_label(label: &str) -> bool {
    LEGACY_CANDIDATES.contains(&label)
}

/// The candidate that reproduces `bytes` EXACTLY, or None.
///
/// This is a verification, not a guess, and that distinction is the whole
/// safety argument. Edit reads a file, edits text, and writes it back through
/// the same codec — so naming an encoding that cannot reproduce the original
/// bytes is how a CP949 source lost every Hangul character to U+FFFD once
/// already. Here a candidate is accepted only when decode-then-encode returns
/// the input byte for byte, which makes the write-back lossless by construction
/// rather than by hoping the detector guessed right.
///
/// `bytes` is a header sample, so a multi-byte sequence can be cut at the end;
/// one or two trailing bytes are dropped before giving up, since a truncated
/// tail says nothing about the encoding of everything before it.
pub fn verified_legacy_encoding(bytes: &[u8]) -> Option<&'static str> {
    verified_legacy_encoding_inner(bytes, 2)
}

/// Same, but the input must round-trip WHOLE — no trailing bytes ignored.
///
/// The trimming variant exists for a 4096-byte HEADER, where a multi-byte
/// sequence really can be cut by the sample boundary. Applied to a whole file it
/// is unsound, and that was a real hole: a label accepted on `raw[..len-2]` was
/// then used to encode `raw`, so Edit could rewrite the final bytes of a file it
/// had never proved it could reproduce. Measured — a shift_jis file ending in a
/// non-round-tripping pair had its last two bytes silently changed by an edit to
/// an unrelated ASCII line. encoding_rs has thousands of such sequences.
pub fn verified_legacy_encoding_exact(bytes: &[u8]) -> Option<&'static str> {
    verified_legacy_encoding_inner(bytes, 0)
}

fn verified_legacy_encoding_inner(bytes: &[u8], max_trim: usize) -> Option<&'static str> {
    for label in LEGACY_CANDIDATES {
        let Some(enc) = encoding_rs::Encoding::for_label(label.as_bytes()) else { continue };
        for trim in 0..=max_trim {
            if trim >= bytes.len() {
                break;
            }
            let sample = &bytes[..bytes.len() - trim];
            if sample.is_empty() {
                break;
            }
            let (text, _, had_errors) = enc.decode(sample);
            if had_errors {
                continue;
            }
            let (back, _, _) = enc.encode(&text);
            if back.as_ref() == sample {
                return Some(label);
            }
        }
    }
    None
}

/// Detect file encoding by reading BOM or analyzing content (memoized, as the
/// Python lru_cache was). Returns 'utf-8', 'utf-16-le', 'utf-8-sig' or
/// 'latin-1'.
pub fn detect_file_encoding(file_path: &str) -> String {
    if let Some(v) = ENCODING_CACHE.lock().get(file_path) {
        return v.clone();
    }
    let enc = match read_header(file_path, 4096) {
        Ok(header) => {
            if header.starts_with(&[0xFF, 0xFE]) {
                "utf-16-le"
            } else if header.starts_with(&[0xFE, 0xFF]) {
                // Without this a UTF-16BE file falls through to latin-1, which
                // round-trips its bytes but reads as mojibake — and file_read_tool
                // already handles \xfe\xff, so the two disagreed about one file.
                "utf-16-be"
            } else if header.starts_with(&[0xEF, 0xBB, 0xBF]) {
                "utf-8-sig"
            } else if std::str::from_utf8(&header).is_ok() || utf8_prefix_ok(&header) {
                "utf-8"
            } else if let Some(label) = verified_legacy_encoding(&header) {
                // A legacy CJK file, proven by round-trip rather than sniffed.
                // Without this a CP949 source decoded as latin-1: bytes were
                // preserved (so Edit was safe) but the text was not the text, so
                // searching it for Hangul matched nothing.
                label
            } else {
                "latin-1"
            }
        }
        Err(e) => {
            log_error(&format!("Error detecting encoding for {file_path}: {e}"), None);
            "utf-8"
        }
    };
    ENCODING_CACHE.lock().insert(file_path.to_string(), enc.to_string());
    enc.to_string()
}

/// A 4096-byte header may cut a multi-byte UTF-8 sequence at the end; Python's
/// header.decode('utf-8') would raise there and misclassify as latin-1 only
/// when the WHOLE header is invalid. Accept a header whose only invalid bytes
/// are an incomplete trailing sequence.
fn utf8_prefix_ok(header: &[u8]) -> bool {
    match std::str::from_utf8(header) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none() && header.len() == 4096,
    }
}

/// Detect line ending type of a file (memoized): 'CRLF' or 'LF'.
pub fn detect_line_endings(file_path: &str) -> &'static str {
    if let Some(v) = LINE_ENDINGS_CACHE.lock().get(file_path) {
        return v;
    }
    let v = match read_header(file_path, 4096) {
        Ok(content) => {
            let crlf_count = content.windows(2).filter(|w| w == b"\r\n").count();
            let lf_count = content.iter().filter(|&&b| b == b'\n').count() - crlf_count;
            if crlf_count > lf_count {
                "CRLF"
            } else {
                "LF"
            }
        }
        Err(e) => {
            log_error(&format!("Error detecting line endings for {file_path}: {e}"), None);
            "LF"
        }
    };
    LINE_ENDINGS_CACHE.lock().insert(file_path.to_string(), v);
    v
}

// === Path Utilities ===

// (Use utils::state::is_in_directory, the single segment-containment
// implementation — the duplicate that lived here in an early Python version
// was dead code and is not ported.)

/// Normalize a file path, fixing common issues:
/// - Fixes paths missing leading slash (e.g., 'home/user' -> '/home/user')
/// - Resolves relative paths against the session cwd
pub fn normalize_file_path(file_path: &str) -> String {
    let mut normalized = file_path.to_string();
    if !Path::new(file_path).is_absolute() {
        for prefix in ["home/", "usr/", "tmp/", "var/", "etc/"] {
            if file_path.starts_with(prefix) {
                normalized = format!("/{file_path}");
                break;
            }
        }
    }
    if Path::new(&normalized).is_absolute() {
        return normalized;
    }
    lexical_abspath(&Path::new(&get_cwd()).join(&normalized))
}

/// os.path.abspath: absolutize + lexically normalize (no symlink resolution).
fn lexical_abspath(p: &Path) -> String {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(&get_cwd()).join(p)
    };
    let mut out = PathBuf::new();
    for comp in abs.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out.to_string_lossy().into_owned()
}

/// Get absolute path from a path string.
pub fn get_absolute_path(path: Option<&str>) -> Option<String> {
    let path = path?;
    if path.is_empty() {
        return None;
    }
    if Path::new(path).is_absolute() {
        Some(path.to_string())
    } else {
        Some(lexical_abspath(&Path::new(&get_cwd()).join(path)))
    }
}

/// os.path.relpath(path, start): relative path, using '..' where needed.
fn relpath(path: &str, start: &str) -> String {
    let path_comps: Vec<String> = Path::new(&lexical_abspath(Path::new(path)))
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let start_comps: Vec<String> = Path::new(&lexical_abspath(Path::new(start)))
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let common = path_comps
        .iter()
        .zip(start_comps.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut parts: Vec<String> = Vec::new();
    for _ in common..start_comps.len() {
        parts.push("..".to_string());
    }
    parts.extend(path_comps[common..].iter().cloned());
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join(std::path::MAIN_SEPARATOR_STR)
    }
}

/// Get both absolute and relative paths.
pub fn get_absolute_and_relative_paths(path: Option<&str>) -> (Option<String>, Option<String>) {
    let absolute_path = get_absolute_path(path);
    let relative_path = absolute_path.as_ref().map(|a| relpath(a, &get_cwd()));
    (absolute_path, relative_path)
}

/// The shortest honest way to name a path on screen (getDisplayPath).
///
/// Three steps, in the source's order: cwd-relative when the file is under the
/// working directory, `~`-prefixed when it is under the home directory, and the
/// absolute path otherwise. The `..` check is what keeps a sibling directory
/// from being displayed as a relative path that is longer and harder to read
/// than the absolute one.
pub fn get_display_path(file_path: &str) -> String {
    if file_path.is_empty() {
        return String::new();
    }
    let (_, relative) = get_absolute_and_relative_paths(Some(file_path));
    if let Some(rel) = relative {
        if !rel.starts_with("..") {
            return rel;
        }
    }
    if let Some(home) = dirs_home() {
        let prefix = format!("{home}{}", std::path::MAIN_SEPARATOR);
        if file_path.starts_with(&prefix) {
            return format!("~{}", &file_path[home.len()..]);
        }
    }
    file_path.to_string()
}

fn dirs_home() -> Option<String> {
    std::env::var("HOME").ok().filter(|h| !h.is_empty())
}

// === File Discovery ===

/// Find a file in subdirectories when the exact path doesn't exist.
///
/// For example, if ai_providers/file.py does not exist, search **/<path> to
/// find deep-eye/ai_providers/file.py. node_modules and .git are filtered out.
pub fn find_file_in_subdirectories(file_path: &str) -> Option<String> {
    // Don't search if path is absolute or already exists.
    if Path::new(file_path).is_absolute() || Path::new(file_path).exists() {
        return None;
    }
    let suffix = Path::new(file_path);
    for entry in walkdir::WalkDir::new(".").follow_links(false) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let s = p.to_string_lossy();
        if s.contains("node_modules") || s.contains(".git") {
            continue;
        }
        if p.ends_with(suffix) {
            // Match the Python glob's relative form (no leading './').
            let cleaned = s.strip_prefix("./").unwrap_or(&s).to_string();
            return Some(cleaned);
        }
    }
    None
}

/// Find files with the same name but different extensions. Returns the
/// FILENAME of a similar file (not its path), or None.
pub fn find_similar_file(file_path: &str) -> Option<String> {
    let p = Path::new(file_path);
    let dir_path = p.parent()?;
    let file_basename = p.file_stem()?.to_string_lossy().into_owned();

    if !dir_path.exists() {
        return None;
    }
    let entries = std::fs::read_dir(dir_path).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let stem = Path::new(&name)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if stem == file_basename && dir_path.join(&name) != p {
            return Some(name);
        }
    }
    None
}

// === Line Numbers ===

/// Add cat -n style line numbers to content (1-indexed start_line).
/// Numbers under 6 digits are right-padded to 6; longer ones print as-is.
pub fn add_line_numbers(content: &str, start_line: usize) -> String {
    if content.is_empty() {
        return String::new();
    }
    content
        .split('\n')
        .enumerate()
        .map(|(i, line)| {
            let line_num = i + start_line;
            let num_str = line_num.to_string();
            if num_str.len() >= 6 {
                format!("{num_str}\t{line}")
            } else {
                format!("{num_str:>6}\t{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// === Directory Utilities ===

/// Check if a directory is empty.
pub fn is_dir_empty(dir_path: &str) -> bool {
    match std::fs::read_dir(dir_path) {
        Ok(mut it) => it.next().is_none(),
        Err(e) => {
            log_error(&format!("Error checking directory {dir_path}: {e}"), None);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("cm-lock-{}-{tag}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    /// Mutual exclusion, and the shape that makes it residue-free. flock is per
    /// open file description, so two Files in ONE process contend exactly as
    /// two processes do — which is also why these assertions carry across
    /// builds: the Python client takes the same flock on the same sidecar path.
    #[test]
    fn a_held_lock_excludes_everyone_and_leaves_nothing_behind() {
        let path = lock_dir("excl").join("config.json.lock");

        let held = FileLock::acquire(&path);
        assert!(held.held, "an uncontended lock must be taken");
        // try_acquire never waits: the answer for a busy lock is immediate.
        assert!(!FileLock::try_acquire(&path).held, "a held lock must not be handed out twice");
        // Waiting for a held lock must END. A blocking LOCK_EX here would hang
        // this test for ever, which is exactly what it would do to a session.
        let t0 = std::time::Instant::now();
        assert!(!FileLock::open(&path, 0.2).held);
        assert!(t0.elapsed() < std::time::Duration::from_secs(2), "the deadline must be honoured");

        // Release is the descriptor closing — the same thing the kernel does for
        // us when a holder is SIGKILLed, which is why no state can go stale.
        drop(held);
        assert!(FileLock::try_acquire(&path).held, "a dropped lock must be free at once");

        // The sidecar is left in place and empty: nothing to reclaim, and the
        // inode every process locks stays the same one. Unlinking it would let
        // the next process lock a FRESH inode while a holder still has the old
        // one — two holders, silently.
        assert!(path.exists(), "the lock file must survive its holder");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0, "a lock must carry no state");
    }

    /// Which file this path names — an inode number on unix. Windows has no
    /// inode: (volume serial, file index) from GetFileInformationByHandle is
    /// the same statement, stable for the life of a file and different for the
    /// file that replaced it, and `std` exposes it only behind the unstable
    /// `windows_by_handle` feature, so the call is declared here.
    ///
    /// The handle is closed before the value is returned. A handle left open on
    /// the document would make the rename that replaces it fail with a sharing
    /// violation, and this test would then pass for the wrong reason.
    #[cfg(unix)]
    fn file_identity(p: &std::path::Path) -> u128 {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(p).unwrap().ino() as u128
    }

    #[cfg(windows)]
    fn file_identity(p: &std::path::Path) -> u128 {
        use std::os::windows::io::AsRawHandle;

        #[repr(C)]
        struct FileTime {
            low: u32,
            high: u32,
        }
        #[repr(C)]
        struct ByHandleFileInformation {
            file_attributes: u32,
            creation_time: FileTime,
            last_access_time: FileTime,
            last_write_time: FileTime,
            volume_serial_number: u32,
            file_size_high: u32,
            file_size_low: u32,
            number_of_links: u32,
            file_index_high: u32,
            file_index_low: u32,
        }
        extern "system" {
            fn GetFileInformationByHandle(
                handle: *mut std::ffi::c_void,
                info: *mut ByHandleFileInformation,
            ) -> i32;
        }

        let fh = std::fs::File::open(p).unwrap();
        let mut info = std::mem::MaybeUninit::<ByHandleFileInformation>::zeroed();
        let ok = unsafe {
            GetFileInformationByHandle(fh.as_raw_handle() as *mut _, info.as_mut_ptr())
        };
        assert!(ok != 0, "GetFileInformationByHandle failed on {}", p.display());
        let info = unsafe { info.assume_init() };
        drop(fh);
        ((info.volume_serial_number as u128) << 64)
            | ((info.file_index_high as u128) << 32)
            | info.file_index_low as u128
    }

    /// The document is replaced by rename, so the lock cannot live on it. This
    /// pins that the write leaves the sidecar's inode alone.
    #[test]
    fn writing_the_document_does_not_replace_the_lock() {
        let dir = lock_dir("inode");
        let doc = dir.join("doc.json");
        let lock = dir.join("doc.json.lock");
        std::fs::write(&doc, b"{}").unwrap();

        let ino = file_identity;
        assert!(update_json_document(&doc, &lock, |cur| {
            let mut m = match cur {
                Some(serde_json::Value::Object(m)) => m,
                _ => serde_json::Map::new(),
            };
            m.insert("a".into(), serde_json::Value::from(1));
            Some(serde_json::Value::Object(m))
        }, false));
        let before = ino(&lock);
        let doc_before = ino(&doc);
        assert!(update_json_document(&doc, &lock, |cur| {
            let mut m = match cur {
                Some(serde_json::Value::Object(m)) => m,
                _ => serde_json::Map::new(),
            };
            m.insert("b".into(), serde_json::Value::from(2));
            Some(serde_json::Value::Object(m))
        }, false));

        assert_eq!(ino(&lock), before, "the lock sidecar must keep its inode");
        assert_ne!(ino(&doc), doc_before, "the document is replaced, which is why the lock is a sidecar");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&doc).unwrap()).unwrap();
        assert_eq!(v["a"], 1, "an earlier change must survive the next one");
        assert_eq!(v["b"], 2);
    }

    /// A document that exists but cannot be read is left alone. Overwriting it
    /// would replace whatever the user hand-edited — hooks, permissions, MCP
    /// servers — with only what this process wanted to add.
    #[test]
    fn an_unreadable_document_is_never_overwritten() {
        let dir = lock_dir("unreadable");
        let doc = dir.join("broken.json");
        let broken = "{ \"hooks\": [ , }";
        std::fs::write(&doc, broken).unwrap();
        let called = std::sync::atomic::AtomicBool::new(false);
        let wrote = update_json_document(&doc, &dir.join("broken.json.lock"), |cur| {
            called.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = cur;
            Some(serde_json::json!({"ours": true}))
        }, false);
        assert!(!wrote, "an unreadable document must not be written");
        assert!(!called.load(std::sync::atomic::Ordering::SeqCst), "mutate must not even run");
        assert_eq!(std::fs::read_to_string(&doc).unwrap(), broken, "left byte for byte");

        // A MISSING file is a different answer: that one is created.
        let fresh = dir.join("fresh.json");
        assert!(update_json_document(&fresh, &dir.join("fresh.json.lock"), |cur| {
            assert!(cur.is_none(), "a missing document reads as None");
            Some(serde_json::json!({"ours": true}))
        }, false));
        assert!(fresh.exists());
    }

    /// `mutate` returning None writes nothing — a caller that finds there is
    /// nothing to change must not churn the file (or its mtime).
    #[test]
    fn a_no_op_update_writes_nothing() {
        let dir = lock_dir("noop");
        let doc = dir.join("n.json");
        std::fs::write(&doc, b"{\"keep\": true}").unwrap();
        assert!(!update_json_document(&doc, &dir.join("n.json.lock"), |_| None, false));
        assert_eq!(std::fs::read_to_string(&doc).unwrap(), "{\"keep\": true}");
    }

    /// The detector must not start CLAIMING a legacy encoding for files that
    /// are not one. A false positive here is not cosmetic: Edit would re-encode
    /// the file through that codec.
    #[test]
    fn only_files_that_really_round_trip_get_a_legacy_label() {
        // Valid UTF-8 must never reach the legacy branch at all, but even if it
        // did, ASCII round-trips everywhere — so the ORDER of checks is what
        // protects it. Assert the branch order via the public detector instead.
        let dir = std::env::temp_dir().join(format!("cm-enc-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let utf8 = dir.join("a.txt");
        std::fs::write(&utf8, "한글 UTF-8 문서".as_bytes()).unwrap();
        assert_eq!(
            detect_file_encoding(&utf8.to_string_lossy()),
            "utf-8",
            "valid UTF-8 must stay UTF-8 — the legacy check comes after it"
        );

        // Bytes valid in NO candidate must still fall back to byte-preserving
        // latin-1 rather than being forced into one.
        assert_eq!(verified_legacy_encoding(&[0xC0, 0x80, 0xFF, 0xFE, 0x00, 0x41]), None);

        // A real CP949 sequence is accepted.
        let (kr, _, _) = encoding_rs::EUC_KR.encode("한글");
        assert_eq!(verified_legacy_encoding(&kr), Some("euc-kr"));

        // Whatever is accepted must reproduce the input exactly — that is the
        // entire safety property Edit leans on.
        for probe in [&kr[..], b"plain ascii", &[0x81u8, 0x40][..]] {
            if let Some(label) = verified_legacy_encoding(probe) {
                let enc = encoding_rs::Encoding::for_label(label.as_bytes()).unwrap();
                let (text, _, err) = enc.decode(probe);
                assert!(!err, "{label} was accepted but decodes with errors");
                assert_eq!(enc.encode(&text).0.as_ref(), probe, "{label} does not round-trip");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn line_numbers_shape() {
        assert_eq!(add_line_numbers("a\nb", 1), "     1\ta\n     2\tb");
        assert_eq!(add_line_numbers("", 1), "");
        assert!(add_line_numbers("x", 123456).starts_with("123456\t"));
    }

    #[test]
    fn relpath_basic() {
        assert_eq!(relpath("/a/b/c", "/a/b"), "c");
        assert_eq!(relpath("/a/b", "/a/b"), ".");
        assert_eq!(relpath("/a/x", "/a/b"), format!("..{}x", std::path::MAIN_SEPARATOR));
    }

    /// The rooted spellings below are what the Python oracle returns on POSIX.
    /// On Windows a leading-slash path is NOT absolute (no drive letter), so
    /// `os.path.abspath` — which `normalize_file_path` falls through to —
    /// attaches the cwd's drive: `/home/user/x` becomes `C:\home\user\x`
    /// (verified against `os.path.abspath` on this platform). The code is
    /// oracle-faithful in both cases; only the expected spelling differs, so
    /// assert the drive-relative shape rather than the POSIX string.
    #[test]
    fn normalize_missing_slash() {
        #[cfg(not(windows))]
        {
            assert_eq!(normalize_file_path("home/user/x"), "/home/user/x");
            assert_eq!(normalize_file_path("/abs/path"), "/abs/path");
        }
        #[cfg(windows)]
        {
            // The missing leading slash is still added, then absolutized.
            let got = normalize_file_path("home/user/x");
            assert!(
                got.ends_with(r"\home\user\x"),
                "expected a path ending in \\home\\user\\x, got {got}"
            );
            assert!(Path::new(&got).is_absolute(), "{got} should be absolute");

            let got = normalize_file_path("/abs/path");
            assert!(
                got.ends_with(r"\abs\path"),
                "expected a path ending in \\abs\\path, got {got}"
            );
            assert!(Path::new(&got).is_absolute(), "{got} should be absolute");
        }
    }

    /// The point of an atomic write is that a reader sees either the old file
    /// or the new one, never neither. Overwriting an existing file has to work
    /// in one step, and nothing may be left behind if it does.
    #[test]
    fn atomic_write_replaces_in_one_step() {
        let dir = std::env::temp_dir().join("codemason_atomic_write_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("config.json");
        let p = path.to_string_lossy().into_owned();

        atomic_write_bytes(&p, b"first").expect("create");
        assert_eq!(std::fs::read(&path).unwrap(), b"first");

        // 0600 on a file this created, matching the Python build's mkstemp. The
        // targets include config.json (API key) and the cut dumps (request
        // material); with no mode set the temp file landed 0644 and, there being
        // no destination mode to restore, the rename published that.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(m, 0o600, "new file mode was {m:o}");
        }

        // An EXISTING file keeps the mode it had: a user who widened a log or a
        // config on purpose does not get it narrowed back by the next write.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
            atomic_write_bytes(&p, b"kept").expect("rewrite");
            let m = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(m, 0o640, "existing file mode became {m:o}");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }

        // The overwrite is the case that used to unlink first on Windows,
        // leaving a window with no file at all.
        atomic_write_bytes(&p, b"second").expect("overwrite");
        assert_eq!(std::fs::read(&path).unwrap(), b"second");

        // Repeated writes must not accumulate temp files next to the target.
        for i in 0..5 {
            atomic_write_bytes(&p, format!("v{i}").as_bytes()).expect("rewrite");
        }
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "config.json")
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

// === Inter-process locks for shared files ===================================
//
// Several codemason processes share one config dir, and they are not all the
// same program: a Python session, a Rust session, two or three of each, on
// different models and different endpoints. Every file in that tree is either
// owned by exactly one writer (a session transcript, an inbox) or it is a
// SHARED DOCUMENT any of them may rewrite — config.json, settings.json,
// mcp.json. A shared document is read, changed and written back, and that
// sequence is not atomic however careful the final write is: two processes that
// read the same bytes and then both write produce whatever the later rename
// happened to carry. Measured on this box, with the two builds saving one
// setting each at the same moment: 150 of 150 trials lost one of the two edits.
//
// Mirrors codemason/utils/file.py (FileLock / update_json_document) — the lock
// PATH is part of the contract, because a name that differs by one byte
// excludes nothing across builds. The rules:
//
// flock, not a lock file whose EXISTENCE is the lock. The kernel releases an
// flock when the descriptor closes — on return, on panic, on exit, on SIGKILL,
// on a segfault, on a power cut — so there is no stale state to reclaim and
// nothing to clean up. A pid file or a create_new lock has to be removed by the
// holder, and the one run that does not remove it wedges every run after it.
// The lock file here carries no content and is never deleted: an empty file
// left behind is not residue, it is the lock we take next time.
//
// A SIDECAR, never the document itself. The document is replaced by rename
// (atomic_write_bytes), which swaps the inode. A lock held on the old inode is
// invisible to a process that opens the path afterwards and gets the new one,
// so both would write while both believed they held the lock — the lock would
// silently do nothing. The sidecar is created once and never renamed, never
// unlinked, so every process locks the same inode forever. For the same reason
// nothing here may unlink a lock file: unlinking a flocked path lets the next
// process create a fresh inode and lock that, which is how you get two holders.
//
// LOCK_NB in a deadline loop, never a blocking LOCK_EX: a blocking acquire
// cannot be interrupted and flock survives SIGSTOP, so one wedged or stopped
// process would park every other session indefinitely with nothing on screen.
//
// A timeout DEGRADES, it does not fail. Not acquiring leaves exactly the
// pre-lock behaviour — a possible lost update, never a corrupt file, because
// the write is still an atomic rename. Refusing to save would turn a rare race
// into a visible failure of a command the user just typed.
//
// Held for the read-modify-write only: a few milliseconds, on a cold path (a
// settings change, a first-run backfill). Never across network I/O, never
// across a prompt, never on the streaming path.

const LOCK_POLL_MS: u64 = 20;
pub const LOCK_TIMEOUT_SECONDS: f64 = 5.0;

/// An exclusive inter-process lock on a sidecar file, released on drop.
/// `held` is false when it could not be taken (see the rules above: the caller
/// carries on, because the write it guards is atomic either way).
pub struct FileLock {
    /// The open descriptor IS the lock. Dropping it releases, which is also
    /// what the kernel does for us if this process dies holding it.
    file: Option<std::fs::File>,
    pub held: bool,
}

impl FileLock {
    /// Take the lock, waiting up to `LOCK_TIMEOUT_SECONDS`.
    pub fn acquire(lock_path: &Path) -> FileLock {
        Self::open(lock_path, LOCK_TIMEOUT_SECONDS)
    }

    /// Take the lock only if it is free right now. For work that another
    /// process doing the same work makes unnecessary (log rotation), where
    /// waiting would be latency spent to do nothing.
    pub fn try_acquire(lock_path: &Path) -> FileLock {
        Self::open(lock_path, 0.0)
    }

    fn open(lock_path: &Path, timeout: f64) -> FileLock {
        if let Some(parent) = lock_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // create (never create_new): the file's existence is not the lock, so a
        // file left by any earlier run is simply the lock we take now.
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        #[cfg(windows)]
        {
            // On Windows the exclusive OPEN is the lock: share_mode(0) is
            // granted only while no other handle is open on the file, in this
            // process or any other, and the OS closes the handle however the
            // process ends. Same per-handle semantics as flock, same
            // release-on-death, and — as on unix — the file itself is left in
            // place, carrying nothing.
            use std::os::windows::fs::OpenOptionsExt;
            opts.share_mode(0);
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
        loop {
            #[cfg(unix)]
            {
                let file = match opts.open(lock_path) {
                    Ok(f) => f,
                    Err(_) => return FileLock { file: None, held: false },
                };
                use std::os::unix::io::AsRawFd;
                // SAFETY: plain flock(2) on a descriptor we own for the lock's lifetime.
                if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                    return FileLock { file: Some(file), held: true };
                }
                match std::io::Error::last_os_error().raw_os_error() {
                    // Held elsewhere: the one case worth waiting out.
                    Some(e) if e == libc::EWOULDBLOCK => {}
                    // A signal landed mid-call; it says nothing about the lock.
                    Some(e) if e == libc::EINTR => continue,
                    // EBADF, or a filesystem with no flock at all. Retrying
                    // cannot change the answer.
                    _ => return FileLock { file: None, held: false },
                }
            }
            #[cfg(not(unix))]
            {
                if let Ok(file) = opts.open(lock_path) {
                    return FileLock { file: Some(file), held: true };
                }
            }
            if std::time::Instant::now() >= deadline {
                return FileLock { file: None, held: false };
            }
            std::thread::sleep(std::time::Duration::from_millis(LOCK_POLL_MS));
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                // Closing the descriptor releases it anyway; this is explicit
                // so the release is visible at the point it happens.
                unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
            }
            drop(file); // the close is the release
        }
        self.held = false;
    }
}

/// Read a shared JSON document, apply `mutate`, write it back atomically.
///
/// The whole read-modify-write runs under `lock_path`, so a second process
/// cannot read the same bytes and overwrite the change with its own view. This
/// is the one way the shared documents in the config dir are written; see the
/// rules above. `mutate` receives the parsed document — None when the file does
/// not exist yet — and returns what to write, or None to write nothing. Returns
/// whether the file was written.
///
/// A file that EXISTS but cannot be read is never overwritten, and `mutate` is
/// not called at all: the caller gets false. Missing and unreadable are not the
/// same answer, and treating them alike is how a one-character typo in a
/// hand-edited settings.json would be "fixed" by replacing the user's hooks and
/// permissions with whatever this process happened to want to add. config.json
/// has refused to do that since it learned the lesson (see config's load_failed);
/// this is that rule for every shared document.
pub fn update_json_document<F>(
    path: &Path,
    lock_path: &Path,
    mutate: F,
    trailing_newline: bool,
) -> bool
where
    F: FnOnce(Option<serde_json::Value>) -> Option<serde_json::Value>,
{
    let _lock = FileLock::acquire(lock_path);
    let current = match std::fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => Some(v),
            Err(_) => return false,
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => return false,
    };
    let Some(updated) = mutate(current) else {
        return false;
    };
    let Ok(mut text) = serde_json::to_string_pretty(&updated) else {
        return false;
    };
    if trailing_newline {
        text.push('\n');
    }
    atomic_write_bytes(&path.to_string_lossy(), text.as_bytes()).is_ok()
}
