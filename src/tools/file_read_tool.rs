//! FileRead Tool for CodeMason
//! Equivalent to codemason/tools/file_read_tool.py — based on
//! src/tools/FileReadTool/FileReadTool.ts (name = 'Read') from Claude Code.
//!
//! Ported faithfully: input schema (verbatim from schemas/tool_schemas.json),
//! READ_DESCRIPTION reuse, cat -n line format (compact prefix: number + tab),
//! MAX_LINES default 2000, offset/limit slicing, the empty / short-offset
//! <system-reminder> warnings, notebook cell rendering, and the same-range
//! read dedup (FILE_UNCHANGED_STUB).
//!
//! SUBSTRATE BOUNDARY (local model may not be multimodal): images / PDFs
//! return a faithful TEXT metadata representation instead of multimodal
//! blocks, exactly as the Python port did.
//!
//! KEPT: population of `context.read_file_state[file_path]` after resolving
//! the path — Edit/Write/NotebookEdit read-before-edit enforcement depends
//! on it.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::{json, Value};

use crate::constants::tool_prompts::READ_DESCRIPTION;
use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ReadFileState, ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::state::get_cwd;

pub const FILE_READ_TOOL_NAME: &str = "Read";

/// limits.ts / prompt.ts
pub const MAX_LINES_TO_READ: usize = 2000;
/// 0.25 MB — utils/file.ts MAX_OUTPUT_SIZE (the maxSizeBytes cap on a full read).
pub const MAX_OUTPUT_SIZE: u64 = (0.25 * 1024.0 * 1024.0) as u64;

/// limits.ts DEFAULT_MAX_OUTPUT_TOKENS + the env override (env > default).
pub const DEFAULT_MAX_OUTPUT_TOKENS: i64 = 25000;

/// prompt.ts FILE_UNCHANGED_STUB (verbatim) — returned by the same-range dedup.
pub const FILE_UNCHANGED_STUB: &str = "File unchanged since last read. The content from the earlier Read tool_result in this conversation is still current — refer to that instead of re-reading.";

/// services/tokenEstimation.ts bytesPerTokenForFileType: JSON family packs ~2
/// bytes/token, everything else ~4.
const JSON_EXTS: &[&str] = &["json", "jsonl", "jsonc"];
/// utils/pdfUtils.ts — PDF_MAX_PAGES_PER_READ
pub const PDF_MAX_PAGES_PER_READ: i64 = 20;

/// utils/notebook.ts LARGE_OUTPUT_THRESHOLD
pub const LARGE_OUTPUT_THRESHOLD: usize = 10000;

pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];
/// [ours] An image file larger than this is not read into memory to be
/// attached. Phone photos are a few MB and the phone channel stops at 20 MB.
pub const MAX_IMAGE_FILE_BYTES: u64 = 50 * 1024 * 1024;

/// constants/files.ts hasBinaryExtension — binary extensions this tool refuses
/// (images / PDF excluded; they render via the substrate metadata path below).
pub const BINARY_EXTENSIONS: &[&str] = &[
    "mp3", "mp4", "avi", "mkv", "mov", "wav", "flac", "ogg", "m4a", "wmv", "flv",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
    "exe", "dll", "so", "dylib", "o", "a", "pyc", "pyo", "class", "jar", "wasm",
    "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "ttf", "otf", "woff", "woff2", "eot",
    "bin", "dat", "db", "sqlite", "sqlite3",
];

// === Embedded tool schemas (shared helper) ===
//
// schemas/tool_schemas.json is the EXACT input_json_schema() output dumped
// from the running Python implementation — embedding it whole guarantees
// byte-parity of what the model sees. Other tool modules use this lookup too.

static TOOL_SCHEMAS: Lazy<Value> = Lazy::new(|| {
    serde_json::from_str(include_str!("../../schemas/tool_schemas.json"))
        .unwrap_or(Value::Null)
});

/// The verbatim input schema for `name` from the Python dump.
pub(crate) fn embedded_tool_schema(name: &str) -> Value {
    TOOL_SCHEMAS
        .get(name)
        .and_then(|t| t.get("schema"))
        .cloned()
        .unwrap_or_else(|| json!({}))
}

// === Helpers (shared with the other file tools) ===

/// expandPath equivalent: make absolute relative to cwd.
pub(crate) fn resolve_path(file_path: &str) -> String {
    if std::path::Path::new(file_path).is_absolute() {
        file_path.to_string()
    } else {
        std::path::Path::new(&get_cwd()).join(file_path).to_string_lossy().into_owned()
    }
}

/// Python os.path.splitext-style extension, lowercased, without the dot.
/// ('.bashrc' -> "", 'a.tar.gz' -> "gz", 'file.' -> "").
pub(crate) fn ext_of(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match base.rfind('.') {
        Some(i) if i > 0 => base[i + 1..].to_lowercase(),
        _ => String::new(),
    }
}

/// File mtime in SECONDS as f64 (the readFileState timestamp unit).
pub(crate) fn mtime_seconds(path: &str) -> Option<f64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs_f64())
}

/// FileReadTool.ts isBlockedDevicePath — path-only (no I/O). Reading these
/// hangs the process or spins forever; /dev/null etc. are intentionally NOT
/// blocked.
const BLOCKED_DEVICE_PATHS: &[&str] = &[
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
];

fn is_blocked_device_path(path: &str) -> bool {
    if BLOCKED_DEVICE_PATHS.contains(&path) {
        return true;
    }
    // /proc/self/fd/{0,1,2} and /proc/<pid>/fd/{0,1,2} are stdio aliases.
    path.starts_with("/proc/")
        && (path.ends_with("/fd/0") || path.ends_with("/fd/1") || path.ends_with("/fd/2"))
}

struct PageRange {
    first_page: i64,
    /// None = open-ended ("N-")
    last_page: Option<i64>,
}

/// parsePDFPageRange — supports "N", "N-M", "N-" (open-ended). 1-indexed.
fn parse_pdf_page_range(pages: &str) -> Option<PageRange> {
    let pages = pages.trim();
    if pages.is_empty() {
        return None;
    }
    if let Some((first_s, last_s)) = pages.split_once('-') {
        let first_s = first_s.trim();
        let last_s = last_s.trim();
        if first_s.is_empty() || !first_s.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let first: i64 = first_s.parse().ok()?;
        let last: Option<i64> = if last_s.is_empty() {
            None
        } else if last_s.chars().all(|c| c.is_ascii_digit()) {
            Some(last_s.parse().ok()?)
        } else {
            return None;
        };
        if first < 1 || matches!(last, Some(l) if l < first) {
            return None;
        }
        return Some(PageRange { first_page: first, last_page: last });
    }
    if !pages.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let n: i64 = pages.parse().ok()?;
    if n < 1 {
        return None;
    }
    Some(PageRange { first_page: n, last_page: Some(n) })
}

/// limits.ts getDefaultFileReadingLimits().maxTokens (env > default).
fn get_max_output_tokens() -> i64 {
    if let Ok(v) = std::env::var("CODEMASON_FILE_READ_MAX_OUTPUT_TOKENS") {
        if !v.is_empty() {
            if let Ok(n) = v.parse::<i64>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    DEFAULT_MAX_OUTPUT_TOKENS
}

/// utils/format.formatFileSize (KB/MB/GB).
pub(crate) fn fmt_size(num_bytes: u64) -> String {
    let mut size = num_bytes as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size < 1024.0 || unit == "TB" {
            if unit == "B" {
                return format!("{}{unit}", size as u64);
            }
            return format!("{size:.1}{unit}");
        }
        size /= 1024.0;
    }
    format!("{num_bytes}B")
}

/// validateContentTokens (FileReadTool.ts:755). Fast-path on a rough
/// bytes/token estimate; only past maxTokens/4 do we pay for a real token
/// count. Returns the MaxFileReadTokenExceededError message when the content
/// exceeds the cap, else None.
fn validate_content_tokens(content: &str, ext: &str) -> Option<String> {
    let max_tokens = get_max_output_tokens();
    let bytes_per_token = if JSON_EXTS.contains(&ext) { 2.0 } else { 4.0 };
    let estimate = (content.chars().count() as f64 / bytes_per_token).round() as i64;
    if estimate == 0 || estimate as f64 <= max_tokens as f64 / 4.0 {
        return None;
    }
    // countTokensWithAPI substrate: an exact local count — the served model's
    // own tokenizer when available, else the CJK-aware heuristic.
    let count = crate::utils::tokens::count_text_tokens(content);
    let count = if count > 0 { count } else { estimate };
    if count > max_tokens {
        return Some(format!(
            "File content ({count} tokens) exceeds maximum allowed tokens ({max_tokens}). Use offset and limit parameters to read specific portions of the file, or search for specific content instead of reading the whole file."
        ));
    }
    None
}

/// addLineNumbers (compact-prefix path): "<lineno>\t<line>" — cat -n format.
/// NOTE: this is the READ tool's own compact form, NOT utils::file's 6-wide
/// variant — the two differ in Python too.
/// [ours - not in upstream] The notice a short read owes the model.
///
/// A full read stops at MAX_LINES_TO_READ and the original says nothing about
/// it, so the model takes the first 2000 lines of a longer file for the whole
/// file and reasons from a view it does not know is partial. The tool already
/// holds the real count; this hands it over. It costs nothing in the fixed
/// prefix - it rides on the tool result, only when a read actually stopped
/// short - which is why the fix belongs here and not in the Read description.
///
/// total_lines follows str.split('\n'), so a file ending in a newline carries
/// one phantom empty element; it is dropped before comparing, or every such
/// file would report a missing line.
fn truncation_notice(
    offset: usize,
    num_lines: usize,
    total_lines: usize,
    ended_with_newline: bool,
) -> Option<String> {
    let file_lines = if ended_with_newline { total_lines.saturating_sub(1) } else { total_lines };
    let last_line = offset.saturating_sub(1) + num_lines;
    if last_line >= file_lines {
        return None;
    }
    Some(format!(
        "\n\n<system-reminder>Showed lines {}-{} of {}. {} more lines were not returned — call Read again with offset={} to continue.</system-reminder>",
        offset, last_line, file_lines, file_lines - last_line, last_line + 1
    ))
}

fn add_line_numbers_compact(content: &str, start_line: usize) -> String {
    if content.is_empty() {
        return String::new();
    }
    content
        .split('\n')
        .enumerate()
        .map(|(i, line)| format!("{}\t{line}", i + start_line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Decode file bytes with the detected encoding, errors replaced.
/// utf-8-sig strips its BOM (Python's utf-8-sig codec does); the utf-16
/// codecs DO NOT — Python's utf-16-le decodes FF FE to U+FEFF and keeps it
/// in the string, so an edit re-encodes it and the file keeps its BOM.
/// (This port used to strip it here and never re-add it: one Edit turned a
/// BOM'd UTF-16-LE file into something the next read misdetected as utf-8.)
pub(crate) fn decode_bytes(bytes: &[u8], enc: &str) -> String {
    match enc {
        "utf-16-le" => {
            let units: Vec<u16> = bytes
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        "utf-16-be" => {
            let units: Vec<u16> = bytes
                .chunks(2)
                .map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]))
                .collect();
            String::from_utf16_lossy(&units)
        }
        "utf-8-sig" => {
            let data = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(bytes);
            String::from_utf8_lossy(data).into_owned()
        }
        // Byte-preserving: every byte maps to the code point of the same value,
        // so an undecodable file round-trips unchanged through decode + encode.
        // Without this arm the fallback below replaced each bad byte with
        // U+FFFD, and Edit wrote that back over the original.
        "latin-1" => bytes.iter().map(|&b| b as char).collect(),
        // A legacy label only reaches here when the caller detected one, which
        // means verified_legacy_encoding PROVED it round-trips — so this decode
        // is exact rather than best-effort. (This comment named
        // detect_file_encoding while the only caller was still using a private
        // BOM sniff that could never produce a legacy label, which is how that
        // gap survived a review: the comment described the intent, not the
        // wiring.)
        other => match encoding_rs::Encoding::for_label(other.as_bytes()) {
            Some(enc) => enc.decode(bytes).0.into_owned(),
            None => String::from_utf8_lossy(bytes).into_owned(),
        },
    }
}

/// Stream `path`, returning (selected_lines, total_lines, ended_with_newline)
/// with ONLY the
/// [line_offset, end_line) window held in memory — the rest is counted, not
/// retained. This is the ranged-read path (limit set): the byte-size cap is
/// intentionally skipped there, so without streaming a `Read(hugefile,
/// limit=N)` would materialize the whole file. Line semantics reproduce
/// str.split('\n') EXACTLY, including the trailing empty element a final
/// newline produces, so total_lines matches the full-read path.
///
/// Substrate note: encodings other than plain utf-8 are decoded whole then
/// windowed (a UTF-16 file can't be split on raw b'\n' safely); utf-8 streams
/// line-by-line as the Python did.
fn read_line_window(
    path: &str,
    enc: &str,
    line_offset: usize,
    end_line: usize,
) -> std::io::Result<(Vec<String>, usize, bool)> {
    use std::io::BufRead;

    let mut selected: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut ended_with_newline = false;
    let mut first = true;

    let mut push_line = |raw: &str, is_newline_terminated: bool, first: &mut bool,
                         total: &mut usize, selected: &mut Vec<String>| {
        let mut s = raw.to_string();
        if *first {
            if let Some(rest) = s.strip_prefix('\u{feff}') {
                s = rest.to_string();
            }
        }
        *first = false;
        if line_offset <= *total && *total < end_line {
            if is_newline_terminated && s.ends_with('\r') {
                s.pop();
            }
            selected.push(s);
        }
        *total += 1;
    };

    if enc == "utf-8" || enc == "utf-8-sig" {
        let f = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(f);
        let mut buf: Vec<u8> = Vec::new();
        loop {
            buf.clear();
            let n = reader.read_until(b'\n', &mut buf)?;
            if n == 0 {
                break;
            }
            let had_nl = buf.last() == Some(&b'\n');
            if had_nl {
                buf.pop();
            }
            ended_with_newline = had_nl;
            let s = String::from_utf8_lossy(&buf).into_owned();
            push_line(&s, had_nl, &mut first, &mut total, &mut selected);
        }
    } else {
        let bytes = std::fs::read(path)?;
        let text = decode_bytes(&bytes, enc);
        ended_with_newline = text.ends_with('\n');
        let body = text.strip_suffix('\n').unwrap_or(&text);
        if !text.is_empty() {
            for line in body.split('\n') {
                push_line(line, true, &mut first, &mut total, &mut selected);
            }
        }
    }

    // Reproduce str.split('\n'): '' -> [''] (1), and a trailing '\n' yields one
    // extra empty element beyond the counted lines.
    if total == 0 {
        total = 1;
        if line_offset == 0 && end_line > 0 {
            selected.push(String::new());
        }
    } else if ended_with_newline {
        if line_offset <= total && total < end_line {
            selected.push(String::new());
        }
        total += 1;
    }
    Ok((selected, total, ended_with_newline))
}

// --- Notebook (faithful port of utils/notebook.ts) ---

fn process_output_text(text: Option<&Value>) -> String {
    match text {
        None | Some(Value::Null) => String::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()))
            .collect(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn extract_image(data: &Value) -> Option<Value> {
    if data.get("image/png").map(|v| v.is_string()).unwrap_or(false) {
        return Some(json!({"media_type": "image/png"}));
    }
    if data.get("image/jpeg").map(|v| v.is_string()).unwrap_or(false) {
        return Some(json!({"media_type": "image/jpeg"}));
    }
    None
}

fn process_output(output: &Value) -> Option<Value> {
    let ot = output.get("output_type").and_then(|v| v.as_str())?;
    match ot {
        "stream" => Some(json!({
            "output_type": ot,
            "text": process_output_text(output.get("text")),
        })),
        "execute_result" | "display_data" => {
            let data = output.get("data").cloned().unwrap_or_else(|| json!({}));
            Some(json!({
                "output_type": ot,
                "text": process_output_text(data.get("text/plain")),
                "image": extract_image(&data),
            }))
        }
        "error" => {
            let traceback = output
                .get("traceback")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            let ename = output.get("ename").map(py_str_of).unwrap_or_else(|| "None".into());
            let evalue = output.get("evalue").map(py_str_of).unwrap_or_else(|| "None".into());
            let text = format!("{ename}: {evalue}\n{traceback}");
            Some(json!({"output_type": ot, "text": text}))
        }
        _ => None,
    }
}

/// Python str() of a JSON scalar (None for null), used in the error-output
/// f-string above.
fn py_str_of(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn is_large_outputs(outputs: &[Option<Value>]) -> bool {
    let mut size = 0usize;
    for o in outputs.iter().flatten() {
        size += o.get("text").and_then(|t| t.as_str()).map(|s| s.len()).unwrap_or(0);
        if size > LARGE_OUTPUT_THRESHOLD {
            return true;
        }
    }
    false
}

fn process_cell(cell: &Value, index: usize, code_language: &str) -> Value {
    let cell_type = cell.get("cell_type").cloned().unwrap_or(Value::Null);
    let source = match cell.get("source") {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()))
            .collect::<String>(),
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    };
    let cell_id = cell
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("cell-{index}"));
    let mut cell_data = json!({
        "cellType": cell_type,
        "source": source,
        "cell_id": cell_id,
    });
    if cell.get("cell_type").and_then(|v| v.as_str()) == Some("code") {
        cell_data["language"] = json!(code_language);
        let outs = cell.get("outputs").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        if !outs.is_empty() {
            let processed: Vec<Option<Value>> = outs.iter().map(process_output).collect();
            if is_large_outputs(&processed) {
                cell_data["outputs"] = json!([{
                    "output_type": "stream",
                    "text": format!("Outputs are too large to include. Use Bash with: cat <notebook_path> | jq '.cells[{index}].outputs'"),
                }]);
            } else {
                let kept: Vec<Value> = processed.into_iter().flatten().collect();
                cell_data["outputs"] = json!(kept);
            }
        }
    }
    cell_data
}

/// Faithful text rendering of cellContentToToolResult + cellOutputToToolResult
/// (image blocks become a metadata placeholder, per the substrate boundary).
fn render_notebook_cells(cells: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for cell in cells {
        let cell_id = cell.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
        let cell_type = cell.get("cellType").and_then(|v| v.as_str()).unwrap_or("");
        let language = cell.get("language").and_then(|v| v.as_str());
        let mut metadata = String::new();
        if cell_type != "code" {
            metadata.push_str(&format!("<cell_type>{cell_type}</cell_type>"));
        }
        if let Some(lang) = language {
            if !lang.is_empty() && lang != "python" && cell_type == "code" {
                metadata.push_str(&format!("<language>{lang}</language>"));
            }
        }
        let source = cell.get("source").and_then(|v| v.as_str()).unwrap_or("");
        let mut block =
            format!("<cell id=\"{cell_id}\">{metadata}{source}</cell id=\"{cell_id}\">");
        if let Some(outputs) = cell.get("outputs").and_then(|v| v.as_array()) {
            for output in outputs {
                if let Some(text) = output.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        block.push_str("\n\n");
                        block.push_str(text);
                    }
                }
                if let Some(image) = output.get("image") {
                    if !image.is_null() {
                        let mt = image.get("media_type").and_then(|v| v.as_str()).unwrap_or("");
                        block.push_str(&format!("\n[Image output: {mt}]"));
                    }
                }
            }
        }
        parts.push(block);
    }
    parts.join("\n")
}

/// Best-effort PDF page count without external deps (count /Type /Page).
/// Scans at most the first 64 MiB.
fn pdf_page_count(path: &str) -> Option<i64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut data = vec![0u8; 64 * 1024 * 1024];
    let mut read_total = 0usize;
    loop {
        match f.read(&mut data[read_total..]) {
            Ok(0) => break,
            Ok(n) => read_total += n,
            Err(_) => return None,
        }
        if read_total == data.len() {
            break;
        }
    }
    data.truncate(read_total);
    static PAGE_RE: Lazy<regex::bytes::Regex> =
        Lazy::new(|| regex::bytes::Regex::new(r"/Type\s*/Page[^s]").unwrap());
    let n = PAGE_RE.find_iter(&data).count() as i64;
    if n == 0 {
        None
    } else {
        Some(n)
    }
}

// === FileRead Tool ===

/// Read a file from the local filesystem (text, image meta, PDF meta, notebook).
pub struct FileReadTool;

fn opt_i64(input: &Value, key: &str) -> Option<i64> {
    input.get(key).and_then(|v| v.as_i64())
}

fn opt_str<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(|v| v.as_str())
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &'static str {
        FILE_READ_TOOL_NAME
    }

    /// Infinity = never persist (FileReadTool.maxResultSizeChars). Read
    /// self-bounds via its own limits, and persisting its output to a file the
    /// model reads back with Read is circular.
    fn max_result_size_chars(&self) -> f64 {
        f64::INFINITY
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(FILE_READ_TOOL_NAME)
    }

    async fn description(&self, _input: &Value) -> String {
        READ_DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        READ_DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Read".to_string()
    }

    /// Path, plus the range when one was asked for (FileReadTool/UI.tsx).
    /// The verb lives in the header ([Read]) and the spinner, so the body is
    /// the argument alone.
    /// The display path (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("file_path")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::file::get_display_path(value))
    }

    /// "Reading <path>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Reading {summary}"),
            None => "Reading file".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let file_path = opt_str(input, "file_path").unwrap_or("");
        if file_path.is_empty() {
            return String::new();
        }
        let mut parts = vec![file_path.to_string()];
        let offset = opt_i64(input, "offset");
        let limit = opt_i64(input, "limit");
        let pages = opt_str(input, "pages").filter(|s| !s.is_empty());
        if let Some(p) = pages {
            parts.push(format!("pages {p}"));
        } else if offset.map(|v| v != 0).unwrap_or(false) || limit.map(|v| v != 0).unwrap_or(false)
        {
            let start = offset.filter(|v| *v != 0).unwrap_or(1);
            match limit.filter(|v| *v != 0) {
                Some(l) => parts.push(format!("lines {start}+{l}")),
                None => parts.push(format!("from line {start}")),
            }
        }
        colorize(&parts.join(" · "), colors::DIM)
    }

    /// One line — what was read, not the file's first few lines
    /// (FileReadTool/UI.tsx). Ctrl+O still expands to the full text.
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        if !data.is_object() || data.get("error").map(|e| !e.is_null()).unwrap_or(false) {
            return None;
        }
        match data.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                let n = data.get("num_lines").and_then(|v| v.as_i64()).unwrap_or(0);
                Some(format!("Read {n} line{}", if n == 1 { "" } else { "s" }))
            }
            Some("file_unchanged") => Some(colorize("Unchanged since last read", colors::DIM)),
            Some("notebook") => {
                let n = data.get("num_cells").and_then(|v| v.as_i64()).unwrap_or(0);
                Some(format!("Read {n} cell{}", if n == 1 { "" } else { "s" }))
            }
            Some("image") => {
                let size = data.get("original_size").and_then(|v| v.as_u64()).unwrap_or(0);
                Some(format!("Read image ({})", fmt_size(size)))
            }
            Some("pdf") => {
                let n = data.get("page_count").and_then(|v| v.as_i64()).unwrap_or(0);
                let size = data.get("original_size").and_then(|v| v.as_u64()).unwrap_or(0);
                Some(format!(
                    "Read PDF ({}, {n} page{})",
                    fmt_size(size),
                    if n == 1 { "" } else { "s" }
                ))
            }
            _ => None,
        }
    }

    /// Faithful subset of FileReadTool.validateInput (pages parse + binary check).
    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let file_path = opt_str(input, "file_path").unwrap_or("");
        let pages = input.get("pages").filter(|v| !v.is_null()).and_then(|v| v.as_str());

        if file_path.is_empty() {
            return ValidationResult::failure("file_path is required");
        }

        // pages parameter (pure string parsing, no I/O).
        if let Some(pages) = pages {
            match parse_pdf_page_range(pages) {
                None => {
                    return ValidationResult::failure(format!(
                        "Invalid pages parameter: \"{pages}\". Use formats like \"1-5\", \"3\", or \"10-20\". Pages are 1-indexed."
                    ));
                }
                Some(parsed) => {
                    let range_size = match parsed.last_page {
                        None => PDF_MAX_PAGES_PER_READ + 1,
                        Some(last) => last - parsed.first_page + 1,
                    };
                    if range_size > PDF_MAX_PAGES_PER_READ {
                        return ValidationResult::failure(format!(
                            "Page range \"{pages}\" exceeds maximum of {PDF_MAX_PAGES_PER_READ} pages per request. Please use a smaller range."
                        ));
                    }
                }
            }
        }

        let full_path = resolve_path(file_path);

        // Block infinite/blocking device files (would hang or OOM the process).
        if is_blocked_device_path(&full_path) {
            return ValidationResult::failure(format!("Cannot read device file: {full_path}"));
        }

        let ext = ext_of(&full_path);

        // Binary extension check (PDF + images excluded — rendered via metadata).
        if BINARY_EXTENSIONS.contains(&ext.as_str()) {
            return ValidationResult::failure(format!(
                "This tool cannot read binary files. The file appears to be a binary .{ext} file. Please use appropriate tools for binary file analysis."
            ));
        }

        ValidationResult::success()
    }

    /// Read the file. Dispatches: notebook / image / pdf / text.
    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let file_path = opt_str(&input, "file_path").unwrap_or("").to_string();
            let raw_offset = opt_i64(&input, "offset");
            let raw_limit = opt_i64(&input, "limit");
            let offset = raw_offset.unwrap_or(1);
            let limit = raw_limit;
            let pages = opt_str(&input, "pages").map(|s| s.to_string());

            let full_path = resolve_path(&file_path);
            let ext = ext_of(&full_path);

            if !std::path::Path::new(&full_path).exists() {
                let sim = crate::utils::file::find_similar_file(&full_path);
                let hint = sim.map(|s| format!(" Did you mean {s}?")).unwrap_or_default();
                yield Ok(ToolResult::Result {
                    result_for_assistant: format!(
                        "File does not exist. Current working directory is {}.{hint}",
                        get_cwd()
                    ),
                    data: Some(json!({"error": "ENOENT", "path": full_path})),
                });
                return;
            }

            if std::path::Path::new(&full_path).is_dir() {
                yield Ok(ToolResult::Result {
                    result_for_assistant: format!(
                        "EISDIR: illegal operation on a directory, read '{full_path}'. Use an ls command via Bash to read a directory."
                    ),
                    data: Some(json!({"error": "EISDIR", "path": full_path})),
                });
                return;
            }

            // The original Read stores the DEFAULTED offset (=1 for a full
            // read), so a Read entry always has a defined offset >= 1 and is
            // therefore never `is_full_read` — that flag is the marker
            // Edit/Write set on their post-write entries (the content-equality
            // fallback keys on it). Keeping this faithful means the dedup below
            // can distinguish Read entries (offset set) from Edit/Write entries
            // (offset None).
            let eff_offset = raw_offset.unwrap_or(1);

            // Read dedup (FileReadTool.ts:540-572): if the SAME range of this
            // file was already Read and its mtime is unchanged, return
            // FILE_UNCHANGED_STUB instead of a second full copy. Only
            // text/notebook reads dedup; images/PDFs aren't cached. Exclude
            // Edit/Write entries (offset None) and partial-view entries.
            let existing = context.read_file_state.lock().get(&full_path).cloned();
            if let Some(existing) = existing {
                if !existing.is_partial_view
                    && existing.offset.is_some()
                    && existing.offset == Some(eff_offset)
                    && existing.limit == raw_limit
                    && !IMAGE_EXTENSIONS.contains(&ext.as_str())
                    && ext != "pdf"
                {
                    if let Some(mtime) = mtime_seconds(&full_path) {
                        if mtime == existing.timestamp {
                            yield Ok(ToolResult::Result {
                                result_for_assistant: FILE_UNCHANGED_STUB.to_string(),
                                data: Some(json!({"type": "file_unchanged", "path": full_path})),
                            });
                            return;
                        }
                    }
                    // stat failed — fall through to a full read
                }
            }

            // KEEP: populate the readFileState substrate so read-before-edit +
            // modified-since-read checks work. content is filled in by the text
            // path below for full reads (the equality fallback);
            // image/pdf/notebook reads leave content "".
            if let Some(mtime) = mtime_seconds(&full_path) {
                context.read_file_state.lock().insert(full_path.clone(), ReadFileState {
                    timestamp: mtime,
                    content: String::new(),
                    offset: Some(eff_offset),
                    limit: raw_limit,
                    is_partial_view: false,
                });
            }

            // --- Notebook ---
            if ext == "ipynb" {
                let text = match std::fs::read(&full_path) {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    Err(e) => {
                        yield Ok(ToolResult::Result {
                            result_for_assistant: format!("Error reading file: {e}"),
                            data: Some(json!({"error": e.to_string()})),
                        });
                        return;
                    }
                };
                let notebook: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        yield Ok(ToolResult::Result {
                            result_for_assistant: format!("Error reading file: {e}"),
                            data: Some(json!({"error": e.to_string()})),
                        });
                        return;
                    }
                };
                let language = notebook
                    .get("metadata")
                    .and_then(|m| m.get("language_info"))
                    .and_then(|l| l.get("name"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("python")
                    .to_string();
                let raw_cells = notebook.get("cells").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let cells: Vec<Value> = raw_cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| process_cell(c, i, &language))
                    .collect();
                let result = render_notebook_cells(&cells);
                let num_cells = cells.len();
                yield Ok(ToolResult::Result {
                    result_for_assistant: result,
                    data: Some(json!({"type": "notebook", "path": full_path,
                                      "cells": cells, "num_cells": num_cells})),
                });
                return;
            }

            // --- Image: real image blocks when the server can see, else metadata ---
            if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                let size = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
                let limit = crate::services::vision::enabled_limit().await;
                let mut attach_error: Option<String> = None;
                if limit > 0 && size > MAX_IMAGE_FILE_BYTES {
                    attach_error = Some(format!(
                        "larger than {} MB; shrink or crop it first",
                        MAX_IMAGE_FILE_BYTES / (1024 * 1024)
                    ));
                } else if limit > 0 {
                    let raw = match std::fs::read(&full_path) {
                        Ok(b) => b,
                        Err(e) => {
                            yield Ok(ToolResult::Result {
                                result_for_assistant: format!("Error reading file: {e}"),
                                data: Some(json!({"error": e.to_string()})),
                            });
                            return;
                        }
                    };
                    // At most half the server's limit from one Read: that is the
                    // batch the oldest images are dropped in, so two Reads in one
                    // step still fit, and a long page cut into tiles cannot push
                    // out images the model has not looked at yet.
                    let slots = (limit / 2).max(1) as usize;
                    // Decode and Lanczos take a while on a big photo; not on an
                    // async worker thread.
                    let (raw, prepared) = tokio::task::spawn_blocking(move || {
                        let r = crate::utils::image_prep::prepare_image(&raw, slots);
                        (raw, r)
                    })
                    .await
                    .unwrap_or_else(|e| (Vec::new(), Err(e.to_string())));
                    match prepared {
                        Ok(prepared) if !prepared.is_empty() => {
                            use base64::Engine;
                            let original_copy = crate::utils::image_store::save_original(
                                &raw, &ext, None, Some(std::path::Path::new(&full_path)),
                            )
                            .display()
                            .to_string();
                            let p0 = &prepared[0];
                            let mut note = if prepared.len() == 1 {
                                format!("{}x{}", p0.width, p0.height)
                            } else {
                                format!(
                                    "cut into {} tiles of about {}x{}, top to bottom / left to right",
                                    prepared.len(), p0.width, p0.height
                                )
                            };
                            if p0.animated {
                                note.push_str(", first frame of an animation");
                            }
                            let images: Vec<Value> = prepared
                                .iter()
                                .map(|p| json!({"type": "image", "source": {
                                    "type": "base64", "media_type": p.media_type,
                                    "data": base64::engine::general_purpose::STANDARD.encode(&p.data)}}))
                                .collect();
                            let tokens: u32 = prepared.iter().map(|p| p.tokens).sum();
                            yield Ok(ToolResult::ResultWithImages {
                                result_for_assistant: format!(
                                    "[Image file: {full_path} ({size} bytes) — shown below, {note}]"
                                ),
                                images,
                                data: Some(json!({"type": "image", "path": full_path,
                                                  "media_type": p0.media_type, "original_size": size,
                                                  "images": prepared.len(), "tokens": tokens,
                                                  "original_copy": original_copy})),
                            });
                            return;
                        }
                        Ok(_) => attach_error = Some("no image in the file".to_string()),
                        Err(e) => attach_error = Some(e),
                    }
                }
                let result = match attach_error {
                    Some(err) => format!("[Image file: {full_path} ({size} bytes) could not be attached: {err}]"),
                    None => format!(
                        "[Image file: {full_path} ({size} bytes, image/{ext}). Multimodal image rendering is not available on this substrate; metadata only.]"
                    ),
                };
                yield Ok(ToolResult::Result {
                    result_for_assistant: result,
                    data: Some(json!({"type": "image", "path": full_path,
                                      "media_type": format!("image/{ext}"), "original_size": size})),
                });
                return;
            }

            // --- PDF (substrate: metadata text, not a document block) ---
            if ext == "pdf" {
                let size = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
                let page_count = pdf_page_count(&full_path);
                let pages_note = page_count.map(|n| format!(", {n} pages")).unwrap_or_default();
                let range_note = pages
                    .as_ref()
                    .filter(|p| !p.is_empty())
                    .map(|p| format!(", pages {p}"))
                    .unwrap_or_default();
                let result = format!(
                    "[PDF file: {full_path} ({size} bytes{pages_note}{range_note}). PDF document rendering is not available on this substrate; metadata only.]"
                );
                yield Ok(ToolResult::Result {
                    result_for_assistant: result,
                    data: Some(json!({"type": "pdf", "path": full_path,
                                      "original_size": size, "page_count": page_count})),
                });
                return;
            }

            // --- Text file (cat -n) ---
            // maxSizeBytes: a NO-LIMIT read of a file whose TOTAL size exceeds
            // the cap returns an error rather than truncating (upstream #21841:
            // the ~100-byte error beats ~25K tokens of clipped content).
            // Limited reads are exempt (limit set => no cap).
            if limit.is_none() {
                let fsize = std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(0);
                if fsize > MAX_OUTPUT_SIZE {
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!(
                            "File content ({}) exceeds maximum allowed size ({}). Use offset and limit parameters to read specific portions of the file, or search for specific content instead of reading the whole file.",
                            fmt_size(fsize), fmt_size(MAX_OUTPUT_SIZE)
                        ),
                        data: Some(json!({"error": "FileTooLarge", "path": full_path,
                                          "size": fsize, "max_size": MAX_OUTPUT_SIZE})),
                    });
                    return;
                }
            }
            // The SHARED detector, not a private BOM sniff. This function used
            // to do its own, documented as "matches _read_file_for_edit's
            // detection so Read and Edit agree" — and that stopped being true
            // the moment Edit moved to the shared one. The result: Edit could
            // read a CP949 file correctly while Read showed the same file as
            // U+FFFD, so the model could edit a file it could not read.
            let enc = crate::utils::file::detect_file_encoding(&full_path);
            let enc: &str = &enc;
            let line_offset = if offset == 0 { 0 } else { (offset - 1).max(0) } as usize;
            let end_line = line_offset + limit.map(|l| l.max(0) as usize).unwrap_or(MAX_LINES_TO_READ);

            let selected: Vec<String>;
            let total_lines: usize;
            let ended_with_newline: bool;
            let mut file_is_empty = false;

            if limit.is_none() {
                // Full read — already size-capped above. Materialize so the
                // full content can be recorded for the modified-since-read
                // equality fallback, then window it.
                let bytes = match std::fs::read(&full_path) {
                    Ok(b) => b,
                    Err(e) => {
                        yield Ok(ToolResult::Result {
                            result_for_assistant: format!("Error reading file: {e}"),
                            data: Some(json!({"error": e.to_string()})),
                        });
                        return;
                    }
                };
                let mut raw = decode_bytes(&bytes, enc);
                // Strip BOM (readFileInRange does this; utf-8-sig/utf-16 already do).
                if let Some(rest) = raw.strip_prefix('\u{feff}') {
                    raw = rest.to_string();
                }
                if raw_offset.is_none() && raw_limit.is_none() {
                    if let Some(st) = context.read_file_state.lock().get_mut(&full_path) {
                        st.content = raw.clone();
                    }
                }
                // Split lines, strip \r (CRLF -> LF), select [offset, offset+limit).
                let all_lines: Vec<&str> = raw.split('\n').collect();
                total_lines = all_lines.len();
                ended_with_newline = all_lines.last() == Some(&"");
                file_is_empty = total_lines == 1 && all_lines == [""];
                selected = all_lines
                    .iter()
                    .skip(line_offset)
                    .take(end_line.saturating_sub(line_offset))
                    .map(|ln| ln.strip_suffix('\r').unwrap_or(ln).to_string())
                    .collect();
            } else {
                // Ranged read: stream only the requested window so a huge file
                // is never fully loaded — the byte cap above does not apply.
                match read_line_window(&full_path, enc, line_offset, end_line) {
                    Ok((sel, total, ends_nl)) => {
                        selected = sel;
                        total_lines = total;
                        ended_with_newline = ends_nl;
                        // DIVERGENCE (documented): the Python's empty-content
                        // branch referenced `all_lines` here, which is unbound
                        // on the ranged path — a latent NameError ("Error
                        // reading file: name 'all_lines' is not defined") for a
                        // ranged read of an empty/1-line file. The INTENDED
                        // check is "is the file empty"; we implement the intent
                        // (size == 0) instead of reproducing the crash.
                        file_is_empty = total_lines == 1
                            && std::fs::metadata(&full_path).map(|m| m.len()).unwrap_or(1) == 0;
                    }
                    Err(e) => {
                        yield Ok(ToolResult::Result {
                            result_for_assistant: format!("Error reading file: {e}"),
                            data: Some(json!({"error": e.to_string()})),
                        });
                        return;
                    }
                }
            }
            let content = selected.join("\n");

            if content.is_empty() {
                // mapToolResultToToolResultBlockParam empty / short-offset warning.
                let warning = if total_lines == 0 || file_is_empty {
                    "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>".to_string()
                } else {
                    format!(
                        "<system-reminder>Warning: the file exists but is shorter than the provided offset ({offset}). The file has {total_lines} lines.</system-reminder>"
                    )
                };
                yield Ok(ToolResult::Result {
                    result_for_assistant: warning,
                    data: Some(json!({"type": "text", "path": full_path,
                                      "total_lines": total_lines, "num_lines": 0,
                                      "start_line": offset})),
                });
                return;
            }

            // maxTokens (validateContentTokens): a read whose token count
            // exceeds the 25k cap returns an error telling the model to use
            // offset/limit. Validated on the selected content, like the original.
            if let Some(token_error) = validate_content_tokens(&content, &ext) {
                yield Ok(ToolResult::Result {
                    result_for_assistant: token_error,
                    data: Some(json!({"error": "MaxFileReadTokenExceeded", "path": full_path})),
                });
                return;
            }

            let mut result = add_line_numbers_compact(&content, offset.max(0) as usize);
            let num_lines = selected.len();
            if let Some(note) =
                truncation_notice(offset.max(0) as usize, num_lines, total_lines, ended_with_newline)
            {
                result.push_str(&note);
            }

            yield Ok(ToolResult::Result {
                result_for_assistant: result,
                data: Some(json!({"type": "text", "path": full_path,
                                  "total_lines": total_lines, "num_lines": num_lines,
                                  "start_line": offset})),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read and Edit must agree about a file's encoding. They did not: Edit
    /// moved to the shared detector while Read kept a private BOM sniff that
    /// could only ever answer utf-8, so the model could EDIT a CP949 file
    /// correctly and still see U+FFFD when it READ the same file.
    #[test]
    fn read_sees_the_same_encoding_edit_does() {
        let dir = std::env::temp_dir().join(format!("cm-read-enc-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ko.txt");
        let (bytes, _, _) = encoding_rs::EUC_KR.encode("# 한글 주석\nTIMEOUT = 60\n");
        std::fs::write(&path, bytes.as_ref()).unwrap();
        let p = path.to_string_lossy().to_string();

        let enc = crate::utils::file::detect_file_encoding(&p);
        assert_eq!(enc, "euc-kr");

        let text = decode_bytes(&std::fs::read(&path).unwrap(), &enc);
        assert!(text.contains("한글 주석"), "Read decoded to {text:?}");
        assert!(!text.contains('\u{FFFD}'), "Read still produces replacement chars");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn schema_matches_dump() {
        let s = embedded_tool_schema("Read");
        assert_eq!(s["title"], "FileReadInput");
        assert_eq!(s["required"], json!(["file_path"]));
        assert_eq!(s["additionalProperties"], json!(false));
    }

    #[test]
    fn pages_parse() {
        assert!(parse_pdf_page_range("1-5").is_some());
        assert!(parse_pdf_page_range("3").is_some());
        let open = parse_pdf_page_range("10-").unwrap();
        assert_eq!(open.first_page, 10);
        assert!(open.last_page.is_none());
        assert!(parse_pdf_page_range("5-3").is_none());
        assert!(parse_pdf_page_range("0").is_none());
        assert!(parse_pdf_page_range("a-b").is_none());
    }

    #[test]
    fn line_numbers_compact() {
        assert_eq!(add_line_numbers_compact("a\nb", 5), "5\ta\n6\tb");
        assert_eq!(add_line_numbers_compact("", 1), "");
    }

    #[test]
    fn ext_helper() {
        assert_eq!(ext_of("/a/b/file.PY"), "py");
        assert_eq!(ext_of("/a/.bashrc"), "");
        assert_eq!(ext_of("archive.tar.gz"), "gz");
        assert_eq!(ext_of("noext"), "");
    }

    #[test]
    fn blocked_devices() {
        assert!(is_blocked_device_path("/dev/zero"));
        assert!(is_blocked_device_path("/proc/self/fd/0"));
        assert!(!is_blocked_device_path("/dev/null"));
    }

    #[tokio::test]
    async fn read_dedup_same_range() {
        use futures::StreamExt;
        let dir = std::env::temp_dir().join(format!("cm-read-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("f.txt");
        std::fs::write(&path, "one\ntwo\n").unwrap();
        let p = path.to_string_lossy().into_owned();

        let ctx = ToolUseContext::default();
        let tool = FileReadTool;
        let input = json!({"file_path": p});

        // first read: content
        let mut s1 = tool.call(input.clone(), ctx.clone(), None);
        let r1 = s1.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, .. } = r1 {
            assert!(result_for_assistant.contains("1\tone"));
        } else {
            panic!("expected result");
        }
        // second read, same range, unchanged mtime: stub
        let mut s2 = tool.call(input, ctx.clone(), None);
        let r2 = s2.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, .. } = r2 {
            assert_eq!(result_for_assistant, FILE_UNCHANGED_STUB);
        } else {
            panic!("expected result");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod bom_roundtrip_tests {
    use super::decode_bytes;
    use crate::tools::file_write_tool::encode_with_encoding;

    /// Python oracle: b"\xff\xfeh\x00i\x00".decode('utf-16-le') == '﻿hi'
    /// and re-encoding restores the identical bytes. The BOM must survive an
    /// edit round-trip or the next read misdetects the file as utf-8.
    #[test]
    fn utf16le_bom_round_trips() {
        let raw: &[u8] = &[0xFF, 0xFE, b'h', 0x00, b'i', 0x00];
        let s = decode_bytes(raw, "utf-16-le");
        assert_eq!(s, "\u{feff}hi");
        assert_eq!(encode_with_encoding(&s, "utf-16-le").unwrap(), raw);
    }

    #[test]
    fn utf16be_bom_round_trips() {
        let raw: &[u8] = &[0xFE, 0xFF, 0x00, b'h', 0x00, b'i'];
        let s = decode_bytes(raw, "utf-16-be");
        assert_eq!(s, "\u{feff}hi");
        assert_eq!(encode_with_encoding(&s, "utf-16-be").unwrap(), raw);
    }

    /// A short read must say so, and a complete one must stay silent. The
    /// phantom trailing element of a newline-terminated file is the trap:
    /// without dropping it, every such file reports one missing line.
    #[test]
    fn truncation_notice_fires_only_on_a_short_read() {
        // 3000-line file read whole: split('\n') counts 3001.
        let n = super::truncation_notice(1, 2000, 3001, true).expect("must warn");
        assert!(n.contains("Showed lines 1-2000 of 3000"), "{n}");
        assert!(n.contains("1000 more lines"), "{n}");
        assert!(n.contains("offset=2001"), "{n}");
        // Continuation read reaches the end.
        assert!(super::truncation_notice(2001, 1001, 3001, true).is_none());
        // Exactly MAX_LINES_TO_READ, newline-terminated: the phantom element
        // must not be mistaken for a 2001st line.
        assert!(super::truncation_notice(1, 2000, 2001, true).is_none());
        // Short files, with and without a final newline.
        assert!(super::truncation_notice(1, 4, 4, true).is_none());
        assert!(super::truncation_notice(1, 3, 3, false).is_none());
        // Explicit limit on a long file still reports the remainder.
        let n = super::truncation_notice(1, 10, 3001, true).expect("must warn");
        assert!(n.contains("2990 more lines"), "{n}");
    }
}
