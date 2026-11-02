//! File Edit Tool for CodeMason
//! Equivalent to codemason/tools/file_edit_tool.py — based on
//! src/tools/FileEditTool/ from Claude Code.
//!
//! The tool name is 'Edit'. The input schema, DESCRIPTION, prompt text,
//! validation behavior (old_string uniqueness, replace_all, read-before-edit,
//! curly-quote normalization, de-sanitization) and the apply/patch logic are
//! ported verbatim. Read-before-edit uses ToolUseContext.read_file_state
//! (populated by FileRead). Line-ending and encoding preservation are
//! reproduced.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::base::{CanUseToolFn, Tool, ToolError, ToolResult, ToolStream};
use crate::types::query::{ReadFileState, ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors, format_edit_diff_preview};
use crate::utils::state::get_cwd;

use super::file_read_tool::{embedded_tool_schema, mtime_seconds};
use super::file_write_tool::encode_with_encoding;

// === Constants (constants.ts) ===

pub const FILE_EDIT_TOOL_NAME: &str = "Edit";

pub const FILE_UNEXPECTEDLY_MODIFIED_ERROR: &str =
    "File has been unexpectedly modified. Read it again before attempting to write it.";

/// NotebookEditTool/constants.ts
pub const NOTEBOOK_EDIT_TOOL_NAME: &str = "NotebookEdit";

/// FileReadTool/prompt.ts
pub const FILE_READ_TOOL_NAME: &str = "Read";

/// V8/Bun string length limit guard. 1 GiB (stat bytes).
pub const MAX_EDIT_FILE_SIZE: u64 = 1024 * 1024 * 1024;

// === Description / prompt (prompt.ts, verbatim) ===

/// FileEditTool.ts description() returns this string.
pub const DESCRIPTION: &str = "A tool for editing files";

/// Local approximation of utils/format.formatFileSize.
fn format_file_size(num_bytes: u64) -> String {
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

/// prompt.ts getPreReadInstruction()
fn get_pre_read_instruction() -> String {
    format!(
        "\n- You must use your `{FILE_READ_TOOL_NAME}` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. "
    )
}

/// prompt.ts getEditToolDescription() -> getDefaultEditDescription().
/// This is the text the API sends as the Edit tool's description (the Python
/// class attr DESCRIPTION = get_edit_tool_description()).
pub fn get_edit_tool_description() -> String {
    // isCompactLinePrefixEnabled() is environment-gated; the default substrate
    // uses the non-compact prefix format.
    let prefix_format = "spaces + line number + arrow";
    // USER_TYPE === 'ant' adds a minimal-uniqueness hint; off by default.
    let minimal_uniqueness_hint = if std::env::var("USER_TYPE").as_deref() == Ok("ant") {
        "\n- Use the smallest old_string that's clearly unique — usually 2-4 adjacent lines is sufficient. Avoid including 10+ lines of context when less uniquely identifies the target."
    } else {
        ""
    };
    format!(
        "Performs exact string replacements in files.\n\nUsage:{}\n- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: {prefix_format}. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.{minimal_uniqueness_hint}\n- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.",
        get_pre_read_instruction()
    )
}

// === Quote normalization (utils.ts) ===

pub const LEFT_SINGLE_CURLY_QUOTE: char = '\u{2018}';
pub const RIGHT_SINGLE_CURLY_QUOTE: char = '\u{2019}';
pub const LEFT_DOUBLE_CURLY_QUOTE: char = '\u{201C}';
pub const RIGHT_DOUBLE_CURLY_QUOTE: char = '\u{201D}';

/// utils.ts normalizeQuotes().
pub fn normalize_quotes(s: &str) -> String {
    s.replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, "\"")
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, "\"")
}

// === De-sanitization / trailing-whitespace normalization (utils.ts) ===

/// utils.ts DESANITIZATIONS — replacements to de-sanitize strings from Claude.
/// Copied verbatim; ITERATION ORDER IS SIGNIFICANT (an ordered slice, matching
/// Python's insertion-ordered dict / JS Object.entries).
pub const DESANITIZATIONS: &[(&str, &str)] = &[
    ("<fnr>", "<function_results>"),
    ("<n>", "<name>"),
    ("</n>", "</name>"),
    ("<o>", "<output>"),
    ("</o>", "</output>"),
    ("<e>", "<error>"),
    ("</e>", "</error>"),
    ("<s>", "<system>"),
    ("</s>", "</system>"),
    ("<r>", "<result>"),
    ("</r>", "</result>"),
    ("< META_START >", "<META_START>"),
    ("< META_END >", "<META_END>"),
    ("< EOT >", "<EOT>"),
    ("< META >", "<META>"),
    ("< SOS >", "<SOS>"),
    ("\n\nH:", "\n\nHuman:"),
    ("\n\nA:", "\n\nAssistant:"),
];

/// utils.ts stripTrailingWhitespace() — strips trailing whitespace from each
/// line while preserving line endings (CRLF, LF, CR).
pub fn strip_trailing_whitespace(s: &str) -> String {
    // Split capturing the line endings, mirroring str.split(/(\r\n|\n|\r)/).
    let mut out = String::with_capacity(s.len());
    let mut segment = String::new();
    let mut chars = s.chars().peekable();
    let mut flush = |segment: &mut String, out: &mut String| {
        out.push_str(segment.trim_end());
        segment.clear();
    };
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                flush(&mut segment, &mut out);
                if chars.peek() == Some(&'\n') {
                    chars.next();
                    out.push_str("\r\n");
                } else {
                    out.push('\r');
                }
            }
            '\n' => {
                flush(&mut segment, &mut out);
                out.push('\n');
            }
            c => segment.push(c),
        }
    }
    flush(&mut segment, &mut out);
    out
}

/// utils.ts desanitizeMatchString() — applies the DESANITIZATIONS
/// replacements, returning the rewritten string and the (from, to) pairs that
/// actually changed something.
pub fn desanitize_match_string(match_string: &str) -> (String, Vec<(&'static str, &'static str)>) {
    let mut result = match_string.to_string();
    let mut applied: Vec<(&'static str, &'static str)> = Vec::new();
    for (frm, to) in DESANITIZATIONS {
        let before = result.clone();
        result = result.replace(frm, to);
        if before != result {
            applied.push((frm, to));
        }
    }
    (result, applied)
}

/// utils.ts normalizeFileEditInput() — pure-string normalization applied
/// before findActualString.
///
/// - strips trailing whitespace from new_string, EXCEPT for .md/.mdx files
///   where two trailing spaces are a hard line break.
/// - if old_string is not an exact substring of the file, retries with the
///   de-sanitized version; when that matches, the same applied replacements
///   are mirrored onto new_string.
///
/// file_content is None when the file could not be read (only the
/// trailing-whitespace normalization applies then).
pub fn normalize_file_edit_input(
    file_path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
    file_content: Option<&str>,
) -> (String, String, bool) {
    // Markdown uses two trailing spaces as a hard line break — stripping would
    // silently change semantics. Skip stripTrailingWhitespace for .md/.mdx.
    let lower = file_path.to_lowercase();
    let is_markdown = lower.ends_with(".md") || lower.ends_with(".mdx");

    let normalized_new_string = if is_markdown {
        new_string.to_string()
    } else {
        strip_trailing_whitespace(new_string)
    };

    let Some(file_content) = file_content else {
        return (old_string.to_string(), normalized_new_string, replace_all);
    };

    // If exact string match works, keep it as is.
    if file_content.contains(old_string) {
        return (old_string.to_string(), normalized_new_string, replace_all);
    }

    // Try de-sanitize string if exact match fails.
    let (desanitized_old_string, applied_replacements) = desanitize_match_string(old_string);

    if file_content.contains(&desanitized_old_string) {
        // Apply the same exact replacements to new_string.
        let mut desanitized_new_string = normalized_new_string;
        for (frm, to) in &applied_replacements {
            desanitized_new_string = desanitized_new_string.replace(frm, to);
        }
        return (desanitized_old_string, desanitized_new_string, replace_all);
    }

    (old_string.to_string(), normalized_new_string, replace_all)
}

/// utils.ts findActualString() — handles quote normalization. Works in CHAR
/// indices (quote normalization maps 1 char -> 1 char, so char offsets in the
/// normalized text equal char offsets in the original).
pub fn find_actual_string(file_content: &str, search_string: &str) -> Option<String> {
    // First try exact match
    if file_content.contains(search_string) {
        return Some(search_string.to_string());
    }

    // Try with normalized quotes
    let normalized_search = normalize_quotes(search_string);
    let normalized_file = normalize_quotes(file_content);

    let byte_idx = normalized_file.find(&normalized_search)?;
    let char_idx = normalized_file[..byte_idx].chars().count();
    let search_chars = search_string.chars().count();
    let actual: String = file_content.chars().skip(char_idx).take(search_chars).collect();
    Some(actual)
}

/// utils.ts isOpeningContext().
fn is_opening_context(chars: &[char], index: usize) -> bool {
    if index == 0 {
        return true;
    }
    matches!(
        chars[index - 1],
        ' ' | '\t' | '\n' | '\r' | '(' | '[' | '{' | '—' | '–'
    )
}

fn apply_curly_double_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| {
            if ch == '"' {
                if is_opening_context(&chars, i) {
                    LEFT_DOUBLE_CURLY_QUOTE
                } else {
                    RIGHT_DOUBLE_CURLY_QUOTE
                }
            } else {
                ch
            }
        })
        .collect()
}

/// Equivalent of /\p{L}/u.test(ch).
fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn apply_curly_single_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| {
            if ch == '\'' {
                let prev_is_letter = i > 0 && is_letter(chars[i - 1]);
                let next_is_letter = i + 1 < chars.len() && is_letter(chars[i + 1]);
                if prev_is_letter && next_is_letter {
                    // Apostrophe in a contraction — right single curly quote
                    RIGHT_SINGLE_CURLY_QUOTE
                } else if is_opening_context(&chars, i) {
                    LEFT_SINGLE_CURLY_QUOTE
                } else {
                    RIGHT_SINGLE_CURLY_QUOTE
                }
            } else {
                ch
            }
        })
        .collect()
}

/// utils.ts preserveQuoteStyle().
pub fn preserve_quote_style(old_string: &str, actual_old_string: &str, new_string: &str) -> String {
    if old_string == actual_old_string {
        return new_string.to_string();
    }

    let has_double_quotes = actual_old_string.contains(LEFT_DOUBLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_DOUBLE_CURLY_QUOTE);
    let has_single_quotes = actual_old_string.contains(LEFT_SINGLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_SINGLE_CURLY_QUOTE);

    if !has_double_quotes && !has_single_quotes {
        return new_string.to_string();
    }

    let mut result = new_string.to_string();
    if has_double_quotes {
        result = apply_curly_double_quotes(&result);
    }
    if has_single_quotes {
        result = apply_curly_single_quotes(&result);
    }
    result
}

// === Edit application (utils.ts) ===

/// utils.ts applyEditToFile().
pub fn apply_edit_to_file(
    original_content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> String {
    let f = |content: &str, search: &str, replace: &str| -> String {
        if replace_all {
            content.replace(search, replace)
        } else {
            content.replacen(search, replace, 1)
        }
    };

    if !new_string.is_empty() {
        return f(original_content, old_string, new_string);
    }

    let with_newline = format!("{old_string}\n");
    let strip_trailing_newline =
        !old_string.ends_with('\n') && original_content.contains(&with_newline);

    if strip_trailing_newline {
        f(original_content, &with_newline, new_string)
    } else {
        f(original_content, old_string, new_string)
    }
}

/// utils.ts getPatchForEdit()/getPatchForEdits() — returns the updated file
/// contents, or Err("String not found in file. Failed to apply edit.").
pub fn get_updated_file_for_edit(
    file_contents: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<String, String> {
    // Special case for empty files.
    if file_contents.is_empty() && old_string.is_empty() && new_string.is_empty() {
        return Ok(String::new());
    }

    let updated_file = if old_string.is_empty() {
        new_string.to_string()
    } else {
        apply_edit_to_file(file_contents, old_string, new_string, replace_all)
    };

    if updated_file == file_contents {
        return Err("String not found in file. Failed to apply edit.".to_string());
    }

    Ok(updated_file)
}

// === Path / line-ending / encoding helpers ===

/// utils/path.expandPath + cwd resolution for relative paths.
pub(crate) fn expand_path(file_path: &str) -> String {
    let expanded = if let Some(rest) = file_path.strip_prefix("~/") {
        match dirs::home_dir() {
            Some(h) => h.join(rest).to_string_lossy().into_owned(),
            None => file_path.to_string(),
        }
    } else if file_path == "~" {
        dirs::home_dir()
            .map(|h| h.to_string_lossy().into_owned())
            .unwrap_or_else(|| file_path.to_string())
    } else {
        file_path.to_string()
    };
    if std::path::Path::new(&expanded).is_absolute() {
        expanded
    } else {
        std::path::Path::new(&get_cwd()).join(&expanded).to_string_lossy().into_owned()
    }
}

/// Returns (content, file_exists, encoding, line_endings).
///
/// Content is normalized to LF for matching (mirrors readFileForEdit + the
/// validateInput byte read that does .replaceAll('\r\n', '\n')). The detected
/// encoding and line-ending style are returned so they can be restored on
/// write.
fn read_file_for_edit(absolute_file_path: &str) -> (String, bool, &'static str, &'static str) {
    let raw = match std::fs::read(absolute_file_path) {
        Ok(b) => b,
        Err(_) => return (String::new(), false, "utf-8", "LF"),
    };

    // Use the shared detector rather than a local utf-16-le-BOM-or-utf-8 guess.
    // The old guess sent every non-UTF-8 file through a lossy UTF-8 decode, and
    // write_text_content_preserving re-encodes and writes back what was decoded
    // — so editing one ASCII line of a CP949/EUC-KR source (Korean Windows'
    // default) replaced every Hangul byte in the file with U+FFFD, permanently.
    // The detector's latin-1 fallback maps bytes 1:1, so an undecodable file now
    // round-trips byte-for-byte. This is also what file_write_tool already does.
    let detected = crate::utils::file::detect_file_encoding(absolute_file_path);
    // The detector only samples the first 4096 bytes, so a file that turns
    // invalid later would still be decoded lossily. Confirm against the whole
    // buffer and drop to latin-1 if it does not hold, which keeps the write-back
    // lossless (Python does the same with a strict decode and an except arm).
    let encoding: &'static str = match detected.as_str() {
        "utf-8" | "utf-8-sig" if std::str::from_utf8(&raw).is_err() => "latin-1",
        // UTF-16 needs the same whole-buffer proof as the legacy labels, and for
        // a sharper reason: Rust's UTF-16 decode CANNOT fail. An unpaired
        // surrogate becomes U+FFFD and a trailing odd byte is padded with 0x00,
        // and both get written back — measured, an edit to an unrelated line
        // rewrote a surrogate and made the file one byte longer. Python drops to
        // latin-1 here because its decode raises; this is that, made explicit.
        "utf-16-le" if crate::utils::file::utf16_round_trips(&raw, true) => "utf-16-le",
        "utf-16-be" if crate::utils::file::utf16_round_trips(&raw, false) => "utf-16-be",
        "utf-16-le" | "utf-16-be" => "latin-1",
        "utf-8-sig" => "utf-8-sig",
        "latin-1" => "latin-1",
        // A legacy label was proved against the detector's 4096-byte HEADER.
        // Edit rewrites the whole file, so it has to hold for the whole file:
        // re-verify against `raw` and drop to byte-preserving latin-1 if it does
        // not. Without this the label fell through to "utf-8" below and a CP949
        // source came back as U+FFFD and was written back that way — the exact
        // data-loss bug the round-trip test in this file exists to catch, and it
        // did catch it.
        other if crate::utils::file::is_legacy_label(other) => {
            crate::utils::file::verified_legacy_encoding_exact(&raw).unwrap_or("latin-1")
        }
        _ => "utf-8",
    };

    let content = super::file_read_tool::decode_bytes(&raw, encoding);
    let line_endings = if content.contains("\r\n") { "CRLF" } else { "LF" };
    // Normalize to LF for matching.
    let content = content.replace("\r\n", "\n");
    (content, true, encoding, line_endings)
}

/// utils/file.writeTextContent — restore line endings + encoding on write.
fn write_text_content_preserving(
    absolute_file_path: &str,
    content: &str,
    encoding: &str,
    line_endings: &str,
) -> Result<(), String> {
    let out = if line_endings == "CRLF" {
        // Normalize first to avoid doubling, then apply CRLF.
        content.replace("\r\n", "\n").replace('\n', "\r\n")
    } else {
        content.to_string()
    };
    let bytes = encode_with_encoding(&out, encoding)?;
    crate::utils::file::atomic_write_bytes(absolute_file_path, &bytes).map_err(|e| e.to_string())
}

/// Python str(bool) — the meta values are capitalized 'True'/'False' strings.
fn py_bool_str(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

fn is_abs(path: &str) -> bool {
    std::path::Path::new(path).is_absolute()
}

// === FileEdit Tool ===

/// A tool for editing files (exact string replacement).
pub struct FileEditTool;

fn s_of<'a>(input: &'a Value, key: &str) -> &'a str {
    input.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &'static str {
        FILE_EDIT_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(FILE_EDIT_TOOL_NAME)
    }

    /// The API description is the full prompt.ts text (the Python class attr
    /// DESCRIPTION = get_edit_tool_description(); note the MODULE-level
    /// 'A tool for editing files' blurb was shadowed by it, exactly as here).
    async fn description(&self, _input: &Value) -> String {
        get_edit_tool_description()
    }

    async fn prompt(&self) -> String {
        get_edit_tool_description()
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Update".to_string()
    }

    /// Faithful port of FileEditTool.ts validateInput().
    async fn validate_input(&self, input: &Value, context: &ToolUseContext) -> ValidationResult {
        let file_path = s_of(input, "file_path");
        let old_string = s_of(input, "old_string");
        let new_string = s_of(input, "new_string");
        let replace_all = input.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

        let full_file_path = expand_path(file_path);

        // old_string === new_string -> no changes (errorCode 1)
        if old_string == new_string {
            return ValidationResult::failure(
                "No changes to make: old_string and new_string are exactly the same.",
            );
        }

        // SECURITY: Skip filesystem operations for UNC paths.
        if full_file_path.starts_with("\\\\") || full_file_path.starts_with("//") {
            return ValidationResult::success();
        }

        // Prevent OOM on multi-GB files (errorCode 10)
        if let Ok(meta) = std::fs::metadata(&full_file_path) {
            let size = meta.len();
            if size > MAX_EDIT_FILE_SIZE {
                return ValidationResult::failure(format!(
                    "File is too large to edit ({}). Maximum editable file size is {}.",
                    format_file_size(size),
                    format_file_size(MAX_EDIT_FILE_SIZE)
                ));
            }
        }

        // Read the file (bytes, encoding-detected, LF-normalized).
        let (file_content, file_exists, _enc, _le) = read_file_for_edit(&full_file_path);

        // File doesn't exist
        if !file_exists {
            // Empty old_string on nonexistent file means new file creation — valid
            if old_string.is_empty() {
                return ValidationResult::success();
            }
            let sim = crate::utils::file::find_similar_file(&full_file_path);
            let message = format!(
                "File does not exist. Current working directory: {}.{}",
                get_cwd(),
                sim.map(|s| format!(" Did you mean {s}?")).unwrap_or_default()
            );
            return ValidationResult {
                result: false,
                message,
                meta: Some(json!({"isFilePathAbsolute": py_bool_str(is_abs(file_path))})),
            };
        }

        // File exists with empty old_string — only valid if file is empty
        if old_string.is_empty() {
            if !file_content.trim().is_empty() {
                return ValidationResult::failure("Cannot create new file - file already exists.");
            }
            return ValidationResult::success();
        }

        // Jupyter notebooks must use NotebookEdit
        if full_file_path.ends_with(".ipynb") {
            return ValidationResult::failure(format!(
                "File is a Jupyter Notebook. Use the {NOTEBOOK_EDIT_TOOL_NAME} to edit this file."
            ));
        }

        // read-before-edit (errorCode 6) + partial-view rejection.
        let read_state = context.read_file_state.lock().get(&full_file_path).cloned();
        let Some(read_state) = read_state.filter(|s| !s.is_partial_view) else {
            return ValidationResult {
                result: false,
                message: "File has not been read yet. Read it first before writing to it."
                    .to_string(),
                meta: Some(json!({"isFilePathAbsolute": py_bool_str(is_abs(file_path))})),
            };
        };

        // Modified-since-read check (errorCode 7) with the full-read content
        // equality fallback: mtime can change without the content changing
        // (cloud sync / AV / linter touch).
        if let Some(last_write_time) = mtime_seconds(&full_file_path) {
            if last_write_time > read_state.timestamp {
                let content_unchanged =
                    read_state.is_full_read() && file_content == read_state.content;
                if !content_unchanged {
                    return ValidationResult::failure(
                        "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.",
                    );
                }
            }
        }

        // normalizeFileEditInput BEFORE findActualString.
        let (old_string, _new_string, replace_all) = normalize_file_edit_input(
            file_path,
            old_string,
            new_string,
            replace_all,
            Some(&file_content),
        );

        // findActualString to handle quote normalization (errorCode 8)
        let Some(actual_old_string) = find_actual_string(&file_content, &old_string) else {
            return ValidationResult {
                result: false,
                message: format!("String to replace not found in file.\nString: {old_string}"),
                meta: Some(json!({"isFilePathAbsolute": py_bool_str(is_abs(file_path))})),
            };
        };

        let matches = file_content.matches(&actual_old_string).count();

        // Multiple matches but replace_all is false (errorCode 9)
        if matches > 1 && !replace_all {
            return ValidationResult {
                result: false,
                message: format!(
                    "Found {matches} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.\nString: {old_string}"
                ),
                meta: Some(json!({
                    "isFilePathAbsolute": py_bool_str(is_abs(file_path)),
                    "actualOldString": actual_old_string,
                })),
            };
        }

        ValidationResult::success()
    }

    /// The edit IS the diff — show it, not the path (FileEditTool/UI.tsx).
    /// The display path (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("file_path")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::file::get_display_path(value))
    }

    /// "Editing <path>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Editing {summary}"),
            None => "Editing file".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let file_path = s_of(input, "file_path");
        let old_string = s_of(input, "old_string");
        let new_string = s_of(input, "new_string");
        if file_path.is_empty() {
            return String::new();
        }
        // format_edit_diff_preview already indents its own lines.
        format_edit_diff_preview(file_path, old_string, new_string)
            .trim_start_matches('\n')
            .to_string()
    }

    /// Post-edit summary; the diff was already shown at tool-use time.
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        let file_path = data.get("filePath").and_then(|v| v.as_str()).filter(|s| !s.is_empty())?;
        let old_s = data.get("oldString").and_then(|v| v.as_str()).unwrap_or("");
        let new_s = data.get("newString").and_then(|v| v.as_str()).unwrap_or("");
        let old_lines: Vec<&str> = old_s.split('\n').collect();
        let new_lines: Vec<&str> = new_s.split('\n').collect();
        let diff = crate::utils::difflib::unified_diff(&old_lines, &new_lines, 0);
        let mut adds = 0usize;
        let mut rems = 0usize;
        for line in &diff {
            if line.starts_with('+') && !line.starts_with("+++") {
                adds += 1;
            } else if line.starts_with('-') && !line.starts_with("---") {
                rems += 1;
            }
        }
        let mut parts: Vec<String> = Vec::new();
        if adds > 0 {
            parts.push(format!("{adds} addition{}", if adds != 1 { "s" } else { "" }));
        }
        if rems > 0 {
            parts.push(format!("{rems} removal{}", if rems != 1 { "s" } else { "" }));
        }
        let summary = if parts.is_empty() { "no changes".to_string() } else { parts.join(" and ") };
        let name = std::path::Path::new(file_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| file_path.to_string());
        Some(format!("Updated {} with {summary}", colorize(&name, colors::BOLD)))
    }

    /// Faithful port of FileEditTool.ts call().
    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let file_path = s_of(&input, "file_path").to_string();
            let old_string = s_of(&input, "old_string").to_string();
            let new_string = s_of(&input, "new_string").to_string();
            let replace_all = input.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

            let absolute_file_path = expand_path(&file_path);

            // Faithful: edit failures (FILE_UNEXPECTEDLY_MODIFIED_ERROR,
            // "String not found", I/O errors) PROPAGATE as a tool error.
            let outcome: Result<ToolResult, String> = (|| {
                // Ensure parent directory exists (matches fs.mkdir(dirname)).
                if let Some(parent) = std::path::Path::new(&absolute_file_path).parent() {
                    if !parent.as_os_str().is_empty() && !parent.is_dir() {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                }

                // 2. Load current state and confirm no changes since last read.
                let (original_file_contents, file_exists, encoding, line_endings) =
                    read_file_for_edit(&absolute_file_path);

                if file_exists {
                    let last_read = context.read_file_state.lock().get(&absolute_file_path).cloned();
                    let last_write_time = mtime_seconds(&absolute_file_path).unwrap_or(0.0);
                    let stale = match &last_read {
                        None => true,
                        Some(lr) => last_write_time > lr.timestamp,
                    };
                    if stale {
                        // Content-equality fallback for full reads (FileEditTool.ts:451-468).
                        let content_unchanged = last_read
                            .as_ref()
                            .map(|lr| lr.is_full_read() && original_file_contents == lr.content)
                            .unwrap_or(false);
                        if !content_unchanged {
                            return Err(FILE_UNEXPECTEDLY_MODIFIED_ERROR.to_string());
                        }
                    }
                }

                // normalizeFileEditInput BEFORE findActualString. When the file
                // doesn't exist, file_content is None so only new_string is
                // normalized (matches the original's ENOENT-returns-original path).
                let (old_string, new_string, replace_all) = normalize_file_edit_input(
                    &file_path,
                    &old_string,
                    &new_string,
                    replace_all,
                    if file_exists { Some(&original_file_contents) } else { None },
                );

                // 3. findActualString to handle quote normalization.
                let actual_old_string = find_actual_string(&original_file_contents, &old_string)
                    .unwrap_or_else(|| old_string.clone());

                // Preserve curly quotes in new_string when the file uses them.
                let actual_new_string =
                    preserve_quote_style(&old_string, &actual_old_string, &new_string);

                // 4. Generate updated file.
                let updated_file = get_updated_file_for_edit(
                    &original_file_contents,
                    &actual_old_string,
                    &actual_new_string,
                    replace_all,
                )?;

                // 5. Write to disk (preserving encoding + line endings).
                write_text_content_preserving(
                    &absolute_file_path,
                    &updated_file,
                    encoding,
                    line_endings,
                )?;

                // 6. Refresh readFileState so a subsequent edit sees the new
                // content as "already read" (full read: store the updated text
                // for the equality fallback).
                if let Some(mtime) = mtime_seconds(&absolute_file_path) {
                    context.read_file_state.lock().insert(absolute_file_path.clone(), ReadFileState {
                        timestamp: mtime,
                        content: updated_file.clone(),
                        offset: None,
                        limit: None,
                        is_partial_view: false,
                    });
                }

                // mapToolResultToToolResultBlockParam
                let result_text = if replace_all {
                    format!(
                        "The file {file_path} has been updated. All occurrences were successfully replaced."
                    )
                } else {
                    format!("The file {file_path} has been updated successfully.")
                };

                Ok(ToolResult::Result {
                    result_for_assistant: result_text,
                    data: Some(json!({
                        "filePath": file_path,
                        "oldString": actual_old_string,
                        "newString": new_string,
                        "originalFile": original_file_contents,
                        "userModified": false,
                        "replaceAll": replace_all,
                    })),
                })
            })();

            match outcome {
                Ok(result) => yield Ok(result),
                Err(message) => {
                    crate::utils::log::log_error(&message, None);
                    yield Err(ToolError::new(message));
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two UTF-16 shapes Rust's lossy decode silently rewrote: an unpaired
    /// surrogate (becomes U+FFFD) and an odd trailing byte (gets padded with
    /// 0x00, so the file GROWS). Both were written back over the user's file.
    #[test]
    fn a_malformed_utf16_file_is_not_rewritten() {
        let dir = std::env::temp_dir().join(format!("cm-u16-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let cases: [(&str, Vec<u8>); 2] = [
            // BOM, "hi", then a lone high surrogate.
            ("surrogate.txt", vec![0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00, 0x00, 0xD8, 0x21, 0x00]),
            // BOM, "hi", then one stray byte — odd length.
            ("odd.txt", vec![0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00, 0x78]),
        ];

        for (name, original) in cases {
            let path = dir.join(name);
            std::fs::write(&path, &original).unwrap();
            let p = path.to_string_lossy().to_string();

            let (text, _crlf, enc, eol) = read_file_for_edit(&p);
            assert_eq!(enc, "latin-1", "{name}: must fall back, not decode lossily");
            // latin-1 is byte-preserving, so a no-op edit must be a no-op write.
            write_text_content_preserving(&p, &text, enc, eol).unwrap();
            let after = std::fs::read(&path).unwrap();
            assert_eq!(
                after, original,
                "{name}: bytes changed with no edit at all — lossy UTF-16 was written back"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Round-trip guard for the Edit path. Editing ONE ascii line of a file must
    /// leave every other byte of it untouched, whatever the file's encoding.
    ///
    /// This is written BEFORE teaching the detector about legacy CJK, on
    /// purpose: that change makes Edit decode and re-encode through a codec
    /// instead of preserving bytes positionally, which is exactly how the CP949
    /// destruction bug worked the first time. If any of these ever fail, the
    /// detector has started guessing an encoding it cannot reproduce.
    #[test]
    fn editing_one_line_preserves_every_other_byte() {
        let dir = std::env::temp_dir().join(format!("cm-edit-rt-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        // (name, original bytes) — each holds non-ASCII the edit must not touch.
        let cp949 = {
            let (b, _, _) = encoding_rs::EUC_KR.encode("주석: 한글 설명
MARKER
끝
");
            b.into_owned()
        };
        let utf8 = "주석: 한글 설명
MARKER
끝
".as_bytes().to_vec();
        let utf16 = {
            let mut v = vec![0xFF, 0xFE];
            for u in "주석
MARKER
끝
".encode_utf16() {
                v.extend_from_slice(&u.to_le_bytes());
            }
            v
        };
        // Bytes that are not valid in ANY of our candidates — must stay exact.
        let junk = b"\x80\x81 MARKER \xfe\xff\n".to_vec();

        for (name, original) in [
            ("cp949.txt", cp949),
            ("utf8.txt", utf8),
            ("utf16.txt", utf16),
            ("junk.bin", junk),
        ] {
            let path = dir.join(name);
            std::fs::write(&path, &original).unwrap();
            let p = path.to_string_lossy().to_string();

            let (text, _crlf, enc, eol) = read_file_for_edit(&p);
            assert!(text.contains("MARKER"), "{name}: the ascii line must be readable");

            let edited = text.replace("MARKER", "EDITED");
            write_text_content_preserving(&p, &edited, enc, eol)
                .unwrap_or_else(|e| panic!("{name}: write-back failed: {e}"));

            let after = std::fs::read(&path).unwrap();
            let expect: Vec<u8> = {
                // The only difference permitted is that one ascii word, encoded
                // the same way the rest of the file is.
                let from = encode_with_encoding("MARKER", enc).unwrap();
                let to = encode_with_encoding("EDITED", enc).unwrap();
                let mut v = Vec::new();
                let mut i = 0;
                while i < original.len() {
                    if original[i..].starts_with(&from[..]) {
                        v.extend_from_slice(&to);
                        i += from.len();
                    } else {
                        v.push(original[i]);
                        i += 1;
                    }
                }
                v
            };
            assert_eq!(
                after, expect,
                "{name}: bytes outside the edit changed — this is the data-loss bug"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    use futures::StreamExt;

    /// Editing one ASCII line of a non-UTF-8 file must not touch the rest of it.
    ///
    /// CP949 is Notepad's default on Korean Windows, so a source file with
    /// Hangul comments is the normal case here, not an edge case. The old
    /// utf-16-le-BOM-or-utf-8 guess decoded such a file lossily and the write
    /// path re-encoded what it decoded, so a one-token rename annihilated every
    /// Hangul byte in the file, permanently and unrecoverably from the file.
    #[test]
    fn edit_preserves_bytes_of_a_non_utf8_file() {
        // "# 한글 주석" in CP949, then an ASCII line to edit.
        let mut original: Vec<u8> = vec![0x23, 0x20];
        original.extend_from_slice(&[0xC7, 0xD1, 0xB1, 0xDB]); // 한글
        original.extend_from_slice(b"\nDEBUG = False\n");

        let dir = std::env::temp_dir().join(format!("cm_edit_enc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("cp949.py");
        std::fs::write(&path, &original).expect("write fixture");
        let p = path.to_string_lossy().into_owned();

        let (content, ok, encoding, endings) = read_file_for_edit(&p);
        assert!(ok);
        assert!(
            !content.contains('\u{FFFD}'),
            "the decode must not introduce replacement characters"
        );

        let edited = content.replace("DEBUG = False", "DEBUG = True");
        write_text_content_preserving(&p, &edited, encoding, endings).expect("write back");

        let after = std::fs::read(&path).expect("read back");
        assert!(
            after.windows(4).any(|w| w == [0xC7, 0xD1, 0xB1, 0xDB]),
            "the CP949 Hangul bytes must survive byte-for-byte"
        );
        assert!(
            !after.windows(3).any(|w| w == [0xEF, 0xBF, 0xBD]),
            "no U+FFFD may be written into the file"
        );
        assert!(
            String::from_utf8_lossy(&after).contains("DEBUG = True"),
            "the edit itself must still be applied"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quote_normalization_find() {
        let file = "say \u{201C}hello\u{201D} now";
        let found = find_actual_string(file, "say \"hello\" now").unwrap();
        assert_eq!(found, "say \u{201C}hello\u{201D} now");
    }

    #[test]
    fn desanitize_ordered() {
        let (out, applied) = desanitize_match_string("<n>x</n>");
        assert_eq!(out, "<name>x</name>");
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn strip_trailing_ws_preserves_endings() {
        assert_eq!(strip_trailing_whitespace("a  \r\nb\t\nc "), "a\r\nb\nc");
    }

    #[test]
    fn markdown_keeps_hard_breaks() {
        let (_o, n, _r) =
            normalize_file_edit_input("doc.md", "x", "line  \nnext", false, Some("x y"));
        assert_eq!(n, "line  \nnext");
        let (_o2, n2, _r2) =
            normalize_file_edit_input("code.py", "x", "line  \nnext", false, Some("x y"));
        assert_eq!(n2, "line\nnext");
    }

    #[test]
    fn empty_new_string_strips_trailing_newline() {
        let content = "a\nb\nc\n";
        let updated = apply_edit_to_file(content, "b", "", false);
        assert_eq!(updated, "a\nc\n");
    }

    #[test]
    fn edit_not_found_errors() {
        let e = get_updated_file_for_edit("abc", "zzz", "y", false).unwrap_err();
        assert_eq!(e, "String not found in file. Failed to apply edit.");
    }

    #[tokio::test]
    async fn uniqueness_and_replace_all() {
        let dir = std::env::temp_dir().join(format!("cm-edit-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("f.txt");
        std::fs::write(&path, "x x x").unwrap();
        let p = path.to_string_lossy().into_owned();
        let ctx = ToolUseContext::default();
        // record a full read so read-before-edit passes
        ctx.read_file_state.lock().insert(p.clone(), ReadFileState {
            timestamp: f64::MAX,
            content: "x x x".into(),
            offset: None,
            limit: None,
            is_partial_view: false,
        });
        let tool = FileEditTool;
        let v = tool
            .validate_input(&json!({"file_path": p, "old_string": "x", "new_string": "y"}), &ctx)
            .await;
        assert!(!v.result);
        assert!(v.message.starts_with("Found 3 matches"));

        let mut s = tool.call(
            json!({"file_path": p, "old_string": "x", "new_string": "y", "replace_all": true}),
            ctx,
            None,
        );
        let r = s.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, .. } = r {
            assert!(result_for_assistant.contains("All occurrences were successfully replaced."));
        } else {
            panic!();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "y y y");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_before_edit_enforced() {
        let dir = std::env::temp_dir().join(format!("cm-edit-rbe-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("f.txt");
        std::fs::write(&path, "content").unwrap();
        let p = path.to_string_lossy().into_owned();
        let ctx = ToolUseContext::default();
        let tool = FileEditTool;
        let v = tool
            .validate_input(&json!({"file_path": p, "old_string": "content", "new_string": "y"}), &ctx)
            .await;
        assert!(!v.result);
        assert_eq!(v.message, "File has not been read yet. Read it first before writing to it.");
        assert_eq!(v.meta.as_ref().unwrap()["isFilePathAbsolute"], "True");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod schema_dump_equality {
    use super::*;
    use crate::tools::base::Tool as _;

    /// The input schema must be byte-equal to the pydantic model_json_schema()
    /// dump (schemas/tool_schemas.json) — the model-facing contract.
    #[test]
    fn schema_matches_python_dump() {
        let dump: serde_json::Value =
            serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap();
        let expected = &dump["Edit"]["schema"];
        let actual = FileEditTool.input_json_schema();
        assert_eq!(&actual, expected, "schema for Edit diverges from the Python dump");
    }
}
