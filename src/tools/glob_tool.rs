//! Glob Tool for CodeMason
//! Equivalent to codemason/tools/glob_tool.py — based on src/tools/GlobTool/
//! (GlobTool.ts, prompt.ts) from Claude Code.
//!
//! Behavior ported verbatim: glob pattern matching, optional `path` search
//! root (validated as an existing dir), results sorted by modification time
//! (oldest first, per ripgrep --sort=modified) capped at `limit` keeping the
//! first (oldest) files, paths relativized under cwd, and the result-block
//! formatting ("No files found" / truncation note).
//!
//! PRIMARY: ripgrep (`rg --files --glob <pat> --no-ignore --hidden
//! --sort=modified`) — same call as Claude Code's utils/glob.ts. FALLBACK: a
//! walk-based matcher with a faithful port of the Python glob→regex
//! translation (which, like ripgrep --hidden, does NOT skip dot-entries).

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::state::get_cwd;

use super::file_read_tool::embedded_tool_schema;

pub const GLOB_TOOL_NAME: &str = "Glob";

pub const DESCRIPTION: &str = "- Fast file pattern matching tool that works with any codebase size\n- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n- Returns matching file paths sorted by modification time\n- Use this tool when you need to find files by name patterns\n- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead";

/// GlobTool.ts: const limit = globLimits?.maxResults ?? 100
pub const DEFAULT_LIMIT: usize = 100;

/// FILE_NOT_FOUND_CWD_NOTE equivalent (utils/file.ts), used in validateInput.
pub const FILE_NOT_FOUND_CWD_NOTE: &str = "Note that the file path should be either an absolute path or a path relative to the current working directory:";

/// expandPath: resolve ~ and make relative paths absolute against cwd,
/// normalized (os.path.normpath).
fn expand_path(path: &str) -> String {
    let expanded = if let Some(rest) = path.strip_prefix("~/") {
        match dirs::home_dir() {
            Some(h) => h.join(rest).to_string_lossy().into_owned(),
            None => path.to_string(),
        }
    } else if path == "~" {
        dirs::home_dir().map(|h| h.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
    } else {
        path.to_string()
    };
    let abs = if std::path::Path::new(&expanded).is_absolute() {
        std::path::PathBuf::from(&expanded)
    } else {
        std::path::Path::new(&get_cwd()).join(&expanded)
    };
    // normpath: fold '.'/'..' lexically.
    let mut out = std::path::PathBuf::new();
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

/// toRelativePath: relativize paths under cwd to save tokens; paths outside
/// cwd stay absolute (no '..' escape).
fn to_relative_path(path: &str) -> String {
    let cwd = get_cwd();
    let p = std::path::Path::new(path);
    let base = std::path::Path::new(&cwd);
    match p.strip_prefix(base) {
        Ok(rel) => {
            let s = rel.to_string_lossy().into_owned();
            if s.is_empty() {
                ".".to_string()
            } else {
                s
            }
        }
        Err(_) => path.to_string(),
    }
}

/// getPath({ path }): path ? expandPath(path) : getCwd().
fn get_path(input: &Value) -> String {
    match input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
        Some(p) => expand_path(p),
        None => get_cwd(),
    }
}

/// Translate a glob pattern into a regex string (no anchors), matched against
/// a '/'-joined relative path. Supports ripgrep-style '**' (spans any number
/// of directory segments, including zero), '*'/'?'/'[...]' within a single
/// segment, and '{a,b,c}' brace alternation (nested; alternatives may contain
/// glob metacharacters).
pub(crate) fn glob_to_regex(pattern: &str) -> String {
    let pattern = pattern.replace(std::path::MAIN_SEPARATOR, "/");
    let chars: Vec<char> = pattern.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut out = String::new();
    while i < n {
        let c = chars[i];
        if c == '*' {
            // Check for '**'.
            if i + 1 < n && chars[i + 1] == '*' {
                i += 2;
                // '**/' (or '**' at end) matches zero or more path segments.
                if i < n && chars[i] == '/' {
                    i += 1;
                    out.push_str("(?:.*/)?");
                } else {
                    out.push_str(".*");
                }
            } else {
                // Single '*': any run of non-separator characters.
                out.push_str("[^/]*");
                i += 1;
            }
        } else if c == '?' {
            out.push_str("[^/]");
            i += 1;
        } else if c == '[' {
            // Character class: pass through to fnmatch-translated form.
            let mut j = i + 1;
            if j < n && (chars[j] == '!' || chars[j] == '^') {
                j += 1;
            }
            if j < n && chars[j] == ']' {
                j += 1;
            }
            while j < n && chars[j] != ']' {
                j += 1;
            }
            if j >= n {
                // Unterminated class: treat '[' literally.
                out.push_str(&regex::escape("["));
                i += 1;
            } else {
                let stuff: String = chars[i + 1..j].iter().collect();
                let stuff = if let Some(rest) = stuff.strip_prefix('!') {
                    format!("^{rest}")
                } else {
                    stuff
                };
                out.push('[');
                out.push_str(&stuff);
                out.push(']');
                i = j + 1;
            }
        } else if c == '{' {
            // Brace alternation: find the matching '}' (depth-aware), split on
            // top-level commas, translate each alternative as a sub-glob.
            let mut depth = 1i32;
            let mut j = i + 1;
            let mut start = j;
            let mut alts: Vec<String> = Vec::new();
            while j < n && depth > 0 {
                let cj = chars[j];
                if cj == '{' {
                    depth += 1;
                } else if cj == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else if cj == ',' && depth == 1 {
                    alts.push(chars[start..j].iter().collect());
                    start = j + 1;
                }
                j += 1;
            }
            if depth != 0 {
                // Unterminated '{': treat literally.
                out.push_str(&regex::escape("{"));
                i += 1;
            } else {
                alts.push(chars[start..j].iter().collect()); // final alternative
                let parts: Vec<String> = alts.iter().map(|a| glob_to_regex(a)).collect();
                out.push_str(&format!("(?:{})", parts.join("|")));
                i = j + 1;
            }
        } else {
            out.push_str(&regex::escape(&c.to_string()));
            i += 1;
        }
    }
    out
}

/// Compile a glob pattern to an anchored regex matched against a '/'-joined
/// relative path (Python: re.compile('(?s:' + body + ')\\Z') used via .match).
fn translate_glob(pattern: &str) -> Option<regex::Regex> {
    regex::Regex::new(&format!("^(?s:{})\\z", glob_to_regex(pattern))).ok()
}

static RG_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Warn once that ripgrep is missing so Glob fell back to the slow walk.
fn warn_rg_missing_once() {
    if !RG_WARNED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        eprintln!(
            "[codemason] WARNING: ripgrep (rg) not found — Glob is using the slow os.walk fallback, which can stall on large trees (e.g. .venv) and make the spinner look frozen. Install it for the fast canonical path: sudo apt install ripgrep"
        );
    }
}

/// Bound the retained path list from one rg --files run. rg emits oldest-first
/// (--sort=modified) and Glob keeps the OLDEST, so a capped tail is exactly
/// what Glob would drop anyway.
const RG_FILES_MAX: usize = 1_000_000;

const RG_TIMEOUT_SECS: u64 = 30;

/// Stream `rg --files --glob`, capping the retained paths (RG_FILES_MAX).
/// A watchdog kills rg at the 30s deadline even if it blocks.
fn rg_files_streamed(rg: &str, rel_pattern: &str, walk_root: &str) -> Result<Vec<String>, String> {
    use std::io::BufRead;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut proc = std::process::Command::new(rg)
        .args(["--files", "--glob", rel_pattern, "--no-ignore", "--hidden", "--sort=modified"])
        .current_dir(walk_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;

    let timed_out = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    {
        let timed_out = timed_out.clone();
        let done = done.clone();
        let pid = proc.id();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(RG_TIMEOUT_SECS);
            while std::time::Instant::now() < deadline {
                if done.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if !done.load(Ordering::SeqCst) {
                timed_out.store(true, Ordering::SeqCst);
                #[cfg(unix)]
                unsafe {
                    // Single process, NOT the group: rg is spawned without
                    // setsid, so it shares this process's group and killpg
                    // would take the session down with it.
                    libc::kill(pid as libc::pid_t, libc::SIGKILL);
                }
                // Windows has no signals, and this arm used to discard the pid:
                // nothing was killed, so the main thread stayed blocked reading
                // rg's stdout until rg finished on its own. There was no
                // timeout on Windows at all — a wedged rg wedged the session.
                // taskkill closes the pipe, which releases the reader.
                #[cfg(windows)]
                crate::utils::windows_shell::kill_process_tree(pid, None);
                #[cfg(not(any(unix, windows)))]
                let _ = pid;
            }
        });
    }

    let mut out: Vec<String> = Vec::new();
    if let Some(stdout) = proc.stdout.take() {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(p) = line else { break };
            if !p.is_empty() {
                let abs = std::path::Path::new(walk_root).join(&p);
                out.push(abs.to_string_lossy().into_owned());
                if out.len() >= RG_FILES_MAX {
                    let _ = proc.kill();
                    break;
                }
            }
        }
    }
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = proc.wait();
    if timed_out.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("ripgrep --files timed out".to_string());
    }
    Ok(out)
}

/// Walk search_dir and return absolute file paths matching the glob pattern.
///
/// Mirrors the ripgrep-backed util: '**' spans any directory depth, hidden
/// files/dirs ARE included and traversed, and only files are returned.
fn glob_files(pattern: &str, search_dir: &str) -> Result<Vec<String>, String> {
    // An absolute pattern is split into a static base dir + relative pattern
    // so the walk roots at the right place (mirrors extractGlobBaseDirectory).
    let (walk_root, rel_pattern) = if std::path::Path::new(pattern).is_absolute() {
        let norm = pattern.replace(std::path::MAIN_SEPARATOR, "/");
        match norm.find(['*', '?', '[', '{']) {
            None => {
                let base = std::path::Path::new(pattern)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "/".to_string());
                let rel = std::path::Path::new(pattern)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (base, rel)
            }
            Some(match_start) => {
                let static_prefix = &norm[..match_start];
                match static_prefix.rfind('/') {
                    None => ("/".to_string(), pattern.to_string()),
                    Some(last_sep) => {
                        let base = if last_sep == 0 { "/" } else { &static_prefix[..last_sep] };
                        (base.to_string(), pattern[last_sep + 1..].to_string())
                    }
                }
            }
        }
    } else {
        (search_dir.to_string(), pattern.to_string())
    };

    // ── PRIMARY: ripgrep — faithful to Claude Code (utils/glob.ts). Native
    //    speed handles huge trees without stalling; --no-ignore --hidden =
    //    same semantics as the reference.
    if let Some(rg) = crate::utils::ripgrep::resolve_ripgrep() {
        match rg_files_streamed(&rg, &rel_pattern, &walk_root) {
            Ok(files) => return Ok(files),
            Err(_) => {} // rg present but errored → fall to the walk
        }
    } else {
        warn_rg_missing_once();
    }

    // ── FALLBACK: walk. Same --hidden semantics: hidden dirs are traversed.
    let Some(regex) = translate_glob(&rel_pattern) else {
        return Err(format!("invalid glob pattern: {rel_pattern}"));
    };
    // ripgrep matches a slash-less glob (e.g. '*.js') against the BASENAME at
    // any depth; the anchored full-path regex alone would only match top-level
    // files. Mirror rg: match the basename for slash-less patterns.
    let basename_only = !rel_pattern.contains('/');
    let mut files: Vec<String> = Vec::new();
    for entry in walkdir::WalkDir::new(&walk_root).follow_links(false) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let abs = entry.path();
        if basename_only {
            let fname = entry.file_name().to_string_lossy();
            if regex.is_match(&fname) {
                files.push(abs.to_string_lossy().into_owned());
            }
            continue;
        }
        let rel = abs
            .strip_prefix(&walk_root)
            .map(|p| p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"))
            .unwrap_or_default();
        if regex.is_match(&rel) {
            files.push(abs.to_string_lossy().into_owned());
        }
    }
    Ok(files)
}

/// Fast file pattern matching by glob pattern.
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        GLOB_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(GLOB_TOOL_NAME)
    }

    async fn description(&self, _input: &Value) -> String {
        DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        GLOB_TOOL_NAME.to_string()
    }

    /// validateInput: if path is provided, it must exist and be a directory.
    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let path = input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
        if let Some(path) = path {
            let absolute_path = expand_path(path);

            // SECURITY: skip filesystem ops for UNC paths to avoid NTLM leaks.
            if absolute_path.starts_with("\\\\") || absolute_path.starts_with("//") {
                return ValidationResult::success();
            }

            if !std::path::Path::new(&absolute_path).exists() {
                return ValidationResult {
                    result: false,
                    message: format!(
                        "Directory does not exist: {path}. {FILE_NOT_FOUND_CWD_NOTE} {}.",
                        get_cwd()
                    ),
                    meta: Some(json!({"errorCode": 1})),
                };
            }

            if !std::path::Path::new(&absolute_path).is_dir() {
                return ValidationResult {
                    result: false,
                    message: format!("Path is not a directory: {path}"),
                    meta: Some(json!({"errorCode": 2})),
                };
            }
        }

        ValidationResult::success()
    }

    /// The pattern, truncated for display (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("pattern")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::formatting::truncate(
            value,
            crate::utils::formatting::TOOL_SUMMARY_MAX_LENGTH,
        ))
    }

    /// "Finding <pattern>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Finding {summary}"),
            None => "Finding files".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
        if pattern.is_empty() {
            return String::new();
        }
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let text = if path.is_empty() {
            pattern.to_string()
        } else {
            format!("{pattern} in {path}")
        };
        colorize(&text, colors::DIM)
    }

    /// Found N files (GlobTool/UI.tsx SearchResultSummary) — the file list is
    /// the model's business; the user gets the count and Ctrl+O.
    fn render_tool_result_message(&self, data: &Value, verbose: bool) -> Option<String> {
        if !data.is_object() || data.get("error").map(|e| !e.is_null()).unwrap_or(false) {
            return None;
        }
        let n = data.get("numFiles").and_then(|v| v.as_i64()).unwrap_or(0);
        let mut line = format!("Found {n} file{}", if n == 1 { "" } else { "s" });
        if data.get("truncated").and_then(|v| v.as_bool()).unwrap_or(false) {
            line.push_str(&colorize(" (truncated)", colors::DIM));
        }
        if !verbose {
            return Some(line);
        }
        let mut lines = vec![line];
        if let Some(filenames) = data.get("filenames").and_then(|v| v.as_array()) {
            for f in filenames {
                lines.push(f.as_str().unwrap_or("").to_string());
            }
        }
        Some(lines.join("\n"))
    }

    /// call: glob the pattern under getPath, sort by modification time (oldest
    /// first, mirroring ripgrep --sort=modified), cap at the limit keeping the
    /// first (oldest) files, relativize paths, and emit the formatted result.
    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let start = std::time::Instant::now();
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let limit = DEFAULT_LIMIT; // globLimits?.maxResults ?? 100
            let search_dir = get_path(&input);

            // The search shells out / walks the tree — run it off the async
            // executor so a slow tree never blocks the loop (the Python
            // blocked; strictly better, output-identical).
            let search = tokio::task::spawn_blocking({
                let pattern = pattern.clone();
                let search_dir = search_dir.clone();
                move || glob_files(&pattern, &search_dir)
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()));

            match search {
                Err(e) => {
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!("Error searching files: {e}"),
                        data: Some(json!({"error": e})),
                    });
                }
                Ok(all_files) => {
                    // Sort by modification time, oldest first (rg
                    // --sort=modified emits OLDEST->NEWEST; the original slices
                    // [offset:offset+limit] keeping the FIRST files).
                    let mut with_mtime: Vec<(String, f64)> = all_files
                        .into_iter()
                        .map(|f| {
                            let mtime = super::file_read_tool::mtime_seconds(&f).unwrap_or(0.0);
                            (f, mtime)
                        })
                        .collect();
                    with_mtime.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    let sorted_files: Vec<String> = with_mtime.into_iter().map(|(f, _)| f).collect();

                    let offset = 0usize;
                    let truncated = sorted_files.len() > offset + limit;
                    let capped: Vec<&String> = sorted_files.iter().skip(offset).take(limit).collect();

                    // Relativize paths under cwd to save tokens (same as GrepTool).
                    let filenames: Vec<String> = capped.iter().map(|f| to_relative_path(f)).collect();

                    let duration_ms = start.elapsed().as_millis() as i64;

                    // mapToolResultToToolResultBlockParam.
                    let result_for_assistant = if filenames.is_empty() {
                        "No files found".to_string()
                    } else {
                        let mut lines = filenames.clone();
                        if truncated {
                            lines.push(
                                "(Results are truncated. Consider using a more specific path or pattern.)".to_string(),
                            );
                        }
                        lines.join("\n")
                    };

                    let num_files = filenames.len();
                    yield Ok(ToolResult::Result {
                        result_for_assistant,
                        data: Some(json!({
                            "filenames": filenames,
                            "durationMs": duration_ms,
                            "numFiles": num_files,
                            "truncated": truncated,
                        })),
                    });
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_translation() {
        let re = translate_glob("**/*.js").unwrap();
        assert!(re.is_match("a.js"));
        assert!(re.is_match(".config/x.js"));
        assert!(re.is_match("a/b/c.js"));
        assert!(!re.is_match("a.ts"));

        let re2 = translate_glob("src/*.ts").unwrap();
        assert!(re2.is_match("src/x.ts"));
        assert!(!re2.is_match("src/a/x.ts"));

        let re3 = translate_glob("**/*.{ts,tsx}").unwrap();
        assert!(re3.is_match("App.tsx"));
        assert!(re3.is_match("src/x.ts"));
        assert!(!re3.is_match("src/x.rs"));
    }

    #[test]
    fn brace_nesting_and_class() {
        let re = translate_glob("{src,test}/**/*.js").unwrap();
        assert!(re.is_match("src/a/b.js"));
        assert!(re.is_match("test/x.js"));
        assert!(!re.is_match("lib/x.js"));

        let re2 = translate_glob("file[0-9].txt").unwrap();
        assert!(re2.is_match("file5.txt"));
        assert!(!re2.is_match("fileA.txt"));

        let re3 = translate_glob("file[!0-9].txt").unwrap();
        assert!(re3.is_match("fileA.txt"));
        assert!(!re3.is_match("file5.txt"));
    }

    #[tokio::test]
    async fn glob_finds_files_oldest_first() {
        use futures::StreamExt;
        let dir = std::env::temp_dir().join(format!("cm-glob-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.rs2"), "x").unwrap();
        std::fs::write(dir.join("sub/b.rs2"), "y").unwrap();

        let tool = GlobTool;
        let input = json!({"pattern": "**/*.rs2", "path": dir.to_string_lossy()});
        let mut s = tool.call(input, ToolUseContext::default(), None);
        let r = s.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, data } = r {
            assert!(result_for_assistant.contains("a.rs2"), "{result_for_assistant}");
            assert!(result_for_assistant.contains("b.rs2"));
            let d = data.unwrap();
            assert_eq!(d["numFiles"], 2);
            assert_eq!(d["truncated"], false);
        } else {
            panic!();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn validate_missing_dir() {
        let tool = GlobTool;
        let v = tool
            .validate_input(
                &json!({"pattern": "*.rs", "path": "/nonexistent-cm-dir-xyz"}),
                &ToolUseContext::default(),
            )
            .await;
        assert!(!v.result);
        assert!(v.message.starts_with("Directory does not exist:"));
        assert_eq!(v.meta.unwrap()["errorCode"], 1);
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
        let expected = &dump["Glob"]["schema"];
        let actual = GlobTool.input_json_schema();
        assert_eq!(&actual, expected, "schema for Glob diverges from the Python dump");
    }
}
