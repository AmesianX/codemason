//! Grep Tool for CodeMason
//! Equivalent to codemason/tools/grep_tool.py — based on src/tools/GrepTool/
//! (GrepTool.ts, prompt.ts) from Claude Code.
//!
//! "Built on ripgrep": shells out to `rg`, reproducing the exact argument
//! construction of the original (--hidden, VCS-directory exclusions,
//! --max-columns 500, -U --multiline-dotall, -i, -l/-c, -n, -A/-B/-C context,
//! -e for dash patterns, --type, --glob splitting). When `rg` is not on PATH,
//! falls back to a pure regex implementation honoring the same parameters
//! (the regex crate IS ripgrep's engine, so fallback syntax matches rg more
//! closely than the Python `re` fallback did).

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::state::get_cwd;

use super::file_read_tool::embedded_tool_schema;

pub const GREP_TOOL_NAME: &str = "Grep";

/// Description ported VERBATIM from prompt.ts getDescription() (Agent/Bash
/// tool names interpolated).
pub const DESCRIPTION: &str = "A powerful search tool built on ripgrep\n\n  Usage:\n  - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n  - Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n  - Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n  - Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n  - Use Agent tool for open-ended searches requiring multiple rounds\n  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`\n";

// === Constants (ported from GrepTool.ts) ===

/// Version control system directories to exclude from searches.
pub const VCS_DIRECTORIES_TO_EXCLUDE: &[&str] = &[".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

/// Default cap on grep results when head_limit is unspecified.
pub const DEFAULT_HEAD_LIMIT: usize = 250;

/// Max line length passed to ripgrep (--max-columns) to prevent
/// base64/minified content from cluttering output.
pub const MAX_COLUMNS: usize = 500;

/// Port of applyHeadLimit. Returns (items, applied_limit).
pub fn apply_head_limit<T: Clone>(
    items: &[T],
    limit: Option<i64>,
    offset: usize,
) -> (Vec<T>, Option<usize>) {
    // Explicit 0 = unlimited escape hatch.
    if limit == Some(0) {
        return (items.iter().skip(offset).cloned().collect(), None);
    }
    let effective_limit = limit.map(|l| l.max(0) as usize).unwrap_or(DEFAULT_HEAD_LIMIT);
    let sliced: Vec<T> = items.iter().skip(offset).take(effective_limit).collect::<Vec<_>>()
        .into_iter()
        .cloned()
        .collect();
    // Only report applied_limit when truncation actually occurred.
    let was_truncated = items.len().saturating_sub(offset) > effective_limit;
    (sliced, if was_truncated { Some(effective_limit) } else { None })
}

/// Port of formatLimitInfo.
pub fn format_limit_info(applied_limit: Option<usize>, applied_offset: Option<usize>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(l) = applied_limit {
        parts.push(format!("limit: {l}"));
    }
    if let Some(o) = applied_offset {
        if o > 0 {
            parts.push(format!("offset: {o}"));
        }
    }
    parts.join(", ")
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

/// toRelativePath (path.ts:96-98): relativize under cwd, but keep paths
/// OUTSIDE cwd absolute rather than emitting '../foo'.
fn to_relative_path(file_path: &str) -> String {
    let cwd = get_cwd();
    match std::path::Path::new(file_path).strip_prefix(std::path::Path::new(&cwd)) {
        Ok(rel) => {
            let s = rel.to_string_lossy().into_owned();
            if s.is_empty() {
                ".".to_string()
            } else {
                s
            }
        }
        Err(_) => file_path.to_string(),
    }
}

/// Index of the `path:content` separator colon in a ripgrep content-mode
/// line, skipping a leading Windows drive prefix (`C:\...`).
fn path_content_colon(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let start = if bytes.len() >= 3
        && bytes[1] == b':'
        && (bytes[0] as char).is_ascii_alphabetic()
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        2
    } else {
        0
    };
    line[start..].find(':').map(|i| i + start)
}

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
    if std::path::Path::new(&expanded).is_absolute() {
        expanded
    } else {
        std::path::Path::new(&get_cwd()).join(&expanded).to_string_lossy().into_owned()
    }
}

/// Port of the glob-splitting logic in call(): split on whitespace, then on
/// commas (unless the pattern contains braces).
pub fn split_glob(glob: &str) -> Vec<String> {
    let mut glob_patterns: Vec<String> = Vec::new();
    for raw_pattern in glob.split_whitespace() {
        if raw_pattern.is_empty() {
            continue;
        }
        if raw_pattern.contains('{') && raw_pattern.contains('}') {
            glob_patterns.push(raw_pattern.to_string());
        } else {
            glob_patterns.extend(raw_pattern.split(',').filter(|p| !p.is_empty()).map(|p| p.to_string()));
        }
    }
    glob_patterns.retain(|g| !g.is_empty());
    glob_patterns
}

#[allow(clippy::too_many_arguments)]
/// Faithful port of the ripgrep arg construction in call().
fn build_rg_args(
    pattern: &str,
    glob: Option<&str>,
    type_: Option<&str>,
    output_mode: &str,
    context_before: Option<i64>,
    context_after: Option<i64>,
    context_c: Option<i64>,
    context: Option<i64>,
    show_line_numbers: bool,
    case_insensitive: bool,
    multiline: bool,
) -> Vec<String> {
    let mut args: Vec<String> = vec!["--hidden".to_string()];

    // Exclude VCS directories to avoid noise from version control metadata.
    for d in VCS_DIRECTORIES_TO_EXCLUDE {
        args.push("--glob".to_string());
        args.push(format!("!{d}"));
    }

    // Limit line length to prevent base64/minified content from cluttering output.
    args.push("--max-columns".to_string());
    args.push(MAX_COLUMNS.to_string());

    // Only apply multiline flags when explicitly requested.
    if multiline {
        args.push("-U".to_string());
        args.push("--multiline-dotall".to_string());
    }

    if case_insensitive {
        args.push("-i".to_string());
    }

    // Output mode flags.
    if output_mode == "files_with_matches" {
        args.push("-l".to_string());
    } else if output_mode == "count" {
        args.push("-c".to_string());
    }

    if show_line_numbers && output_mode == "content" {
        args.push("-n".to_string());
    }

    // Context flags (-C/context takes precedence over before/after).
    if output_mode == "content" {
        if let Some(c) = context {
            args.push("-C".to_string());
            args.push(c.to_string());
        } else if let Some(c) = context_c {
            args.push("-C".to_string());
            args.push(c.to_string());
        } else {
            if let Some(b) = context_before {
                args.push("-B".to_string());
                args.push(b.to_string());
            }
            if let Some(a) = context_after {
                args.push("-A".to_string());
                args.push(a.to_string());
            }
        }
    }

    // If pattern starts with dash, use -e so rg does not treat it as an option.
    if pattern.starts_with('-') {
        args.push("-e".to_string());
        args.push(pattern.to_string());
    } else {
        args.push(pattern.to_string());
    }

    if let Some(t) = type_ {
        args.push("--type".to_string());
        args.push(t.to_string());
    }

    if let Some(glob) = glob {
        for g in split_glob(glob) {
            args.push("--glob".to_string());
            args.push(g);
        }
    }

    args
}

/// Merge a second ripgrep pass into the first, per output mode.
///
/// ripgrep can only be told ONE encoding for a whole run, so covering a tree
/// that mixes UTF-8 and a legacy encoding means running it twice. The two
/// result sets overlap: an ASCII pattern matches the same line in both passes,
/// because ASCII decodes identically either way. So this is not a concatenation.
///
/// - content / files_with_matches: identical lines are the same finding; keep
///   first-seen order so the primary pass still reads first.
/// - count: the LARGER of the two counts, not their sum. Summing was wrong and
///   measured wrong — it reported every file at exactly double. The premise was
///   inverted: the two passes do not find different matches in a file, they
///   find THE SAME lines. ASCII is byte-identical under both encodings, so an
///   ASCII pattern matches the same lines in both passes (measured: 2 and 2,
///   summed to 4 for a file with 2). For a pattern that is not ASCII only the
///   correctly-decoded pass finds anything at all, and the other reports 0
///   (measured: 1 and 0). Either way one pass's result contains the other's, so
///   the larger is the count and the sum is double-counting.
///
///   The case max would under-report is a file holding real matches under BOTH
///   interpretations at once — different lines valid only as UTF-8 and only as
///   CP949 in the same file. ASCII lines decode identically so they cannot
///   differ, which leaves only deliberately malformed input; under-reporting
///   there is a far better failure than doubling every ordinary count.
use std::collections::{HashMap, HashSet};

/// Run ripgrep twice — once as it comes, once in the configured legacy
/// encoding — and merge.
///
/// ripgrep's automatic detection only reads a BOM, so a CP949 file is searched
/// as if it were UTF-8 and Hangul inside it matches nothing at all (measured on
/// Windows against a file a Windows editor actually wrote). Passing -E is not a
/// fix by itself: measured, `-E euc-kr` finds that file and MISSES the UTF-8
/// one, trading one blind spot for another. Both passes together are what
/// covers a mixed tree.
///
/// Split out of the caller so it can be measured against real ripgrep on a real
/// mixed tree — the version that lived inline could only be exercised through a
/// model turn.
fn run_ripgrep_both_encodings(
    rg_path: &str,
    args: &[String],
    cwd: &str,
    output_mode: &str,
) -> Result<Vec<String>, String> {
    let primary = run_ripgrep(rg_path, args, cwd)?;
    let legacy = crate::utils::config::get_config().grep_legacy_encoding;
    let legacy = legacy.trim().to_string();
    if legacy.is_empty() {
        return Ok(primary);
    }
    let mut largs = args.to_vec();
    largs.push("--encoding".to_string());
    largs.push(legacy);
    // A failure of the extra pass must never lose the results of the one that
    // worked — this is an addition to the search, not a precondition for it.
    let secondary = run_ripgrep(rg_path, &largs, cwd).unwrap_or_default();
    Ok(merge_rg_passes(primary, secondary, output_mode))
}

fn merge_rg_passes(primary: Vec<String>, secondary: Vec<String>, output_mode: &str) -> Vec<String> {
    if secondary.is_empty() {
        return primary;
    }
    if output_mode != "count" {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out = Vec::with_capacity(primary.len() + secondary.len());
        for line in primary.into_iter().chain(secondary) {
            if seen.insert(line.clone()) {
                out.push(line);
            }
        }
        return out;
    }
    // count: the larger per file, preserving first-seen file order.
    let mut order: Vec<String> = Vec::new();
    let mut totals: HashMap<String, i64> = HashMap::new();
    for line in primary.into_iter().chain(secondary) {
        let Some(colon) = line.rfind(':') else { continue };
        if colon == 0 {
            continue;
        }
        let (file, count) = (&line[..colon], &line[colon + 1..]);
        let Ok(n) = count.trim().parse::<i64>() else { continue };
        if !totals.contains_key(file) {
            order.push(file.to_string());
        }
        let slot = totals.entry(file.to_string()).or_insert(0);
        *slot = (*slot).max(n);
    }
    order.into_iter().map(|f| { let n = totals[&f]; format!("{f}:{n}") }).collect()
}

/// Bound the search (Glob uses the same); the original execFile has a timeout.
const RIPGREP_TIMEOUT_SEC: u64 = 30;

/// Hard cap on lines held from a single ripgrep run — a broad pattern over a
/// huge tree could otherwise buffer millions of match lines into RAM.
const RIPGREP_MAX_LINES: usize = 200_000;

/// Invoke ripgrep and return its stdout lines (excluding trailing empty).
///
/// Streams stdout with a hard line cap and a watchdog timer that kills rg at
/// the deadline even if it is blocked producing nothing.
fn run_ripgrep(rg_path: &str, args: &[String], cwd: &str) -> Result<Vec<String>, String> {
    use std::io::{BufRead, Read};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut proc = std::process::Command::new(rg_path)
        .args(args)
        .arg(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to launch ripgrep: {e}"))?;

    let timed_out = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    {
        let timed_out = timed_out.clone();
        let done = done.clone();
        let pid = proc.id();
        std::thread::spawn(move || {
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_secs(RIPGREP_TIMEOUT_SEC);
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

    let mut lines: Vec<String> = Vec::new();
    let mut capped = false;
    if let Some(stdout) = proc.stdout.take() {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.split(b'\n') {
            let Ok(bytes) = line else { break };
            lines.push(String::from_utf8_lossy(&bytes).into_owned());
            if lines.len() >= RIPGREP_MAX_LINES {
                capped = true;
                let _ = proc.kill();
                break;
            }
        }
    }
    let mut stderr = String::new();
    if let Some(mut err) = proc.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    let status = proc.wait().ok();

    if timed_out.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(format!(
            "ripgrep timed out after {RIPGREP_TIMEOUT_SEC}s — search incomplete; narrow the pattern or path."
        ));
    }
    // rg exit codes: 0 = matches, 1 = no matches, 2 = error. A deliberate kill
    // after hitting the line cap is a successful (truncated) search.
    let code = status.and_then(|s| s.code());
    if !matches!(code, Some(0) | Some(1)) && !capped {
        let msg = stderr.trim();
        return Err(if msg.is_empty() {
            format!("ripgrep exited {}", code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into()))
        } else {
            msg.to_string()
        });
    }
    Ok(lines)
}

// === Fallback (used when `rg` is not on PATH) ===

const FALLBACK_SKIP_DIRS: &[&str] = &[".git", ".svn", ".hg", ".bzr", ".jj", ".sl", "node_modules", "__pycache__"];

/// ripgrep's built-in type definitions are extensive; reproduce the most
/// common ones referenced by the description.
fn fallback_type_globs(type_: &str) -> &'static [&'static str] {
    match type_ {
        "py" => &["*.py", "*.pyi", "*.pyw"],
        "js" => &["*.js", "*.jsx", "*.mjs", "*.cjs", "*.vue"],
        "ts" => &["*.ts", "*.tsx", "*.cts", "*.mts"],
        "rust" => &["*.rs"],
        "go" => &["*.go"],
        "java" => &["*.java", "*.jsp"],
        "c" => &["*.c", "*.h"],
        "cpp" => &["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx", "*.C", "*.H"],
        "cs" => &["*.cs"],
        "rb" => &["*.rb"],
        "php" => &["*.php", "*.php3", "*.php4", "*.php5", "*.phtml"],
        "html" => &["*.html", "*.htm", "*.xhtml"],
        "css" => &["*.css", "*.scss", "*.sass", "*.less"],
        "json" => &["*.json", "*.jsonl"],
        "yaml" => &["*.yaml", "*.yml"],
        "md" => &["*.md", "*.markdown", "*.mkd"],
        "sh" => &["*.sh", "*.bash", "*.zsh"],
        "sql" => &["*.sql"],
        "toml" => &["*.toml"],
        "xml" => &["*.xml"],
        _ => &[],
    }
}

/// Python fnmatch.fnmatch (POSIX case-sensitive): '*' matches ANY chars
/// (including '/'), '?' one char, '[seq]'/'[!seq]' classes.
fn fnmatch(name: &str, pat: &str) -> bool {
    let mut re = String::from("^(?s:");
    let chars: Vec<char> = pat.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        match chars[i] {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '[' => {
                let mut j = i + 1;
                if j < chars.len() && chars[j] == '!' {
                    j += 1;
                }
                if j < chars.len() && chars[j] == ']' {
                    j += 1;
                }
                while j < chars.len() && chars[j] != ']' {
                    j += 1;
                }
                if j >= chars.len() {
                    re.push_str(&regex::escape("["));
                } else {
                    let stuff: String = chars[i + 1..j].iter().collect();
                    let stuff = if let Some(rest) = stuff.strip_prefix('!') {
                        format!("^{rest}")
                    } else {
                        stuff
                    };
                    re.push('[');
                    re.push_str(&stuff);
                    re.push(']');
                    i = j;
                }
            }
            c => re.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    re.push_str(")\\z");
    regex::Regex::new(&re).map(|r| r.is_match(name)).unwrap_or(false)
}

fn fallback_iter_files(root: &str, globs: &[String], type_: Option<&str>) -> Vec<String> {
    let type_globs = type_.map(fallback_type_globs).unwrap_or(&[]);
    // rg matches a glob containing '/' against the path relative to the search
    // root, not just the basename.
    let name_globs: Vec<&String> = globs.iter().filter(|g| !g.contains('/')).collect();
    let path_globs: Vec<&String> = globs.iter().filter(|g| g.contains('/')).collect();

    let matches = |relpath: &str, name: &str| -> bool {
        if !globs.is_empty()
            && !(name_globs.iter().any(|g| fnmatch(name, g))
                || path_globs.iter().any(|g| fnmatch(relpath, g)))
        {
            return false;
        }
        if !type_globs.is_empty() && !type_globs.iter().any(|g| fnmatch(name, g)) {
            return false;
        }
        true
    };

    if std::path::Path::new(root).is_file() {
        return vec![root.to_string()];
    }

    let mut files: Vec<String> = Vec::new();
    let walker = walkdir::WalkDir::new(root).follow_links(false).into_iter();
    for entry in walker.filter_entry(|e| {
        !(e.file_type().is_dir()
            && FALLBACK_SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref()))
    }) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let full = entry.path();
        let rel = full
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"))
            .unwrap_or_default();
        let name = entry.file_name().to_string_lossy();
        if matches(&rel, &name) {
            files.push(full.to_string_lossy().into_owned());
        }
    }
    files
}

#[allow(clippy::too_many_arguments)]
/// Pure fallback honoring the same params. Returns rg-equivalent lines:
/// - files_with_matches: absolute file paths
/// - count: 'path:count'
/// - content: 'path:n:line' (or 'path:line') with context lines, '-' separated
fn fallback_search(
    root: &str,
    pattern: &str,
    globs: &[String],
    type_: Option<&str>,
    output_mode: &str,
    context_before: usize,
    context_after: usize,
    show_line_numbers: bool,
    case_insensitive: bool,
    multiline: bool,
) -> Result<Vec<String>, String> {
    // ripgrep's --multiline-dotall makes `.` match newlines WITHOUT changing
    // `^`/`$` anchoring, so only dot_matches_new_line (not multi_line).
    let regex = regex::RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .dot_matches_new_line(multiline)
        .build()
        .map_err(|e| e.to_string())?;

    let files = fallback_iter_files(root, globs, type_);
    let mut out_lines: Vec<String> = Vec::new();

    for fp in files {
        let Ok(bytes) = std::fs::read(&fp) else { continue };
        // Use the shared detector rather than assuming UTF-8. Unlike the ripgrep
        // path above — one stream carrying lines from many files, where no
        // single encoding is right — here we read ONE known file, so its
        // encoding is knowable.
        //
        // What this actually fixes, measured: a UTF-16 file was INVISIBLE to
        // this search. Decoded as UTF-8 its ASCII text keeps the NUL padding
        // bytes, the binary probe below sees \0 and skips the file entirely, so
        // a pattern plainly present in it returned no match at all. With the BOM
        // honoured it decodes and matches normally.
        //
        // It does NOT fix legacy CJK: the detector answers latin-1 for anything
        // it cannot prove is UTF-8, which preserves bytes (what Edit needs to
        // round-trip safely) but is not the text, so Hangul in a CP949 file
        // still does not match a Hangul pattern. That needs real charset
        // detection plus a CJK codec, i.e. a new dependency. Read-only either
        // way: nothing is written back.
        let enc = crate::utils::file::detect_file_encoding(&fp);
        let text = crate::tools::file_read_tool::decode_bytes(&bytes, &enc);
        // Skip likely-binary content.
        // BYTE index into a String panics unless it lands on a char boundary,
        // and 8192 lands mid-character in any file of CJK text — measured, a
        // 9000-byte Korean file panicked the whole search, not just that file
        // (the panic is inside spawn_blocking, so Grep returned "task panicked"
        // and found nothing anywhere). Take characters, which is also what the
        // Python side has always done, so the two agree on what "binary" means.
        let probe: String = text.chars().take(8192).collect();
        let probe = probe.as_str();
        if probe.contains('\0') {
            continue;
        }

        let lines: Vec<&str> = text.split_terminator(['\n']).map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();

        let (matched, match_line_idxs): (bool, std::collections::BTreeSet<usize>) = if multiline {
            let mut idxs = std::collections::BTreeSet::new();
            let mut any = false;
            for m in regex.find_iter(&text) {
                any = true;
                let start_line = text[..m.start()].matches('\n').count();
                idxs.insert(start_line);
            }
            (any, idxs)
        } else {
            let idxs: std::collections::BTreeSet<usize> = lines
                .iter()
                .enumerate()
                .filter(|(_, ln)| regex.is_match(ln))
                .map(|(i, _)| i)
                .collect();
            (!idxs.is_empty(), idxs)
        };

        if !matched {
            continue;
        }

        match output_mode {
            "files_with_matches" => out_lines.push(fp.clone()),
            "count" => out_lines.push(format!("{fp}:{}", match_line_idxs.len())),
            _ => {
                // content
                let mut shown = std::collections::BTreeSet::new();
                for &idx in &match_line_idxs {
                    let lo = idx.saturating_sub(context_before);
                    let hi = (idx + context_after + 1).min(lines.len());
                    for j in lo..hi {
                        if !shown.insert(j) {
                            continue;
                        }
                        let is_match = match_line_idxs.contains(&j);
                        let sep = if is_match { ":" } else { "-" };
                        // Truncate long lines to MAX_COLUMNS (rg --max-columns 500).
                        let line_text: String = lines[j].chars().take(MAX_COLUMNS).collect();
                        if show_line_numbers {
                            out_lines.push(format!("{fp}{sep}{}{sep}{line_text}", j + 1));
                        } else {
                            out_lines.push(format!("{fp}{sep}{line_text}"));
                        }
                    }
                }
            }
        }
    }

    Ok(out_lines)
}

/// A powerful search tool built on ripgrep.
pub struct GrepTool;

fn oi64(input: &Value, a: &str, b: &str) -> Option<i64> {
    input
        .get(a)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_i64())
        .or_else(|| input.get(b).filter(|v| !v.is_null()).and_then(|v| v.as_i64()))
}

fn obool(input: &Value, a: &str, b: &str) -> Option<bool> {
    input
        .get(a)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_bool())
        .or_else(|| input.get(b).filter(|v| !v.is_null()).and_then(|v| v.as_bool()))
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        GREP_TOOL_NAME
    }

    /// GrepTool.maxResultSizeChars (lower than the 50k default → persists sooner).
    fn max_result_size_chars(&self) -> f64 {
        20_000.0
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(GREP_TOOL_NAME)
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
        "Search".to_string()
    }

    /// If path is provided, validate that it exists (port of validateInput).
    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let path = input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
        if let Some(path) = path {
            let absolute_path = expand_path(path);
            // SECURITY: skip fs ops for UNC paths.
            if absolute_path.starts_with("\\\\") || absolute_path.starts_with("//") {
                return ValidationResult::success();
            }
            if !std::path::Path::new(&absolute_path).exists() {
                return ValidationResult::failure(format!(
                    "Path does not exist: {path}. Note: paths are relative to the current working directory {}.",
                    get_cwd()
                ));
            }
        }
        ValidationResult::success()
    }

    /// pattern (in path) — GrepTool/UI.tsx renders the args, not a sentence.
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

    /// "Searching for <pattern>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Searching for {summary}"),
            None => "Searching".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
        if pattern.is_empty() {
            return String::new();
        }
        let path = input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
        let summary = match path {
            Some(p) => format!("{pattern} in {p}"),
            None => pattern.to_string(),
        };
        colorize(&summary, colors::DIM)
    }

    /// SearchResultSummary (GrepTool/UI.tsx): a count per output_mode, and the
    /// matched content only in verbose (Ctrl+O expands it otherwise).
    fn render_tool_result_message(&self, data: &Value, verbose: bool) -> Option<String> {
        if !data.is_object() || data.get("error").map(|e| !e.is_null()).unwrap_or(false) {
            return None;
        }
        let mode = data.get("mode").and_then(|v| v.as_str()).unwrap_or("files_with_matches");
        let line = match mode {
            "content" => {
                let n = data.get("numLines").and_then(|v| v.as_i64()).unwrap_or(0);
                format!("Found {n} line{}", if n == 1 { "" } else { "s" })
            }
            "count" => {
                let n = data.get("numMatches").and_then(|v| v.as_i64()).unwrap_or(0);
                let files = data.get("numFiles").and_then(|v| v.as_i64()).unwrap_or(0);
                format!(
                    "Found {n} match{} across {files} file{}",
                    if n == 1 { "" } else { "es" },
                    if files == 1 { "" } else { "s" }
                )
            }
            _ => {
                let n = data.get("numFiles").and_then(|v| v.as_i64()).unwrap_or(0);
                format!("Found {n} file{}", if n == 1 { "" } else { "s" })
            }
        };
        if !verbose {
            return Some(line);
        }
        let content = data
            .get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                data.get("filenames")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .map(|f| f.as_str().unwrap_or("").to_string())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default()
            });
        if content.is_empty() {
            Some(line)
        } else {
            let mut lines = vec![line];
            lines.extend(content.split('\n').map(|s| s.to_string()));
            Some(lines.join("\n"))
        }
    }

    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let path = input.get("path").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());

            let glob = input.get("glob").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
            let type_ = input.get("type").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
            let output_mode = input
                .get("output_mode")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("files_with_matches")
                .to_string();

            let context_before = oi64(&input, "-B", "context_before");
            let context_after = oi64(&input, "-A", "context_after");
            let context_c = oi64(&input, "-C", "context_c");
            let context_val = input.get("context").filter(|v| !v.is_null()).and_then(|v| v.as_i64());

            let show_line_numbers = obool(&input, "-n", "line_numbers").unwrap_or(true);
            let case_insensitive = obool(&input, "-i", "case_insensitive").unwrap_or(false);

            let head_limit = input.get("head_limit").filter(|v| !v.is_null()).and_then(|v| v.as_i64());
            let offset = input.get("offset").filter(|v| !v.is_null()).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
            let multiline = input.get("multiline").filter(|v| !v.is_null()).and_then(|v| v.as_bool()).unwrap_or(false);

            let absolute_path = path.as_deref().map(expand_path).unwrap_or_else(get_cwd);

            // Search off the executor (subprocess / tree walk).
            let search = tokio::task::spawn_blocking({
                let pattern = pattern.clone();
                let glob = glob.clone();
                let type_ = type_.clone();
                let output_mode = output_mode.clone();
                let absolute_path = absolute_path.clone();
                move || -> Result<Vec<String>, String> {
                    match crate::utils::ripgrep::resolve_ripgrep() {
                        Some(rg_path) => {
                            let args = build_rg_args(
                                &pattern,
                                glob.as_deref(),
                                type_.as_deref(),
                                &output_mode,
                                context_before,
                                context_after,
                                context_c,
                                context_val,
                                show_line_numbers,
                                case_insensitive,
                                multiline,
                            );
                            run_ripgrep_both_encodings(
                                &rg_path, &args, &absolute_path, &output_mode,
                            )
                        }
                        None => {
                            // Resolve effective context for fallback (mirrors precedence).
                            let (cb, ca) = if output_mode == "content" {
                                if let Some(c) = context_val {
                                    (c.max(0) as usize, c.max(0) as usize)
                                } else if let Some(c) = context_c {
                                    (c.max(0) as usize, c.max(0) as usize)
                                } else {
                                    (
                                        context_before.unwrap_or(0).max(0) as usize,
                                        context_after.unwrap_or(0).max(0) as usize,
                                    )
                                }
                            } else {
                                (0, 0)
                            };
                            let globs = glob.as_deref().map(split_glob).unwrap_or_default();
                            fallback_search(
                                &absolute_path,
                                &pattern,
                                &globs,
                                type_.as_deref(),
                                &output_mode,
                                cb,
                                ca,
                                show_line_numbers,
                                case_insensitive,
                                multiline,
                            )
                        }
                    }
                }
            })
            .await
            .unwrap_or_else(|e| Err(e.to_string()));

            let results = match search {
                Ok(r) => r,
                Err(e) => {
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!("Error searching: {e}"),
                        data: Some(json!({"error": e})),
                    });
                    return;
                }
            };

            if output_mode == "content" {
                let (limited, applied_limit) = apply_head_limit(&results, head_limit, offset);
                let final_lines: Vec<String> = limited
                    .iter()
                    .map(|line| match path_content_colon(line) {
                        Some(colon) if colon > 0 => {
                            format!("{}{}", to_relative_path(&line[..colon]), &line[colon..])
                        }
                        _ => line.clone(),
                    })
                    .collect();
                let content = final_lines.join("\n");
                let applied_offset = if offset > 0 { Some(offset) } else { None };
                let limit_info = format_limit_info(applied_limit, applied_offset);
                let result_content = if content.is_empty() { "No matches found".to_string() } else { content.clone() };
                let final_content = if limit_info.is_empty() {
                    result_content
                } else {
                    format!("{result_content}\n\n[Showing results with pagination = {limit_info}]")
                };
                let num_lines = final_lines.len();
                yield Ok(ToolResult::Result {
                    result_for_assistant: final_content,
                    data: Some(json!({
                        "mode": "content",
                        "numFiles": 0,
                        "filenames": [],
                        "content": content,
                        "numLines": num_lines,
                        "appliedLimit": applied_limit,
                        "appliedOffset": applied_offset,
                    })),
                });
                return;
            }

            if output_mode == "count" {
                let (limited, applied_limit) = apply_head_limit(&results, head_limit, offset);
                let final_count_lines: Vec<String> = limited
                    .iter()
                    .map(|line| match line.rfind(':') {
                        Some(colon) if colon > 0 => {
                            format!("{}{}", to_relative_path(&line[..colon]), &line[colon..])
                        }
                        _ => line.clone(),
                    })
                    .collect();

                let mut total_matches: i64 = 0;
                let mut file_count: i64 = 0;
                for line in &final_count_lines {
                    if let Some(colon) = line.rfind(':') {
                        if colon > 0 {
                            if let Ok(n) = line[colon + 1..].parse::<i64>() {
                                total_matches += n;
                                file_count += 1;
                            }
                        }
                    }
                }

                let applied_offset = if offset > 0 { Some(offset) } else { None };
                let limit_info = format_limit_info(applied_limit, applied_offset);
                let raw_content = if final_count_lines.is_empty() {
                    "No matches found".to_string()
                } else {
                    final_count_lines.join("\n")
                };
                let summary = format!(
                    "\n\nFound {total_matches} total {} across {file_count} {}.{}",
                    if total_matches == 1 { "occurrence" } else { "occurrences" },
                    if file_count == 1 { "file" } else { "files" },
                    if limit_info.is_empty() { String::new() } else { format!(" with pagination = {limit_info}") }
                );
                let content_joined = final_count_lines.join("\n");
                yield Ok(ToolResult::Result {
                    result_for_assistant: format!("{raw_content}{summary}"),
                    data: Some(json!({
                        "mode": "count",
                        "numFiles": file_count,
                        "filenames": [],
                        "content": content_joined,
                        "numMatches": total_matches,
                        "appliedLimit": applied_limit,
                        "appliedOffset": applied_offset,
                    })),
                });
                return;
            }

            // files_with_matches mode (default). Sort by mtime desc, filename tiebreak.
            let mut sorted_matches: Vec<String> = results;
            sorted_matches.sort_by(|a, b| {
                let ma = super::file_read_tool::mtime_seconds(a).unwrap_or(0.0);
                let mb = super::file_read_tool::mtime_seconds(b).unwrap_or(0.0);
                mb.partial_cmp(&ma)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
            let (final_matches, applied_limit) = apply_head_limit(&sorted_matches, head_limit, offset);
            let relative_matches: Vec<String> =
                final_matches.iter().map(|p| to_relative_path(p)).collect();
            let num_files = relative_matches.len();

            let applied_offset = if offset > 0 { Some(offset) } else { None };
            let limit_info = format_limit_info(applied_limit, applied_offset);

            let result = if num_files == 0 {
                "No files found".to_string()
            } else {
                format!(
                    "Found {num_files} {}{}\n{}",
                    plural(num_files, "file"),
                    if limit_info.is_empty() { String::new() } else { format!(" {limit_info}") },
                    relative_matches.join("\n")
                )
            };

            yield Ok(ToolResult::Result {
                result_for_assistant: result,
                data: Some(json!({
                    "mode": "files_with_matches",
                    "filenames": relative_matches,
                    "numFiles": num_files,
                    "appliedLimit": applied_limit,
                    "appliedOffset": applied_offset,
                })),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point, measured against REAL ripgrep on a REAL mixed tree:
    /// searching for Hangul must find it in BOTH the UTF-8 file and the CP949
    /// file. Either pass alone finds exactly one of them.
    #[test]
    fn a_mixed_tree_is_searched_in_both_encodings() {
        let Some(rg) = crate::utils::ripgrep::resolve_ripgrep() else { return };
        let dir = std::env::temp_dir().join(format!("cm-mix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (cp949, _, _) = encoding_rs::EUC_KR.encode("한글 주석 하나\n");
        std::fs::write(dir.join("a_cp949.txt"), cp949.as_ref()).unwrap();
        std::fs::write(dir.join("b_utf8.txt"), "한글 주석 둘\n".as_bytes()).unwrap();
        let cwd = dir.to_string_lossy().to_string();

        let args = build_rg_args(
            "한글", None, None, "files_with_matches",
            None, None, None, None, false, false, false,
        );

        // Each pass alone sees only its own half — this is what makes the
        // merge necessary rather than merely tidy.
        let plain = run_ripgrep(&rg, &args, &cwd).unwrap_or_default();
        let mut legacy_args = args.clone();
        legacy_args.push("--encoding".to_string());
        legacy_args.push("euc-kr".to_string());
        let legacy_only = run_ripgrep(&rg, &legacy_args, &cwd).unwrap_or_default();
        let joined = |v: &Vec<String>| v.join(" ");
        assert!(joined(&plain).contains("b_utf8"), "default pass should see the UTF-8 file");
        assert!(!joined(&plain).contains("a_cp949"), "default pass cannot see CP949 — the bug");
        assert!(joined(&legacy_only).contains("a_cp949"), "legacy pass should see CP949");
        assert!(
            !joined(&legacy_only).contains("b_utf8"),
            "legacy-only would MISS the UTF-8 file, which is why -E alone is not the fix"
        );

        // Together they cover the tree. The guard holds the setting for the
        // rest of the test, including the "" window below, so no other test
        // reads it mid-toggle; it restores the previous value on drop.
        let _enc = crate::utils::config::pin_legacy_encoding("euc-kr");
        let both = run_ripgrep_both_encodings(&rg, &args, &cwd, "files_with_matches").unwrap();
        let both_s = both.join(" ");
        assert!(both_s.contains("a_cp949"), "merged result lost the CP949 file: {both:?}");
        assert!(both_s.contains("b_utf8"), "merged result lost the UTF-8 file: {both:?}");

        // Turning it off restores the single-pass behaviour exactly.
        crate::utils::config::update_config(|c| c.grep_legacy_encoding = String::new());
        let off = run_ripgrep_both_encodings(&rg, &args, &cwd, "files_with_matches").unwrap();
        assert_eq!(off, plain, "empty setting must mean one pass, unchanged");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The count bug, end to end against real ripgrep: an ASCII term present
    /// twice in each of two files must total 4, not 8. This is measured through
    /// run_ripgrep_both_encodings rather than the merge helper alone, because
    /// the unit tests passed while the real path reported double — the wrong
    /// premise was baked into what the unit test asserted.
    #[test]
    fn an_ascii_term_is_not_counted_twice_per_file() {
        let Some(rg) = crate::utils::ripgrep::resolve_ripgrep() else { return };
        let dir = std::env::temp_dir().join(format!("cm-cnt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let body = "SHARED_TERM 한글\nx\nSHARED_TERM 둘\n";
        std::fs::write(dir.join("modern_utf8.py"), body.as_bytes()).unwrap();
        let (cp949, _, _) = encoding_rs::EUC_KR.encode(body);
        std::fs::write(dir.join("legacy_cp949.py"), cp949.as_ref()).unwrap();
        let cwd = dir.to_string_lossy().to_string();

        let _enc = crate::utils::config::pin_legacy_encoding("euc-kr");
        let args = build_rg_args(
            "SHARED_TERM", None, None, "count",
            None, None, None, None, false, false, false,
        );
        let lines = run_ripgrep_both_encodings(&rg, &args, &cwd, "count").unwrap();

        let total: i64 = lines
            .iter()
            .filter_map(|l| l.rfind(':').map(|c| l[c + 1..].trim().parse::<i64>().unwrap_or(0)))
            .sum();
        assert_eq!(total, 4, "each file holds 2; doubling means the merge summed: {lines:?}");
        assert_eq!(lines.len(), 2, "each file must appear once: {lines:?}");
        for line in &lines {
            let n: i64 = line[line.rfind(':').unwrap() + 1..].parse().unwrap();
            assert_eq!(n, 2, "per-file count is wrong in {line}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The two passes overlap, so merging is not concatenation. An ASCII
    /// pattern matches the same line in both, because ASCII decodes the same
    /// either way.
    #[test]
    fn the_second_pass_merges_rather_than_duplicating() {
        let primary = vec!["a.rs:1:hit".to_string(), "b.rs:2:hit".to_string()];
        let secondary = vec!["a.rs:1:hit".to_string(), "ko.txt:3:한글".to_string()];
        let merged = merge_rg_passes(primary, secondary, "content");
        assert_eq!(
            merged,
            vec!["a.rs:1:hit", "b.rs:2:hit", "ko.txt:3:한글"],
            "a line found by both passes is ONE finding, and order follows the first pass"
        );

        // files_with_matches is the same rule.
        let merged = merge_rg_passes(
            vec!["a.rs".to_string()],
            vec!["a.rs".to_string(), "ko.txt".to_string()],
            "files_with_matches",
        );
        assert_eq!(merged, vec!["a.rs", "ko.txt"]);
    }

    /// count takes the LARGER of the two passes, never their sum. Summing was
    /// measured reporting every file at exactly double, because the passes do
    /// not find different matches — they find the same lines.
    #[test]
    fn counts_take_the_larger_pass_not_the_sum() {
        // The ASCII case, measured with real ripgrep: both passes see the same
        // lines and report the same number. Summing doubled it.
        let merged = merge_rg_passes(
            vec!["a.rs:2".to_string()],
            vec!["a.rs:2".to_string()],
            "count",
        );
        assert_eq!(merged, vec!["a.rs:2"], "identical counts must not become 4");

        // The non-ASCII case: only the correctly-decoded pass finds anything.
        let merged = merge_rg_passes(
            vec!["utf8.py:1".to_string()],
            vec!["cp949.py:1".to_string()],
            "count",
        );
        assert_eq!(merged, vec!["utf8.py:1", "cp949.py:1"]);

        // A file only the second pass can read keeps its count, and file order
        // still follows first sight.
        let merged = merge_rg_passes(
            vec!["a.rs:2".to_string(), "b.rs:1".to_string()],
            vec!["a.rs:3".to_string(), "ko.txt:5".to_string()],
            "count",
        );
        assert_eq!(merged, vec!["a.rs:3", "b.rs:1", "ko.txt:5"]);

        // A path containing a colon must still split on the LAST one.
        let merged = merge_rg_passes(
            vec!["dir:x/a.rs:2".to_string()],
            vec!["dir:x/a.rs:4".to_string()],
            "count",
        );
        assert_eq!(merged, vec!["dir:x/a.rs:4"]);
    }

    /// An empty second pass must be a no-op, not a rewrite of the first.
    #[test]
    fn an_empty_second_pass_changes_nothing() {
        let primary = vec!["a.rs:1:x".to_string()];
        assert_eq!(merge_rg_passes(primary.clone(), vec![], "content"), primary);
        let counts = vec!["a.rs:2".to_string()];
        assert_eq!(merge_rg_passes(counts.clone(), vec![], "count"), counts);
    }

    /// A byte index into a String panics off a char boundary. 8192 lands
    /// mid-character in any CJK file, and the panic took the WHOLE search down
    /// rather than skipping one file.
    #[test]
    fn a_large_cjk_file_does_not_panic_the_binary_probe() {
        let dir = std::env::temp_dir().join(format!("cm-probe-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("big_ko.txt");
        // 9000 bytes: byte 8192 falls inside a 3-byte character.
        std::fs::write(&path, "가".repeat(3000)).unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > 8192);

        // Exercise the probe expression itself, which is what panicked.
        let text = std::fs::read_to_string(&path).unwrap();
        let probe: String = text.chars().take(8192).collect();
        assert!(!probe.contains('\0'), "not binary");
        assert_eq!(probe.chars().count(), 3000, "shorter than 8192 chars, so all of it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The point of teaching the detector legacy CJK: a CP949 source is what a
    /// Korean Windows editor writes by default, and searching one for Hangul
    /// used to match nothing at all — the detector answered latin-1, which
    /// preserves bytes but is not the text.
    #[test]
    fn hangul_in_a_cp949_file_is_findable() {
        let dir = std::env::temp_dir().join(format!("cm-grep-kr-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ko.txt");
        let (bytes, _, _) = encoding_rs::EUC_KR.encode("이것은 한글 주석입니다\nplain ascii\n");
        std::fs::write(&path, bytes.as_ref()).unwrap();

        let enc = crate::utils::file::detect_file_encoding(&path.to_string_lossy());
        assert_eq!(enc, "euc-kr", "a CP949 file must be recognised, not filed as latin-1");

        let raw = std::fs::read(&path).unwrap();
        let text = crate::tools::file_read_tool::decode_bytes(&raw, &enc);
        assert!(text.contains("한글 주석"), "decoded text lost the Hangul: {text:?}");

        // The old behaviour, for contrast — this is why it matched nothing.
        let as_latin1: String = raw.iter().map(|&b| b as char).collect();
        assert!(
            !as_latin1.contains("한글"),
            "latin-1 must NOT contain the Hangul, or this test proves nothing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A UTF-16 file was invisible to the fallback search: decoded as UTF-8 its
    /// ASCII text keeps the NUL padding, the binary probe sees it and skips the
    /// file, so a pattern plainly inside it returned nothing.
    #[test]
    fn a_utf16_file_is_searchable_not_skipped_as_binary() {
        let dir = std::env::temp_dir().join(format!("cm-grep-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("u16.txt");
        // UTF-16LE with a BOM, which is what the detector keys on.
        let mut bytes = vec![0xFF, 0xFE];
        for u in "needle here\nsecond line\n".encode_utf16() {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();

        let enc = crate::utils::file::detect_file_encoding(&path.to_string_lossy());
        assert_eq!(enc, "utf-16-le", "the BOM must be recognised");
        let decoded = crate::tools::file_read_tool::decode_bytes(&bytes, &enc);
        assert!(decoded.contains("needle here"), "decoded text lost the content");
        assert!(!decoded.contains('\0'), "NULs left in — the binary probe would skip this");

        // The old path: what the file looked like before the detector was used.
        let lossy = String::from_utf8_lossy(&bytes).into_owned();
        assert!(
            lossy.contains('\0'),
            "without the detector this must still look binary, or the test proves nothing"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rg_args_construction() {
        let args = build_rg_args(
            "-foo", Some("*.rs *.py"), Some("rust"), "content",
            Some(1), Some(2), None, None, true, true, true,
        );
        let joined = args.join(" ");
        assert!(joined.starts_with("--hidden --glob !.git --glob !.svn --glob !.hg --glob !.bzr --glob !.jj --glob !.sl --max-columns 500"));
        assert!(joined.contains("-U --multiline-dotall"));
        assert!(joined.contains(" -i "));
        assert!(joined.contains(" -n "));
        assert!(joined.contains("-B 1"));
        assert!(joined.contains("-A 2"));
        assert!(joined.contains("-e -foo"));
        assert!(joined.contains("--type rust"));
        assert!(joined.ends_with("--glob *.rs --glob *.py"));
    }

    #[test]
    fn context_precedence() {
        let args = build_rg_args(
            "p", None, None, "content", Some(1), Some(2), Some(3), Some(4),
            false, false, false,
        );
        let joined = args.join(" ");
        assert!(joined.contains("-C 4"));
        assert!(!joined.contains("-B 1"));
    }

    #[test]
    fn glob_splitting() {
        assert_eq!(split_glob("*.js, *.ts"), vec!["*.js", "*.ts"]);
        assert_eq!(split_glob("*.{ts,tsx}"), vec!["*.{ts,tsx}"]);
        assert_eq!(split_glob("a.js b.py,c.rb"), vec!["a.js", "b.py", "c.rb"]);
    }

    #[test]
    fn head_limit_semantics() {
        let items: Vec<i32> = (0..300).collect();
        let (kept, limit) = apply_head_limit(&items, None, 0);
        assert_eq!(kept.len(), 250);
        assert_eq!(limit, Some(250));
        let (all, none) = apply_head_limit(&items, Some(0), 10);
        assert_eq!(all.len(), 290);
        assert!(none.is_none());
        let (few, no_trunc) = apply_head_limit(&items[..5].to_vec(), Some(10), 0);
        assert_eq!(few.len(), 5);
        assert!(no_trunc.is_none());
    }

    #[test]
    fn drive_colon_skipped() {
        // Oracle-verified against the Python: _path_content_colon returns 9 and
        // 7 for these exact vectors. (An earlier full-suite failure here was a
        // stale mixed-build artifact from concurrent cluster integration; the
        // function is pure and deterministic.)
        assert_eq!(path_content_colon("C:\\x\\y.rs:12:code"), Some(9));
        assert_eq!(path_content_colon("/a/b.rs:3:code"), Some(7));
    }

    #[test]
    fn fnmatch_semantics() {
        assert!(fnmatch("a/b.js", "*.js")); // '*' crosses '/'
        assert!(fnmatch("x.py", "*.p?"));
        assert!(fnmatch("file5", "file[0-9]"));
        assert!(!fnmatch("fileA", "file[0-9]"));
    }

    #[tokio::test]
    async fn grep_content_mode_end_to_end() {
        use futures::StreamExt;
        let dir = std::env::temp_dir().join(format!("cm-grep-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world\nsecond line\n").unwrap();

        let tool = GrepTool;
        let input = json!({
            "pattern": "hello",
            "path": dir.to_string_lossy(),
            "output_mode": "content",
        });
        let mut s = tool.call(input, ToolUseContext::default(), None);
        let r = s.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, data } = r {
            assert!(result_for_assistant.contains("hello world"), "{result_for_assistant}");
            assert!(result_for_assistant.contains(":1:"), "line numbers on by default");
            assert_eq!(data.unwrap()["numLines"], 1);
        } else {
            panic!();
        }
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
        let expected = &dump["Grep"]["schema"];
        let actual = GrepTool.input_json_schema();
        assert_eq!(&actual, expected, "schema for Grep diverges from the Python dump");
    }
}
