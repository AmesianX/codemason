//! Formatting utilities for CodeMason CLI output.
//! Equivalent to codemason/utils/formatting.py.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::io::IsTerminal;

/// Unified-diff hunk header: @@ -oldStart[,oldCount] +newStart[,newCount] @@.
static DIFF_HUNK_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@").unwrap());

// === ANSI Color Support ===

/// Pure-ASCII compatibility mode.
///
/// When on, output degrades to 7-bit ASCII so it stays legible on ancient/dumb
/// terminals: no ANSI colour or escape sequences, no markdown rendering, no
/// emoji/braille/box-drawing. Toggled by CODEMASON_PLAIN / CODEMASON_ASCII env
/// vars; the --plain/--ascii CLI flag sets CODEMASON_PLAIN early in main().
pub fn plain_mode() -> bool {
    let v = std::env::var("CODEMASON_PLAIN")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("CODEMASON_ASCII").ok().filter(|s| !s.is_empty()));
    match v {
        Some(s) => matches!(s.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => false,
    }
}

/// Output glyphs: (pretty unicode, plain ASCII). sym() picks per plain_mode().
const SYMBOLS: &[(&str, &str, &str)] = &[
    ("connector", "L", "L"), // tool-result branch connector (1-cell ASCII everywhere)
    ("warn", "⚠", "!"),
    ("bolt", "⚡", "*"),
    ("bar_full", "█", "#"),
    ("bar_empty", "░", "-"),
    ("rule", "⎯", "-"),
    ("dash", "—", "-"),   // em dash
    ("dot", "·", "-"),    // middle dot separator
    ("ellipsis", "…", "..."),
    ("arrow", "→", "->"),
    ("up", "↑", "^"),
    ("down", "↓", "v"),
    ("think", "✻", "*"), // reasoning/thinking token marker (spinner)
];

/// Return a UI glyph as pretty unicode, or plain ASCII in plain_mode().
pub fn sym(name: &str) -> &'static str {
    for (n, pretty, plain) in SYMBOLS {
        if *n == name {
            return if plain_mode() { plain } else { pretty };
        }
    }
    ""
}

/// In CJK locales most terminals render East-Asian 'Ambiguous'-width chars as
/// 2 cells. Detect that from the locale env vars. Returning on the FIRST *set*
/// variable is intentional and POSIX-correct (LC_ALL > LC_CTYPE > LANG); a set
/// LC_ALL=C must NOT be overridden by a CJK LANG.
fn ambiguous_is_wide() -> bool {
    for var in ["LC_ALL", "LC_CTYPE", "LANG"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                let vl = v.to_lowercase();
                return ["ko", "ja", "zh", "cjk"].iter().any(|t| vl.contains(t));
            }
        }
    }
    false
}

/// Best-effort terminal cell width of a string: wide (W/F) chars count 2,
/// East-Asian ambiguous (A) count 2 in CJK locales else 1, zero-width
/// combining marks count 0.
pub fn display_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthChar;
    let amb_wide = ambiguous_is_wide();
    // unicode-width already treats W/F as 2 and combining as 0; ambiguous chars
    // are width 1 by default (the crate's CJK variant was removed), so widen
    // them by locale using the East_Asian_Width=Ambiguous test below.
    s.chars()
        .map(|ch| {
            let base = UnicodeWidthChar::width(ch).unwrap_or(0);
            if amb_wide && base == 1 && is_east_asian_ambiguous(ch) {
                2
            } else {
                base
            }
        })
        .sum()
}

/// Clip a (possibly styled) line to `width` cells.
///
/// Both pickers render into prompt_toolkit `Window(..., wrap_lines=False)` on
/// the Python side: a line wider than the terminal is CUT, never wrapped.  The
/// Rust pickers repaint by walking back `painted` ROWS with ESC[nA, so one
/// wrapped line leaves the cursor a row short and the whole block marches down
/// the screen on every refresh — clipping keeps logical lines and physical rows
/// equal.  SGR sequences occupy no cells and are copied through; if the cut
/// lands inside styled text the style is closed.
/// How long a one-line tool summary may be (constants/toolLimits.ts
/// TOOL_SUMMARY_MAX_LENGTH). Every tool's get_activity_description truncates to
/// this, so a pasted 4KB regex cannot push the spinner off the screen.
pub const TOOL_SUMMARY_MAX_LENGTH: usize = 50;

/// Cut `text` to `max_width` columns, marking the cut (utils/truncate.ts).
///
/// Width, not characters — the same reason clip_to_width counts columns. The
/// ellipsis replaces the last columns rather than being added past the limit,
/// so the result never exceeds `max_width`.
///
/// The ellipsis costs what THIS process says it costs. The original reserves a
/// hardcoded 1 column (`maxWidth - 1`) because its width function — the
/// `string-width` package — calls U+2026 one cell wide everywhere.
/// codemason's display_width is locale-aware, and '…' is East-Asian
/// *ambiguous*: two cells wherever ambiguous counts wide. Reserving 1 there
/// returned max_width + 1 columns and broke the one promise this function makes.
pub fn truncate(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    let reserve = display_width("…").max(1);
    format!(
        "{}…",
        clip_to_width(text, max_width.saturating_sub(reserve))
    )
}

pub fn clip_to_width(s: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0usize;
    let mut styled = false;
    let mut cut = false;
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            out.push(ch);
            styled = true;
            for c2 in chars.by_ref() {
                out.push(c2);
                if c2.is_ascii_alphabetic() {
                    break; // CSI final byte
                }
            }
            continue;
        }
        let w = display_width(&ch.to_string());
        if used + w > width {
            cut = true;
            break;
        }
        out.push(ch);
        used += w;
    }
    if cut && styled && !out.ends_with("\x1b[0m") {
        out.push_str("\x1b[0m");
    }
    out
}

/// East_Asian_Width=Ambiguous membership for the ranges that matter in
/// terminal output (Latin-1 signs, Greek, Cyrillic, box drawing, common
/// punctuation/symbols). Mirrors unicodedata.east_asian_width == 'A' for the
/// characters this UI actually prints.
fn is_east_asian_ambiguous(ch: char) -> bool {
    let c = ch as u32;
    matches!(c,
        0x00A1 | 0x00A4 | 0x00A7 | 0x00A8 | 0x00AA | 0x00AD | 0x00AE
        | 0x00B0..=0x00B4 | 0x00B6..=0x00BA | 0x00BC..=0x00BF | 0x00C6 | 0x00D0
        | 0x00D7 | 0x00D8 | 0x00DE..=0x00E1 | 0x00E6 | 0x00E8..=0x00EA
        | 0x00EC | 0x00ED | 0x00F0 | 0x00F2 | 0x00F3 | 0x00F7..=0x00FA | 0x00FC | 0x00FE
        | 0x0391..=0x03A9 | 0x03B1..=0x03C9
        | 0x0401 | 0x0410..=0x044F | 0x0451
        | 0x2010 | 0x2013..=0x2016 | 0x2018 | 0x2019 | 0x201C | 0x201D
        | 0x2020..=0x2022 | 0x2024..=0x2027 | 0x2030 | 0x2032 | 0x2033 | 0x2035
        | 0x203B | 0x203E | 0x2074 | 0x207F | 0x2081..=0x2084 | 0x20AC
        | 0x2103 | 0x2105 | 0x2109 | 0x2113 | 0x2116 | 0x2121 | 0x2122 | 0x2126 | 0x212B
        | 0x2153 | 0x2154 | 0x215B..=0x215E | 0x2160..=0x216B | 0x2170..=0x2179
        | 0x2189 | 0x2190..=0x2199 | 0x21B8 | 0x21B9 | 0x21D2 | 0x21D4 | 0x21E7
        | 0x2200 | 0x2202 | 0x2203 | 0x2207 | 0x2208 | 0x220B | 0x220F | 0x2211
        | 0x2215 | 0x221A | 0x221D..=0x2220 | 0x2223 | 0x2225 | 0x2227..=0x222C
        | 0x222E | 0x2234..=0x2237 | 0x223C | 0x223D | 0x2248 | 0x224C | 0x2252
        | 0x2260 | 0x2261 | 0x2264..=0x2267 | 0x226A | 0x226B | 0x226E | 0x226F
        | 0x2282 | 0x2283 | 0x2286 | 0x2287 | 0x2295 | 0x2299 | 0x22A5 | 0x22BF
        | 0x2312 | 0x2460..=0x24E9 | 0x24EB..=0x254B | 0x2550..=0x2573
        | 0x2580..=0x258F | 0x2592..=0x2595 | 0x25A0 | 0x25A1 | 0x25A3..=0x25A9
        | 0x25B2 | 0x25B3 | 0x25B6 | 0x25B7 | 0x25BC | 0x25BD | 0x25C0 | 0x25C1
        | 0x25C6..=0x25C8 | 0x25CB | 0x25CE..=0x25D1 | 0x25E2..=0x25E5 | 0x25EF
        | 0x2605 | 0x2606 | 0x2609 | 0x260E | 0x260F | 0x261C | 0x261E
        | 0x2640 | 0x2642 | 0x2660 | 0x2661 | 0x2663..=0x2665 | 0x2667..=0x266A
        | 0x266C | 0x266D | 0x266F | 0x269E | 0x269F | 0x26BF | 0x26C6..=0x26CD
        | 0x26CF..=0x26D3 | 0x26D5..=0x26E1 | 0x26E3 | 0x26E8 | 0x26E9
        | 0x26EB..=0x26F1 | 0x26F4 | 0x26F6..=0x26F9 | 0x26FB | 0x26FC | 0x26FE | 0x26FF
        | 0x273D | 0x2776..=0x277F | 0x2B56..=0x2B59
        | 0x3248..=0x324F | 0xE000..=0xF8FF | 0xFE00..=0xFE0F | 0xFFFD
    )
}

/// Check if terminal supports colors.
pub fn supports_color() -> bool {
    if plain_mode() {
        return false; // pure-ASCII: no ANSI escapes at all
    }
    if std::env::var("NO_COLOR").map(|v| !v.is_empty()).unwrap_or(false) {
        return false;
    }
    // FORCE_COLOR is checked BEFORE the isatty gate so it can force color on a
    // pipe/redirect (its whole purpose).
    if std::env::var("FORCE_COLOR").map(|v| !v.is_empty()).unwrap_or(false) {
        return true;
    }
    std::io::stdout().is_terminal()
}

pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const GRAY: &str = "\x1b[90m";
    pub const RED: &str = "\x1b[91m";
    pub const GREEN: &str = "\x1b[92m";
    pub const YELLOW: &str = "\x1b[93m";
    pub const BLUE: &str = "\x1b[94m";
    pub const MAGENTA: &str = "\x1b[95m";
    pub const CYAN: &str = "\x1b[96m";
    pub const DIM: &str = "\x1b[2m";
    pub const BOLD: &str = "\x1b[1m";
    // Background colors for diff display
    pub const BG_GREEN: &str = "\x1b[48;5;22m"; // Dark green background
    pub const BG_RED: &str = "\x1b[48;5;52m";   // Dark red background
}

/// Apply color if supported.
pub fn colorize(text: &str, color: &str) -> String {
    if supports_color() {
        format!("{color}{text}{}", colors::RESET)
    } else {
        text.to_string()
    }
}

/// Nearest xterm-256 index for a #rrggbb colour. ABSOLUTE 256 (ESC[38;5;Nm)
/// bypasses the terminal's palette remap — named ANSI colours collapse under a
/// theme; 256 does not.
pub fn hex_to_ansi256(hexstr: &str) -> u8 {
    let h = hexstr.trim_start_matches('#');
    let parse = |a: usize| u32::from_str_radix(h.get(a..a + 2).unwrap_or("00"), 16).unwrap_or(0);
    let (r, g, b) = (parse(0), parse(2), parse(4));
    if r == g && g == b {
        // grayscale ramp (232-255)
        if r < 8 {
            return 16;
        }
        if r > 248 {
            return 231;
        }
        return (((r - 8) as f64 / 247.0 * 24.0).round() as u32 + 232) as u8;
    }
    let levels = [0u32, 95, 135, 175, 215, 255]; // 6x6x6 colour cube
    let lvl = |v: u32| {
        (0..6usize)
            .min_by_key(|i| (levels[*i] as i64 - v as i64).abs())
            .unwrap_or(0) as u32
    };
    (16 + 36 * lvl(r) + 6 * lvl(g) + lvl(b)) as u8
}

/// SGR escape setting foreground to the nearest 256-colour for a hex value.
pub fn fg256_hex(hexstr: &str) -> String {
    format!("\x1b[38;5;{}m", hex_to_ansi256(hexstr))
}

/// Format number with comma separators.
pub fn format_number(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// Print text safely, handling encoding errors (a lossy fallback keeps the
/// program alive on a console that can't take the bytes).
pub fn safe_print(text: &str) {
    // Routed through term_out, never straight to stdout: while the interactive
    // REPL is running, the prompt engine owns the bottom of the screen, and a
    // direct write lands INSIDE that block and is erased by the next repaint —
    // which silently swallowed the model's answer. This is the Rust stand-in
    // for the Python's patch_stdout, and every display path must go through it.
    crate::utils::term_out::emit(&format!("{text}\n"));
}

// Escape sequences + control bytes that corrupt the terminal when RAW tool
// output is echoed to the console. Applied ONLY to untrusted tool-result text
// before display — never to our own colorized output.
static RE_ANSI_CSI: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").unwrap());
static RE_ANSI_OSC: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)").unwrap());
static RE_ANSI_OTHER: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\x1b[@-Z\\-_]").unwrap());
static RE_CTRL: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\u{9f}]").unwrap());

/// Strip terminal-corrupting escape sequences and control bytes from untrusted
/// text, keeping \t \n \r and all printable characters.
pub fn sanitize_terminal_text(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let t = RE_ANSI_CSI.replace_all(text, "");
    let t = RE_ANSI_OSC.replace_all(&t, "");
    let t = RE_ANSI_OTHER.replace_all(&t, "");
    RE_CTRL.replace_all(&t, "").into_owned()
}

static COLOR_SYSTEM_CACHE: Lazy<Mutex<Option<&'static str>>> = Lazy::new(|| Mutex::new(None));

/// Pick the color depth the terminal actually advertises (truecolor only if
/// COLORTERM says so; 256 for *-256color; else 16-color 'standard'). Memoized.
pub fn detect_color_system() -> &'static str {
    let mut cache = COLOR_SYSTEM_CACHE.lock();
    if let Some(v) = *cache {
        return v;
    }
    let v = if matches!(
        std::env::var("COLORTERM").unwrap_or_default().to_lowercase().as_str(),
        "truecolor" | "24bit"
    ) {
        "truecolor"
    } else {
        let term = std::env::var("TERM").unwrap_or_default();
        if term.contains("256color") || term.contains("256") {
            "256"
        } else {
            "standard"
        }
    };
    *cache = Some(v);
    v
}

/// Render markdown to ANSI for the terminal (delegates to utils::markdown, the
/// rich-equivalent renderer). Falls back to the raw text when color isn't
/// supported (piped / non-tty / plain_mode).
pub fn render_markdown(text: &str) -> String {
    if text.is_empty() || !supports_color() {
        return text.to_string();
    }
    crate::utils::markdown::render_markdown_ansi(text)
}

// === Diff Formatting ===

struct DiffBody {
    num_additions: usize,
    num_removals: usize,
    lines: Vec<String>,
}

/// Shared diff-body renderer for format_edit_diff / format_edit_diff_preview.
fn render_diff_body(old_string: &str, new_string: &str, start_line: usize) -> DiffBody {
    let old_lines: Vec<&str> = old_string.split('\n').collect();
    let new_lines: Vec<&str> = new_string.split('\n').collect();

    let diff_lines = crate::utils::difflib::unified_diff(&old_lines, &new_lines, 3);

    let num_additions = diff_lines
        .iter()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
        .count();
    let num_removals = diff_lines
        .iter()
        .filter(|l| l.starts_with('-') && !l.starts_with("---"))
        .count();

    let has_color = supports_color();
    let mut diff_output = Vec::new();
    let hunk_lines: Vec<&String> = diff_lines
        .iter()
        .filter(|l| !l.starts_with("---") && !l.starts_with("+++"))
        .collect();

    let mut old_line = start_line;
    let mut new_line = start_line;
    let max_line_num = start_line + old_lines.len() + new_lines.len();
    let line_num_width = max_line_num.to_string().len();

    for line in hunk_lines {
        if line.starts_with("@@") {
            // Re-anchor the counters to THIS hunk's start offsets. Without this
            // a multi-hunk diff kept counting sequentially across the omitted
            // gap, so every line number in the 2nd+ hunk was too low.
            if let Some(m) = DIFF_HUNK_RE.captures(line) {
                old_line = start_line + m[1].parse::<usize>().unwrap_or(1) - 1;
                new_line = start_line + m[2].parse::<usize>().unwrap_or(1) - 1;
            }
            continue;
        } else if let Some(content) = line.strip_prefix('-') {
            let ln = format!("{:>width$}", old_line, width = line_num_width);
            if has_color {
                diff_output.push(format!(
                    "     {}{ln}{} {}{}-{content}{}",
                    colors::DIM,
                    colors::RESET,
                    colors::BG_RED,
                    colors::RED,
                    colors::RESET
                ));
            } else {
                diff_output.push(format!("     {ln} -{content}"));
            }
            old_line += 1;
        } else if let Some(content) = line.strip_prefix('+') {
            let ln = format!("{:>width$}", new_line, width = line_num_width);
            if has_color {
                diff_output.push(format!(
                    "     {}{ln}{} {}{}+{content}{}",
                    colors::DIM,
                    colors::RESET,
                    colors::BG_GREEN,
                    colors::GREEN,
                    colors::RESET
                ));
            } else {
                diff_output.push(format!("     {ln} +{content}"));
            }
            new_line += 1;
        } else if let Some(content) = line.strip_prefix(' ') {
            let ln = format!("{:>width$}", new_line, width = line_num_width);
            if has_color {
                diff_output.push(format!(
                    "     {}{ln}{}  {content}",
                    colors::DIM,
                    colors::RESET
                ));
            } else {
                diff_output.push(format!("     {ln}  {content}"));
            }
            old_line += 1;
            new_line += 1;
        }
    }

    DiffBody { num_additions, num_removals, lines: diff_output }
}

fn additions_removals_summary(num_additions: usize, num_removals: usize) -> String {
    let mut parts = Vec::new();
    if num_additions > 0 {
        parts.push(format!(
            "{num_additions} addition{}",
            if num_additions != 1 { "s" } else { "" }
        ));
    }
    if num_removals > 0 {
        parts.push(format!(
            "{num_removals} removal{}",
            if num_removals != 1 { "s" } else { "" }
        ));
    }
    if parts.is_empty() {
        "no changes".to_string()
    } else {
        parts.join(" and ")
    }
}

/// Format an edit operation as a colored diff display, Claude Code style.
/// (Reads the file AFTER the edit to locate new_string for line numbers.)
pub fn format_edit_diff(file_path: &str, old_string: &str, new_string: &str) -> String {
    let filename = std::path::Path::new(file_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| file_path.to_string());

    let start_line = std::fs::read_to_string(file_path)
        .ok()
        .and_then(|content| {
            content
                .find(new_string)
                .map(|pos| content[..pos].matches('\n').count() + 1)
        })
        .unwrap_or(1);

    let body = render_diff_body(old_string, new_string, start_line);
    let summary = additions_removals_summary(body.num_additions, body.num_removals);
    let mut result = vec![format!(
        "  {} Updated {} with {summary}",
        sym("connector"),
        colorize(&filename, colors::BOLD)
    )];
    if !body.lines.is_empty() {
        result.push(String::new());
        result.extend(body.lines);
    }
    result.join("\n")
}

/// Format diff preview BEFORE edit is applied (for permission request).
/// Unlike format_edit_diff(), this reads the ORIGINAL file to find line numbers.
pub fn format_edit_diff_preview(file_path: &str, old_string: &str, new_string: &str) -> String {
    let filename = std::path::Path::new(file_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| file_path.to_string());

    let start_line = std::fs::read_to_string(file_path)
        .ok()
        .and_then(|content| {
            content
                .find(old_string)
                .map(|pos| content[..pos].matches('\n').count() + 1)
        })
        .unwrap_or(1);

    let body = render_diff_body(old_string, new_string, start_line);
    let summary = additions_removals_summary(body.num_additions, body.num_removals);
    let mut result = vec![format!(
        "  {} {} {summary}",
        colorize(&filename, colors::DIM),
        sym("dash")
    )];
    if !body.lines.is_empty() {
        result.push(String::new());
        result.extend(body.lines);
    }
    result.join("\n")
}

// === Tool Display ===

/// Every tool renders its own tool-use line; this is only the fallback for a
/// tool that defines none: show the argument that identifies the call, else a
/// compact key=value line — never a raw dict repr.
const PRIMARY_ARG_KEYS: &[&str] = &[
    "command", "query", "url", "pattern", "file_path", "notebook_path", "path", "task_id",
    "shell_id", "skill",
];

fn value_brief(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn is_empty_value(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

pub fn summarize_tool_input(tool_input: &Value, max_len: usize) -> String {
    let mut text = String::new();
    if let Value::Object(map) = tool_input {
        for key in PRIMARY_ARG_KEYS {
            if let Some(v) = map.get(*key) {
                if !is_empty_value(v) {
                    text = value_brief(v);
                    break;
                }
            }
        }
        if text.is_empty() {
            text = map
                .iter()
                .filter(|(_, v)| !is_empty_value(v))
                .map(|(k, v)| format!("{k}={}", value_brief(v)))
                .collect::<Vec<_>>()
                .join(" ");
        }
    } else if !tool_input.is_null() {
        text = value_brief(tool_input);
    }
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > max_len {
        format!("{}…", chars[..max_len].iter().collect::<String>())
    } else {
        text
    }
}

// === Token Display ===

/// Compact token count: 2600 -> '2.6k', 10000 -> '10k', 262144 -> '262.1k',
/// 999 -> '999'.
pub fn format_tokens_compact(n: i64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = format!("{:.1}k", n as f64 / 1000.0);
    if s.ends_with(".0k") {
        format!("{}k", &s[..s.len() - 3])
    } else {
        s
    }
}

/// Format token usage for display.
pub fn format_token_display(
    usage: Option<&crate::types::messages::BetaUsage>,
    duration_ms: i64,
    show_details: bool,
    fallback_tokens: i64,
) -> String {
    let (input_tokens, output_tokens, cache_read, cache_create) = match usage {
        Some(u) => (
            u.input_tokens,
            u.output_tokens,
            u.cache_read_input_tokens,
            u.cache_creation_input_tokens,
        ),
        None => (0, 0, 0, 0),
    };

    let mut total_tokens = input_tokens + output_tokens + cache_read + cache_create;
    if total_tokens == 0 && fallback_tokens > 0 {
        total_tokens = fallback_tokens;
    }

    let duration_sec = duration_ms as f64 / 1000.0;

    if show_details {
        let mut parts = vec![format!("{}{}", sym("down"), format_tokens_compact(input_tokens))];
        if cache_read > 0 {
            parts.push(format!("(+{} cached)", format_tokens_compact(cache_read)));
        }
        parts.push(format!(
            "{} {}{}",
            sym("arrow"),
            sym("up"),
            format_tokens_compact(output_tokens)
        ));
        parts.push(format!("[{duration_sec:.1}s]"));
        parts.join(" ")
    } else {
        // Default footer: how full the context window is, compact k form —
        // measured against the AUTO-COMPACT THRESHOLD, not the raw window (the
        // number you can act on; matches the startup banner).
        let limit = crate::services::compaction::get_auto_compact_threshold()
            .unwrap_or_else(|| crate::utils::tokens::get_context_size());
        let pct = if limit > 0 {
            format!(" ({}%)", ((total_tokens as f64 / limit as f64) * 100.0).round() as i64)
        } else {
            String::new()
        };
        format!(
            "{} {} {} / {}{pct} {} {duration_sec:.1}s",
            sym("rule"),
            sym("down"),
            format_tokens_compact(total_tokens),
            format_tokens_compact(limit),
            sym("dot")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_compact() {
        assert_eq!(format_tokens_compact(999), "999");
        assert_eq!(format_tokens_compact(2600), "2.6k");
        assert_eq!(format_tokens_compact(10000), "10k");
        assert_eq!(format_tokens_compact(262144), "262.1k");
    }

    #[test]
    fn hex256() {
        assert_eq!(hex_to_ansi256("#000000"), 16);
        assert_eq!(hex_to_ansi256("#ffffff"), 231);
    }

    #[test]
    fn summarize_primary_key() {
        let v = serde_json::json!({"command": "ls -la", "description": "list"});
        assert_eq!(summarize_tool_input(&v, 160), "ls -la");
    }
}

#[cfg(test)]
mod clip_tests {
    use super::*;

    #[test]
    fn clip_keeps_short_lines_intact() {
        assert_eq!(clip_to_width("abc", 10), "abc");
        assert_eq!(clip_to_width("", 10), "");
    }

    #[test]
    fn clip_cuts_at_cell_width() {
        assert_eq!(clip_to_width("abcdef", 3), "abc");
        // CJK cells are two wide: only one fits in three cells
        assert_eq!(clip_to_width("한글", 3), "한");
        assert_eq!(clip_to_width("한글", 4), "한글");
    }

    #[test]
    fn clip_passes_sgr_through_and_closes_a_cut_style() {
        // styling costs no cells
        assert_eq!(clip_to_width("\x1b[1mabc\x1b[0m", 3), "\x1b[1mabc\x1b[0m");
        // cutting inside styled text closes the style
        assert_eq!(clip_to_width("\x1b[1mabcdef", 3), "\x1b[1mabc\x1b[0m");
        // unstyled text is left alone
        assert_eq!(clip_to_width("abcdef", 3), "abc");
    }

    /// Regression: the resume picker's header is the line that overflows a phone
    /// terminal, because "↑/↓" are East-Asian *ambiguous* and take two cells
    /// under a CJK locale.  Wrapped, it desynced the ESC[nA repaint and the list
    /// marched down the screen once per second.
    #[test]
    fn picker_header_is_clipped_to_the_window() {
        let header = "Resume a session (↑/↓ move, Enter select, d delete, Esc cancel)";
        for width in [30usize, 40, 55, 62, 64, 80] {
            let clipped = clip_to_width(header, width);
            assert!(
                display_width(&clipped) <= width,
                "header still exceeds {width} cells"
            );
        }
    }
}

#[cfg(test)]
mod menu_clip_tests {
    use super::*;

    /// Regression: a completion entry whose meta is a full sentence (skill
    /// commands) is far wider than a phone terminal.  Wrapped, it took two
    /// PHYSICAL rows while the prompt engine counted one LOGICAL line, so its
    /// erase walked back a row short and the block marched down the screen on
    /// every repaint — typing "/t" reproduced it instantly.
    #[test]
    fn long_completion_entry_fits_the_window() {
        let entry = "  /test-driven-development  \x1b[38;5;245mTDD — enforce RED → GREEN → REFACTOR, write the test before the code.\x1b[0m";
        for cols in [40usize, 62, 80] {
            let clipped = clip_to_width(entry, cols);
            let bare = clipped.replace("\x1b[38;5;245m", "").replace("\x1b[0m", "");
            assert!(
                display_width(&bare) <= cols,
                "entry still exceeds {cols} cells: {} ", display_width(&bare)
            );
        }
    }
}
