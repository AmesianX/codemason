//! DSML-safe incremental stream filter for realtime (-rt/--realtime) rendering.
//! Equivalent to codemason/utils/realtime_render.py — see that file's module
//! doc for what must never reach the screen (DSML tool-call markup in both
//! polarities, leaked <think> blocks, <|special|> control tokens) and why the
//! filter holds back a trailing fragment that could still grow into a marker.

use once_cell::sync::Lazy;
use regex::Regex;

const PIPE: &str = "[|｜]";

/// Triggers that, once seen, suppress the REST of the message (tool call
/// body). BOTH polarities: the server's tool-call parser can terminate a call
/// early and spill the remainder into `content`, so the message can arrive
/// carrying only the ORPHAN closing tags.
static DSML_START: Lazy<Regex> =
    Lazy::new(|| Regex::new(&format!("^</?{PIPE}DSML{PIPE}")).unwrap());
/// Inline strippable control token: <|word|> (ascii or full-width pipe).
static SPECIAL_TOKEN: Lazy<Regex> =
    Lazy::new(|| Regex::new(&format!("^<{PIPE}[^|｜<>]*{PIPE}>")).unwrap());
/// Bare-XML tool-call markup, the form GLM emits: <tool_call>, <arg_key>,
/// <arg_value> and their closers, with no pipes anywhere.
///
/// Everything else stripped here is pipe-delimited (<|DSML|>, <|word|>), and
/// this family matched none of it. Observed 2026-09-07: a turn whose visible
/// answer ended "dessin</arg_value></tool_call>" — the call itself had been
/// parsed correctly on the structured channel and its commands ran, so only
/// the rendered text was wrong, but it puts unexplained markup in the middle
/// of the model's prose. The module comment already noted that a message can
/// arrive carrying ORPHAN closing tags; it only ever handled the DSML spelling.
static XML_TOOL_TAG: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^</?(?:tool_call|tool_response|arg_key|arg_value|args)\s*>").unwrap()
});

static THINK_OPEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)^<think>").unwrap());
static THINK_CLOSE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)</think>").unwrap());
/// Prefix of a special/DSML token: '<' (optionally '/') + pipe + letters, no
/// closing yet.
static MARKER_PREFIX: Lazy<Regex> =
    Lazy::new(|| Regex::new(&format!("^</?{PIPE}[^|｜<>]*$")).unwrap());

/// Longest marker this filter can ever hold for, in bytes.
///
/// The XML family is a fixed list (longest: "</tool_response>", 16 bytes), but
/// SPECIAL_TOKEN / MARKER_PREFIX are `<|` + arbitrary text + `|>` — the grammar
/// puts no bound on the middle, so a `<|` whose closing pipe never arrives
/// would hold the ENTIRE rest of the stream in `pending`, and every later delta
/// re-lowercases all of it. That is O(n^2) on a buffer that never stops
/// growing, i.e. a live hang, not just slow rendering.
///
/// Measured: across both bundled tokenizers (DSV4-Flash, glm-5.3-flash) the
/// longest special token is 29 bytes (`<|place_holder_mm_span_0435|>`). 64
/// leaves better than 2x headroom, so no real marker is ever cut short, while
/// anything longer is treated as what it is — prose that happens to start '<'.
const MAX_MARKER_BYTES: usize = 64;

/// True if s (starting with '<') might still grow into a suppressible/strip
/// marker, so we should hold it instead of emitting.
fn could_be_marker_prefix(s: &str) -> bool {
    // Bound the hold BEFORE any whole-string work (the to_lowercase below).
    if s.len() > MAX_MARKER_BYTES {
        return false;
    }
    if MARKER_PREFIX.is_match(s) {
        return true;
    }
    if s == "<" || s == "</" {
        return true;
    }
    // Prefix of <think> / </think> or of a bare-XML tool tag. Held rather than
    // emitted, so a tag split across two stream chunks is not printed in
    // halves before it can be recognised.
    let lower = s.to_lowercase();
    for tag in [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<args>",
        "</args>",
    ] {
        if tag.starts_with(&lower) && s.chars().count() < tag.len() {
            return true;
        }
    }
    false
}

/// Length (in bytes) of the longest suffix of s that is a proper prefix of
/// marker (case-insensitive).
fn tail_partial(s: &str, marker: &str) -> usize {
    // Measured against the ORIGINAL string, not a lowercased copy. The caller
    // slices `s` at the returned offset, and to_lowercase() is not
    // length-preserving — U+212A (KELVIN SIGN) is three bytes and lowercases to
    // a one-byte 'k' — so a byte count taken from the lowercased form could
    // land inside a character of `s` and panic the moment a model chunk ended
    // with one. Walking `s`'s own char boundaries makes every returned value a
    // valid split point by construction.
    let marker_lower = marker.to_lowercase();
    // Only a suffix shorter than the marker can be a PROPER prefix of it, so
    // there is no reason to look further back than that.
    let start = s.len().saturating_sub(marker_lower.len());
    for (idx, _) in s.char_indices() {
        if idx < start {
            continue;
        }
        let suffix = &s[idx..];
        if suffix.len() >= marker_lower.len() {
            continue;
        }
        if marker_lower.starts_with(&suffix.to_lowercase()) {
            return suffix.len();
        }
    }
    0
}

#[derive(Default)]
pub struct RealtimeStreamFilter {
    /// unclassified tail (may hold a partial marker)
    pending: String,
    /// DSML tool call started → drop everything after
    suppress_all: bool,
    /// inside <think> … </think> → drop inner
    in_think: bool,
}

impl RealtimeStreamFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns text safe to print now.
    pub fn feed(&mut self, delta: &str) -> String {
        if self.suppress_all || delta.is_empty() {
            return String::new();
        }
        self.pending.push_str(delta);
        let mut out = String::new();
        while !self.pending.is_empty() {
            if self.in_think {
                match THINK_CLOSE.find(&self.pending) {
                    Some(m) => {
                        self.pending = self.pending[m.end()..].to_string();
                        self.in_think = false;
                        continue;
                    }
                    None => {
                        // keep a tail that could be a partial '</think>'
                        let keep = tail_partial(&self.pending, "</think>");
                        self.pending = if keep > 0 {
                            self.pending[self.pending.len() - keep..].to_string()
                        } else {
                            String::new()
                        };
                        return out;
                    }
                }
            }

            match self.pending.find('<') {
                None => {
                    out.push_str(&self.pending);
                    self.pending.clear();
                    break;
                }
                Some(lt) => {
                    // emit safe text before the '<'
                    if lt > 0 {
                        out.push_str(&self.pending[..lt]);
                        self.pending = self.pending[lt..].to_string();
                    }
                    // now pending starts with '<'
                    if DSML_START.is_match(&self.pending) {
                        self.suppress_all = true;
                        self.pending.clear();
                        break;
                    }
                    if let Some(m) = THINK_OPEN.find(&self.pending) {
                        self.pending = self.pending[m.end()..].to_string();
                        self.in_think = true;
                        continue;
                    }
                    if let Some(m) = SPECIAL_TOKEN.find(&self.pending) {
                        self.pending = self.pending[m.end()..].to_string(); // strip <|...|>
                        continue;
                    }
                    // Bare-XML tool markup: strip the tag, keep going. Unlike
                    // DSML this does NOT suppress the rest — the leak seen in
                    // practice was orphan CLOSING tags after real prose, and
                    // swallowing everything behind them would eat the answer.
                    if let Some(m) = XML_TOOL_TAG.find(&self.pending) {
                        self.pending = self.pending[m.end()..].to_string();
                        continue;
                    }
                    // incomplete? hold and wait for more.
                    if could_be_marker_prefix(&self.pending) {
                        return out;
                    }
                    // definitely not a marker — the '<' is real text
                    out.push('<');
                    self.pending = self.pending[1..].to_string();
                }
            }
        }
        out
    }

    /// End of stream: any held-back tail never completed a marker, so it was
    /// real text (unless we're mid-DSML/think, in which case it's dropped).
    pub fn flush(&mut self) -> String {
        if self.suppress_all || self.in_think {
            self.pending.clear();
            return String::new();
        }
        let rest = std::mem::take(&mut self.pending);
        // drop only if it's a bare incomplete control-token opener. Same bound
        // as could_be_marker_prefix: past it, this is prose and must survive.
        if rest.len() <= MAX_MARKER_BYTES && MARKER_PREFIX.is_match(&rest) {
            return String::new();
        }
        // A complete tool tag held to the very end is markup, not text.
        if XML_TOOL_TAG.is_match(&rest) {
            return String::new();
        }
        rest
    }
}

/// Groups streamed safe text into complete MARKDOWN BLOCKS so each can be
/// rendered (render_markdown) and committed as a whole — instead of raw
/// per-char writes. A block is a blank-line-delimited paragraph, or a whole
/// ``` fenced code block (blank lines inside a fence are NOT boundaries).
/// feed() returns the blocks that are now complete; flush() returns the
/// trailing partial block at stream end.
#[derive(Default)]
pub struct MarkdownBlockBuffer {
    /// bytes not yet split into complete lines
    pending: String,
    /// complete lines of the block being built
    cur: Vec<String>,
    in_fence: bool,
    /// the opening fence marker (``` or ~~~)
    fence: String,
}

impl MarkdownBlockBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.pending.push_str(text);
        let mut blocks = Vec::new();
        while let Some(nl) = self.pending.find('\n') {
            let line = self.pending[..nl].to_string();
            self.pending = self.pending[nl + 1..].to_string();
            let stripped = line.trim_start();
            if stripped.starts_with("```") || stripped.starts_with("~~~") {
                if !self.in_fence {
                    self.in_fence = true;
                    self.fence = stripped.chars().take(3).collect();
                    self.cur.push(line);
                } else if stripped.starts_with(&self.fence) {
                    self.in_fence = false;
                    self.cur.push(line);
                    blocks.push(self.cur.join("\n")); // complete code block
                    self.cur.clear();
                } else {
                    self.cur.push(line);
                }
                continue;
            }
            if !self.in_fence && line.trim().is_empty() {
                let body = self.cur.join("\n");
                let body = body.trim_matches('\n');
                if !body.trim().is_empty() {
                    blocks.push(body.to_string()); // complete paragraph
                }
                self.cur.clear();
                continue;
            }
            self.cur.push(line);
        }
        blocks
    }

    pub fn flush(&mut self) -> String {
        if !self.pending.is_empty() {
            self.cur.push(std::mem::take(&mut self.pending));
        }
        let body = self.cur.join("\n");
        self.cur.clear();
        body.trim_matches('\n').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A '<|' whose closing pipe never arrives must NOT hold the stream.
    /// Before the bound, MARKER_PREFIX matched any length, so `pending` grew
    /// without limit and every delta re-lowercased the whole buffer — O(n^2)
    /// on a growing buffer, which is a hang, not slow rendering.
    #[test]
    fn an_unclosed_pipe_marker_does_not_hold_the_stream() {
        let mut f = RealtimeStreamFilter::new();
        // Well under the bound: still held, so a split marker is not printed
        // in halves.
        assert_eq!(f.feed("<|tool"), "");

        // Past the bound: this can no longer become any real special token
        // (longest measured is 29 bytes), so it must be emitted as prose.
        let mut f = RealtimeStreamFilter::new();
        let long = format!("<|{}", "a".repeat(4000));
        let out = f.feed(&long) + &f.flush();
        assert_eq!(out, long, "prose starting '<|' must survive, not be held");
    }

    /// tail_partial's result is used to slice the ORIGINAL string, so it has to
    /// be a byte offset that is valid there.
    ///
    /// It used to be measured on a lowercased copy, and to_lowercase() is not
    /// length-preserving: U+212A (KELVIN SIGN) is three bytes and becomes a
    /// one-byte 'k'. A chunk ending "</THIN\u{212a}" lowercased to "</think",
    /// giving keep = 7 — seven bytes back from the end of the original lands
    /// inside the three-byte character, and the slice panics. The stream is
    /// model-controlled, so this is reachable input, not a hypothetical.
    #[test]
    fn tail_partial_offsets_are_valid_in_the_original() {
        // The exact hazard: a character that shrinks when lowercased.
        let s = "text </THIN\u{212a}";
        let keep = tail_partial(s, "</think>");
        assert!(
            s.is_char_boundary(s.len() - keep),
            "offset {keep} must be a char boundary in the original"
        );
        // And the slice the caller performs must not panic.
        let _ = &s[s.len() - keep..];

        // Ordinary behaviour is unchanged: a real partial marker is kept.
        let s = "hello </thi";
        let keep = tail_partial(s, "</think>");
        assert_eq!(&s[s.len() - keep..], "</thi");
        // Case-insensitive, as before.
        let s = "hello </THI";
        let keep = tail_partial(s, "</think>");
        assert_eq!(&s[s.len() - keep..], "</THI");
        // A complete marker is NOT a proper prefix, so nothing is held back.
        assert_eq!(tail_partial("x </think>", "</think>"), 0);
        // No partial marker at all.
        assert_eq!(tail_partial("plain text", "</think>"), 0);
        // Multi-byte tail that is not a marker must still return a valid offset.
        let s = "안녕하세요";
        let keep = tail_partial(s, "</think>");
        assert!(s.is_char_boundary(s.len() - keep));
    }

    #[test]
    fn dsml_suppresses_rest() {
        let mut f = RealtimeStreamFilter::new();
        let out = f.feed("hello <｜DSML｜invoke name=\"Bash\"> more");
        assert_eq!(out, "hello ");
        assert_eq!(f.feed("anything"), "");
        assert_eq!(f.flush(), "");
    }

    #[test]
    fn orphan_closing_tag_suppresses() {
        let mut f = RealtimeStreamFilter::new();
        let out = f.feed("text </｜DSML｜parameter> spill");
        assert_eq!(out, "text ");
    }

    #[test]
    fn think_block_dropped_across_chunks() {
        let mut f = RealtimeStreamFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("a<thi"));
        out.push_str(&f.feed("nk>hidden</th"));
        out.push_str(&f.feed("ink>b"));
        out.push_str(&f.flush());
        assert_eq!(out, "ab");
    }

    #[test]
    fn special_token_stripped_split() {
        let mut f = RealtimeStreamFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("x<|en"));
        out.push_str(&f.feed("d|>y"));
        out.push_str(&f.flush());
        assert_eq!(out, "xy");
    }

    #[test]
    fn real_lt_is_emitted() {
        let mut f = RealtimeStreamFilter::new();
        let mut out = String::new();
        out.push_str(&f.feed("a < b and 1<2"));
        out.push_str(&f.flush());
        assert_eq!(out, "a < b and 1<2");
    }

    #[test]
    fn block_buffer_paragraphs_and_fences() {
        let mut b = MarkdownBlockBuffer::new();
        let blocks = b.feed("para1 line\n\n```py\ncode\n\nmore\n```\npara2");
        assert_eq!(blocks, vec!["para1 line".to_string(), "```py\ncode\n\nmore\n```".to_string()]);
        assert_eq!(b.flush(), "para2");
    }
}

#[cfg(test)]
mod xml_tool_tag_tests {
    use super::*;

    /// Orphan tool-call markup must not reach the screen as prose. Observed
    /// 2026-09-07: an answer ended "dessin</arg_value></tool_call>". Every
    /// other marker stripped here is pipe-delimited, so this bare-XML family
    /// matched nothing.
    #[test]
    fn orphan_tool_tags_are_stripped_without_eating_the_answer() {
        let mut f = RealtimeStreamFilter::new();
        let out = f.feed("dessin</arg_value></tool_call>") + &f.flush();
        assert_eq!(out.trim(), "dessin", "got {out:?}");
    }

    /// Split across chunks the way the stream actually delivers it.
    #[test]
    fn a_tag_split_across_chunks_is_still_stripped() {
        let mut f = RealtimeStreamFilter::new();
        let mut out = String::new();
        for chunk in ["ok</arg_", "value><tool_", "call>rest"] {
            out.push_str(&f.feed(chunk));
        }
        out.push_str(&f.flush());
        assert!(!out.contains("arg_value"), "leaked: {out:?}");
        assert!(!out.contains("tool_call"), "leaked: {out:?}");
        assert!(out.contains("ok") && out.contains("rest"), "lost text: {out:?}");
    }

    /// A '<' that is ordinary text still survives.
    #[test]
    fn real_angle_brackets_are_untouched() {
        let mut f = RealtimeStreamFilter::new();
        let out = f.feed("if a < b && c > d then") + &f.flush();
        assert!(out.contains("a < b"), "got {out:?}");
    }
}
