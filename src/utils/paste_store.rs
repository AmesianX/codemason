//! Pasted-text placeholder store — based on Claude Code's history.ts.
//! Equivalent to codemason/utils/paste_store.py.
//!
//! A large or multi-line paste is replaced in the input box by a compact
//! `[Pasted text #N +M lines]` placeholder; the full text is kept here and
//! spliced back in when the message is actually submitted to the model.
//! Image paste is intentionally out of scope (needs a vision-capable model).

use once_cell::sync::Lazy;
use std::collections::BTreeMap;

/// Per-session id -> full pasted text. Ids count up from 1 and are stable for
/// the life of the REPL so a placeholder can always be expanded.
#[derive(Default)]
pub struct PasteStore {
    store: BTreeMap<u64, String>,
    next_id: u64,
}

/// Cap the retained pastes so a very long REPL session can't grow the store
/// without bound. Past this many pastes the oldest is dropped.
const MAX_PASTES: usize = 1000;

impl PasteStore {
    pub fn new() -> Self {
        PasteStore { store: BTreeMap::new(), next_id: 1 }
    }

    pub fn add(&mut self, text: &str) -> u64 {
        let pid = self.next_id;
        self.next_id += 1;
        self.store.insert(pid, text.to_string());
        if self.store.len() > MAX_PASTES {
            if let Some(&oldest) = self.store.keys().next() {
                self.store.remove(&oldest); // evict oldest id
            }
        }
        pid
    }

    pub fn get(&self, pid: u64) -> Option<&String> {
        self.store.get(&pid)
    }

    pub fn contains(&self, pid: u64) -> bool {
        self.store.contains_key(&pid)
    }
}

/// Number of line breaks (`line1\nline2\nline3` -> 2), matching history.ts's
/// getPastedTextRefNumLines (it counts breaks, not lines).
pub fn pasted_num_lines(text: &str) -> usize {
    static RE: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\r\n|\r|\n").unwrap());
    RE.find_iter(text).count()
}

/// `[Pasted text #N]` for a single line, `[Pasted text #N +M lines]` else.
pub fn format_pasted_ref(pid: u64, num_lines: usize) -> String {
    if num_lines == 0 {
        format!("[Pasted text #{pid}]")
    } else {
        format!("[Pasted text #{pid} +{num_lines} lines]")
    }
}

/// Matches the placeholder we emit. (history.ts also parses Image / Truncated
/// text refs; we only handle text paste here.)
static REF_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\[Pasted text #(\d+)(?: \+\d+ lines)?\]").unwrap());

/// Return (pid, matched_str, start_index) for each placeholder in `text`.
pub fn parse_references(text: &str) -> Vec<(u64, String, usize)> {
    REF_RE
        .captures_iter(text)
        .filter_map(|m| {
            let pid: u64 = m[1].parse().ok()?;
            if pid > 0 {
                let whole = m.get(0).unwrap();
                Some((pid, whole.as_str().to_string(), whole.start()))
            } else {
                None
            }
        })
        .collect()
}

/// Replace `[Pasted text #N]` placeholders with their stored full content.
/// Splices at the original offsets in reverse so earlier indices stay valid,
/// and so placeholder-looking strings inside pasted content are never
/// re-expanded (faithful to history.ts's expandPastedTextRefs).
pub fn expand_pasted_refs(text: &str, store: &PasteStore) -> String {
    let refs = parse_references(text);
    let mut expanded = text.to_string();
    for (pid, matched, index) in refs.iter().rev() {
        let Some(content) = store.get(*pid) else {
            continue; // unknown id (e.g. restored history) — leave as-is
        };
        expanded.replace_range(*index..*index + matched.len(), content);
    }
    expanded
}

/// A paste becomes a placeholder when it spans multiple lines OR is longer
/// than this many characters. Small single-line pastes stay inline.
pub const PASTE_PLACEHOLDER_MIN_CHARS: usize = 200;

pub fn should_placeholder(text: &str) -> bool {
    pasted_num_lines(text) >= 1 || text.len() > PASTE_PLACEHOLDER_MIN_CHARS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_expand() {
        let mut store = PasteStore::new();
        let pid = store.add("full\ncontent");
        let ph = format_pasted_ref(pid, pasted_num_lines("full\ncontent"));
        assert_eq!(ph, "[Pasted text #1 +1 lines]");
        let msg = format!("before {ph} after");
        assert_eq!(expand_pasted_refs(&msg, &store), "before full\ncontent after");
    }

    #[test]
    fn nested_placeholder_not_reexpanded() {
        let mut store = PasteStore::new();
        let inner = store.add("INNER");
        let _ = inner;
        let outer = store.add("contains [Pasted text #1]");
        let msg = format_pasted_ref(outer, 0);
        assert_eq!(expand_pasted_refs(&msg, &store), "contains [Pasted text #1]");
    }
}
