//! SessionSearch tool — full-text recall over this project's past sessions.
//! Equivalent to codemason/tools/session_search_tool.py.
//!
//! The local counterpart of claude.ai's conversation_search: transcripts are
//! JSONL files on disk here, so past-conversation recall is a search over them
//! (utils/session_index.rs owns the index). The tool is read-only and scoped
//! to the current project directory's sessions; the current session is never
//! in the results.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::i18n::{tr, trf, FmtArg};
use crate::utils::session_index::{search, SearchGroup};

pub const SESSION_SEARCH_TOOL_NAME: &str = "SessionSearch";

#[allow(dead_code)]
const MAX_LIMIT: usize = 20;

static SCHEMA: Lazy<Value> = Lazy::new(|| {
    let all: Value =
        serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default();
    all.get(SESSION_SEARCH_TOOL_NAME)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
});

/// First `n` CHARACTERS, not bytes.
///
/// `len()` counts bytes, so a byte-length check says nothing about where a
/// character ends: `&s[..8]` on a string whose 8th byte falls inside a
/// multi-byte character panics with "byte index is not a char boundary". Both
/// fields sliced below are written by codemason as ASCII (a uuid, an ISO-8601
/// timestamp), but they are READ BACK from the session store, and this tool's
/// whole job is to display whatever is on disk — a truncated write, a
/// hand-edited transcript or a file from another version is enough. Cutting on
/// character boundaries costs nothing and cannot fail.
fn first_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

fn short_id(session_id: &str) -> &str {
    first_chars(session_id, 8)
}

/// Timestamps are ISO-8601 Z; the date part is enough to place a hit in time.
fn day(ts: &str) -> &str {
    first_chars(ts, 10)
}

fn format_for_assistant(groups: &[SearchGroup], query: &str) -> String {
    if groups.is_empty() {
        return format!(
            "No past-session messages match \"{query}\" in this project. \
             Loosen the query (fewer words, or a * prefix wildcard) before \
             concluding the topic was never discussed here."
        );
    }
    let hit_count: usize = groups.iter().map(|g| g.hits.len()).sum();
    let mut lines = vec![format!(
        "Found {hit_count} matching message(s) in {} past session(s) \
         for \"{query}\". Matched terms are [bracketed]; excerpts are from this \
         project's transcripts, best match first.",
        groups.len()
    )];
    for (i, g) in groups.iter().enumerate() {
        let title = if g.first_prompt.is_empty() { "(no prompt)" } else { &g.first_prompt };
        let newest = g.hits.iter().map(|h| h.ts.as_str()).max().unwrap_or("");
        let mut head = format!("{}. session {} — \"{title}\"", i + 1, short_id(&g.session_id));
        if !newest.is_empty() {
            head.push_str(&format!(" ({})", day(newest)));
        }
        lines.push(String::new());
        lines.push(head);
        for h in &g.hits {
            let d = day(&h.ts);
            let stamp = if d.is_empty() { String::new() } else { format!(" {d}") };
            lines.push(format!("   - [{}{stamp}] {}", h.role, h.snippet));
        }
    }
    lines.push(String::new());
    lines.push("The user can reopen any of these sessions with /resume.".to_string());
    lines.join("\n")
}

pub struct SessionSearchTool;

#[async_trait]
impl Tool for SessionSearchTool {
    fn name(&self) -> &'static str {
        SESSION_SEARCH_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        SCHEMA.clone()
    }

    async fn description(&self, _input: &Value) -> String {
        "Search past sessions of this project".to_string()
    }

    /// SessionSearch has no `DESCRIPTION` in the Python, so the wire description
    /// resolves through TOOL_DESCRIPTIONS / the simple blurb rather than this
    /// tool's (much longer) system prompt. Returning None selects that chain.
    async fn wire_description(&self) -> Option<String> {
        None
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn prompt(&self) -> String {
        concat!(
            "Searches the transcripts of this project's PAST sessions (earlier ",
            "conversations in this working directory) by keyword and returns ",
            "matching excerpts of user and assistant messages, grouped by ",
            "session.\n\n",
            "Usage:\n",
            "- Use this when the user refers to earlier work you have no context ",
            "for: \"as we discussed\", \"last time\", \"that bug we fixed\", ",
            "\"continue where we left off\". Search BEFORE answering that you ",
            "don't know or asking the user to re-explain.\n",
            "- Also use it to recover your own past findings in this project ",
            "(a diagnosis, a decision, a command that worked) instead of ",
            "re-deriving them.\n",
            "- query: words are ANDed and match by prefix (auth matches ",
            "authentication; a Korean word matches itself plus attached ",
            "particles). \"Quote a word or phrase\" for exact matching.\n",
            "- Matched terms come back [bracketed]. The current session is never ",
            "included, and only this project's sessions are searched.\n",
            "- MEMORY.md holds durable saved facts; this tool recovers specifics ",
            "that were never saved as memories. If a recovered fact is worth ",
            "keeping, save it as a memory.\n",
            "- Results are excerpts, not full turns. To pick a session back up, ",
            "tell the user which one and suggest /resume — you cannot load it ",
            "yourself."
        )
        .to_string()
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let q = input.get("query").and_then(|q| q.as_str()).unwrap_or("");
        colorize(q, colors::DIM)
    }

    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        let groups = data.get("groups").and_then(|g| g.as_array())?;
        if groups.is_empty() {
            return Some(colorize(&tr("No matches in past sessions"), colors::DIM));
        }
        let hits: usize = groups
            .iter()
            .map(|g| g.get("hits").and_then(|h| h.as_array()).map(|a| a.len()).unwrap_or(0))
            .sum();
        Some(colorize(
            &trf(
                "{hits} match(es) in {sessions} past session(s)",
                &[("hits", FmtArg::from(hits)), ("sessions", FmtArg::from(groups.len()))],
            ),
            colors::DIM,
        ))
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let q = input.get("query").and_then(|q| q.as_str()).unwrap_or("");
        if q.trim().is_empty() {
            let mut v = ValidationResult::failure("query is required");
            v.meta = Some(serde_json::json!({"errorCode": 1}));
            return v;
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let query = input
                .get("query")
                .and_then(|q| q.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let limit = input.get("limit").and_then(|l| l.as_i64()).unwrap_or(8).max(1) as usize;

            // First search after new sessions accumulate re-indexes them —
            // file IO and sqlite work that must not stall the event loop.
            let q2 = query.clone();
            let result =
                tokio::task::spawn_blocking(move || search(&q2, limit)).await;
            let groups = match result {
                Ok(Ok(groups)) => groups,
                Ok(Err(e)) => {
                    // Python: except ValueError → 'Invalid query: {e}'.
                    let data = serde_json::json!({"query": query, "groups": []});
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!("Invalid query: {e}"),
                        data: Some(data),
                    });
                    return;
                }
                Err(join_err) => {
                    let data = serde_json::json!({"query": query, "groups": []});
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!("Invalid query: {join_err}"),
                        data: Some(data),
                    });
                    return;
                }
            };

            let groups_json: Vec<Value> = groups
                .iter()
                .map(|g| {
                    serde_json::json!({
                        "session_id": g.session_id,
                        "first_prompt": g.first_prompt,
                        "hits": g.hits.iter().map(|h| serde_json::json!({
                            "role": h.role,
                            "ts": h.ts,
                            "snippet": h.snippet,
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let data = serde_json::json!({"query": query, "groups": groups_json});
            yield Ok(ToolResult::Result {
                result_for_assistant: format_for_assistant(&groups, &query),
                data: Some(data),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::session_index::SearchHit;

    #[test]
    fn empty_groups_message() {
        let out = format_for_assistant(&[], "foo bar");
        assert!(out.starts_with("No past-session messages match \"foo bar\""));
        assert!(out.contains("* prefix wildcard"));
    }

    #[test]
    fn grouped_formatting() {
        let groups = vec![SearchGroup {
            session_id: "0123456789abcdef".into(),
            first_prompt: "fix the bug".into(),
            hits: vec![SearchHit {
                role: "user".into(),
                ts: "2026-08-01T10:00:00Z".into(),
                snippet: "the [bug] was here".into(),
            }],
        }];
        let out = format_for_assistant(&groups, "bug");
        assert!(out.contains("Found 1 matching message(s) in 1 past session(s)"));
        assert!(out.contains("1. session 01234567 — \"fix the bug\" (2026-08-01)"));
        assert!(out.contains("   - [user 2026-08-01] the [bug] was here"));
        assert!(out.ends_with("The user can reopen any of these sessions with /resume."));
    }

    /// A byte-length check is not a character-boundary check. These fields are
    /// read back from the session store, so the tool has to survive whatever is
    /// on disk rather than assuming it wrote it.
    #[test]
    fn truncation_cuts_on_character_boundaries() {
        // ASCII behaviour is unchanged.
        assert_eq!(short_id("0123456789abcdef"), "01234567");
        assert_eq!(short_id("short"), "short");
        assert_eq!(day("2026-08-01T12:34:56Z"), "2026-08-01");
        assert_eq!(day("2026"), "2026");
        assert_eq!(day(""), "");

        // Multi-byte: the cut lands on a character, and the result is a prefix
        // of the input rather than a mangled one.
        for s in ["한글로된세션아이디", "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", "aé한🔥bcdefghij", "\u{00A0}\u{00A0}\u{00A0}"] {
            let id = short_id(s);
            let d = day(s);
            assert!(s.starts_with(id), "{id:?} is not a prefix of {s:?}");
            assert!(s.starts_with(d), "{d:?} is not a prefix of {s:?}");
            assert!(id.chars().count() <= 8);
            assert!(d.chars().count() <= 10);
        }
    }
}
