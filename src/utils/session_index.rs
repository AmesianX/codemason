//! Full-text search over this project's past session transcripts.
//! Equivalent to codemason/utils/session_index.py.
//!
//! The transcripts already exist (session_storage writes one JSONL per
//! session); this module makes them recallable. claude.ai gives the model a
//! conversation_search tool over past chats — locally the chats are files on
//! disk, so the same capability is an SQLite FTS5 index next to them:
//!
//!     <project_dir>/search-index.db
//!         msg            fts5(text, session_id, ts, role, file)
//!         indexed_files  (file, mtime, size, first_prompt)
//!
//! Only user/assistant TEXT is indexed — not tool results, not control
//! payloads (<task-notification> etc.), not compact summaries (their
//! originals are in the same file; indexing both would double every hit).
//! The index is synced lazily on each search: files whose (mtime, size)
//! changed are re-indexed, vanished files are dropped. The CURRENT session is
//! excluded on both the index and the query side — it is this context, not a
//! memory — and a concurrent session in the same project may index it early,
//! which is why the query side filters too.
//!
//! FTS5 can be compiled out of a distro's sqlite3; that is a real boundary,
//! so search() degrades to a linear scan of the transcripts there
//! (list_sessions already full-reads them all for /resume, so the scale is
//! accepted). rusqlite here is the bundled build, which carries FTS5, but the
//! probe is kept so a system-sqlite build degrades exactly like Python.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rusqlite::Connection;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::session_storage::{extract_first_prompt, get_project_dir, load_transcript_file};
use super::state::{get_original_cwd, get_session_id};

const DB_NAME: &str = "search-index.db";

/// Snippet window FTS5 builds around matches (in tokens); matched terms are
/// marked with [brackets] so the model can see WHY a row matched.
const SNIPPET_TOKENS: i64 = 24;

/// Control payloads that render as user turns but aren't the user speaking —
/// same shape session_storage skips when picking a first prompt.
static CONTROL_TEXT: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"^\s*(?:<[a-z][\w-]*[\s>]|\[Request interrupted)").unwrap());

/// Per-message indexing cap. A pasted log can be huge; past 16k chars a match
/// adds recall noise, not signal.
const TEXT_CAP: usize = 16_000;

static FTS5_OK: Lazy<Mutex<Option<bool>>> = Lazy::new(|| Mutex::new(None));

/// Take the index down to 0600, like the transcripts it is built from.
///
/// SQLite creates the file with the process umask, normally 0644, while
/// session_storage writes the transcripts themselves 0600. The index holds the
/// SAME text — the fts5 table stores every message body — so the stricter of
/// the two is the right mode for it.
///
/// Defence in depth rather than a live leak: the containing project directory
/// is already 0700, so another local user cannot traverse into it today. What
/// this covers is the directory mode being loosened later, and the file being
/// copied somewhere flatter (a backup, an archive, a bug report), where the
/// mode travels with it.
///
/// Best effort and unix-only. Windows ACLs are inherited from the directory and
/// have no chmod equivalent; failing to tighten is not a reason to fail a
/// search.
fn restrict_index_permissions(db_path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // The -wal and -shm sidecars carry the same content between
        // checkpoints, so tightening only the main file would leave the data
        // readable in the one next to it.
        for suffix in ["", "-wal", "-shm"] {
            let p = if suffix.is_empty() {
                db_path.to_path_buf()
            } else {
                std::path::PathBuf::from(format!("{}{}", db_path.display(), suffix))
            };
            if p.exists() {
                let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = db_path;
}

fn fts5_available() -> bool {
    let mut cache = FTS5_OK.lock();
    if let Some(ok) = *cache {
        return ok;
    }
    let ok = (|| -> rusqlite::Result<()> {
        let probe = Connection::open_in_memory()?;
        probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)", [])?;
        Ok(())
    })()
    .is_ok();
    *cache = Some(ok);
    ok
}

/// The searchable text of one transcript entry ([] for most entries).
fn entry_texts(entry: &Value) -> Vec<String> {
    if entry.get("isMeta").and_then(|v| v.as_bool()).unwrap_or(false)
        || entry
            .get("isCompactSummary")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    {
        return Vec::new();
    }
    let etype = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(etype, "user" | "assistant") {
        return Vec::new();
    }
    let content = entry.get("message").and_then(|m| m.get("content"));
    let mut texts: Vec<String> = Vec::new();
    match content {
        Some(Value::String(s)) => texts.push(s.clone()),
        Some(Value::Array(blocks)) => {
            for block in blocks {
                if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                    if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                        if !t.is_empty() {
                            texts.push(t.to_string());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    texts
        .into_iter()
        .filter(|t| !t.trim().is_empty() && !CONTROL_TEXT.is_match(t))
        .collect()
}

fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS msg USING fts5(\
         text, session_id UNINDEXED, ts UNINDEXED, role UNINDEXED, file UNINDEXED)",
        [],
    )?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS indexed_files(\
         file TEXT PRIMARY KEY, mtime REAL, size INTEGER, first_prompt TEXT)",
        [],
    )?;
    Ok(())
}

fn file_mtime_f64(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn jsonl_files(project_dir: &Path, exclude_stem: &str) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(project_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
                .filter(|p| {
                    p.file_stem().and_then(|s| s.to_str()).map(|s| s != exclude_stem).unwrap_or(true)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn char_cap(s: &str, cap: usize) -> String {
    if s.chars().count() <= cap {
        s.to_string()
    } else {
        s.chars().take(cap).collect()
    }
}

fn sync(conn: &Connection, project_dir: &Path, exclude_stem: &str) -> rusqlite::Result<()> {
    let files: HashMap<String, std::path::PathBuf> = jsonl_files(project_dir, exclude_stem)
        .into_iter()
        .filter_map(|p| {
            let name = p.file_name()?.to_str()?.to_string();
            Some((name, p))
        })
        .collect();
    let mut known: HashMap<String, (f64, i64)> = HashMap::new();
    {
        let mut stmt = conn.prepare("SELECT file, mtime, size FROM indexed_files")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, (row.get::<_, f64>(1)?, row.get::<_, i64>(2)?)))
        })?;
        for row in rows.flatten() {
            known.insert(row.0, row.1);
        }
    }
    let file_names: HashSet<&String> = files.keys().collect();
    for name in known.keys().filter(|k| !file_names.contains(k)) {
        conn.execute("DELETE FROM msg WHERE file = ?1", [name])?;
        conn.execute("DELETE FROM indexed_files WHERE file = ?1", [name])?;
    }
    for (name, path) in &files {
        let Ok(st) = std::fs::metadata(path) else {
            continue;
        };
        let mtime = file_mtime_f64(&st);
        let size = st.len() as i64;
        if known.get(name).map(|(m, s)| *m == mtime && *s == size).unwrap_or(false) {
            continue;
        }
        let Ok((messages, order)) = load_transcript_file(path) else {
            continue; // oversized/unreadable — skipped, not fatal (list_sessions rule)
        };
        let entries: Vec<&Value> = order.iter().map(|u| &messages[u]).collect();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let sid = entries
            .iter()
            .find_map(|e| {
                e.get("sessionId")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
            })
            .unwrap_or(stem);
        conn.execute("DELETE FROM msg WHERE file = ?1", [name])?;
        {
            let mut ins = conn.prepare(
                "INSERT INTO msg(text, session_id, ts, role, file) VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for e in &entries {
                let ts = e.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
                let role = e.get("type").and_then(|v| v.as_str()).unwrap_or("");
                for text in entry_texts(e) {
                    ins.execute(rusqlite::params![char_cap(&text, TEXT_CAP), sid, ts, role, name])?;
                }
            }
        }
        let entries_owned: Vec<Value> = entries.iter().map(|e| (*e).clone()).collect();
        conn.execute(
            "INSERT OR REPLACE INTO indexed_files(file, mtime, size, first_prompt) \
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![name, mtime, size, extract_first_prompt(&entries_owned)],
        )?;
    }
    // (Python commits explicitly; rusqlite autocommits outside a transaction.)
    Ok(())
}

static QUERY_TOKEN_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r#""[^"]*"|\S+"#).unwrap());

/// User words -> an FTS5 MATCH that cannot be a syntax error.
///
/// Every bare token is quoted (so `config.json` or `--flag` don't parse as
/// operators) and searched as a PREFIX: the unicode61 tokenizer splits on
/// whitespace, so in agglutinative text a word and its particle are ONE token
/// ('텔레그램으로') and an exact-word query for the bare word would miss it —
/// and prefix also gives 'auth' → 'authentication' for free, with bm25 still
/// ranking the exact hits first. User quotes opt back into exact word/phrase
/// matching. Tokens are implicitly ANDed, which is FTS5's default.
fn fts_match_expr(query: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for token in QUERY_TOKEN_RE.find_iter(query).map(|m| m.as_str()) {
        if token.starts_with('"') && token.ends_with('"') && token.len() > 2 {
            parts.push(token.to_string());
            continue;
        }
        let bare = token.trim_end_matches('*').replace('"', "");
        if !bare.is_empty() {
            parts.push(format!("\"{bare}\"*"));
        }
    }
    parts.join(" ")
}

/// One matched message inside a session group.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub role: String,
    pub ts: String,
    pub snippet: String,
}

/// One session's group of hits.
#[derive(Debug, Clone)]
pub struct SearchGroup {
    pub session_id: String,
    pub first_prompt: String,
    pub hits: Vec<SearchHit>,
}

type RawRow = (String, String, String, String, String); // snippet, sid, ts, role, file

/// Rank-ordered rows -> at most `limit` session groups, best-first, at most
/// `per_session` snippets each (one long session must not eat the whole
/// result).
fn group_hits(
    raw: Vec<RawRow>,
    first_prompts: &HashMap<String, String>,
    limit: usize,
    per_session: usize,
) -> Vec<SearchGroup> {
    let mut groups: Vec<SearchGroup> = Vec::new();
    let mut index_by_sid: HashMap<String, usize> = HashMap::new();
    for (snippet, sid, ts, role, file) in raw {
        let idx = match index_by_sid.get(&sid) {
            Some(i) => *i,
            None => {
                if groups.len() >= limit {
                    continue;
                }
                groups.push(SearchGroup {
                    session_id: sid.clone(),
                    first_prompt: first_prompts.get(&file).cloned().unwrap_or_default(),
                    hits: Vec::new(),
                });
                index_by_sid.insert(sid.clone(), groups.len() - 1);
                groups.len() - 1
            }
        };
        if groups[idx].hits.len() < per_session {
            groups[idx].hits.push(SearchHit { role, ts, snippet });
        }
    }
    groups
}

/// Session groups matching `query`, best-first. Err on a query with no
/// searchable tokens (Python ValueError); degrades to a scan without FTS5.
pub fn search(query: &str, limit: usize) -> Result<Vec<SearchGroup>, String> {
    let project_dir = get_project_dir(&get_original_cwd());
    let current = get_session_id();
    if !project_dir.is_dir() {
        return Ok(Vec::new());
    }
    if !fts5_available() {
        return scan_search(&project_dir, &current, query, limit);
    }
    let match_expr = fts_match_expr(query);
    if match_expr.is_empty() {
        return Err("query has no searchable words".to_string());
    }
    let fts_result = (|| -> rusqlite::Result<(Vec<RawRow>, HashMap<String, String>)> {
        let db_path = project_dir.join(DB_NAME);
        let conn = Connection::open(&db_path)?;
        restrict_index_permissions(&db_path);
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        ensure_schema(&conn)?;
        sync(&conn, &project_dir, &current)?;
        let sql = format!(
            "SELECT snippet(msg, 0, '[', ']', ' … ', {SNIPPET_TOKENS}), \
             session_id, ts, role, file FROM msg \
             WHERE msg MATCH ?1 AND session_id != ?2 ORDER BY rank LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql)?;
        let raw: Vec<RawRow> = stmt
            .query_map(
                rusqlite::params![match_expr, current, (limit * 4) as i64],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )?
            .flatten()
            .collect();
        let mut first_prompts = HashMap::new();
        let mut stmt2 = conn.prepare("SELECT file, first_prompt FROM indexed_files")?;
        let rows = stmt2.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            ))
        })?;
        for row in rows.flatten() {
            first_prompts.insert(row.0, row.1);
        }
        Ok((raw, first_prompts))
    })();
    match fts_result {
        Ok((raw, first_prompts)) => Ok(group_hits(raw, &first_prompts, limit, 3)),
        // A locked/corrupt index must not make recall unavailable; the
        // transcripts themselves are the source of truth.
        Err(_) => scan_search(&project_dir, &current, query, limit),
    }
}

/// No-FTS5 path: case-insensitive AND-of-words over a linear read of every
/// transcript. Newest sessions first (no rank without FTS), same group shape.
fn scan_search(
    project_dir: &Path,
    current: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchGroup>, String> {
    let words: Vec<String> = QUERY_TOKEN_RE
        .find_iter(query)
        .map(|m| m.as_str().trim_matches('"').to_lowercase())
        .map(|w| w.trim_end_matches('*').to_string())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return Err("query has no searchable words".to_string());
    }
    let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> =
        jsonl_files(project_dir, current)
            .into_iter()
            .filter_map(|p| {
                let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?;
                Some((mtime, p))
            })
            .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));
    let mut groups: Vec<SearchGroup> = Vec::new();
    for (_, path) in files {
        if groups.len() >= limit {
            break;
        }
        let Ok((messages, order)) = load_transcript_file(&path) else {
            continue;
        };
        let entries: Vec<Value> = order.iter().map(|u| messages[u].clone()).collect();
        let mut hits: Vec<SearchHit> = Vec::new();
        for e in &entries {
            for text in entry_texts(e) {
                let low = text.to_lowercase();
                if !words.iter().all(|w| low.contains(w)) {
                    continue;
                }
                // Byte offsets from the lowercased haystack can split a UTF-8
                // boundary in `text`; snap to char boundaries (semantics
                // identical to Python's char slicing on the clip window).
                let at = low.find(&words[0]).unwrap_or(0);
                let start = snap_floor(&text, at.saturating_sub(60));
                let end = snap_floor(&text, std::cmp::min(text.len(), at + 120));
                let clip = text[start..end].replace('\n', " ").trim().to_string();
                hits.push(SearchHit {
                    role: e.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    ts: e.get("timestamp").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    snippet: format!("{}{clip}…", if start > 0 { "…" } else { "" }),
                });
                break;
            }
            if hits.len() >= 3 {
                break;
            }
        }
        if !hits.is_empty() {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let sid = entries
                .iter()
                .find_map(|e| {
                    e.get("sessionId")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                })
                .unwrap_or(stem);
            groups.push(SearchGroup {
                session_id: sid,
                first_prompt: extract_first_prompt(&entries),
                hits,
            });
        }
    }
    Ok(groups)
}

/// Largest char boundary <= idx.
fn snap_floor(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The index stores every message body, the same text as the transcripts,
    /// which session_storage writes 0600. SQLite creates it with the umask —
    /// 0644 on a normal machine — so it was the loosest copy of the data.
    #[cfg(unix)]
    #[test]
    fn the_index_is_not_left_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("cm-idx-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let db = dir.join("probe.db");
        std::fs::write(&db, b"").unwrap();
        std::fs::set_permissions(&db, std::fs::Permissions::from_mode(0o644)).unwrap();

        // A -wal sidecar carries the same rows between checkpoints.
        let wal = std::path::PathBuf::from(format!("{}-wal", db.display()));
        std::fs::write(&wal, b"").unwrap();
        std::fs::set_permissions(&wal, std::fs::Permissions::from_mode(0o644)).unwrap();

        restrict_index_permissions(&db);

        let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&db), 0o600, "the index kept group/other read");
        assert_eq!(mode(&wal), 0o600, "the -wal sidecar kept group/other read");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fts_expr_quoting() {
        assert_eq!(fts_match_expr("config.json --flag"), "\"config.json\"* \"--flag\"*");
        assert_eq!(fts_match_expr("\"exact phrase\" auth*"), "\"exact phrase\" \"auth\"*");
        assert_eq!(fts_match_expr(""), "");
    }

    #[test]
    fn entry_texts_filters() {
        let meta = serde_json::json!({"type":"user","isMeta":true,
            "message":{"content":"hidden"}});
        assert!(entry_texts(&meta).is_empty());
        let ctl = serde_json::json!({"type":"user",
            "message":{"content":"<task-notification> done"}});
        assert!(entry_texts(&ctl).is_empty());
        let ok = serde_json::json!({"type":"assistant",
            "message":{"content":[{"type":"text","text":"an answer"}]}});
        assert_eq!(entry_texts(&ok), vec!["an answer".to_string()]);
    }

    #[test]
    fn fts5_probe_ok_with_bundled() {
        // Bundled rusqlite carries FTS5; the probe should say so.
        assert!(fts5_available());
    }

    #[test]
    fn group_hits_caps() {
        let raw: Vec<RawRow> = (0..10)
            .map(|i| {
                (
                    format!("snip{i}"),
                    format!("s{}", i % 2),
                    String::new(),
                    "user".to_string(),
                    "f.jsonl".to_string(),
                )
            })
            .collect();
        let groups = group_hits(raw, &HashMap::new(), 8, 3);
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|g| g.hits.len() == 3));
    }
}
