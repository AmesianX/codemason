//! Skill discovery + frontmatter parsing for CodeMason.
//! Equivalent to codemason/utils/skills.py. Based on Claude Code's skill
//! loading (utils/frontmatterParser.ts + skills/loadSkillsDir.ts).
//!
//! A skill is a markdown file `<skills_dir>/<name>/SKILL.md` with YAML
//! frontmatter (description, context: inline|fork, agent, ...) plus a body
//! that is the skill's prompt. `context: inline` (default) expands into the
//! conversation; `context: fork` runs the skill in a sub-agent.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_yaml::Value as Yaml;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::utils::state::get_cwd;

/// utils/frontmatterParser.ts FRONTMATTER_REGEX
static FRONTMATTER_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^---\s*\n((?s:.)*?)---\s*\n?").unwrap());

/// frontmatterParser.ts YAML_SPECIAL_CHARS — chars/sequences that require
/// quoting in an unquoted YAML value. Note ': ' (colon+space) is matched as a
/// sequence so that '12:34' times and 'https://' URLs stay unquoted.
static YAML_SPECIAL_CHARS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[{}\[\]*&#!|>%@`]|: ").unwrap());

/// frontmatterParser.ts quoteProblematicValues line matcher: simple
/// `key: value` lines (not indented, not list items, not block scalars).
static SIMPLE_KV_LINE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^([a-zA-Z_-]+):\s+(.+)$").unwrap());

/// Port of frontmatterParser.ts quoteProblematicValues (lines 85-121).
///
/// Pre-process frontmatter text to quote values that contain special YAML
/// characters, so glob patterns like **/*.{ts,tsx} parse correctly.
fn quote_problematic_values(frontmatter_text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in frontmatter_text.split('\n') {
        if let Some(m) = SIMPLE_KV_LINE.captures(line) {
            let key = &m[1];
            let value = &m[2];
            if key.is_empty() || value.is_empty() {
                out.push(line.to_string());
                continue;
            }
            // Skip if already quoted.
            if (value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\''))
            {
                out.push(line.to_string());
                continue;
            }
            // Quote if it contains special YAML characters. Escape backslashes
            // first, then double-quotes.
            if YAML_SPECIAL_CHARS.is_match(value) {
                let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
                out.push(format!("{key}: \"{escaped}\""));
                continue;
            }
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

/// parseFrontmatter — (frontmatter mapping, content without frontmatter).
///
/// On YAML failure, retries after quoting problematic values (so a single
/// special char in one value no longer nukes the whole frontmatter).
pub fn parse_frontmatter(markdown: &str) -> (HashMap<String, Yaml>, String) {
    let Some(m) = FRONTMATTER_REGEX.captures(markdown) else {
        return (HashMap::new(), markdown.to_string());
    };
    let whole = m.get(0).map(|g| g.as_str()).unwrap_or("");
    let fm_text = m.get(1).map(|g| g.as_str()).unwrap_or("");
    let content = markdown[whole.len()..].to_string();

    let parse = |text: &str| -> Option<HashMap<String, Yaml>> {
        match serde_yaml::from_str::<Yaml>(text) {
            Ok(Yaml::Mapping(map)) => {
                let mut out = HashMap::new();
                for (k, v) in map {
                    if let Yaml::String(ks) = k {
                        out.insert(ks, v);
                    } else {
                        // Non-string key: stringify like Python dict access would
                        out.insert(yaml_to_string(&k), v);
                    }
                }
                Some(out)
            }
            _ => None,
        }
    };

    if let Some(fm) = parse(fm_text) {
        return (fm, content);
    }
    // YAML parsing failed - try again after quoting problematic values.
    if let Some(fm) = parse(&quote_problematic_values(fm_text)) {
        return (fm, content);
    }
    (HashMap::new(), content)
}

fn yaml_to_string(v: &Yaml) -> String {
    match v {
        Yaml::String(s) => s.clone(),
        Yaml::Bool(b) => {
            // Python str(True) -> 'True'; keep display parity where it leaks
            // into display_name (never happens for real skills, but stay close)
            if *b { "True".into() } else { "False".into() }
        }
        Yaml::Number(n) => n.to_string(),
        Yaml::Null => "None".into(),
        other => serde_yaml::to_string(other).unwrap_or_default().trim_end().to_string(),
    }
}

/// frontmatterParser.ts parseBooleanFrontmatter: true only for True/'true'.
fn parse_bool_frontmatter(value: Option<&Yaml>) -> bool {
    match value {
        Some(Yaml::Bool(true)) => true,
        Some(Yaml::String(s)) => s == "true",
        _ => false,
    }
}

/// argumentSubstitution.ts parseArgumentNames: frontmatter `arguments` as a
/// space-separated string OR a list. Drops empty + numeric-only names (those
/// collide with the $0/$1 indexed shorthand).
pub fn parse_argument_names(value: Option<&Yaml>) -> Vec<String> {
    let valid = |n: &str| -> bool {
        let t = n.trim();
        !t.is_empty() && !t.chars().all(|c| c.is_ascii_digit())
    };
    match value {
        None | Some(Yaml::Null) => Vec::new(),
        Some(Yaml::Sequence(seq)) => seq
            .iter()
            .filter_map(|v| match v {
                Yaml::String(s) if valid(s) => Some(s.trim().to_string()),
                _ => None,
            })
            .collect(),
        Some(Yaml::String(s)) => s
            .split_whitespace()
            .filter(|n| valid(n))
            .map(|n| n.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

/// Port of loadSkillsDir.ts parseSkillPaths. Returns the list of path patterns
/// (or empty). Substrate gap: codemason cannot yet detect which files the
/// model touches, so skills with non-empty paths are activated nowhere and are
/// excluded from the default listing (see discover_skills).
fn parse_skill_paths(fm: &HashMap<String, Yaml>) -> Vec<String> {
    let raw = match fm.get("paths") {
        None | Some(Yaml::Null) => return Vec::new(),
        Some(v) => v,
    };
    let items: Vec<String> = match raw {
        Yaml::Sequence(seq) => seq.iter().map(yaml_to_string).collect(),
        other => {
            // Comma-separated string. (We don't expand brace patterns here —
            // the presence/absence of patterns is all that matters.)
            yaml_to_string(other)
                .split(',')
                .map(|p| p.trim().to_string())
                .collect()
        }
    };
    let mut out = Vec::new();
    for mut p in items {
        // Strip /** suffix to match ignore-library path semantics.
        if p.ends_with("/**") {
            p.truncate(p.len() - 3);
        }
        if !p.is_empty() {
            out.push(p);
        }
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub context: String, // 'inline' | 'fork'
    pub agent: Option<String>,
    pub path: String,
    pub display_name: Option<String>,
    pub disable_model_invocation: bool,
    pub paths: Vec<String>,
    pub argument_names: Vec<String>,
    /// the skill's directory; ${CODEMASON_SKILL_DIR}
    pub base_dir: String,
}

/// The skills shipped inside codemason.
///
/// Lowest-precedence root, so a user/project skill of the same name overrides
/// the bundled one. SUBSTRATE (Rust binary vs Python package data): the SKILL.md
/// files are embedded in the binary at compile time and materialized to
/// `<CONFIG_DIR>/bundled_skills/` — rewritten when the on-disk copy drifts, the
/// same self-materializing idiom as keep_going_installer — so skills that
/// reference ${CODEMASON_SKILL_DIR} still get a real directory.
pub fn bundled_skills_dir() -> PathBuf {
    crate::utils::config::CONFIG_DIR.join("bundled_skills")
}

/// (relative path, content) of every bundled skill file, embedded verbatim
/// from the repository's skills/ tree (the Python package's codemason/skills).
const BUNDLED_SKILL_FILES: &[(&str, &str)] = &[
    ("arxiv/SKILL.md", include_str!("../../skills/arxiv/SKILL.md")),
    (
        "blocked-page-recovery/SKILL.md",
        include_str!("../../skills/blocked-page-recovery/SKILL.md"),
    ),
    ("code-review/SKILL.md", include_str!("../../skills/code-review/SKILL.md")),
    (
        "github-code-review/SKILL.md",
        include_str!("../../skills/github-code-review/SKILL.md"),
    ),
    (
        "github-issue-to-pr/SKILL.md",
        include_str!("../../skills/github-issue-to-pr/SKILL.md"),
    ),
    ("github-issues/SKILL.md", include_str!("../../skills/github-issues/SKILL.md")),
    (
        "github-pr-workflow/SKILL.md",
        include_str!("../../skills/github-pr-workflow/SKILL.md"),
    ),
    (
        "grounded-citations/SKILL.md",
        include_str!("../../skills/grounded-citations/SKILL.md"),
    ),
    ("plan/SKILL.md", include_str!("../../skills/plan/SKILL.md")),
    ("simplify-code/SKILL.md", include_str!("../../skills/simplify-code/SKILL.md")),
    ("skill-creator/SKILL.md", include_str!("../../skills/skill-creator/SKILL.md")),
    ("spike/SKILL.md", include_str!("../../skills/spike/SKILL.md")),
    (
        "systematic-debugging/SKILL.md",
        include_str!("../../skills/systematic-debugging/SKILL.md"),
    ),
    (
        "test-driven-development/SKILL.md",
        include_str!("../../skills/test-driven-development/SKILL.md"),
    ),
];

/// Materialize the embedded bundled skills to bundled_skills_dir() (idempotent;
/// rewrites only drifted files). Best-effort — a read-only config dir just
/// means no bundled skills, never an error.
fn ensure_bundled_skills() {
    let root = bundled_skills_dir();
    for (rel, content) in BUNDLED_SKILL_FILES {
        let dest = root.join(rel);
        let current = std::fs::read_to_string(&dest).ok();
        if current.as_deref() != Some(*content) {
            if let Some(parent) = dest.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // Atomic: this runs at startup, so a second session starting at the
            // same moment is reading these very files while they are written, and
            // a plain write truncates first — the loader then saw a SKILL.md with
            // no frontmatter and silently dropped the skill. Every writer writes
            // identical bytes, so the rename needs no lock: it publishes the whole
            // file or nothing.
            let _ = crate::utils::file::atomic_write_bytes(&dest.to_string_lossy(), content.as_bytes());
        }
    }
}

/// Skill search roots, in first-wins precedence order.
///
/// By default only our own dirs are searched, user before project:
///
///     user/.codemason -> project/.codemason -> bundled
///
/// With config.import_claude_dirs on, each scope also searches the matching
/// `.claude` dir, so skills authored for the other tool are picked up:
///
///     user/.codemason -> user/.claude -> project/.codemason -> project/.claude
///
/// The native dir wins over an imported one on a name collision, and user wins
/// over project (matching the original's managed -> user -> project order;
/// codemason has no managed dir). The user-level `.codemason` root honors
/// CODEMASON_CONFIG_DIR via CONFIG_DIR.
fn skill_dirs() -> Vec<PathBuf> {
    ensure_bundled_skills();
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    let cwd = PathBuf::from(get_cwd());
    // Another product's directories are read ONLY when asked for. This is our
    // program; inheriting a foreign layout by default means a skill we never
    // shipped can appear in our tool list without the user choosing that.
    let import_foreign = crate::utils::config::get_config().import_claude_dirs;
    let mut candidates: Vec<PathBuf> = vec![
        crate::utils::config::CONFIG_DIR.join("skills"), // user, native
    ];
    if import_foreign {
        candidates.push(home.join(".claude").join("skills")); // user, opt-in import
    }
    candidates.push(cwd.join(".codemason").join("skills")); // project, native
    if import_foreign {
        candidates.push(cwd.join(".claude").join("skills")); // project, opt-in import
    }
    candidates.push(bundled_skills_dir()); // shipped with codemason (lowest)
    let mut dirs_out = Vec::new();
    let mut seen = HashSet::new();
    for d in candidates {
        let rp = d.to_string_lossy().into_owned();
        if seen.contains(&rp) {
            continue;
        }
        seen.insert(rp);
        if d.is_dir() {
            dirs_out.push(d);
        }
    }
    dirs_out
}

struct SkillCache {
    skills: Option<Vec<Skill>>,
    sig: Option<Vec<(String, i64, i64)>>,
}

static CACHE: Lazy<Mutex<SkillCache>> =
    Lazy::new(|| Mutex::new(SkillCache { skills: None, sig: None }));

/// Drop the discover_skills() cache (call if skill dirs change mid-session).
pub fn refresh_skills() {
    let mut c = CACHE.lock();
    c.skills = None;
    c.sig = None;
}

fn mtime_ns(p: &Path) -> i64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// A cheap fingerprint of the skill roots and their immediate skill dirs, so a
/// skill CREATED or REMOVED mid-session invalidates the cache automatically —
/// without re-parsing every SKILL.md on each call. Content edits to an
/// existing SKILL.md still need refresh_skills() (or a `codemason skill` run,
/// which is a fresh process).
fn skills_signature() -> Vec<(String, i64, i64)> {
    let mut sig = Vec::new();
    for d in skill_dirs() {
        sig.push((d.to_string_lossy().into_owned(), mtime_ns(&d), 0));
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut subs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        subs.sort();
        for sub in subs {
            if sub.is_dir() {
                let md = sub.join("SKILL.md");
                let md_m = if md.is_file() { mtime_ns(&md) } else { 0 };
                sig.push((sub.to_string_lossy().into_owned(), mtime_ns(&sub), md_m));
            }
        }
    }
    sig
}

/// Scan the skill dirs for `<name>/SKILL.md` files (cached, auto-refreshed).
///
/// Memoized per process, but the cache is invalidated automatically when the
/// skill directories change (a skill added or removed), keyed on a cheap mtime
/// signature — so a skill created mid-session becomes visible without a
/// restart. Call refresh_skills() to force a re-scan.
pub fn discover_skills() -> Vec<Skill> {
    let sig = skills_signature();
    {
        let c = CACHE.lock();
        if let (Some(skills), Some(cached_sig)) = (&c.skills, &c.sig) {
            if *cached_sig == sig {
                return skills.clone();
            }
        }
    }

    let mut skills: Vec<Skill> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    for d in skill_dirs() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut subs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        subs.sort();
        for sub in subs {
            let skill_md = sub.join("SKILL.md");
            if !(sub.is_dir() && skill_md.is_file()) {
                continue;
            }
            let name = sub
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if seen_names.contains(&name) {
                // First-wins: an earlier dir (user) already provided this name.
                continue;
            }
            let Ok(raw_bytes) = std::fs::read(&skill_md) else {
                continue;
            };
            let raw = String::from_utf8_lossy(&raw_bytes).into_owned();
            let (fm, content) = parse_frontmatter(&raw);
            let display_name = fm.get("name").filter(|v| !v.is_null()).map(yaml_to_string);
            let desc = fm
                .get("description")
                .filter(|v| !v.is_null())
                .map(yaml_to_string)
                .unwrap_or_default()
                .trim()
                .to_string();
            let ctx_raw = fm
                .get("context")
                .filter(|v| !v.is_null())
                .map(yaml_to_string)
                .unwrap_or_else(|| "inline".to_string());
            let ctx = if ctx_raw == "inline" || ctx_raw == "fork" {
                ctx_raw
            } else {
                "inline".to_string()
            };
            let agent = fm
                .get("agent")
                .filter(|v| !v.is_null())
                .map(yaml_to_string);
            seen_names.insert(name.clone());
            skills.push(Skill {
                name,
                description: desc,
                content: content.trim().to_string(),
                context: ctx,
                agent,
                path: skill_md.to_string_lossy().into_owned(),
                display_name,
                disable_model_invocation: parse_bool_frontmatter(
                    fm.get("disable-model-invocation"),
                ),
                paths: parse_skill_paths(&fm),
                argument_names: parse_argument_names(fm.get("arguments")),
                base_dir: sub.to_string_lossy().into_owned(),
            });
        }
    }

    let mut c = CACHE.lock();
    c.sig = Some(sig);
    c.skills = Some(skills.clone());
    skills
}

/// Skills the model may invoke via the Skill tool: exclude
/// disable-model-invocation skills and conditional (paths) skills.
///
/// - disable-model-invocation: excluded from model invocation entirely
///   (loadSkillsDir.ts / commands.ts getSkillToolCommands).
/// - non-empty paths: conditional skills that only activate when matching
///   files are touched (loadSkillsDir.ts:771-794). Substrate gap: codemason
///   can't detect touched files, so these are excluded from the listing here.
pub fn model_invocable_skills() -> Vec<Skill> {
    discover_skills()
        .into_iter()
        .filter(|s| !s.disable_model_invocation && s.paths.is_empty())
        .collect()
}

/// The skill names the MODEL can see for the life of this session.
///
/// The tool block is built once per tool-set and reused (services/openai_common),
/// because it sits at the very front of the server's cached prefix: rebuilding it
/// mid-session re-sends every tool definition and forces the next reply to
/// re-read the whole conversation. So the Skill tool's listing is whatever
/// existed at the first request.
///
/// A skill created after that is still runnable — `get_skill` scans live — it is
/// simply absent from the listing the model reads. This snapshot is how the REPL
/// can tell the two apart and say so instead of leaving the user wondering why a
/// skill they just wrote is being ignored.
static SESSION_SKILLS: Lazy<HashSet<String>> =
    Lazy::new(|| model_invocable_skills().into_iter().map(|s| s.name).collect());

/// Take the snapshot now, so it reflects session start rather than whenever the
/// first lookup happens to occur.
pub fn pin_session_skills() {
    Lazy::force(&SESSION_SKILLS);
}

/// Was this skill visible to the model when the session began?
pub fn is_session_skill(name: &str) -> bool {
    SESSION_SKILLS.contains(name)
}

/// Look up a model-invocable skill by name (excludes disable-model-invocation
/// and conditional/paths skills).
pub fn get_skill(name: &str) -> Option<Skill> {
    model_invocable_skills().into_iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_basic() {
        let md = "---\nname: test\ndescription: a skill\n---\nBody here";
        let (fm, content) = parse_frontmatter(md);
        assert_eq!(fm.get("name").map(yaml_to_string).as_deref(), Some("test"));
        assert_eq!(content, "Body here");
    }

    #[test]
    fn frontmatter_special_chars_requoted() {
        let md = "---\npaths: **/*.{ts,tsx}\n---\nB";
        let (fm, _) = parse_frontmatter(md);
        assert!(fm.contains_key("paths"));
    }

    #[test]
    fn no_frontmatter_passthrough() {
        let (fm, content) = parse_frontmatter("just text");
        assert!(fm.is_empty());
        assert_eq!(content, "just text");
    }

    #[test]
    fn argument_names_variants() {
        assert_eq!(
            parse_argument_names(Some(&Yaml::String("a b 12 c".into()))),
            vec!["a", "b", "c"]
        );
        let seq = Yaml::Sequence(vec![
            Yaml::String("x".into()),
            Yaml::String("3".into()),
            Yaml::String(" y ".into()),
        ]);
        assert_eq!(parse_argument_names(Some(&seq)), vec!["x", "y"]);
    }
}
