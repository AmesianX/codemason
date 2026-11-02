//! Persistent memory (memdir) — the original's memory directive, rooted
//! locally. Equivalent to codemason/utils/memory.py.
//!
//! Claude Code gives the model a per-project memory directory and injects a
//! directive plus the MEMORY.md index into every session's system prompt; the
//! memories themselves are ordinary files the model writes with its own Write
//! tool. That mechanism ports cleanly — the directory lives next to the
//! project's transcripts (same sanitized-cwd key, so "this project" means the
//! same thing to memory and to sessions), and no new tool is needed on the
//! write side.
//!
//! What is injected each session:
//!   - the directive (how to save, the frontmatter shape, what NOT to save)
//!   - the contents of MEMORY.md, the one-line-per-memory index
//!
//! The memory FILES are never injected — the index line is the recall hook,
//! and the model Reads the file when a line looks relevant. That is what
//! keeps the prompt cost flat as memories accumulate.
//!
//! The read side for things that were never saved as memories is the
//! SessionSearch tool (utils/session_index.rs) — the directive points there.

use std::path::{Path, PathBuf};

use super::session_storage::get_project_dir;
use super::state::get_original_cwd;

/// MEMORY.md is an index (one line per memory), so this cap is generous —
/// hitting it means content leaked into the index, and truncation is the
/// right pressure.
const INDEX_MAX_CHARS: usize = 8_000;

pub const MEMORY_FILE: &str = "MEMORY.md";

pub fn get_memory_dir() -> PathBuf {
    get_project_dir(&get_original_cwd()).join("memory")
}

fn directive(memory_dir: &Path) -> String {
    format!(
        r#"# Memory

You have a persistent file-based memory at `{}`. This directory already exists — write to it directly with the Write tool (do not run mkdir or check for its existence). Each memory is one file holding one fact, with frontmatter:

```markdown
---
name: <short-kebab-case-slug>
description: <one-line summary, used to decide relevance during recall>
metadata:
  type: user | feedback | project | reference
---

<the fact; for feedback/project, follow with **Why:** and **How to apply:** lines. Link related memories with [[their-name]].>
```

`user`: who the user is (role, expertise, preferences). `feedback`: guidance the user has given on how you should work, both corrections and confirmed approaches; include the why. `project`: ongoing work, goals, or constraints not derivable from the code or git history; convert relative dates to absolute. `reference`: pointers to external resources (URLs, dashboards, tickets).

After writing the file, add a one-line pointer in `MEMORY.md` (`- [Title](file.md) — hook`). `MEMORY.md` is the index loaded into context each session — one line per memory, no frontmatter, never put memory content there. When an index line looks relevant to the task at hand, Read that memory file before relying on it.

Write a memory only at these moments: the user asks you to remember something; the user tells you something durable about themselves, their preferences, or how they want you to work; the user settles a decision or a constraint that the code will not record; or the standing state of ongoing work changes in a way a later session would have to be told. Nothing else is a trigger. A turn that went well is not one. An interesting result is not one. Reaching the end of a session is not one, and neither is a stretch of work you want on the record.

One subject, one file, and that file stays short — a handful of lines that state the fact and why it matters. If you have already written about a piece of work, do not add a second memory about it; revise the one you wrote so it still says one thing. Revising is not appending: a file that grows an entry per turn, or carries dated entries, or reads as a record of what happened, has stopped being a memory and become a log, and a log belongs in the repo or in the transcript, not here. When the standing state changes, rewrite the sentence that is now wrong rather than adding the update beneath it.

Separate, unrelated facts are separate files, but they are rare: most sessions leave none, and a session that leaves more than one or two is almost certainly saving its own work rather than a fact. A session whose every turn produces something new produces no more memories than a quiet one — what you derived, proved, measured, or ruled out along the way is the work, not the standing state, and the transcript already holds it. Save only the conclusion that would change what a later session does, never the reasoning that reached it. When unsure, don't write: a memory you skipped costs one question later, a memory you should not have written misleads every session that reads it.

Before saving, check for an existing file that already covers it. Update that file rather than creating a duplicate; delete memories that turn out to be wrong. Don't save what the repo already records (code structure, past fixes, git history) or what only matters to this conversation; if asked to remember one of those, ask what was non-obvious about it and save that instead. Memories reflect what was true when written — if one names a file, function, or flag, verify it still exists before recommending it. For specifics from a past conversation that were never saved as a memory, use the SessionSearch tool."#,
        memory_dir.display()
    )
}

/// The system-prompt Memory section, or None when the substrate is broken.
///
/// Always emitted when it can be — an empty memory is how every memory
/// starts, and the directive is what teaches the model to fill it. Never
/// raises: memory must not be able to stop a session from starting.
pub fn build_memory_section() -> Option<String> {
    let result = (|| -> std::io::Result<String> {
        let memory_dir = get_memory_dir();
        // Created here so the directive's "already exists" is true by the time
        // the model reads it.
        std::fs::create_dir_all(&memory_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&memory_dir, std::fs::Permissions::from_mode(0o700));
        }
        let index_path = memory_dir.join(MEMORY_FILE);
        let mut contents = String::new();
        if index_path.is_file() {
            let bytes = std::fs::read(&index_path)?;
            contents = String::from_utf8_lossy(&bytes).trim().to_string();
        }
        if contents.chars().count() > INDEX_MAX_CHARS {
            let head: String = contents.chars().take(INDEX_MAX_CHARS).collect();
            contents = format!(
                "{head}\n… (index truncated — MEMORY.md holds content that belongs in memory files; move it there)"
            );
        }
        if contents.is_empty() {
            contents = "(empty — no memories saved yet)".to_string();
        }
        Ok(format!(
            "{}\n\n## MEMORY.md\n{contents}",
            directive(&memory_dir)
        ))
    })();
    match result {
        Ok(section) => Some(section),
        Err(_) => {
            crate::utils::log::log_error(&"memory: failed to build the Memory section", None);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_names_the_dir() {
        let d = directive(Path::new("/tmp/x/memory"));
        assert!(d.starts_with("# Memory"));
        assert!(d.contains("`/tmp/x/memory`"));
        assert!(d.contains("SessionSearch"));
    }
}
