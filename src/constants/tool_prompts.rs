//! Tool descriptions — based on Claude Code tool prompts.
//! Equivalent to codemason/constants/tool_prompts.py.
//!
//! Source of truth: extracted/source/src/tools/<Tool>/prompt.ts
//!
//! These are the descriptions sent to the model in the tool schema (the
//! original sends each tool's full prompt, not a one-line blurb).
//! Dynamic/feature-gated bits are resolved for the EXTERNAL local build:
//!   - hasEmbeddedSearchTools() = False  → steer toward Glob/Grep
//!   - SandboxManager disabled            → no sandbox section
//!   - isPDFSupported() = True            → include PDF guidance
//!   - USER_TYPE != 'ant'                 → no ant-only hints
//!
//! The Python module assembled these once at import from f-strings over the
//! tool-name constants; the resolved strings are embedded here verbatim via
//! include_str! (dumped from the running Python module), which preserves every
//! byte without re-running the interpolation.

// === Read (FileReadTool/prompt.ts renderPromptTemplate) ===

pub const MAX_LINES_TO_READ: usize = 2000;

pub const READ_DESCRIPTION: &str = include_str!("text/read_description.txt");

// === Write (FileWriteTool/prompt.ts getWriteToolDescription) ===

pub const WRITE_DESCRIPTION: &str = include_str!("text/write_description.txt");

// === Edit (FileEditTool/prompt.ts getDefaultEditDescription) ===
// prefixFormat resolves to 'line number + tab': for external builds
// isCompactLinePrefixEnabled() defaults to True (killswitch off), and this
// matches the actual tab-prefixed Read output.

pub const EDIT_DESCRIPTION: &str = include_str!("text/edit_description.txt");

// === Glob (GlobTool/prompt.ts DESCRIPTION) ===

pub const GLOB_DESCRIPTION: &str = include_str!("text/glob_description.txt");

// === Grep (GrepTool/prompt.ts getDescription) ===

pub const GREP_DESCRIPTION: &str = include_str!("text/grep_description.txt");

// === Bash (BashTool/prompt.ts getSimplePrompt) ===
// Resolved external: embedded=False, sandbox disabled, MONITOR_TOOL feature
// off. Default timeout 120000ms (2 min), max 600000ms (10 min). The external
// prompt appends getCommitAndPRInstructions() — the FULL inline git commit/PR
// workflow block (prompt.ts:81-160) with the standard external attribution
// texts, all captured in the embedded text.

pub const BASH_DEFAULT_TIMEOUT_MS: u64 = 120_000;
pub const BASH_MAX_TIMEOUT_MS: u64 = 600_000;

/// Standard external attribution texts (getAttributionTexts external default).
pub const COMMIT_ATTRIBUTION: &str = "Co-Authored-By: CodeMason <noreply@codemason.local>";
pub const PR_ATTRIBUTION: &str = "🤖 Generated with CodeMason";

pub const BASH_DESCRIPTION: &str = include_str!("text/bash_description.txt");

// === Agent (AgentTool/prompt.ts getPrompt) ===
// Adapted to codemason's actual Agent: a single-`prompt`, foreground, stateless
// subprocess — parameters codemason's Agent does not have are deliberately
// omitted so a local model isn't told to call parameters that don't exist. One
// deliberate, non-canonical addition is kept: the "Write the agent prompt in
// the user's language" note. See the Python reference for the full rationale.

// No AGENT_DESCRIPTION here. AgentTool supplies its own wire description
// (tools::built_in_agents::build_agent_description, returned from its prompt()),
// so tool_description() never resolves Agent and a copy here could only drift
// out of it — it had, by two usage notes and the wording of the language rule.

/// Map tool name -> ported description, for the API request builder
/// (TOOL_DESCRIPTIONS).
pub fn tool_description(tool_name: &str) -> Option<&'static str> {
    use crate::constants::prompts::{
        BASH_TOOL_NAME, FILE_EDIT_TOOL_NAME, FILE_READ_TOOL_NAME,
        FILE_WRITE_TOOL_NAME, GLOB_TOOL_NAME, GREP_TOOL_NAME,
    };
    match tool_name {
        n if n == FILE_READ_TOOL_NAME => Some(READ_DESCRIPTION),
        n if n == FILE_WRITE_TOOL_NAME => Some(WRITE_DESCRIPTION),
        n if n == FILE_EDIT_TOOL_NAME => Some(EDIT_DESCRIPTION),
        n if n == GLOB_TOOL_NAME => Some(GLOB_DESCRIPTION),
        n if n == GREP_TOOL_NAME => Some(GREP_DESCRIPTION),
        n if n == BASH_TOOL_NAME => Some(BASH_DESCRIPTION),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_not_truncated() {
        // Head + tail anchors per description guard against truncation.
        assert!(READ_DESCRIPTION.starts_with("Reads a file from the local filesystem."));
        assert!(READ_DESCRIPTION.ends_with("in place of file contents."));
        assert!(READ_DESCRIPTION.contains("up to 2000 lines"));
        assert!(WRITE_DESCRIPTION.starts_with("Writes a file to the local filesystem."));
        assert!(WRITE_DESCRIPTION.ends_with("Avoid writing emojis to files unless asked."));
        assert!(EDIT_DESCRIPTION.starts_with("Performs exact string replacements in files."));
        assert!(EDIT_DESCRIPTION.contains("line number + tab"));
        assert!(GLOB_DESCRIPTION.contains("sorted by modification time"));
        assert!(GREP_DESCRIPTION.contains(r"interface\{\}"));
        assert!(GREP_DESCRIPTION.contains(r"struct \{[\s\S]*?field"));
        assert!(BASH_DESCRIPTION.contains("timeout after 120000ms (2 minutes)"));
        assert!(BASH_DESCRIPTION.contains(COMMIT_ATTRIBUTION));
        assert!(BASH_DESCRIPTION.contains(PR_ATTRIBUTION));
        assert!(BASH_DESCRIPTION.ends_with("gh api repos/foo/bar/pulls/123/comments"));
        // Agent is deliberately absent: it carries its own, and a copy here
        // drifted out of it before. description_map_resolves pins the absence.
    }

    #[test]
    fn description_map_resolves() {
        assert_eq!(tool_description("Read"), Some(READ_DESCRIPTION));
        // Agent resolves nowhere here on purpose: it carries its own wire
        // description, and the copy this map used to hold had drifted out of it.
        assert_eq!(tool_description("Agent"), None, "Agent must not resolve here");
        assert_eq!(tool_description("Bash"), Some(BASH_DESCRIPTION));
        assert_eq!(tool_description("Nope"), None);
    }
}
