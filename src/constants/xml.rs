//! XML tag names for the blocks injected into the conversation —
//! adapted from src/constants/xml.ts.
//! Equivalent to codemason/constants/xml.py.
//!
//! These are protocol, not formatting: the task-notification block is parsed
//! back out in print mode, so a tag that differs between the writer and the
//! reader is a silently dropped completion. They live in one module for the
//! same reason the source keeps them in one: three subsystems write these
//! blocks (shell tasks, agent tasks, the poll pipeline) and they have to agree.

pub const TASK_NOTIFICATION_TAG: &str = "task-notification";
pub const TASK_ID_TAG: &str = "task-id";
pub const TOOL_USE_ID_TAG: &str = "tool-use-id";
pub const TASK_TYPE_TAG: &str = "task-type";
pub const OUTPUT_FILE_TAG: &str = "output-file";
pub const STATUS_TAG: &str = "status";
pub const SUMMARY_TAG: &str = "summary";

/// Worktree isolation is not implemented in codemason, so nothing writes these
/// yet. Named here anyway: the notification format is shared with the reader,
/// and an agent run in a worktree reports where its changes landed.
pub const WORKTREE_TAG: &str = "worktree";
pub const WORKTREE_PATH_TAG: &str = "worktreePath";
pub const WORKTREE_BRANCH_TAG: &str = "worktreeBranch";
