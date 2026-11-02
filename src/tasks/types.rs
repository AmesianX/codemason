//! TaskState union — adapted from src/tasks/types.ts.
//! Equivalent to codemason/tasks/types.py.
//!
//! The source union spans local_bash / local_agent / remote_agent / teammate /
//! workflow / monitor / dream. codemason registers the two that actually run
//! locally: the background shell and the sub-agent. The remaining types have no
//! implementation here (no remote execution, no workflows, no teammates), so
//! they are absent from the union rather than stubbed.
//!
//! TypeScript discriminates the union on `type` and narrows with the guards
//! (isLocalShellTask / EJ); Rust makes it an enum and `as_local_shell` /
//! `as_local_agent` are those guards. The payload's own `base.task_type` stays
//! the discriminant on the wire, so a task read back from state still says what
//! it is.

use crate::task::TaskStateBase;

pub use super::local_agent_task::LocalAgentTaskState;
pub use super::local_shell_task::LocalShellTaskState;

#[derive(Debug, Clone)]
pub enum TaskState {
    LocalShell(LocalShellTaskState),
    LocalAgent(LocalAgentTaskState),
}

impl TaskState {
    /// The fields every task shares (the spread of TaskStateBase).
    pub fn base(&self) -> &TaskStateBase {
        match self {
            TaskState::LocalShell(t) => &t.base,
            TaskState::LocalAgent(t) => &t.base,
        }
    }

    pub fn base_mut(&mut self) -> &mut TaskStateBase {
        match self {
            TaskState::LocalShell(t) => &mut t.base,
            TaskState::LocalAgent(t) => &mut t.base,
        }
    }

    /// `type` on the discriminated union.
    pub fn task_type(&self) -> &str {
        &self.base().task_type
    }

    /// Narrow to a background shell task (isLocalShellTask).
    pub fn as_local_shell(&self) -> Option<&LocalShellTaskState> {
        match self {
            TaskState::LocalShell(t) => Some(t),
            _ => None,
        }
    }

    /// Narrow to a sub-agent task (EJ).
    pub fn as_local_agent(&self) -> Option<&LocalAgentTaskState> {
        match self {
            TaskState::LocalAgent(t) => Some(t),
            _ => None,
        }
    }

    /// True when this task is backgrounded. Both variants carry the flag, and
    /// callers that only need to know "will someone collect this later?" should
    /// not have to match first.
    pub fn is_backgrounded(&self) -> bool {
        match self {
            TaskState::LocalShell(t) => t.is_backgrounded,
            TaskState::LocalAgent(t) => t.is_backgrounded,
        }
    }
}

impl From<LocalShellTaskState> for TaskState {
    fn from(t: LocalShellTaskState) -> Self {
        TaskState::LocalShell(t)
    }
}

impl From<LocalAgentTaskState> for TaskState {
    fn from(t: LocalAgentTaskState) -> Self {
        TaskState::LocalAgent(t)
    }
}
