//! Task registry — adapted from src/tasks.ts (getAllTasks / getTaskByType).
//! Equivalent to codemason/tasks/registry.py.
//!
//! The source registers six task types (shell, local/remote agent, dream,
//! workflow, monitor). codemason registers the two with a local implementation:
//! the background shell and the sub-agent. The dispatch contract
//! (getTaskByType → impl with .kill) is what lets stop_task stop either one
//! without knowing which it has.

use crate::task::Task;
use crate::tasks::local_agent_task::LOCAL_AGENT_TASK;
use crate::tasks::local_shell_task::LOCAL_SHELL_TASK;

/// Dispatch to a task implementation by its type (getTaskByType).
pub fn get_task_by_type(task_type: &str) -> Option<&'static dyn Task> {
    get_all_tasks().into_iter().find(|t| t.task_type() == task_type)
}

/// All registered task implementations (getAllTasks).
pub fn get_all_tasks() -> Vec<&'static dyn Task> {
    vec![&*LOCAL_SHELL_TASK, &*LOCAL_AGENT_TASK]
}
