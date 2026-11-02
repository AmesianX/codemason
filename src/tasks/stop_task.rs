//! Shared stop logic — adapted from src/tasks/stopTask.ts.
//! Equivalent to codemason/tasks/stop_task.py.
//!
//! Look up a task by id, validate it is running, kill it via its task impl,
//! and (for shell tasks) suppress the kill notification. Used by TaskStop.

use crate::state::app_state::{get_app_state, set_app_state, AppState};
use crate::tasks::local_shell_task::is_local_shell_task;
use crate::tasks::registry::get_task_by_type;

/// Raised when a task cannot be stopped. `code` distinguishes the reason.
#[derive(Debug, Clone)]
pub struct StopTaskError {
    pub message: String,
    /// 'not_found' | 'not_running' | 'unsupported_type'
    pub code: &'static str,
}

impl StopTaskError {
    fn new(message: impl Into<String>, code: &'static str) -> Self {
        StopTaskError { message: message.into(), code }
    }
}

impl std::fmt::Display for StopTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Result of a successful stop.
#[derive(Debug)]
pub struct StopResult {
    pub task_id: String,
    pub task_type: String,
    /// The command (shell task) or description.
    pub command: Option<String>,
}

/// Kill a running task, returning its identity (stopTask).
///
/// Errors (with `.code`) when the task is not found, not running, or of an
/// unsupported type.
pub async fn stop_task(task_id: &str) -> Result<StopResult, StopTaskError> {
    let app_state = get_app_state();
    let task = app_state
        .tasks
        .get(task_id)
        .cloned()
        .ok_or_else(|| StopTaskError::new(format!("No task found with ID: {task_id}"), "not_found"))?;

    if task.base().status != "running" {
        return Err(StopTaskError::new(
            format!("Task {task_id} is not running (status: {})", task.base().status),
            "not_running",
        ));
    }

    let task_impl = get_task_by_type(&task.base().task_type)
        .ok_or_else(|| StopTaskError::new(format!("Unsupported task type: {}", task.base().task_type), "unsupported_type"))?;

    task_impl.kill(task_id).await;

    // Bash: suppress the "exit code 137" notification (noise). kill_task already
    // set notified=True, so the late completion notice is dropped — this block
    // mirrors stopTask's explicit suppression. (The original additionally emits
    // an SDK task_terminated event here; codemason has no SDK event queue —
    // substrate — so that emit is dropped.)
    let shell = task.as_local_shell().filter(|t| is_local_shell_task(t));
    if shell.is_some() {
        let tid = task_id.to_string();
        set_app_state(move |mut prev: AppState| {
            if let Some(prev_task) = prev.tasks.get(&tid) {
                if !prev_task.base().notified {
                    let mut next = prev_task.clone();
                    next.base_mut().notified = true;
                    prev.tasks.insert(tid, next);
                }
            }
            prev
        });
    }

    // A shell task is named by its command; everything else — a sub-agent —
    // by its description, which is the brief it was spawned with.
    let command = match shell {
        Some(t) => Some(t.command.clone()),
        None => Some(task.base().description.clone()),
    };
    Ok(StopResult { task_id: task_id.to_string(), task_type: task.base().task_type.clone(), command })
}

#[cfg(test)]
mod agent_tests {
    //! Stopping a sub-agent has to actually stop it, and must NOT suppress its
    //! notification the way a shell kill does.
    use super::*;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::tasks::local_agent_task::register_agent_foreground;
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn fake_agent() -> AgentDefinition {
        AgentDefinition {
            agent_type: "Explore".into(),
            when_to_use: String::new(),
            get_system_prompt: Arc::new(|| String::new()),
            tools: None,
            disallowed_tools: Vec::new(),
            source: "built-in".into(),
        }
    }

    #[tokio::test]
    async fn task_stop_reaches_a_sub_agent() {
        let _guard = test_lock();
        reset_app_state();
        let id = create_agent_id(None);
        let token = CancellationToken::new();
        register_agent_foreground(
            &id, "look around", "find the thing", fake_agent(), None,
            Some(token.clone()), None,
        );

        let stopped = stop_task(&id).await.expect("a running agent can be stopped");
        assert_eq!(stopped.task_type, "local_agent");
        assert_eq!(
            stopped.command.as_deref(),
            Some("look around"),
            "an agent is named by the brief it was spawned with"
        );

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap().as_local_agent().unwrap();
        assert_eq!(task.base.status, "killed");
        assert!(token.is_cancelled(), "the agent itself was told to stop");
        // The source suppresses the notification for a shell task only: "Agent
        // tasks: don't suppress — the AbortError catch sends a notification
        // carrying extractPartialResult(agentMessages), which is the payload
        // not noise."
        assert!(
            !task.base.notified,
            "a stopped agent stays announceable, so its partial result is not lost"
        );

        let again = stop_task(&id).await;
        assert_eq!(again.err().map(|e| e.code), Some("not_running"));
        reset_app_state();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::create_task_state_base;
    use crate::tasks::local_shell_task::LocalShellTaskState;
    use crate::utils::task::framework::register_task;

    #[tokio::test]
    async fn not_found() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        let e = stop_task("bzzz00000").await.unwrap_err();
        assert_eq!(e.code, "not_found");
    }

    #[tokio::test]
    async fn not_running() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        let id = "b12345678";
        let mut base = create_task_state_base(id, "local_bash", "x", None);
        base.id = id.into();
        base.status = "completed".into();
        register_task(LocalShellTaskState { base, kind: "bash".into(), ..Default::default() }.into());
        let e = stop_task(id).await.unwrap_err();
        assert_eq!(e.code, "not_running");
    }

    #[tokio::test]
    async fn stops_running_and_suppresses() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        let id = "b87654321";
        let mut base = create_task_state_base(id, "local_bash", "x", None);
        base.id = id.into();
        base.status = "running".into();
        register_task(LocalShellTaskState { base, command: "sleep 100".into(), kind: "bash".into(), pid: None, ..Default::default() }.into());
        let r = stop_task(id).await.unwrap();
        assert_eq!(r.task_type, "local_bash");
        assert_eq!(r.command.as_deref(), Some("sleep 100"));
        let st = get_app_state();
        let t = st.tasks.get(id).unwrap();
        assert_eq!(t.base().status, "killed");
        assert!(t.base().notified);
    }
}
