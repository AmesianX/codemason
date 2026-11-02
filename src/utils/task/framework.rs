//! Task framework helpers — adapted from src/utils/task/framework.ts.
//! Equivalent to codemason/utils/task/framework.py.
//!
//! The AppState.tasks update primitives every task mechanism builds on:
//! update_task_state / register_task / evict_terminal_task / get_running_tasks.
//!
//! The poll pipeline (generate_task_attachments / apply_task_offsets_and_evictions
//! / poll_tasks) is ported too, including the part that surprises: it produces
//! no attachments. The source builds the list and never pushes to it, and its
//! own comment says why — each task type enqueues its own completion
//! notification, so generating one here would race those callbacks and deliver
//! the same completion twice. What the pass actually does is keep the books:
//! advance each running task's output offset past what has been read, and evict
//! terminal tasks whose notification has been consumed and whose grace period
//! has passed.

use crate::state::app_state::{get_app_state, set_app_state, AppState};

/// Standard polling interval for all tasks (POLL_INTERVAL_MS).
pub const POLL_INTERVAL_MS: u64 = 1000;

/// How long a killed task stays visible before eviction (STOPPED_DISPLAY_MS).
pub const STOPPED_DISPLAY_MS: i64 = 3_000;

/// Grace period for a terminal local_agent task (PANEL_GRACE_MS). The source
/// names it for the coordinator panel that holds the task on screen; here it is
/// the window in which a finished agent's record — status, result, counts — can
/// still be read back before eviction.
pub const PANEL_GRACE_MS: i64 = 30_000;
use crate::task::{is_terminal_task_status, now_ms};
use crate::tasks::types::TaskState;

/// Update a task's state in AppState (updateTaskState).
///
/// `updater(&task)` returns `Some(next)` to replace it, or `None` for a no-op
/// (the reference-equality skip in the source; Rust has no reference identity
/// across a clone, so "None = unchanged" carries it). If the task is absent,
/// the store is left untouched. Callers that need to observe whether a change
/// happened capture it in the closure (Cell/Mutex), matching the Python idiom.
pub fn update_task_state<F>(task_id: &str, updater: F)
where
    F: FnOnce(&TaskState) -> Option<TaskState>,
{
    let task_id = task_id.to_string();
    set_app_state(move |mut prev: AppState| {
        let Some(task) = prev.tasks.get(&task_id) else {
            return prev;
        };
        if let Some(updated) = updater(task) {
            prev.tasks.insert(task_id, updated);
        }
        prev
    });
}

/// Register a new task in AppState (registerTask).
///
/// Re-registering an id that is already present is a REPLACEMENT, not a new
/// start — resuming an agent builds a fresh task state and registers it over
/// the old one. The fields the UI and the message history own are carried
/// forward, because the replacement does not know them: `retain` is the user's
/// hold, `start_time` keeps the ordering stable, `messages` and `disk_loaded`
/// are the transcript as it stands (a just-appended prompt lives in `messages`
/// and is not on disk yet), and `pending_messages` is a queue that may already
/// hold something for the agent to read. Overwriting them would drop a queued
/// message on the floor and reset a hold the user placed.
///
/// The original also emits a `task_started` event here and skips it on a
/// replacement, so a resume does not look like a second spawn. codemason routes
/// its notifications through utils::message_queue_manager and nothing
/// subscribes to task starts yet, so there is no emit — but the replacement is
/// still distinguished, because the carry-forward depends on it.
pub fn register_task(task: TaskState) {
    set_app_state(move |mut prev: AppState| {
        let id = task.base().id.clone();
        let merged = match (prev.tasks.get(&id), &task) {
            (Some(TaskState::LocalAgent(existing)), TaskState::LocalAgent(next)) => {
                let mut merged = next.clone();
                merged.retain = existing.retain;
                merged.base.start_time = existing.base.start_time;
                merged.messages = existing.messages.clone();
                merged.disk_loaded = existing.disk_loaded;
                merged.pending_messages = existing.pending_messages.clone();
                TaskState::LocalAgent(merged)
            }
            _ => task,
        };
        prev.tasks.insert(id, merged);
        prev
    });
}

/// Eagerly evict a terminal, notified task from AppState (evictTerminalTask).
///
/// Three guards, all from the source:
///   * not terminal                   -> keep (still running)
///   * not notified                   -> keep (an unconsumed completion is
///                                       never dropped)
///   * eviction deadline not reached  -> keep
///
/// The third applies to tasks that carry an `evict_after` deadline, which in
/// the source is exactly the agent tasks (it tests `"retain" in task`). A
/// terminal agent task gets `evict_after = now + PANEL_GRACE_MS`, so its record
/// — status, result, token and tool counts — outlives the run by that grace
/// period and the parent can still read it back. An agent task with NO deadline
/// is a retained one, and a retained task is never evicted at all; that is why
/// an unset deadline means keep, not evict (the source's
/// `evictAfter ?? Infinity`).
pub fn evict_terminal_task(task_id: &str) {
    let task_id = task_id.to_string();
    set_app_state(move |mut prev: AppState| {
        let should = prev
            .tasks
            .get(&task_id)
            .map(|t| {
                if !is_terminal_task_status(&t.base().status) || !t.base().notified {
                    return false;
                }
                match t.as_local_agent() {
                    Some(agent) => agent.evict_after.map(|at| at <= now_ms()).unwrap_or(false),
                    None => true,
                }
            })
            .unwrap_or(false);
        if should {
            prev.tasks.remove(&task_id);
        }
        prev
    });
}

/// All running tasks (getRunningTasks).
pub fn get_running_tasks(state: &AppState) -> Vec<TaskState> {
    state
        .tasks
        .values()
        .filter(|t| t.base().status == "running")
        .cloned()
        .collect()
}

/// Convenience: read a single task's current state.
pub fn get_task(task_id: &str) -> Option<TaskState> {
    get_app_state().tasks.get(task_id).cloned()
}


/// A task status update, as the attachment pipeline would emit it
/// (TaskAttachment).
///
/// Built by nothing today — see `generate_task_attachments` — but it is the
/// shape `render_task_status_attachment` consumes, and the compaction path
/// builds the same JSON by hand.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskAttachment {
    pub task_id: String,
    pub task_type: String,
    pub status: String,
    pub description: String,
    pub delta_summary: Option<String>,
    pub tool_use_id: Option<String>,
}

/// What one bookkeeping pass found.
#[derive(Debug, Default)]
pub struct TaskPollFindings {
    pub attachments: Vec<TaskAttachment>,
    pub updated_task_offsets: std::collections::HashMap<String, i64>,
    pub evicted_task_ids: Vec<String>,
}

/// Walk the task map, returning what to patch and what to drop
/// (generateTaskAttachments).
///
/// `attachments` is always empty, exactly as in the source, and the source's
/// own comment is the reason: "Completed tasks are NOT notified here — each
/// task type handles its own completion notification via
/// enqueuePendingNotification(). Generating attachments here would race with
/// those per-type callbacks, causing dual delivery (one inline attachment + one
/// separate API turn)." The field is kept because `poll_tasks` iterates it and
/// a future task type may want it.
///
/// Only the offsets are returned, never the tasks: a task can transition to
/// completed during the disk read below, and writing back a whole stale
/// snapshot would clobber that transition — a zombie task, running forever in
/// a map that has already forgotten it finished.
pub fn generate_task_attachments(state: &AppState) -> TaskPollFindings {
    let mut found = TaskPollFindings::default();

    for task in state.tasks.values() {
        let base = task.base();
        if base.notified {
            if is_terminal_task_status(&base.status) {
                // Consumed — it can go.
                found.evicted_task_ids.push(base.id.clone());
                continue;
            }
            if base.status == "pending" {
                // Keep it: it has not run yet, but the parent already knows.
                continue;
            }
        }

        if base.status == "running" {
            let (content, new_offset) =
                crate::utils::task::disk_output::get_task_output_delta(&base.id, base.output_offset);
            if !content.is_empty() {
                found.updated_task_offsets.insert(base.id.clone(), new_offset);
            }
        }
    }

    found
}

/// Apply what `generate_task_attachments` found (applyTaskOffsetsAndEvictions).
///
/// Every condition is re-checked against the state as it is NOW, not as it was
/// when the walk started: the disk read in between is a window in which a task
/// can finish, or a resume can replace it. An offset for a task that is no
/// longer running is moot, and an eviction decided before a replacement landed
/// would delete the replacement.
pub fn apply_task_offsets_and_evictions(
    updated_task_offsets: std::collections::HashMap<String, i64>,
    evicted_task_ids: Vec<String>,
) {
    if updated_task_offsets.is_empty() && evicted_task_ids.is_empty() {
        return;
    }
    set_app_state(move |mut prev: AppState| {
        let mut changed = false;

        for (id, offset) in updated_task_offsets.iter() {
            if let Some(fresh) = prev.tasks.get_mut(id) {
                if fresh.base().status == "running" {
                    fresh.base_mut().output_offset = *offset;
                    changed = true;
                }
            }
        }

        for id in evicted_task_ids.iter() {
            let evictable = match prev.tasks.get(id) {
                None => false,
                Some(fresh) => {
                    if !is_terminal_task_status(&fresh.base().status) || !fresh.base().notified {
                        false
                    } else {
                        match fresh.as_local_agent() {
                            Some(agent) => {
                                agent.evict_after.map(|at| at <= now_ms()).unwrap_or(false)
                            }
                            None => true,
                        }
                    }
                }
            };
            if evictable {
                prev.tasks.remove(id);
                changed = true;
            }
        }

        let _ = changed;
        prev
    });
}

/// Human-readable status for a notification (getStatusText).
pub fn get_status_text(status: &str) -> &str {
    match status {
        "completed" => "completed successfully",
        "failed" => "failed",
        "killed" => "was stopped",
        "running" => "is running",
        "pending" => "is pending",
        other => other,
    }
}

/// Enqueue a <task-notification> for a task status update
/// (enqueueTaskNotification).
///
/// Reachable only through `poll_tasks`, which iterates a list that is always
/// empty — see `generate_task_attachments`. Ported so the pipeline is whole,
/// and so a task type that does want inline delivery has the writer already.
pub fn enqueue_task_notification(attachment: &TaskAttachment) {
    let tool_use_id_line = match &attachment.tool_use_id {
        Some(id) => format!("\n<tool-use-id>{id}</tool-use-id>"),
        None => String::new(),
    };
    let output_path = crate::utils::task::disk_output::get_task_output_path(&attachment.task_id);
    let message = format!(
        "<task-notification>\n\
         <task-id>{}</task-id>{}\n\
         <task-type>{}</task-type>\n\
         <output-file>{}</output-file>\n\
         <status>{}</status>\n\
         <summary>Task \"{}\" {}</summary>\n\
         </task-notification>",
        attachment.task_id,
        tool_use_id_line,
        attachment.task_type,
        output_path.to_string_lossy(),
        attachment.status,
        attachment.description,
        get_status_text(&attachment.status),
    );
    crate::utils::message_queue_manager::enqueue_pending_notification(
        &message,
        Some("task-notification"),
    );
}

/// One bookkeeping pass over every task (pollTasks).
///
/// Advances the running tasks' output offsets and drops the terminal ones that
/// have been consumed. Not a notification loop — see
/// `generate_task_attachments` for why the attachment list it walks is empty.
pub fn poll_tasks() {
    let found = generate_task_attachments(&get_app_state());
    for attachment in &found.attachments {
        enqueue_task_notification(attachment);
    }
    apply_task_offsets_and_evictions(found.updated_task_offsets, found.evicted_task_ids);
}


#[cfg(test)]
mod poll_tests {
    //! The bookkeeping pass. The part that looks wrong until you read why: it
    //! emits no attachments, because each task type announces its own
    //! completion and a second announcement here would deliver it twice.
    use super::*;
    use crate::state::app_state::{reset_app_state, test_lock};
    use crate::task::create_task_state_base;
    use crate::tasks::local_agent_task::{
        complete_agent_task, register_agent_foreground,
    };
    use crate::tasks::local_shell_task::LocalShellTaskState;
    use crate::tools::built_in_agents::AgentDefinition;
    use crate::utils::agent_id::create_agent_id;
    use crate::utils::task::disk_output::get_task_output_path;
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

    fn spawn() -> String {
        let id = create_agent_id(None);
        register_agent_foreground(
            &id, "look around", "find the thing", fake_agent(), None,
            Some(CancellationToken::new()), None,
        );
        id
    }

    fn offset_of(id: &str) -> i64 {
        get_app_state().tasks.get(id).unwrap().base().output_offset
    }

    fn patch_agent<F: Fn(&mut crate::tasks::local_agent_task::LocalAgentTaskState)>(id: &str, f: F) {
        update_task_state(id, |t| {
            let mut next = t.as_local_agent()?.clone();
            f(&mut next);
            Some(next.into())
        });
    }

    #[test]
    fn a_pass_advances_the_offset_and_announces_nothing() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        let _ = std::fs::write(get_task_output_path(&id), b"first line\n");

        let found = generate_task_attachments(&get_app_state());
        assert!(found.attachments.is_empty(), "its own type announces it");
        assert_eq!(found.updated_task_offsets.get(&id), Some(&11));
        assert!(found.evicted_task_ids.is_empty());

        poll_tasks();
        assert_eq!(offset_of(&id), 11, "the pass writes the offset back");

        let _ = std::fs::write(get_task_output_path(&id), b"first line\nsecond\n");
        poll_tasks();
        assert_eq!(offset_of(&id), 18, "and advances past only the new bytes");

        poll_tasks();
        assert_eq!(offset_of(&id), 18, "nothing new leaves it alone");
        reset_app_state();
    }

    #[test]
    fn eviction_waits_for_notified_and_the_grace_period() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        complete_agent_task(&id, serde_json::json!({"text": "done"}));

        poll_tasks();
        assert!(
            get_app_state().tasks.contains_key(&id),
            "a finished but un-notified task is kept — its completion is unread"
        );

        patch_agent(&id, |t| t.base.notified = true);
        poll_tasks();
        assert!(
            get_app_state().tasks.contains_key(&id),
            "and still kept while its grace period runs"
        );

        patch_agent(&id, |t| t.evict_after = Some(now_ms() - 1));
        poll_tasks();
        assert!(!get_app_state().tasks.contains_key(&id), "then dropped");
        reset_app_state();
    }

    /// The write-back re-reads the state it is writing into, because the disk
    /// read is a window in which the task can finish.
    #[test]
    fn an_offset_for_a_task_that_finished_mid_pass_is_dropped() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        let _ = std::fs::write(get_task_output_path(&id), b"xxxxxxxxxxxxxxxxxxxx");
        let found = generate_task_attachments(&get_app_state());
        assert!(!found.updated_task_offsets.is_empty());

        complete_agent_task(&id, serde_json::json!({"text": "raced"}));
        apply_task_offsets_and_evictions(found.updated_task_offsets, found.evicted_task_ids);

        let state = get_app_state();
        let task = state.tasks.get(&id).unwrap();
        assert_eq!(task.base().status, "completed", "the finish is not clobbered");
        assert_eq!(task.base().output_offset, 0, "and the stale offset is not written");
        reset_app_state();
    }

    #[test]
    fn a_task_that_came_back_to_life_mid_pass_is_not_evicted() {
        let _guard = test_lock();
        reset_app_state();
        let id = spawn();
        complete_agent_task(&id, serde_json::json!({"text": "done"}));
        patch_agent(&id, |t| {
            t.base.notified = true;
            t.evict_after = Some(now_ms() - 1);
        });
        let found = generate_task_attachments(&get_app_state());
        assert_eq!(found.evicted_task_ids, vec![id.clone()]);

        // A resume replaces it in the window.
        patch_agent(&id, |t| {
            t.base.status = "running".into();
            t.base.notified = false;
            t.evict_after = None;
        });
        apply_task_offsets_and_evictions(found.updated_task_offsets, found.evicted_task_ids);
        assert!(get_app_state().tasks.contains_key(&id));
        reset_app_state();
    }

    #[test]
    fn a_consumed_shell_task_is_evicted_with_no_grace_period() {
        let _guard = test_lock();
        reset_app_state();
        let mut base = create_task_state_base("b99999999", "local_bash", "run tests", None);
        base.status = "completed".into();
        base.notified = true;
        register_task(
            LocalShellTaskState { base, command: "pytest".into(), kind: "bash".into(), ..Default::default() }
                .into(),
        );
        poll_tasks();
        assert!(!get_app_state().tasks.contains_key("b99999999"));
        reset_app_state();
    }

    #[test]
    fn the_status_wording_is_the_sources() {
        assert_eq!(get_status_text("completed"), "completed successfully");
        assert_eq!(get_status_text("failed"), "failed");
        assert_eq!(get_status_text("killed"), "was stopped");
        assert_eq!(get_status_text("running"), "is running");
        assert_eq!(get_status_text("pending"), "is pending");
    }
}
