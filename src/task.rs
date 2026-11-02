//! Task base types — adapted from src/Task.ts.
//! Equivalent to codemason/task.py.
//!
//! Defines the task type/status vocabulary, the shared TaskStateBase fields
//! stored in AppState.tasks, task-id generation (prefixed, crypto-random), and
//! the Task dispatch contract (`kill`).

use async_trait::async_trait;

/// TaskType — codemason runs tasks in-process synchronously for agents, so the
/// only task that actually lives in AppState.tasks is the background shell
/// ('local_bash'). The remaining types are kept in the prefix table for id
/// fidelity but no implementation is registered (see tasks/registry.rs).
pub type TaskType = String;

/// TaskStatus lifecycle: pending -> running -> completed | failed | killed.
pub type TaskStatus = String;

/// True when a task will not transition further (isTerminalTaskStatus).
pub fn is_terminal_task_status(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "killed")
}

/// Millisecond epoch — the stand-in for the source's Date.now().
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Task ID prefixes (TASK_ID_PREFIXES). 'b' for local_bash is kept for
/// backward compatibility with persisted/transcript task ids.
pub const TASK_ID_PREFIXES: &[(&str, &str)] = &[
    ("local_bash", "b"),
    ("local_agent", "a"),
    ("remote_agent", "r"),
    ("in_process_teammate", "t"),
    ("local_workflow", "w"),
    ("monitor_mcp", "m"),
    ("dream", "d"),
];

/// Case-insensitive-safe alphabet (digits + lowercase) for task IDs.
/// 36^8 ≈ 2.8 trillion combinations, sufficient to resist brute-force symlink
/// attacks on the output directory.
pub const TASK_ID_ALPHABET: &str = "0123456789abcdefghijklmnopqrstuvwxyz";

fn get_task_id_prefix(task_type: &str) -> &'static str {
    TASK_ID_PREFIXES
        .iter()
        .find(|(t, _)| *t == task_type)
        .map(|(_, p)| *p)
        .unwrap_or("x")
}

/// Generate a prefixed, crypto-random task id (generateTaskId).
///
/// Uses a rejection-sampled uniform draw over the alphabet rather than
/// urandom-byte % len: 256 % 36 != 0, so the modulo biased ids toward the
/// first few symbols, shrinking the effective keyspace.
pub fn generate_task_id(task_type: &str) -> String {
    use rand::Rng;
    let prefix = get_task_id_prefix(task_type);
    let alphabet: Vec<char> = TASK_ID_ALPHABET.chars().collect();
    let mut rng = rand::rngs::OsRng;
    let body: String = (0..8)
        .map(|_| alphabet[rng.gen_range(0..alphabet.len())])
        .collect();
    format!("{prefix}{body}")
}

/// Base fields shared by all task states (TaskStateBase). Concrete task states
/// (e.g. LocalShellTaskState) embed this struct.
#[derive(Debug, Clone, Default)]
pub struct TaskStateBase {
    pub id: String,
    pub task_type: String,
    pub status: String,
    pub description: String,
    pub tool_use_id: Option<String>,
    pub start_time: i64,
    pub end_time: Option<i64>,
    pub total_paused_ms: Option<i64>,
    pub output_file: String,
    pub output_offset: i64,
    pub notified: bool,
}

/// Build the base fields for a new task (createTaskStateBase). Callers embed
/// the returned struct and override fields (e.g. status -> 'running'),
/// mirroring the source's `{ ...createTaskStateBase(...), status: 'running' }`.
pub fn create_task_state_base(
    task_id: &str,
    task_type: &str,
    description: &str,
    tool_use_id: Option<String>,
) -> TaskStateBase {
    TaskStateBase {
        id: task_id.to_string(),
        task_type: task_type.to_string(),
        status: "pending".to_string(),
        description: description.to_string(),
        tool_use_id,
        start_time: now_ms(),
        end_time: None,
        total_paused_ms: None,
        output_file: crate::utils::task::disk_output::get_task_output_path(task_id)
            .to_string_lossy()
            .into_owned(),
        output_offset: 0,
        notified: false,
    }
}

/// Task dispatch contract (Task type in Task.ts).
///
/// `get_task_by_type` dispatches to a Task implementation for `kill`. Kill
/// drives the registry through the app-state store.
#[async_trait]
pub trait Task: Send + Sync {
    fn name(&self) -> &'static str;
    fn task_type(&self) -> &'static str;
    async fn kill(&self, task_id: &str);
}
