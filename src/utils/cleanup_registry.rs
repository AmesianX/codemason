//! Cleanup registry — adapted from utils/cleanupRegistry.ts.
//! Equivalent to codemason/utils/cleanup_registry.py.
//!
//! The source keeps one module-level `Set` of cleanup callbacks.
//! `registerCleanup` adds one and returns the function that removes it again;
//! `runCleanupFunctions` awaits every registered callback during graceful
//! shutdown. That module is kept separate from the shutdown path to avoid a
//! circular dependency, and so is this one.
//!
//! This is what makes a task's cleanup handle mean something: a sub-agent
//! registers "kill me" at spawn, so a process that dies mid-run still tears the
//! agent down, and the handle cancels that registration once the agent has
//! finished on its own. Without the pair one of the two lifecycle bugs is
//! unavoidable — either the agent outlives the process that owns it, or its
//! cleanup fires after it already completed.
//!
//! Divergence from the Python: a closure cannot live in a `Clone + Debug` task
//! state, so the task stores the handle's id and "calling the handle" is
//! `unregister_cleanup(id)`. Same semantics, and the id is what makes the
//! call idempotent — every terminal transition releases the registration, and
//! task removal may try again.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

type Cleanup = Box<dyn Fn() + Send + Sync>;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static CLEANUPS: Mutex<Vec<(u64, Cleanup)>> = Mutex::new(Vec::new());

/// A registration's id — the Rust stand-in for the source's returned remover.
/// `Copy` so a task state can hold it without owning a closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanupHandle(pub u64);

/// Register `f` to run during shutdown; returns its handle (registerCleanup).
pub fn register_cleanup<F>(f: F) -> CleanupHandle
where
    F: Fn() + Send + Sync + 'static,
{
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    CLEANUPS.lock().push((id, Box::new(f)));
    CleanupHandle(id)
}

/// Cancel a registration. Idempotent: an already-released handle is a no-op.
pub fn unregister_cleanup(handle: CleanupHandle) {
    CLEANUPS.lock().retain(|(id, _)| *id != handle.0);
}

/// How many cleanups are currently registered (tests / diagnostics).
pub fn cleanup_function_count() -> usize {
    CLEANUPS.lock().len()
}

/// Run every registered cleanup, draining the registry (runCleanupFunctions).
///
/// Best-effort and panic-isolated: this runs on the way out, and each
/// registration belongs to a different subsystem, so one failing must not stop
/// the others.
pub fn run_cleanup_functions() {
    let pending: Vec<(u64, Cleanup)> = std::mem::take(&mut *CLEANUPS.lock());
    for (_, f) in pending {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    }
}
