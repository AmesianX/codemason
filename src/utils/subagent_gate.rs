//! How many sub-agents may run at once — across EVERY codemason process on
//! this machine.
//!
//! A sub-agent is a second tenant in the server's prefix cache: its own tool
//! block, its own system prompt, its own history, none of it shared with the
//! parent. The pool we serve against holds 1.62 conversations (vLLM,
//! 2026-09-07: "GPU KV cache size: 847,533 tokens, maximum concurrency for
//! 524,288 tokens per request: 1.62x"). One parent plus one sub-agent is
//! already tight; a second sub-agent — or a second codemason session's — evicts
//! the parent outright, and the parent then re-prefills its whole history when
//! control comes back (measured: ~200s at 267k, 1,334 tok/s cold).
//!
//! The limit is a set of advisory file locks under the config dir, one per
//! slot, so it holds between processes too. A spawn that finds every slot taken
//! waits, polling, and reports that it is waiting — a silent stall is a defect
//! in unattended runs.
//!
//! Two ways of holding a slot, one meaning. On unix it is flock(LOCK_EX |
//! LOCK_NB), which is per open file description, so two spawns inside one
//! process contend just as two processes do. Windows has no flock and std
//! exposes no LockFileEx, so the lock there is an exclusive open of the slot
//! file itself — share_mode(0), which the OS grants only while no other handle
//! is open on that file, in this process or any other. Same per-handle
//! semantics, and the same release-on-death: both are held by an open handle,
//! so a crash or a taskkill frees the slot instead of stranding it. (This is
//! why the slot file's EXISTENCE is deliberately not the lock, the way
//! remote::inbox's sidecar is — that design needs FILE_FLAG_DELETE_ON_CLOSE
//! and a reclaim path to survive a hard kill, and here there is nothing to
//! reclaim.)

use crate::utils::config::get_config;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// A held slot; the lock is released on drop.
pub struct SubagentSlot {
    _file: std::fs::File,
}

fn slots() -> usize {
    get_config().max_concurrent_subagents.max(1)
}

/// Slot `i` of the pool for THIS endpoint.
///
/// The pool is keyed by the model server, because that is whose capacity it
/// rations: the limit exists because one vLLM's KV cache holds 1.62
/// conversations. A single global pool got that wrong as soon as two sessions
/// pointed at different endpoints — the default is ONE slot, so a sub-agent on
/// endpoint A blocked every sub-agent on endpoint B from starting at all, for a
/// GPU it was not using. Same correction, same reason, as the Telegram poll lock
/// being keyed by bot token instead of by home directory.
///
/// Slot files are empty and are never deleted; one set per endpoint ever used is
/// a handful of 0-byte files, and deleting a flocked path is how you get two
/// holders of one slot. See utils::file for the rules.
/// `endpoint` is the tag of the pool — passed in rather than read here, so one
/// scan for a free slot cannot straddle an endpoint switch and compare slot 0 of
/// one pool with slot 1 of another.
fn slot_path(endpoint: &str, i: usize) -> std::path::PathBuf {
    let dir = crate::utils::config::LOCKS_DIR.clone();
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("subagent.{endpoint}.{i}.lock"))
}

fn try_slot(endpoint: &str, i: usize) -> Option<SubagentSlot> {
    let path = slot_path(endpoint, i);

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let file =
            std::fs::OpenOptions::new().create(true).read(true).write(true).open(&path).ok()?;
        // SAFETY: plain flock(2) on a descriptor we own for the file's lifetime.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            Some(SubagentSlot { _file: file })
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        // The open IS the lock: share_mode(0) asks for the file with no sharing,
        // which Windows grants only when nobody else has it open, and refuses
        // with a sharing violation otherwise. That refusal is exactly the
        // "slot busy" answer LOCK_NB gives, without blocking.
        //
        // A scanner or indexer holding the file open for a moment can therefore
        // read as busy when it is not. The cost of that is one more 2s poll in
        // acquire(), never a lost or double-granted slot, so it is left alone.
        use std::os::windows::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .share_mode(0)
            .open(&path)
            .ok()?;
        Some(SubagentSlot { _file: file })
    }
}

/// A free slot right now, or None without waiting.
pub fn try_acquire() -> Option<SubagentSlot> {
    try_acquire_from(&crate::utils::config::endpoint_tag(None))
}

/// try_acquire against a named pool. The endpoint is resolved once per scan:
/// `acquire` re-resolves it on every poll, so switching endpoints while waiting
/// moves the wait to the pool the request will actually use.
fn try_acquire_from(endpoint: &str) -> Option<SubagentSlot> {
    (0..slots()).find_map(|i| try_slot(endpoint, i))
}

/// Take a slot, waiting (polling) for one if all are busy. Returns None only
/// if the parent aborted while waiting. Callers announce the wait themselves —
/// try_acquire() first, and if that fails, say so, then call this.
pub async fn acquire(abort: &CancellationToken) -> Option<SubagentSlot> {
    acquire_pool(None, abort).await
}

/// `pool: None` re-resolves the endpoint on every poll, so switching endpoints
/// while waiting moves the wait to the pool the request will actually use. A
/// named pool is for tests, which must not depend on a config value another test
/// in the same process is free to change.
async fn acquire_pool(pool: Option<&str>, abort: &CancellationToken) -> Option<SubagentSlot> {
    loop {
        let slot = match pool {
            Some(p) => try_acquire_from(p),
            None => try_acquire(),
        };
        if let Some(slot) = slot {
            return Some(slot);
        }
        tokio::select! {
            _ = abort.cancelled() => return None,
            _ = tokio::time::sleep(Duration::from_secs(2)) => {}
        }
    }
}

impl Drop for SubagentSlot {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: descriptor still open; the File closes right after.
            unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
        }
        // Windows needs no counterpart: closing the handle — which happens as
        // this File drops, a line from here — is what drops the exclusive open.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn one_slot_serialises_and_frees_on_drop() {
        let dir = std::env::temp_dir().join(format!("cm-gate-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("CODEMASON_CONFIG_DIR", &dir);
        // A named pool, not whatever endpoint the live config happens to hold:
        // the pool is keyed by the model server now, and another test in this
        // process may be pointing the config at a stub server of its own.
        let pool = "test-endpoint";
        // Fill the pool instead of assuming its size. get_config() is a cached
        // singleton, so a config.json written here is ignored once any other test
        // in this process has loaded one; and the default is per line (1 where
        // sub-agents are serial, 3 where they are not). What the case is about is
        // that the pool has a bottom and that a drop returns a slot to it, which
        // holds at any size.
        let mut held: Vec<SubagentSlot> = Vec::new();
        while let Some(s) = try_acquire_from(pool) {
            held.push(s);
            assert!(held.len() <= 64, "the pool must have a bottom");
        }
        assert_eq!(held.len(), slots(), "the pool holds exactly what it advertises");
        let first = held.pop().expect("at least one slot");
        // Every slot taken but the one just popped — which is still held by
        // `first`, so a fresh acquire must still come back empty.
        assert!(try_acquire_from(pool).is_none());
        // A sub-agent on a DIFFERENT endpoint must not be blocked by this one:
        // the pool it waits on is the server whose capacity it is spending.
        assert!(
            try_acquire_from("another-endpoint").is_some(),
            "a second endpoint has its own pool"
        );
        // ...and the waiting form must give up when the parent aborts, rather
        // than waiting forever.
        let aborted = CancellationToken::new();
        aborted.cancel();
        assert!(acquire_pool(Some(pool), &aborted).await.is_none());
        drop(first);
        assert!(try_acquire_from(pool).is_some());
    }
}
