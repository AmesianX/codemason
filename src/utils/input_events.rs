//! Single-owner terminal input.
//!
//! SUBSTRATE (Rust-only; the Python side gets this from prompt_toolkit): the
//! Python original runs on prompt_toolkit's contract of *one Application per
//! terminal at a time* — the resident prompt is an Application, every picker is
//! an Application, and `pause_prompt()` genuinely EXITS the resident one
//! (`app.exit(_PAUSE)`) before a sub-UI starts, so there is never a moment with
//! two input readers.  Python upholds that contract by convention; nothing
//! stops a second Application except discipline.
//!
//! crossterm has the same underlying constraint — its event reader is a
//! process-wide singleton, and constructing a second `EventStream` while one is
//! live blocks forever inside `EventStream::new()` — but the port originally
//! let the prompt engine and each picker construct their own stream.  The
//! result was exactly what the contract predicts: the second constructor hung,
//! and when both streams did exist, keystrokes were delivered to whichever
//! reader polled first (pickers missing keys, lists repainting badly).
//!
//! This module makes the contract a TYPE instead of a convention: the only way
//! to read terminal events is to `acquire()` the process-wide token, and the
//! `EventStream` lives strictly inside it.  Two owners cannot exist; a would-be
//! second owner waits (async) until the first drops.  While nobody owns the
//! input, no `EventStream` exists at all, so the tty is genuinely free — an
//! external pager (`less`) can own the terminal, and anything typed in a gap
//! sits in the kernel's tty buffer for the next owner, which is precisely what
//! happens between two prompt_toolkit Applications.
use crossterm::event::Event;
use futures::StreamExt;
use once_cell::sync::Lazy;
use std::sync::Arc;

/// The process-wide ownership token. `tokio::sync::Mutex` so a waiting
/// acquirer parks the task, not the thread.
static OWNER: Lazy<Arc<tokio::sync::Mutex<()>>> = Lazy::new(|| Arc::new(tokio::sync::Mutex::new(())));

/// Exclusive handle on terminal input. Constructing the `EventStream` only
/// after the lock is held — and dropping it before the lock releases (field
/// order below) — is the whole safety argument.
pub struct InputEvents {
    stream: crossterm::event::EventStream,
    _guard: tokio::sync::OwnedMutexGuard<()>, // dropped AFTER the stream
}

/// Take exclusive ownership of terminal input, waiting until the current
/// owner (if any) releases it.
pub async fn acquire() -> InputEvents {
    let guard = OWNER.clone().lock_owned().await;
    InputEvents { stream: crossterm::event::EventStream::new(), _guard: guard }
}

impl InputEvents {
    /// Next terminal event. Read errors are skipped (the resident loop's old
    /// `else { continue }` behavior); `None` means the stream is closed.
    pub async fn next(&mut self) -> Option<Event> {
        loop {
            match self.stream.next().await {
                Some(Ok(ev)) => return Some(ev),
                Some(Err(_)) => continue,
                None => return None,
            }
        }
    }
}
