//! Generator utilities for CodeMason
//! Equivalent to codemason/utils/generators.py (src/utils/generators.ts),
//! reformulated over futures Streams (the Rust shape of async generators).

use futures::stream::{Stream, StreamExt};
use futures::FutureExt;
use std::pin::Pin;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Grace window a tool gets to finish cooperatively after abort fires, before
/// its in-flight pull is dropped. Cooperative tools (bash/powershell watch the
/// abort token and SIGKILL their subprocess, then return an "interrupted"
/// result); the window lets that clean teardown run instead of dropping the
/// stream mid-await — which could orphan the subprocess. Non-cooperative tools
/// (a bare fetch) never return, so they hit the force-drop after the window.
const ABORT_GRACE: Duration = Duration::from_millis(500);

/// Yield from `stream`, but stop when `signal` is cancelled — dropping whatever
/// await the stream is stuck on if it doesn't finish cooperatively within the
/// grace window.
///
/// WHY: the abort_controller is otherwise only polled BETWEEN tools (query.rs).
/// A tool sitting in one long await would run to completion before the
/// interrupt is noticed, so ESC "does nothing" mid-tool. Racing each pull
/// against the cancellation catches the abort immediately. Applied once at the
/// single tool.call() dispatch point, so EVERY tool becomes abort-aware
/// uniformly. signal=None degrades to a plain passthrough.
pub fn abortable<T: Send + 'static>(
    mut stream: Pin<Box<dyn Stream<Item = T> + Send>>,
    signal: Option<CancellationToken>,
    grace: Option<Duration>,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    let Some(signal) = signal else {
        return stream;
    };
    let grace = grace.unwrap_or(ABORT_GRACE);
    Box::pin(async_stream::stream! {
        loop {
            let pull = stream.next();
            tokio::select! {
                item = pull => {
                    match item {
                        Some(v) => {
                            yield v;
                            if signal.is_cancelled() {
                                // deliver the item that completed alongside
                                // abort, then stop
                                return;
                            }
                        }
                        None => return,
                    }
                }
                _ = signal.cancelled() => {
                    // Abort fired with a pull in flight. Give the tool the grace
                    // window to finish cooperatively (kill its subprocess and
                    // return), then force it by dropping the stream.
                    match tokio::time::timeout(grace, stream.next()).await {
                        Ok(Some(v)) => {
                            yield v; // the tool's own "interrupted" result
                            return;
                        }
                        Ok(None) | Err(_) => return,
                    }
                }
            }
        }
    })
}

/// Run all streams concurrently up to a concurrency cap, yielding values as
/// they come in. Equivalent to src/utils/generators.ts all().
///
/// The Python port had to hand-roll task cleanup; futures' select_all/drop
/// semantics give us the same guarantee — dropping the combined stream drops
/// every inner stream (and their subprocess/file state) — so the explicit
/// finally block has a direct structural equivalent here.
///
/// Error propagation: the Python version re-raised the first stream error and
/// closed the rest. Streams of Result keep that shape — the caller stops on the
/// first Err it sees, and dropping this stream tears the survivors down.
pub fn all_streams<T: Send + 'static>(
    generators: Vec<Pin<Box<dyn Stream<Item = T> + Send>>>,
    concurrency_cap: usize,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    // futures::stream::select_all polls every stream; to honor the cap we feed
    // streams in as slots free up, mirroring the Python waiting/pending sets.
    Box::pin(async_stream::stream! {
        let mut waiting: std::collections::VecDeque<_> = generators.into();
        let mut active: futures::stream::SelectAll<Pin<Box<dyn Stream<Item = T> + Send>>> =
            futures::stream::SelectAll::new();
        let mut live = 0usize;
        while live < concurrency_cap {
            match waiting.pop_front() {
                Some(g) => {
                    active.push(g);
                    live += 1;
                }
                None => break,
            }
        }
        loop {
            match active.next().await {
                Some(item) => {
                    yield item;
                    // SelectAll drops exhausted streams internally; top up from
                    // the waiting queue when capacity frees.
                    let current = active.len();
                    if current < live {
                        live = current;
                    }
                    while live < concurrency_cap {
                        match waiting.pop_front() {
                            Some(g) => {
                                active.push(g);
                                live += 1;
                            }
                            None => break,
                        }
                    }
                }
                None => {
                    if let Some(g) = waiting.pop_front() {
                        active.push(g);
                        live = active.len();
                    } else {
                        return;
                    }
                }
            }
        }
    })
}

/// Convenience: a future's ready value as a single-item stream (used to adapt
/// one-shot results into the dispatch pipeline).
pub fn once_stream<T: Send + 'static>(
    fut: impl std::future::Future<Output = T> + Send + 'static,
) -> Pin<Box<dyn Stream<Item = T> + Send>> {
    Box::pin(fut.into_stream())
}
