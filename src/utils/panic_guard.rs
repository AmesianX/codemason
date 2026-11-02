//! Marks a region where a panic is expected and handled by a catch_unwind.
//!
//! The panic hook installed in main restores the terminal, because a panic
//! unwinds past the prompt engine's teardown and would otherwise leave the user
//! in a shell with no echo, no cursor and bracketed paste still on. But a hook
//! runs for EVERY panic, including one a `catch_unwind` is about to swallow
//! deliberately — and restoring the terminal there is wrong: the program
//! carries on with the engine still believing it owns a raw terminal.
//!
//! Scoped to the thread rather than global: a panic on one worker says nothing
//! about what another thread is doing, and a global flag would let a swallowed
//! panic on one thread suppress the restoration a genuine panic on another
//! thread needs.

use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Guard object; the region ends when it drops.
pub struct ExpectedPanic;

impl Drop for ExpectedPanic {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Enter a region whose panics are caught and handled by the caller.
pub fn expected_panic() -> ExpectedPanic {
    DEPTH.with(|d| d.set(d.get() + 1));
    ExpectedPanic
}

/// True while this thread is inside such a region.
pub fn in_expected_panic() -> bool {
    DEPTH.with(|d| d.get()) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard has to nest and unwind correctly, or a swallowed panic leaves
    /// the flag stuck and a LATER real panic silently skips the terminal
    /// restoration — turning this safety net off exactly when it is needed.
    #[test]
    fn guard_is_scoped_and_survives_a_panic() {
        assert!(!in_expected_panic());
        {
            let _g = expected_panic();
            assert!(in_expected_panic());
            {
                let _h = expected_panic();
                assert!(in_expected_panic());
            }
            assert!(in_expected_panic(), "the outer guard must still hold");
        }
        assert!(!in_expected_panic(), "both guards dropped");

        // A panic caught inside the region must still leave the flag clear.
        let _ = std::panic::catch_unwind(|| {
            let _g = expected_panic();
            panic!("expected");
        });
        assert!(!in_expected_panic(), "unwinding must drop the guard");
    }

    /// It is thread-local: a swallowed panic on one thread must not suppress
    /// the restoration a genuine panic on another thread needs.
    #[test]
    fn guard_does_not_leak_across_threads() {
        let _g = expected_panic();
        assert!(in_expected_panic());
        let other = std::thread::spawn(|| in_expected_panic()).join().unwrap();
        assert!(!other, "another thread must not see this thread's guard");
    }
}
