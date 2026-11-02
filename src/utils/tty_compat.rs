//! Cross-platform TTY control.
//! Equivalent to codemason/utils/tty_compat.py.
//!
//! The Python module shimmed POSIX termios/tty so native Windows didn't crash
//! at import. In Rust, crossterm already abstracts raw-mode/echo control on
//! both platforms, so this module is the thin seam callers use instead of
//! touching termios directly — on failure every operation degrades to "no
//! terminal control", the same fallback the Python except-clauses took.

/// Enable raw mode; Err = not a TTY / unsupported (callers fall back).
pub fn enable_raw_mode() -> std::io::Result<()> {
    crossterm::terminal::enable_raw_mode().map_err(std::io::Error::other)
}

/// Disable raw mode (no-op on failure, like the Python tcsetattr stub).
pub fn disable_raw_mode() {
    let _ = crossterm::terminal::disable_raw_mode();
}

/// Flush pending input (termios.tcflush(TCIFLUSH) equivalent): drain any
/// buffered key events without blocking.
pub fn flush_input() {
    while let Ok(true) = crossterm::event::poll(std::time::Duration::from_millis(0)) {
        let _ = crossterm::event::read();
    }
}

/// Terminal echo suppression for the startup window.
///
/// DIVERGENCE (Rust-only): the prompt engine cannot own the terminal until it
/// has something to draw, and startup does real work first (model resolve,
/// remote channel). Until raw mode begins the tty is canonical, so the KERNEL
/// echoes anything typed — painting it across the banner and leaving it queued.
/// `flush_input` drops the queued keys, but only silencing the echo keeps them
/// off the screen. Echo is suppressed for the startup window and restored on
/// exit; canonical mode is otherwise untouched, so banner output is unaffected.
#[cfg(unix)]
mod echo {
    use once_cell::sync::Lazy;
    use parking_lot::Mutex;

    static SAVED: Lazy<Mutex<Option<libc::termios>>> = Lazy::new(|| Mutex::new(None));

    pub fn suppress() {
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
                return; // not a tty
            }
            let mut saved = SAVED.lock();
            if saved.is_none() {
                *saved = Some(t);
            }
            t.c_lflag &= !libc::ECHO;
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
        }
    }

    pub fn restore() {
        if let Some(t) = SAVED.lock().take() {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
            }
        }
    }
}

#[cfg(not(unix))]
mod echo {
    pub fn suppress() {}
    pub fn restore() {}
}

/// Stop the kernel echoing keystrokes typed before the prompt exists.
pub fn suppress_echo() {
    echo::suppress();
}

/// Put echo back the way it was found.
pub fn restore_echo() {
    echo::restore();
}
