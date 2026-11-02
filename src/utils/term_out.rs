//! Global output router — the Rust substrate for prompt_toolkit's
//! patch_stdout(raw=True).
//!
//! In the Python port, every print() during the interactive REPL was proxied
//! ABOVE the live prompt by patch_stdout. Rust has no stdout monkeypatching,
//! so the equivalent is an explicit router: display code prints through
//! `cmprintln!` / `cmprint!` (or `emit`), and while a PromptEngine is active
//! those writes are handed to it — the engine erases its bottom prompt block,
//! writes the text into scrollback, and repaints the prompt. With no engine
//! active the macros degrade to plain stdout, so headless (-p) and plain-mode
//! output is byte-identical to a normal print.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::sync::Arc;

/// Sink installed by the interactive prompt engine. Receives COMPLETE text
/// chunks (may contain embedded newlines; a trailing newline means println).
pub trait OutputSink: Send + Sync {
    fn write_above_prompt(&self, text: &str);
}

static SINK: Lazy<Mutex<Option<Arc<dyn OutputSink>>>> = Lazy::new(|| Mutex::new(None));

/// Install (or clear) the engine sink. The interactive REPL installs it for
/// its lifetime; everything else leaves it None.
pub fn set_sink(sink: Option<Arc<dyn OutputSink>>) {
    *SINK.lock() = sink;
}

pub fn sink_active() -> bool {
    SINK.lock().is_some()
}

/// Route one chunk of output: above the live prompt when an engine is active,
/// else straight to stdout.
pub fn emit(text: &str) {
    let sink = SINK.lock().clone();
    match sink {
        Some(s) => s.write_above_prompt(text),
        None => {
            use std::io::Write;
            let mut out = std::io::stdout();
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }
    }
}

/// println through the router (the interactive-safe replacement for println!).
#[macro_export]
macro_rules! cmprintln {
    () => { $crate::utils::term_out::emit("\n") };
    ($($arg:tt)*) => {{
        let mut s = format!($($arg)*);
        s.push('\n');
        $crate::utils::term_out::emit(&s);
    }};
}

/// print (no newline) through the router.
#[macro_export]
macro_rules! cmprint {
    ($($arg:tt)*) => {{
        let s = format!($($arg)*);
        $crate::utils::term_out::emit(&s);
    }};
}
