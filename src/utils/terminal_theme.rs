//! Optional terminal colour-theme application via OSC escape sequences.
//! Equivalent to codemason/utils/terminal_theme.py — see that file's module
//! doc for the restore strategy (atexit + SIGTERM/SIGHUP, deliberately not
//! SIGINT) and the honest SIGKILL limitation.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{IsTerminal, Write};

/// macOS Terminal.app "Homebrew" profile: bright lime-green text on black.
/// 'prompt'/'input' are NOT terminal OSC colours — they're consumed by the
/// REPL's prompt style (cli.rs) to keep the '>' marker and typed text from
/// disappearing into the all-green body.
fn themes() -> HashMap<&'static str, HashMap<&'static str, &'static str>> {
    let mut homebrew = HashMap::new();
    homebrew.insert("fg", "#28fe14");
    homebrew.insert("bg", "#000000");
    homebrew.insert("cursor", "#28fe14");
    homebrew.insert("prompt", "#ffb000");
    homebrew.insert("input", "#f5f0e8");
    let mut m = HashMap::new();
    m.insert("homebrew", homebrew);
    m
}

struct ThemeState {
    applied: bool,
    active_name: Option<String>,
}

static STATE: Lazy<Mutex<ThemeState>> =
    Lazy::new(|| Mutex::new(ThemeState { applied: false, active_name: None }));

/// Colour map of the theme applied this session (or {} if none). Lets the
/// REPL tint its prompt/input to match.
pub fn active_theme_colors() -> HashMap<&'static str, &'static str> {
    let state = STATE.lock();
    match &state.active_name {
        Some(name) => themes().get(name.as_str()).cloned().unwrap_or_default(),
        None => HashMap::new(),
    }
}

/// Build an OSC sequence terminated with ST (ESC \).
fn osc(code: &str, value: &str) -> String {
    if value.is_empty() {
        format!("\x1b]{code}\x1b\\")
    } else {
        format!("\x1b]{code};{value}\x1b\\")
    }
}

fn write_seq(seq: &str) {
    let mut out = std::io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// Revert fg/bg/cursor/palette to the terminal's OWN configured defaults via
/// OSC reset codes — no need to know the original values. Idempotent.
pub fn restore() {
    {
        let mut state = STATE.lock();
        if !state.applied {
            return;
        }
        state.applied = false;
        state.active_name = None;
    }
    write_seq(&osc("110", "")); // reset default foreground
    write_seq(&osc("111", "")); // reset default background
    write_seq(&osc("112", "")); // reset cursor colour
    write_seq(&osc("104", "")); // reset colour palette
}

/// Apply a named terminal theme for this session and arrange restoration.
///
/// Returns true if applied. No-op (false) for an unknown theme, a non-tty
/// stdout, or pure-ASCII/plain mode (dumb terminals don't do OSC).
pub fn apply_theme(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let name = name.trim().to_lowercase();
    let all = themes();
    let Some(theme) = all.get(name.as_str()) else {
        return false;
    };
    // Don't paint a non-interactive stream or a dumb/plain terminal.
    if crate::utils::formatting::plain_mode() || !std::io::stdout().is_terminal() {
        return false;
    }

    write_seq(&osc("11", theme["bg"])); // default background
    write_seq(&osc("10", theme["fg"])); // default foreground
    write_seq(&osc("12", theme["cursor"])); // cursor
    {
        let mut state = STATE.lock();
        state.active_name = Some(name);
        state.applied = true;
    }

    // Restore on terminating signals. SIGTERM/SIGHUP only — Ctrl+C in the REPL
    // often means "cancel the current query", not "quit"; the normal-exit path
    // calls restore() from the CLI's shutdown (the atexit equivalent). SIGHUP
    // does not exist on native Windows.
    #[cfg(unix)]
    {
        for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGHUP] {
            // Emit the reset then re-raise the default behavior; register is
            // unsafe-free via signal_hook's low-level flag pattern — here we
            // use signal_hook::flag + a small emergency handler.
            let _ = unsafe {
                signal_hook::low_level::register(sig, move || {
                    // async-signal-safe enough for our purpose: write() of a
                    // static byte string.
                    let seq = b"\x1b]110\x1b\\\x1b]111\x1b\\\x1b]112\x1b\\\x1b]104\x1b\\";
                    unsafe {
                        libc::write(1, seq.as_ptr() as *const _, seq.len());
                        libc::signal(sig, libc::SIG_DFL);
                        libc::raise(sig);
                    }
                })
            };
        }
    }
    true
}
