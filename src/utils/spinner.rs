//! Animated spinner with real-time token count display.
//! Equivalent to the StreamingSpinner class in codemason/cli.py (plus its
//! `_spinner_frames` / `_terminal_supports_braille` helpers).
//!
//! Two render modes, exactly as the Python:
//!   - owned (`external_render=false`): the spinner \r-repaints the terminal
//!     itself, animated by a background task (sequential/--plain mode).
//!   - external (`external_render=true`): the interactive REPL renders it in
//!     the prompt engine's message block, repainted on the engine's
//!     refresh_interval — this object must then never write to the terminal.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::utils::formatting::{
    clip_to_width, colorize, colors, display_width, format_tokens_compact, plain_mode, sym,
};

const BRAILLE_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const ASCII_FRAMES: &[&str] = &["|", "/", "-", "\\"];

static CACHED_FRAMES: Lazy<Mutex<Option<&'static [&'static str]>>> = Lazy::new(|| Mutex::new(None));

/// Best-effort check for braille (U+2800–28FF) glyph rendering.
///
/// HONEST LIMITS — there is NO terminal capability that reports font glyph
/// coverage. So 'auto' can only check:
///   1. UTF-8 output — necessary, not sufficient (reliable: a non-UTF-8
///      terminal cannot render braille at all).
///   2. A cursor-advance (DSR ESC[6n) probe — catches braille drawn at the
///      wrong cell width. It does NOT catch the 'tofu box' case, where a
///      missing glyph still occupies exactly 1 cell.
/// Set CODEMASON_SPINNER=ascii to force ASCII there.
/// Usable columns for the one-line repaint.
///
/// Read fresh on every tick rather than cached: the window can be resized while
/// a turn runs, which is exactly when the stale-line bug showed up. One column
/// is held back because some terminals wrap as soon as the last cell is filled
/// rather than when the next character arrives. The fallback of 80 is the
/// conservative choice — clipping too narrow leaves the line short, clipping
/// too wide brings the stacking back.
fn terminal_columns() -> usize {
    crossterm::terminal::size()
        .map(|(c, _)| c as usize)
        .unwrap_or(80)
        .saturating_sub(1)
        .max(20)
}

fn terminal_supports_braille() -> bool {
    // Rust always writes UTF-8, so step 1 reduces to "is the terminal able to
    // show it" — approximated by the locale, the same signal Python's
    // sys.stdout.encoding carried.
    let enc = std::env::var("LC_ALL")
        .or_else(|_| std::env::var("LC_CTYPE"))
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default()
        .to_lowercase();
    if !enc.is_empty() && !enc.contains("utf") {
        return false; // reliable: non-UTF-8 can't render braille at all
    }
    if !(std::io::stdout().is_terminal() && std::io::stdin().is_terminal()) {
        return true; // can't probe (piped); prefer the nicer spinner
    }

    // DSR probe: measure the cursor advance of one braille cell.
    let probe = || -> Option<bool> {
        crate::utils::tty_compat::enable_raw_mode().ok()?;
        let result = (|| -> Option<bool> {
            let col = |out: &mut std::io::Stdout| -> Option<u16> {
                write!(out, "\x1b[6n").ok()?;
                out.flush().ok()?;
                read_dsr_col(std::time::Duration::from_millis(150))
            };
            let mut out = std::io::stdout();
            let c0 = col(&mut out)?;
            write!(out, "⠿").ok()?; // a fully-set braille cell
            out.flush().ok()?;
            let c1 = col(&mut out);
            write!(out, "\x08 \x08").ok()?; // erase the probe glyph
            out.flush().ok()?;
            let c1 = c1?;
            Some(c1.saturating_sub(c0) == 1) // exactly one cell
        })();
        crate::utils::tty_compat::disable_raw_mode();
        result
    };
    probe().unwrap_or(true) // no answer / failure → don't penalise braille
}

/// Read a Device Status Report reply (ESC[<row>;<col>R) within `timeout`.
fn read_dsr_col(timeout: std::time::Duration) -> Option<u16> {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        match crossterm::event::poll(remaining) {
            Ok(true) => {
                if let Ok(crossterm::event::Event::Key(k)) = crossterm::event::read() {
                    let _ = k; // crossterm parses CPR into a CursorPosition event on some
                               // backends; fall through to the direct read below.
                }
            }
            _ => break,
        }
    }
    // crossterm exposes the cursor position query directly; prefer it.
    crossterm::cursor::position().ok().map(|(c, _r)| c + 1)
}

/// Pick spinner frames. Default braille; CODEMASON_SPINNER overrides with
/// 'braille' | 'ascii' | 'auto'. Result is cached (the probe runs at most once).
pub fn spinner_frames() -> &'static [&'static str] {
    let mut cache = CACHED_FRAMES.lock();
    if let Some(f) = *cache {
        return f;
    }
    if plain_mode() {
        *cache = Some(ASCII_FRAMES); // pure-ASCII: never braille/probe
        return ASCII_FRAMES;
    }
    let mode = std::env::var("CODEMASON_SPINNER")
        .unwrap_or_else(|_| "auto".into())
        .trim()
        .to_lowercase();
    let frames = match mode.as_str() {
        "ascii" => ASCII_FRAMES,
        "braille" => BRAILLE_FRAMES,
        _ => {
            if terminal_supports_braille() {
                BRAILLE_FRAMES
            } else {
                ASCII_FRAMES
            }
        }
    };
    *cache = Some(frames);
    frames
}

/// The label when no caller pinned one — i.e. while the MODEL is working,
/// not a tool.
///
/// Upstream always samples a whimsical verb here. Off by default: the words
/// are chosen at random per turn, so the line says something different every
/// time while nothing about the work has changed, and next to a tool label
/// that IS meaningful it reads as noise. "Thinking" is what is actually
/// happening. config.spinner_random_verb brings the sampling back.
fn pick_verb() -> String {
    if !crate::utils::config::get_config().spinner_random_verb {
        return "Thinking".to_string();
    }
    use rand::seq::SliceRandom;
    let verbs = crate::constants::spinner_verbs::get_spinner_verbs();
    verbs
        .choose(&mut rand::thread_rng())
        .map(|v| v.to_string())
        .unwrap_or_else(|| "Thinking".to_string())
}

/// Shared spinner state (the REPL holds one across the whole session).
/// Prefill big enough that the wait is the prompt rather than the model, and
/// so worth saying out loud without -dbg. At the cold rate measured for this
/// deployment (1,335 tok/s) this is roughly 15 seconds before the first token
/// -- about where an unexplained spinner starts reading as a hang. Below it the
/// status line stays as clean as it was.
const PREFILL_NOTEWORTHY_TOKENS: i64 = 20_000;

pub struct StreamingSpinner {
    frames: &'static [&'static str],
    frame_idx: AtomicUsize,
    token_count: AtomicI64,
    /// Reasoning tokens for THIS turn. Stays 0 on servers/turns that don't
    /// think, and the status line then looks exactly as it did before.
    thinking_tokens: AtomicI64,
    start_time: Mutex<Option<std::time::Instant>>,
    active: AtomicBool,
    last_line_length: AtomicUsize,
    /// The REPL renders this spinner in the prompt engine's block; this object
    /// must then never write to the terminal itself.
    external_render: bool,
    /// Whimsical verb shown while working, picked once per turn and held stable.
    verb: Mutex<String>,
    /// Set when a caller PINNED the verb via `start(Some(..))`. A pinned verb
    /// names work the user cannot infer from anything on screen (compaction),
    /// so display_verb returns it instead of the in_progress todo's
    /// activeForm — that todo describes the task compaction INTERRUPTED, so
    /// showing it through a reduction reports the wrong thing entirely.
    ///
    /// Atomic rather than a sentinel inside `verb`: display_verb must test it
    /// without taking a lock, because `render()` reads the verb and a setter
    /// holding `verb` across a render is the paint-vs-verb inversion.
    verb_pinned: AtomicBool,
    /// The label `pin_verb` displaced, restored when it unpins. Without it an
    /// unpin had to invent a new whimsical verb, so a turn interrupted by a
    /// compaction came back wearing a different name than it went in with.
    verb_saved: Mutex<Option<String>>,
    animation: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl StreamingSpinner {
    pub fn new(external_render: bool) -> Arc<Self> {
        Arc::new(StreamingSpinner {
            frames: spinner_frames(),
            frame_idx: AtomicUsize::new(0),
            token_count: AtomicI64::new(0),
            thinking_tokens: AtomicI64::new(0),
            start_time: Mutex::new(None),
            active: AtomicBool::new(false),
            last_line_length: AtomicUsize::new(0),
            external_render,
            verb: Mutex::new(pick_verb()),
            verb_pinned: AtomicBool::new(false),
            verb_saved: Mutex::new(None),
            animation: Mutex::new(None),
        })
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Claude Code verb priority (Spinner.tsx:169): the in_progress todo's
    /// activeForm (authored by the model in TodoWrite) takes precedence, else
    /// its content, else the stable random whimsical verb. Resolved LIVE on
    /// each render so it tracks todo changes mid-turn.
    fn display_verb(&self) -> String {
        let fallback = self.verb.lock().clone();
        // A pinned verb wins. Without this, `start(Some("Compacting"))` was
        // silently overridden whenever a todo was in_progress — the caller
        // asked for a name and got the interrupted task's name instead.
        if self.verb_pinned.load(Ordering::SeqCst) {
            return fallback;
        }
        // The MAIN conversation's todos — the same key TodoWrite writes under.
        // This used to read get_session_id(), a UUID that nothing ever stored
        // against, so the branch below never ran.
        let key = crate::utils::todo_store::todo_key(None);
        for t in crate::utils::todo_store::get_todos(&key) {
            if t.get("status").and_then(|s| s.as_str()) == Some("in_progress") {
                let v = t
                    .get("activeForm")
                    .and_then(|s| s.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .or_else(|| t.get("content").and_then(|s| s.as_str()))
                    .unwrap_or(&fallback)
                    .trim()
                    .to_string();
                return if v.is_empty() { fallback } else { v };
            }
        }
        fallback
    }

    /// Start the spinner (resets token count + picks a fresh verb). `verb`
    /// pins the label instead of picking a whimsical one, for work that is not
    /// a turn and whose name is the useful part (compaction).
    pub fn start(self: &Arc<Self>, verb: Option<&str>) {
        *self.start_time.lock() = Some(std::time::Instant::now());
        self.token_count.store(0, Ordering::SeqCst);
        self.thinking_tokens.store(0, Ordering::SeqCst);
        self.frame_idx.store(0, Ordering::SeqCst);
        self.active.store(true, Ordering::SeqCst);
        *self.verb.lock() = verb.map(|v| v.to_string()).unwrap_or_else(pick_verb);
        // Guard dropped above before render(): never hold `verb` into a render.
        self.verb_pinned.store(verb.is_some(), Ordering::SeqCst);
        *self.verb_saved.lock() = None; // a new turn carries no displaced label
        self.render();
        self.start_animation();
    }

    /// Re-pin the label MID-TURN without touching the counters, for work that
    /// interrupts a turn and then hands it back (a compaction). `None` unpins
    /// and restores the label the turn went in with.
    ///
    /// start() cannot be used for this: it resets the elapsed clock, the token
    /// count and the thinking count, so a reduction in the middle of a turn
    /// would throw away the turn's own progress display.
    ///
    /// The elapsed clock stays the TURN's, deliberately. It is the one number
    /// here that keeps describing the turn rather than the interruption, so a
    /// long summary reads as "Compacting... (152s)" where the 152s is the
    /// turn's age. Resetting it instead would make the turn's time jump
    /// backwards the moment the label is handed back, which is worse.
    ///
    /// Lock discipline, and the whole reason this is a method and not two
    /// field writes at the call site: each guard is taken, used and DROPPED
    /// before the next is taken, and `verb` is never held into `render()` --
    /// render() reads it back through display_verb(), so a setter holding it
    /// across a render is the only inversion this object makes available.
    /// Same shape as start(). No new lock is introduced: `verb` already
    /// existed and `verb_pinned` is an atomic, which cannot block.
    pub fn pin_verb(self: &Arc<Self>, verb: Option<&str>) {
        match verb {
            Some(v) => {
                let prev = { self.verb.lock().clone() };
                {
                    let mut saved = self.verb_saved.lock();
                    if saved.is_none() {
                        *saved = Some(prev);
                    }
                }
                *self.verb.lock() = v.to_string();
                self.verb_pinned.store(true, Ordering::SeqCst);
            }
            None => {
                let restore = { self.verb_saved.lock().take() };
                if let Some(prev) = restore {
                    *self.verb.lock() = prev;
                }
                self.verb_pinned.store(false, Ordering::SeqCst);
            }
        }
        // Every guard above is out of scope here.
        self.render();
    }

    /// Resume without resetting the token count (for Agent progress).
    pub fn resume(self: &Arc<Self>) {
        if !self.active.load(Ordering::SeqCst) {
            self.active.store(true, Ordering::SeqCst);
            let mut st = self.start_time.lock();
            if st.is_none() {
                *st = Some(std::time::Instant::now());
            }
            drop(st);
            self.render();
            self.start_animation();
        }
    }

    fn start_animation(self: &Arc<Self>) {
        if self.external_render {
            return; // the engine repaints on refresh_interval; no task needed
        }
        let mut slot = self.animation.lock();
        if slot.as_ref().map(|h| !h.is_finished()).unwrap_or(false) {
            return;
        }
        let me = self.clone();
        // Requires a tokio runtime; in a non-async context the spinner simply
        // does not animate (the Python's RuntimeError branch).
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            *slot = Some(handle.spawn(async move {
                while me.active.load(Ordering::SeqCst) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    if me.active.load(Ordering::SeqCst) {
                        me.frame_idx.fetch_add(1, Ordering::SeqCst);
                        me.render();
                    }
                }
            }));
        }
    }

    /// Update token count and re-render.
    pub fn update(&self, token_count: i64) {
        if !self.active.load(Ordering::SeqCst) {
            return;
        }
        self.token_count.store(token_count, Ordering::SeqCst);
        self.render();
    }

    /// Update the reasoning-token count and re-render. Only ever called while
    /// the server is emitting reasoning, so a turn with no thinking never
    /// touches this and the counter stays hidden.
    pub fn update_thinking(&self, thinking_tokens: i64) {
        if !self.active.load(Ordering::SeqCst) {
            return;
        }
        self.thinking_tokens.store(thinking_tokens, Ordering::SeqCst);
        self.render();
    }

    /// ' — fetching example.com', or '' when no tool has said what it is
    /// waiting on. Sits next to the elapsed time, where the question is asked.
    fn wait_segment(&self) -> String {
        match crate::utils::tokens::wait_note() {
            Some(n) if !n.is_empty() => format!(" {} {n}", sym("dash")),
            _ => String::new(),
        }
    }

    /// ' · ↑ 73.1k prefill' while the server is still reading the prompt and
    /// nothing has come back yet.
    ///
    /// Before the first output token there is no way to tell a warm request
    /// (answer in ~2s) from one whose cached prefix broke (73k re-read, 55s) —
    /// both show a spinner and a clock. This says which, and how much is left
    /// to read, so a long wait is legible instead of looking like a freeze.
    /// It disappears the moment real output starts.
    fn prefill_segment(&self) -> String {
        // Diagnostic, not everyday furniture: only under -dbg.
        //
        // It earned its place by settling an argument — 297 seconds of
        // "Calculating…" turned out to be "↑ 996 to prefill, 54.3k cached ·
        // ↓ 0 tokens", i.e. neither prefill nor decode, which is what finally
        // pointed at a tool waiting on itself. That question is answered now,
        // and a number nobody is currently asking about is just noise on the
        // line. -dbg brings it back, along with turns.log.
        if self.token_count.load(Ordering::SeqCst) > 0 {
            return String::new();
        }
        let p = crate::utils::tokens::pending_prefill();
        if p <= 0 {
            return String::new();
        }
        // Under -dbg: always, as a diagnostic. Otherwise only when the wait is
        // the PROMPT and not the model, which is the one case where silence is
        // indistinguishable from a freeze.
        //
        // The clean-line argument above still holds for ordinary turns, and
        // they stay clean: a warm request re-reads a few thousand tokens and
        // never crosses the bar. What it did not cover is a reduction clearing
        // an early tool result, which invalidates every cached token after it:
        // on 2026-10-03 17:55 that was 242,795 tokens at 0.0% cache hit and 182
        // seconds before the first token, spent showing the turn's own label
        // and no reason at all.
        if !crate::utils::log::debug_enabled() && p < PREFILL_NOTEWORTHY_TOKENS {
            return String::new();
        }
        let cached = crate::utils::tokens::pending_cached();
        // '~' marks our own estimate; no mark means vLLM counted it.
        let mark = if crate::utils::tokens::prefill_is_measured() { "" } else { "~" };
        let hit = if cached > 0 {
            format!(", {} cached", format_tokens_compact(cached))
        } else {
            String::new()
        };
        format!(
            " {} {} {}{} to prefill{}",
            sym("dot"),
            sym("up"),
            mark,
            format_tokens_compact(p),
            hit
        )
    }

    /// ' · ✻ 234 thinking', or '' when this turn did no thinking. The thinking
    /// tokens are a SUBSET of token_count (a breakdown, not an extra total).
    fn thinking_segment(&self) -> String {
        let t = self.thinking_tokens.load(Ordering::SeqCst);
        if t <= 0 {
            return String::new();
        }
        format!(" {} {} {} thinking", sym("dot"), sym("think"), format_tokens_compact(t))
    }

    /// Status line for the prompt engine's block (the external_render
    /// counterpart of render). Frame derived from elapsed time.
    pub fn toolbar_text(&self) -> String {
        let start = *self.start_time.lock();
        let (Some(start), true) = (start, self.active.load(Ordering::SeqCst)) else {
            return String::new();
        };
        let elapsed = start.elapsed().as_secs_f64();
        let frame = self.frames[((elapsed * 10.0) as usize) % self.frames.len()];
        format!(
            "{frame} {}{} ({elapsed:.1}s {} {} {} tokens{}{}){}",
            self.display_verb(),
            sym("ellipsis"),
            sym("dot"),
            sym("down"),
            format_tokens_compact(self.token_count.load(Ordering::SeqCst)),
            self.thinking_segment(),
            self.prefill_segment(),
            self.wait_segment()
        )
    }

    /// Render the spinner to the terminal (owned mode only).
    fn render(&self) {
        if self.external_render {
            return;
        }
        let start = *self.start_time.lock();
        let (Some(start), true) = (start, self.active.load(Ordering::SeqCst)) else {
            return;
        };
        let elapsed = start.elapsed().as_secs_f64();
        let frame = self.frames[self.frame_idx.load(Ordering::SeqCst) % self.frames.len()];

        // Claude Code form: "✽ Cogitating… (16s · ↓ 759 tokens …)" — the ↓
        // marks tokens streaming IN to the client.
        let status = format!(
            "{frame} {}{} ({elapsed:.1}s {} {} {} tokens{}{}){}",
            self.display_verb(),
            sym("ellipsis"),
            sym("dot"),
            sym("down"),
            format_tokens_compact(self.token_count.load(Ordering::SeqCst)),
            self.thinking_segment(),
            self.prefill_segment(),
            self.wait_segment()
        );

        // Clip to the terminal, and measure in COLUMNS rather than chars.
        //
        // The repaint is `\r`, spaces, `\r`, which only reaches the line the
        // cursor is on. Let the status run past the window and the terminal
        // wraps it; `\r` then lands at the start of the LAST visual row and the
        // rows above it are never cleared, so every tick leaves another copy
        // behind. Reported 2026-09-13 from a narrowed window: eight stale
        // "Thinking…" lines stacked up between two tool calls.
        //
        // Two separate mistakes, both fixed here. Clipping keeps the line on one
        // row so `\r` can always reach all of it. And the blank run is sized by
        // display_width, not chars().count(): Hangul and the box/arrow glyphs in
        // this line occupy two columns each, so counting characters wiped about
        // half of what was actually drawn.
        let width = terminal_columns();
        let status = clip_to_width(&status, width);
        let clear = " ".repeat(self.last_line_length.load(Ordering::SeqCst).min(width));
        let mut out = std::io::stdout();
        let _ = write!(out, "\r{clear}\r");
        let _ = write!(out, "{}", colorize(&status, colors::CYAN));
        let _ = out.flush();
        self.last_line_length.store(display_width(&status), Ordering::SeqCst);
    }

    /// Stop the spinner and clear the line.
    pub fn stop(&self) {
        if !self.active.load(Ordering::SeqCst) {
            return;
        }
        self.active.store(false, Ordering::SeqCst);
        if let Some(h) = self.animation.lock().take() {
            h.abort();
        }
        if self.external_render {
            return; // nothing on the terminal to clear
        }
        let clear = " ".repeat(
            self.last_line_length.load(Ordering::SeqCst).min(terminal_columns()),
        );
        let mut out = std::io::stdout();
        let _ = write!(out, "\r{clear}\r");
        let _ = out.flush();
    }

    /// Final elapsed time and token count.
    pub fn get_final_stats(&self) -> (f64, i64) {
        match *self.start_time.lock() {
            Some(start) => (start.elapsed().as_secs_f64(), self.token_count.load(Ordering::SeqCst)),
            None => (0.0, 0),
        }
    }
}

#[cfg(test)]
mod repaint_width_tests {
    use super::*;

    /// The repaint is `\r`, blanks, `\r`, which only reaches the cursor's own
    /// row. A status longer than the window wraps, `\r` then lands on the last
    /// visual row, and the rows above it survive every tick — reported
    /// 2026-09-13 as stale "Thinking…" lines stacking up in a narrowed window.
    /// So the line must never exceed the window.
    #[test]
    fn the_status_line_never_exceeds_the_window() {
        let long = "생각하는 중… · ↓ 286 tokens · ✻ 1.2k thinking · ↑ 118.5k to prefill, 110k cached";
        for width in [20usize, 40, 60, 80, 120] {
            let clipped = clip_to_width(long, width);
            assert!(
                display_width(&clipped) <= width,
                "width {width}: clipped to {} columns",
                display_width(&clipped)
            );
        }
    }

    /// The blank run is sized in COLUMNS. Hangul and the arrow/star glyphs in
    /// this line are two columns each, so chars().count() wiped about half of
    /// what had been drawn and left the tail on screen.
    #[test]
    fn wide_glyphs_are_measured_in_columns_not_characters() {
        let s = "생각하는 중… ↓ ✻";
        assert!(
            display_width(s) > s.chars().count(),
            "this sample must contain wide glyphs for the test to mean anything"
        );
    }

    /// A window narrower than the floor still gets a sane width rather than 0,
    /// and the held-back column keeps us off the last cell.
    #[test]
    fn terminal_columns_stays_usable() {
        let c = terminal_columns();
        assert!(c >= 20, "floor holds: {c}");
    }
}

#[cfg(test)]
mod wait_note_display_tests {
    use super::*;

    /// The wait note is APPENDED after the closing paren. The token and
    /// prefill readouts inside it are what made a 500-second stall legible in
    /// the first place, so they must render byte-identically with or without
    /// a note.
    #[test]
    fn a_wait_note_does_not_disturb_the_token_or_prefill_readout() {
        let _guard = crate::utils::tokens::spinner_state_lock();
        std::env::set_var("CODEMASON_DEBUG", "1"); // the segment is -dbg only
        crate::utils::tokens::reset_prefill_state();
        crate::utils::tokens::set_pending_prefill_measured(73_100, 12_000);
        crate::utils::tokens::set_wait_note(None);
        let s = StreamingSpinner::new(true);
        s.start(Some("WebFetch"));
        let without = s.toolbar_text();

        crate::utils::tokens::set_wait_note(Some("fetching example.com".into()));
        let with = s.toolbar_text();

        assert!(with.starts_with(without.trim_end()), "note must only append");
        assert!(with.ends_with("fetching example.com"));
        assert!(without.contains("to prefill"), "prefill readout kept: {without}");

        crate::utils::tokens::set_wait_note(None);
        crate::utils::tokens::reset_prefill_state();
        std::env::remove_var("CODEMASON_DEBUG");
    }
}

#[cfg(test)]
mod pin_verb_tests {
    use super::*;

    /// pin_verb must name the interruption without discarding the turn. start()
    /// would have zeroed the token count and restarted the clock, which is why
    /// a mid-turn compaction cannot use it.
    #[test]
    fn pinning_renames_the_line_and_keeps_the_turn_handed_back_intact() {
        let _guard = crate::utils::tokens::spinner_state_lock();
        crate::utils::tokens::set_wait_note(None);
        let s = StreamingSpinner::new(true);
        s.start(None);
        let turn_verb = s.display_verb();
        s.update(500);
        assert_eq!(s.token_count.load(Ordering::SeqCst), 500);

        s.pin_verb(Some("Compacting"));
        assert_eq!(s.display_verb(), "Compacting", "the pin must win");
        assert_eq!(
            s.token_count.load(Ordering::SeqCst),
            500,
            "pinning must not reset the turn's counters"
        );

        // Idempotent: a second pin must not overwrite the saved label with
        // "Compacting", or unpinning would hand back the wrong name.
        s.pin_verb(Some("Compacting"));
        s.pin_verb(None);
        assert_eq!(s.display_verb(), turn_verb, "the turn's own label returns");
        assert_eq!(s.token_count.load(Ordering::SeqCst), 500);

        // Unpinning an unpinned spinner is a no-op, not a relabel.
        s.pin_verb(None);
        assert_eq!(s.display_verb(), turn_verb);
    }

    /// Without -dbg the readout appears only when the prompt is what the turn
    /// is waiting on. A warm request must leave the line exactly as clean as it
    /// was before the gate was opened.
    #[test]
    fn the_prefill_readout_is_quiet_when_warm_and_speaks_when_cold() {
        // The third test in this file to touch the same process-global prefill
        // state, and the one that was left out when the other two were given
        // the lock: cargo runs them on parallel threads, so an unguarded one
        // reads whatever a sibling's reset_prefill_state happened to leave. It
        // failed roughly once in five full runs and passed alone every time,
        // which is the shape of exactly this and nothing else.
        let _guard = crate::utils::tokens::spinner_state_lock();
        std::env::remove_var("CODEMASON_DEBUG");
        crate::utils::tokens::set_wait_note(None);
        let s = StreamingSpinner::new(true);
        s.start(Some("Thinking"));

        crate::utils::tokens::reset_prefill_state();
        crate::utils::tokens::set_pending_prefill_measured(2_000, 400_000);
        let warm = s.toolbar_text();
        assert!(!warm.contains("to prefill"), "warm request must stay quiet: {warm}");

        crate::utils::tokens::reset_prefill_state();
        crate::utils::tokens::set_pending_prefill_measured(242_795, 0);
        let cold = s.toolbar_text();
        assert!(cold.contains("to prefill"), "a cold prefill must say so: {cold}");

        crate::utils::tokens::reset_prefill_state();
    }
}
