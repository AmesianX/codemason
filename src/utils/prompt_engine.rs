//! Persistent-prompt terminal engine — the Rust substrate for the Python
//! port's prompt_toolkit PromptSession + patch_stdout combination.
//!
//! ONE engine runs for the whole interactive REPL as the single owner of
//! stdin. It paints a bottom block — optional spinner line, an input rule,
//! parked pending-message preview lines, then the '> ' prompt with the live
//! input buffer and an rprompt on the right — and repaints it on a
//! refresh_interval tick (0.1s) so the spinner animates. All model/tool
//! output flows through utils::term_out and is written ABOVE the block: the
//! engine erases its own lines, writes the text into scrollback, and repaints.
//!
//! Substrate divergence (documented): prompt_toolkit's Buffer/Application are
//! replaced by a hand-rolled line editor (emacs-ish basics, multiline via
//! Ctrl+J, history, bracketed paste, completion menu, auto-suggest ghost
//! text). The KEY BEHAVIOR CONTRACT is carried over from cli.py verbatim —
//! Enter submit vs mid-turn steering, double-Esc clear/rewind arming, single
//! y/n/a keypress permission mode, Ctrl+O pager hook, Ctrl+C hard-stop,
//! Ctrl+D EOF — via the EngineEvent callbacks the REPL (cli.rs) installs.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::io::Write;
use std::sync::Arc;
use unicode_width::UnicodeWidthChar;

use crate::utils::formatting::display_width;

/// One styled fragment of the bottom block ((style, text) — style is a raw
/// ANSI SGR prefix, "" for plain).
#[derive(Debug, Clone)]
pub struct Frag {
    pub style: String,
    pub text: String,
}

impl Frag {
    pub fn plain(text: impl Into<String>) -> Self {
        Frag { style: String::new(), text: text.into() }
    }
    pub fn styled(style: impl Into<String>, text: impl Into<String>) -> Self {
        Frag { style: style.into(), text: text.into() }
    }
}

/// What the engine reports to the REPL. Mirrors the prompt_toolkit key-binding
/// handlers in cli._make_prompt_kb — the engine handles EDITING keys itself
/// and surfaces the semantic events.
pub enum EngineEvent {
    /// Enter on a non-empty buffer (raw text, placeholders unexpanded).
    Submit(String),
    /// Esc pressed (engine already applied double-press arming semantics):
    /// a running turn should be interrupted.
    Interrupt,
    /// Double-Esc on an EMPTY buffer: open the conversation-rewind selector.
    RewindRequested,
    /// Single keypress answered a pending permission ask ('y' | 'n' | 'a').
    Permission(char),
    /// Ctrl+O: open the recent-tool-output pager.
    OpenOutputs,
    /// Ctrl+C: hard stop.
    HardStop,
    /// Ctrl+D on an empty buffer: EOF (quit).
    Eof,
    /// Ctrl+X Ctrl+K: the source's chord for the bulk stop of background
    /// agents. The engine only reports the chord — the two-press window and
    /// the kill itself belong to the REPL's gesture, which Ctrl+C shares.
    KillAgents,
}

pub type EventHandler = Arc<dyn Fn(EngineEvent) + Send + Sync>;

/// Callbacks that let the REPL own the CONTENT while the engine owns the
/// terminal: the message block above the input (spinner/rule/pending lines),
/// the right-prompt, and the mode flags the key dispatch depends on.
pub struct EngineHooks {
    /// Lines painted above the input line each repaint (already styled).
    pub message_lines: Arc<dyn Fn() -> Vec<Frag> + Send + Sync>,
    /// Right-aligned fragment on the input line ('(Esc again to clear)',
    /// context gauge).
    pub rprompt: Arc<dyn Fn() -> Vec<Frag> + Send + Sync>,
    /// The prompt marker for the input line ('> ', or the permission question).
    pub prompt_marker: Arc<dyn Fn() -> Vec<Frag> + Send + Sync>,
    /// True while a tool-permission ask is pending: input keys are swallowed,
    /// y/n/a answer, Esc denies.
    pub permission_pending: Arc<dyn Fn() -> bool + Send + Sync>,
    /// True while a turn is running (Esc = interrupt; Enter = steer).
    pub processing: Arc<dyn Fn() -> bool + Send + Sync>,
    /// True while a sub-UI (resume picker, pager) owns the terminal: the
    /// engine parks — erases its block and reads no input — until unpaused.
    pub paused: Arc<dyn Fn() -> bool + Send + Sync>,
    pub on_event: EventHandler,
    /// Large-paste hook: given the pasted text, return Some(placeholder) to
    /// insert instead (the caller stores the full text), or None to insert
    /// verbatim.
    pub on_paste: Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
    /// Completion hook: (text_before_cursor, explicit_tab) -> candidates as
    /// (replace_len_chars, insert, display, meta).
    pub completions: Arc<dyn Fn(&str, bool) -> Vec<(usize, String, String, String)> + Send + Sync>,
    /// Auto-suggest hook: full buffer text -> ghost suffix.
    pub auto_suggest: Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
}

/// The live input buffer + editing state.
#[derive(Default)]
struct EditState {
    text: String,
    cursor: usize, // byte offset, always on a char boundary
    history: Vec<String>,
    history_pos: Option<usize>,
    history_stash: String,
    /// double-press Esc arming (cli.py esc_state)
    armed_at: Option<std::time::Instant>,
    empty_armed_at: Option<std::time::Instant>,
    /// Ctrl+X was pressed and is waiting for the second half of a chord.
    /// prompt_toolkit does this for us in the Python ('c-x', 'c-k' is one
    /// binding); crossterm hands us one key at a time, so the prefix is state.
    chord_ctrl_x: bool,
    /// completion menu: (candidates, selected index). `menu_navigated` is
    /// false while the menu is merely SHOWING (live as-you-type completion) and
    /// true once the user has explicitly moved through it with Tab/arrows —
    /// which is what decides whether Enter accepts a completion or submits the
    /// line, exactly as prompt_toolkit does. Without the flag, live completion
    /// silently eats every Enter and nothing can be submitted.
    menu: Option<(Vec<(usize, String, String, String)>, usize)>,
    menu_navigated: bool,
}

impl EditState {
    fn insert(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }
    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = prev_boundary(&self.text, self.cursor);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
    }
    fn delete(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        let next = next_boundary(&self.text, self.cursor);
        self.text.replace_range(self.cursor..next, "");
    }
    fn left(&mut self) {
        self.cursor = prev_boundary(&self.text, self.cursor);
    }
    fn right(&mut self) {
        self.cursor = next_boundary(&self.text, self.cursor);
    }
    fn home(&mut self) {
        // to start of current line (multiline buffer)
        let upto = &self.text[..self.cursor];
        self.cursor = upto.rfind('\n').map(|i| i + 1).unwrap_or(0);
    }
    fn end(&mut self) {
        let rest = &self.text[self.cursor..];
        self.cursor += rest.find('\n').unwrap_or(rest.len());
    }
    fn word_left(&mut self) {
        let bytes = self.text.as_bytes();
        let mut i = self.cursor;
        while i > 0 && !bytes[prev_boundary(&self.text, i)].is_ascii_alphanumeric() {
            i = prev_boundary(&self.text, i);
        }
        while i > 0 && bytes[prev_boundary(&self.text, i)].is_ascii_alphanumeric() {
            i = prev_boundary(&self.text, i);
        }
        self.cursor = i;
    }
    fn word_right(&mut self) {
        let bytes = self.text.as_bytes();
        let n = self.text.len();
        let mut i = self.cursor;
        while i < n && !bytes[i].is_ascii_alphanumeric() {
            i = next_boundary(&self.text, i);
        }
        while i < n && bytes[i].is_ascii_alphanumeric() {
            i = next_boundary(&self.text, i);
        }
        self.cursor = i;
    }
    fn kill_to_end(&mut self) {
        let end = {
            let rest = &self.text[self.cursor..];
            self.cursor + rest.find('\n').unwrap_or(rest.len())
        };
        self.text.replace_range(self.cursor..end, "");
    }
    fn kill_to_start(&mut self) {
        let start = {
            let upto = &self.text[..self.cursor];
            upto.rfind('\n').map(|i| i + 1).unwrap_or(0)
        };
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }
    fn kill_word_back(&mut self) {
        let end = self.cursor;
        self.word_left();
        self.text.replace_range(self.cursor..end, "");
    }
    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_pos {
            None => {
                self.history_stash = self.text.clone();
                self.history_pos = Some(self.history.len() - 1);
            }
            Some(0) => return,
            Some(p) => self.history_pos = Some(p - 1),
        }
        if let Some(p) = self.history_pos {
            self.text = self.history[p].clone();
            self.cursor = self.text.len();
        }
    }
    fn history_down(&mut self) {
        match self.history_pos {
            None => {}
            Some(p) if p + 1 < self.history.len() => {
                self.history_pos = Some(p + 1);
                self.text = self.history[p + 1].clone();
                self.cursor = self.text.len();
            }
            Some(_) => {
                self.history_pos = None;
                self.text = std::mem::take(&mut self.history_stash);
                self.cursor = self.text.len();
            }
        }
    }
    fn append_history(&mut self) {
        if !self.text.is_empty() && self.history.last() != Some(&self.text) {
            self.history.push(self.text.clone());
        }
        self.history_pos = None;
    }
    fn reset(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.history_pos = None;
        self.menu = None;
        self.menu_navigated = false;
    }
}

fn prev_boundary(s: &str, i: usize) -> usize {
    if i == 0 {
        return 0;
    }
    let mut j = i - 1;
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}

fn next_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut j = i + 1;
    while j < s.len() && !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

/// Shared paint state — how many terminal lines the bottom block currently
/// occupies, so writes above the prompt know what to erase.
struct PaintState {
    painted_lines: usize,
    /// Physical row (0-based from the block's top) where the HARDWARE cursor
    /// is parked after a repaint. It is parked on the software cursor's cell —
    /// not the block's last row — because an IME anchors its composition
    /// preview at the terminal's cursor position: parked at the end, a Hangul
    /// syllable visibly composed to the RIGHT of the inverse-cell cursor and
    /// only jumped into place on commit. prompt_toolkit avoids this the same
    /// way (its real cursor sits at the logical cursor).
    park_row: usize,
    /// suppress painting entirely (paused for a sub-UI)
    hidden: bool,
    /// Last frame actually written, with the width it was rendered for.
    /// A repaint whose composed frame is IDENTICAL is skipped outright:
    /// prompt_toolkit only redraws on invalidation, and the 10Hz tick
    /// rewriting an unchanged block had a real casualty — a web terminal
    /// shows the IME's Hangul composition preview at the cursor, and every
    /// gratuitous rewrite wiped it, so long-input typing showed each
    /// syllable only after the NEXT keystroke committed it.  None = not on
    /// screen (erased/hidden): always draw.
    last_frame: Option<(usize, Vec<String>)>,
}

pub struct PromptEngine {
    hooks: EngineHooks,
    edit: Mutex<EditState>,
    paint: Mutex<PaintState>,
    /// double-press window (cli.py: 0.7s)
    double_press_window: std::time::Duration,
    /// Buffer text externally set (rewind selector drops the picked prompt
    /// back into the input box).
    pending_set_text: Mutex<Option<String>>,
    running: std::sync::atomic::AtomicBool,
    /// Input-park handshake (Python pause_prompt/_park parity, cli.py:1908):
    /// `suspended` is the request, `parked` the loop's confirmation that it has
    /// RELEASED the input token, `wake` stirs the loop for suspend/resume/stop.
    suspended: std::sync::atomic::AtomicBool,
    parked: std::sync::atomic::AtomicBool,
    wake: tokio::sync::Notify,
    parked_note: tokio::sync::Notify,
}

/// Terminal writer — reachable ONLY inside this closure, and ONLY with the
/// paint guard already in hand.
///
/// WHY IT IS A CLOSURE. The engine has two locks that every painting path
/// touches: `paint` and std's own stdout lock, always taken paint -> stdout.
/// `write_above` and the Ctrl+L arm used to bind the StdoutLock to a local and
/// then release `paint` (or `edit`) while that local was still alive, so the
/// rest of the body — including `repaint()`, which reaches for `paint` — ran
/// holding stdout. That is the one path where the order was stdout -> paint.
/// Under -rt the background RemoteProgress task repaints while the turn task
/// prints a tool result, and the two orders meet: one thread holds stdout and
/// waits for paint, the other holds paint and waits for stdout. Neither
/// parking_lot nor std detects it, so every runtime thread parks and the whole
/// process hangs with no output and no prompt — no way out but a kill
/// (observed 2026-09-29 under `cm2 -rt -dsp`: two threads wedged exactly so).
/// Handing the writer to a closure means the guard cannot outlive the call, so
/// the inverted order is no longer expressible.
fn with_out<R>(
    paint: &mut PaintState,
    f: impl FnOnce(&mut std::io::StdoutLock<'static>, &mut PaintState) -> R,
) -> R {
    let mut o = std::io::stdout().lock();
    f(&mut o, paint)
}

static ENGINE: Lazy<Mutex<Option<Arc<PromptEngine>>>> = Lazy::new(|| Mutex::new(None));

struct EngineSink(Arc<PromptEngine>);

impl crate::utils::term_out::OutputSink for EngineSink {
    fn write_above_prompt(&self, text: &str) {
        self.0.write_above(text);
    }
}

impl PromptEngine {
    pub fn new(hooks: EngineHooks) -> Arc<Self> {
        Arc::new(PromptEngine {
            hooks,
            edit: Mutex::new(EditState::default()),
            paint: Mutex::new(PaintState {
                painted_lines: 0,
                park_row: 0,
                hidden: false,
                last_frame: None,
            }),
            double_press_window: std::time::Duration::from_millis(700),
            pending_set_text: Mutex::new(None),
            running: std::sync::atomic::AtomicBool::new(false),
            suspended: std::sync::atomic::AtomicBool::new(false),
            parked: std::sync::atomic::AtomicBool::new(false),
            wake: tokio::sync::Notify::new(),
            parked_note: tokio::sync::Notify::new(),
        })
    }

    /// Install as the global term_out sink and enter raw mode. Runs the input +
    /// repaint loops until `stop()`.
    pub async fn run(self: &Arc<Self>) {
        use crossterm::event::{Event, KeyEvent, KeyEventKind};
        use futures::StreamExt;

        self.running.store(true, std::sync::atomic::Ordering::SeqCst);
        *ENGINE.lock() = Some(self.clone());
        crate::utils::term_out::set_sink(Some(Arc::new(EngineSink(self.clone()))));
        if crate::utils::tty_compat::enable_raw_mode().is_err() {
            // Not a tty: the caller should have used the plain loop; degrade to
            // doing nothing (events never arrive).
        }
        // Discard anything typed before the prompt existed.  Until raw mode is
        // on the tty is still in canonical mode: the kernel echoes keystrokes
        // and buffers them, and they would all be delivered the instant the
        // event stream opens — landing in a buffer the user never saw being
        // filled (and, mid-startup, printed across the banner).  Startup can
        // take seconds (backend probe, remote channel), so this is easy to hit.
        crate::utils::tty_compat::flush_input();
        // Bracketed paste so a paste arrives as one Event::Paste.
        {
            let mut paint = self.paint.lock();
            with_out(&mut paint, |o, _| {
                let _ = crossterm::execute!(o, crossterm::event::EnableBracketedPaste);
            });
        }

        let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        self.repaint();

        // Outer loop = one input-ownership session per iteration (the Python
        // input loop's `while True: prompt_async()` with _park() between
        // Applications, cli.py:1926).  The input token is held only inside the
        // inner loop; a suspend request drops it, confirms via `parked`, and
        // waits to be resumed — so a sub-UI's acquire() proceeds the moment we
        // park, never against us.
        'sessions: while self.running.load(std::sync::atomic::Ordering::SeqCst) {
            if self.suspended.load(std::sync::atomic::Ordering::SeqCst) {
                // Park: confirm the token is free, then sleep till resume/stop.
                self.parked.store(true, std::sync::atomic::Ordering::SeqCst);
                self.parked_note.notify_waiters();
                loop {
                    let woken = self.wake.notified(); // register before check
                    if !self.suspended.load(std::sync::atomic::Ordering::SeqCst)
                        || !self.running.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        break;
                    }
                    woken.await;
                }
                self.parked.store(false, std::sync::atomic::Ordering::SeqCst);
                continue 'sessions;
            }

            // Each ownership session begins by putting the terminal back into
            // raw mode: a sub-UI that just released input (picker, pager)
            // restored the cooked state on its way out, exactly as a
            // prompt_toolkit Application does — and prompt_toolkit's resident
            // session re-enters raw on every prompt_async() for the same
            // reason.  Without this, the first /resume selection left the tty
            // canonical: no echo, no events until Enter — a dead prompt.
            let _ = crate::utils::tty_compat::enable_raw_mode();
            let mut events = crate::utils::input_events::acquire().await;
            loop {
            if !self.running.load(std::sync::atomic::Ordering::SeqCst)
                || self.suspended.load(std::sync::atomic::Ordering::SeqCst)
            {
                continue 'sessions; // drops `events` = releases the input token
            }
            tokio::select! {
                _ = self.wake.notified() => {
                    // suspend()/stop() poked us: loop back to the checks above.
                }
                _ = tick.tick() => {
                    // refresh_interval repaint (spinner frame, rprompt hints).
                    if let Some(t) = self.pending_set_text.lock().take() {
                        let mut e = self.edit.lock();
                        e.cursor = t.len();
                        e.text = t;
                    }
                    self.repaint();
                }
                ev = events.next() => {
                    let Some(ev) = ev else { continue };
                    if (self.hooks.paused)() {
                        continue; // pause raced an in-flight event; drop it
                    }
                    match ev {
                        Event::Paste(data) => {
                            // Normalise CR first (Windows/RDP clipboards carry
                            // CRLF; a literal ^M would weld lines together).
                            let data = data.replace("\r\n", "\n").replace('\r', "\n");
                            let insert = (self.hooks.on_paste)(&data).unwrap_or(data);
                            self.edit.lock().insert(&insert);
                            self.repaint();
                        }
                        Event::Key(KeyEvent { code, modifiers, kind, .. }) => {
                            if kind == KeyEventKind::Release {
                                continue;
                            }
                            self.on_key(code, modifiers);
                        }
                        Event::Resize(_, _) => {
                            // The terminal reflowed our old cells; the cached
                            // frame no longer reflects the screen.
                            //
                            // AND THAT IS ALL WE DO. Read this before adding a
                            // re-measure here — it was tried twice (2026-09-13,
                            // 816fb8e..92a442c and the reverts between them) and
                            // both attempts were withdrawn.
                            //
                            // The pull is real: erase_block_locked walks up
                            // `park_row` rows, and that count was measured at the
                            // width the frame was DRAWN at. Narrow the window and
                            // a one-row input rule becomes two, the erase still
                            // walks up one, and the top of the old block survives
                            // above the prompt. Re-measuring the stored frame at
                            // the new width fixes exactly that.
                            //
                            // What it costs: the re-measure assumes the terminal
                            // reflowed, and only NARROWING reflows. Widening does
                            // not un-wrap rows that are already on screen, so the
                            // re-measure reports fewer rows than the screen holds,
                            // the erase walks up from above the block, and \x1b[0J
                            // takes real scrollback with it. Measured on Windows
                            // Terminal + WSL2: dragging the window wider emptied
                            // the display. Gating it to narrowing only (92a442c)
                            // kept the engine acting on a guess about another
                            // program's reflow behaviour, which is what it had
                            // just been burned by.
                            //
                            // Why it was dropped rather than gated further: the
                            // terminal changed. This was chased on Windows
                            // Terminal + WSL2, and the session moved to wsltty,
                            // where it is not the problem it was. So the whole
                            // thing went out the same day — the fix was for a
                            // terminal that is no longer in use, and carrying
                            // width arithmetic that can wipe a screen to cover a
                            // stale row nobody is seeing is the wrong trade.
                            //
                            // If it comes back on some terminal: a stale row is
                            // cosmetic and Ctrl+L clears it, a wiped screen is
                            // lost work, so do not re-measure on a guess. Ask the
                            // terminal where the cursor actually IS after the
                            // resize (DSR \x1b[6n) and derive the row count from
                            // the answer.
                            self.paint.lock().last_frame = None;
                            self.repaint();
                        }
                        _ => {}
                    }
                }
            }
            }
        }

        // Teardown: erase the block, restore the terminal.
        self.erase_block();
        {
            let mut paint = self.paint.lock();
            with_out(&mut paint, |o, _| {
                let _ = write!(o, "\x1b[?25h"); // restore the hardware cursor
                let _ = crossterm::execute!(o, crossterm::event::DisableBracketedPaste);
            });
        }
        crate::utils::tty_compat::disable_raw_mode();
        crate::utils::tty_compat::restore_echo();
        crate::utils::term_out::set_sink(None);
        *ENGINE.lock() = None;
    }

    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_waiters(); // a parked or selecting loop must see it
    }

    /// Park the engine: release terminal input for a sub-UI and wait until the
    /// run loop CONFIRMS it has (Python pause_prompt parity, cli.py:1908-1915 —
    /// app.exit(_PAUSE) then `await self._idle_evt.wait()`).  After this
    /// returns, `input_events::acquire()` cannot be contended by the engine.
    pub async fn suspend(&self) {
        if !self.running.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        self.suspended.store(true, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_waiters();
        loop {
            // Register the waiter BEFORE re-checking, or a notify landing
            // between check and await is lost and this waits forever.
            let parked = self.parked_note.notified();
            if self.parked.load(std::sync::atomic::Ordering::SeqCst)
                || !self.running.load(std::sync::atomic::Ordering::SeqCst)
            {
                break;
            }
            parked.await;
        }
    }

    /// Undo `suspend`: the run loop re-acquires input and repaints (Python
    /// resume_prompt parity — `_resume_evt.set()`, cli.py:1917).
    pub fn unsuspend(&self) {
        self.suspended.store(false, std::sync::atomic::Ordering::SeqCst);
        self.wake.notify_waiters();
    }

    /// Externally set the input buffer (rewind selector).
    pub fn set_buffer_text(&self, text: &str) {
        *self.pending_set_text.lock() = Some(text.to_string());
    }

    pub fn buffer_text(&self) -> String {
        self.edit.lock().text.clone()
    }

    fn emit_event(&self, ev: EngineEvent) {
        (self.hooks.on_event)(ev);
    }

    fn on_key(&self, code: crossterm::event::KeyCode, m: crossterm::event::KeyModifiers) {
        use crossterm::event::{KeyCode, KeyModifiers};
        let perm = (self.hooks.permission_pending)();

        if perm {
            // Permission-ask mode: answer with a SINGLE keypress and accept
            // ONLY y/n/a (+Esc=deny). Everything else — including Enter and
            // ordinary typing — is swallowed, so text the user was composing
            // for the NEXT message can't leak into the ask.
            match code {
                KeyCode::Char(c @ ('y' | 'Y' | 'n' | 'N' | 'a' | 'A')) => {
                    self.emit_event(EngineEvent::Permission(c.to_ascii_lowercase()));
                }
                KeyCode::Esc => self.emit_event(EngineEvent::Interrupt),
                KeyCode::Char('c') if m.contains(KeyModifiers::CONTROL) => {
                    self.emit_event(EngineEvent::HardStop)
                }
                _ => {}
            }
            return;
        }

        let processing = (self.hooks.processing)();
        let mut e = self.edit.lock();

        // Ctrl+X is a chord PREFIX (the source binds chat:killAgents to
        // ctrl+x ctrl+k). It is consumed here and the next key decides:
        // Ctrl+K fires the gesture, anything else drops the prefix and is
        // handled as itself — so a mistyped chord costs nothing but the
        // Ctrl+X. Checked before everything else, including the completion
        // menu, because a prefix that something else can eat is not a prefix.
        if e.chord_ctrl_x {
            e.chord_ctrl_x = false;
            if matches!(code, KeyCode::Char('k')) && m.contains(KeyModifiers::CONTROL) {
                drop(e);
                self.emit_event(EngineEvent::KillAgents);
                return;
            }
        } else if matches!(code, KeyCode::Char('x')) && m.contains(KeyModifiers::CONTROL) {
            e.chord_ctrl_x = true;
            return;
        }

        // Completion menu navigation swallows Up/Down/Tab/Enter first.
        if let Some((cands, sel)) = e.menu.clone() {
            match code {
                KeyCode::Down | KeyCode::Tab => {
                    // First Tab/Down selects the highlighted row rather than
                    // skipping past it (prompt_toolkit's first-press behavior).
                    let ns = if e.menu_navigated { (sel + 1) % cands.len() } else { sel };
                    e.menu_navigated = true;
                    e.menu = Some((cands, ns));
                    drop(e);
                    self.repaint();
                    return;
                }
                KeyCode::Up => {
                    let ns = (sel + cands.len() - 1) % cands.len();
                    e.menu_navigated = true;
                    e.menu = Some((cands, ns));
                    drop(e);
                    self.repaint();
                    return;
                }
                // Enter accepts the completion ONLY when the user navigated the
                // menu. A menu that is merely showing (live completion) must
                // let Enter submit the line.
                KeyCode::Enter if e.menu_navigated => {
                    let (replace, insert, _d, _m) = cands[sel].clone();
                    // replace `replace` chars before cursor with `insert`
                    let mut start = e.cursor;
                    for _ in 0..replace {
                        start = prev_boundary(&e.text, start);
                    }
                    let cur = e.cursor;
                    e.text.replace_range(start..cur, &insert);
                    e.cursor = start + insert.len();
                    e.menu = None;
                    e.menu_navigated = false;
                    drop(e);
                    self.repaint();
                    return;
                }
                KeyCode::Esc => {
                    e.menu = None;
                    e.menu_navigated = false;
                    drop(e);
                    self.repaint();
                    return;
                }
                _ => {
                    // any other key (Enter included, when not navigated) closes
                    // the menu and falls through to the normal handler
                    e.menu = None;
                    e.menu_navigated = false;
                }
            }
        }

        match code {
            KeyCode::Enter if m.contains(KeyModifiers::CONTROL) => {
                // Ctrl+Enter/Ctrl+J: newline
                e.insert("\n");
            }
            KeyCode::Char('j') if m.contains(KeyModifiers::CONTROL) => {
                e.insert("\n");
            }
            KeyCode::Enter => {
                let text = e.text.clone();
                if text.trim().is_empty() {
                    // Empty/whitespace idle Enter: TRUE no-op (drop whitespace).
                    e.reset();
                    drop(e);
                    self.repaint();
                    return;
                }
                e.append_history();
                e.reset();
                drop(e);
                self.repaint();
                self.emit_event(EngineEvent::Submit(text));
                return;
            }
            KeyCode::Esc => {
                if processing {
                    // Esc during a running turn fires IMMEDIATELY (eager).
                    drop(e);
                    self.emit_event(EngineEvent::Interrupt);
                    return;
                }
                let now = std::time::Instant::now();
                if e.text.is_empty() {
                    // Empty input: double-Esc opens the conversation-rewind
                    // selector. First Esc arms + hints via rprompt.
                    e.armed_at = None;
                    let fire = matches!(e.empty_armed_at, Some(t) if now.duration_since(t) < self.double_press_window);
                    if fire {
                        e.empty_armed_at = None;
                        drop(e);
                        self.emit_event(EngineEvent::RewindRequested);
                        return;
                    }
                    e.empty_armed_at = Some(now);
                } else {
                    // Non-empty: double-Esc clears (saved to history first so
                    // Up-arrow recovers it).
                    e.empty_armed_at = None;
                    let fire = matches!(e.armed_at, Some(t) if now.duration_since(t) < self.double_press_window);
                    if fire {
                        e.append_history();
                        e.reset();
                        e.armed_at = None;
                    } else {
                        e.armed_at = Some(now);
                    }
                }
            }
            KeyCode::Char('c') if m.contains(KeyModifiers::CONTROL) => {
                drop(e);
                self.emit_event(EngineEvent::HardStop);
                return;
            }
            KeyCode::Char('d') if m.contains(KeyModifiers::CONTROL) => {
                if e.text.is_empty() {
                    drop(e);
                    self.emit_event(EngineEvent::Eof);
                    return;
                }
                e.delete();
            }
            KeyCode::Char('o') if m.contains(KeyModifiers::CONTROL) => {
                drop(e);
                self.emit_event(EngineEvent::OpenOutputs);
                return;
            }
            KeyCode::Char('a') if m.contains(KeyModifiers::CONTROL) => e.home(),
            KeyCode::Char('e') if m.contains(KeyModifiers::CONTROL) => e.end(),
            KeyCode::Char('k') if m.contains(KeyModifiers::CONTROL) => e.kill_to_end(),
            KeyCode::Char('u') if m.contains(KeyModifiers::CONTROL) => e.kill_to_start(),
            KeyCode::Char('w') if m.contains(KeyModifiers::CONTROL) => e.kill_word_back(),
            KeyCode::Char('l') if m.contains(KeyModifiers::CONTROL) => {
                // clear screen, keep buffer.
                //
                // The clear itself now happens INSIDE the paint lock. The
                // earlier shape took stdout first and was still holding it
                // across `self.paint.lock()` and `repaint()` below — the same
                // stdout -> paint inversion `with_out` now rules out. Fixing
                // the edit -> paint order here (below) was necessary but not
                // sufficient; there were two inversions in this one arm.
                //
                // Release `edit` BEFORE taking `paint`. repaint() locks paint
                // then edit, so holding edit here while reaching for paint is a
                // lock-order inversion: the engine task on Ctrl+L and a turn
                // task printing a tool result through write_above/repaint could
                // each hold what the other wants, and parking_lot has no
                // deadlock detection — the TUI freezes with no way out but a
                // kill. Every other arm in this match already drops `e` before
                // doing anything outside the edit lock; this one did not.
                drop(e);
                {
                    let mut paint = self.paint.lock();
                    with_out(&mut paint, |o, paint| {
                        let _ = crossterm::execute!(
                            o,
                            crossterm::terminal::Clear(crossterm::terminal::ClearType::All),
                            crossterm::cursor::MoveTo(0, 0)
                        );
                        paint.painted_lines = 0;
                        // The screen no longer holds the frame we cached, so
                        // the skip in repaint() must not treat it as still
                        // drawn. Without this Ctrl+L cleared the terminal and
                        // the repaint right below it decided the frame was
                        // unchanged and drew nothing: a blank screen until the
                        // next keystroke happened to change the frame.
                        paint.last_frame = None;
                    });
                }
                self.repaint();
                return;
            }
            KeyCode::Tab => {
                // explicit completion request
                let before = e.text[..e.cursor].to_string();
                let cands = (self.hooks.completions)(&before, true);
                if cands.len() == 1 {
                    let (replace, insert, _d, _m) = cands[0].clone();
                    let mut start = e.cursor;
                    for _ in 0..replace {
                        start = prev_boundary(&e.text, start);
                    }
                    let cur = e.cursor;
                    e.text.replace_range(start..cur, &insert);
                    e.cursor = start + insert.len();
                }
                // Multi-candidate menu disabled with the rest of autocomplete;
                // an ambiguous Tab does nothing.
            }
            KeyCode::Left if m.contains(KeyModifiers::CONTROL) => e.word_left(),
            KeyCode::Right if m.contains(KeyModifiers::CONTROL) => e.word_right(),
            KeyCode::Left => e.left(),
            KeyCode::Right => e.right(), // ghost accept removed with autocomplete
            KeyCode::Home => e.home(),
            KeyCode::End => e.end(),
            KeyCode::Up => e.history_up(),
            KeyCode::Down => e.history_down(),
            KeyCode::Backspace => e.backspace(),
            KeyCode::Delete => e.delete(),
            KeyCode::Char(c) => {
                if m.contains(KeyModifiers::CONTROL) {
                    // unhandled control chord: swallow
                } else {
                    let mut buf = [0u8; 4];
                    e.insert(c.encode_utf8(&mut buf));
                    // Live as-you-type completion is DISABLED for now: the
                    // pop-up menu growing/shrinking under the prompt is what
                    // kept desyncing the repaint on narrow terminals.  Tab
                    // still completes a unique candidate.  Re-enable by
                    // restoring the complete_while_typing block here (see
                    // git history) once the renderer can survive it.
                }
            }
            _ => {}
        }
        drop(e);
        self.repaint();
    }

    /// Write text into scrollback above the live prompt block (patch_stdout).
    pub fn write_above(&self, text: &str) {
        {
            let mut paint = self.paint.lock();
            with_out(&mut paint, |o, paint| {
                Self::erase_block_locked(o, paint);
                // In raw mode, '\n' does not return the carriage — write CRLF.
                let normalized = text.replace('\n', "\r\n");
                let _ = o.write_all(normalized.as_bytes());
                let _ = o.flush();
            });
        }
        // BOTH guards are released before repaint(). repaint() takes `paint`
        // again, and arriving there with stdout still in hand is precisely the
        // deadlock this shape exists to prevent — the old code dropped `paint`
        // here but let the stdout guard live to the end of the function.
        self.repaint();
    }

    fn erase_block(&self) {
        let mut paint = self.paint.lock();
        with_out(&mut paint, |o, paint| {
            Self::erase_block_locked(o, paint);
            let _ = o.flush();
        });
    }

    fn erase_block_locked(o: &mut impl Write, paint: &mut PaintState) {
        if paint.painted_lines > 0 {
            // the cursor is parked on `park_row`; move to the block's top and
            // clear downward.
            let up = paint.park_row.min(paint.painted_lines - 1);
            if up > 0 {
                let _ = write!(o, "\x1b[{}A", up);
            }
            let _ = write!(o, "\r\x1b[0J");
            paint.painted_lines = 0;
            paint.park_row = 0;
            paint.last_frame = None;
        }
    }

    /// Temporarily hide the block (sub-UI takes the terminal); unhide repaints.
    pub fn set_hidden(&self, hidden: bool) {
        {
            let mut paint = self.paint.lock();
            if hidden {
                with_out(&mut paint, |o, paint| {
                    Self::erase_block_locked(o, paint);
                    let _ = o.flush();
                });
            }
            paint.hidden = hidden;
        }
        if !hidden {
            self.repaint();
        }
    }

    /// Paint the bottom block: message lines + prompt line (with input buffer,
    /// ghost suggestion, rprompt) + completion menu.
    ///
    /// The HARDWARE cursor is hidden while the engine runs and stays parked on
    /// the block's LAST painted line (which keeps the erase math trivial); the
    /// visible cursor is drawn as an inverse-video cell inside the input line —
    /// the same software-cursor approach prompt_toolkit uses when it owns the
    /// screen region.
    pub fn repaint(&self) {
        let mut paint = self.paint.lock();
        if paint.hidden {
            return;
        }
        let cols = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(80).max(20);

        let e = self.edit.lock();
        let mut lines: Vec<String> = Vec::new();

        for frag_line in group_lines((self.hooks.message_lines)()) {
            lines.push(frag_line);
        }

        // Input line(s): marker + buffer (multiline splits) + ghost + rprompt.
        let marker: String = (self.hooks.prompt_marker)()
            .iter()
            .map(|f| format!("{}{}{}", f.style, f.text, if f.style.is_empty() { "" } else { "\x1b[0m" }))
            .collect();
        let marker_width: usize = (self.hooks.prompt_marker)().iter().map(|f| display_width(&f.text)).sum();

        // Ghost auto-suggest disabled with the rest of autocomplete (its
        // unbudgeted width is what first pushed the input line past the right
        // edge).  The rendering below keeps its ghost branches, dormant.
        let ghost = String::new();
        let rp: String = {
            let frags = (self.hooks.rprompt)();
            frags
                .iter()
                .map(|f| format!("{}{}{}", f.style, f.text, if f.style.is_empty() { "" } else { "\x1b[0m" }))
                .collect()
        };
        let rp_width: usize = (self.hooks.rprompt)().iter().map(|f| display_width(&f.text)).sum();

        // Cursor position within the buffer, as (line index, byte col range).
        let upto = &e.text[..e.cursor];
        let cur_line_idx = upto.matches('\n').count();

        // Render each buffer line, drawing the software cursor as an inverse
        // cell over the character at the cursor (or a trailing space).
        // `cursor_cell` records where that cell lands — (index into `lines`,
        // cells before it on that line) — so the HARDWARE cursor can be parked
        // on the same cell after painting (IME composition anchors there).
        let mut cursor_cell: Option<(usize, String)> = None;
        let buffer_lines: Vec<&str> = e.text.split('\n').collect();
        let mut consumed = 0usize; // byte offset of the current line start
        for (i, bl) in buffer_lines.iter().enumerate() {
            let mut line = String::new();
            let prefix_width;
            if i == 0 {
                line.push_str(&marker);
                prefix_width = marker_width;
            } else {
                line.push_str("  ");
                prefix_width = 2;
            }
            if i == cur_line_idx {
                let col = e.cursor - consumed; // byte offset in this line
                let (before, after) = bl.split_at(col.min(bl.len()));
                // Record the text that PRECEDES the cursor on this rendered
                // line; its row/column come from layout_line, so a wide glyph
                // pushed to the next row is accounted for exactly as the
                // terminal does it.
                cursor_cell = Some((lines.len(), format!("{}{}", " ".repeat(prefix_width), before)));
                line.push_str(before);
                let mut chars = after.chars();
                match chars.next() {
                    Some(c) => {
                        line.push_str(&format!("\x1b[7m{c}\x1b[27m"));
                        line.push_str(chars.as_str());
                        if i == buffer_lines.len() - 1 && !ghost.is_empty() {
                            line.push_str(&format!("\x1b[38;5;240m{ghost}\x1b[0m"));
                        }
                    }
                    None => {
                        // cursor at end of line: inverse the ghost's first char
                        // or a blank cell
                        let mut g = ghost.chars();
                        match (i == buffer_lines.len() - 1).then(|| g.next()).flatten() {
                            Some(gc) => {
                                line.push_str(&format!("\x1b[7m\x1b[38;5;240m{gc}\x1b[27m{}\x1b[0m", g.as_str()));
                            }
                            None => line.push_str("\x1b[7m \x1b[27m"),
                        }
                    }
                }
            } else {
                line.push_str(bl);
                if i == buffer_lines.len() - 1 && !ghost.is_empty() {
                    line.push_str(&format!("\x1b[38;5;240m{ghost}\x1b[0m"));
                }
            }
            // rprompt on the FIRST input line, right-aligned, only if room
            if i == 0 && rp_width > 0 {
                // The ghost suggestion is drawn on this line too, so it has to
                // be in the budget: leaving it out padded "> /cl" as if it were
                // 5 cells when the ghost "ear" made it 8, pushing the line past
                // the right edge.  It then wrapped onto a second physical row
                // while painted_lines counted one, so erase_block walked back a
                // row short and left a stale row per keystroke.
                // With a ghost the software cursor inverts its first cell and
                // costs no extra column; without one it adds a blank cell.
                let ghost_w = if i == buffer_lines.len() - 1 { display_width(&ghost) } else { 0 };
                let used = prefix_width + display_width(bl) + if ghost_w > 0 { ghost_w } else { 1 };
                if used + rp_width + 2 < cols {
                    let pad = cols - used - rp_width - 1;
                    line.push_str(&" ".repeat(pad));
                    line.push_str(&rp);
                }
            }
            lines.push(line);
            consumed += bl.len() + 1; // +1 for the '\n'
        }

        // completion menu below the input (up to 8 rows)
        if let Some((cands, sel)) = &e.menu {
            for (idx, (_r, _ins, disp, meta)) in cands.iter().take(8).enumerate() {
                let mark = if idx == *sel { "\x1b[7m" } else { "" };
                let meta_part = if meta.is_empty() {
                    String::new()
                } else {
                    format!("  \x1b[38;5;245m{meta}\x1b[0m")
                };
                // Clip to the window, like the Python menu's
                // wrap_lines=False Window.  A long entry (skill commands carry
                // a full sentence of meta) would otherwise wrap onto a second
                // PHYSICAL row while painted_lines counts one LOGICAL line, so
                // erase_block walks back a row short and leaves a stale row
                // behind on every repaint — the list then marches down screen.
                lines.push(crate::utils::formatting::clip_to_width(
                    &format!("  {mark}{disp}\x1b[0m{meta_part}"),
                    cols,
                ));
            }
        }
        drop(e);

        // Unchanged frame at the same width: nothing to do.  Skipping is what
        // lets an IME's composition preview live at the cursor between
        // keystrokes — see the last_frame field note.
        if paint
            .last_frame
            .as_ref()
            .is_some_and(|(w, f)| *w == cols && *f == lines)
        {
            return;
        }

        // ONE frame: composed into a buffer, then handed to the terminal in as
        // few writes as possible, wrapped in synchronized-output markers.
        //
        // WHY. out() is std's Stdout, which is a LineWriter: it flushes every
        // time it is handed a newline. This block used to be written line by
        // line with `\r\n` between the lines, so a single repaint reached the
        // terminal as four separately-flushed states — erased, spinner only,
        // spinner+rule, then the whole block. At the 0.1s refresh tick that is
        // ten times a second, and because the spinner line carries a tenth-second
        // timer the frame is never identical while a turn runs, so the
        // last_frame skip above never fires and the rule and the prompt line —
        // which did not change at all — were torn down and rebuilt with it.
        // Reported as the input rule and prompt visibly shaking during a reply.
        //
        // Not a Windows bug, though that is where it was seen: nothing on this
        // path is cfg'd, and the same four flushes happen on Linux. There a pty
        // is a plain byte pipe and the emulator paints on its own ~60Hz tick, so
        // four flushes a millisecond apart are usually coalesced into one
        // presented frame — luck, not immunity, and it runs out over ssh or on a
        // loaded machine. Windows has ConPTY between us and the terminal, which
        // parses the stream into its own buffer and re-emits on a cadence of its
        // own that our flush boundaries land squarely on.
        //
        // Buffering does not fully close it by itself: LineWriter::write_all
        // still writes through the LAST newline immediately and leaves the tail
        // buffered, and our tail is the prompt line. Hence DECSET 2026
        // (synchronized output): the terminal holds the display until the
        // closing 2026l, so however many writes the frame takes, it appears at
        // once. Terminals that do not know the mode ignore it, and those that do
        // also time it out (~150ms), so an unpaired open cannot wedge a display.
        //
        // The open and the close go in the SAME buffer on purpose. Written
        // separately, any early return added between them later would leave 2026
        // open and freeze the terminal; here that is not expressible.
        let mut buf: Vec<u8> = Vec::with_capacity(512);
        let _ = write!(buf, "\x1b[?2026h");
        Self::erase_block_locked(&mut buf, &mut paint);
        let _ = write!(buf, "\x1b[?25l"); // hide hardware cursor (software cursor drawn)
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                let _ = write!(buf, "\r\n");
            }
            let _ = write!(buf, "\r\x1b[2K{line}");
        }

        // Park the (hidden) HARDWARE cursor on the software cursor's cell.
        // Terminals anchor an IME's composition preview at the cursor
        // POSITION regardless of DECTCEM visibility; parked at the block's
        // end (as this used to), a Hangul syllable visibly composed to the
        // right of the inverse-cell cursor. Wrap-aware: rows/cols are derived
        // with the same terminal-accurate measure as wrapped_rows.
        let total_rows = lines.iter().map(|l| wrapped_rows(l, cols)).sum::<usize>().max(1);
        let park_row = match cursor_cell {
            Some((line_idx, ref before)) => {
                let rows_above: usize =
                    lines[..line_idx].iter().map(|l| wrapped_rows(l, cols)).sum();
                // Same layout the terminal just performed: rows consumed by the
                // text before the cursor, and the column it ends on.
                let (rows_used, col) = layout_line(before, cols);
                let row = rows_above + rows_used - 1;
                let up = (total_rows - 1).saturating_sub(row);
                if up > 0 {
                    let _ = write!(buf, "\x1b[{up}A");
                }
                let _ = write!(buf, "\r");
                if col > 0 {
                    let _ = write!(buf, "\x1b[{col}C");
                }
                row.min(total_rows - 1)
            }
            None => total_rows - 1, // no input line rendered: park at the end
        };
        let _ = write!(buf, "\x1b[?2026l");

        with_out(&mut paint, |o, paint| {
            let _ = o.write_all(&buf);
            let _ = o.flush();
            paint.park_row = park_row;
            paint.painted_lines = total_rows;
            paint.last_frame = Some((cols, lines));
        });
    }
}

/// Physical rows a rendered line occupies at `cols`.
///
/// `erase_block` walks back by ROW (ESC[nA), so the block's height has to be
/// counted in rows, not in logical lines.  A line wider than the window is
/// wrapped by the terminal onto extra rows; counting it as one row leaves the
/// erase a row short and a stale row on screen per repaint, and the block then
/// marches down the screen.  Menu rows are clipped so they never wrap, but the
/// INPUT line cannot be clipped — a long prompt has to stay fully visible — so
/// this is what keeps a wrapped input line honest.  SGR sequences occupy no
/// cells and are stripped before measuring.
/// Lay a rendered line out the way the terminal does, returning
/// (rows occupied, column the cursor ends on).
///
/// This SIMULATES wrapping instead of dividing total width by `cols`, because
/// a double-width glyph is never split: when one would straddle the right
/// edge the terminal moves it WHOLE to the next row and leaves the last cell
/// blank.  Division misses those blanks, and with Hangul (every glyph two
/// cells) the error compounds — a long Korean line reports fewer rows than it
/// occupies, so the erase leaves stale rows and the cursor parks off-target,
/// which is why short input was fine and multi-row input was not.
///
/// Widths come from unicode-width, NOT display_width(): the latter widens
/// East-Asian *ambiguous* characters under a CJK locale while terminals draw
/// them narrow (the input rule is U+2500 repeated cols-1 times — counted wide
/// it would measure ~2*cols and the erase would walk into live scrollback).
fn layout_line(line: &str, cols: usize) -> (usize, usize) {
    use unicode_width::UnicodeWidthChar;
    static SGR: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap());
    let plain = SGR.replace_all(line, "");
    let mut rows = 1usize;
    let mut col = 0usize;
    for ch in plain.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w == 0 {
            continue; // combining marks and the like occupy no cell
        }
        if col + w > cols {
            rows += 1;
            col = 0;
        }
        col += w;
    }
    (rows, col)
}

/// Physical rows a rendered line occupies at `cols`.
fn wrapped_rows(line: &str, cols: usize) -> usize {
    layout_line(line, cols).0
}

/// Merge a Frag list into rendered physical lines (Frags may contain '\n').
fn group_lines(frags: Vec<Frag>) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for f in frags {
        let styled_open = !f.style.is_empty();
        for (i, part) in f.text.split('\n').enumerate() {
            if i > 0 {
                lines.push(std::mem::take(&mut cur));
            }
            if !part.is_empty() {
                if styled_open {
                    cur.push_str(&f.style);
                }
                cur.push_str(part);
                if styled_open {
                    cur.push_str("\x1b[0m");
                }
            }
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Cell width helper for a single char (used by tests).
#[allow(dead_code)]
fn char_width(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// A double-width glyph is never split: one that would straddle the right
    /// edge moves whole to the next row, leaving the last cell blank. Dividing
    /// total width by cols misses those blanks — with Hangul the error
    /// compounds and a long line under-reports its rows.
    #[test]
    fn wide_glyph_never_straddles_the_edge() {
        // 45 cols, 45 Hangul (90 cells). Each row fits 22 glyphs (44 cells)
        // and wastes the 45th, so it is 22 + 22 + 1 = THREE rows.
        // 90/45 = 2 was the old (wrong) answer.
        let s: String = "가".repeat(45);
        assert_eq!(wrapped_rows(&s, 45), 3);
        // even width divides cleanly: 44 cols fits exactly 22 per row
        assert_eq!(wrapped_rows(&"가".repeat(44), 44), 2);
    }

    #[test]
    fn ascii_layout_matches_division() {
        assert_eq!(wrapped_rows(&"a".repeat(40), 40), 1);
        assert_eq!(wrapped_rows(&"a".repeat(41), 40), 2);
        assert_eq!(wrapped_rows("", 40), 1);
    }

    #[test]
    fn layout_reports_the_ending_column() {
        // 3 Hangul = 6 cells on a wide line
        assert_eq!(layout_line("가나다", 40), (1, 6));
        // 45-col line: 22 glyphs fill 44, the 23rd wraps and sits at col 2
        assert_eq!(layout_line(&"가".repeat(23), 45), (2, 2));
    }

    #[test]
    fn sgr_and_combining_marks_occupy_no_cells() {
        assert_eq!(layout_line("\x1b[7m가\x1b[0m", 40), (1, 2));
        assert_eq!(layout_line("e\u{301}", 40), (1, 1)); // e + combining acute
    }
}

/// The lock-order regression this file exists to not repeat.
///
/// Two locks, `paint` and std's stdout, taken by every painting path. As long
/// as every path takes them in the same order there is no cycle; the moment one
/// path holds stdout and then reaches for paint, two threads can wedge. It hung
/// `cm2 -rt -dsp` on 2026-09-29 with no output and no prompt, and nothing in
/// the process noticed: parking_lot has no deadlock detection and std's stdout
/// lock has none either, so the only symptom is silence.
///
/// This test drives the two paths that met in that hang — a turn thread
/// printing through `write_above`, a progress thread calling `repaint` — from
/// two real threads, and fails on a watchdog instead of hanging the suite.
#[cfg(test)]
mod lock_order_tests {
    use super::*;

    fn test_hooks() -> EngineHooks {
        EngineHooks {
            message_lines: Arc::new(Vec::new),
            rprompt: Arc::new(Vec::new),
            prompt_marker: Arc::new(|| vec![Frag::plain("> ")]),
            permission_pending: Arc::new(|| false),
            processing: Arc::new(|| false),
            paused: Arc::new(|| false),
            on_event: Arc::new(|_| {}),
            on_paste: Arc::new(|_| None),
            completions: Arc::new(|_, _| Vec::new()),
            auto_suggest: Arc::new(|_| None),
        }
    }

    #[test]
    fn write_above_and_repaint_never_deadlock() {
        let engine = PromptEngine::new(test_hooks());
        let rounds = 400;

        let (done_tx, done_rx) = std::sync::mpsc::channel::<&'static str>();

        let printer = {
            let e = engine.clone();
            let tx = done_tx.clone();
            std::thread::spawn(move || {
                for _ in 0..rounds {
                    e.write_above("tool result\n");
                }
                let _ = tx.send("printer");
            })
        };
        let painter = {
            let e = engine.clone();
            let tx = done_tx.clone();
            std::thread::spawn(move || {
                for _ in 0..rounds {
                    e.repaint();
                }
                let _ = tx.send("painter");
            })
        };
        drop(done_tx);

        // Watchdog. A deadlock here is silence, so the assertion has to be a
        // deadline: generous enough for a loaded CI box, far below "forever".
        let deadline = std::time::Duration::from_secs(20);
        let mut finished = Vec::new();
        for _ in 0..2 {
            match done_rx.recv_timeout(deadline) {
                Ok(who) => finished.push(who),
                Err(_) => panic!(
                    "deadlock: {:?} finished, the other thread is wedged. \
                     A painting path is holding stdout while reaching for `paint` \
                     (or the reverse) — see with_out().",
                    finished
                ),
            }
        }
        printer.join().unwrap();
        painter.join().unwrap();
    }
}
