//! Inline arrow-key picker (radio / checkbox).
//! Equivalent to codemason/utils/arrow_picker.py.
//!
//! Substrate divergence (documented): the Python built this on a short-lived
//! prompt_toolkit Application; Rust renders the same inline UI directly with
//! crossterm (raw mode + cursor-up repaint) — not full-screen, so REPL
//! scrollback survives. In the interactive REPL it must run while the
//! persistent prompt session is PAUSED (one reader per terminal at a time).
//! Falls back to numbered input when raw mode can't engage (e.g. non-tty).
//!
//! Used by AskUserQuestion so the user picks with ↑/↓/enter instead of typing
//! a number into a raw read that fights the REPL's input loop.

use std::collections::BTreeSet;
use std::io::{IsTerminal, Write};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::utils::i18n::{tr, trf, FmtArg};

pub struct ArrowSelectOptions<'a> {
    pub descriptions: Option<&'a [Option<String>]>,
    pub previews: Option<&'a [Option<String>]>,
    pub multi: bool,
    pub header: Option<&'a str>,
    /// Takes the picker down from outside — used when the same question was
    /// answered somewhere else (the phone). It exits the picker the way Esc
    /// would, so the screen is put back.
    pub close_when: Option<CancellationToken>,
}

impl Default for ArrowSelectOptions<'_> {
    fn default() -> Self {
        ArrowSelectOptions { descriptions: None, previews: None, multi: false, header: None, close_when: None }
    }
}

/// Present an inline picker over `options`.
///
/// Returns the selected option indices (a single-element list for
/// single-select), or None if the user cancelled (Esc / Ctrl+C). For
/// multi-select, Enter with nothing toggled selects the highlighted row.
pub async fn arrow_select(
    title: &str,
    options: &[String],
    opts: ArrowSelectOptions<'_>,
) -> Option<Vec<usize>> {
    if options.is_empty() {
        return None;
    }
    match select_interactive(title, options, &opts).await {
        Ok(result) => result,
        // Any terminal failure (no tty, raw-mode refusal) → plain numbers.
        Err(_) => select_numbered(title, options, opts.descriptions, opts.multi, opts.header),
    }
}

struct PickerState {
    index: usize,
    selected: BTreeSet<usize>,
}

fn render_lines(
    title: &str,
    options: &[String],
    opts: &ArrowSelectOptions<'_>,
    state: &PickerState,
    max_prev: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(h) = opts.header {
        lines.push(format!("\x1b[1m[{h}]\x1b[0m"));
    }
    lines.push(format!("\x1b[1m{title}\x1b[0m"));
    for (i, opt) in options.iter().enumerate() {
        let cur = i == state.index;
        let cursor = if cur { "❯" } else { " " };
        let body = if opts.multi {
            let boxmark = if state.selected.contains(&i) { "[x]" } else { "[ ]" };
            format!("{cursor} {boxmark} {opt}")
        } else {
            format!("{cursor} {opt}")
        };
        if cur {
            lines.push(format!("\x1b[1m{body}\x1b[0m"));
        } else {
            lines.push(body);
        }
        // Every option carries its description, not just the highlighted one
        // (three of four unexplained choices right when the choice is being
        // made read wrong — see the Python comment).
        if let Some(descs) = opts.descriptions {
            if let Some(Some(desc)) = descs.get(i) {
                // fg:#8a8580 → nearest 256
                let idx = crate::utils::formatting::hex_to_ansi256("8a8580");
                lines.push(format!("  \x1b[38;5;{idx}m— {desc}\x1b[0m"));
            }
        }
    }
    // fixed preview area for the highlighted row, reserved at max height so
    // the inline block doesn't jump as the cursor moves
    if max_prev > 0 {
        let pv: Vec<String> = opts
            .previews
            .and_then(|p| p.get(state.index))
            .and_then(|p| p.as_ref())
            .map(|p| p.split('\n').map(|s| s.to_string()).collect())
            .unwrap_or_default();
        for j in 0..max_prev {
            let seg = pv.get(j).map(|s| s.as_str()).unwrap_or("");
            lines.push(format!("    │ {seg}"));
        }
    }
    let hint = if opts.multi {
        tr("↑/↓ move · space toggle · enter confirm · esc cancel")
    } else {
        tr("↑/↓ move · enter select · esc cancel")
    };
    lines.push(hint);
    lines
}

async fn select_interactive(
    title: &str,
    options: &[String],
    opts: &ArrowSelectOptions<'_>,
) -> std::io::Result<Option<Vec<usize>>> {
    if !std::io::stdout().is_terminal() {
        return Err(std::io::Error::other("not a tty"));
    }
    let n = options.len();
    let mut state = PickerState { index: 0, selected: BTreeSet::new() };

    // Reserve a fixed preview area (max lines over all options).
    let max_prev = opts
        .previews
        .map(|ps| {
            ps.iter()
                .filter_map(|p| p.as_ref())
                .map(|p| p.split('\n').count())
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);

    crate::utils::tty_compat::enable_raw_mode()?;
    // Exclusive input token (utils/input_events.rs); the engine is parked by
    // the caller (pause_prompt), so acquisition is immediate.
    let mut events = crate::utils::input_events::acquire().await;
    let mut painted_lines = 0usize;
    let result: Option<Vec<usize>>;

    let mut paint = |state: &PickerState, painted_lines: &mut usize| -> std::io::Result<()> {
        let lines = render_lines(title, options, opts, state, max_prev);
        // Clip to the window like Python's wrap_lines=False Window: a wrapped
        // line would desync the ESC[nA walk-back below and march the block down
        // the screen once per repaint.
        let width = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(80).max(20);
        let mut out = std::io::stdout();
        if *painted_lines > 0 {
            write!(out, "\x1b[{}A", painted_lines)?;
        }
        for line in &lines {
            // \r + clear-to-end so a shorter repaint leaves no residue.
            let line = crate::utils::formatting::clip_to_width(line, width);
            write!(out, "\r\x1b[2K{line}\n")?;
        }
        out.flush()?;
        *painted_lines = lines.len();
        Ok(())
    };

    paint(&state, &mut painted_lines)?;

    loop {
        let ev = if let Some(token) = &opts.close_when {
            tokio::select! {
                e = events.next() => e,
                _ = token.cancelled() => {
                    result = None;
                    break;
                }
            }
        } else {
            events.next().await
        };
        let Some(event) = ev else {
            result = None;
            break;
        };
        if let Event::Key(KeyEvent { code, modifiers, .. }) = event {
            match code {
                KeyCode::Up => {
                    state.index = (state.index + n - 1) % n;
                }
                KeyCode::Down => {
                    state.index = (state.index + 1) % n;
                }
                KeyCode::Char(' ') if opts.multi => {
                    if !state.selected.remove(&state.index) {
                        state.selected.insert(state.index);
                    }
                }
                KeyCode::Enter => {
                    result = if opts.multi {
                        let sel: Vec<usize> = state.selected.iter().copied().collect();
                        Some(if sel.is_empty() { vec![state.index] } else { sel })
                    } else {
                        Some(vec![state.index])
                    };
                    break;
                }
                KeyCode::Esc => {
                    result = None;
                    break;
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    result = None;
                    break;
                }
                _ => {}
            }
            paint(&state, &mut painted_lines)?;
        }
    }

    crate::utils::tty_compat::disable_raw_mode();
    Ok(result)
}

/// Fallback for terminals where the interactive picker can't run.
fn select_numbered(
    title: &str,
    options: &[String],
    descriptions: Option<&[Option<String>]>,
    multi: bool,
    header: Option<&str>,
) -> Option<Vec<usize>> {
    if let Some(h) = header {
        crate::cmprintln!("[{h}]");
    }
    crate::cmprintln!("{title}");
    for (i, opt) in options.iter().enumerate() {
        crate::cmprintln!("  {}. {opt}", i + 1);
        if let Some(descs) = descriptions {
            if let Some(Some(d)) = descs.get(i) {
                crate::cmprintln!("     {d}");
            }
        }
    }
    let hint = if multi { tr("numbers comma-separated") } else { tr("a number") };
    loop {
        crate::cmprint!("  {}", trf("Select {hint}: ", &[("hint", FmtArg::Str(hint.clone()))]));
        let _ = std::io::stdout().flush();
        let mut raw = String::new();
        if std::io::stdin().read_line(&mut raw).is_err() {
            return None;
        }
        if raw.is_empty() {
            return None; // EOF
        }
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        let picks: Vec<&str> = if multi { raw.split(',').map(|p| p.trim()).collect() } else { vec![raw] };
        let mut idxs = Vec::new();
        let mut ok = true;
        for p in picks {
            match p.parse::<usize>() {
                Ok(v) if v >= 1 && v <= options.len() => idxs.push(v - 1),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && !idxs.is_empty() {
            return Some(idxs);
        }
    }
}
