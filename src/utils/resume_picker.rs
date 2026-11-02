//! Interactive session picker for `resume` — adapted from
//! components/ResumeTask.tsx's session list UX.
//! Equivalent to codemason/utils/resume_picker.py; the prompt_toolkit
//! Application substrate becomes a crossterm inline repaint loop (same
//! divergence as arrow_picker.rs — the RENDERED UI and key contract are
//! identical: ↑/↓ move, Enter pick, d-d delete with double-press confirm,
//! Esc/q cancel, 1s refresh so the age column ticks).
//!
//! Falls back to a numbered text prompt when the terminal can't run the
//! interactive list (e.g. dumb terminals / no tty).

use std::io::{IsTerminal, Write};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;

use crate::utils::formatting::display_width;
use crate::utils::session_storage::{list_sessions, SessionInfo};

/// Cap the list like the original picker viewport; extension point: paging.
pub const MAX_SESSIONS: usize = 50;

/// Relative modified time, always down to seconds ('42s ago', '5m 3s ago',
/// '2h 14m 3s ago', '3d 7h 12m 5s ago') — every row visibly ticks with the
/// picker's 1s refresh, day-old sessions included.
fn format_age(iso_ts: &str) -> String {
    let ts = chrono::DateTime::parse_from_rfc3339(&iso_ts.replace('Z', "+00:00"));
    let Ok(ts) = ts else {
        return "?".to_string();
    };
    let delta = chrono::Utc::now().signed_duration_since(ts.with_timezone(&chrono::Utc));
    let secs = delta.num_seconds().max(0);
    let (d, rem) = (secs / 86400, secs % 86400);
    let (h, rem2) = (rem / 3600, rem % 3600);
    let (m, s) = (rem2 / 60, rem2 % 60);
    let mut parts: Vec<String> = Vec::new();
    if d > 0 {
        parts.push(format!("{d}d"));
    }
    if h > 0 || !parts.is_empty() {
        parts.push(format!("{h}h"));
    }
    if m > 0 || !parts.is_empty() {
        parts.push(format!("{m}m"));
    }
    parts.push(format!("{s}s"));
    format!("{} ago", parts.join(" "))
}

fn row_text(s: &SessionInfo, width: usize) -> String {
    let age = format_age(&s.modified);
    let count = s.message_count;
    let prompt = if s.first_prompt.is_empty() { "No prompt" } else { &s.first_prompt };
    let head = format!("{age:>16}  {count:>4} msg  ");
    let room = width.saturating_sub(display_width(&head) + 4).max(10);
    let mut shown = prompt.to_string();
    if display_width(prompt) > room {
        // display_width-aware truncation (CJK chars are 2 cells wide)
        let mut out = String::new();
        let mut used = 0usize;
        for ch in prompt.chars() {
            let w = display_width(&ch.to_string());
            if used + w > room.saturating_sub(1) {
                break;
            }
            out.push(ch);
            used += w;
        }
        shown = format!("{out}…");
    }
    format!("{head}{shown}")
}

/// First row of the viewport, given the selection and how many rows fit.
///
/// The list is repainted by walking back `painted` rows with ESC[nA, which only
/// reaches rows still on screen.  A block TALLER than the window scrolls its own
/// top away, the walk-back can no longer reach it, and every 1s refresh leaves
/// another full copy of the list behind — with 100+ stored sessions that is what
/// the phone terminal showed.  Keeping the drawn block inside the window is the
/// fix; this slides it so the selected row stays visible.
fn viewport_offset(index: usize, shown: usize, total: usize) -> usize {
    if shown >= total || index < shown {
        0
    } else {
        (index + 1 - shown).min(total - shown)
    }
}

/// Arrow-key session picker. Returns the chosen session, or None if
/// cancelled / nothing to resume.
pub async fn pick_session() -> Option<SessionInfo> {
    let mut sessions = list_sessions(Some(MAX_SESSIONS));
    // Don't offer the ACTIVE session — resuming the conversation you're in is
    // a no-op (and adopt would needlessly reload it).
    let current = crate::utils::state::get_session_id();
    sessions.retain(|s| s.session_id != current);
    if sessions.is_empty() {
        crate::cmprintln!("No stored sessions for this directory.");
        return None;
    }

    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        // No tty (piped/dumb) — go straight to the numbered prompt.
        return pick_with_numbers(&sessions);
    }
    match pick_interactive(&mut sessions).await {
        Ok(r) => r,
        Err(_) => pick_with_numbers(&sessions),
    }
}

async fn pick_interactive(sessions: &mut Vec<SessionInfo>) -> std::io::Result<Option<SessionInfo>> {
    let (width, height) = crossterm::terminal::size()
        .map(|(c, r)| (c as usize, r as usize))
        .unwrap_or((80, 24));
    // The block is repainted by walking back `painted` rows with ESC[nA, which
    // only reaches rows still on screen.  Drawing a block TALLER than the window
    // scrolls its top away, the walk-back can no longer reach it, and each 1s
    // refresh leaves another full copy of the list behind.  With 100+ stored
    // sessions the list is far taller than a phone terminal, so show a viewport
    // that always fits and scroll it with the selection (prompt_toolkit clips
    // its Window to the screen for the same reason).
    let viewport = height.saturating_sub(2).max(3);
    let mut index = 0usize;
    // delete_armed: index awaiting a second 'd' to confirm deletion.
    let mut delete_armed: Option<usize> = None;

    crate::utils::tty_compat::enable_raw_mode()?;
    // The exclusive input token (see utils/input_events.rs).  The caller has
    // parked the engine via pause_prompt(), so this resolves immediately; if a
    // caller ever forgets, this WAITS instead of deadlocking crossterm.
    let mut events = crate::utils::input_events::acquire().await;
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
    let mut painted = 0usize;
    let result: Option<SessionInfo>;

    let paint = |sessions: &[SessionInfo], index: usize, delete_armed: Option<usize>, painted: &mut usize| -> std::io::Result<()> {
        // Slide the viewport so the selected row is always inside it.
        let shown = viewport.min(sessions.len());
        let offset = viewport_offset(index, shown, sessions.len());
        let mut o = std::io::stdout().lock();
        if *painted > 0 {
            write!(o, "\x1b[{}A", painted)?;
        }
        let header = crate::utils::formatting::clip_to_width(
            "Resume a session (↑/↓ move, Enter select, d delete, Esc cancel)",
            width,
        );
        write!(o, "\r\x1b[2K\x1b[1m{header}\x1b[0m\r\n")?;
        for (i, s) in sessions.iter().enumerate().skip(offset).take(shown) {
            let row = row_text(s, width);
            let (style, text) = if i == index {
                if delete_armed == Some(i) {
                    ("\x1b[31;7m", format!("❯ {row}  (d again to delete)"))
                } else {
                    ("\x1b[7m", format!("❯ {row}"))
                }
            } else {
                ("", format!("  {row}"))
            };
            let text = crate::utils::formatting::clip_to_width(&text, width);
            let reset = if style.is_empty() { "" } else { "\x1b[0m" };
            write!(o, "\r\x1b[2K{style}{text}{reset}\r\n")?;
        }
        o.flush()?;
        *painted = shown + 1;
        Ok(())
    };

    paint(sessions, index, delete_armed, &mut painted)?;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                // 1s refresh so the "Ns ago" column ticks in real time.
                paint(sessions, index, delete_armed, &mut painted)?;
            }
            ev = events.next() => {
                let Some(Event::Key(KeyEvent { code, modifiers, .. })) = ev else { continue };
                match code {
                    KeyCode::Up => {
                        index = (index + sessions.len() - 1) % sessions.len();
                        delete_armed = None; // moving away disarms a pending delete
                    }
                    KeyCode::Down => {
                        index = (index + 1) % sessions.len();
                        delete_armed = None;
                    }
                    KeyCode::Enter => {
                        result = Some(sessions[index].clone());
                        break;
                    }
                    KeyCode::Char('d') => {
                        if delete_armed != Some(index) {
                            delete_armed = Some(index); // first press arms; second confirms
                        } else {
                            delete_armed = None;
                            if crate::utils::session_storage::delete_session(std::path::Path::new(
                                &sessions[index].full_path,
                            )) {
                                sessions.remove(index);
                                if sessions.is_empty() {
                                    result = None;
                                    break;
                                }
                                index = index.min(sessions.len() - 1);
                                // Repaint leaves one stale row at the bottom;
                                // clear downward after moving up next paint.
                                let mut o = std::io::stdout().lock();
                                write!(o, "\x1b[{}A\r\x1b[0J", painted)?;
                                o.flush()?;
                                painted = 0;
                            }
                            // already gone / unreadable — just leave the row
                        }
                    }
                    KeyCode::Esc | KeyCode::Char('q') => {
                        result = None;
                        break;
                    }
                    KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                        result = None;
                        break;
                    }
                    _ => {}
                }
                paint(sessions, index, delete_armed, &mut painted)?;
            }
        }
    }

    crate::utils::tty_compat::disable_raw_mode();
    Ok(result)
}

/// Plain fallback for terminals where the interactive list can't run.
fn pick_with_numbers(sessions: &[SessionInfo]) -> Option<SessionInfo> {
    let width = crossterm::terminal::size().map(|(c, _)| c as usize).unwrap_or(80);
    crate::cmprintln!("Stored sessions:");
    for (i, s) in sessions.iter().enumerate() {
        crate::cmprintln!("  {:>2}. {}", i + 1, row_text(s, width));
    }
    crate::cmprint!("Session number (empty to cancel): ");
    let mut raw = String::new();
    if std::io::stdin().read_line(&mut raw).is_err() {
        return None;
    }
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let Ok(idx) = raw.parse::<usize>() else {
        crate::cmprintln!("Not a number.");
        return None;
    };
    if idx < 1 || idx > sessions.len() {
        crate::cmprintln!("Out of range.");
        return None;
    }
    Some(sessions[idx - 1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn age_formats() {
        let now = chrono::Utc::now() - chrono::Duration::seconds(42);
        let s = format_age(&now.to_rfc3339());
        assert!(s.ends_with("s ago"), "{s}");
        let old = chrono::Utc::now() - chrono::Duration::seconds(2 * 3600 + 14 * 60 + 3);
        let s2 = format_age(&old.to_rfc3339());
        assert!(s2.starts_with("2h 14m"), "{s2}");
    }
}

#[cfg(test)]
mod viewport_tests {
    use super::*;

    #[test]
    fn everything_visible_starts_at_the_top() {
        assert_eq!(viewport_offset(0, 20, 12), 0);
        assert_eq!(viewport_offset(11, 20, 12), 0);
    }

    #[test]
    fn selection_stays_inside_the_viewport() {
        let (shown, total) = (10usize, 50usize);
        for index in 0..total {
            let off = viewport_offset(index, shown, total);
            assert!(index >= off && index < off + shown,
                    "row {index} outside viewport {off}..{}", off + shown);
            assert!(off + shown <= total, "viewport runs past the end at {index}");
        }
    }

    #[test]
    fn viewport_does_not_scroll_past_the_last_row() {
        assert_eq!(viewport_offset(49, 10, 50), 40);
        assert_eq!(viewport_offset(9, 10, 50), 0);
        assert_eq!(viewport_offset(10, 10, 50), 1);
    }
}
