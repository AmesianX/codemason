//! The server's abliteration switch, driven from the turn loop.
//!
//! The served checkpoint has its refusal behaviour intact; the edit that
//! removes it is a rank-24 adapter the server can add to `o_proj` per
//! request (`POST /ab {"scale":0|1}`, see the lane's AB-SWITCH.md). The reason
//! it is a switch and not a checkpoint choice: removing refusal also removes
//! the model's ability to say a thing does not exist, because declining a
//! request and denying a premise run through the same direction in the
//! weights. Measured on the served pair — 4/60 fabrications with it off,
//! 18-22/60 with it on. So the model that judges a tool result, a link or a
//! diff should be the restored one.
//!
//! Which leaves exactly one REQUEST that needs the edit: the first one of a
//! user turn, the one that decides whether to accept. Hence `auto` — on for
//! that request, off the moment it answers:
//!
//!     user turn starts       -> arm (no request)
//!     turn's first request   -> on
//!     it answers, anything   -> off
//!     every request after    -> stays off
//!     next user turn         -> arm again
//!
//! The off is paired with the on, inside one request, and that pairing is the
//! whole point. It used to hang off the first TOOL CALL instead, on the
//! assumption that a turn which accepts goes on to call one. Turns that never
//! call a tool — a plain text answer, an empty turn, an abort, a generation the
//! degenerate guard cuts — left the switch ON, and nothing else ever turned it
//! off: not the end of the turn, not the next turn (which pushes the same 1.0
//! and is skipped as already-there), not process exit.
//!
//! Measured 2026-10-01 on this install. The last tool call of the session was
//! 16:34:51; everything after it was question-and-answer, so the edit stayed on
//! from there. At 16:40:20 the reasoning channel collapsed into a single
//! repeated character, and all seven repetition retries went out on the edited
//! weights — the off was unreachable in precisely the failure mode that needed
//! it, because a collapsed turn cannot call a tool. The turn ended on
//! REPETITION_GIVEUP_NOTE, the next user message did the same, and `GET /ab`
//! still read `{"scale":1.0,"mode":"abliterated"}` an hour later: the exact
//! "switch left on for hours" this module was written to avoid. Compare the
//! same day before the switch shipped — 13 collapses between 10:35 and 10:40,
//! 13 escapes.
//!
//! `on` and `off` exist to pin the state for A/B measurement, not for daily
//! use.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::time::Duration;

/// What the user asked for, not what the server currently holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Never touch the switch.
    Off,
    /// Pin it on, for measurement.
    On,
    /// On for the turn's first request, off as soon as it answers (the default).
    Auto,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_lowercase().as_str() {
            "off" | "0" | "false" => Some(Mode::Off),
            "on" | "1" | "true" => Some(Mode::On),
            "auto" => Some(Mode::Auto),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::On => "on",
            Mode::Auto => "auto",
        }
    }
}

/// `auto` by default: the lane serves the restored checkpoint, so without the
/// switch every request that needs the edit gets refused. A server with no
/// `/ab` route (or an llama.cpp backend) simply fails the push and is left
/// alone, so this default costs nothing where the feature does not exist.
/// Override with `CODEMASON_AB=off|on|auto` for a run.
static MODE: Lazy<Mutex<Mode>> = Lazy::new(|| {
    let from_env = std::env::var("CODEMASON_AB").ok().and_then(|v| Mode::parse(&v));
    Mutex::new(from_env.unwrap_or(Mode::Auto))
});
/// Last scale we pushed, so the common case (already there) costs no request.
static PUSHED: Lazy<Mutex<Option<f32>>> = Lazy::new(|| Mutex::new(None));
/// Set when a user turn starts, spent by that turn's first request.
///
/// A latch rather than a push: arming costs no request, and nothing can be
/// left on by arming alone, because only `begin_request` turns the switch on
/// and it hands the caller the obligation to turn it back off.
static ARMED: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

pub fn mode() -> Mode {
    *MODE.lock()
}

/// Set the mode and bring the server in line with it right away.
pub async fn set_mode(m: Mode) -> Result<(), String> {
    *MODE.lock() = m;
    // Forced: a command typed by the user must reach the server, never be
    // answered from our guess about what it already holds. See `push_inner`.
    match m {
        Mode::On => push_inner(1.0, true).await,
        // Leaving auto or turning it off must not strand the server on.
        Mode::Off | Mode::Auto => push_inner(0.0, true).await,
    }
}

/// A user turn is starting: arm the edit for that turn's FIRST request.
///
/// Nothing is pushed here. The push belongs to the request, so that the off can
/// be paired with it — see `begin_request` / `end_request`.
pub async fn on_user_turn() {
    if mode() == Mode::Auto {
        *ARMED.lock() = true;
    }
}

/// About to send a request: under `auto`, turn the edit on iff this is the
/// turn's first one.
///
/// Returns whether it was turned on. The caller MUST hand that back to
/// `end_request`, which is what makes "on" impossible to leak: there is no
/// path that turns it on and then waits for some later event to turn it off.
pub async fn begin_request() -> bool {
    if mode() != Mode::Auto {
        return false;
    }
    // Scope the guard: it must be dropped before the await below.
    let armed = {
        let mut armed = ARMED.lock();
        let was = *armed;
        *armed = false;
        was
    };
    if !armed {
        return false;
    }
    let _ = push(1.0).await;
    true
}

/// That request has answered — off, whatever it answered with.
///
/// A tool call, a sentence, an empty turn, an abort and a generation the
/// degenerate guard cut all reach here. That is the point: the previous version
/// turned off only on a tool call, which the collapsed turns of 2026-10-01
/// never reached (see the module doc).
pub async fn end_request(turned_on: bool) {
    if !turned_on {
        return;
    }
    // The mode can change while the request is in flight, and this off was
    // armed under the mode that held when it STARTED. `/ab on` typed mid-turn
    // pins the edit deliberately; undoing it a moment later, on the strength of
    // a decision the user has since overridden, is how the switch ended up
    // stuck off while the REPL said "on" (2026-11-20):
    //
    //   turn in flight       begin_request pushed 1.0
    //   user types /ab on    set_mode(On) -> 1.0, prints "/ab mode: on"
    //   turn answers         end_request pushed 0.0   <-- stomps the pin
    //   from then on         mode is On, so begin_request returns false and
    //                        NOTHING ever pushes 1.0 again -- every later turn
    //                        refuses, under a display that reads "on"
    //
    // `Off` still pushes: parking the server is always safe, and never
    // stranding it on is the point of the pairing.
    if mode() == Mode::On {
        return;
    }
    let _ = push(0.0).await;
}

/// Ask the server what it currently holds. Used by `/ab` with no argument, so
/// the reported state is the server's, not our idea of it.
pub async fn query() -> Option<(f32, bool)> {
    let (url, key) = endpoint()?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build().ok()?;
    let mut req = client.get(&url);
    if !key.is_empty() {
        req = req.bearer_auth(key);
    }
    let body: serde_json::Value = req.send().await.ok()?.json().await.ok()?;
    effective(&body)
}

/// The scale the server will actually apply, read out of an `/ab` body.
///
/// The lane's switch went per-token (`ab_switch.py`): the value that reaches
/// the forward pass is `_default_scale`, which `fill_token_scales` writes into
/// the step's vector for every request that did not send its own `ab_scale`,
/// and which the server mirrors into each worker's `details[].default_scale`.
/// The top-level `scale` is `get_scale()` — the legacy scalar `_scale`, which
/// `_token_scale()` stops allocating the moment the per-token buffer is
/// reserved, so it reads `0.0` for the life of the process.
///
/// Reading that field printed `server: off (restored)` over a server holding
/// the edit ON (measured 2026-11-21: `POST {"scale":1}` answered
/// `{"scale":1.0,"mode":"abliterated"}`, the `GET` right after it answered
/// `{"scale":0.0,"mode":"original","details":[{"default_scale":1.0,..}]}`).
/// It is the mirror image of the 2026-11-20 bug: the display said `on` over a
/// server at 0.0, the fix was to believe the server instead of `PUSHED`, and
/// the field we started believing had quietly stopped meaning anything.
///
/// `consistent` has the same hole — the server derives it from those
/// always-zero scalars, so ranks holding different defaults still look agreed.
/// Compare the per-worker defaults instead.
///
/// A body without `per_token` is a server from before that change (or another
/// server entirely), and there the top-level pair is the right answer.
fn effective(body: &serde_json::Value) -> Option<(f32, bool)> {
    let per_token = body
        .get("per_token")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if per_token {
        if let Some(details) = body.get("details").and_then(|v| v.as_array()) {
            let defaults: Vec<f32> = details
                .iter()
                .filter_map(|d| d.get("default_scale").and_then(|v| v.as_f64()))
                .map(|v| v as f32)
                .collect();
            if let Some(&head) = defaults.first() {
                let consistent = defaults.iter().all(|s| (s - head).abs() <= 1e-6);
                return Some((head, consistent));
            }
        }
    }
    Some((
        body.get("scale")?.as_f64()? as f32,
        body.get("consistent").and_then(|v| v.as_bool()).unwrap_or(true),
    ))
}

fn endpoint() -> Option<(String, String)> {
    let cfg = crate::utils::config::get_config();
    let base = cfg.model_base_url.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return None;
    }
    Some((format!("{base}/ab"), cfg.model_api_key.clone()))
}

/// Push a scale, skipping the request when we believe the server has it.
async fn push(scale: f32) -> Result<(), String> {
    push_inner(scale, false).await
}

/// Push a scale. `force` sends it even when `PUSHED` says it is already there.
///
/// `PUSHED` is a cache over a value this process does not own, and it has no
/// invalidation: the server resets to `GLM53_AB_SCALE` (0) whenever it
/// restarts, and every other client sharing it pushes its own scales. So it is
/// a guess, and it goes stale silently.
///
/// Measured 2026-11-20 on the live lane. `/ab on` printed `/ab mode: on` and
/// the model went on refusing, for hours. `set_mode(On)` had called
/// `push(1.0)`, `PUSHED` was already `Some(1.0)` from before the vLLM
/// container was restarted the previous morning, so the shortcut returned
/// `Ok(())` without sending anything; `GET /ab` read `0.0` the whole time. The
/// REPL has no way to tell that apart from a push that landed, because `Ok`
/// meant "we think so", not "the server says so".
///
/// Two changes, both about not trusting our own guess:
///   * an explicit user command forces the request (`set_mode`);
///   * `PUSHED` is set from the scale the SERVER echoes, not from what we
///     sent, and cleared when the server does not say -- so the next push
///     cannot be skipped on the strength of an unconfirmed one.
///
/// The shortcut itself stays for `auto`: there both pushes are ours, inside
/// one turn, and the pair is re-pushed on the next turn anyway.
///
/// A failure is reported to the caller but never blocks the turn: a server
/// without the plugin (or an llama.cpp backend) has no `/ab`, and that is a
/// normal configuration, not an error worth interrupting work over.
async fn push_inner(scale: f32, force: bool) -> Result<(), String> {
    if !force && *PUSHED.lock() == Some(scale) {
        return Ok(());
    }
    let (url, key) = match endpoint() {
        Some(v) => v,
        None => return Err("model_base_url is not set".into()),
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client.post(&url).json(&serde_json::json!({ "scale": scale }));
    if !key.is_empty() {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        let code = resp.status();
        let body = resp.text().await.unwrap_or_default();
        // The push did not land, and we no longer know what the server holds.
        *PUSHED.lock() = None;
        return Err(format!("{code}: {}", body.trim()));
    }
    // Believe the server, not ourselves. Our `/ab` answers with
    // {"scale":..,"workers":[..],"consistent":bool}; anything else (another
    // server, an empty 200) leaves us with no confirmation, so record none and
    // let the next push go out rather than be skipped.
    let body = resp.text().await.unwrap_or_default();
    let parsed: Option<serde_json::Value> = serde_json::from_str(&body).ok();
    // Through `effective`, so a POST that ever grows the per-token diagnostics
    // is read the same way the GET is. On today's reply (scale/workers/
    // consistent/mode) it is the top-level pair, unchanged.
    let served = parsed.as_ref().and_then(effective);
    *PUSHED.lock() = served.map(|(s, _)| s);
    if let Some((s, consistent)) = served {
        if (s - scale).abs() > 1e-6 {
            return Err(format!("server took {s}, not {scale}"));
        }
        if !consistent {
            return Err("workers disagree on the scale".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_three_modes_and_rejects_the_rest() {
        assert_eq!(Mode::parse("auto"), Some(Mode::Auto));
        assert_eq!(Mode::parse("ON"), Some(Mode::On));
        assert_eq!(Mode::parse(" off "), Some(Mode::Off));
        assert_eq!(Mode::parse("maybe"), None);
        // Round trip, so /ab with no argument reports what /ab took.
        for m in [Mode::Off, Mode::On, Mode::Auto] {
            assert_eq!(Mode::parse(m.as_str()), Some(m));
        }
    }

    /// The live lane's own answer, pasted from `GET /ab` on 2026-11-21 while
    /// the edit was ON: top-level `scale` 0.0, `default_scale` 1.0. Reading the
    /// top level printed `server: off (restored)` over an abliterated server.
    #[test]
    fn per_token_body_reports_the_default_not_the_dead_scalar() {
        let body: serde_json::Value = serde_json::from_str(
            r#"{"scale":0.0,"workers":[0.0,0.0],"consistent":true,"mode":"original",
                "per_token":true,"buffer_tokens":9236,"fills":[41329,41329],
                "details":[{"scale":0.0,"default_scale":1.0,"per_token":true},
                           {"scale":0.0,"default_scale":1.0,"per_token":true}]}"#,
        )
        .unwrap();
        assert_eq!(effective(&body), Some((1.0, true)));
    }

    /// `consistent` from the server is computed over the always-zero scalars,
    /// so it says `true` while the ranks hold different defaults — which at TP=2
    /// means the halves of the model disagree about which weights they are.
    #[test]
    fn per_token_disagreement_is_caught_even_when_the_server_says_consistent() {
        let body: serde_json::Value = serde_json::from_str(
            r#"{"scale":0.0,"consistent":true,"per_token":true,
                "details":[{"default_scale":1.0},{"default_scale":0.0}]}"#,
        )
        .unwrap();
        assert_eq!(effective(&body), Some((1.0, false)));
    }

    /// A server from before the per-token change, and the POST reply, carry no
    /// `per_token`: the top-level pair is the answer there and must stay so.
    #[test]
    fn legacy_body_still_reads_the_top_level_pair() {
        let body: serde_json::Value = serde_json::from_str(
            r#"{"scale":1.0,"workers":[1.0,1.0],"consistent":false,"mode":"abliterated"}"#,
        )
        .unwrap();
        assert_eq!(effective(&body), Some((1.0, false)));
        // per_token with the details missing is not something the plugin sends
        // (it adds both or neither), but guessing 0.0 there would be the same
        // mistake again, so fall back rather than invent.
        let partial: serde_json::Value =
            serde_json::from_str(r#"{"scale":1.0,"per_token":true}"#).unwrap();
        assert_eq!(effective(&partial), Some((1.0, true)));
    }
}

#[cfg(test)]
mod hook_tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    /// These tests drive process-global state -- `MODE`, `PUSHED`, `ARMED` and
    /// the global `model_base_url` -- so they cannot run beside each other.
    /// Without this they interleave: each one's pushes land on the other's stub
    /// and both assertions read the wrong list. Async mutex, because the guard
    /// is held across awaits.
    static TEST_LOCK: Lazy<tokio::sync::Mutex<()>> = Lazy::new(|| tokio::sync::Mutex::new(()));

    /// Minimal HTTP server: enough to record the scales we post, without
    /// pulling in a web framework for one test.
    ///
    /// It has to tolerate traffic that is not ours. This test publishes the
    /// stub's port through the GLOBAL config (model_base_url), which every other
    /// test in the process reads too — so the context-window probe's
    /// `GET /slots` lands here, with no body at all. It used to unwrap the JSON
    /// parse of that empty body: the spawned thread panicked, the listener went
    /// with it, and this test's own later requests then reached a closed port.
    /// That is the whole mechanism behind "passes alone, fails in a full run".
    /// So: never unwrap a request, keep serving, and record only the requests
    /// that carry a `scale` — the ones this test is actually about.
    fn stub(seen: std::sync::Arc<Mutex<Vec<f32>>>) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    let low = line.to_ascii_lowercase();
                    if let Some(v) = low.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                    if line == "\r\n" || line == "\n" {
                        break;
                    }
                }
                let mut buf = vec![0u8; len];
                if std::io::Read::read_exact(&mut reader, &mut buf).is_err() {
                    continue;
                }
                let body: serde_json::Value =
                    serde_json::from_slice(&buf).unwrap_or(serde_json::Value::Null);
                if let Some(scale) = body.get("scale").and_then(|v| v.as_f64()) {
                    seen.lock().push(scale as f32);
                }
                let mut out = stream;
                let payload = if body.is_null() { "{}".to_string() } else { body.to_string() };
                let _ = write!(
                    out,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
            }
        });
        port
    }

    /// The auto lifecycle: arming costs nothing, the turn's FIRST request is
    /// the only one turned on, and it is turned off the moment it answers --
    /// with no tool call anywhere in the test, because the off must not depend
    /// on one. That dependency is the 2026-10-01 regression in the module doc:
    /// seven retries on the edited weights and the server left at 1.0.
    #[tokio::test]
    async fn auto_is_on_for_the_turns_first_request_and_off_when_it_answers() {
        let _serial = TEST_LOCK.lock().await;
        let seen = std::sync::Arc::new(Mutex::new(Vec::<f32>::new()));
        let port = stub(seen.clone());
        // Restored at the end: the endpoint is global, so leaving it pointed at
        // this stub makes every test that runs afterwards probe a dead port.
        let previous_base_url = crate::utils::config::get_config().model_base_url;
        crate::utils::config::update_config(|c| {
            c.model_base_url = format!("http://127.0.0.1:{port}");
        });
        *PUSHED.lock() = None;
        *ARMED.lock() = false;

        set_mode(Mode::Auto).await.unwrap(); // entering auto parks it off
        on_user_turn().await;                // arms only
        assert_eq!(*seen.lock(), vec![0.0], "arming must cost no request");

        let on = begin_request().await;      // 1.0
        assert!(on, "the turn's first request gets the edit");
        end_request(on).await;               // 0.0  (no tool call involved)
        assert_eq!(*seen.lock(), vec![0.0, 1.0, 0.0]);

        // Every later request of the same turn: the latch is spent, so the
        // switch is not touched at all -- including the repetition retries that
        // used to run on the edited weights.
        for _ in 0..3 {
            let again = begin_request().await;
            assert!(!again, "only the first request of a turn is armed");
            end_request(again).await;
        }
        assert_eq!(*seen.lock(), vec![0.0, 1.0, 0.0], "no further requests");

        // Pinned off: arming does nothing, so neither does the request pair.
        // The mode command itself always goes out, even though we believe the
        // server is already at 0.0 -- a user command is never answered from our
        // guess (see push_inner).
        set_mode(Mode::Off).await.unwrap();
        on_user_turn().await;
        assert!(!begin_request().await);
        assert_eq!(*seen.lock(), vec![0.0, 1.0, 0.0, 0.0]);

        crate::utils::config::update_config(|c| c.model_base_url = previous_base_url);
    }

    /// `/ab on` must reach the server even when we already think it is on.
    ///
    /// The 2026-11-20 bug: the server had been restarted under a session whose
    /// `PUSHED` still read 1.0, so it was back at `GLM53_AB_SCALE=0` while we
    /// believed it was edited. `set_mode(On)` hit the already-there shortcut,
    /// returned Ok without sending a byte, and the REPL printed "/ab mode: on"
    /// over a server at 0.0 -- the model went on refusing and the switch looked
    /// broken. Nothing in the process can detect that state, which is exactly
    /// why the command may not be cached.
    #[tokio::test]
    async fn an_explicit_mode_command_is_never_answered_from_the_cache() {
        let _serial = TEST_LOCK.lock().await;
        let seen = std::sync::Arc::new(Mutex::new(Vec::<f32>::new()));
        let port = stub(seen.clone());
        let previous_base_url = crate::utils::config::get_config().model_base_url;
        crate::utils::config::update_config(|c| {
            c.model_base_url = format!("http://127.0.0.1:{port}");
        });

        // Stand in for the stale belief: we think the server is at 1.0. It is
        // not -- a restart put it back to 0.0 -- but nothing told us.
        *PUSHED.lock() = Some(1.0);
        *ARMED.lock() = false;

        set_mode(Mode::On).await.unwrap();
        assert_eq!(*seen.lock(), vec![1.0], "the user typed it; it must be sent");

        // Same for the other direction, and for re-entering auto: both park the
        // server, and neither may be skipped on a belief.
        set_mode(Mode::Auto).await.unwrap();
        set_mode(Mode::Auto).await.unwrap();
        assert_eq!(*seen.lock(), vec![1.0, 0.0, 0.0]);

        // The per-request shortcut inside `auto` is still a shortcut: both
        // pushes of the pair are ours, within one turn.
        on_user_turn().await;
        let on = begin_request().await;
        end_request(on).await;
        on_user_turn().await;
        let on = begin_request().await;
        end_request(on).await;
        assert_eq!(*seen.lock(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        crate::utils::config::update_config(|c| c.model_base_url = previous_base_url);
    }

    /// `/ab on` typed while a turn is in flight must survive that turn ending.
    ///
    /// The off is armed at `begin_request` under the mode that held then. When
    /// the user pins the edit mid-turn, the pending off is a decision they have
    /// since overridden -- and because nothing re-pushes under `On`, letting it
    /// through strands the server off for good, with the REPL reading "on".
    /// That is the 2026-11-20 report: refusals that began mid-session and then
    /// would not go away no matter how often `/ab on` was typed.
    #[tokio::test]
    async fn pinning_the_edit_mid_turn_survives_that_turn_ending() {
        let _serial = TEST_LOCK.lock().await;
        let seen = std::sync::Arc::new(Mutex::new(Vec::<f32>::new()));
        let port = stub(seen.clone());
        let previous_base_url = crate::utils::config::get_config().model_base_url;
        crate::utils::config::update_config(|c| {
            c.model_base_url = format!("http://127.0.0.1:{port}");
        });
        *PUSHED.lock() = None;
        *ARMED.lock() = false;

        set_mode(Mode::Auto).await.unwrap(); // 0.0
        on_user_turn().await;
        let on = begin_request().await; // 1.0 — the turn is now in flight
        assert!(on);
        assert_eq!(*seen.lock(), vec![0.0, 1.0]);

        // The user pins it while that request is still being answered.
        set_mode(Mode::On).await.unwrap();

        // The turn answers. Its off must not undo the pin.
        end_request(on).await;
        assert_eq!(
            *seen.lock(),
            vec![0.0, 1.0, 1.0],
            "the pending off must not stomp a mode the user pinned mid-turn"
        );

        // And it stays on: under `On` nothing re-pushes, so a stomp here would
        // never be corrected.
        on_user_turn().await;
        let again = begin_request().await;
        assert!(!again, "On does not arm per-request");
        end_request(again).await;
        assert_eq!(*seen.lock(), vec![0.0, 1.0, 1.0], "still pinned on");

        set_mode(Mode::Auto).await.unwrap(); // park it for the next test
        crate::utils::config::update_config(|c| c.model_base_url = previous_base_url);
    }
}
