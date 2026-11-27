//! The abliteration edit, asked for per request.
//!
//! The served checkpoint has its refusal behaviour intact; the edit that
//! removes it is a rank-24 adapter the server adds at `o_proj`, and a request
//! gets it by carrying `{"vllm_xargs": {"ab_scale": 1}}` (see the lane's
//! AB-SWITCH.md). It applies to that request's tokens and nobody else's. The
//! reason it is a per-request choice and not a checkpoint choice: removing
//! refusal also removes the model's ability to say a thing does not exist,
//! because declining a request and denying a premise run through the same
//! direction in the weights. Measured on the served pair — 4/60 fabrications
//! with it off, 18-22/60 with it on. So the model that judges a tool result, a
//! link or a diff should be the restored one.
//!
//! Which leaves exactly one REQUEST that needs the edit: the first one of a
//! user turn, the one that decides whether to accept. Hence `auto`:
//!
//!     user turn starts       -> arm (no request)
//!     turn's first request   -> ab_scale 1
//!     every request after    -> ab_scale 0 (retries, tool loop, sub-agents)
//!     next user turn         -> arm again
//!
//! Zero trust (2026-11-27). This module used to drive a SERVER-WIDE switch,
//! `POST /ab {"scale":0|1}`: on before the turn's first request, off after it.
//! That switch was the engine's default for every request that did not carry
//! its own scale, i.e. every other client's — one user's `/ab on` changed
//! everyone's model, and measured on the live lane it did. A client cannot be
//! trusted to leave a shared knob alone (this code is MIT; anyone can send
//! anything), so the server no longer has one (`POST /ab` answers 410) and this
//! client never writes server state. It only decides what to put on ITS OWN
//! requests. That also retires the failure the old design was built around —
//! the edit left on for hours because nothing turned it off (2026-10-01): there
//! is no "off" to forget any more, because there is no state left behind.
//!
//! `on` and `off` exist to pin the choice for A/B measurement, not for daily
//! use. Only the conversation loop's requests carry the key; compaction and
//! helper calls carry none and get the server default (the original weights).

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
/// edit every request that needs it gets refused. A server without the plugin
/// (or an llama.cpp backend) ignores the key, so this default costs nothing
/// where the feature does not exist. Override with `CODEMASON_AB=off|on|auto`.
static MODE: Lazy<Mutex<Mode>> = Lazy::new(|| {
    let from_env = std::env::var("CODEMASON_AB").ok().and_then(|v| Mode::parse(&v));
    Mutex::new(from_env.unwrap_or(Mode::Auto))
});
/// Set when a user turn starts, spent by that turn's first request.
static ARMED: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

pub fn mode() -> Mode {
    *MODE.lock()
}

/// Set the mode. Local only: nothing is sent anywhere, now or later — the
/// choice rides on each request (`request_scale`).
///
/// A pending arm is dropped: a mode typed mid-turn is the user's latest word,
/// and a stale arm from the mode before it must not fire on their behalf.
pub fn set_mode(m: Mode) {
    *MODE.lock() = m;
    *ARMED.lock() = false;
}

/// A user turn is starting: under `auto`, the turn's FIRST request gets the
/// edit. Costs nothing — it only sets a latch that `request_scale` spends.
pub async fn on_user_turn() {
    if mode() == Mode::Auto {
        *ARMED.lock() = true;
    }
}

/// The `ab_scale` to put on the conversation request about to be sent.
///
/// `on` 1, `off` 0, `auto` 1 for the turn's first request and 0 for every one
/// after it. Always an explicit value, never "whatever the server defaults
/// to": the default is not ours, so the request says what it wants.
pub fn request_scale() -> f32 {
    match mode() {
        Mode::On => 1.0,
        Mode::Off => 0.0,
        Mode::Auto => {
            let mut armed = ARMED.lock();
            let was = *armed;
            *armed = false;
            if was {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Read the server's DEFAULT (what a request with no `ab_scale` gets), for
/// `/ab` with no argument. Read-only: `GET /ab`. The server's default is fixed
/// at its startup; nothing here can change it.
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
/// server at 0.0, the fix was to believe the server instead of the client's cache, and
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

    /// A server from before the per-token change carries no
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

    /// These tests drive process-global state -- `MODE`, `ARMED` and the
    /// global `model_base_url` -- so they cannot run beside each other. Async
    /// mutex, because the guard is held across awaits.
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

    /// Zero trust (2026-11-27): the client must never write the server's
    /// switch. `POST /ab` set the engine-wide default, which every request
    /// without its own scale follows -- one client's `/ab on` changed every
    /// other client's model. The server now refuses it; the client must not
    /// rely on it either. Any request carrying a `scale` reaching the stub is a
    /// failure, whatever the mode and whatever the turn does.
    ///
    /// Failed on the code before this change: `[1.0, 0.0, 0.0, 1.0]`.
    #[tokio::test]
    async fn changing_the_mode_never_writes_the_server() {
        let _serial = TEST_LOCK.lock().await;
        let seen = std::sync::Arc::new(Mutex::new(Vec::<f32>::new()));
        let port = stub(seen.clone());
        let previous_base_url = crate::utils::config::get_config().model_base_url;
        crate::utils::config::update_config(|c| {
            c.model_base_url = format!("http://127.0.0.1:{port}");
        });
        for m in [Mode::On, Mode::Off, Mode::Auto, Mode::On, Mode::Auto] {
            set_mode(m);
            on_user_turn().await;
            let _ = request_scale();
            let _ = request_scale();
        }
        // Give any stray background push time to land before we look.
        tokio::time::sleep(Duration::from_millis(200)).await;
        crate::utils::config::update_config(|c| c.model_base_url = previous_base_url);
        set_mode(Mode::Auto);
        assert!(seen.lock().is_empty(), "client wrote the server switch: {:?}", seen.lock());
    }

    /// `auto`: the turn's first request only, every later one explicitly 0 --
    /// retries, the tool loop and sub-agents included -- and the next turn
    /// arms again. No tool call anywhere: nothing depends on one.
    #[tokio::test]
    async fn auto_gives_the_edit_to_the_turns_first_request_only() {
        let _serial = TEST_LOCK.lock().await;
        set_mode(Mode::Auto);
        assert_eq!(request_scale(), 0.0, "nothing armed before a user turn");
        on_user_turn().await;
        assert_eq!(request_scale(), 1.0, "the deciding request");
        assert_eq!(request_scale(), 0.0, "the request after it");
        assert_eq!(request_scale(), 0.0);
        on_user_turn().await;
        assert_eq!(request_scale(), 1.0, "next turn arms again");
        assert_eq!(request_scale(), 0.0);
        set_mode(Mode::Auto);
    }

    /// `on` and `off` pin every request, and a user turn does not arm them.
    #[tokio::test]
    async fn on_and_off_pin_every_request() {
        let _serial = TEST_LOCK.lock().await;
        set_mode(Mode::On);
        on_user_turn().await;
        assert_eq!([request_scale(), request_scale(), request_scale()], [1.0; 3]);
        set_mode(Mode::Off);
        on_user_turn().await;
        assert_eq!([request_scale(), request_scale()], [0.0; 2]);
        set_mode(Mode::Auto);
    }

    /// A mode typed mid-turn wins over an arm left by the mode before it.
    #[tokio::test]
    async fn a_mode_typed_mid_turn_drops_the_pending_arm() {
        let _serial = TEST_LOCK.lock().await;
        set_mode(Mode::Auto);
        on_user_turn().await;
        set_mode(Mode::Off);
        assert_eq!(request_scale(), 0.0);
        set_mode(Mode::Auto);
        assert_eq!(request_scale(), 0.0, "the old arm must not fire after switching back");
        set_mode(Mode::Auto);
    }
}
