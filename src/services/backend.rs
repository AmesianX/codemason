//! Backend gate — routes model queries to the right service implementation.
//! Equivalent to codemason/services/backend.py.
//!
//!     llama.cpp server  ->  services/llamacpp.rs   (original, unchanged)
//!     vLLM server       ->  services/vllm.rs        (continuous_usage_stats etc.)
//!
//! Selection order:
//!   1. Explicit: config.backend / env CODEMASON_BACKEND = 'vllm' | 'llamacpp'.
//!      Set this when you configure the endpoint — you already know whether the
//!      URL is a vLLM or a llama-server.
//!   2. Fallback auto-detect (config.backend == 'auto', the default): probe the
//!      server once. /slots is llama.cpp-only; vLLM 404s it. The result is
//!      cached; a connection error is NOT cached (assume vLLM for that call,
//!      retry later).
//!
//! Callers import query_model from HERE.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::types::query::AssistantMessage;

use super::openai_common::QueryModelParams;

/// cached auto-detect result ('vllm' | 'llamacpp')
static DETECTED: Lazy<Mutex<Option<&'static str>>> = Lazy::new(|| Mutex::new(None));

fn normalize(value: &str) -> &'static str {
    match value.trim().to_lowercase().as_str() {
        "llamacpp" | "llama.cpp" | "llama-server" | "llama_cpp" | "llama" => "llamacpp",
        "vllm" => "vllm",
        _ => "auto",
    }
}

/// Probe the configured server: /slots (200 + list) => llama.cpp, else vLLM.
async fn auto_detect(cfg: &crate::utils::config::Config) -> &'static str {
    if let Some(d) = *DETECTED.lock() {
        return d;
    }
    let client = get_shared_client();
    let mut req = client
        .get(format!("{}/slots", cfg.model_base_url))
        .timeout(Duration::from_secs_f64(5.0));
    if !cfg.model_api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", cfg.model_api_key));
    }
    match req.send().await {
        Ok(resp) => {
            let is_llamacpp = resp.status().as_u16() == 200
                && resp
                    .json::<serde_json::Value>()
                    .await
                    .map(|v| v.is_array())
                    .unwrap_or(false);
            let d = if is_llamacpp { "llamacpp" } else { "vllm" };
            *DETECTED.lock() = Some(d);
            d
        }
        Err(_) => {
            // Server unreachable — assume vLLM for this call but don't cache the
            // guess, so the next call re-probes once the server is up.
            "vllm"
        }
    }
}

/// Resolve the active backend: 'vllm' or 'llamacpp'.
pub async fn get_backend() -> &'static str {
    let cfg = crate::utils::config::get_config();
    let choice = normalize(&cfg.backend);
    if choice == "vllm" || choice == "llamacpp" {
        return choice;
    }
    auto_detect(&cfg).await
}

/// Neutral model-query entry point. Every call site imports this; it dispatches
/// to the backend-specific shim (query_llamacpp / query_vllm) based on the
/// detected/configured server.
pub async fn query_model(params: QueryModelParams) -> AssistantMessage {
    if get_backend().await == "llamacpp" {
        super::llamacpp::query_llamacpp(params).await
    } else {
        super::vllm::query_vllm(params).await
    }
}

// --- Shared HTTP client + connect retry --------------------------------------
// Both service backends stream chat completions over one HTTP client. Opening a
// NEW client (connection) per request meant a fresh TCP+TLS handshake for every
// turn — and the more handshakes, the higher the chance one transiently fails
// (SYN loss / TLS reset / momentary path blip), surfacing as a connect error
// and killing the turn (no retry). These helpers fix both halves: a persistent
// keep-alive client so a rapid loop REUSES one connection (few handshakes), and
// a connect-phase retry so a lone blip is absorbed instead of fatal.
// Env-tunable: CODEMASON_MODEL_CONNECT_RETRIES /
// CODEMASON_MODEL_RETRY_BACKOFF_BASE / CODEMASON_MODEL_RETRY_BACKOFF_CAP
// (the legacy LLAMACPP_* names for these still work).

fn env_raw(new: &str, old: &str) -> Option<String> {
    std::env::var(new).ok().or_else(|| std::env::var(old).ok())
}

/// Total time to keep retrying a transient failure, in seconds.
///
/// The count-based schedule alone gave up after a few seconds, which is right
/// for a person watching a dead server and wrong for the way this is mostly
/// run: started and left for hours. A server restart, a gateway coming back, a
/// reconnecting VPN — all of those take minutes, not seconds, and losing hours
/// of unattended work to a blip that healed 30 seconds later is the worst
/// possible trade.
///
/// The deadline is the last of utils::escalation's retry checkpoints — two
/// hours. That is not a guess at how long outages last, it is the cheaper way
/// to be wrong: calling a temporary outage permanent throws away everything the
/// run has done, while calling a permanent one temporary idles a machine nobody
/// is watching. The early attempts are still fast (half a second, then one, then
/// two), so a blip that clears immediately costs almost nothing.
/// Env: CODEMASON_MODEL_RETRY_BUDGET. 0 restores count-only behaviour.
pub(crate) fn http_retry_budget_secs() -> f64 {
    env_raw("CODEMASON_MODEL_RETRY_BUDGET", "LLAMACPP_RETRY_BUDGET")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v >= 0.0)
        .unwrap_or_else(crate::utils::escalation::retry_deadline_secs)
}

/// Statuses worth trying again: the server is there but not ready.
///
/// Deliberately NOT 500 — a deterministic application error would just burn the
/// whole budget re-asking a question that has already been answered. These four
/// are the ones a restarting backend, a gateway in front of it, or a rate
/// limiter actually emit while recovering.
pub(crate) fn is_retryable_status(code: u16) -> bool {
    matches!(code, 429 | 502 | 503 | 504)
}

/// Backoff for `attempt` (0-based), before jitter.
pub(crate) fn retry_delay_secs(attempt: u32, base: f64, cap: f64) -> f64 {
    cap.min(base * 2f64.powi(attempt as i32))
}

pub(crate) fn http_retry_cfg() -> (u32, f64, f64) {
    let i = |new: &str, old: &str, default: u32| {
        env_raw(new, old)
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(default)
    };
    let f = |new: &str, old: &str, default: f64| {
        env_raw(new, old)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(default)
    };
    (
        i("CODEMASON_MODEL_CONNECT_RETRIES", "LLAMACPP_CONNECT_RETRIES", 3),
        f("CODEMASON_MODEL_RETRY_BACKOFF_BASE", "LLAMACPP_RETRY_BACKOFF_BASE", 0.5),
        f("CODEMASON_MODEL_RETRY_BACKOFF_CAP", "LLAMACPP_RETRY_BACKOFF_CAP", 30.0),
    )
}

/// HTTP timeout for model streams. Large-context requests (e.g. ~1M-token
/// auto-compaction summaries) need far more than the old 600s: measured a real
/// 1M compaction at ~410s, and 600s cut it off mid-generation. Default 1800s;
/// env-overridable via CODEMASON_MODEL_HTTP_TIMEOUT (legacy
/// LLAMACPP_HTTP_TIMEOUT still honoured).
pub(crate) fn http_timeout_secs() -> f64 {
    env_raw("CODEMASON_MODEL_HTTP_TIMEOUT", "LLAMACPP_HTTP_TIMEOUT")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1800.0)
}

/// Session-lived reqwest::Client with a keep-alive pool. Reused across all LLM
/// calls so rapid turns share connections instead of re-handshaking each.
///
/// Substrate note: httpx set one `timeout` covering each phase (connect / read
/// gap / write); reqwest splits them, so the connect and per-read timeouts are
/// both set to the configured value here and no TOTAL deadline is imposed — the
/// same effective behavior (a stream stays alive as long as chunks keep
/// arriving within the window).
static SHARED_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    let t = Duration::from_secs_f64(http_timeout_secs());
    reqwest::Client::builder()
        .pool_max_idle_per_host(32)
        // Must stay BELOW the proxy's keep-alive timeout, with margin. This was
        // 90.0 — which is also reqwest's own default, and also the nginx
        // default the server in front of us runs. A dead heat. When the
        // server's timer won, the pool handed out a socket the server had
        // already closed and the next request died on a reset BEFORE it was
        // written, so it never reached the server: the access log showed 88/88
        // requests 200 and not one 499 or 5xx, while reqwest raised a bare
        // `kind: Request`. Verified 2026-10-14 against our own gateway: idle
        // 30s reuses fine, idle 95s and 120s both come back with the remote end
        // closed. nginx sends no `Keep-Alive: timeout=` header, so this cannot
        // be negotiated — it has to be a number we pick, and it has to be the
        // smaller one. Expiring first is what makes the race unreachable; the
        // exact value only sets how much slack absorbs clock skew and a slow
        // write. Keep this in step with the Python `keepalive_expiry`.
        .pool_idle_timeout(Duration::from_secs_f64(60.0))
        .connect_timeout(t)
        .read_timeout(t)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
});

pub fn get_shared_client() -> reqwest::Client {
    SHARED_CLIENT.clone()
}

// aclose_shared_client / the atexit hook have no Rust equivalent: reqwest's
// pool closes with the process and holds no interpreter-level resource that
// warns at exit (the Python atexit dance existed only to silence httpx's
// unclosed-transport warning).

/// Send `request` retrying ONLY the connect / response-header phase on
/// transient network errors. Once the response is returned, streaming errors
/// are NOT retried — content may already have been emitted, so a mid-stream
/// failure propagates to the caller. This is what lets a fast loop survive a
/// momentary connect blip instead of dying on the first connect error.
///
/// (reqwest's send() resolves when response headers arrive and can be retried
/// safely — the equivalent of retrying httpx's `__aenter__` — so the whole
/// pre-body phase is covered, matching the retryable set in the Python:
/// ConnectError/ConnectTimeout/PoolTimeout/Read/Write/RemoteProtocolError.)
///
/// `signal` is the turn's abort token (Esc / Ctrl+C). It is checked after every
/// failed attempt and raced against the backoff instead of sleeping through it.
/// Without it this ladder was unstoppable: the key cancels the token, the loop
/// never looked at it, and a dropped connection parked the session for the full
/// two-hour budget with no way out but killing the process. The stream BODY has
/// been racing the same token since the prefill fix (utils::generators::
/// abortable); the connect phase in front of it was simply never covered.
pub(crate) async fn send_with_retry<F>(
    build: F,
    signal: Option<&CancellationToken>,
) -> Result<reqwest::Response, String>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    let (retries, base, cap) = http_retry_cfg();
    let min_attempts = retries.max(1);
    let budget = Duration::from_secs_f64(http_retry_budget_secs());
    let started = std::time::Instant::now();
    // No initial value: the loop always assigns before anything reads it.
    let mut last_err;
    let mut attempt: u32 = 0;
    let mut announced = false;
    let mut reported_checkpoints = 0usize;
    loop {
        match build().send().await {
            Ok(resp) => {
                // A response whose STATUS says "not ready" is as transient as a
                // refused connection: a restarting backend or the gateway in
                // front of it answers 502/503 for exactly as long as it takes to
                // come back. The body has not been read yet, so retrying is
                // still clean.
                if !is_retryable_status(resp.status().as_u16()) {
                    if announced {
                        crate::utils::log::log_error(
                            &format!(
                                "model server recovered after {:.0}s / {attempt} attempts",
                                started.elapsed().as_secs_f64()
                            ),
                            Some("backend: retry recovered"),
                        );
                    }
                    return Ok(resp);
                }
                last_err = format!("HTTP {}", resp.status().as_u16());
            }
            // send() failed before any body was read → safe to retry cleanly.
            Err(e) => last_err = format!("{e:?}"),
        }
        attempt += 1;
        let elapsed = started.elapsed();
        // Stopped by the user. Nothing below is worth doing: no checkpoint
        // line, no backoff, no reconnect. Logged like the other two outcomes so
        // a post-mortem can tell "someone stopped it" from "it gave up" and
        // from "it is still going". query.rs checks the same token the moment
        // this returns, so the user sees the normal interrupt line rather than
        // a connection error.
        if signal.is_some_and(|s| s.is_cancelled()) {
            crate::utils::log::log_error(
                &format!(
                    "model server retry stopped by the user ({last_err}) after {:.0}s / {attempt} attempts",
                    elapsed.as_secs_f64()
                ),
                Some("backend: retry aborted"),
            );
            return Err(last_err);
        }
        // Keep going while either the count floor or the time budget still
        // allows it — the count covers a blip on a fast link, the budget covers
        // an outage measured in minutes.
        if attempt >= min_attempts && elapsed >= budget {
            // The retry wait is announced with cmprintln, i.e. to the TERMINAL
            // only. Scrollback is gone by morning and a post-mortem then has no
            // trace of a multi-minute wait at all — it looks like the process
            // simply stopped (no socket, every thread parked, nothing logged).
            // The outcome therefore goes to the error log too.
            crate::utils::log::log_error(
                &format!(
                    "model server never recovered ({last_err}) — gave up after {:.0}s / {attempt} attempts",
                    elapsed.as_secs_f64()
                ),
                Some("backend: retry budget exhausted"),
            );
            return Err(last_err);
        }
        let delay = retry_delay_secs(attempt - 1, base, cap);
        use rand::Rng;
        let jitter = rand::rngs::OsRng.gen_range(0.0..=delay.max(f64::MIN_POSITIVE));
        let mut wait = Duration::from_secs_f64(delay + jitter);
        // Never sleep past the budget once the count floor is met; otherwise a
        // 30s backoff could overshoot the deadline by half a minute.
        if attempt >= min_attempts {
            wait = wait.min(budget.saturating_sub(elapsed));
        }
        // Say it once at the start, then again at each checkpoint. A silent
        // multi-hour wait is indistinguishable from a hang, and the checkpoints
        // widen (5, 10, 15, 20, 30, 60, 90 min) so the reporting is dense while
        // someone might still be watching and sparse long after they have gone.
        if !budget.is_zero() {
            let crossed = crate::utils::escalation::retry_checkpoints_passed(elapsed.as_secs_f64());
            if !announced {
                announced = true;
                reported_checkpoints = crossed;
                // Also on disk: see the note at the budget-exhausted return.
                crate::utils::log::log_error(
                    &format!(
                        "model server not responding ({last_err}) — retrying for up to {:.0}s",
                        budget.as_secs_f64()
                    ),
                    Some("backend: retry started"),
                );
                crate::cmprintln!(
                    "  ⎿  {}",
                    crate::utils::i18n::trf(
                        "Model server not responding ({err}) — retrying for up to {mins} min. Press Esc or Ctrl+C to stop.",
                        &[
                            ("err", crate::utils::i18n::FmtArg::Str(last_err.chars().take(80).collect())),
                            ("mins", crate::utils::i18n::FmtArg::Int(
                                crate::utils::escalation::minutes_for_message(
                                    budget.as_secs_f64()) as i64)),
                        ]
                    )
                );
            } else if crossed > reported_checkpoints {
                reported_checkpoints = crossed;
                crate::utils::log::log_error(
                    &format!(
                        "still no response from the model server ({last_err}) — {:.0}s elapsed, {attempt} attempts",
                        elapsed.as_secs_f64()
                    ),
                    Some("backend: retry continuing"),
                );
                crate::cmprintln!(
                    "  ⎿  {}",
                    crate::utils::i18n::trf(
                        "Still no response — retrying ({mins} min left). Press Esc or Ctrl+C to stop.",
                        &[("mins", crate::utils::i18n::FmtArg::Int(
                            crate::utils::escalation::retry_minutes_left(elapsed.as_secs_f64())
                                as i64))]
                    )
                );
            }
        }
        // The wait itself has to be interruptible, not just the moment before
        // it: with a 30s backoff cap, checking only between attempts would
        // leave Esc dead for half a minute at a time.
        match signal {
            Some(sig) => {
                tokio::select! {
                    _ = tokio::time::sleep(wait) => {}
                    _ = sig.cancelled() => {
                        crate::utils::log::log_error(
                            &format!(
                                "model server retry stopped by the user ({last_err}) after {:.0}s / {attempt} attempts",
                                started.elapsed().as_secs_f64()
                            ),
                            Some("backend: retry aborted"),
                        );
                        return Err(last_err);
                    }
                }
            }
            None => tokio::time::sleep(wait).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Esc / Ctrl+C during the retry ladder must end it NOW, not at the end of
    /// the current backoff and not at the end of the two-hour budget.
    ///
    /// This is the bug the token was added for: the key cancelled the turn's
    /// token, nothing in this loop read it, and a dropped connection left the
    /// session retrying for two hours with no way out but killing the process.
    /// The assertion is on wall time because that is what the complaint was —
    /// an Err alone would also come back from simply giving up.
    #[tokio::test]
    async fn the_user_can_stop_the_retry_ladder() {
        // A port with nothing on it: every attempt fails at once, so the loop
        // spends essentially all its time in the backoff — exactly where the
        // old code was deaf.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let url = format!("http://127.0.0.1:{port}/");

        let token = CancellationToken::new();
        let stopper = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            stopper.cancel();
        });

        let started = std::time::Instant::now();
        let resp = send_with_retry(
            || get_shared_client().post(&url).body("{}"),
            Some(&token),
        )
        .await;
        let elapsed = started.elapsed();

        assert!(resp.is_err(), "a stopped ladder must not pretend it got a response");
        // The count floor alone (3 attempts, 0.5s + 1s backoff plus up to as
        // much again in jitter) runs to ~3s before the budget even applies.
        // Coming back under a second means the cancel was acted on, not waited
        // out. Derived from the constants so this keeps meaning something if
        // the backoff changes.
        let (min_attempts, base, cap) = http_retry_cfg();
        let count_only_min: f64 = (0..min_attempts.saturating_sub(1))
            .map(|i| retry_delay_secs(i, base, cap))
            .sum();
        assert!(
            elapsed.as_secs_f64() < count_only_min.max(0.5),
            "stop took {:.2}s — it waited out the backoff instead of being cancelled",
            elapsed.as_secs_f64()
        );
    }

    /// Against a REAL server that answers 503 and then recovers: the request
    /// must come back 200 without the caller seeing an error.
    ///
    /// The server is spun up here rather than gated behind an env var — a test
    /// that silently returns when the variable is missing is a test that never
    /// runs, and this one passed "instantly" in exactly that way once.
    #[tokio::test]
    async fn a_server_that_is_down_then_comes_back_is_ridden_out() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        // Refuse for longer than the COUNT floor alone could ever survive, so
        // passing this means the time budget is what carried it. Derived rather
        // than hardcoded: the worst case is every delay plus its maximum jitter
        // (jitter is uniform over 0..=delay), and deriving it keeps the test
        // honest if the backoff constants change — a hardcoded 5s would quietly
        // stop proving anything if the delays grew past it. It is also the
        // shortest outage that still proves it, which matters because this test
        // sets the floor for the whole suite's wall time.
        let (min_attempts, base, cap) = http_retry_cfg();
        let count_only_max: f64 = (0..min_attempts.saturating_sub(1))
            .map(|i| retry_delay_secs(i, base, cap) * 2.0)
            .sum();
        let outage = count_only_max + 0.5;
        let down_until = std::time::Instant::now() + Duration::from_secs_f64(outage);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let down = std::time::Instant::now() < down_until;
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = sock.read(&mut buf).await;
                    let resp: &[u8] = if down {
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
                    } else {
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
                    };
                    let _ = sock.write_all(resp).await;
                    let _ = sock.flush().await;
                });
            }
        });

        let url = format!("http://127.0.0.1:{port}/");
        let started = std::time::Instant::now();
        let resp = send_with_retry(|| get_shared_client().post(&url).body("{}"), None).await;
        let elapsed = started.elapsed();

        let resp = resp.expect("the retry loop gave up on a server that recovered");
        assert_eq!(resp.status().as_u16(), 200, "recovered server must yield its real answer");
        assert!(
            elapsed.as_secs_f64() >= count_only_max,
            "returned in {elapsed:?}, within what the count floor alone ({count_only_max}s) \
             could survive — this run does not show the budget doing anything"
        );
    }

    /// The retry deadline must come from the shared ladder, not a local number.
    /// The two ladders exist as one named pair precisely so nobody re-tunes one
    /// of them in isolation.
    #[test]
    fn the_retry_deadline_is_the_shared_ladder() {
        assert_eq!(
            http_retry_budget_secs(),
            crate::utils::escalation::retry_deadline_secs(),
            "the default deadline must be the ladder's, not a copy of the number"
        );
        assert_eq!(http_retry_budget_secs(), 7200.0, "two hours");
    }

    /// The budget is what lets an unattended run survive a restart. The old
    /// schedule gave up in a couple of seconds, which loses hours of work to a
    /// blip that healed on its own.
    #[test]
    fn the_retry_window_covers_a_restart_not_just_a_blip() {
        assert!(http_retry_budget_secs() >= 60.0, "seconds is not enough for a restart");
        let (_, base, cap) = http_retry_cfg();
        // Early attempts stay fast, so a blip that clears costs almost nothing.
        assert!(retry_delay_secs(0, base, cap) <= 1.0);
        // Backoff must actually grow, and then stop growing.
        assert!(retry_delay_secs(3, base, cap) > retry_delay_secs(1, base, cap));
        assert_eq!(retry_delay_secs(30, base, cap), cap, "must settle at the cap");
    }

    /// A restarting backend answers with a status, not a refused connection —
    /// so the status has to be part of "transient" or the budget never applies
    /// to the most common case.
    #[test]
    fn only_the_not_ready_statuses_are_retried() {
        for code in [429, 502, 503, 504] {
            assert!(is_retryable_status(code), "{code} means come back shortly");
        }
        // 500 is deliberately excluded: a deterministic app error would burn
        // the whole budget re-asking a question already answered.
        for code in [200, 400, 401, 404, 422, 500] {
            assert!(!is_retryable_status(code), "{code} must not be retried");
        }
    }
}
