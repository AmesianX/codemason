//! What the server is actually about to re-read, asked of the server.
//!
//! Before the first output token arrives there is nothing on screen but a
//! spinner and a clock, and a warm request (answer in ~2s) looks exactly like
//! one whose cached prefix broke. On this deployment the difference is 55
//! seconds for a 73k conversation and 175 for a 233k one, so "is it prefilling
//! or is it stuck" is the question every long wait raises and none of them
//! answer.
//!
//! vLLM already knows. Two counters move the moment a request is SCHEDULED,
//! seconds before any token comes back:
//!
//!   vllm:prefix_cache_queries_total   tokens this prompt asked about
//!   vllm:prefix_cache_hits_total      tokens it could serve from cache
//!
//! Their difference is the work that has to be computed — the server's own
//! number, not a guess from our side of the wire. Measured 2026-09-07 on a
//! deliberately cold 59,384-token prompt: queries +59,384, hits +0, scheduled
//! 4.7s in, answered at 44.8s. Exactly the shape a stalled turn has.
//!
//! Caveat, and the reason this only ever informs a display: the counters are
//! global. With another request in flight the delta belongs to whichever was
//! scheduled, so the number can be wrong when the server is shared. It is a
//! diagnostic, never an input to behaviour.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static PROBING: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

fn metrics_url_and_key() -> Option<(String, String)> {
    let cfg = crate::utils::config::get_config();
    let base = cfg.model_base_url.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return None;
    }
    Some((format!("{base}/metrics"), cfg.model_api_key.clone()))
}

/// (queried, hits) cumulative, or None when the server has no such counters
/// (llama.cpp) or is unreachable.
async fn read_counters(url: &str, key: &str) -> Option<(i64, i64)> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build().ok()?;
    let mut req = client.get(url);
    if !key.is_empty() {
        req = req.bearer_auth(key);
    }
    let body = req.send().await.ok()?.text().await.ok()?;
    let (mut q, mut h) = (None, None);
    for line in body.lines() {
        let val = || line.rsplit(' ').next().and_then(|v| v.parse::<f64>().ok());
        if line.starts_with("vllm:prefix_cache_queries_total") {
            q = val();
        } else if line.starts_with("vllm:prefix_cache_hits_total") {
            h = val();
        }
    }
    Some((q? as i64, h? as i64))
}

/// Watch for THIS request being scheduled and publish how much of it the
/// server must compute. Returns immediately; the work runs in the background
/// and stops on its own once the answer is known or the deadline passes.
pub fn probe_next_request() {
    // One at a time: concurrent probes would read each other's deltas.
    if PROBING.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async move {
        let done = || PROBING.store(false, Ordering::SeqCst);
        let Some((url, key)) = metrics_url_and_key() else {
            return done();
        };
        let Some((q0, h0)) = read_counters(&url, &key).await else {
            return done(); // no counters: the client-side estimate stands
        };
        // Scheduling lands within a few seconds; past that the request is
        // queued behind something else and the delta would not be ours.
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(700)).await;
            // The answer already started: the number is no longer interesting.
            if crate::utils::tokens::output_started() {
                break;
            }
            let Some((q, h)) = read_counters(&url, &key).await else { break };
            if q > q0 {
                let (queried, cached) = (q - q0, h - h0);
                let compute = queried - cached;
                crate::utils::tokens::set_pending_prefill_measured(compute, cached);
                // A cold or badly-degraded prefix is the single most expensive
                // thing that happens without saying so. Record it where an
                // unattended run leaves a trace: at ~1,335 tok/s a 73k re-read
                // is 55 seconds of silence that otherwise looks like a hang,
                // and afterwards nothing in the transcript says it happened.
                let hit = if queried > 0 { 100.0 * cached as f64 / queried as f64 } else { 100.0 };
                if compute > 20_000 && hit < 50.0 {
                    crate::utils::log::log_error(
                        &format!(
                            "cold prefill: {compute} of {queried} prompt tokens must be recomputed \
                             ({hit:.1}% served from cache) — about {:.0}s before the first token",
                            compute as f64 / 1335.0
                        ),
                        Some("prefill_probe: vllm /metrics"),
                    );
                }
                crate::utils::log::debug_log(
                    "PREFILL",
                    Some(&serde_json::json!({
                        "queried": queried, "cached": cached, "compute": compute,
                        "hit_percent": hit,
                    })),
                );
                break;
            }
        }
        done();
    });
}
