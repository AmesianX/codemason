//! Vision capability — how many images the configured server accepts per request
//! (codemason/services/vision.py).
//!
//! The limit belongs to the server (vLLM --limit-mm-per-prompt), so ask the
//! server instead of trusting a config value: send one request carrying more
//! tiny images than any server allows. vLLM checks the count while parsing the
//! request, before it decodes a single image, and answers
//!
//!     400 "At most N image(s) may be provided in one prompt."
//!
//! N == 0 means the server runs text-only (--language-model-only), as does "is
//! not a multimodal model". The result is cached for the process. An answer that
//! says neither (connection error, 401, 5xx, a 400 for some other reason) is NOT
//! cached as a limit: the next call asks again once the server is up (same
//! policy as services/backend.rs), but not sooner than RETRY_AFTER — every
//! request and every image Read asks, and while the server is down each would
//! otherwise send a probe of its own.
//!
//! Env CODEMASON_VISION_MAX_IMAGES overrides the probe: 0 turns vision off, N
//! trusts N. (An env var, not a config.json field: the config file holds
//! measured user values and is not extended for this.) Only the vLLM backend
//! calls this.

use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::json;

use super::backend::get_shared_client;

/// 28x28 white PNG: one 28-pixel cell, the smallest size every processor takes
/// (Qwen-style processors reject an image under their 28-pixel factor, which
/// would read as "text-only"). Never decoded when the probe works as intended
/// (the count check rejects the request first); if a server accepts this many,
/// each costs only the minimum vision tokens.
const TINY_PNG: &str = "data:image/png;base64,\
iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAKElEQVR42mP8//8/A7UBEwMNwKiho4aOGjpq\
6Kiho4aOGjpq6OAzFADRYgM1sivIJAAAAABJRU5ErkJggg==";
const PROBE_COUNT: i64 = 65;
/// Before an inconclusive probe is repeated.
const RETRY_AFTER: Duration = Duration::from_secs(30);

static LIMIT: Lazy<Mutex<Option<i64>>> = Lazy::new(|| Mutex::new(None));
/// When the last probe that could not tell ran.
static LAST_INCONCLUSIVE: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));

/// vLLM's words for a model with no image input at all (multimodal/registry.py).
fn says_text_only(body: &str) -> bool {
    body.to_lowercase().contains("is not a multimodal model")
}

/// "At most N image" anywhere in the body -> N.
fn parse_limit(body: &str) -> Option<i64> {
    let lower = body.to_lowercase();
    let at = lower.find("at most ")?;
    let digits: String = lower[at + 8..].chars().take_while(|c| c.is_ascii_digit()).collect();
    let rest = &lower[at + 8 + digits.len()..];
    if digits.is_empty() || !rest.trim_start().starts_with("image") {
        return None;
    }
    digits.parse().ok()
}

/// Ask the server. None = could not tell (do not cache).
///
/// Sent the way every model request is (services/openai_common.rs): bearer key
/// when set, and the body sealed when model_aes_key is set — a gateway that
/// refuses plaintext would otherwise answer 400 and read as "no images". Error
/// bodies come back in the clear (deploy/encrypt_gateway.py).
async fn probe(cfg: &crate::utils::config::Config) -> Option<i64> {
    let mut content = vec![json!({"type": "text", "text": "."})];
    for _ in 0..PROBE_COUNT {
        content.push(json!({"type": "image_url", "image_url": {"url": TINY_PNG}}));
    }
    let body = json!({
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
    });
    let mut req = get_shared_client()
        .post(format!("{}/v1/chat/completions", cfg.model_base_url))
        .timeout(Duration::from_secs(30));
    if cfg.model_aes_key.is_empty() {
        req = req.json(&body);
    } else {
        let key = crate::utils::body_crypto::load_key(&cfg.model_aes_key).ok()?;
        req = req
            .header("Content-Type", "application/octet-stream")
            .header("X-Body-Encrypted", "aesgcm")
            .body(crate::utils::body_crypto::encrypt(&key, body.to_string().as_bytes()).into_bytes());
    }
    if !cfg.model_api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", cfg.model_api_key));
    }
    let resp = req.send().await.ok()?;
    let status = resp.status().as_u16();
    if status == 200 {
        return Some(PROBE_COUNT); // accepts at least this many
    }
    let text = resp.text().await.unwrap_or_default();
    if let Some(n) = parse_limit(&text) {
        return Some(n);
    }
    if (status == 400 || status == 422) && says_text_only(&text) {
        return Some(0);
    }
    None // 401/5xx/another 400 — cannot tell yet
}

/// Images the server accepts in one request; 0 = no vision.
pub async fn image_limit() -> i64 {
    if let Ok(v) = std::env::var("CODEMASON_VISION_MAX_IMAGES") {
        if let Ok(n) = v.trim().parse::<i64>() {
            return n.max(0);
        }
    }
    if let Some(n) = *LIMIT.lock() {
        return n;
    }
    if matches!(*LAST_INCONCLUSIVE.lock(), Some(t) if t.elapsed() < RETRY_AFTER) {
        return 0;
    }
    let cfg = crate::utils::config::get_config();
    match probe(&cfg).await {
        Some(n) => {
            *LIMIT.lock() = Some(n);
            n
        }
        None => {
            *LAST_INCONCLUSIVE.lock() = Some(Instant::now());
            0
        }
    }
}

/// Last result of `enabled_limit`, for sync callers: the sub-agent prefix (the
/// main prompt is refreshed first each turn) and the display, which runs in a
/// repaint where a probe would freeze the prompt while the server is down.
static ENABLED_LIMIT: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(0));

/// Images per request when the active backend is vLLM and its server takes
/// images; 0 otherwise.
pub async fn enabled_limit() -> i64 {
    let n = if super::backend::get_backend().await == "vllm" { image_limit().await } else { 0 };
    *ENABLED_LIMIT.lock() = n;
    n
}

/// What `enabled_limit` last returned (0 before it ever ran).
pub fn last_enabled_limit() -> i64 {
    *ENABLED_LIMIT.lock()
}

#[cfg(test)]
mod tests {
    use super::{parse_limit, says_text_only};

    #[test]
    fn only_text_only_wording_means_no_images() {
        assert!(says_text_only(r#"{"message":"Qwen/x is not a multimodal model"}"#));
        assert!(!says_text_only("height:1 and width:1 must be larger than factor:28"));
    }

    #[test]
    fn parses_vllm_limit_message() {
        let body = r#"{"error":{"message":"At most 8 image(s) may be provided in one prompt.","code":400}}"#;
        assert_eq!(parse_limit(body), Some(8));
        assert_eq!(parse_limit("At most 0 image(s) may be provided"), Some(0));
        assert_eq!(parse_limit("At most 3 video(s) may be provided"), None);
        assert_eq!(parse_limit("maximum context length"), None);
    }
}
