//! WebSearch Tool for CodeMason
//! Equivalent to codemason/tools/web_search_tool.py (a reconstruction of
//! src/tools/WebSearchTool/ on the local substrate).
//!
//! The original WebSearchTool is NOT a single literal search: its call() spins
//! up a sub-LLM that drives Anthropic's server-side web_search tool, expanding
//! the intent into several queries and reading page CONTENT. codemason
//! reconstructs that structure locally:
//!
//!   1. query expansion   - one small LLM call rewrites the intent into N
//!                          focused queries (falls back to the literal query).
//!   2. serialized SERP   - SearXNG-first (JSON API, X-API-Key), ddgs-style
//!                          scrape fallback across the merged backend list,
//!                          throttled/backed-off per config.websearch_*.
//!   3. top-K page fetch  - highest-ranked results reduced to short extracts
//!                          via the web_fetch substrate.
//!
//! Domain filtering is two-layer (site:/-site: operators for recall + a
//! hostname filter for precision). Schema, validation (incl. the
//! allowed/blocked mutual exclusion), the description and the trailing
//! REMINDER line are ported faithfully.
//!
//! Substrate divergence (documented): the Python delegated the fallback SERP
//! scrape to the `ddgs` library. This port implements the scrape natively for
//! the engines the config names (duckduckgo html endpoint, brave, mojeek;
//! google is attempted but is generally consent-walled — the same practical
//! outcome ddgs has). Engines are queried in list order and results
//! aggregated/deduped until max_results, standing in for ddgs's merge-and-rank;
//! the retry/backoff/throttle posture around the scrape is identical. The
//! all-engines-empty error string is ddgs's verbatim "No results found.".
//! The result cache is the same JSON shape ({"ts": ..., "results": [...]})
//! and the cache KEY replicates json.dumps(sort_keys=True) byte-for-byte, so
//! caches written by the Python version are read by this one and vice versa.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::services::backend::query_model;
use crate::services::openai_common::{py_json_dumps, QueryModelParams, QueryOptions};
use crate::tools::base::{Tool, ToolResult, ToolStream, PermissionResult};
use crate::types::messages::ContentBlock;
use crate::types::query::{Message, ToolUseContext, ValidationResult};
use crate::utils::config::{get_config, Config, CONFIG_DIR};
use crate::utils::formatting::{colorize, colors};
use crate::utils::messages::create_user_message_text;

pub const WEB_SEARCH_TOOL_NAME: &str = "WebSearch";

const MAX_RESULTS_PER_QUERY: usize = 8;
// Internal caps (not user-tunable): final merged result cap, how many top
// results get a page fetch, and per-page extract length.
const MAX_MERGED: usize = 12;
// 3 x 600 chars gave the model ~1.8k of page body for an entire search, and
// the extract is taken from the top of the page — on most sites that budget is
// spent entirely on nav chrome before any real content.
const FETCH_TOP_K: usize = 5;
const EXTRACT_CHARS: usize = 4000;

/// One search hit (the Python dict {'title','url','body',['extract'],['extract_error']}).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchHit {
    #[serde(default)]
    pub title: String,
    pub url: String,
    #[serde(default)]
    pub body: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_error: Option<String>,
}

/// codemason Config for WebSearch settings, read at call time (config.json +
/// env override). The Python fell back to a defaults instance if config was
/// unavailable; the Rust get_config() cannot fail, so the fallback is moot.
fn ws_config() -> Config {
    get_config()
}

/// (url, timeout, lang, api_key) for the SearXNG primary backend.
fn searxng_conf() -> (String, f64, String, String) {
    let c = ws_config();
    (
        c.searxng_url.trim().to_string(),
        c.searxng_timeout,
        if c.searxng_lang.is_empty() { "auto".to_string() } else { c.searxng_lang },
        c.searxng_api_key.trim().to_string(),
    )
}

/// Cache root, under codemason's config home by default (relocatable via
/// config.websearch_cache_dir / CODEMASON_WEBSEARCH_CACHE_DIR / CODEMASON_CONFIG_DIR).
fn cache_dir() -> PathBuf {
    let dir = ws_config().websearch_cache_dir;
    if !dir.is_empty() {
        PathBuf::from(dir)
    } else {
        CONFIG_DIR.join("cache").join("websearch")
    }
}

/// Cache key: sha256 of json.dumps({'q','a','b'}, sort_keys=True), first 32
/// hex chars — byte-identical to the Python (keys inserted in sorted order
/// a,b,q; py_json_dumps gives the ", "/": " separators + ensure_ascii).
fn cache_key(query: &str, allowed: &[String], blocked: &[String]) -> String {
    use sha2::Digest;
    let mut a: Vec<String> = allowed.iter().map(|d| d.to_lowercase()).collect();
    a.sort();
    let mut b: Vec<String> = blocked.iter().map(|d| d.to_lowercase()).collect();
    b.sort();
    let mut obj = serde_json::Map::new();
    obj.insert("a".to_string(), json!(a));
    obj.insert("b".to_string(), json!(b));
    obj.insert("q".to_string(), json!(query));
    let payload = py_json_dumps(&Value::Object(obj));
    let digest = sha2::Sha256::digest(payload.as_bytes());
    hex::encode(digest)[..32].to_string()
}

fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Return cached results if a fresh entry exists, else None. Any I/O or parse
/// error degrades to a miss (never raises into the search path).
fn cache_get(key: &str) -> Option<Vec<SearchHit>> {
    let ttl = ws_config().websearch_cache_ttl;
    if ttl <= 0 {
        return None;
    }
    let path = cache_dir().join(format!("{key}.json"));
    let raw = std::fs::read_to_string(path).ok()?;
    let obj: Value = serde_json::from_str(&raw).ok()?;
    let ts = obj.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if now_secs() - ts > ttl as f64 {
        return None;
    }
    let results = obj.get("results")?;
    let hits: Vec<SearchHit> = serde_json::from_value(results.clone()).ok()?;
    if hits.is_empty() {
        None
    } else {
        Some(hits)
    }
}

/// Persist a non-empty result set. Best-effort: failures are swallowed.
fn cache_put(key: &str, results: &[SearchHit]) {
    if ws_config().websearch_cache_ttl <= 0 || results.is_empty() {
        return;
    }
    let write = || -> std::io::Result<()> {
        let d = cache_dir();
        std::fs::create_dir_all(&d)?;
        // Per-process temp name. The cache is shared by every codemason process on
        // this machine, so two of them missing on the SAME query wrote the same
        // `<key>.json.tmp` at the same time: each truncated the other's bytes and
        // whichever renamed first published a half-written entry, which then
        // failed to parse for the rest of its TTL — a cache that quietly stopped
        // being a cache. The rename is still the atomic publish; it is only the
        // staging file that has to be private.
        let tmp = d.join(format!("{key}.json.{}.tmp", std::process::id()));
        let body = serde_json::to_string(&json!({"ts": now_secs(), "results": results}))
            .unwrap_or_default();
        if let Err(e) = std::fs::write(&tmp, body) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, d.join(format!("{key}.json"))) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    };
    let _ = write();
}

/// Ported from upstream getWebSearchPrompt (prompt.ts) — EXCEPT the trailing
/// "Search in the language that fits the topic" block, which is a codemason
/// addition (see the Python module for the rationale).
fn get_description() -> String {
    let current_month_year = chrono::Local::now().format("%B %Y").to_string();
    format!(
        r#"
- Allows the assistant to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond the model's knowledge cutoff
- Runs a web search and fetches the most relevant pages to ground the answer

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported to include or block specific websites

IMPORTANT - Use the correct year in search queries:
  - The current month is {current_month_year}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation" with the current year, NOT last year

IMPORTANT - Search in the language that fits the topic:
  - Search in the language most likely to surface the best sources. A topic rooted in one country or language — its people, media, news, products, local services, regulations — is documented in that language first, and the best sources may never be translated. Search in it.
  - Transliterating a name into the Latin alphabet usually loses those sources: the local spelling is what the local pages actually use, so search the name as it is written at home.
  - When two languages could both help, issue BOTH queries so coverage runs in parallel instead of committing to one.
  - For globally English or technical topics (programming, standards, international products), English is usually best — do NOT force a local language where English clearly wins.
"#
    )
}

/// Translate domain filters into SERP operators the engines understand.
///
/// allowed_domains -> `(site:a.com OR site:b.com) <query>`, blocked_domains ->
/// `<query> -site:a.com ...`. allowed/blocked are mutually exclusive (rejected
/// at validation), but applying both independently here is harmless.
fn build_query(query: &str, allowed: &[String], blocked: &[String]) -> String {
    let mut q = query.to_string();
    if !allowed.is_empty() {
        let sites = allowed
            .iter()
            .filter(|d| !d.is_empty())
            .map(|d| format!("site:{}", d.to_lowercase().trim_start_matches('.')))
            .collect::<Vec<_>>()
            .join(" OR ");
        if !sites.is_empty() {
            q = format!("({sites}) {q}");
        }
    }
    if !blocked.is_empty() {
        for d in blocked.iter().filter(|d| !d.is_empty()) {
            q.push_str(&format!(" -site:{}", d.to_lowercase().trim_start_matches('.')));
        }
    }
    q
}

fn host_of(u: &str) -> String {
    url::Url::parse(u)
        .ok()
        .and_then(|p| p.host_str().map(|h| h.to_lowercase()))
        .unwrap_or_default()
}

/// host == domain or a subdomain of it (suffix match on a label boundary).
fn domain_match(host: &str, domains: &[String]) -> bool {
    let host = host.to_lowercase();
    for d in domains {
        let d = d.to_lowercase();
        let d = d.trim_start_matches('.');
        if !d.is_empty() && (host == d || host.ends_with(&format!(".{d}"))) {
            return true;
        }
    }
    false
}

/// Query a self-hosted SearXNG instance's JSON API.
///
/// Returns (results, error): (Some(list), None) when the instance answered
/// (list may be empty); (None, Some(err)) when unavailable — the caller's
/// signal to fall back to the scrape path.
async fn searxng_search(
    query: &str,
    max_results: usize,
) -> (Option<Vec<SearchHit>>, Option<String>) {
    let (url_base, timeout, lang, api_key) = searxng_conf();
    if url_base.is_empty() {
        return (None, Some("searxng disabled".to_string()));
    }
    let base = url_base.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs_f64(timeout.max(0.1)))
        .build();
    let client = match client {
        Ok(c) => c,
        Err(e) => return (None, Some(format!("searxng unavailable: {e}"))),
    };
    let mut req = client
        .get(format!("{base}/search"))
        .query(&[("q", query), ("format", "json"), ("safesearch", "0"), ("language", &lang)])
        .header("User-Agent", "codemason-websearch/1.0")
        .header("Accept", "application/json");
    // API-key auth: the reverse proxy in front of a public instance gates on
    // this shared-secret header and 403s anything without it.
    if !api_key.is_empty() {
        req = req.header("X-API-Key", api_key);
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return (None, Some(format!("searxng unavailable: {e}"))),
    };
    if resp.status().as_u16() != 200 {
        return (None, Some(format!("searxng HTTP {}", resp.status().as_u16())));
    }
    let raw = match resp.text().await {
        Ok(t) => t,
        Err(e) => return (None, Some(format!("searxng unavailable: {e}"))),
    };
    // A JSON-format-disabled instance answers 200 with an HTML page; the parse
    // failure correctly reports unavailable -> scrape fallback.
    let data: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return (None, Some(format!("searxng unavailable: {e}"))),
    };

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for h in data.get("results").and_then(|r| r.as_array()).unwrap_or(&Vec::new()) {
        let u = h.get("url").and_then(|v| v.as_str()).unwrap_or("");
        if u.is_empty() || !seen.insert(u.to_string()) {
            continue;
        }
        out.push(SearchHit {
            title: h.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            url: u.to_string(),
            body: h.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            extract: None,
            extract_error: None,
        });
        if out.len() >= max_results {
            break;
        }
    }
    (Some(out), None)
}

// --- ddgs-equivalent scrape fallback ---------------------------------------

fn scrape_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        .build()
        .unwrap_or_default()
}

fn text_of(el: scraper::ElementRef<'_>) -> String {
    el.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ")
}

/// DuckDuckGo html endpoint (the primary engine of the ddgs library).
async fn engine_duckduckgo(query: &str, max_results: usize) -> Result<Vec<SearchHit>, String> {
    let client = scrape_client();
    let resp = client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query), ("b", "")])
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("duckduckgo HTTP {}", resp.status().as_u16()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let doc = scraper::Html::parse_document(&body);
    let sel_result = scraper::Selector::parse("div.result").map_err(|e| e.to_string())?;
    let sel_a = scraper::Selector::parse("a.result__a").map_err(|e| e.to_string())?;
    let sel_snip = scraper::Selector::parse(".result__snippet").map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in doc.select(&sel_result) {
        let Some(a) = r.select(&sel_a).next() else { continue };
        let href = a.value().attr("href").unwrap_or("");
        // DDG wraps targets as //duckduckgo.com/l/?uddg=<encoded>&...
        let target = if href.contains("uddg=") {
            url::Url::parse(&format!("https:{href}"))
                .ok()
                .and_then(|u| {
                    u.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.into_owned())
                })
                .unwrap_or_else(|| href.to_string())
        } else {
            href.to_string()
        };
        if target.is_empty() || !target.starts_with("http") {
            continue;
        }
        let snippet = r.select(&sel_snip).next().map(text_of).unwrap_or_default();
        out.push(SearchHit {
            title: text_of(a),
            url: target,
            body: snippet,
            extract: None,
            extract_error: None,
        });
        if out.len() >= max_results {
            break;
        }
    }
    Ok(out)
}

/// Mojeek (no-JS-friendly HTML results).
async fn engine_mojeek(query: &str, max_results: usize) -> Result<Vec<SearchHit>, String> {
    let client = scrape_client();
    let resp = client
        .get("https://www.mojeek.com/search")
        .query(&[("q", query)])
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("mojeek HTTP {}", resp.status().as_u16()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let doc = scraper::Html::parse_document(&body);
    let sel_li = scraper::Selector::parse("ul.results-standard li").map_err(|e| e.to_string())?;
    let sel_a = scraper::Selector::parse("h2 a, a.title, a.ob").map_err(|e| e.to_string())?;
    let sel_p = scraper::Selector::parse("p.s").map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for li in doc.select(&sel_li) {
        let Some(a) = li.select(&sel_a).next() else { continue };
        let href = a.value().attr("href").unwrap_or("");
        if !href.starts_with("http") {
            continue;
        }
        out.push(SearchHit {
            title: text_of(a),
            url: href.to_string(),
            body: li.select(&sel_p).next().map(text_of).unwrap_or_default(),
            extract: None,
            extract_error: None,
        });
        if out.len() >= max_results {
            break;
        }
    }
    Ok(out)
}

/// Brave search HTML (may be bot-walled; failures are just skipped).
async fn engine_brave(query: &str, max_results: usize) -> Result<Vec<SearchHit>, String> {
    let client = scrape_client();
    let resp = client
        .get("https://search.brave.com/search")
        .query(&[("q", query), ("source", "web")])
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("brave HTTP {}", resp.status().as_u16()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let doc = scraper::Html::parse_document(&body);
    let sel_r = scraper::Selector::parse("div.snippet[data-type=\"web\"], div#results .snippet")
        .map_err(|e| e.to_string())?;
    let sel_a = scraper::Selector::parse("a").map_err(|e| e.to_string())?;
    let sel_desc =
        scraper::Selector::parse(".snippet-description, .snippet-content").map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in doc.select(&sel_r) {
        let Some(a) = r.select(&sel_a).find(|a| {
            a.value().attr("href").map(|h| h.starts_with("http")).unwrap_or(false)
        }) else {
            continue;
        };
        let href = a.value().attr("href").unwrap_or("").to_string();
        out.push(SearchHit {
            title: text_of(a),
            url: href,
            body: r.select(&sel_desc).next().map(text_of).unwrap_or_default(),
            extract: None,
            extract_error: None,
        });
        if out.len() >= max_results {
            break;
        }
    }
    Ok(out)
}

/// Google HTML (generally consent-walled for bot traffic; attempted for parity
/// with the configured backend list, failure is expected and swallowed).
async fn engine_google(query: &str, max_results: usize) -> Result<Vec<SearchHit>, String> {
    let client = scrape_client();
    let resp = client
        .get("https://www.google.com/search")
        .query(&[("q", query), ("num", &max_results.to_string())])
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("google HTTP {}", resp.status().as_u16()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    let doc = scraper::Html::parse_document(&body);
    let sel_g = scraper::Selector::parse("div.g").map_err(|e| e.to_string())?;
    let sel_a = scraper::Selector::parse("a[href^=\"http\"]").map_err(|e| e.to_string())?;
    let sel_h3 = scraper::Selector::parse("h3").map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for g in doc.select(&sel_g) {
        let (Some(a), Some(h3)) = (g.select(&sel_a).next(), g.select(&sel_h3).next()) else {
            continue;
        };
        out.push(SearchHit {
            title: text_of(h3),
            url: a.value().attr("href").unwrap_or("").to_string(),
            body: String::new(),
            extract: None,
            extract_error: None,
        });
        if out.len() >= max_results {
            break;
        }
    }
    Ok(out)
}

/// The scrape fallback for one query across the merged backend list — the
/// stand-in for `DDGS().text(query, max_results=..., backend=backends)`.
/// Returns (results, error); when every engine comes back empty the error is
/// ddgs's verbatim "No results found." (genuine-empty and all-rate-limited are
/// indistinguishable at this layer, exactly as in the library).
async fn ddgs_search(query: &str, max_results: usize) -> (Vec<SearchHit>, Option<String>) {
    let c = ws_config();
    let backends = c.websearch_backends.clone();
    let attempts = c.websearch_retry_attempts.max(1);
    let (backoff_base, backoff_cap) = (c.websearch_backoff_base, c.websearch_backoff_cap);
    let engines: Vec<String> =
        backends.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();

    let mut err: Option<String> = None;
    // Say which attempt is running. Worst case here is 15s per engine sweep
    // times `attempts`, plus backoff between them — 51 seconds on the shipped
    // defaults, all of it previously behind an unchanging spinner.
    let _clear = crate::utils::tokens::WaitNoteGuard;
    for attempt in 0..attempts {
        crate::utils::tokens::set_wait_note(Some(format!(
            "searching (attempt {}/{attempts})",
            attempt + 1
        )));
        // One "attempt" = one sweep of the merged engine list, aggregated and
        // deduped (the library's merge-across-whichever-respond behavior).
        let mut out: Vec<SearchHit> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut last_engine_err: Option<String> = None;
        for engine in &engines {
            let res = match engine.as_str() {
                "duckduckgo" => engine_duckduckgo(query, max_results).await,
                "mojeek" => engine_mojeek(query, max_results).await,
                "brave" => engine_brave(query, max_results).await,
                "google" => engine_google(query, max_results).await,
                other => Err(format!("unknown backend: {other}")),
            };
            match res {
                Ok(hits) => {
                    for h in hits {
                        if h.url.is_empty() || !seen.insert(h.url.clone()) {
                            continue;
                        }
                        out.push(h);
                    }
                }
                Err(e) => last_engine_err = Some(e),
            }
            if out.len() >= max_results {
                break;
            }
        }
        if !out.is_empty() {
            out.truncate(max_results);
            return (out, None);
        }
        // All engines empty/failed this attempt — the library raises
        // DDGSException('No results found.').
        err = Some(last_engine_err.unwrap_or_else(|| "No results found.".to_string()));
        // Back off before the next attempt (never after the last). Immediate
        // retries just re-hit the same rate limit.
        if attempt < attempts - 1 {
            let delay = backoff_cap.min(backoff_base * 2f64.powi(attempt as i32));
            let jitter: f64 = {
                use rand::Rng;
                rand::thread_rng().gen_range(0.0..=delay.max(f64::MIN_POSITIVE))
            };
            crate::utils::tokens::set_wait_note(Some(format!(
                "search rate-limited, retrying in {:.0}s",
                delay + jitter
            )));
            tokio::time::sleep(std::time::Duration::from_secs_f64(delay + jitter)).await;
        }
    }
    (Vec::new(), err)
}

/// Extract concatenated text blocks from a query_model AssistantMessage.
fn message_text(msg: &crate::types::query::AssistantMessage) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in &msg.message.content {
        if let ContentBlock::Text(t) = b {
            if !t.text.is_empty() {
                parts.push(t.text.clone());
            }
        }
    }
    parts.join("\n").trim().to_string()
}

/// Pull a JSON array of query strings out of the planner model's output.
fn parse_query_list(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    static ARR: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"(?s)\[.*\]").unwrap());
    let Some(m) = ARR.find(text) else {
        return Vec::new();
    };
    let Ok(arr) = serde_json::from_str::<Vec<Value>>(m.as_str()) else {
        return Vec::new();
    };
    arr.into_iter()
        .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Rewrite the user's intent into several focused queries via one small LLM
/// call. Always returns at least [query]; degrades to exactly that whenever no
/// model is reachable, so the tool stays functional headless and in tests.
async fn expand_queries(query: &str, context: Option<&ToolUseContext>) -> Vec<String> {
    let max_exp = ws_config().websearch_max_expansions;
    let mut expansions: Vec<String> = Vec::new();
    if let Some(ctx) = context {
        if max_exp > 0 {
            let today = chrono::Local::now().format("%B %Y").to_string();
            let prompt = format!(
                "Today is {today}. A user wants to web-search for: \"{query}\".\n\
                 Produce up to {max_exp} focused web-search queries that together \
                 cover this intent (synonyms, sub-questions, recency-scoped variants). \
                 Use the current year where recency matters. \
                 LANGUAGE RULE (important): the anchor query is searched verbatim, so \
                 at least one expansion MUST be a natural TRANSLATION into another \
                 language rather than a same-language rephrase. \
                 If the query is not in English, translate it into English. \
                 If the query is in English but the topic belongs to a particular \
                 country or language community (a person, outlet, product, place or \
                 regulation documented there), translate it into that language, \
                 written the way it is written locally. \
                 This spans both audiences instead of committing to one. \
                 For purely global or technical English topics, keep the expansion in English. \
                 Respond with ONLY a JSON array of query strings and nothing else."
            );
            let params = QueryModelParams {
                messages: vec![Message::User(create_user_message_text(&prompt))],
                system_prompt: vec![
                    "You are a search-query planner for a web search tool.".to_string()
                ],
                tools: vec![],
                signal: Some(ctx.abort_controller.clone()),
                // 256 truncated Korean expansions to an EMPTY response: Korean
                // tokenizes longer and the model emits reasoning, so the JSON
                // array never fit. 768 leaves headroom; the planner still stops
                // early on its short array.
                options: QueryOptions { max_tokens: Some(768), ..Default::default() },
                ..Default::default()
            };
            let msg = query_model(params).await;
            if !msg.is_api_error_message {
                expansions = parse_query_list(&message_text(&msg));
            }
        }
    }

    // Verbatim original first (anchor), then expansions; dedup preserving order.
    let mut ordered: Vec<String> = Vec::new();
    for q in std::iter::once(query.to_string()).chain(expansions) {
        let q = q.trim().to_string();
        if !q.is_empty() && !ordered.contains(&q) {
            ordered.push(q);
        }
        if ordered.len() >= (max_exp.max(0) as usize) + 1 {
            break;
        }
    }
    ordered
}

const BOT_CHALLENGE_MARKERS: &[&str] = &[
    "establishing a secure connection",
    "challenge loading",
    "bunny-shield",
    "just a moment...",
    "checking your browser",
    "cf-browser-verification",
    "enable javascript and cookies to continue",
    "verifying you are human",
    "ddos-guard",
];

/// True if the page is an anti-bot interstitial rather than real content.
/// Only applied to short pages — a long article legitimately discussing
/// Cloudflare must not be discarded.
fn is_bot_challenge(flat_text: &str) -> bool {
    if flat_text.chars().count() > 1500 {
        return false;
    }
    let low = flat_text.to_lowercase();
    BOT_CHALLENGE_MARKERS.iter().any(|m| low.contains(m))
}

/// Fetch a result page and reduce it to a short whitespace-collapsed extract,
/// reusing the web_fetch substrate. Returns (extract, error) — the reason a
/// page came back empty is surfaced rather than collapsed to ''.
async fn fetch_extract(
    url: &str,
    signal: Option<tokio_util::sync::CancellationToken>,
    max_chars: usize,
) -> (String, Option<String>) {
    use crate::tools::web_fetch_tool::{get_url_markdown_content, FetchOutcome};
    match get_url_markdown_content(url, signal).await {
        Ok(FetchOutcome::Redirect(_)) => (String::new(), Some("redirected off-origin".to_string())),
        Ok(FetchOutcome::Content(res)) => {
            let content = res.content.trim().to_string();
            if content.is_empty() {
                return (String::new(), Some("empty page".to_string()));
            }
            let flat = content.split_whitespace().collect::<Vec<_>>().join(" ");
            if is_bot_challenge(&flat) {
                // Cloudflare / Bunny Shield / DDoS-Guard interstitials return
                // HTTP 200 with a few hundred characters of "Establishing a
                // secure connection" — passed through, they read like a real
                // (if useless) page.
                return (
                    String::new(),
                    Some("bot challenge — site requires a JS browser".to_string()),
                );
            }
            (flat.chars().take(max_chars).collect(), None)
        }
        Err(e) => (String::new(), Some(e)),
    }
}

/// Render the merged results as the assistant-facing text. Shared by the live
/// search path and the cache-hit path so both emit byte-identical output
/// (minus the '(searched: ...)' line, which only the live multi-query path has).
fn format_results(query: &str, merged: &[SearchHit], searched: Option<&[String]>) -> String {
    let mut lines = vec![format!("Search results for \"{query}\":"), String::new()];
    if let Some(qs) = searched {
        if qs.len() > 1 {
            lines.push(format!("(searched: {})", qs.join("; ")));
            lines.push(String::new());
        }
    }
    for r in merged {
        lines.push(format!("- [{}]({})", r.title, r.url));
        let snippet = r.extract.as_deref().filter(|s| !s.is_empty()).unwrap_or(&r.body);
        if !snippet.is_empty() {
            lines.push(format!("  {snippet}"));
        }
        if let Some(err) = r.extract_error.as_deref().filter(|s| !s.is_empty()) {
            // Say so rather than silently showing a title with no body.
            lines.push(format!("  (page body not retrieved — {err})"));
        }
    }
    // Verbatim reminder the original adds in
    // mapToolResultToToolResultBlockParam (WebSearchTool.ts:426-427).
    lines.push(
        "\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks."
            .to_string(),
    );
    lines.join("\n")
}

fn str_list(input: &Value, key: &str) -> Vec<String> {
    input
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default()
}

/// Search the web (SearXNG-first, scrape fallback) with intent expansion +
/// page fetch.
pub struct WebSearchTool;

#[async_trait::async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        WEB_SEARCH_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        json!({
            "additionalProperties": false,
            "properties": {
                "query": {
                    "description": "The search query to use",
                    "minLength": 2,
                    "title": "Query",
                    "type": "string"
                },
                "allowed_domains": {
                    "anyOf": [
                        {"items": {"type": "string"}, "type": "array"},
                        {"type": "null"}
                    ],
                    "default": null,
                    "description": "Only include search results from these domains",
                    "title": "Allowed Domains"
                },
                "blocked_domains": {
                    "anyOf": [
                        {"items": {"type": "string"}, "type": "array"},
                        {"type": "null"}
                    ],
                    "default": null,
                    "description": "Never include search results from these domains",
                    "title": "Blocked Domains"
                }
            },
            "required": ["query"],
            "title": "WebSearchInput",
            "type": "object"
        })
    }

    async fn description(&self, _input: &Value) -> String {
        get_description()
    }

    async fn prompt(&self) -> String {
        get_description()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn check_permissions(&self, _input: &Value, _context: &ToolUseContext) -> PermissionResult {
        // Ask, despite being read-only — same reasoning as WebFetch: nothing
        // changes locally, but the query is model-controlled and leaves this
        // machine, so it is an egress channel however small. Under -dsp this is
        // never reached.
        PermissionResult::ask()
    }

    fn needs_permissions(&self) -> bool {
        true // network access
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "WebSearch".to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let q = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if q.is_empty() || q.chars().count() < 2 {
            return ValidationResult::failure("query must be at least 2 characters");
        }
        // WebSearchTool.ts:244-251 (errorCode 2): allowed and blocked domains
        // are mutually exclusive.
        if !str_list(input, "allowed_domains").is_empty() && !str_list(input, "blocked_domains").is_empty()
        {
            return ValidationResult::failure(
                "Error: Cannot specify both allowed_domains and blocked_domains in the same request",
            );
        }
        ValidationResult::success()
    }

    /// The query, truncated for display (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("query")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::formatting::truncate(
            value,
            crate::utils::formatting::TOOL_SUMMARY_MAX_LENGTH,
        ))
    }

    /// "Searching for <query>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Searching for {summary}"),
            None => "Searching the web".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, _verbose: bool) -> String {
        let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
        if query.is_empty() {
            String::new()
        } else {
            colorize(query, colors::DIM)
        }
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<crate::tools::base::CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let allowed = str_list(&input, "allowed_domains");
            let blocked = str_list(&input, "blocked_domains");
            let signal = Some(context.abort_controller.clone());

            // 0. cache lookup. A coding agent re-searches near-identical
            //    queries constantly; a fresh hit skips every upstream request.
            let key = cache_key(&query, &allowed, &blocked);
            if let Some(cached) = cache_get(&key) {
                let content: Vec<Value> = cached
                    .iter()
                    .map(|r| json!({"title": r.title, "url": r.url}))
                    .collect();
                yield Ok(ToolResult::result(
                    format_results(&query, &cached, None),
                    Some(json!({"query": query, "cached": true, "content": content})),
                ));
                return;
            }

            // 1. expand intent -> multiple queries (recovers the multi-search loop).
            let queries = expand_queries(&query, Some(&context)).await;

            // 2. SERIALIZED SERP, SearXNG-first; scrape fallback per query with
            //    throttle. Once SearXNG is proven unreachable in this call, stop
            //    trying it for the remaining queries.
            let built: Vec<String> =
                queries.iter().map(|q| build_query(q, &allowed, &blocked)).collect();
            let mut searched: Vec<(Vec<SearchHit>, Option<String>)> = Vec::new();
            let mut sources: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let wc = ws_config();
            let lo = wc.websearch_throttle_min.min(wc.websearch_throttle_max);
            let hi = wc.websearch_throttle_min.max(wc.websearch_throttle_max);
            let mut searxng_live = !searxng_conf().0.is_empty(); // cleared on first unreachable
            let mut ddgs_called = false; // gate the inter-call throttle
            for b in &built {
                let mut results: Vec<SearchHit> = Vec::new();
                let mut err: Option<String> = None;

                if searxng_live {
                    let (sx, sx_err) = searxng_search(b, MAX_RESULTS_PER_QUERY).await;
                    match sx {
                        None => {
                            // Host unreachable / JSON disabled — don't retry it
                            // this call.
                            searxng_live = false;
                            err = sx_err;
                        }
                        Some(sx_hits) if !sx_hits.is_empty() => {
                            results = sx_hits;
                            sources.insert("searxng".to_string());
                        }
                        Some(_) => {}
                    }
                }

                if results.is_empty() {
                    // SearXNG off / unavailable / empty -> scrape fallback.
                    // Space out successive scrapes (burst avoidance); the first
                    // of the search doesn't wait.
                    if ddgs_called && hi > 0.0 {
                        let jitter = {
                            use rand::Rng;
                            rand::thread_rng().gen_range(lo..=hi.max(lo + f64::MIN_POSITIVE))
                        };
                        tokio::time::sleep(std::time::Duration::from_secs_f64(jitter)).await;
                    }
                    ddgs_called = true;
                    let (dd_results, dd_err) = ddgs_search(b, MAX_RESULTS_PER_QUERY).await;
                    results = dd_results;
                    if !results.is_empty() {
                        sources.insert("ddgs".to_string());
                    }
                    err = if results.is_empty() { dd_err.or(err) } else { None };
                }

                searched.push((results, err));
            }

            let mut merged: Vec<SearchHit> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let mut errors: Vec<String> = Vec::new();
            for (results, err) in &searched {
                if let Some(e) = err {
                    errors.push(e.clone());
                }
                for r in results {
                    if !seen.insert(r.url.clone()) {
                        continue;
                    }
                    merged.push(r.clone());
                }
            }

            // Enforce the domain contract: the site:/-site: operators bias the
            // engines toward the right pages, but not every backend honors them
            // — this filter is the precision guarantee on top of that bias.
            if !allowed.is_empty() {
                merged.retain(|r| domain_match(&host_of(&r.url), &allowed));
            }
            if !blocked.is_empty() {
                merged.retain(|r| !domain_match(&host_of(&r.url), &blocked));
            }

            if merged.is_empty() {
                let msg = if !errors.is_empty() && errors.len() == built.len() {
                    let mut uniq: Vec<String> = errors.clone();
                    uniq.sort();
                    uniq.dedup();
                    format!(
                        "Web search is temporarily unavailable — all backends failed ({}). \
                         This is usually transient rate-limiting; try again in a moment.",
                        uniq.join("; ")
                    )
                } else {
                    format!("No search results found for \"{query}\".")
                };
                yield Ok(ToolResult::result(
                    msg,
                    Some(json!({"query": query, "content": []})),
                ));
                return;
            }

            merged.truncate(MAX_MERGED);

            // 3. top-K page fetch -> short extract (recovers page-content reading).
            let top_k = merged.len().min(FETCH_TOP_K);
            let fetches = merged[..top_k]
                .iter()
                .map(|r| fetch_extract(&r.url, signal.clone(), EXTRACT_CHARS));
            let extracts = futures::future::join_all(fetches).await;
            for (r, (ex, err)) in merged[..top_k].iter_mut().zip(extracts) {
                if !ex.is_empty() {
                    r.extract = Some(ex);
                } else if let Some(e) = err {
                    r.extract_error = Some(e);
                }
            }

            // Cache the completed, non-empty result set (cache_put never stores
            // an empty set, so a transient block isn't pinned as "no results").
            cache_put(&key, &merged);

            let content: Vec<Value> =
                merged.iter().map(|r| json!({"title": r.title, "url": r.url})).collect();
            let source: Vec<String> = sources.into_iter().collect();
            yield Ok(ToolResult::result(
                format_results(&query, &merged, Some(&queries)),
                Some(json!({"query": query, "source": source, "content": content})),
            ));
        })
    }
}

pub static WEB_SEARCH_TOOL: WebSearchTool = WebSearchTool;

// Silence the unused-import lint for Arc used only via QueryModelParams
// defaults in some cfgs.
#[allow(unused)]
fn _arc_marker(_: Option<Arc<()>>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_matches_python() {
        // Oracle values from the Python implementation:
        // json.dumps({'q':'한글 query','a':[],'b':[]}, sort_keys=True)
        //   == '{"a": [], "b": [], "q": "\\ud55c\\uae00 query"}'
        //   sha256[:32] == 'f623d56d746c3447b2727bfea93cb352'
        assert_eq!(cache_key("한글 query", &[], &[]), "f623d56d746c3447b2727bfea93cb352");
    }

    #[test]
    fn build_query_operators() {
        assert_eq!(
            build_query("rust", &["Docs.rs".to_string(), ".github.com".to_string()], &[]),
            "(site:docs.rs OR site:github.com) rust"
        );
        assert_eq!(
            build_query("rust", &[], &["Spam.com".to_string()]),
            "rust -site:spam.com"
        );
    }

    #[test]
    fn domain_matching() {
        assert!(domain_match("docs.rs", &["docs.rs".to_string()]));
        assert!(domain_match("api.docs.rs", &["docs.rs".to_string()]));
        assert!(!domain_match("notdocs.rs", &["docs.rs".to_string()]));
        assert!(!domain_match("docs.rs.evil.com", &["docs.rs".to_string()]));
    }

    #[test]
    fn parse_query_list_extracts_json() {
        assert_eq!(
            parse_query_list("Here you go:\n[\"a\", \"b\"]\ndone"),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(parse_query_list("no array here").is_empty());
        assert!(parse_query_list("").is_empty());
    }

    #[test]
    fn bot_challenge_detection() {
        assert!(is_bot_challenge("Just a moment... checking your browser"));
        assert!(!is_bot_challenge(&format!(
            "long article about cloudflare {}",
            "x".repeat(2000)
        )));
    }

    #[test]
    fn format_results_shape() {
        let hits = vec![
            SearchHit {
                title: "T1".into(),
                url: "https://a.com".into(),
                body: "snippet".into(),
                extract: None,
                extract_error: None,
            },
            SearchHit {
                title: "T2".into(),
                url: "https://b.com".into(),
                body: String::new(),
                extract: None,
                extract_error: Some("empty page".into()),
            },
        ];
        let out = format_results("q", &hits, Some(&["q".to_string(), "q2".to_string()]));
        assert!(out.starts_with("Search results for \"q\":\n\n(searched: q; q2)\n\n"));
        assert!(out.contains("- [T1](https://a.com)\n  snippet"));
        assert!(out.contains("- [T2](https://b.com)\n  (page body not retrieved — empty page)"));
        assert!(out.ends_with(
            "\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks."
        ));
        // single-query (cache) path has no searched line
        let out2 = format_results("q", &hits, None);
        assert!(!out2.contains("(searched:"));
    }

    #[test]
    fn cache_roundtrip_and_ttl() {
        // Isolate the cache dir via env (CODEMASON_WEBSEARCH_CACHE_DIR is read
        // through config; use the field default path but a unique key instead).
        let key = format!("testkey{}", std::process::id());
        let hits = vec![SearchHit {
            title: "T".into(),
            url: "https://a.com".into(),
            body: "b".into(),
            extract: Some("ex".into()),
            extract_error: None,
        }];
        cache_put(&key, &hits);
        if ws_config().websearch_cache_ttl > 0 {
            let got = cache_get(&key).expect("fresh entry readable");
            assert_eq!(got[0].url, "https://a.com");
            assert_eq!(got[0].extract.as_deref(), Some("ex"));
            // Python-written shape: {"ts": ..., "results": [...]} — verify keys.
            let path = cache_dir().join(format!("{key}.json"));
            let raw = std::fs::read_to_string(&path).unwrap();
            let v: Value = serde_json::from_str(&raw).unwrap();
            assert!(v.get("ts").is_some() && v.get("results").is_some());
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn backoff_math() {
        // delay = min(cap, base * 2^attempt) — the Python loop's values.
        let base = 0.8f64;
        let cap = 8.0f64;
        let delays: Vec<f64> = (0..4).map(|a| cap.min(base * 2f64.powi(a))).collect();
        assert_eq!(delays, vec![0.8, 1.6, 3.2, 6.4]);
        assert_eq!(cap.min(base * 2f64.powi(10)), 8.0);
    }
}
