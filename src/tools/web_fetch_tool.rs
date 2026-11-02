//! WebFetch Tool for CodeMason
//! Equivalent to codemason/tools/web_fetch_tool.py (src/tools/WebFetchTool/).
//!
//! Flow (WebFetchTool.ts): fetch URL -> HTML->markdown -> run the caller's
//! prompt on the content via a small model -> return the model's response. The
//! DESCRIPTION and make_secondary_model_prompt are ported verbatim. The "small
//! fast model" is the local model (query_model).
//!
//! Cross-host redirect detection (isPermittedRedirect / getWithPermittedRedirects),
//! the preapproved-host list (preapproved.ts), the 15-minute URL cache
//! (utils.ts URL_CACHE), and validateURL / content-length caps are ported
//! faithfully. Substrate swap: markdownify (itself a turndown stand-in) ->
//! a scraper-crate DOM walk (see html_to_markdown for the documented rendering
//! divergences).

use std::collections::{HashMap, HashSet};

use std::time::{SystemTime, UNIX_EPOCH};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::services::backend::query_model;
use crate::services::openai_common::{QueryModelParams, QueryOptions};
use crate::tools::base::{PermissionResult, Tool, ToolResult, ToolStream};
use crate::types::query::{Message, ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};
use crate::utils::messages::create_user_message_text;

pub const WEB_FETCH_TOOL_NAME: &str = "WebFetch";

// ---------------------------------------------------------------------------
// Preapproved hosts — faithful port of preapproved.ts.
//
// WHAT THIS LIST ACTUALLY DOES HERE, because the upstream wording below invites
// the wrong reading and I got it wrong myself once: it is NOT a permission gate.
// check_permissions asks for EVERY WebFetch regardless of this list (and -dsp
// skips the question for every URL, equally regardless). The only thing being
// on it changes is that markdown under the size cap is returned RAW instead of
// being summarized through a model call. So an entry here buys a cheaper, more
// faithful read of a site — not access to one.
//
// Upstream's rationale, kept for provenance: "For legal and security concerns,
// WebFetch typically only accesses domains the user has provided. We make an
// exception for a list of preapproved code-related domains." That held where
// the list did gate access; in this port it does not.
//
// Some entries are path-scoped (e.g. "vercel.com/docs"); those are matched with
// path-segment boundaries so "/docs" does not match "/docs-evil/malware".
// ---------------------------------------------------------------------------
pub const PREAPPROVED_HOSTS: &[&str] = &[
    // Open specs this program implements. Another product's own documentation
    // and repositories used to sit here too; they were removed on identity
    // grounds — this is not that product — NOT as a security measure, since the
    // list gates nothing. The cost of removing them is that those three sites
    // now get summarized rather than read raw.
    "modelcontextprotocol.io",
    "agentskills.io",
    // Top Programming Languages
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "www.typescriptlang.org",
    // Web & JavaScript Frameworks/Libraries
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    // Python Frameworks & Libraries
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    // PHP Frameworks
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    // Java Frameworks & Libraries
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    // .NET & C# Frameworks
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    // Mobile Development
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    // Data Science & Machine Learning
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    // Databases
    "www.mongodb.com",
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    // Cloud & DevOps
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com",
    // Testing & Monitoring
    "cypress.io",
    "selenium.dev",
    // Game Development
    "docs.unity.com",
    "docs.unrealengine.com",
    // Other Essential Tools
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org",
];

/// Split entries into hostname-only (O(1) set membership) and path-scoped
/// (host -> list of path prefixes), mirroring preapproved.ts's module-load split.
static PREAPPROVED_SPLIT: Lazy<(HashSet<&'static str>, HashMap<&'static str, Vec<&'static str>>)> =
    Lazy::new(|| {
        let mut hosts = HashSet::new();
        let mut paths: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        for entry in PREAPPROVED_HOSTS {
            match entry.find('/') {
                None => {
                    hosts.insert(*entry);
                }
                Some(slash) => {
                    let (host, path) = entry.split_at(slash);
                    paths.entry(host).or_default().push(path);
                }
            }
        }
        (hosts, paths)
    });

/// Port of preapproved.ts isPreapprovedHost.
pub fn is_preapproved_host(hostname: &str, pathname: &str) -> bool {
    let (hosts, paths) = &*PREAPPROVED_SPLIT;
    if hosts.contains(hostname) {
        return true;
    }
    if let Some(prefixes) = paths.get(hostname) {
        for p in prefixes {
            // Enforce path segment boundaries: "/docs" must not match
            // "/docs-evil/malware". Only exact match or "/" after the
            // prefix is allowed.
            if pathname == *p || pathname.starts_with(&format!("{p}/")) {
                return true;
            }
        }
    }
    false
}

pub const DESCRIPTION: &str = r#"
- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL
  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.
  - For GitHub URLs, prefer using the gh CLI via Bash instead (e.g., gh pr view, gh issue view, gh api).
"#;

/// Truncate to not spend too many tokens (utils.ts MAX_MARKDOWN_LENGTH).
pub const MAX_MARKDOWN_LENGTH: usize = 100_000;

/// validateURL caps (utils.ts).
pub const MAX_URL_LENGTH: usize = 2000;

/// Cap the HTTP read at 10MB (utils.ts MAX_HTTP_CONTENT_LENGTH).
pub const MAX_HTTP_CONTENT_LENGTH: usize = 10 * 1024 * 1024;

/// Cap same-host redirect hops (utils.ts MAX_REDIRECTS).
pub const MAX_REDIRECTS: usize = 10;

// utils.ts URL_CACHE: LRUCache with a 15-minute TTL and a 50MB byte-size cap
// (CACHE_TTL_MS / MAX_CACHE_SIZE_BYTES). Keyed by the ORIGINAL url (not the
// upgraded/redirected one). Ported as an order-tracked map with byte
// accounting + LRU eviction, exactly like the Python OrderedDict port.
const CACHE_TTL_S: f64 = 15.0 * 60.0;
const MAX_CACHE_SIZE_BYTES: usize = 50 * 1024 * 1024; // 50MB

/// A fetched page (utils.ts FetchedContent).
#[derive(Debug, Clone)]
pub struct FetchedContent {
    pub content: String,
    pub bytes: usize,
    pub code: u16,
    pub code_text: String,
    pub content_type: String,
}

/// Mirror of utils.ts RedirectInfo.
#[derive(Debug, Clone)]
pub struct RedirectInfo {
    pub original_url: String,
    pub redirect_url: String,
    pub status_code: u16,
}

pub enum FetchOutcome {
    Content(FetchedContent),
    Redirect(RedirectInfo),
}

struct UrlCache {
    /// insertion/LRU order, oldest first
    order: Vec<String>,
    map: HashMap<String, (f64, FetchedContent)>,
    bytes: usize,
}

static URL_CACHE: Lazy<Mutex<UrlCache>> =
    Lazy::new(|| Mutex::new(UrlCache { order: Vec::new(), map: HashMap::new(), bytes: 0 }));

fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// sizeCalculation: byte length of the cached markdown content.
fn entry_size(value: &FetchedContent) -> usize {
    value.content.len().max(1)
}

impl UrlCache {
    fn evict(&mut self, url: &str) {
        if let Some((_, v)) = self.map.remove(url) {
            self.bytes = self.bytes.saturating_sub(entry_size(&v));
        }
        self.order.retain(|k| k != url);
    }

    fn get(&mut self, url: &str) -> Option<FetchedContent> {
        let (ts, _) = self.map.get(url)?;
        if now_secs() - ts > CACHE_TTL_S {
            self.evict(url);
            return None;
        }
        // LRU touch
        self.order.retain(|k| k != url);
        self.order.push(url.to_string());
        self.map.get(url).map(|(_, v)| v.clone())
    }

    fn set(&mut self, url: &str, value: FetchedContent) {
        let now = now_secs();
        if self.map.contains_key(url) {
            self.evict(url);
        }
        // Drop expired entries first (TTL), then enforce the byte cap by
        // evicting least-recently-used entries until under MAX_CACHE_SIZE_BYTES.
        let expired: Vec<String> = self
            .map
            .iter()
            .filter(|(_, (ts, _))| now - ts > CACHE_TTL_S)
            .map(|(k, _)| k.clone())
            .collect();
        for k in expired {
            self.evict(&k);
        }
        self.bytes += entry_size(&value);
        self.map.insert(url.to_string(), (now, value));
        self.order.push(url.to_string());
        while self.bytes > MAX_CACHE_SIZE_BYTES && self.map.len() > 1 {
            let Some(oldest) = self.order.first().cloned() else { break };
            if oldest == url {
                break;
            }
            self.evict(&oldest);
        }
    }
}

/// Port of makeSecondaryModelPrompt (prompt.ts:23-46).
pub fn make_secondary_model_prompt(
    markdown_content: &str,
    prompt: &str,
    is_preapproved_domain: bool,
) -> String {
    let guidelines = if is_preapproved_domain {
        "Provide a concise response based on the content above. Include relevant details, code examples, and documentation excerpts as needed.".to_string()
    } else {
        "Provide a concise response based only on the content above. In your response:\n - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is ok as long as we respect the license.\n - Use quotation marks for exact language from articles; any language outside of the quotation should never be word-for-word the same.\n - You are not a lawyer and never comment on the legality of your own prompts and responses.\n - Never produce or reproduce exact song lyrics.".to_string()
    };
    format!("\nWeb page content:\n---\n{markdown_content}\n---\n\n{prompt}\n\n{guidelines}\n")
}

fn parse_maybe_schemeless(url: &str) -> Result<url::Url, url::ParseError> {
    if url.contains("://") {
        url::Url::parse(url)
    } else {
        url::Url::parse(&format!("https://{url}"))
    }
}

/// Port of utils.ts isPreapprovedUrl.
pub fn is_preapproved_url(u: &str) -> bool {
    match parse_maybe_schemeless(u) {
        Ok(parsed) => match parsed.host_str() {
            Some(host) => {
                let path = if parsed.path().is_empty() { "/" } else { parsed.path() };
                is_preapproved_host(host, path)
            }
            None => false,
        },
        Err(_) => false,
    }
}

/// Is this v4 address in any range Python's ipaddress classifies as
/// private/loopback/link_local/reserved/multicast/unspecified?
fn blocked_v4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_private()               // 10/8, 172.16/12, 192.168/16
        || ip.is_loopback()       // 127/8
        || ip.is_link_local()     // 169.254/16
        || ip.is_multicast()      // 224/4
        || ip.is_broadcast()      // 255.255.255.255
        || ip.is_unspecified()    // 0.0.0.0
        || o[0] == 0              // 0.0.0.0/8 ("this network", is_private in Python)
        || o[0] >= 240            // 240/4 reserved
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)   // 192.0.0.0/24 (incl. /29)
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)   // 192.0.2.0/24 TEST-NET-1
        || (o[0] == 198 && (o[1] & 0xFE) == 18)      // 198.18/15 benchmarking
        || (o[0] == 198 && o[1] == 51 && o[2] == 100) // 198.51.100/24 TEST-NET-2
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)  // 203.0.113/24 TEST-NET-3
}

/// NAT64 well-known prefix 64:ff9b::/96 (RFC 6052).
fn nat64_embedded_v4(ip: std::net::Ipv6Addr) -> Option<std::net::Ipv4Addr> {
    let seg = ip.segments();
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0 && seg[3] == 0 && seg[4] == 0 && seg[5] == 0
    {
        let o = ip.octets();
        Some(std::net::Ipv4Addr::new(o[12], o[13], o[14], o[15]))
    } else {
        None
    }
}

fn blocked_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => blocked_v4(v4),
        std::net::IpAddr::V6(v6) => {
            // NAT64 (RFC 6052 well-known prefix). On an IPv6-only access network
            // with 464XLAT the resolver hands back 64:ff9b::<v4> for ordinary
            // public sites; unwrap to the embedded IPv4 and judge THAT — a
            // NAT64-mapped 192.168.x is still blocked, a NAT64-mapped public
            // address is not.
            if let Some(v4) = nat64_embedded_v4(v6) {
                return blocked_v4(v4);
            }
            if let Some(v4) = v6.to_ipv4_mapped() {
                return blocked_v4(v4);
            }
            let seg = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (seg[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
                || (seg[0] & 0xffc0) == 0xfe80 // fe80::/10 link local
                || seg[0] == 0x0000            // ::/8 reserved
                || seg[0] == 0x2001 && seg[1] == 0x0db8 // 2001:db8::/32 doc
        }
    }
}

/// True if the host is loopback/private/link-local/reserved or an internal
/// name. The original relied on per-domain USER APPROVAL as the SSRF boundary;
/// this port auto-allows read-only tools, so a hard internal-address guard
/// replaces it (127.0.0.1 / 192.168.x / 169.254.169.254 cloud metadata / etc).
pub fn is_internal_host(hostname: &str) -> bool {
    let h = hostname.trim_matches(|c| c == '[' || c == ']').to_lowercase();
    if h.is_empty() {
        return true;
    }
    if h == "localhost"
        || h.ends_with(".localhost")
        || h.ends_with(".internal")
        || h.ends_with(".local")
    {
        return true;
    }

    // Literal IP host (strip a %zone suffix like Python did).
    let bare = h.split('%').next().unwrap_or("");
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return blocked_ip(ip);
    }

    // Resolve the name; block if ANY resolved address is internal. (DNS is only
    // queried once here; DNS-rebinding TOCTOU between this and the fetch is out
    // of scope for a local tool.) Blocking DNS call, same as the Python
    // getaddrinfo — wrapped so a multithreaded runtime isn't stalled.
    let resolve = || {
        use std::net::ToSocketAddrs;
        format!("{h}:443").to_socket_addrs().map(|it| it.map(|sa| sa.ip()).collect::<Vec<_>>())
    };
    let addrs = match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(resolve)
        }
        _ => resolve(),
    };
    match addrs {
        Ok(addrs) => addrs.into_iter().any(blocked_ip),
        Err(_) => false, // unresolvable → the fetch will fail on its own
    }
}

/// Port of utils.ts validateURL, plus an internal-address SSRF guard (the
/// original's per-domain approval gate is not reproduced in this console port).
pub fn validate_url(u: &str) -> bool {
    if u.len() > MAX_URL_LENGTH {
        return false;
    }
    let Ok(parsed) = parse_maybe_schemeless(u) else {
        return false;
    };

    // Only http(s) — the fetch upgrades http->https; block file:/ftp:/gopher: etc.
    if !matches!(parsed.scheme(), "http" | "https") {
        return false;
    }

    // Block URLs carrying username/password (userinfo) credentials.
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return false;
    }

    // Require a publicly-resolvable, dotted hostname (>= 2 labels).
    let hostname = parsed.host_str().unwrap_or("");
    if hostname.split('.').count() < 2 {
        return false;
    }

    // SSRF guard: reject loopback/private/link-local/reserved/metadata targets.
    if is_internal_host(hostname) {
        return false;
    }

    true
}

/// Port of utils.ts isPermittedRedirect. Allows only same-origin redirects
/// and www-toggle redirects (add/remove leading 'www.').
pub fn is_permitted_redirect(original_url: &str, redirect_url: &str) -> bool {
    let (Ok(po), Ok(pr)) = (url::Url::parse(original_url), url::Url::parse(redirect_url)) else {
        return false;
    };
    if pr.scheme() != po.scheme() {
        return false;
    }
    if pr.port() != po.port() {
        return false;
    }
    if !pr.username().is_empty() || pr.password().is_some() {
        return false;
    }
    let strip_www = |host: &str| host.strip_prefix("www.").unwrap_or(host).to_string();
    strip_www(po.host_str().unwrap_or("")) == strip_www(pr.host_str().unwrap_or(""))
}

/// HTML -> markdown, the Rust counterpart of the Python port's markdownify
/// (itself standing in for the original's turndown).
///
/// Documented rendering divergences from markdownify: script/style/head/
/// noscript/template CONTENT is dropped entirely (markdownify could leak raw
/// script text; turndown upstream also drops it); attribute-level options
/// (heading_style etc.) follow markdownify's DEFAULTS — h1/h2 render SETEXT
/// (underlined with ===/---), h3+ ATX (###); bullets '*'; nested lists indent
/// with a tab. Whitespace normalization differs at the margins; the trailing
/// `\n{3,}` collapse matches the Python post-pass exactly.
/// The charset a response declares, lowercased, or None.
///
/// Header first, then a `<meta>` in the document head — the order the browsers
/// use, and the order that matters when the two disagree. Only the head is
/// scanned: a charset declaration is required to appear early, and a later
/// "charset=" inside body text is prose, not a declaration.
pub fn declared_charset(raw: &[u8], content_type: &str) -> Option<String> {
    if let Some(cs) = charset_from_content_type(content_type) {
        return Some(cs);
    }
    // ASCII-compatible enough to find the declaration in any of these encodings;
    // that is exactly why the standard requires it inside the first bytes.
    let head = &raw[..raw.len().min(2048)];
    let head = String::from_utf8_lossy(head).to_lowercase();
    let idx = head.find("charset")?;
    let rest = &head[idx + "charset".len()..];
    let rest = rest.trim_start().strip_prefix('=')?.trim_start();
    let rest = rest.trim_start_matches(['"', '\'']);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .unwrap_or(rest.len());
    let cs = &rest[..end];
    (!cs.is_empty()).then(|| cs.to_string())
}

/// `charset=` out of a Content-Type header value, lowercased.
fn charset_from_content_type(content_type: &str) -> Option<String> {
    let lower = content_type.to_lowercase();
    let idx = lower.find("charset=")?;
    let rest = &lower[idx + "charset=".len()..];
    let rest = rest.trim_start_matches(['"', '\'']);
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .unwrap_or(rest.len());
    let cs = &rest[..end];
    (!cs.is_empty()).then(|| cs.to_string())
}

/// Decode a fetched body using the charset it declares.
///
/// Everything used to go through a lossy UTF-8 decode, so a page served as
/// EUC-KR, Shift_JIS or GBK reached the model as a wall of replacement
/// characters — it could not answer about the page, and nothing said why. The
/// declaration is authoritative here (unlike a local file, which declares
/// nothing and would need guessing), so this is a lookup, not a heuristic.
///
/// An unrecognised or absent charset keeps the previous behaviour exactly:
/// lossy UTF-8, which is also what the standard defaults to.
pub fn decode_body(raw: &[u8], content_type: &str) -> String {
    if let Some(label) = declared_charset(raw, content_type) {
        if let Some(enc) = encoding_rs::Encoding::for_label(label.as_bytes()) {
            let (text, _, _) = enc.decode(raw);
            return text.into_owned();
        }
    }
    String::from_utf8_lossy(raw).into_owned()
}

pub fn html_to_markdown(content: &str, content_type: &str) -> String {
    if !content_type.to_lowercase().contains("html") {
        return content.to_string();
    }
    let doc = scraper::Html::parse_document(content);
    let mut out = String::new();
    walk_children(doc.root_element(), &mut out, &WalkCtx::default());
    // markdownify's output then goes through re.sub(r'\n\s*\n\s*\n+', '\n\n')
    static COLLAPSE: Lazy<regex::Regex> =
        Lazy::new(|| regex::Regex::new(r"\n[ \t\r\x0b\x0c]*\n[ \t\r\x0b\x0c]*(\n[ \t\r\x0b\x0c]*)+").unwrap());
    let collapsed = COLLAPSE.replace_all(&out, "\n\n");
    collapsed.trim().to_string()
}

#[derive(Default, Clone)]
struct WalkCtx {
    /// inside <pre> — keep whitespace verbatim
    pre: bool,
    /// current list nesting depth (bullets/numbers indent by depth-1 tabs)
    list_depth: usize,
}

fn append_text(t: &str, out: &mut String, ctx: &WalkCtx) {
    if ctx.pre {
        out.push_str(t);
        return;
    }
    // Collapse runs of whitespace to one space, dropping leading whitespace
    // after a newline / at the start.
    let mut last_ws = out.ends_with('\n') || out.ends_with(' ') || out.is_empty();
    for c in t.chars() {
        if c.is_whitespace() {
            if !last_ws {
                out.push(' ');
                last_ws = true;
            }
        } else {
            out.push(c);
            last_ws = false;
        }
    }
}

fn walk_children(el: scraper::ElementRef<'_>, out: &mut String, ctx: &WalkCtx) {
    for child in el.children() {
        match child.value() {
            scraper::Node::Text(t) => append_text(t, out, ctx),
            scraper::Node::Element(_) => {
                if let Some(cel) = scraper::ElementRef::wrap(child) {
                    walk_element(cel, out, ctx);
                }
            }
            _ => {}
        }
    }
}

fn block_sep(out: &mut String) {
    while out.ends_with(' ') || out.ends_with('\t') {
        out.pop();
    }
    if !out.is_empty() && !out.ends_with("\n\n") {
        while out.ends_with('\n') {
            out.pop();
        }
        out.push_str("\n\n");
    }
}

fn collect_inline_text(el: scraper::ElementRef<'_>, ctx: &WalkCtx) -> String {
    let mut buf = String::new();
    walk_children(el, &mut buf, ctx);
    buf
}

fn walk_element(el: scraper::ElementRef<'_>, out: &mut String, ctx: &WalkCtx) {
    let name = el.value().name();
    match name {
        "script" | "style" | "head" | "noscript" | "template" => {}
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
            let level = name[1..].parse::<usize>().unwrap_or(1);
            let text = collect_inline_text(el, ctx);
            let text = text.trim();
            if !text.is_empty() {
                block_sep(out);
                if level == 1 {
                    // markdownify default heading_style=UNDERLINED for h1/h2
                    out.push_str(&format!("{text}\n{}\n", "=".repeat(text.chars().count())));
                } else if level == 2 {
                    out.push_str(&format!("{text}\n{}\n", "-".repeat(text.chars().count())));
                } else {
                    out.push_str(&format!("{} {text}\n", "#".repeat(level)));
                }
                out.push('\n');
            }
        }
        "p" | "div" | "section" | "article" | "main" | "header" | "footer" | "nav"
        | "aside" | "figure" | "figcaption" | "form" | "fieldset" => {
            block_sep(out);
            walk_children(el, out, ctx);
            block_sep(out);
        }
        "br" => out.push('\n'),
        "hr" => {
            block_sep(out);
            out.push_str("---\n\n");
        }
        "strong" | "b" => {
            let text = collect_inline_text(el, ctx);
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push_str(&format!("**{trimmed}**"));
            }
        }
        "em" | "i" => {
            let text = collect_inline_text(el, ctx);
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push_str(&format!("*{trimmed}*"));
            }
        }
        "code" if !ctx.pre => {
            let text = collect_inline_text(el, ctx);
            if !text.is_empty() {
                out.push_str(&format!("`{text}`"));
            }
        }
        "pre" => {
            block_sep(out);
            let mut inner = String::new();
            let mut pre_ctx = ctx.clone();
            pre_ctx.pre = true;
            walk_children(el, &mut inner, &pre_ctx);
            out.push_str("```\n");
            out.push_str(inner.trim_end_matches('\n'));
            out.push_str("\n```\n\n");
        }
        "blockquote" => {
            block_sep(out);
            let mut inner = String::new();
            walk_children(el, &mut inner, ctx);
            for line in inner.trim().split('\n') {
                out.push_str("> ");
                out.push_str(line);
                out.push('\n');
            }
            out.push('\n');
        }
        "ul" | "ol" => {
            if ctx.list_depth == 0 {
                block_sep(out);
            } else if !out.ends_with('\n') {
                out.push('\n');
            }
            let mut l_ctx = ctx.clone();
            l_ctx.list_depth += 1;
            let mut counter = 1u32;
            for child in el.children() {
                match child.value() {
                    scraper::Node::Element(cel) if cel.name() == "li" => {
                        if let Some(li) = scraper::ElementRef::wrap(child) {
                            let indent = "\t".repeat(l_ctx.list_depth - 1);
                            let marker = if name == "ol" {
                                let m = format!("{counter}. ");
                                counter += 1;
                                m
                            } else {
                                "* ".to_string()
                            };
                            let mut item = String::new();
                            walk_children(li, &mut item, &l_ctx);
                            let item = item.trim();
                            out.push_str(&format!("{indent}{marker}{item}\n"));
                        }
                    }
                    scraper::Node::Element(_) => {
                        if let Some(cel) = scraper::ElementRef::wrap(child) {
                            walk_element(cel, out, &l_ctx);
                        }
                    }
                    scraper::Node::Text(t) => append_text(t, out, &l_ctx),
                    _ => {}
                }
            }
            if ctx.list_depth == 0 {
                out.push('\n');
            }
        }
        "a" => {
            let text = collect_inline_text(el, ctx);
            let text = text.trim();
            let href = el.value().attr("href").unwrap_or("");
            if !text.is_empty() {
                if href.is_empty() {
                    out.push_str(text);
                } else {
                    out.push_str(&format!("[{text}]({href})"));
                }
            }
        }
        "img" => {
            let alt = el.value().attr("alt").unwrap_or("");
            let src = el.value().attr("src").unwrap_or("");
            if !src.is_empty() {
                out.push_str(&format!("![{alt}]({src})"));
            }
        }
        "table" => {
            block_sep(out);
            let mut rows: Vec<Vec<String>> = Vec::new();
            collect_table_rows(el, ctx, &mut rows);
            if !rows.is_empty() {
                let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
                for (i, row) in rows.iter().enumerate() {
                    out.push('|');
                    for c in 0..cols {
                        out.push_str(&format!(" {} |", row.get(c).map(|s| s.as_str()).unwrap_or("")));
                    }
                    out.push('\n');
                    if i == 0 {
                        out.push('|');
                        for _ in 0..cols {
                            out.push_str(" --- |");
                        }
                        out.push('\n');
                    }
                }
                out.push('\n');
            }
        }
        _ => walk_children(el, out, ctx),
    }
}

fn collect_table_rows(el: scraper::ElementRef<'_>, ctx: &WalkCtx, rows: &mut Vec<Vec<String>>) {
    for child in el.children() {
        if let scraper::Node::Element(cel) = child.value() {
            let cname = cel.name().to_string();
            let Some(celref) = scraper::ElementRef::wrap(child) else { continue };
            match cname.as_str() {
                "tr" => {
                    let mut row = Vec::new();
                    for cell in celref.children() {
                        if let scraper::Node::Element(inner) = cell.value() {
                            if matches!(inner.name(), "td" | "th") {
                                if let Some(cellref) = scraper::ElementRef::wrap(cell) {
                                    let text = collect_inline_text(cellref, ctx);
                                    row.push(text.split_whitespace().collect::<Vec<_>>().join(" "));
                                }
                            }
                        }
                    }
                    rows.push(row);
                }
                "thead" | "tbody" | "tfoot" => collect_table_rows(celref, ctx, rows),
                _ => {}
            }
        }
    }
}

/// Port of utils.ts getWithPermittedRedirects. Does NOT blindly follow
/// redirects (reqwest Policy::none()); same-host / www-toggle redirects are
/// followed manually (capped at MAX_REDIRECTS hops). A cross-host redirect is
/// surfaced to the caller as a RedirectInfo.
async fn get_with_permitted_redirects(
    url_in: String,
    signal: Option<tokio_util::sync::CancellationToken>,
) -> Result<RawFetchOutcome, String> {
    use futures::StreamExt;

    let mut url = url_in;
    for _depth in 0..=MAX_REDIRECTS {
        // SSRF guard on EVERY hop, not just the initial URL. A permitted
        // www-toggle redirect could resolve to an internal/private IP;
        // re-validate the host before each fetch.
        let host = url::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
            .unwrap_or_default();
        if is_internal_host(&host) {
            return Err(format!("Blocked fetch to internal host: {host}"));
        }

        static FETCH_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default()
        });
        // Say which host is being waited on. A slow one gets the full 60s
        // above, and until now that was 60 seconds of unexplained spinner.
        crate::utils::tokens::set_wait_note(Some(format!("fetching {host}")));
        let _clear = crate::utils::tokens::WaitNoteGuard;
        let resp = FETCH_CLIENT
            .get(&url)
            .header("Accept", "text/markdown, text/html, */*")
            .header("User-Agent", "CodeMason/0.1 (+local)")
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let status = resp.status().as_u16();
        if matches!(status, 301 | 302 | 307 | 308) {
            let location = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let Some(location) = location else {
                return Err("Redirect missing Location header".to_string());
            };
            let redirect_url = url::Url::parse(&url)
                .ok()
                .and_then(|base| base.join(&location).ok())
                .map(|u| u.to_string())
                .unwrap_or(location);
            if is_permitted_redirect(&url, &redirect_url) {
                url = redirect_url;
                continue;
            }
            return Ok(RawFetchOutcome::Redirect(RedirectInfo {
                original_url: url,
                redirect_url,
                status_code: status,
            }));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let code_text = resp.status().canonical_reason().unwrap_or("").to_string();
        // Cap the read at MAX_HTTP_CONTENT_LENGTH; honor the user's abort
        // mid-download — a large/slow fetch was otherwise uncancellable.
        let mut chunks: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if let Some(sig) = &signal {
                if sig.is_cancelled() {
                    return Err("Fetch aborted".to_string());
                }
            }
            let chunk = chunk.map_err(|e| e.to_string())?;
            chunks.extend_from_slice(&chunk);
            if chunks.len() > MAX_HTTP_CONTENT_LENGTH {
                return Err(format!(
                    "Response exceeds maximum content length ({MAX_HTTP_CONTENT_LENGTH} bytes)"
                ));
            }
        }
        return Ok(RawFetchOutcome::Raw { raw: chunks, content_type, code: status, code_text });
    }
    Err(format!("Too many redirects (exceeded {MAX_REDIRECTS})"))
}

enum RawFetchOutcome {
    Raw { raw: Vec<u8>, content_type: String, code: u16, code_text: String },
    Redirect(RedirectInfo),
}

/// Port of utils.ts getURLMarkdownContent. Returns either FetchedContent or a
/// RedirectInfo. Uses the 15-minute URL cache keyed by the ORIGINAL URL.
pub async fn get_url_markdown_content(
    url: &str,
    signal: Option<tokio_util::sync::CancellationToken>,
) -> Result<FetchOutcome, String> {
    if !validate_url(url) {
        return Err("Invalid URL".to_string());
    }

    if let Some(cached) = URL_CACHE.lock().get(url) {
        return Ok(FetchOutcome::Content(cached));
    }

    // HTTP -> HTTPS upgrade (the cache key stays the original URL).
    let mut upgraded = url.to_string();
    if let Ok(mut parsed) = parse_maybe_schemeless(url) {
        if parsed.scheme() == "http" {
            let _ = parsed.set_scheme("https");
            upgraded = parsed.to_string();
        } else if !url.contains("://") {
            upgraded = parsed.to_string();
        }
    }

    let response = get_with_permitted_redirects(upgraded, signal).await?;
    match response {
        RawFetchOutcome::Redirect(r) => Ok(FetchOutcome::Redirect(r)),
        RawFetchOutcome::Raw { raw, content_type, code, code_text } => {
            let html_content = decode_body(&raw, &content_type);
            let markdown_content = html_to_markdown(&html_content, &content_type);
            let entry = FetchedContent {
                content: markdown_content,
                bytes: raw.len(),
                code,
                code_text,
                content_type,
            };
            URL_CACHE.lock().set(url, entry.clone());
            Ok(FetchOutcome::Content(entry))
        }
    }
}

/// Fetch a URL and process its content with the model.
pub struct WebFetchTool;

#[async_trait::async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        WEB_FETCH_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        json!({
            "additionalProperties": false,
            "properties": {
                "url": {
                    "description": "The URL to fetch content from",
                    "title": "Url",
                    "type": "string"
                },
                "prompt": {
                    "description": "The prompt to run on the fetched content",
                    "title": "Prompt",
                    "type": "string"
                }
            },
            "required": ["url", "prompt"],
            "title": "WebFetchInput",
            "type": "object"
        })
    }

    async fn description(&self, _input: &Value) -> String {
        DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn needs_permissions(&self) -> bool {
        true // network access
    }

    async fn check_permissions(&self, _input: &Value, _context: &ToolUseContext) -> PermissionResult {
        // Ask, despite being read-only. Nothing changes locally, but the URL is
        // model-controlled and the request leaves this machine — so a Read of
        // any file (Read has no path restriction) can be posted to an arbitrary
        // host on the next call, with neither step prompting. The SSRF guard
        // below is thorough about reaching INWARD (scheme allowlist, userinfo
        // rejection, DNS-resolved private/loopback/metadata blocking,
        // same-origin redirects) and says nothing about egress, because the
        // upstream boundary for that was per-domain user approval, which this
        // port dropped when it auto-allowed read-only tools.
        //
        // Under -dsp this is never reached: has_permissions_to_use_tool returns
        // before the read-only branch, so skipping permissions still skips this.
        PermissionResult::ask()
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "WebFetch".to_string()
    }

    async fn validate_input(&self, input: &Value, _context: &ToolUseContext) -> ValidationResult {
        let url = input.get("url").and_then(|v| v.as_str()).unwrap_or("");
        let ok = parse_maybe_schemeless(url).map(|p| p.host_str().is_some()).unwrap_or(false);
        if !ok {
            return ValidationResult::failure(format!(
                "Error: Invalid URL \"{url}\". The URL provided could not be parsed."
            ));
        }
        ValidationResult::success()
    }

    /// url, then what we asked of the page (WebFetchTool/UI.tsx: "url: prompt").
    /// The url, truncated for display (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("url")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::formatting::truncate(
            value,
            crate::utils::formatting::TOOL_SUMMARY_MAX_LENGTH,
        ))
    }

    /// "Fetching <url>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Fetching {summary}"),
            None => "Fetching web page".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, verbose: bool) -> String {
        let url = input.get("url").and_then(|v| v.as_str()).unwrap_or("");
        if url.is_empty() {
            return String::new();
        }
        let prompt_raw = input.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let mut prompt = prompt_raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if !prompt.is_empty() && !verbose && prompt.chars().count() > 100 {
            prompt = format!("{}…", prompt.chars().take(100).collect::<String>());
        }
        let mut lines = vec![colorize(url, colors::DIM)];
        if !prompt.is_empty() {
            lines.push(colorize(&prompt, colors::DIM));
        }
        lines.join("\n")
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<crate::tools::base::CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let url = input.get("url").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let user_prompt = input.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let start = std::time::Instant::now();
            let signal = Some(context.abort_controller.clone());

            let response = match get_url_markdown_content(&url, signal.clone()).await {
                Ok(r) => r,
                Err(e) => {
                    yield Ok(ToolResult::result(
                        format!("Error fetching {url}: {e}"),
                        Some(json!({"error": e, "url": url})),
                    ));
                    return;
                }
            };

            // Cross-host redirect: do not follow. Tell the model to re-issue
            // WebFetch with the redirect URL (WebFetchTool.ts:217-249).
            let fetched = match response {
                FetchOutcome::Redirect(r) => {
                    let status_text = match r.status_code {
                        301 => "Moved Permanently",
                        308 => "Permanent Redirect",
                        307 => "Temporary Redirect",
                        _ => "Found",
                    };
                    let message = format!(
                        "REDIRECT DETECTED: The URL redirects to a different host.\n\n\
                         Original URL: {orig}\n\
                         Redirect URL: {redir}\n\
                         Status: {code} {status_text}\n\n\
                         To complete your request, I need to fetch content from the redirected URL. \
                         Please use WebFetch again with these parameters:\n\
                         - url: \"{redir}\"\n\
                         - prompt: \"{prompt}\"",
                        orig = r.original_url,
                        redir = r.redirect_url,
                        code = r.status_code,
                        prompt = user_prompt,
                    );
                    yield Ok(ToolResult::result(
                        message,
                        Some(json!({
                            "url": url,
                            "code": r.status_code,
                            "code_text": status_text,
                            "duration_ms": start.elapsed().as_millis() as i64,
                        })),
                    ));
                    return;
                }
                FetchOutcome::Content(c) => c,
            };

            let mut content = fetched.content.clone();
            let content_type = fetched.content_type.clone();
            let is_preapproved = is_preapproved_url(&url);

            // Preapproved + text/markdown + under cap: return raw content, no
            // model call (WebFetchTool.ts:261-278).
            let result_text = if is_preapproved
                && content_type.contains("text/markdown")
                && content.chars().count() < MAX_MARKDOWN_LENGTH
            {
                content
            } else {
                // Truncate content to avoid "Prompt is too long" errors
                // (utils.ts applyPromptToMarkdown). Python sliced by CHARS.
                if content.chars().count() > MAX_MARKDOWN_LENGTH {
                    content = format!(
                        "{}\n\n[Content truncated due to length...]",
                        content.chars().take(MAX_MARKDOWN_LENGTH).collect::<String>()
                    );
                }
                let params = QueryModelParams {
                    messages: vec![Message::User(create_user_message_text(
                        &make_secondary_model_prompt(&content, &user_prompt, is_preapproved),
                    ))],
                    system_prompt: vec![
                        "You are a helpful assistant that analyzes web page content and answers questions about it."
                            .to_string(),
                    ],
                    tools: vec![],
                    signal: signal.clone(),
                    options: QueryOptions { dangerously_skip_permissions: false, ..Default::default() },
                    ..Default::default()
                };
                let model_response = query_model(params).await;
                if model_response.is_api_error_message {
                    // The Python except path: report the fetch size + failure.
                    let err = crate::utils::messages::get_text_from_message(&Message::Assistant(
                        model_response.clone(),
                    ));
                    format!("Fetched {} chars but failed to process: {err}", content.chars().count())
                } else {
                    let mut text = String::new();
                    for b in &model_response.message.content {
                        if let crate::types::messages::ContentBlock::Text(t) = b {
                            text.push_str(&t.text);
                        }
                    }
                    if text.is_empty() { "No response from model".to_string() } else { text }
                }
            };

            yield Ok(ToolResult::result(
                result_text,
                Some(json!({
                    "url": url,
                    "bytes": fetched.bytes,
                    "code": fetched.code,
                    "code_text": fetched.code_text,
                    "duration_ms": start.elapsed().as_millis() as i64,
                })),
            ));
        })
    }
}

pub static WEB_FETCH_TOOL: WebFetchTool = WebFetchTool;

#[cfg(test)]
mod tests {
    use super::*;

    /// A page served in a legacy CJK charset used to reach the model as
    /// replacement characters — it could not answer about the page and nothing
    /// said why. This is the Korean case specifically, because that is the
    /// environment this is used in.
    #[test]
    fn a_legacy_charset_page_decodes_instead_of_becoming_garbage() {
        let (euckr, _, _) = encoding_rs::EUC_KR.encode("한글 문서입니다");
        let raw = euckr.into_owned();

        // Declared in the header.
        assert_eq!(decode_body(&raw, "text/html; charset=euc-kr"), "한글 문서입니다");

        // The old path, for contrast: this is what the model was being handed.
        let lossy = String::from_utf8_lossy(&raw);
        assert!(lossy.contains('\u{FFFD}'), "the bug must reproduce, or this proves nothing");
    }

    /// Declared in the document when the header is silent, which is how most
    /// legacy pages are actually served.
    #[test]
    fn a_meta_charset_is_honoured_when_the_header_is_silent() {
        let (body, _, _) =
            encoding_rs::EUC_KR.encode("<html><head><meta charset=\"euc-kr\"></head><body>한글</body></html>");
        let raw = body.into_owned();
        assert_eq!(declared_charset(&raw, "text/html").as_deref(), Some("euc-kr"));
        assert!(decode_body(&raw, "text/html").contains("한글"));
    }

    /// The header wins over the document, and an unknown or absent label must
    /// fall back to exactly the old behaviour rather than erroring.
    #[test]
    fn charset_precedence_and_fallback() {
        let utf8 = "<meta charset=\"euc-kr\">한글".as_bytes().to_vec();
        assert_eq!(
            declared_charset(&utf8, "text/html; charset=utf-8").as_deref(),
            Some("utf-8"),
            "the header is authoritative when both are present"
        );
        assert_eq!(declared_charset(b"no declaration here", "text/html"), None);
        // Unknown label: unchanged behaviour, not a panic and not an error.
        assert_eq!(decode_body("plain".as_bytes(), "text/html; charset=not-a-charset"), "plain");
        // A "charset=" appearing late in body prose is not a declaration.
        let late = format!("{}charset=euc-kr", "x".repeat(4096));
        assert_eq!(declared_charset(late.as_bytes(), "text/html"), None);
    }

    #[test]
    fn preapproved_path_boundaries() {
        assert!(is_preapproved_host("docs.python.org", "/3/library"));
        // Path-scoped entry: the segment must match as a segment.
        assert!(is_preapproved_host("vercel.com", "/docs"));
        assert!(is_preapproved_host("vercel.com", "/docs/frameworks"));
        assert!(!is_preapproved_host("vercel.com", "/docs-evil/malware"));
        assert!(!is_preapproved_host("vercel.com", "/pricing"));
        assert!(!is_preapproved_host("github.com", "/torvalds/linux"));
    }

    #[test]
    fn validate_url_rules() {
        assert!(!validate_url("ftp://example.com/file"));
        assert!(!validate_url("https://user:pass@example.com/"));
        assert!(!validate_url("https://localhost/x"));
        assert!(!validate_url("https://127.0.0.1/x"));
        assert!(!validate_url("https://192.168.1.66/x"));
        assert!(!validate_url("https://169.254.169.254/meta"));
        assert!(!validate_url("https://singlelabel/x"));
        assert!(!validate_url(&format!("https://example.com/{}", "a".repeat(2100))));
    }

    #[test]
    fn nat64_unwrap() {
        // NAT64-mapped public address is NOT blocked; mapped private IS.
        let pub_ip: std::net::IpAddr = "64:ff9b::8.8.8.8".parse().unwrap();
        assert!(!blocked_ip(pub_ip));
        let priv_ip: std::net::IpAddr = "64:ff9b::192.168.0.1".parse().unwrap();
        assert!(blocked_ip(priv_ip));
    }

    #[test]
    fn permitted_redirects() {
        assert!(is_permitted_redirect("https://foo.com/a", "https://foo.com/b"));
        assert!(is_permitted_redirect("https://www.foo.com/a", "https://foo.com/b"));
        assert!(is_permitted_redirect("https://foo.com/a", "https://www.foo.com/b"));
        assert!(!is_permitted_redirect("https://foo.com/a", "http://foo.com/b"));
        assert!(!is_permitted_redirect("https://foo.com/a", "https://bar.com/b"));
        assert!(!is_permitted_redirect("https://foo.com/a", "https://foo.com:8443/b"));
        assert!(!is_permitted_redirect("https://foo.com/a", "https://u:p@foo.com/b"));
    }

    #[test]
    fn html_to_md_basics() {
        let html = "<html><head><script>evil()</script><style>b{}</style></head><body>\
            <h1>Title</h1><h3>Sub</h3><p>Hello <strong>world</strong> and <em>more</em>.</p>\
            <ul><li>one</li><li>two</li></ul>\
            <ol><li>first</li><li>second</li></ol>\
            <p><a href=\"https://x.com\">link</a> and <code>inline</code></p>\
            <pre><code>let x = 1;\nlet y = 2;</code></pre>\
            <blockquote>quoted text</blockquote>\
            <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>\
            </body></html>";
        let md = html_to_markdown(html, "text/html");
        assert!(md.contains("Title\n====="), "{md}");
        assert!(md.contains("### Sub"), "{md}");
        assert!(md.contains("**world**"));
        assert!(md.contains("*more*"));
        assert!(md.contains("* one\n* two"));
        assert!(md.contains("1. first\n2. second"));
        assert!(md.contains("[link](https://x.com)"));
        assert!(md.contains("`inline`"));
        assert!(md.contains("```\nlet x = 1;\nlet y = 2;\n```"));
        assert!(md.contains("> quoted text"));
        assert!(md.contains("| A | B |"));
        assert!(md.contains("| --- | --- |"));
        assert!(!md.contains("evil()"));
        assert!(!md.contains("b{}"));
        // \n{3,} collapsed to exactly two
        assert!(!md.contains("\n\n\n"));
    }

    #[test]
    fn non_html_passthrough() {
        assert_eq!(html_to_markdown("# already md", "text/markdown"), "# already md");
    }

    #[test]
    fn cache_ttl_and_lru() {
        let mut cache = UrlCache { order: Vec::new(), map: HashMap::new(), bytes: 0 };
        let entry = FetchedContent {
            content: "x".repeat(100),
            bytes: 100,
            code: 200,
            code_text: "OK".into(),
            content_type: "text/html".into(),
        };
        cache.set("https://a.com", entry.clone());
        assert!(cache.get("https://a.com").is_some());
        // expire it manually
        cache.map.get_mut("https://a.com").unwrap().0 -= CACHE_TTL_S + 1.0;
        assert!(cache.get("https://a.com").is_none());
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn secondary_prompt_shapes() {
        let p = make_secondary_model_prompt("MD", "ask", false);
        assert!(p.starts_with("\nWeb page content:\n---\nMD\n---\n\nask\n\n"));
        assert!(p.contains("strict 125-character maximum"));
        let p2 = make_secondary_model_prompt("MD", "ask", true);
        assert!(p2.contains("Include relevant details, code examples, and documentation excerpts"));
    }
}
