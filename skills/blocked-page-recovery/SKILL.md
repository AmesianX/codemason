---
name: blocked-page-recovery
description: Recover blocked, paywalled, or WAF'd pages via a fallback ladder, preserving provenance.
---

# Blocked-Page Recovery

When a `WebFetch` returns a block page, paywall, CAPTCHA, or empty JS shell, work
down this ladder instead of giving up — and always record *how* you got the copy,
because a snapshot is not the live page.

## The ladder

```
1. firecrawl scrape   — server-side render of the LIVE page (handles JS/soft blocks)
2. Wayback Machine     — archive.org "available" API (snapshot + timestamp)
3. archive.today       — domain rotation: archive.ph → .md → .li → .is (paywalls)
4. API / feed pivot    — /api, /graphql, .json, RSS, sitemap on the same host
5. Firecrawl search    — find the same content mirrored elsewhere
```

## Provenance discipline (non-negotiable)

| Route | Provenance | How to cite |
|-------|-----------|-------------|
| firecrawl scrape / live fetch | `live` | cite normally |
| Wayback / archive.today | `snapshot` | cite WITH the snapshot date ("as archived 2026-08-06"); never present it as live |

If the user needs *current* data (prices, availability, breaking news), a
snapshot is context, not an answer — say so and note its age.

## Routes

**1. firecrawl scrape (try first for live content).** The firecrawl scrape tool
renders the page server-side and returns markdown, defeating most JS-gated and
soft-blocked pages. This is usually the fastest genuine recovery.

**2. Wayback Machine** (best provenance for archived content):

```bash
# closest snapshot as JSON, then fetch archived_snapshots.closest.url
curl -sL "https://archive.org/wayback/available?url=<URL>"
# enumerate snapshots / recover deleted pages via the CDX index
curl -sL "https://web.archive.org/cdx/search/cdx?url=<URL>&output=json&limit=10"
```

Works for any publicly crawled URL. Fails for robots-blocked, never-crawled, or
JS-only pages. CDX returns 503 under load — fall back to the `available` API,
don't retry-hammer.

**3. archive.today** (paywalled news that Wayback lacks). Rate-limits (429) and
rotates domains, so iterate:

```bash
for d in archive.ph archive.md archive.li archive.is; do
  code=$(curl -sL --max-time 20 "https://$d/newest/<URL>" -o /tmp/pg.html -w "%{http_code}")
  [ "$code" = 200 ] && grep -qi "<expected title words>" /tmp/pg.html && break
done
```

**4. API / feed pivot.** WAFs guard the HTML far more than the data behind it.
After 2–3 blocked HTML attempts, stop fighting it and look for `/api/…`,
`/graphql`, a `.json` variant, an RSS/Atom feed (`/feed`, `/rss`), or
`/sitemap.xml`.

**5. Firecrawl search.** Search the exact headline/quote — the same content is
often mirrored on a site that isn't blocked.

## Validate the body, not the status code

A 200 with a plausible body is often NOT the page:

- **Google Cache is dead** (since mid-2024) — `webcache.googleusercontent.com`
  returns a Search interstitial, never a cache. Don't use it.
- **AMP caches** (`*.cdn.ampproject.org`) often return a ~300-byte redirect stub
  back to the blocked URL — treating it as success creates a fetch loop.
- **Rate-limit pages** (archive.today 429) are multi-KB HTML. Check for the
  target's ACTUAL content (title words, expected strings), not just byte size.

Reject any body under a sane size floor, any meta-refresh/JS-redirect stub
pointing back at the original host, and interstitial titles ("Just a moment",
"Redirecting", "Attention Required"). Do not route through anonymous proxy
relays — they lie about success and leak the request.
