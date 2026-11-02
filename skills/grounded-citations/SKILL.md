---
name: grounded-citations
description: Ground answers and documents in cited, verifiable sources — register sources, cite inline, verify.
---

# Grounded Citations

When an answer or document makes factual claims from external sources, every
such claim must trace to a source the reader can check. Register sources as you
retrieve them and cite while drafting — reconstructing citations from memory
afterward is the failure mode this skill exists to prevent.

## When to use

Research write-ups, literature summaries, any deliverable whose claims the reader
must be able to verify, and explicitly whenever the user asks for citations or
fact-checking. Claims from your own reasoning need no citation — this is about
external facts.

## Procedure

**① Keep a source ledger.** Maintain a numbered list of the sources you actually
retrieved. Start it at the beginning of the task (or reuse the numbering already
in a draft you're continuing). Each entry is `[n] <title> — <URL>`.

**② Register every source at retrieval time.** After each `WebSearch`,
`WebFetch`, or firecrawl scrape/search, add the URLs you'll rely on to the ledger
*before* writing prose. Only pages you actually opened and read go in — never a
URL you merely saw in a results list.

**③ Cite while drafting.** Put the bracketed id(s) immediately after each
sentence a source supports:

```
Ice floats because it is less dense than liquid water.[1][2]
```

- No space before the bracket; each id in its own brackets; at most 3 per sentence.
- Cite per sentence, not one dump at the end.
- Only ids that exist in the ledger. Never invent an id or a URL.
- Conflicting sources: present both readings, each with its own id, and say which
  you weight and why.
- Quote exact figures, dates, and names as the source states them. Flag gaps
  explicitly ("no source found for X") instead of smoothing them over.

**④ Append a Sources block** built from the ledger, not retyped from memory:

```
## Sources
[1] Title — https://example.com/a
[2] Title — https://example.com/b
```

**⑤ Verify before delivering.** Check that every `[n]` in the prose exists in the
Sources block and vice versa, that no id is invented, and that load-bearing
claims are actually cited. Fix and re-read.

## Fact-checking mode

For medical, legal, financial, safety, or disputed claims — or when the user asks
to fact-check — upgrade from citations to **evidence**:

- **Attach a verbatim quote per claim.** After fetching a page, copy the exact
  sentence(s) that carry each claim (copy-paste, never retype) and keep them with
  the source. A paraphrase is not evidence.
- **Mark unsourceable claims `[unverified]`** instead of citing them — declared
  provenance for every claim is the goal, not a fake citation.
- **Cross-check disputed facts** against a second *independent* source. One
  source is reporting; two independent sources are corroboration.
- **Render an evidence block** that prints each source's supporting quote beneath
  its URL, so the deliverable shows claim → source → exact text with nothing
  taken on faith.

## Pitfalls

Citing a search-result snippet you never opened; citing from memory after
writing; inventing plausible-looking URLs; one citation dump at the end instead
of per-sentence; presenting a stale archived snapshot as the live page (see the
`blocked-page-recovery` skill for provenance discipline).
