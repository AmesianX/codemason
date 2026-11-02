---
name: spike
description: Throwaway experiments to validate feasibility before committing to a build.
---

# Spike

A spike is disposable code written to answer a feasibility question — "will this
approach work?" — not to ship. Build the smallest thing that produces observable
output, learn from it, then throw it away.

## When NOT to use

Skip spikes when the path is already clear, or when the task is production work.
A spike is for genuine uncertainty. Never let spike code become the real
implementation — it has no tests, no error handling, and hardcoded everything.

## Core method

```
decompose → (align) → research → build → verdict   (iterate on findings)
```

### 1. Decompose

Break the idea into **2–5 independent feasibility questions**; each is one spike.
Frame each as Given/When/Then with observable output, and order by risk — the
spike most likely to KILL the idea runs first.

| # | Spike | Validates (Given/When/Then) | Risk |
|---|-------|-----------------------------|------|
| 001 | ws-streaming | Given a WS conn, when tokens stream, then client sees chunks <100ms | High |

A **comparison** spike answers one question with different approaches (suffix
`a`/`b`). Skip decomposition only if the user already knows exactly what to spike.

### 2. Align (multi-spike ideas)

Show the table and ask the user to drop/reorder/re-frame before you write code.

### 3. Research (per spike, before building)

Enough to pick the right approach, then stop. Brief the spike in 2–3 sentences,
surface competing tools/libraries (maintained vs abandoned), pick one and say
why. Use `WebSearch`/`WebFetch` to find and read the actual docs, and `Bash`
(`pip show X`, `npm ls X`) to see what's already installed. Skip research for
pure logic with no external dependency.

### 4. Build

One standalone directory per spike, under `spikes/`:

```
spikes/001-ws-streaming/{README.md, main.py}
```

**Bias toward something the user can interact with**, in order of preference:
a runnable CLI that prints observable output; a minimal HTML page; a one-endpoint
server; or a test with recognizable assertions. A spike whose only output is
"it works" in a log has failed to be a spike.

**Depth over speed.** Never declare success after one happy-path run — probe edge
cases and follow surprising findings; the verdict is only trustworthy if the
investigation was honest. **Hardcode everything** — no build tools, Docker, env
files, or config systems unless the question is specifically about them.

### 5. Verdict

Per spike, record one of: **VALIDATED** (approach works, here's the evidence),
**PARTIAL** (works with caveats X, Y), **INVALIDATED** (doesn't work, because Z).
End with a recommendation on whether/how to proceed to a real build — and delete
or clearly quarantine the spike code so it never ships.
