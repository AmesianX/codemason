---
name: systematic-debugging
description: 4-phase root-cause debugging — understand the bug before fixing it.
---

# Systematic Debugging

Random fixes waste time and create new bugs. Quick patches mask the real issue.

**Core principle: ALWAYS find the root cause before attempting a fix. A symptom
fix is a failure.**

## The Iron Law

```
NO FIXES WITHOUT ROOT-CAUSE INVESTIGATION FIRST.
```

If you have not completed Phase 1, you may not propose a fix.

## The feedback-loop rule

The feedback loop IS the debugging work. Before reading code to build a theory,
create or identify a **tight** command that goes RED on the user's exact symptom
and GREEN when the bug is fixed. A tight loop is fast, deterministic,
runnable by you, and specific enough to catch *this* bug — not merely "doesn't
crash". When a clean repro is hard, spend disproportionate effort building the
loop; guessing without a red-capable loop is the failure mode this skill exists
to prevent.

## When to use

Any technical issue: test failures, production bugs, unexpected behavior,
performance problems, build/CI failures, flaky tests. Use it ESPECIALLY when
under time pressure, when "just one quick fix" seems obvious, or when a previous
fix did not work. Simple bugs have root causes too.

## Phase 1 — Root-cause investigation

1. **Read the error carefully.** Full stack trace, line numbers, file paths,
   error codes. The message often contains the answer. Use `Read` on the named
   files and `Grep` for the error string.
2. **Build the tight feedback loop.** A failing test at the seam is best; else a
   curl/HTTP script against a dev server, a CLI invocation diffing output, a
   headless-browser script, a replayed trace, or a throwaway harness that boots
   the smallest slice and calls the failing path. Run it with `Bash`. For a
   flaky bug the immediate goal is a *higher reproduction rate* (loop it 100×,
   add stress, pin time, seed randomness) — a 50% flake is debuggable, a 1% one
   usually is not.
3. **Trace the data flow.** For deep errors, `Grep` the references and follow the
   value from source to symptom.

**Phase 1 completion checklist:** you can state WHAT is wrong, WHERE it happens,
and WHY — and you have a loop that goes red on it. STOP: do not proceed until
you understand *why*.

## Phase 2 — Pattern analysis

Minimize the reproduction to the smallest input that still fails. Find a working
example nearby (`Grep` for comparable code), compare it against the broken path,
and name the exact difference and the dependencies involved.

## Phase 3 — Hypothesis and testing

Form ranked, falsifiable hypotheses ("if X, then changing Y flips the loop").
Test the cheapest one minimally, confirm the loop's color changes as predicted,
and discard a hypothesis the moment evidence contradicts it. If you don't know,
say so and gather more data — do not guess.

## Phase 4 — Implementation

1. Make the loop RED with a committed test that captures the root cause.
2. Implement a SINGLE fix at the root cause, not the symptom.
3. Verify the loop is GREEN and the full test suite still passes.
4. **Rule of three:** if three distinct fixes have failed, STOP and question the
   architecture — the bug may be a design problem. Discuss with the user before
   trying more fixes.

## Red flags — STOP and return to Phase 1

Trying fixes to "see if they work"; changing code you don't understand; adding
`try/except` to make an error disappear; the same bug reappearing after a fix;
"it's probably just X" without evidence.
