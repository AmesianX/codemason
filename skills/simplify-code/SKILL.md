---
name: simplify-code
description: Parallel multi-agent cleanup of recent code changes (quality, not bug-hunting).
---

# Simplify Code — parallel review & cleanup

Review the code that changed and remove waste — duplication, dead code,
needless complexity, inefficiency — without changing behavior. This is a
*quality* pass, not a bug hunt.

## When to use

After writing or editing a chunk of code, before committing, or when the user
asks to "clean up", "simplify", or "tidy" recent changes.

## Phase 1 — Identify the changes

Capture the diff to review, choosing the source by what the user asked for:

```bash
git diff                 # default: uncommitted working-tree changes
git diff HEAD            # if the above is empty, include staged
git diff --staged        # "staged changes"
git diff HEAD~1          # "the last commit"
git diff main...HEAD     # "this branch" / "my PR"
git diff -- path/to/f    # specific files
```

If there is no diff and no repo, fall back to the files the user named or that
were edited this session. If there is genuinely nothing changed, say so and stop.
If the diff is very large (>~2000 lines), warn the user and offer to scope it
down before fanning out.

## Phase 2 — Review from four angles

Launch four reviewers **in parallel** with the `Agent` tool (one message, four
Agent calls), each given the COMPLETE diff plus the repo path so they can search
the wider codebase. The four angles:

1. **Duplication & dead code** — repeated logic, unused imports/vars/functions,
   commented-out code, pass-through wrappers.
2. **Needless complexity** — over-abstraction, deep nesting, premature
   generalization, YAGNI violations.
3. **Efficiency** — N+1 patterns, repeated work, needless allocation/IO, wrong
   data structure.
4. **Reuse & consistency** — reinvented helpers that already exist, patterns that
   diverge from the surrounding code.

**If you cannot delegate** (leaf context, delegation disabled): do all four
angles yourself, sequentially, and say in the summary that it was a single-pass
inline review, not the parallel fan-out.

Tell every reviewer to search the codebase for evidence (not reason from the
diff alone), to apply **Chesterton's Fence** — `git blame` a line before
proposing its removal; mark `confidence: low` if the original purpose is unclear —
and to report each finding as:

```
file:line → problem → cost (what it duplicates/wastes/makes harder) → fix | confidence: high/med/low | risk: SAFE/CAREFUL/RISKY
```

- **SAFE** = provably behavior-neutral (unused imports, dead code, pass-throughs) → auto-apply.
- **CAREFUL** = semantics-preserving (rename local, flatten ternary, extract helper) → apply, then run tests.
- **RISKY** = may change behavior or break a public contract → flag for the user, do NOT auto-apply.

## Phase 3 — Apply and verify

Dedupe findings across reviewers. Apply SAFE and CAREFUL fixes, run the test
suite (`Bash`) after the CAREFUL ones, and present the RISKY ones to the user
with the reviewer's reasoning. Summarize what changed and what you deliberately
left alone.
