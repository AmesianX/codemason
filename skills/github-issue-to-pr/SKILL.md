---
name: github-issue-to-pr
description: Carry a GitHub issue end-to-end to a tested, verified PR with honest CI state.
---

# GitHub Issue → Pull Request

Turn an issue into a tested, verified PR. This skill owns the end-to-end
discipline — premise validation, duplicate sweeps, class-level fixes, honest CI
reporting; `github-pr-workflow` owns the PR mechanics and
`systematic-debugging` / `test-driven-development` own the fix mechanics.

## When to use

"Fix issue #123 and open a PR", "implement this feature request", "take this bug
from issue to green CI". Not for reviewing an existing PR or answering a code
question with no requested change.

## Procedure

### 1. Read the live issue — body AND full thread

`gh issue view <N> --comments`. The body is a snapshot; the newest comments carry
the live state (partial fixes merged, new root-cause analyses, maintainer
decisions, questions aimed at you). Read repo instructions too (`AGENTS.md`,
`CONTRIBUTING`). Done when the requested behavior, non-goals, and any open thread
questions are known.

### 2. Sweep for duplicate / existing work

```bash
gh pr list --search "#<N>" --state all
gh pr list --search "<subsystem> <symptom>" --state open   # + a synonym variant
git log --oneline -20 -- <relevant files>
```

Popular issues attract multiple independent fixes; a duplicate wastes the work.
Done when you know every open PR and recent commit touching this issue.

### 3. Validate the premise against current code

Reproduce the bug (or demonstrate the missing behavior) on the current default
branch with a failing test, tracing the path via `Grep`/`Read`. Then check the
second question: is the "bug" actually deliberate design? `git log -p -S
"<symbol>"` on the code the issue wants changed and read the original intent — a
restriction is often the feature. Challenge stale issue prose instead of
implementing it blindly. Done when the root cause is demonstrated in current code
AND the change doesn't fight an intentional design.

### 4. Define acceptance & risk

List acceptance criteria, interface/migration/compatibility/security impact, and
rollback. Map every criterion to a test or explicit verification.

### 5. Implement the smallest complete change — and fix the class

Work on an isolated branch. Load `test-driven-development` or
`systematic-debugging` as the bug class calls for. Add regression tests first,
then implement. Then `Grep` for the SAME bug shape at sibling call sites and fix
the whole class in this PR — an incomplete fix that leaves known siblings broken
is worse than none. Every changed line must trace to the issue; no drive-by
cleanup.

### 6. Prove the regression test bites

Temporarily restore the old (broken) behavior, run the new test, confirm it
FAILS; restore the fix, confirm it passes. A test that passes with and without
the fix proves nothing.

### 7. Quality gates, then open the PR immediately

Run the formatter, lint, typecheck, and the repo's test entrypoint on affected
areas; run the `code-review` skill on the diff. Then push and open the PR right
away (`github-pr-workflow`) — the PR dispatches CI and CI latency is the long
pole. The PR body links the issue with problem, approach, tests, risk, and
exclusions. Read the PR back and verify head SHA, base, title, files.

### 8. Shepherd CI honestly and close the loop

`gh pr checks` / `gh run view --log-failed`. Distinguish your failures from
pre-existing baseline/infra ones; reproduce on `main` when unsure; rerun once for
genuine infra flakes only. Never claim "green"/"merged" without live evidence.
When it lands, comment on the issue with the PR link and a one-line explanation.

## Verification checklist

- [ ] Full issue thread read; newest comment state reflected in the plan.
- [ ] Duplicate-PR sweep run (issue number + a keyword variant).
- [ ] Premise reproduced on current code; design intent checked via git history.
- [ ] Regression test proven to fail without the fix.
- [ ] Sibling call sites fixed or explicitly ruled out.
- [ ] Every changed line traces to the issue.
- [ ] CI state reported from live evidence only; issue commented with the PR link.
