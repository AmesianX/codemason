---
name: github-code-review
description: Review a GitHub pull request — fetch the diff, inspect, post an inline/summary review via gh.
---

# GitHub Code Review

Review someone else's pull request (or your own local changes before pushing).
Uses the `gh` CLI. For a pre-commit gate on your OWN change use the `code-review`
skill; this one is about reviewing a PR on GitHub.

## Reviewing local changes (pre-push)

```bash
git diff --staged            # what would be committed
git diff main...HEAD         # what a PR would contain
git diff --stat main...HEAD  # per-file insertions/deletions
```

Quick machine scan before reading:

```bash
git diff main...HEAD | grep -nE 'TODO|FIXME|console\.log|print\(|import pdb'   # leftover debug
git diff main...HEAD | grep -nE '(api|secret|token|password)[_-]?key?\s*=\s*["'\'']'  # secrets
git diff main...HEAD | grep -nE '^(<<<<<<<|=======|>>>>>>>)'                    # conflict markers
```

## Reviewing a PR on GitHub

```bash
gh pr view <N> --json title,author,body,baseRefName,headRefName
gh pr diff <N>                       # the full diff
gh pr view <N> --json files -q '.files[].path'   # changed files
gh pr checkout <N>                   # check it out locally to run tests / Read / Grep
```

Read the diff in the context of the wider codebase — `Grep` for the changed
symbols, `Read` the surrounding files, and run the tests. Cross-file issues hide
in the gaps between hunks.

## Review checklist

- **Correctness:** logic matches intent; edge cases and error paths handled.
- **Security:** no secrets; input validated; parameterized SQL; no `eval`/`exec`
  or unsafe deserialization on untrusted input; paths guarded against traversal.
- **Tests:** new behavior is covered; the tests would actually fail without the change.
- **Clarity:** names, duplication, dead code, needless complexity.
- **Scope:** every changed line traces to the PR's stated purpose; no drive-by churn.

## Posting the review

```bash
# inline comment on a specific line
gh pr review <N> --comment --body "..."          # general comment
gh pr review <N> --request-changes --body "$(cat <<'MD'
## Blocking
- path/to/file.py:42 — <problem> → <fix>

## Suggestions (non-blocking)
- ...
MD
)"
gh pr review <N> --approve --body "LGTM — ..."   # only when it genuinely passes
```

Separate **blocking** findings (must fix before merge: correctness, security,
failing tests) from **non-blocking** suggestions, each with a `file:line`
reference. Approve only when the blocking list is empty.
