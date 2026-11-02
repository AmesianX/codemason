---
name: github-issues
description: Create, view, triage, label, assign, and close GitHub issues via the gh CLI.
---

# GitHub Issues

Manage issues on a GitHub repo with the `gh` CLI (`gh auth status` to confirm
login first).

## Viewing

```bash
gh issue list                                  # open issues
gh issue list --label bug --assignee @me       # filtered
gh issue list --search "crash in parser in:title"
gh issue view <N> --comments                   # body AND the live thread
```

Always read the comments, not just the body: the body is a snapshot from filing
time; the newest comments carry the current state — partial fixes, new analysis,
maintainer decisions, questions aimed at you.

## Creating

Write a clear, reproducible issue. Bug and feature templates:

**Bug**
```markdown
## Description
One or two sentences on what's wrong.

## Steps to reproduce
1. ...
2. ...

## Expected
What should happen.

## Actual
What happens instead (error text, stack trace).

## Environment
version / OS / relevant config.
```

**Feature**
```markdown
## Problem
The need or pain, from the user's point of view.

## Proposed solution
What you suggest.

## Alternatives considered
Other options and why they're worse.
```

```bash
gh issue create --title "parser crashes on empty input" \
  --body "$(cat <<'MD'
## Description
...
MD
)" --label bug
```

## Triage & management

```bash
gh issue edit <N> --add-label "priority:high" --add-label bug
gh issue edit <N> --remove-label needs-triage
gh issue edit <N> --add-assignee @me
gh label list                       # what labels the repo has
gh issue comment <N> --body "Reproduced on main @ <sha>; root cause is ..."
gh issue close <N> --reason completed
gh issue reopen <N>
```

## Good triage habits

Reproduce before labeling `bug`; link duplicates (`gh issue comment <N> --body
"Duplicate of #M"` then close); when closing via a fix, comment the PR/commit
link so the reporter gets a traceable resolution; keep labels consistent with
what `gh label list` already offers rather than inventing new ones.
