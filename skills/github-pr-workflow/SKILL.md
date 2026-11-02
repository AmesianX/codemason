---
name: github-pr-workflow
description: GitHub PR lifecycle — branch, commit, push, open PR, watch CI, merge (via gh CLI).
---

# GitHub Pull Request Workflow

Take a change from a branch to a merged PR. Uses the `gh` CLI; check it is
authenticated first with `gh auth status` (if not, `gh auth login`).

## 1. Branch

Never build on the default branch. Start fresh from an up-to-date base:

```bash
git checkout main && git pull
git checkout -b feat/short-description
```

## 2. Commit

Make the change with `Write`/`Edit`, stage explicitly, and write a
Conventional Commit message:

```bash
git add path/to/changed_files
git commit -m "feat(auth): add JWT-based login

- login/register endpoints, password hashing
- auth middleware for protected routes
- unit tests for the auth flow"
```

Format: `type(scope): short imperative subject` (≤72 chars), blank line, body
explaining WHY, wrapped ~72 cols. Types: `feat fix refactor docs test ci chore
perf`.

## 3. Push & open the PR

```bash
git push -u origin HEAD
gh pr create --title "feat(auth): add JWT-based login" --body "$(cat <<'MD'
## Summary
- what changed and why

## Test plan
- [ ] unit tests pass
- [ ] manual check of X

Closes #42
MD
)"
```

Useful flags: `--draft`, `--reviewer u1,u2`, `--label enhancement`, `--base develop`.
Read the PR back (`gh pr view --json number,title,baseRefName,headRefName,files`)
and confirm the number, base, title, and files are what you intended.

## 4. Watch CI honestly

The PR is what dispatches CI, and CI latency is the long pole — open it as soon
as the work is ready, then watch:

```bash
gh pr checks --watch      # polls until all checks finish
gh pr checks              # one-shot snapshot
```

Report status only from live evidence. Never say "green" or "merged" without
having just observed that exact state.

## 5. Fix CI failures

```bash
gh run list --branch "$(git branch --show-current)" --limit 5
gh run view <run-id> --log-failed
```

Distinguish failures your diff introduced from pre-existing baseline/infra
failures — reproduce on `main` when unsure, and rerun only for genuine infra
flakes. Fix, commit, push; CI re-runs automatically.

## 6. Merge

```bash
gh pr merge --squash --delete-branch      # cleanest for feature branches
# or let it land when checks pass:
gh pr merge --squash --auto --delete-branch
git checkout main && git pull             # sync local afterward
```

Only merge when the user asked for it (or a durable policy allows it) and checks
are actually green.
