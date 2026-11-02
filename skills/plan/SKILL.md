---
name: plan
description: Write a markdown implementation plan to .codemason/plans/ — design only, no code.
---

# Plan

Produce a written implementation plan. **Do not write or edit production code
while running this skill** — the output is the plan document, nothing else.

## Core behavior

- Investigate first, then write. Read the relevant code before proposing changes.
- Design only: no edits to source files, no running of build/deploy commands.
- The deliverable is one markdown file the user (or a later session) can execute
  step by step.

## Investigate before writing

```bash
# understand the structure and the neighbours of what you'll change
ls and Glob for the relevant area
Grep for the feature's current call sites and similar existing features
Read the key files and their tests
```

Base the plan on what the code actually does, not on assumptions.

## Save location

Write the plan to `.codemason/plans/<short-kebab-title>.md` (create the
directory if needed). Tell the user the path.

## Document structure

```markdown
# <Feature> — Implementation Plan

## Goal
One paragraph: what we're building and why. The user-visible outcome.

## Context
Key files, current behavior, constraints, and anything non-obvious learned
while investigating (with `file:line` references).

## Approach
The chosen design in prose, and the main alternative(s) considered with the
reason each was rejected.

## Steps
Bite-sized, ordered, independently verifiable tasks. Each step names the files
it touches and how to confirm it works (the test or command).
1. ...
2. ...

## Risks / open questions
What could go wrong, and any decision that still needs the user.
```

## Principles

- **Bite-sized steps.** Each step should be a small, reviewable change that
  leaves the system working — not "build the whole backend".
- **YAGNI.** Plan only what the goal needs; don't design for imagined futures.
- **Verifiable.** Every step states how you'll know it's done (a test, a command,
  an observable behavior).
- Match the plan's depth to the task: a small change needs a short plan.

## When done

Present the path and a brief summary. Ask whether to proceed with implementation
(a separate step, outside this skill).
