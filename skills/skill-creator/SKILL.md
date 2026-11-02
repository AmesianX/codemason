---
name: skill-creator
description: Author a codemason skill — scaffold a SKILL.md, write good frontmatter and body.
---

# Skill Creator

Create a new codemason skill. A skill is a `SKILL.md` markdown file with YAML
frontmatter; its body IS the prompt that expands into the conversation when the
skill is invoked (inline) or handed to a sub-agent (fork). A good skill makes the
agent's process **predictable** — it reliably follows the same useful discipline.

## When to use

The user wants to capture a reusable workflow, checklist, or domain procedure as
a skill; or wants to edit an existing skill. Don't write a skill whose only
content is "load skill X instead" — a routing/index skill just adds an
indirection hop; the skill catalog already lists everything.

## Where skills live (pick a root)

| Root | Scope | How |
|------|-------|-----|
| `~/.codemason/skills/<name>/` | user, native (default) | `codemason skill new <name>` |
| `<cwd>/.codemason/skills/<name>/` | project | `codemason skill new <name> --project` |
| `~/.claude/skills/` or `<cwd>/.claude/skills/` | imported (opt-in) | add `--claude` |
| `codemason/skills/<name>/` | bundled, ships with codemason | `Write` + `git add` (repo checkout only) |

`.codemason` wins over an imported `.claude`, user wins over project, and both
win over bundled — so a user skill overrides a bundled one of the same name.

The `.claude` roots belong to another product and are read only when
`import_claude_dirs` is turned on (`CODEMASON_IMPORT_CLAUDE_DIRS=1`, or the key
in `~/.codemason/config.json`); the default is off. `--claude` still writes
there — naming it on the command line is the opt-in for that one call — but the
skill will not load until importing is enabled.

## 1. Scaffold

```bash
codemason skill new my-skill --description "One line on when to use it." \
  --arguments "target"        # optional positional arg names
```

This writes a starter `SKILL.md`. Edit it with `Edit`/`Write`.

## 2. Frontmatter

codemason reads these keys (only `name` and `description` are required):

```yaml
---
name: my-skill                  # lowercase, hyphens, matches the directory name
description: Concise capability statement, ideally under ~60 chars.
arguments: target other         # optional: space-separated positional arg names
context: inline                 # inline (default) = runs in this conversation
                                # fork = runs in a sub-agent, returns a result
agent: general-purpose          # optional: which agent runs a fork skill
disable-model-invocation: false # true = user can /invoke it, model cannot auto-pick it
---
```

**description rules** — it's paid for on every turn and shown in the skill index:
one sentence stating the *capability* (not the implementation), don't repeat the
skill name, no marketing words ("powerful", "comprehensive", "seamless"). If it
contains a `:`, wrap the whole value in double quotes or YAML mis-parses it.

## 3. Body

The body after the closing `---` is the skill's instructions. Recommended
sections (use what fits — a simple skill needs fewer):

```
# <Skill> — one line
2-3 sentences: what it does, what it does NOT do.

## When to use     — bulleted triggers, plus "Don't use for:" counter-triggers
## Procedure       — numbered steps, each ending in a checkable completion criterion
## Pitfalls        — known limits; things that look broken but aren't
## Verification    — how to prove the skill worked
```

**Arguments** declared in `arguments:` (and positionals `${1}`, `${2}`) are
substituted into the body at invocation; `${CODEMASON_SKILL_DIR}` expands to the
skill's own directory (use it to reference bundled reference files).

**Reference codemason's tools by name** where the skill needs a capability:
`Read`, `Grep`, `Glob`, `Bash`, `Write`, `Edit`, `Agent` (sub-agents),
`WebSearch`, `WebFetch`, and any MCP tools (`mcp__<server>__<tool>`). Never bake
in machine-local absolute paths — write repo-relative or `${CODEMASON_SKILL_DIR}`.

## Writing-quality principles

1. **Optimize for process predictability.** If a line doesn't change behavior, cut it.
2. **End steps with completion criteria** — checkable ("every changed file
   accounted for") beats vague ("summarize changes").
3. **Co-locate rules with the concept they govern.**
4. **Use strong leading words** (`root cause`, `tight loop`, `regression test`)
   instead of long repeated explanations.
5. **Prune no-ops.** "Be careful" and "use best practices" don't change behavior —
   replace with a checkable criterion or delete.
6. Keep it tight: ~100 lines for a simple skill, ~200 for a complex one. Put bulky
   material in a `references/*.md` next to the SKILL.md and point to it.

## 4. Verify

```bash
codemason skill show my-skill        # re-read the file
codemason skill list                 # confirm it's discovered and model-invocable
```

If it doesn't appear as invocable, check: frontmatter parses (starts with `---`
on the first byte, closes with `---`), `name` and `description` present, and
`disable-model-invocation` is not accidentally true. Then dry-run it with
`/my-skill` and refine the body based on what the model actually did.
