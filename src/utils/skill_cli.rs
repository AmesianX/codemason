//! ``codemason skill`` management subcommand.
//! Equivalent to codemason/utils/skill_cli.py.
//!
//! Intercepted at the top of cli::main() (before arg parsing, which owns the
//! positional ``prompt``). The skill *runtime* — discovery, frontmatter
//! parsing, argument substitution, the Skill tool — lives in utils/skills.rs
//! and tools/skill_tool.rs; this only adds the ergonomics for authoring and
//! managing the SKILL.md files it reads.
//!
//! Skills live in ``<skills_dir>/<name>/SKILL.md`` where skills_dir is
//! ``~/.codemason/skills`` (user, default) or ``<cwd>/.codemason/skills``
//! (``--project``). ``--claude`` writes to the matching ``.claude`` dir
//! instead; naming it on the command line is itself the opt-in, but the
//! runtime only READS those dirs when config.import_claude_dirs is on.
//!
//!     codemason skill list
//!     codemason skill show <name>
//!     codemason skill new <name> [--description D] [--arguments "a b"]
//!                                 [--context inline|fork] [--agent NAME] [--project]
//!     codemason skill remove <name> [--project]
//!     codemason skill enable <name>
//!     codemason skill disable <name>

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::utils::config::CONFIG_DIR;
use crate::utils::skills::{bundled_skills_dir, discover_skills, model_invocable_skills, refresh_skills};
use crate::utils::state::get_cwd;

/// (label, dir) for every skill root, in the same first-wins precedence the
/// runtime uses (utils/skills::skill_dirs): user before project, ours before
/// any imported layout.
///
/// The foreign roots appear only when importing them is enabled, so `list`
/// shows what the runtime will actually load rather than advertising
/// directories it is going to ignore. `--claude` on `install` is unaffected:
/// naming it on the command line IS the opt-in for that one call.
fn roots() -> Vec<(&'static str, PathBuf)> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    let cwd = PathBuf::from(get_cwd());
    let import_foreign = crate::utils::config::get_config().import_claude_dirs;
    let mut all: Vec<(&'static str, PathBuf)> =
        vec![("user:.codemason", CONFIG_DIR.join("skills"))];
    if import_foreign {
        all.push(("user:.claude", home.join(".claude").join("skills")));
    }
    all.push(("project:.codemason", cwd.join(".codemason").join("skills")));
    if import_foreign {
        all.push(("project:.claude", cwd.join(".claude").join("skills")));
    }
    all.push(("bundled", bundled_skills_dir()));
    // De-dupe by path (e.g. run from home: user and project resolve equal).
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (label, d) in all {
        let key = d.to_string_lossy().into_owned();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        out.push((label, d));
    }
    out
}

/// Destination root for `new`/`remove` given the scope/compat flags.
fn target_dir(project: bool, claude: bool) -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    let cwd = PathBuf::from(get_cwd());
    if project {
        if claude {
            cwd.join(".claude").join("skills")
        } else {
            cwd.join(".codemason").join("skills")
        }
    } else if claude {
        home.join(".claude").join("skills")
    } else {
        CONFIG_DIR.join("skills")
    }
}

fn skill_md(name: &str, project: bool, claude: bool) -> PathBuf {
    target_dir(project, claude).join(name).join("SKILL.md")
}

/// Return the SKILL.md path for ``name`` in precedence order across roots.
fn find_skill_md(name: &str) -> Option<PathBuf> {
    for (_label, base) in roots() {
        let p = base.join(name).join("SKILL.md");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn scope_of(path: &Path) -> &'static str {
    let s = path.to_string_lossy();
    for (label, base) in roots() {
        if s.starts_with(&*base.to_string_lossy()) {
            return label;
        }
    }
    "?"
}

// ── list ────────────────────────────────────────────────────────────────────
fn cmd_list(_args: &[String]) -> i32 {
    refresh_skills();
    let skills = discover_skills();
    if skills.is_empty() {
        println!("No skills found. Create one with: codemason skill new <name>");
        println!(
            "  (searched {})",
            roots()
                .iter()
                .map(|(_l, d)| d.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return 0;
    }
    let invocable: HashSet<String> =
        model_invocable_skills().into_iter().map(|s| s.name).collect();
    for s in &skills {
        let scope = scope_of(Path::new(&s.path));
        let mut tags: Vec<String> = Vec::new();
        if s.context == "fork" {
            tags.push("fork".into());
        }
        if !invocable.contains(&s.name) {
            tags.push("not-model-invocable".into());
        }
        if !s.argument_names.is_empty() {
            tags.push(format!("args:{}", s.argument_names.join(",")));
        }
        let suffix = if tags.is_empty() {
            String::new()
        } else {
            format!("  [{}]", tags.join(", "))
        };
        let desc = s.description.trim();
        println!("  {:<24} ({scope}) {desc}{suffix}", s.name);
    }
    0
}

// ── show ────────────────────────────────────────────────────────────────────
fn cmd_show(args: &[String]) -> i32 {
    let Some(name) = args.first() else {
        eprintln!("usage: codemason skill show <name>");
        return 2;
    };
    let Some(p) = find_skill_md(name) else {
        eprintln!("error: no such skill: {name}");
        return 1;
    };
    println!("# {name}  ({})", scope_of(&p));
    println!("# {}\n", p.display());
    let text = std::fs::read(&p)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    println!("{text}");
    0
}

// ── new ─────────────────────────────────────────────────────────────────────
fn template(name: &str, description: &str, extra: &str) -> String {
    format!(
        "---\nname: {name}\ndescription: {description}\n{extra}---\n<!-- The body below IS the skill: it expands into the prompt when the skill is\n     invoked. Reference arguments as ${{1}}, ${{2}} (or names you declared in\n     `arguments:`), and the skill's own directory as ${{CODEMASON_SKILL_DIR}}. -->\n\nWhen this skill runs, do the following:\n\n1. ...\n2. ...\n"
    )
    .replace("${{", "${")
    .replace("}}", "}")
}

static NAME_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z0-9._-]+$").unwrap());

fn cmd_new(args: &[String]) -> i32 {
    let Some(name) = args.first().cloned() else {
        eprintln!(
            "usage: codemason skill new <name> [--description D] [--arguments \"a b\"] \
             [--context inline|fork] [--agent NAME] [--project] [--claude]"
        );
        return 2;
    };
    let rest = &args[1..];
    let mut description = "TODO: one line on when to use this skill.".to_string();
    let mut arguments: Option<String> = None;
    let mut agent: Option<String> = None;
    let mut context = "inline".to_string();
    let mut project = false;
    let mut claude = false;
    let mut i = 0;
    while i < rest.len() {
        let tok = rest[i].as_str();
        macro_rules! next {
            () => {{
                i += 1;
                match rest.get(i) {
                    Some(v) => v.clone(),
                    None => {
                        eprintln!("unknown option: {tok}");
                        return 2;
                    }
                }
            }};
        }
        match tok {
            "--description" => description = next!(),
            "--arguments" => arguments = Some(next!()),
            "--context" => context = next!(),
            "--agent" => agent = Some(next!()),
            "--project" => project = true,
            "--claude" => claude = true,
            _ => {
                eprintln!("unknown option: {tok}");
                return 2;
            }
        }
        i += 1;
    }
    if context != "inline" && context != "fork" {
        eprintln!("error: --context must be 'inline' or 'fork'");
        return 2;
    }
    if !NAME_RE.is_match(&name) {
        eprintln!("error: skill name must be [A-Za-z0-9._-]");
        return 2;
    }

    let path = skill_md(&name, project, claude);
    if path.exists() {
        eprintln!("error: skill already exists: {}", path.display());
        return 1;
    }
    let mut extra = String::new();
    if let Some(a) = &arguments {
        if !a.is_empty() {
            extra.push_str(&format!("arguments: {a}\n"));
        }
    }
    if context == "fork" {
        extra.push_str("context: fork\n");
    }
    if let Some(a) = &agent {
        if !a.is_empty() {
            extra.push_str(&format!("agent: {a}\n"));
        }
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&path, template(&name, &description, &extra)) {
        eprintln!("error: could not write {}: {e}", path.display());
        return 1;
    }
    println!("Created skill '{name}' ({}) at:", scope_of(&path));
    println!("  {}", path.display());
    println!("Edit the body, then it is available to the model and via /{name}");
    0
}

// ── remove ──────────────────────────────────────────────────────────────────
fn cmd_remove(args: &[String]) -> i32 {
    let project = args.iter().any(|a| a == "--project");
    let claude = args.iter().any(|a| a == "--claude");
    let names: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let Some(name) = names.first() else {
        eprintln!("usage: codemason skill remove <name> [--project] [--claude]");
        return 2;
    };
    let mut path = skill_md(name, project, claude);
    if !path.exists() {
        // fall back to wherever it actually is
        match find_skill_md(name) {
            Some(found) => path = found,
            None => {
                eprintln!("error: no such skill: {name}");
                return 1;
            }
        }
    }
    let skill_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or(path);
    if let Err(e) = std::fs::remove_dir_all(&skill_dir) {
        eprintln!("error: could not remove {}: {e}", skill_dir.display());
        return 1;
    }
    println!("Removed skill '{name}' ({})", skill_dir.display());
    0
}

// ── enable / disable (toggle disable-model-invocation frontmatter) ───────────
static FM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(---\s*\n)((?s:.)*?\n)(---\s*\n)").unwrap());
static DISABLE_LINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*disable-model-invocation\s*:").unwrap());

/// Add/remove ``disable-model-invocation: true`` in the frontmatter block.
///
/// Returns (new_text, changed). Targeted line edit — the rest of the file
/// (body, other frontmatter keys, comments) is left byte-for-byte intact.
fn set_disable_flag(text: &str, disable: bool) -> (String, bool) {
    let Some(m) = FM_RE.captures(text) else {
        return (text.to_string(), false);
    };
    let whole_len = m.get(0).map(|g| g.end()).unwrap_or(0);
    let open_d = m.get(1).map(|g| g.as_str()).unwrap_or("");
    let body = m.get(2).map(|g| g.as_str()).unwrap_or("");
    let close_d = m.get(3).map(|g| g.as_str()).unwrap_or("");

    let lines: Vec<&str> = body.split('\n').collect();
    let mut kept: Vec<String> = lines
        .iter()
        .filter(|ln| !DISABLE_LINE_RE.is_match(ln))
        .map(|s| s.to_string())
        .collect();
    let had = kept.len() != lines.len();
    let changed;
    if disable {
        // ensure exactly one flag line
        if !kept.is_empty() && kept.last().map(|s| s.is_empty()).unwrap_or(false) {
            let idx = kept.len() - 1;
            kept.insert(idx, "disable-model-invocation: true".to_string());
        } else {
            kept.push("disable-model-invocation: true".to_string());
        }
        changed = true;
    } else {
        changed = had; // removing only changes if it was there
    }
    let new_body = kept.join("\n");
    let new_text = format!("{open_d}{new_body}{close_d}{}", &text[whole_len..]);
    (new_text, changed)
}

fn toggle(name: &str, disable: bool) -> i32 {
    let Some(p) = find_skill_md(name) else {
        eprintln!("error: no such skill: {name}");
        return 1;
    };
    let Ok(text) = std::fs::read_to_string(&p) else {
        eprintln!("error: no such skill: {name}");
        return 1;
    };
    let (new_text, changed) = set_disable_flag(&text, disable);
    if !changed {
        println!(
            "skill '{name}' already {} for model invocation.",
            if disable { "disabled" } else { "enabled" }
        );
        return 0;
    }
    if let Err(e) = std::fs::write(&p, new_text) {
        eprintln!("error: could not write {}: {e}", p.display());
        return 1;
    }
    println!(
        "{} model invocation for skill '{name}'.",
        if disable { "Disabled" } else { "Enabled" }
    );
    0
}

fn cmd_enable(args: &[String]) -> i32 {
    let Some(name) = args.first() else {
        eprintln!("usage: codemason skill enable <name>");
        return 2;
    };
    toggle(name, false)
}

fn cmd_disable(args: &[String]) -> i32 {
    let Some(name) = args.first() else {
        eprintln!("usage: codemason skill disable <name>");
        return 2;
    };
    toggle(name, true)
}

/// Dispatch ``codemason skill <sub> ...``. ``argv`` is the args after "skill".
pub fn handle_skill_cli(argv: &[String]) -> i32 {
    if argv.is_empty() || matches!(argv[0].as_str(), "-h" | "--help" | "help") {
        println!(
            "codemason skill — author and manage skills\n  roots (first-wins): ~/.codemason/skills, <cwd>/.codemason/skills\n  (~/.claude/skills and <cwd>/.claude/skills are read only with import_claude_dirs on)\n\n  list                     list discovered skills\n  show <name>              print a skill's SKILL.md\n  new <name> [opts]        scaffold a new skill (--project, --claude)\n  remove <name>            delete a skill (--project, --claude)\n  enable <name>            allow model invocation\n  disable <name>           block model invocation\n"
        );
        return 0;
    }
    let sub = argv[0].as_str();
    let rest = &argv[1..];
    match sub {
        "list" | "ls" => cmd_list(rest),
        "show" | "cat" => cmd_show(rest),
        "new" | "create" => cmd_new(rest),
        "remove" | "rm" => cmd_remove(rest),
        "enable" => cmd_enable(rest),
        "disable" => cmd_disable(rest),
        _ => {
            eprintln!("unknown skill subcommand: {sub}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disable_flag_roundtrip() {
        let text = "---\nname: x\ndescription: d\n---\nbody\n";
        let (disabled, changed) = set_disable_flag(text, true);
        assert!(changed);
        assert!(disabled.contains("disable-model-invocation: true"));
        assert!(disabled.ends_with("---\nbody\n"));
        let (enabled, changed2) = set_disable_flag(&disabled, false);
        assert!(changed2);
        assert_eq!(enabled, text);
    }

    #[test]
    fn template_has_placeholders() {
        let t = template("n", "d", "");
        assert!(t.contains("${1}"));
        assert!(t.contains("${CODEMASON_SKILL_DIR}"));
        assert!(t.starts_with("---\nname: n\ndescription: d\n---\n"));
    }
}
