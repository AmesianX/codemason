//! Tab / live completion for the interactive REPL prompt.
//! Equivalent to codemason/utils/completion.py.
//!
//! Substrate divergence (documented): the Python plugged a prompt_toolkit
//! Completer/AutoSuggest into the PromptSession; here the same candidate logic
//! feeds the PromptEngine's completion/auto-suggest hooks. Two kinds of
//! completion, unchanged:
//!
//!   * **Slash commands** — when the line's first token starts with `/`,
//!     complete the REPL built-ins and every discovered skill name, shown with
//!     its description. Live (as-you-type).
//!   * **File paths** — otherwise, on an explicit Tab, complete the filesystem
//!     path under the cursor. Gated to Tab (or clearly path-like tokens) so it
//!     never interrupts normal prose typing.
//!
//! Candidate functions never raise — a completion error must not break the
//! prompt.

/// Built-in slash commands and a one-line meta for the menu. Keep in sync with
/// cli::BUILTIN_COMMANDS / the /help text.
pub const BUILTIN_META: &[(&str, &str)] = &[
    ("help", "show help"),
    ("exit", "quit"),
    ("quit", "quit"),
    ("clear", "new session (clear history)"),
    ("compact", "summarize to free context"),
    ("resume", "pick a stored session to continue"),
    ("expand", "print last N tool outputs"),
    ("export", "save the conversation to Markdown"),
    ("language", "set UI language"),
    ("lang", "set UI language"),
    ("keepgoing", "toggle keep-going"),
    ("dsp", "toggle skipping tool approvals"),
    ("effort", "reasoning effort: low|medium|high|off"),
    ("reasoning", "reasoning effort: low|medium|high|off"),
    ("think", "reasoning effort: low|medium|high|off"),
    ("goal", "set/show/clear a persistent goal"),
    ("remote", "remote channel status/toggle"),
    ("remote-control", "remote channel status/toggle"),
    ("rc", "remote channel (alias)"),
];

/// Commands whose single argument has a fixed set of options — completed after
/// `/<cmd> `. Keep in sync with the handlers in cli::dispatch_builtin.
const EFFORT_ARGS: &[&str] = &["low", "medium", "high", "off", "status"];
const ONOFF_ARGS: &[&str] = &["on", "off", "status"];

pub fn command_args(cmd: &str) -> Option<&'static [&'static str]> {
    match cmd {
        "effort" | "reasoning" | "think" => Some(EFFORT_ARGS),
        "dsp" | "keepgoing" | "remote" | "remote-control" | "rc" => Some(ONOFF_ARGS),
        _ => None,
    }
}

/// (name, description) for every discovered skill; empty on any error.
fn skill_completions() -> Vec<(String, String)> {
    crate::utils::skills::discover_skills()
        .into_iter()
        .map(|s| (s.name, s.description.trim().to_string()))
        .collect()
}

/// A completion candidate in the engine's shape:
/// (chars_to_replace_before_cursor, insert, display, meta).
pub type Candidate = (usize, String, String, String);

/// Complete a filesystem path fragment (the token under the cursor).
fn path_completions(word: &str) -> Vec<Candidate> {
    let expanded = if let Some(rest) = word.strip_prefix("~/") {
        match dirs::home_dir() {
            Some(h) => h.join(rest).to_string_lossy().into_owned(),
            None => word.to_string(),
        }
    } else {
        word.to_string()
    };
    let (dirname, partial) = match expanded.rfind('/') {
        Some(i) => (&expanded[..i], &expanded[i + 1..]),
        None => ("", expanded.as_str()),
    };
    let search_dir = if dirname.is_empty() { "." } else { dirname };
    let Ok(entries) = std::fs::read_dir(search_dir) else {
        return Vec::new();
    };
    // Preserve the user's typed ~ / dir prefix when inserting.
    let prefix_len = word.chars().count() - partial.chars().count();
    let prefix: String = word.chars().take(prefix_len).collect();
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                format!("{name}/")
            } else {
                name
            }
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .filter(|display| {
            let name = display.trim_end_matches('/');
            if name.starts_with('.') && !partial.starts_with('.') {
                return false; // hide dotfiles unless the user started typing one
            }
            name.starts_with(partial)
        })
        .map(|display| {
            (
                word.chars().count(),
                format!("{prefix}{display}"),
                display.clone(),
                String::new(),
            )
        })
        .collect()
}

fn looks_like_path(word: &str) -> bool {
    !word.is_empty()
        && (word.contains('/')
            || word.starts_with('~')
            || word.starts_with('.'))
}

/// The engine completion hook: candidates for `text_before_cursor`.
/// Never raises (any I/O error yields no candidates).
pub fn get_completions(text_before_cursor: &str, explicit_tab: bool) -> Vec<Candidate> {
    let text = text_before_cursor;
    let stripped = text.trim_start();

    if stripped.starts_with('/') && !stripped.contains('\n') {
        // First token still being typed (no space yet): complete the
        // command / skill name.
        if !stripped.contains(' ') {
            let word = stripped; // includes the leading '/'
            let prefix = word[1..].to_lowercase();
            let mut out = Vec::new();
            for (name, meta) in BUILTIN_META {
                if name.starts_with(&prefix) {
                    out.push((
                        word.chars().count(),
                        format!("/{name}"),
                        format!("/{name}"),
                        meta.to_string(),
                    ));
                }
            }
            for (name, desc) in skill_completions() {
                if name.to_lowercase().starts_with(&prefix) {
                    out.push((
                        word.chars().count(),
                        format!("/{name}"),
                        format!("/{name}"),
                        desc,
                    ));
                }
            }
            return out;
        }
        // Past the command: complete its argument from the fixed option set,
        // but only for the FIRST argument (single-arg commands).
        let parts: Vec<&str> = stripped.split_whitespace().collect();
        let Some(opts) = command_args(&parts[0][1..].to_lowercase()) else {
            return Vec::new();
        };
        let ends_ws = stripped.ends_with(' ');
        let args = &parts[1..];
        // args fully entered (not counting the partial being typed)
        let completed = args.len() as i64 - if ends_ws { 0 } else { 1 };
        if completed == 0 {
            let arg = if ends_ws { String::new() } else { parts.last().unwrap().to_lowercase() };
            return opts
                .iter()
                .filter(|o| o.starts_with(&arg))
                .map(|o| (arg.chars().count(), o.to_string(), o.to_string(), String::new()))
                .collect();
        }
        return Vec::new();
    }

    // File path — the last whitespace-delimited token. Offer it on an explicit
    // Tab, or immediately when the token is clearly a path.
    let word = if !text.is_empty() && !text.ends_with(char::is_whitespace) {
        text.split_whitespace().last().unwrap_or("")
    } else {
        ""
    };
    if !word.is_empty() && (explicit_tab || looks_like_path(word)) {
        path_completions(word)
    } else {
        Vec::new()
    }
}

/// Inline "ghost text" suggestion for the next input (accept with →).
///
/// - While typing a `/` command, suggest the first matching built-in or skill
///   name (`/sy` → ghosts `stematic-debugging`).
/// - Otherwise, fall back to the most recent history line that starts with
///   what you've typed (fish-shell style).
pub fn get_suggestion(text: &str, history: &[String]) -> Option<String> {
    if text.is_empty() || text.contains('\n') {
        return None;
    }
    let stripped = text.trim_start();
    if stripped.starts_with('/') && !stripped.contains(' ') {
        let prefix = stripped[1..].to_lowercase();
        if prefix.is_empty() {
            return None;
        }
        let mut names: Vec<String> = BUILTIN_META.iter().map(|(n, _)| n.to_string()).collect();
        names.extend(skill_completions().into_iter().map(|(n, _)| n));
        for name in names {
            let lower = name.to_lowercase();
            if lower.starts_with(&prefix) && name.chars().count() > prefix.chars().count() {
                return Some(name.chars().skip(prefix.chars().count()).collect());
            }
        }
        return None;
    }

    // non-slash: history-based suggestion (most recent match wins)
    for line in history.iter().rev() {
        if line.starts_with(text) && line.len() > text.len() {
            return Some(line[text.len()..].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_builtin_completion() {
        let c = get_completions("/hel", false);
        assert!(c.iter().any(|(_, ins, _, _)| ins == "/help"));
    }

    #[test]
    fn arg_completion_first_only() {
        let c = get_completions("/dsp o", false);
        let opts: Vec<&str> = c.iter().map(|(_, i, _, _)| i.as_str()).collect();
        assert!(opts.contains(&"on") && opts.contains(&"off"));
        assert!(get_completions("/dsp on off", false).is_empty());
    }

    #[test]
    fn ghost_from_history() {
        let hist = vec!["run the tests".to_string()];
        assert_eq!(get_suggestion("run th", &hist), Some("e tests".to_string()));
    }
}
