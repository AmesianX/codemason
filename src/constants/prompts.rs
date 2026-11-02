//! Prompts for CodeMason — based on Claude Code system prompt.
//! Equivalent to codemason/constants/prompts.py.
//!
//! Source of truth:
//!   extracted/source/src/constants/prompts.ts
//!   extracted/source/src/constants/systemPromptSections.ts
//!   extracted/source/src/constants/cyberRiskInstruction.ts
//!
//! The original assembles the system prompt from modular sections. Many
//! sections are gated on `process.env.USER_TYPE === 'ant'` or `feature(...)`
//! flags; in an external build those resolve to false and get
//! dead-code-eliminated. This port reproduces the EXTERNAL build's prompt
//! verbatim, keeping the gates as `PromptConfig` flags so they can be flipped
//! without re-porting.
//!
//! Static section texts are embedded via include_str! from src/constants/text/
//! — those files were dumped from the running Python module, so they are
//! byte-identical to the reference. Dynamic sections (tool-set dependent,
//! environment info) are assembled in code and covered by fixture tests.
//!
//! NOTE: SDK/Anthropic-only machinery (prompt caching scopes, MCP, skills,
//! proactive/KAIROS, verification agents, growthbook features) is intentionally
//! omitted — none of it exists in the local environment. memdir is the
//! exception: memories are plain files, so it ports (utils/memory.rs via
//! load_memory_prompt below).

use std::collections::HashSet;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::tools::base::Tool;
use crate::utils::config::get_config;
use crate::utils::state::get_cwd;

// === Product / macros ===

pub const PRODUCT_NAME: &str = "CodeMason";
/// MACRO.ISSUES_EXPLAINER is a build-time injected macro in the original.
pub const MACRO_ISSUES_EXPLAINER: &str =
    "report the issue at https://github.com/AmesianX/codemason/issues";

/// Owned by the Safeguards team in the original — ported verbatim, do not edit.
pub const CYBER_RISK_INSTRUCTION: &str = include_str!("text/cyber_risk_instruction.txt");

// === Tool name constants (extracted/.../tools/*/*.ts) ===

pub const BASH_TOOL_NAME: &str = "Bash";
pub const FILE_READ_TOOL_NAME: &str = "Read";
pub const FILE_WRITE_TOOL_NAME: &str = "Write";
pub const FILE_EDIT_TOOL_NAME: &str = "Edit";
pub const GLOB_TOOL_NAME: &str = "Glob";
pub const GREP_TOOL_NAME: &str = "Grep";
pub const AGENT_TOOL_NAME: &str = "Agent";
pub const LEGACY_AGENT_TOOL_NAME: &str = "Task";
pub const ASK_USER_QUESTION_TOOL_NAME: &str = "AskUserQuestion";
pub const TODO_WRITE_TOOL_NAME: &str = "TodoWrite";
pub const TASK_CREATE_TOOL_NAME: &str = "TaskCreate";
pub const SKILL_TOOL_NAME: &str = "Skill";

// === Build/feature gates (default = external local build) ===

/// Resolved feature/USER_TYPE gates. Defaults match an external build.
#[derive(Debug, Clone, Default)]
pub struct PromptConfig {
    pub user_type_ant: bool,
    pub repl_mode: bool,
    pub embedded_search_tools: bool,
    pub fork_subagent: bool,
    pub explore_plan_agents: bool,
    pub non_interactive: bool,
    /// omit the dynamic-boundary marker
    pub global_cache_scope: bool,
}

pub fn get_prompt_config() -> PromptConfig {
    PromptConfig::default()
}

// === Bullet helper (prependBullets) ===

/// A bullet item: a top-level string (' - x') or a nested list ('  - x').
pub enum BulletItem {
    Item(String),
    Sub(Vec<String>),
}

/// Port of prependBullets: strings -> ' - x', nested lists -> '  - x'.
pub fn prepend_bullets(items: &[BulletItem]) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        match item {
            BulletItem::Item(s) => out.push(format!(" - {s}")),
            BulletItem::Sub(subs) => {
                for sub in subs {
                    out.push(format!("  - {sub}"));
                }
            }
        }
    }
    out
}

// === Static sections ===

pub fn get_hooks_section() -> &'static str {
    include_str!("text/hooks_section.txt")
}

/// output_style is not ported (always absent locally); the parameter is kept so
/// the alternative wording stays reachable, matching the Python signature.
pub fn get_simple_intro_section(output_style_present: bool) -> String {
    let helps = if output_style_present {
        "according to your \"Output Style\" below, which describes how you should respond to user queries."
    } else {
        "with software engineering tasks."
    };
    format!(
        "\nYou are CodeMason, an AI coding assistant. You are an interactive agent that helps users {helps} Use the instructions below and the tools available to you to assist the user.\n\n{CYBER_RISK_INSTRUCTION}\nIMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files."
    )
}

pub fn get_simple_system_section() -> &'static str {
    include_str!("text/simple_system_section.txt")
}

pub fn get_simple_doing_tasks_section(_config: &PromptConfig) -> &'static str {
    // Ant-only comment-discipline subitems are DCE'd in external builds; the
    // dumped text is the external assembly.
    include_str!("text/simple_doing_tasks_section.txt")
}

pub fn get_actions_section() -> &'static str {
    include_str!("text/actions_section.txt")
}

/// [ours — not in upstream external build] Execute-don't-describe discipline
/// for the DeepSeek substrate; see the Python reference for the full design
/// note. `autonomous` (wired to --dangerously-skip-permissions) appends the
/// autonomous-mode block (routine-action confirmation dropped, AskUserQuestion
/// reserved for real blockers, impasse-stop legalized).
/// [ours — not in upstream external build] Covers the ONE request per turn
/// that runs on abliterated weights. services::ab runs `auto`: the turn's
/// first request goes out at scale=1 and the edit is pushed off the moment it
/// answers, so a tool-using turn is already protected — but a turn that calls
/// no tool spends its whole answer in that window with nothing holding it
/// back (44% fabrication on the held-out invented subjects; 0 with this
/// section, and the unedited requests after it lose nothing: 0/39 either
/// way). The edit does not remove knowledge — a forced-choice
/// existence probe over 20 invented subjects scores 88% at scale=1 and 100% on
/// the real controls — it removes the disposition to act on it: asked the same
/// subjects as open questions the model describes them confidently ~75% of the
/// time. Held-out set of 20 non-software domains, 19 tools wired, every row
/// hand-read: no directive 44% fabrication, first sentence alone 21%, both
/// sentences 0%, unedited weights 0%. The second sentence is load-bearing —
/// every fabrication in the study came from a turn with no tool call (0 in 54
/// tool-calling turns). Phrased without naming any category to verify: an
/// earlier draft enumerated "options, APIs, identifiers" and fixed exactly
/// those while collapsing verification elsewhere (3/3 -> 0/6 tool use, 0 -> 6/6
/// fabrication), because an enumeration reads as an exemption list. Measured
/// costs: over-denial of real things 0/200 rows; extra searches on real
/// subjects 50-70% vs 20% unedited (~1 added search per turn). Not a
/// guarantee: inside the real CLI an invented Korean verb still got a
/// fabricated entry — no felt uncertainty, so no lookup — and no rewording
/// removed that shape. Korean, in an English
/// prompt, on purpose: the same sentences in English searched 23/34 and
/// fabricated 4 where Korean searched 34/34 and fabricated 0 against Korean
/// questions. Do not "tidy" it into English without rerunning that. See the
/// Python reference for the full design note.
pub fn get_fabrication_guard_section() -> &'static str {
    include_str!("text/fabrication_guard.txt")
}

/// [ours] Paragraphs 3 and 4 of the base text are 2.1.88's USER_TYPE='ant'
/// counterweights, reworded: the thoroughness one (prompts.ts:210, marked
/// "un-gate once validated on external via A/B") and the assertiveness one
/// (prompts.ts:224). They are gone from the prompt by 2.1.208, which dates
/// them to a model that stopped needing them; this substrate has not. The
/// false-claims bullet (prompts.ts:237) is deliberately NOT copied whole -
/// its first half already exists in paragraph 2. Keep in step with the
/// Python reference, which carries the full design note.
pub fn get_execution_discipline_section(autonomous: bool) -> String {
    let base = include_str!("text/execution_discipline_base.txt");
    if autonomous {
        format!(
            "{base}{}",
            include_str!("text/execution_discipline_autonomous_suffix.txt")
        )
    } else {
        base.to_string()
    }
}

pub fn get_using_your_tools_section(
    enabled_tools: &HashSet<String>,
    config: &PromptConfig,
) -> String {
    let task_tool_name = [TASK_CREATE_TOOL_NAME, TODO_WRITE_TOOL_NAME]
        .into_iter()
        .find(|n| enabled_tools.contains(*n));

    let task_item = |task_tool_name: &str| {
        format!(
            "Break down and manage your work with the {task_tool_name} tool. These tools are helpful for planning your work and helping the user track your progress. Mark each task as completed as soon as you are done with the task. Do not batch up multiple tasks before marking them as completed."
        )
    };

    if config.repl_mode {
        let mut items = Vec::new();
        if let Some(n) = task_tool_name {
            items.push(BulletItem::Item(task_item(n)));
        }
        if items.is_empty() {
            return String::new();
        }
        let mut lines = vec!["# Using your tools".to_string()];
        lines.extend(prepend_bullets(&items));
        return lines.join("\n");
    }

    let embedded = config.embedded_search_tools;

    let mut provided_tool_subitems = vec![
        format!("To read files use {FILE_READ_TOOL_NAME} instead of cat, head, tail, or sed"),
        format!("To edit files use {FILE_EDIT_TOOL_NAME} instead of sed or awk"),
        format!(
            "To create files use {FILE_WRITE_TOOL_NAME} instead of cat with heredoc or echo redirection"
        ),
    ];
    if !embedded {
        provided_tool_subitems.push(format!(
            "To search for files use {GLOB_TOOL_NAME} instead of find or ls"
        ));
        provided_tool_subitems.push(format!(
            "To search the content of files, use {GREP_TOOL_NAME} instead of grep or rg"
        ));
    }
    provided_tool_subitems.push(format!(
        "Reserve using the {BASH_TOOL_NAME} exclusively for system commands and terminal operations that require shell execution. If you are unsure and there is a relevant dedicated tool, default to using the dedicated tool and only fallback on using the {BASH_TOOL_NAME} tool for these if it is absolutely necessary."
    ));

    let mut items: Vec<BulletItem> = vec![
        BulletItem::Item(format!(
            "Do NOT use the {BASH_TOOL_NAME} to run commands when a relevant dedicated tool is provided. Using dedicated tools allows the user to better understand and review your work. This is CRITICAL to assisting the user:"
        )),
        BulletItem::Sub(provided_tool_subitems),
    ];
    if let Some(n) = task_tool_name {
        items.push(BulletItem::Item(task_item(n)));
    }
    items.push(BulletItem::Item(
        "You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel. Maximize use of parallel tool calls where possible to increase efficiency. However, if some tool calls depend on previous calls to inform dependent values, do NOT call these tools in parallel and instead call them sequentially. For instance, if one operation must complete before another starts, run these operations sequentially instead."
            .to_string(),
    ));
    let mut lines = vec!["# Using your tools".to_string()];
    lines.extend(prepend_bullets(&items));
    lines.join("\n")
}

pub fn get_agent_tool_section(config: &PromptConfig) -> &'static str {
    if config.fork_subagent {
        include_str!("text/agent_tool_section_fork.txt")
    } else {
        include_str!("text/agent_tool_section_default.txt")
    }
}

pub fn get_simple_tone_and_style_section(_config: &PromptConfig) -> &'static str {
    include_str!("text/tone_and_style_section.txt")
}

pub fn get_output_efficiency_section(_config: &PromptConfig) -> &'static str {
    include_str!("text/output_efficiency_section.txt")
}

// === Dynamic sections ===

pub fn get_session_specific_guidance_section(
    enabled_tools: &HashSet<String>,
    config: &PromptConfig,
) -> Option<String> {
    let has_ask_user_question = enabled_tools.contains(ASK_USER_QUESTION_TOOL_NAME);
    let has_agent_tool = enabled_tools.contains(AGENT_TOOL_NAME);
    let has_skill = enabled_tools.contains(SKILL_TOOL_NAME);

    let mut items: Vec<BulletItem> = Vec::new();
    if has_skill {
        items.push(BulletItem::Item(format!(
            "When the user types `/<skill-name>` (e.g., /commit), that is shorthand to invoke a user-invocable skill, which expands into a full prompt. Use the {SKILL_TOOL_NAME} tool to execute it. IMPORTANT: Only use {SKILL_TOOL_NAME} for skills listed in its user-invocable skills section — do not guess or use built-in CLI commands."
        )));
    }
    if has_ask_user_question {
        items.push(BulletItem::Item(format!(
            "If you do not understand why the user has denied a tool call, use the {ASK_USER_QUESTION_TOOL_NAME} to ask them."
        )));
    }
    if !config.non_interactive {
        items.push(BulletItem::Item(
            "If you need the user to run a shell command themselves (e.g., an interactive login like `gcloud auth login`), suggest they type `! <command>` in the prompt — the `!` prefix runs the command in this session so its output lands directly in the conversation."
                .to_string(),
        ));
    }
    if has_agent_tool {
        items.push(BulletItem::Item(get_agent_tool_section(config).to_string()));
    }
    if items.is_empty() {
        return None;
    }
    let mut lines = vec!["# Session-specific guidance".to_string()];
    lines.extend(prepend_bullets(&items));
    Some(lines.join("\n"))
}

/// [ours — not in upstream external build] How to be called back by work this
/// session did not start.
///
/// Gated on Bash, because every sentence of it is a shell command. Its own
/// section rather than a bullet spliced into the ported guidance: that section
/// reproduces the external build verbatim and its fixture is what attests to
/// that, which a local addition inside it would quietly stop being true.
///
/// Deliberately says nothing about what the model can or cannot do between
/// turns. The harness already wakes it for background commands and sub-agents,
/// and an earlier draft that claimed otherwise contradicted the Bash tool's own
/// "you will be notified — do not poll", which would have sent the model back
/// to the sleep-polling those tools exist to replace.
pub fn get_background_callback_section(enabled_tools: &HashSet<String>) -> Option<String> {
    if !enabled_tools.contains(BASH_TOOL_NAME) {
        return None;
    }
    Some(
        "# Being called back\n\nWork this session did not start — a job on another machine, a scheduled task, a queue — cannot notify you the way a background command here does. If you tell the user you will report on something like that, wire the trigger first: `\"$CODEMASON_BIN\" inject --session \"$CODEMASON_SESSION_ID\" TEXT` queues TEXT for this session, which picks it up within a second; it prints the session it reached, so the wiring is checkable without waiting. Both variables are exported to everything you spawn. Keep TEXT your own short description, one message per event: it arrives as the user's own words, so piping a log or another program's output in hands that authority to whatever produced it, and the queue keeps only the most recent messages. `\"$CODEMASON_BIN\" notify TITLE` sends outward instead, to the user's own channels, and needs no session — it leaves the machine, so ask before using it for anything but an alert they asked for."
            .to_string(),
    )
}

/// [ours] What seeing images means here: how many fit in one request and what
/// happens past that (services/openai_common.rs, images_to_evict). `n` is the
/// server limit (services/vision.rs enabled_limit).
///
/// Its own section, not a line in the Read tool description: that description
/// is the ported original and says only that images can be read. None when the
/// server takes no images (llama.cpp, a text-only vLLM), so a text-only setup
/// gets the same prompt as before.
///
/// Measured without it on the live server: asked to read ten images and name
/// each colour, the model read all ten in one go, found the oldest replaced by
/// text, re-read them, pushed others out, and either chased its tail (30 reads
/// for 10 files) or named colours of images it was never shown. The limit and
/// the "write it down while attached" rule are what it was missing.
///
/// "Into your reply text": the template drops reasoning before the last user
/// message (clear_thinking), so a note kept only in thinking is gone next turn.
/// (The 20-reads-for-10-files run that prompted this wording also hit two bugs
/// since fixed — reasoning sent under a key vLLM ignores, and tool images in a
/// user message — which blanked thinking inside the tool loop.)
pub fn get_vision_section(n: i64) -> Option<String> {
    if n <= 0 {
        return None;
    }
    Some(format!(
        "# Images\n\nYou can see images. They reach you from the user's messages and from the Read tool on an image file, each labelled [Image N] in order of arrival. A very tall or wide image is read as several tiles; each tile is one image.\n - At most {n} images are attached to one request. Past that the oldest are replaced by a line \"[Image K: …; saved at PATH]\" that says whether you saw it earlier. Such an image is not in front of you: if you saw it, work from what you noted about it then; if you need to look again, or never saw it, Read the saved PATH — that attaches it as a new image and pushes an older one out.\n - So read at most {n} images at a time, and while they are attached write what you need from each into your reply text — not only into your thinking, which is dropped when the turn ends — instead of reading everything first and describing it afterwards.\n - If a task needs more than {n} images in view at once, say so rather than working from images that are no longer attached."
    ))
}

pub fn get_language_section(language: Option<&str>) -> Option<String> {
    let language = language?;
    if language.is_empty() {
        return None;
    }
    // The wording scopes the rule to "communications with the user", so the
    // model treats tool inputs as out of scope and writes Agent prompts in
    // English even with the Agent tool's own "same language as the user" rule.
    // The last sentence closes that gap. Gated on a configured language: unset
    // means no section at all, so upstream behaviour is untouched.
    Some(format!(
        "# Language\nAlways respond in {language}. Use {language} for all explanations, comments, and communications with the user. Technical terms and code identifiers should remain in their original form. This also covers the tool inputs you author for other agents to read — write Agent prompts and descriptions in {language} too, never defaulting to English."
    ))
}

pub const SUMMARIZE_TOOL_RESULTS_SECTION: &str =
    include_str!("text/summarize_tool_results_section.txt");

// === Environment info ===

fn get_is_git() -> bool {
    // `git rev-parse --is-inside-work-tree` with the Python port's 2s timeout
    // (poll try_wait; std::process has no native timeout), falling back to a
    // .git directory check on any failure.
    let cwd = get_cwd();
    let spawned = std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(&cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn();
    if let Ok(mut child) = spawned {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let mut out = String::new();
                    if let Some(mut stdout) = child.stdout.take() {
                        use std::io::Read;
                        let _ = stdout.read_to_string(&mut out);
                    }
                    return status.success() && out.trim() == "true";
                }
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(_) => break,
            }
        }
    }
    std::path::Path::new(&cwd).join(".git").is_dir()
}

/// Python sys.platform values, which the prompt prints verbatim.
pub fn sys_platform() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(windows) {
        "win32"
    } else {
        std::env::consts::OS
    }
}

/// Port of getUnameSR — `uname -sr` equivalent (platform.system() +
/// platform.release()).
pub fn get_uname_sr() -> String {
    #[cfg(unix)]
    {
        let mut buf: libc::utsname = unsafe { std::mem::zeroed() };
        if unsafe { libc::uname(&mut buf) } == 0 {
            let cstr = |a: &[libc::c_char]| {
                unsafe { std::ffi::CStr::from_ptr(a.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            };
            return format!("{} {}", cstr(&buf.sysname), cstr(&buf.release));
        }
    }
    #[cfg(windows)]
    {
        // platform.system()/release() shape; the release number is not exposed
        // by std — substrate divergence documented: "Windows" with the raw
        // version from `cmd /c ver` when available.
        if let Ok(out) = std::process::Command::new("cmd").args(["/c", "ver"]).output() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(v) = s.split_whitespace().last() {
                return format!("Windows {}", v.trim_end_matches(']'));
            }
        }
        return "Windows".to_string();
    }
    #[allow(unreachable_code)]
    {
        format!("{} unknown", std::env::consts::OS)
    }
}

pub fn get_shell_info_line() -> String {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
    let shell_name = if shell.contains("zsh") {
        "zsh".to_string()
    } else if shell.contains("bash") {
        "bash".to_string()
    } else {
        shell
    };
    if cfg!(windows) {
        format!(
            "Shell: {shell_name} (use Unix shell syntax, not Windows — e.g., /dev/null not NUL, forward slashes in paths)"
        )
    } else {
        format!("Shell: {shell_name}")
    }
}

/// No marketing-name registry locally; fall to the generic form (faithful to
/// the external path for an unknown model id). The original ends this line
/// with a period (prompts.ts:666 `...the model ${modelId}.`).
fn model_description(model_id: &str) -> Option<String> {
    Some(format!("You are powered by the model {model_id}."))
}

/// The served model's training cutoff, for the environment block, or None to
/// leave the line out.
///
/// This was a hardcoded table of another product's model ids. It could never
/// match: we serve whatever `model_name` points at (glm-*, DSV4-*, ...), so
/// every branch was dead and the line was never emitted. Carrying another
/// vendor's release calendar in our prompt builder taught the next reader that
/// this program runs those models.
///
/// It is a setting now, empty by default — so the line stays absent exactly as
/// it was, and nothing about the prompt changes until someone fills in the
/// cutoff for the model they actually serve.
/// Env: CODEMASON_MODEL_KNOWLEDGE_CUTOFF.
pub fn get_knowledge_cutoff(_model_id: &str) -> Option<String> {
    let cutoff = crate::utils::config::get_config().model_knowledge_cutoff;
    let cutoff = cutoff.trim();
    (!cutoff.is_empty()).then(|| cutoff.to_string())
}

pub fn compute_simple_env_info(
    model_id: &str,
    additional_working_directories: Option<&[String]>,
) -> String {
    let is_git = get_is_git();
    let uname_sr = get_uname_sr();
    let cwd = get_cwd();
    let cutoff_msg =
        get_knowledge_cutoff(model_id).map(|c| format!("Assistant knowledge cutoff is {c}."));
    let has_add = additional_working_directories.map(|a| !a.is_empty()).unwrap_or(false);

    let mut env_items: Vec<BulletItem> = vec![
        BulletItem::Item(format!("Primary working directory: {cwd}")),
        BulletItem::Sub(vec![format!("Is a git repository: {}", is_git)]),
    ];
    if has_add {
        env_items.push(BulletItem::Item("Additional working directories:".to_string()));
        env_items.push(BulletItem::Sub(
            additional_working_directories.unwrap_or(&[]).to_vec(),
        ));
    }
    env_items.push(BulletItem::Item(format!("Platform: {}", sys_platform())));
    env_items.push(BulletItem::Item(get_shell_info_line()));
    env_items.push(BulletItem::Item(format!("OS Version: {uname_sr}")));
    if let Some(desc) = model_description(model_id) {
        env_items.push(BulletItem::Item(desc));
    }
    if let Some(m) = cutoff_msg {
        env_items.push(BulletItem::Item(m));
    }
    env_items.push(BulletItem::Item(format!(
        "{PRODUCT_NAME} is a command-line AI coding assistant."
    )));

    let mut lines = vec![
        "# Environment".to_string(),
        "You have been invoked in the following environment: ".to_string(),
    ];
    lines.extend(prepend_bullets(&env_items));
    lines.join("\n")
}

/// The <env> block form used by agent prompts (computeEnvInfo).
pub fn compute_env_info(
    model_id: &str,
    additional_working_directories: Option<&[String]>,
) -> String {
    let is_git = get_is_git();
    let uname_sr = get_uname_sr();
    let add_info = match additional_working_directories {
        Some(dirs) if !dirs.is_empty() => {
            format!("Additional working directories: {}\n", dirs.join(", "))
        }
        _ => String::new(),
    };
    let cutoff_msg = get_knowledge_cutoff(model_id)
        .map(|c| format!("\n\nAssistant knowledge cutoff is {c}."))
        .unwrap_or_default();
    let model_desc = model_description(model_id).unwrap_or_default();
    format!(
        "Here is useful information about the environment you are running in:\n<env>\nWorking directory: {}\nIs directory a git repo: {}\n{}Platform: {}\n{}\nOS Version: {}\n{}</env>\n{}{}",
        get_cwd(),
        if is_git { "Yes" } else { "No" },
        add_info,
        sys_platform(),
        get_shell_info_line(),
        uname_sr,
        wsl_windows_note(),
        model_desc,
        cutoff_msg
    )
}

/// What running under WSL means for reaching Windows, or "" anywhere else.
///
/// A Windows program that needs administrator rights cannot be elevated from
/// here without a UAC prompt, and UAC is drawn on the secure desktop precisely
/// so that no process can dismiss it. Nothing in this program can change that.
/// What it CAN do is say so, because the model does not know: asked to capture
/// packets from WSL it reached for `Start-Process -Verb RunAs` on every command,
/// and a user spent a session clicking a dialog per invocation before saying
/// "매우매우 귀찮아" (2026-09-07).
///
/// The way through is a scheduled task registered once at RunLevel Highest —
/// running an already-elevated task is not a request to elevate, so no prompt
/// appears — which needs the user to register it. So the note names both: do
/// not reach for RunAs, and ask for what would actually work.
fn wsl_windows_note() -> String {
    wsl_windows_note_for(crate::utils::platform::get_platform())
}

/// Split out so the wording can be tested on a host that is not WSL.
fn wsl_windows_note_for(platform: &str) -> String {
    if platform != "wsl" {
        return String::new();
    }
    concat!(
        "Running under WSL: Windows programs are reachable (cmd.exe, ",
        "powershell.exe, schtasks.exe), and Windows drives are under /mnt/. But ",
        "a Windows program needing administrator rights CANNOT be elevated from ",
        "here silently. `Start-Process -Verb RunAs` opens a UAC dialog on the ",
        "Windows secure desktop that only a human can accept, once per call — do ",
        "not reach for it in a loop. Ask the user to register the privileged ",
        "command once as a scheduled task at RunLevel Highest, then run it with ",
        "`schtasks.exe /run /tn <name>`, which needs no prompt because it is not ",
        "requesting elevation.\n"
    )
        .to_string()
}

// === Memory (memdir) ===

/// Memory directive + the MEMORY.md index, rooted at the project dir like the
/// transcripts (utils/memory.rs). None only when the substrate is broken — an
/// empty memory still gets the directive, or nothing ever teaches the model to
/// start one.
fn strip_html_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + 3..],
            None => return out, // unterminated: drop the tail
        }
    }
    out.push_str(rest);
    out
}

/// Operator directive — free text from outside the binary, pinned to the END of
/// the system prompt.
///
/// This exists because the posture an agent needs is a property of the JOB, not
/// of the program. An authorised pentest engagement wants "the tools are the
/// work, run them"; a shared box wants the opposite. Baking either one in makes
/// the binary wrong for the other, so the operator supplies it per run.
///
/// LAST on purpose. Sections read in order and the later ones sit closer to the
/// conversation, so an operator line that contradicts a built-in section has to
/// come after it to win. It is also why this is not merged into any existing
/// section.
///
/// Read once, at startup, and never rebuilt: the system prompt is the cached
/// prefix (see services::openai_common::build_system_prompt_with_cwd_emphasis —
/// 72.1s against 3.4s at 96k when the front of it moves). Changing it mid-session
/// would re-prefill the whole conversation, so the CLI sets the env before the
/// first prompt is assembled and nothing writes it afterwards.
/// Shipped default, seeded into the config dir on first run so the user edits a
/// FILE rather than a rebuilt binary. Compiled in because a release binary has no
/// repo to read it from.
const DIRECTIVE_TEMPLATE: &str = include_str!("../../directive.md");

/// `~/.codemason/directive.md` (CODEMASON_CONFIG_DIR relocates it).
pub fn directive_path() -> std::path::PathBuf {
    crate::utils::config::CONFIG_DIR.join("directive.md")
}

pub fn load_operator_directive() -> Option<String> {
    // Env (set by --directive/--directive-file, or inherited) wins over the file.
    let text = match std::env::var("CODEMASON_DIRECTIVE") {
        Ok(v) if !v.trim().is_empty() => v,
        // Explicitly emptied -> the operator means "no directive", not "fall back".
        Ok(_) => return None,
        Err(_) => {
            let path = directive_path();
            if !path.exists() {
                // Seed once. A failure here is not worth a diagnostic: the run
                // simply carries on with no directive.
                let _ = std::fs::create_dir_all(&*crate::utils::config::CONFIG_DIR);
                // Atomic: two sessions starting together both seed it, and the
                // loser of that race must not be able to read a half-written
                // directive and paste it into a system prompt. Identical content
                // from every writer, so the rename alone is enough.
                let _ = crate::utils::file::atomic_write_bytes(
                    &path.to_string_lossy(),
                    DIRECTIVE_TEMPLATE.as_bytes(),
                );
            }
            std::fs::read_to_string(&path).ok()?
        }
    };
    // HTML comments carry the file's own instructions to the user, not the model's.
    let text = strip_html_comments(&text);
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    Some(format!(
        "# Operator directive\n\n         The operator of this session set the following. It describes the work \
         you are doing and the posture it calls for. Where it is more specific \
         than a rule above, follow it.\n\n{text}"
    ))
}

pub async fn load_memory_prompt() -> Option<String> {
    crate::utils::memory::build_memory_section()
}

// === Main assembly ===

/// Port of getSystemPrompt — returns the system prompt as a list of sections.
pub async fn get_system_prompt(
    tools: &[Arc<dyn Tool>],
    model: Option<&str>,
    additional_working_directories: Option<&[String]>,
) -> Vec<String> {
    let config = get_prompt_config();
    let model = model
        .map(|m| m.to_string())
        .unwrap_or_else(|| get_config().model_name);
    let enabled_tools: HashSet<String> = tools.iter().map(|t| t.name().to_string()).collect();
    let output_style_present = false; // output styles not ported

    let mut sections: Vec<Option<String>> = vec![
        Some(get_simple_intro_section(output_style_present)),
        Some(get_simple_system_section().to_string()),
        if output_style_present {
            None
        } else {
            Some(get_simple_doing_tasks_section(&config).to_string())
        },
        Some(get_actions_section().to_string()),
        // [ours] Execute-don't-describe discipline for the DeepSeek substrate.
        // Always on; --dangerously-skip-permissions additionally drops routine
        // confirmation.
        Some(get_execution_discipline_section(
            get_config().dangerously_skip_permissions,
        )),
        Some(get_using_your_tools_section(&enabled_tools, &config)),
        Some(get_simple_tone_and_style_section(&config).to_string()),
        Some(get_output_efficiency_section(&config).to_string()),
    ];

    sections.push(get_session_specific_guidance_section(&enabled_tools, &config));
    sections.push(get_background_callback_section(&enabled_tools));
    sections.push(get_vision_section(crate::services::vision::enabled_limit().await));
    sections.push(load_memory_prompt().await);
    sections.push(Some(compute_simple_env_info(
        &model,
        additional_working_directories,
    )));
    // Language section is driven by the user's configured preferred language
    // (getLanguageSection(settings.language)). None when unset.
    sections.push(get_language_section(get_config().language.as_deref()));
    sections.push(Some(SUMMARIZE_TOOL_RESULTS_SECTION.to_string()));
    // [ours] Last of the built-ins: the abliterated checkpoint keeps its
    // knowledge but loses the disposition to check before asserting. Measured
    // in this position (appended after the full prompt).
    sections.push(Some(get_fabrication_guard_section().to_string()));
    // Last: an operator directive has to outrank the built-in sections above.
    sections.push(load_operator_directive());

    sections.into_iter().flatten().collect()
}

// === Agent prompt ===

pub const DEFAULT_AGENT_PROMPT: &str = include_str!("text/default_agent_prompt.txt");

/// Python name: _AGENT_NOTES (imported by the Agent tool).
pub const AGENT_NOTES: &str = include_str!("text/agent_notes.txt");

/// Port of DEFAULT_AGENT_PROMPT + enhanceSystemPromptWithEnvDetails.
pub async fn get_agent_prompt(
    model: Option<&str>,
    additional_working_directories: Option<&[String]>,
) -> Vec<String> {
    let model = model
        .map(|m| m.to_string())
        .unwrap_or_else(|| get_config().model_name);
    let env_info = compute_env_info(&model, additional_working_directories);
    vec![
        DEFAULT_AGENT_PROMPT.to_string(),
        AGENT_NOTES.to_string(),
        env_info,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cutoff line must stay ABSENT by default. It always was — the old
    /// table only listed another product's model ids, so nothing we serve ever
    /// matched — and replacing that table must not start emitting a line the
    /// prompt never had.
    #[test]
    fn no_knowledge_cutoff_line_unless_it_is_configured() {
        let previous = crate::utils::config::get_config().model_knowledge_cutoff;
        crate::utils::config::update_config(|c| c.model_knowledge_cutoff = String::new());
        assert_eq!(get_knowledge_cutoff("glm-5.3-flash"), None);
        // Any id at all, including one that used to be special-cased.
        assert_eq!(get_knowledge_cutoff("some-other-model-4-6"), None, "no id is special now");

        crate::utils::config::update_config(|c| c.model_knowledge_cutoff = "March 2025".into());
        assert_eq!(get_knowledge_cutoff("glm-5.3-flash").as_deref(), Some("March 2025"));

        // Whitespace-only must read as unset, not as a blank cutoff line.
        crate::utils::config::update_config(|c| c.model_knowledge_cutoff = "   ".into());
        assert_eq!(get_knowledge_cutoff("glm-5.3-flash"), None);

        crate::utils::config::update_config(|c| c.model_knowledge_cutoff = previous);
    }

    fn tools18() -> HashSet<String> {
        [
            "Agent", "Bash", "Glob", "Grep", "Read", "Edit", "Write", "TodoWrite",
            "AskUserQuestion", "NotebookEdit", "WebFetch", "WebSearch", "SessionSearch",
            "EnterPlanMode", "ExitPlanMode", "Skill", "TaskOutput", "TaskStop",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    /// The embedded prompt corpus must be LF-only on every platform.
    ///
    /// `include_str!` bakes the working tree's bytes into the binary, so a
    /// CRLF checkout ships every system-prompt and tool-description line to the
    /// model ending in \r\n — a byte-for-byte divergence from the Python
    /// original. A clean POSIX tree cannot produce it and the fixture tests
    /// below only catch it once it has already been built, so guard the corpus
    /// directly. `.gitattributes` prevents the contaminated checkout; this
    /// catches a tree that predates it, which renormalization does NOT fix on
    /// its own (see the procedure in that file).
    #[test]
    fn embedded_prompt_text_is_lf_only() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/constants/text");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("read src/constants/text") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                continue;
            }
            let bytes = std::fs::read(&path).expect("read embedded text file");
            assert!(
                !bytes.contains(&b'\r'),
                "{} contains CR: this checkout is CRLF-contaminated and include_str! \
                 will bake it into the binary",
                path.display()
            );
            checked += 1;
        }
        assert!(checked >= 20, "expected the text/ corpus, found only {checked} files");

        // The files on disk are clean; confirm what actually compiled in is too,
        // covering both modules that embed from this directory.
        for (name, text) in [
            ("CYBER_RISK_INSTRUCTION", CYBER_RISK_INSTRUCTION),
            ("BASH_DESCRIPTION", crate::constants::tool_prompts::BASH_DESCRIPTION),
            ("READ_DESCRIPTION", crate::constants::tool_prompts::READ_DESCRIPTION),
            ("EDIT_DESCRIPTION", crate::constants::tool_prompts::EDIT_DESCRIPTION),
        ] {
            assert!(!text.contains('\r'), "{name} was embedded with a CR");
        }
    }

    #[test]
    fn using_your_tools_matches_python_fixture() {
        let got = get_using_your_tools_section(&tools18(), &PromptConfig::default());
        assert_eq!(got, include_str!("text/fixture_using_your_tools_18.txt"));
    }

    #[test]
    fn session_specific_matches_python_fixture() {
        let got =
            get_session_specific_guidance_section(&tools18(), &PromptConfig::default()).unwrap();
        assert_eq!(got, include_str!("text/fixture_session_specific_18.txt"));
    }

    #[test]
    fn language_section_matches_python_fixture() {
        let got = get_language_section(Some("Korean")).unwrap();
        assert_eq!(got, include_str!("text/fixture_language_korean.txt"));
    }

    #[test]
    fn intro_section_matches_python_dump() {
        assert_eq!(
            get_simple_intro_section(false),
            include_str!("text/simple_intro_section.txt")
        );
    }

    #[test]
    fn execution_discipline_composes() {
        let base = get_execution_discipline_section(false);
        let auto = get_execution_discipline_section(true);
        assert!(auto.starts_with(&base));
        assert!(auto.contains("Being stuck is different from being undecided."));
        assert!(base.starts_with("# Carrying out requested actions"));
    }

    #[test]
    fn static_sections_not_truncated() {
        assert!(CYBER_RISK_INSTRUCTION.contains("CTF"));
        assert!(get_actions_section().contains("measure twice, cut once."));
        assert!(get_simple_system_section().starts_with("# System"));
        assert!(get_simple_doing_tasks_section(&PromptConfig::default())
            .contains(MACRO_ISSUES_EXPLAINER));
        assert!(get_output_efficiency_section(&PromptConfig::default())
            .starts_with("# Output efficiency"));
        assert!(SUMMARIZE_TOOL_RESULTS_SECTION.starts_with("When working with tool results"));
        assert!(AGENT_NOTES.starts_with("Notes:"));
        assert!(DEFAULT_AGENT_PROMPT.starts_with("You are an agent for CodeMason"));
    }
}

// === Sub-agent prefix ===

/// The session-constant tail of every sub-agent / forked-skill system prompt:
/// agent notes, environment, language. Computed ONCE per process and reused
/// for every spawn.
///
/// Each spawn used to rebuild this — compute_env_info() re-read the cwd, the
/// language section re-read effective_language() — so two sub-agents in the
/// same session could carry different prefixes for no reason the user could
/// see. The prefix (tool block + this prompt) is ~20k tokens and sits ahead of
/// everything the sub-agent does; when it matches the previous spawn's the
/// server serves it from cache, when it does not the whole thing prefills cold
/// and pushes that many of the PARENT's blocks out of a pool that only holds
/// 1.62 conversations. The cwd the sub-agent should work in is stated in the
/// prompt the parent writes; the env block only needs to be stable.
pub fn agent_prefix_sections(model_id: &str) -> Vec<String> {
    // (image section it was built with, sections)
    static SECTIONS: Lazy<Mutex<Option<(Option<String>, Vec<String>)>>> = Lazy::new(|| Mutex::new(None));
    // The image section follows the limit the main prompt last asked for
    // (main_prompt_for_turn runs first each turn): it may go from absent to
    // present once, when the server was down at startup.
    let vision = get_vision_section(crate::services::vision::last_enabled_limit());
    let mut cached = SECTIONS.lock();
    if let Some((built_with, v)) = cached.as_ref() {
        if *built_with == vision {
            return v.clone();
        }
    }
    let mut v = vec![AGENT_NOTES.to_string(), compute_env_info(model_id, None)];
    if let Some(s) = &vision {
        v.push(s.clone());
    }
    if let Some(lang) = get_language_section(crate::utils::language::session_language().as_deref()) {
        v.push(lang);
    }
    *cached = Some((vision, v.clone()));
    v
}

const VISION_HEADING: &str = "# Images\n";

/// The main system prompt rebuilt with the image section the server now calls
/// for, when `built` carries another one (constants/prompts.py
/// refresh_system_prompt).
///
/// The prompt is built once per session and is the cached prefix, so it is not
/// rebuilt per turn. But it is built at startup, and a server that could not be
/// asked then (restarting, unreachable) gets no "# Images" section — for the
/// whole session, while images still go out once the server answers, to a
/// model that does not know its limit. Called at the start of each user turn;
/// it asks the server only while the limit is unknown. The rebuilt prompt is
/// kept, so a normal session rebuilds at most once (absent -> present) and
/// every later turn sends the same bytes.
pub async fn main_prompt_for_turn(
    built: &[String],
    tools: &[Arc<dyn Tool>],
    model: Option<&str>,
) -> Vec<String> {
    static REBUILT: Lazy<Mutex<Option<Vec<String>>>> = Lazy::new(|| Mutex::new(None));
    let want = get_vision_section(crate::services::vision::enabled_limit().await);
    if vision_section_of(built) == want {
        *REBUILT.lock() = None;
        return built.to_vec();
    }
    if let Some(p) = REBUILT.lock().as_ref() {
        if vision_section_of(p) == want {
            return p.clone();
        }
    }
    let p = get_system_prompt(tools, model, None).await;
    *REBUILT.lock() = Some(p.clone());
    p
}

fn vision_section_of(prompt: &[String]) -> Option<String> {
    prompt.iter().find(|s| s.starts_with(VISION_HEADING)).cloned()
}

#[cfg(test)]
mod vision_section_tests {
    use super::get_vision_section;

    /// What the Python port (constants/prompts.py get_vision_section) prints
    /// for 8, as a JSON string; pinned in its tests/test_vision_followups.py
    /// too, so a change on either side fails here or there.
    const PY_SECTION_8: &str = r##""# Images\n\nYou can see images. They reach you from the user's messages and from the Read tool on an image file, each labelled [Image N] in order of arrival. A very tall or wide image is read as several tiles; each tile is one image.\n - At most 8 images are attached to one request. Past that the oldest are replaced by a line \"[Image K: …; saved at PATH]\" that says whether you saw it earlier. Such an image is not in front of you: if you saw it, work from what you noted about it then; if you need to look again, or never saw it, Read the saved PATH — that attaches it as a new image and pushes an older one out.\n - So read at most 8 images at a time, and while they are attached write what you need from each into your reply text — not only into your thinking, which is dropped when the turn ends — instead of reading everything first and describing it afterwards.\n - If a task needs more than 8 images in view at once, say so rather than working from images that are no longer attached.""##;

    /// Text-only setups get no section; the text matches the Python port byte
    /// for byte.
    #[test]
    fn gated_and_matches_python() {
        assert_eq!(get_vision_section(0), None);
        let py: String = serde_json::from_str(PY_SECTION_8).unwrap();
        assert_eq!(get_vision_section(8).unwrap(), py);
    }
}

#[cfg(test)]
mod wsl_note_tests {
    use super::*;

    /// The note is a fact about the host, not about the moment. It sits in the
    /// system prompt, which is built once per session and forms the base of the
    /// server's cached prefix, so anything that varied between calls would move
    /// the front of every request. Two calls must be byte-identical.
    #[test]
    fn the_env_block_is_stable_across_calls() {
        let a = compute_env_info("glm-5.3-flash", None);
        let b = compute_env_info("glm-5.3-flash", None);
        assert_eq!(a, b, "the env block feeds the cached prefix");
    }

    /// Present under WSL, absent everywhere else — never a paragraph about
    /// Windows on a machine that has none.
    #[test]
    fn the_note_matches_the_host() {
        // Checked directly, so the wording is covered on a build machine that is
        // not itself WSL — otherwise this test would assert nothing here.
        let wsl = wsl_windows_note_for("wsl");
        assert!(wsl.contains("schtasks.exe /run"), "names the way through: {wsl}");
        assert!(wsl.contains("Start-Process -Verb RunAs"), "names what to avoid: {wsl}");
        assert!(wsl.contains("secure desktop"), "says why it cannot be automated: {wsl}");
        assert!(wsl.ends_with('\n'), "closes its line inside <env>");

        for other in ["linux", "macos", "windows", "unknown"] {
            assert!(
                wsl_windows_note_for(other).is_empty(),
                "no Windows advice on {other}"
            );
        }

        // And the live host agrees with itself.
        assert_eq!(
            wsl_windows_note(),
            wsl_windows_note_for(crate::utils::platform::get_platform())
        );
    }
}
