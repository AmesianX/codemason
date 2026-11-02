//! ``codemason whoami`` / ``inject`` / ``notify`` — the self-wake surface.
//! Equivalent to codemason/utils/wake_cli.py.
//!
//! Intercepted at the top of cli::main() alongside ``mcp`` and ``skill``
//! (before arg parsing, which owns the positional ``prompt``).
//!
//! WHY THIS EXISTS. A session can already be woken from outside: the remote
//! poller files a message into ``~/.codemason/remote/inbox/<id>.jsonl`` and
//! the session drains that into the queue its prompt reads from, once a
//! second (remote/input.rs, DRAIN_INTERVAL). Outbound alerts have the same
//! shape through remote/notify.rs. Both were reachable only from inside the
//! Rust process, so an agent that wanted to report a result hours later —
//! a finished build, a job on another machine — had no way to arrange it and
//! no way to learn the machinery was there. Asked directly, it answered that no
//! such path existed while reading a message that had arrived through it.
//! These three commands are that machinery, named and callable from a shell
//! script, so the agent can wire its own callback instead of waiting to be
//! asked.
//!
//!     codemason whoami                      this session's id, cwd, remote state
//!     codemason inject [--session ID] TEXT  wake a session with TEXT
//!     codemason notify TITLE [--body B]     push an alert to the phone
//!
//! `inject` reaches a LIVE session's queue; `notify` reaches the phone whether
//! or not a session is running. A watcher wanting both — triage when someone
//! is home, a buzz when nobody is — calls both.

use serde_json::json;

use crate::remote::{inbox, notify};
use crate::remote::events::{EventKind, Sensitivity};

/// Name of the environment variable a session exports for its children, so a
/// script it spawns can address it without being told the id.
pub const SESSION_ENV: &str = "CODEMASON_SESSION_ID";

/// Absolute path of the running binary, exported beside SESSION_ENV.
///
/// The command's NAME is not knowable from inside: this ships as `codemason`
/// but is installed behind whatever symlink the site chose (`cm2` here), and
/// that symlink sits in ~/.local/bin, which a login shell has on PATH and a
/// cron line or an ssh command does not. A watcher told to run `codemason
/// inject` therefore dies with "command not found" in exactly the unattended
/// context it was written for. The absolute path has neither problem.
pub const BIN_ENV: &str = "CODEMASON_BIN";

/// Export both, so a child process can address this session and re-enter this
/// program without knowing either.
pub fn export_self(session_id: &str) {
    std::env::set_var(SESSION_ENV, session_id);
    if let Ok(exe) = std::env::current_exe() {
        std::env::set_var(BIN_ENV, exe);
    }
}

/// Withdraw the id when the session stops being reachable.
///
/// `/remote off` unregisters, which stops the drain; a watcher started before
/// that holds the id in its own environment and cannot be told. But anything
/// this session spawns AFTER the variable is cleared reads the absence and
/// refuses, instead of addressing a session that no longer takes messages.
pub fn unexport_self() {
    std::env::remove_var(SESSION_ENV);
}

const WHOAMI_HELP: &str = "codemason whoami — this session's identity and delivery state

  --session-id    print only the session id (empty if none), for scripts
  --json          machine-readable";

const INJECT_HELP: &str = "codemason inject — wake a session with a message

  codemason inject [--session ID] [--source NAME] TEXT
  codemason inject [--session ID] -            read TEXT from stdin

The message lands in the session's inbox and is picked up within a second,
exactly as a message from the phone is. This is local: no account, no network.
The session must be running — with nothing alive this fails rather than
queueing for a session that may never come back. Pair it with
`codemason notify` when the alert must arrive either way.

Target resolution, in order: --session, $CODEMASON_SESSION_ID, then — only when
run at a terminal, where a wrong guess is seen — the single live session.
Unaddressed from a script it is an error, as is an ambiguous or dead target.

TEXT ARRIVES AS THE USER'S OWN WORDS. The session cannot tell it apart from
something the person typed, so it carries the authority of an instruction.
Piping a log or any other program's output into it hands that
authority to whatever wrote it: say what happened and where to look, rather
than pasting the bytes in.

The queue keeps only the most recent messages (200, or 256 KiB), dropping
oldest-first, so a watcher should send one short message per event rather
than a stream.";

const NOTIFY_HELP: &str = "codemason notify — push an alert to the user

  codemason notify [--body TEXT] [--kind KIND] [--sensitivity LEVEL] TITLE
  codemason notify --body - TITLE                 read the body from stdin

  --kind         info (default), task_done, error
  --sensitivity  public (default), internal, secret

Goes out over whatever channels this machine has configured; it does not need a
session to be running and does not need credentials to be read or passed.";

fn run_async<F: std::future::Future<Output = i32>>(fut: F) -> i32 {
    match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt.block_on(fut),
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn wants_help(argv: &[String]) -> bool {
    argv.first().map(|s| matches!(s.as_str(), "-h" | "--help" | "help")).unwrap_or(false)
}

/// Read stdin to end; used for `-` arguments so a script can pipe a body in
/// rather than fight the shell over quoting it.
fn read_stdin() -> String {
    use std::io::Read;
    let mut s = String::new();
    let _ = std::io::stdin().read_to_string(&mut s);
    s
}

/// The session this process should address, and why.
///
/// Returns Err with a message meant to be printed as-is: either nothing is
/// running, or several are and the caller has to say which. Both are refusals,
/// not fallbacks — delivering a report to the wrong session is worse
/// than not delivering it, because it looks delivered.
fn resolve_session(explicit: Option<&str>) -> Result<String, String> {
    let live = inbox::live_sessions();
    let env_id = std::env::var(SESSION_ENV).unwrap_or_default();
    pick_session(
        &live,
        explicit,
        if env_id.is_empty() { None } else { Some(env_id.as_str()) },
        at_a_terminal(),
    )
}

/// Whether a person is watching this invocation.
///
/// Decides whether "the only live session" is a convenience or a hazard. Typed
/// at a prompt it is obviously right and saves an argument nobody wants to look
/// up. Run from a watcher, a cron line or a `-p` process it is a guess about
/// someone else's conversation, made where no one will see it be wrong — and a
/// `-p` run never registers a session, so it has no id of its own to prefer and
/// would take whatever interactive session happened to be open.
fn at_a_terminal() -> bool {
    #[cfg(unix)]
    {
        // SAFETY: isatty reads a descriptor's type and has no side effects.
        unsafe { libc::isatty(libc::STDIN_FILENO) == 1 && libc::isatty(libc::STDERR_FILENO) == 1 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// The decision itself, with the world passed in: which live session a wake is
/// for. Separated from the lookup so the refusals can be tested — they are the
/// whole point of the function, and both of them are silent failures if they
/// ever turn into fallbacks.
fn pick_session(
    live: &[inbox::SessionEntry],
    explicit: Option<&str>,
    env_id: Option<&str>,
    interactive: bool,
) -> Result<String, String> {
    if let Some(id) = explicit {
        return match live.iter().find(|s| s.session_id == id) {
            Some(s) => Ok(s.session_id.clone()),
            None => Err(format!("no live session {id}\n{}", describe_live(live))),
        };
    }
    // A session exports its own id to everything it spawns, so a watcher the
    // agent wrote from inside the session needs no argument at all.
    if let Some(env_id) = env_id {
        if let Some(s) = live.iter().find(|s| s.session_id == env_id) {
            return Ok(s.session_id.clone());
        }
        return Err(format!(
            "{SESSION_ENV} names {env_id}, which is not running\n{}",
            describe_live(live)
        ));
    }
    if live.is_empty() {
        return Err(
            "no live session to wake (start one, or send an alert with `notify` instead — it needs none)"
                .into(),
        );
    }
    // Unattended and unaddressed: refuse. The convenience below is only a
    // convenience when someone is there to see it pick wrong.
    if !interactive {
        return Err(format!(
            "no target: {SESSION_ENV} is not set and this is not a terminal, so there is \
             nobody to catch a wrong guess. Pass --session, or run from a process that \
             inherited {SESSION_ENV} from the session you mean.\n{}",
            describe_live(live)
        ));
    }
    if live.len() == 1 {
        return Ok(live[0].session_id.clone());
    }
    Err(format!(
        "several sessions are live; name one with --session\n{}",
        describe_live(live)
    ))
}

fn describe_live(live: &[inbox::SessionEntry]) -> String {
    if live.is_empty() {
        return "  (no live sessions)".into();
    }
    live.iter()
        .map(|s| format!("  {}  pid {}  {}", s.session_id, s.pid, s.label))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn handle_whoami_cli(argv: &[String]) -> i32 {
    if wants_help(argv) {
        println!("{WHOAMI_HELP}");
        return 0;
    }
    let id_only = argv.iter().any(|a| a == "--session-id");
    let as_json = argv.iter().any(|a| a == "--json");

    let live = inbox::live_sessions();
    let env_id = std::env::var(SESSION_ENV).unwrap_or_default();
    // Outside a session there is no "me"; a single live session is still a
    // useful answer for a script, so report it and say where it came from.
    let (me, via) = if !env_id.is_empty() {
        (env_id, SESSION_ENV)
    } else if live.len() == 1 {
        (live[0].session_id.clone(), "the only live session")
    } else {
        (String::new(), "nothing claimed")
    };

    if id_only {
        println!("{me}");
        return if me.is_empty() { 1 } else { 0 };
    }

    let mine = live.iter().find(|s| s.session_id == me);
    let configured = notify::is_configured();
    let off = notify::off_reason();
    let pool = notify::all_adapters().len();

    if as_json {
        let v = json!({
            "session_id": me,
            "resolved_via": via,
            "pid": mine.map(|s| s.pid),
            "cwd": mine.map(|s| s.cwd.clone()),
            "label": mine.map(|s| s.label.clone()),
            "remote_configured": configured,
            "remote_off_reason": off,
            "bot_pool": pool,
            "live_sessions": live.iter().map(|s| json!({
                "session_id": s.session_id, "pid": s.pid, "label": s.label, "cwd": s.cwd,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        return 0;
    }

    if me.is_empty() {
        println!("session: (none — {via})");
    } else {
        println!("session: {me}  ({via})");
        if let Some(s) = mine {
            println!("  pid {}  cwd {}  label {}", s.pid, s.cwd, s.label);
        }
    }
    println!(
        "remote : {}",
        if configured {
            format!("on, {pool} bot(s) configured")
        } else if off.is_empty() {
            "not configured (~/.codemason/remote.env)".to_string()
        } else {
            format!("off — {off}")
        }
    );
    println!("live   :\n{}", describe_live(&live));
    0
}

pub fn handle_inject_cli(argv: &[String]) -> i32 {
    if argv.is_empty() || wants_help(argv) {
        println!("{INJECT_HELP}");
        return if argv.is_empty() { 2 } else { 0 };
    }
    let mut session: Option<String> = None;
    let mut source = "inject".to_string();
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--session" | "-s" => {
                i += 1;
                match argv.get(i) {
                    Some(v) => session = Some(v.clone()),
                    None => {
                        eprintln!("--session needs a session id");
                        return 2;
                    }
                }
            }
            "--source" => {
                i += 1;
                match argv.get(i) {
                    Some(v) => source = v.clone(),
                    None => {
                        eprintln!("--source needs a name");
                        return 2;
                    }
                }
            }
            other => parts.push(other.to_string()),
        }
        i += 1;
    }

    let text = if parts.len() == 1 && parts[0] == "-" {
        read_stdin()
    } else {
        parts.join(" ")
    };
    let text = text.trim_end_matches('\n').to_string();
    if text.trim().is_empty() {
        eprintln!("nothing to inject (give TEXT, or `-` to read stdin)");
        return 2;
    }

    let target = match resolve_session(session.as_deref()) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            return 1;
        }
    };

    // `source` rides along so the session can tell a machine's message from a
    // person's; the delivery path is the one the phone already uses.
    let msg = inbox::InboundMessage::new(text, source, json!({}));
    if inbox::deliver(&target, &msg) {
        println!("delivered to {target}");
        0
    } else {
        eprintln!("could not write to the inbox of {target}");
        1
    }
}

pub fn handle_notify_cli(argv: &[String]) -> i32 {
    if argv.is_empty() || wants_help(argv) {
        println!("{NOTIFY_HELP}");
        return if argv.is_empty() { 2 } else { 0 };
    }
    let mut body = String::new();
    let mut kind = EventKind::Info;
    let mut sensitivity = Sensitivity::Public;
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--body" | "-b" => {
                i += 1;
                match argv.get(i) {
                    Some(v) if v == "-" => body = read_stdin(),
                    Some(v) => body = v.clone(),
                    None => {
                        eprintln!("--body needs text (or `-` for stdin)");
                        return 2;
                    }
                }
            }
            "--kind" | "-k" => {
                i += 1;
                kind = match argv.get(i).map(|s| s.as_str()) {
                    Some("info") => EventKind::Info,
                    Some("task_done") => EventKind::TaskDone,
                    Some("error") => EventKind::Error,
                    Some(other) => {
                        eprintln!("unknown --kind {other} (info, task_done, error)");
                        return 2;
                    }
                    None => {
                        eprintln!("--kind needs a value");
                        return 2;
                    }
                };
            }
            "--sensitivity" => {
                i += 1;
                sensitivity = match argv.get(i).map(|s| s.as_str()) {
                    Some("public") => Sensitivity::Public,
                    Some("internal") => Sensitivity::Internal,
                    Some("secret") => Sensitivity::Secret,
                    Some(other) => {
                        eprintln!("unknown --sensitivity {other} (public, internal, secret)");
                        return 2;
                    }
                    None => {
                        eprintln!("--sensitivity needs a value");
                        return 2;
                    }
                };
            }
            other => parts.push(other.to_string()),
        }
        i += 1;
    }

    let title = parts.join(" ");
    let title = title.trim();
    if title.is_empty() {
        eprintln!("nothing to send (give TITLE)");
        return 2;
    }
    if !notify::is_configured() {
        let off = notify::off_reason();
        eprintln!(
            "the remote channel cannot deliver: {}",
            if off.is_empty() { "not configured (~/.codemason/remote.env)".to_string() } else { off }
        );
        return 1;
    }

    let title = title.to_string();
    run_async(async move {
        notify::notify_simple(kind, &title, &body, sensitivity).await;
        // A one-shot process exits the moment the call returns, which would
        // drop the delivery still in flight; this is the same wait the REPL
        // does on its way out.
        notify::drain_at_exit().await;
        println!("sent");
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, pid: i64) -> inbox::SessionEntry {
        inbox::SessionEntry {
            session_id: id.to_string(),
            pid,
            cwd: "/tmp".into(),
            label: "~".into(),
            started: 0.0,
        }
    }

    /// With one session running and nothing said, that session is the target —
    /// the ordinary case, and the one a watcher script relies on.
    #[test]
    fn a_lone_session_needs_no_argument() {
        let live = vec![entry("aaa", 1)];
        assert_eq!(pick_session(&live, None, None, true).unwrap(), "aaa");
    }

    /// Two sessions and no choice given must REFUSE. Picking one would deliver
    /// a report to the wrong conversation, which reads as delivered and
    /// is therefore worse than not delivering it.
    #[test]
    fn an_ambiguous_target_is_refused_not_guessed() {
        let live = vec![entry("aaa", 1), entry("bbb", 2)];
        let err = pick_session(&live, None, None, true).unwrap_err();
        assert!(err.contains("--session"), "the refusal must say how to resolve it: {err}");
        assert!(err.contains("aaa") && err.contains("bbb"), "it must list the candidates: {err}");
    }

    /// The environment variable wins over "the only one", so a script spawned
    /// by session A keeps addressing A even after B starts beside it.
    #[test]
    fn the_exported_id_decides_before_the_count_does() {
        let live = vec![entry("aaa", 1), entry("bbb", 2)];
        assert_eq!(pick_session(&live, None, Some("bbb"), true).unwrap(), "bbb");
    }

    /// A named session that has since exited is an error, not a fallback to
    /// whatever else is running: the watcher is reporting to a conversation
    /// that no longer exists and has to be told so.
    #[test]
    fn a_dead_target_does_not_fall_back_to_a_live_one() {
        let live = vec![entry("aaa", 1)];
        assert!(pick_session(&live, Some("gone"), None, true).is_err());
        assert!(pick_session(&live, None, Some("gone"), true).is_err());
    }

    /// Unattended and unaddressed: the "only live session" convenience is a
    /// guess about someone else's conversation, and a script is exactly where
    /// nobody is watching it be wrong. A `-p` run is the sharp case — it
    /// registers no session, so it has no id of its own and would otherwise
    /// take whatever happened to be open.
    #[test]
    fn a_script_with_no_target_refuses_rather_than_taking_the_only_session() {
        let live = vec![entry("aaa", 1)];
        let err = pick_session(&live, None, None, false).unwrap_err();
        assert!(err.contains("--session") && err.contains(SESSION_ENV), "{err}");
        // An inherited or explicit target is a target, terminal or not.
        assert_eq!(pick_session(&live, None, Some("aaa"), false).unwrap(), "aaa");
        assert_eq!(pick_session(&live, Some("aaa"), None, false).unwrap(), "aaa");
    }

    /// Nothing running at all: say so, and point at the command that still
    /// works without a session.
    #[test]
    fn with_nothing_running_it_names_the_other_route() {
        let err = pick_session(&[], None, None, true).unwrap_err();
        assert!(err.contains("notify"), "it must point at the session-free route: {err}");
    }
}
