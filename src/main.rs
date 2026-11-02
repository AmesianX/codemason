//! CodeMason — AI-powered coding assistant CLI (Rust port).
//! Entry point; equivalent of codemason/cli.py main().

mod cli;
mod cli_repl;
mod constants;
mod locales;
mod print_runner;
mod query;
mod remote;
mod services;
mod state;
mod task;
mod tasks;
mod tools;
mod types;
mod utils;

/// atexit trampoline: Python registers tasks.local_shell_task.shutdown_all
/// with atexit at import (local_shell_task.py:626) so background commands are
/// reaped on ANY normal exit path, not only the tidy one. libc::atexit is the
/// direct equivalent — it also runs under std::process::exit, which several
/// early-exit paths use.
extern "C" fn shutdown_all_at_exit() {
    crate::tasks::local_shell_task::shutdown_all();
    // The same guarantee for sub-agents, through the registry the source uses
    // for them (pq/xj7): each spawn registers "kill me", so an agent cannot
    // outlive the process that owns it. Python gets this from
    // utils/cleanup_registry.py's own atexit.register.
    crate::utils::cleanup_registry::run_cleanup_functions();
}

/// Restore the terminal before the default panic handler prints.
///
/// The prompt engine puts the terminal in raw mode, hides the hardware cursor
/// and enables bracketed paste, and undoes all three only on its normal
/// teardown path. A panic unwinds straight past that, so any panic anywhere —
/// a slice bug in a tool, an invariant broken deep in a turn — dropped the user
/// back to a shell with no echo, no visible cursor and paste brackets still on.
/// They then have to blind-type `reset`, which makes a crash materially worse
/// than the crash itself.
///
/// Deliberately a hook rather than a Drop guard: a Drop on the engine would only
/// cover unwinding through the engine's own frame, and would do nothing for
/// panic=abort. The hook runs first, then the default handler prints the message
/// and backtrace as usual — the panic is not swallowed.
///
/// SCOPE, measured on Windows rather than assumed: this restores the TERMINAL,
/// not the session. Most panics here happen on a tokio worker, and a worker
/// panic kills only that task — the process keeps running with its input engine
/// dead, so the screen is readable and correct but nothing responds, and the
/// user still has to interrupt or kill it. That is much better than being
/// dropped into an invisible shell, which is what happened before, but "made
/// survivable" means the terminal survives, not the session.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic a catch_unwind is about to swallow on purpose must NOT drag
        // the terminal out of raw mode: the program carries on, and the engine
        // would be left believing it still owns a raw terminal. The hook runs
        // before unwinding, so catch_unwind alone does not keep us out.
        if !crate::utils::panic_guard::in_expected_panic() {
            // Leave the cause on disk BEFORE touching the terminal. stderr in an
            // overnight run is scrollback that may be gone by morning, and the
            // default handler prints no stack unless RUST_BACKTRACE was set —
            // so without this a crash leaves one line and no location. The run
            // is lost either way; the reason for it does not have to be.
            let message = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic with no message".to_string());
            let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
            crate::utils::log::log_panic(
                &message,
                location,
                std::env::var("CODEMASON_SESSION_ID").ok(),
            );

            // Restore in the reverse order the engine sets them up. Each is
            // already a no-op when it was never enabled, and none can panic.
            crate::utils::tty_compat::disable_raw_mode();
            crate::utils::tty_compat::restore_echo();
            // Only when stderr is a terminal. With it redirected there is no
            // raw mode and no bracketed paste to undo, so these would be pure
            // noise — and `-p` runs with stderr captured to a log are a normal
            // way to use this, where the escapes land as 14 stray bytes at the
            // top of the file. Measured on Windows: the sequences themselves
            // are pure ASCII and stderr has VT processing on by that point, so
            // on a real console they are interpreted rather than printed.
            if crossterm::tty::IsTty::is_tty(&std::io::stderr()) {
                use std::io::Write;
                let mut o = std::io::stderr();
                let _ = write!(o, "\x1b[?25h"); // show the hardware cursor
                let _ = write!(o, "\x1b[?2004l"); // disable bracketed paste
                let _ = o.flush();
            }
        }
        previous(info);
    }));
}

/// Stack for the thread the whole CLI runs on.
///
/// query() recurses once per tool-calling round: every tool result re-enters it
/// with the grown history (query.rs, "Recursive query with tool results"), and
/// the recovery paths add their own levels on top. The frames are async
/// generator states, so they are large and they all stay live — the outer
/// frames are suspended at `while let Some(m) = inner.next().await`, not
/// returned. Depth therefore tracks the number of tool rounds in a turn, which
/// on a long unattended run is hundreds.
///
/// The default 8 MB main-thread stack does not survive that. Observed
/// 2026-09-06 at 271k tokens of context, mid-task:
///
///     thread 'main' has overflowed its stack
///     fatal runtime error: stack overflow, aborting
///
/// which kills the process outright — no panic hook, no terminal restore, no
/// transcript flush of the turn in flight. Exactly the failure an unattended
/// run cannot survive, and the longer the run the likelier it gets.
///
/// 256 MB is reserved address space, not committed memory: pages are faulted in
/// only as the stack actually grows, so a short session pays nothing. This
/// raises the ceiling rather than removing it — the real fix is to drive the
/// tool loop iteratively instead of recursively, which is a restructuring of
/// query() and is tracked separately.
const MAIN_STACK_BYTES: usize = 256 * 1024 * 1024;

fn main() {
    unsafe {
        libc::atexit(shutdown_all_at_exit);
    }
    install_panic_hook();
    // Run everything on a thread with a stack sized for the recursion above.
    // std::thread::Builder is the only portable way to set it; the main
    // thread's stack is fixed by the OS before any Rust code runs.
    let child = std::thread::Builder::new()
        .name("codemason-main".into())
        .stack_size(MAIN_STACK_BYTES)
        .spawn(|| {
            crate::cli::main();
        })
        .expect("spawn main thread");
    // Propagate a panic so the exit status still reflects failure; the hook has
    // already run and restored the terminal by this point.
    if child.join().is_err() {
        std::process::exit(101);
    }
}
