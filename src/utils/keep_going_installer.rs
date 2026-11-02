//! Self-installer for the keep-going Stop hook.
//! Equivalent to codemason/utils/keep_going_installer.py.
//!
//! keep-going is OFF by default (see config.rs). CodeMason auto-installs this
//! hook only when keep-going is effectively enabled (REPL `keepgoing on` / env
//! / config), so a fresh checkout materializes it on first such run. Turning
//! keep-going OFF does NOT uninstall it — the hook's own `_enabled()` check
//! short-circuits to a no-op (identical to having no Stop hook), which is
//! cheap and trivially reversible.
//!
//! Why embed the script as a string instead of shipping a data file: the whole
//! policy then travels inside version-controlled source. Move the binary to
//! another box and the first run re-materializes the hook there — no
//! package-data wiring, no stale file left behind on the old machine to chase.
//!
//! The hook is rewritten whenever its content drifts from HOOK_SCRIPT, so a
//! code update to the policy below propagates on the next run without a manual
//! reinstall.
//!
//! NOTE (unchanged from the Python original): the on-disk hook IS a Python
//! script and settings.json runs it with `python3` — the hook contract is
//! stdin-JSON/stdout-JSON, independent of what codemason itself is written in.
//! A host without python3 simply has a Stop hook that fails to run, which the
//! hooks layer reports as a non-blocking error.

use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::path::PathBuf;

use crate::utils::config::{CONFIG_DIR, SETTINGS_FILE};

pub static HOOKS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("hooks"));
pub static HOOK_FILE: Lazy<PathBuf> = Lazy::new(|| HOOKS_DIR.join("keep_going.py"));

/// What settings.json runs. Use the resolved hook path (honors
/// $CODEMASON_CONFIG_DIR) so it matches wherever the single config dir lives.
pub fn hook_command() -> String {
    format!("python3 \"{}\"", HOOK_FILE.display())
}

/// The Stop hook script, embedded verbatim (byte-identical to the Python
/// original's HOOK_SCRIPT). ensure_installed() rewrites the on-disk copy
/// whenever it differs from this string.
pub const HOOK_SCRIPT: &str = r##"#!/usr/bin/env python3
"""
CodeMason "Stop" hook — keep the model working until the task is actually done.

How it is wired: ~/.codemason/settings.json -> hooks.Stop runs this script every
time the model ends a turn WITHOUT calling a tool (the loop's natural stop
point). CodeMason feeds this script a JSON object on stdin and reads the result:

  - exit 0 with no output                -> let the model stop (turn ends)
  - stdout {"decision":"block","reason"} -> inject `reason` and CONTINUE the model
  - stdout {"continue": false}           -> force-stop the turn

stdin fields we use:
  - last_assistant_message : text of the model's final message this turn
  - used_tools             : True if the turn ran any tool since the last user
                             input (real agentic work vs a chat reply)
  - stop_hook_active       : True when the model is ALREADY continuing because
                             of this hook (set after a previous block). Use it
                             to avoid infinite loops.

Decision logic here = the "0 conversational gate + 1 sentinel + 2 progress-
safety" hybrid:
  0. If the turn did NO tool work since the last user input (used_tools=False),
     it's a pure conversational reply (greeting / plain Q&A) -> allow stop. This
     keeps keep-going from hijacking chat: only turns that actually ran tools are
     candidates for nudging. (Engine reports `used_tools` in the hook input.)
  1. If the model declared completion (printed TASK_COMPLETE) -> allow stop.
  2. Else, if we already nudged once (stop_hook_active) and it STILL ended with
     no tool use, treat it as done/stuck -> allow stop (loop guard). Note:
     productive tool turns reset stop_hook_active to False, so as long as the
     model keeps doing real tool work it keeps getting nudged; only TWO
     consecutive no-tool stops end it.
  3. Otherwise -> block and tell it to keep going.

There is NO hard turn cap (the hook owns termination).
The model emitting TASK_COMPLETE is the intended clean exit; otherwise the
progress-safety in step 2 stops it. You can always interrupt with Ctrl-C.

---------------------------------------------------------------------------
UPGRADE to "objective test gate" (best for a specific coding project): judge
completion by GROUND TRUTH instead of the model's word. Replace step 1 with a
real check, e.g.:

    import subprocess
    r = subprocess.run(['pytest', '-q'], cwd=d.get('cwd', '.'),
                       capture_output=True, text=True)
    if r.returncode == 0:
        sys.exit(0)                                   # tests pass -> done
    if d.get('stop_hook_active'):
        sys.exit(0)                                   # safety: still failing -> give up
    print(json.dumps({'decision': 'block',
        'reason': f'Tests still failing - keep fixing:\n{r.stdout[-1500:]}'}))
    sys.exit(0)

Keep that as a PER-PROJECT hook, not here -- this file is global and would run
the wrong test command in other projects.
---------------------------------------------------------------------------
"""
import json
import os
import sys
from pathlib import Path


def _enabled():
    """Effective keep-going state, resolved with this precedence:
       1. CODEMASON_KEEP_GOING env  (session override: CLI flag / REPL command)
       2. ~/.codemason/config.json "keep_going"  (persistent default)
       3. True  (default ON)
    OFF makes this hook a clean no-op -- identical to having no Stop hook."""
    env = os.environ.get('CODEMASON_KEEP_GOING')
    if env is not None:
        return env.strip().lower() not in ('0', 'false', 'no', 'off', '')
    try:
        cfg_dir = Path(os.environ.get('CODEMASON_CONFIG_DIR') or (Path.home() / '.codemason'))
        cfg = json.loads((cfg_dir / 'config.json').read_text())
        return bool(cfg.get('keep_going', False))
    except Exception:
        return False


# Cleanly disabled -> behave exactly like no Stop hook (let the model stop).
if not _enabled():
    sys.exit(0)

try:
    d = json.load(sys.stdin)
except Exception:
    # Can't read the hook input -> never block (fail open, let the model stop).
    sys.exit(0)

last = d.get('last_assistant_message') or ''

# 0) Pure conversational turn (no tool work since the last user input) -> stop.
#    `used_tools` defaults missing->True so an older engine that doesn't send it
#    keeps the prior always-nudge behavior rather than silently going quiet.
if not d.get('used_tools', True):
    sys.exit(0)

# 1) Model declared completion -> allow stop.
if 'TASK_COMPLETE' in last:
    sys.exit(0)

# 2) Loop guard: already nudged and it still ended with no tool use -> stop.
if d.get('stop_hook_active'):
    sys.exit(0)

# 3) Otherwise, push it to keep going.
print(json.dumps({
    'decision': 'block',
    'reason': (
        'Keep working toward the goal. Do not stop to ask for confirmation '
        'and do not summarize prematurely. Continue using tools until the '
        'task is fully done. When it is genuinely complete, end your final '
        'message with the exact line: TASK_COMPLETE'
    ),
}))
"##;

fn read_settings() -> Value {
    if !SETTINGS_FILE.exists() {
        return json!({});
    }
    std::fs::read_to_string(&*SETTINGS_FILE)
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or_else(|| json!({}))
}

fn stop_hook_registered(settings: &Value) -> bool {
    let cmd = hook_command();
    let stop = settings
        .get("hooks")
        .and_then(|h| h.get("Stop"))
        .and_then(|s| s.as_array());
    let Some(stop) = stop else {
        return false;
    };
    for entry in stop {
        let Some(hooks) = entry.get("hooks").and_then(|h| h.as_array()) else {
            continue;
        };
        for h in hooks {
            if h.get("command").and_then(|c| c.as_str()) == Some(cmd.as_str()) {
                return true;
            }
        }
    }
    false
}

/// True if both the hook file exists AND settings.json registers it.
pub fn is_installed() -> bool {
    if !HOOK_FILE.exists() {
        return false;
    }
    stop_hook_registered(&read_settings())
}

/// Materialize the hook on this machine. Idempotent: writes the script only
/// when missing/drifted, registers the Stop hook only when absent. Returns a
/// list of human-readable actions taken (empty = already fully installed).
pub fn ensure_installed() -> Vec<String> {
    let mut actions = Vec::new();

    let _ = std::fs::create_dir_all(&*HOOKS_DIR);
    let current = std::fs::read_to_string(&*HOOK_FILE).ok();
    if current.as_deref() != Some(HOOK_SCRIPT) {
        // Atomic, because this file is EXECUTED by whoever runs the hook. A
        // plain write truncates first, so a second session starting in that
        // window — or the hook runner itself — ran a half-written Python script.
        // Identical content from every writer, so no lock is needed: the rename
        // publishes the whole file or nothing.
        if crate::utils::file::atomic_write_bytes(
            &HOOK_FILE.to_string_lossy(),
            HOOK_SCRIPT.as_bytes(),
        )
        .is_ok()
        {
            actions.push(format!("wrote {}", HOOK_FILE.display()));
        }
    }

    // The cheap pre-check keeps the common case (already installed) off the lock
    // entirely; the authoritative one is inside the callback, where it cannot
    // straddle another process's write. Two sessions that both read "not
    // registered" and then both appended left the Stop hook registered TWICE,
    // and settings.json holds the hook and permission configuration, so a plain
    // write also risked publishing a truncated version of it.
    if !stop_hook_registered(&read_settings()) {
        if let Some(parent) = SETTINGS_FILE.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let registered = crate::utils::file::update_json_document(
            &SETTINGS_FILE,
            &crate::utils::config::lock_path("settings.json"),
            |current| {
                let mut settings = match current {
                    Some(v) if v.is_object() => v,
                    _ => json!({}),
                };
                if stop_hook_registered(&settings) {
                    return None; // another process got there first
                }
                // settings.setdefault('hooks', {}).setdefault('Stop', []) —
                // coercing a non-list Stop entry to a list, exactly like the Python.
                let obj = settings.as_object_mut()?;
                let hooks = obj.entry("hooks".to_string()).or_insert_with(|| json!({}));
                if !hooks.is_object() {
                    *hooks = json!({});
                }
                let hooks_obj = hooks.as_object_mut()?;
                let stop = hooks_obj.entry("Stop".to_string()).or_insert_with(|| json!([]));
                if !stop.is_array() {
                    *stop = json!([]);
                }
                stop.as_array_mut()?.push(json!({
                    "hooks": [{"type": "command", "command": hook_command()}]
                }));
                Some(settings)
            },
            false,
        );
        if registered {
            actions.push(format!("registered Stop hook in {}", SETTINGS_FILE.display()));
        } else if SETTINGS_FILE.exists() && read_settings().as_object().map_or(true, |m| m.is_empty())
        {
            // The document exists but did not parse, so it was left alone rather
            // than replaced with just our hook (see utils::file). Say so: the
            // alternative is a keep-going that silently never fires.
            actions.push(format!("NOT registered: {} could not be read", SETTINGS_FILE.display()));
        }
    }

    actions
}
