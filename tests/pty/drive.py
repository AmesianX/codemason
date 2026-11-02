#!/usr/bin/env python3
"""Drive the interactive REPL under a PTY and dump the rendered screen.

Development tool (not part of `cargo test`): the REPL keeps a resident prompt
block that it repaints continuously, so neither pipe capture nor an
"output went quiet" heuristic works — the byte stream never goes quiet, it just
redraws the same cells.  pyte replays the ANSI stream into a virtual screen, and
we key every decision off the *rendered screen* instead of the byte stream.

Usage:
    .venv-test/bin/python tests/pty/drive.py --cwd DIR -- STEP [STEP ...]

A STEP is `send:TEXT` (typed as-is; \\r, \\t, \\x1b[B are unescaped) or
`wait:SECONDS`.  Sends wait for the prompt to be idle first, so keystrokes are
never delivered before the prompt engine is listening.
"""
import argparse
import fcntl
import os
import pty
import re
import struct
import termios
import select
import sys
import time

import pyte


# Both implementations probe the terminal with a cursor-position request and
# refuse to take over stdin until it is answered (prompt_toolkit prints
# "your terminal doesn't support cursor position requests" and falls back).
# pyte does not answer, so we answer for it from the rendered cursor.
CPR_RE = re.compile(rb"\x1b\[6n")


def answer_cpr(fd, data, screen):
    for _ in CPR_RE.findall(data):
        reply = f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R".encode()
        try:
            os.write(fd, reply)
        except OSError:
            return


def snapshot(screen):
    """Rendered screen as text, trailing blank lines removed."""
    lines = list(screen.display)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--bin", default="target/release/codemason")
    ap.add_argument("--args", default="", help="extra argv for the binary, space split")
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--idle", type=float, default=1.5,
                    help="seconds the rendered screen must stay unchanged to count as idle")
    ap.add_argument("--settle", type=float, default=8.0,
                    help="idle seconds with no steps left that ends the run")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--raw", help="also write the raw PTY byte stream here")
    ap.add_argument("steps", nargs="*")
    args = ap.parse_args()

    binary = os.path.abspath(args.bin)
    argv = [binary] + [a for a in args.args.split(" ") if a]
    screen = pyte.Screen(args.cols, args.rows)
    stream = pyte.ByteStream(screen)
    raw = open(args.raw, "wb") if args.raw else None

    pid, fd = pty.fork()
    if pid == 0:
        # The child's controlling terminal must report the geometry we render
        # into, or crossterm/prompt_toolkit fall back to 80x24 and the wrap math
        # no longer matches the pyte screen.
        fcntl.ioctl(0, termios.TIOCSWINSZ,
                    struct.pack("HHHH", args.rows, args.cols, 0, 0))
        os.chdir(args.cwd)
        env = dict(os.environ, TERM="xterm-256color",
                   COLUMNS=str(args.cols), LINES=str(args.rows))
        os.execve(binary, argv, env)
        os._exit(127)

    steps = list(args.steps)
    deadline = time.time() + args.timeout
    last_change = time.time()
    prev = snapshot(screen)
    dead = False

    try:
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    dead = True
                    break
                if not data:
                    dead = True
                    break
                if raw:
                    raw.write(data)
                    raw.flush()
                stream.feed(data)
                answer_cpr(fd, data, screen)
                cur = snapshot(screen)
                if cur != prev:          # repaints of identical cells are not change
                    prev = cur
                    last_change = time.time()
                continue

            idle_for = time.time() - last_change
            if steps and idle_for >= args.idle:
                step = steps.pop(0)
                if step.startswith("send:"):
                    text = step[5:].encode().decode("unicode_escape")
                    os.write(fd, text.encode())
                elif step.startswith("wait:"):
                    time.sleep(float(step[5:]))
                last_change = time.time()
                continue

            if not steps and idle_for >= args.settle:
                break
    finally:
        if raw:
            raw.close()
        if not dead:
            try:
                os.write(fd, b"\x03")     # Ctrl-C, then let it go
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass
        for _ in range(20):
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass

    print(snapshot(screen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
