#!/usr/bin/env python3
"""The cached prefix must not move for the life of a session.

The tool block and the system prompt sit AHEAD of the entire message history in
every request, so they are the base of the server's prefix cache. One changed
byte there throws away every cached token behind it. Measured 2026-09-06 on a
121k-token session: a single flip cost a full re-prefill, ~123 seconds before
the first token appeared, and the two cold prefills in a 27-request session were
exactly the two requests where the tool block changed.

WHY THE TEST LOOKS LIKE THIS. The regression is only visible ACROSS REQUESTS
INSIDE ONE PROCESS. Comparing separate runs proves nothing: two sessions started
in different directories are allowed to differ, and an earlier version of this
test failed on exactly that and reported three phantom regressions.

So it reproduces the real thing. Asking a question in Korean about an English
file forces the flip in one session:

    request 1: last user message = the Korean question   -> language detected
    request 2: last user message = the English tool result -> not detected

The detected language is read from the LAST user message, and inside a tool loop
that is a tool RESULT, so it swings with whatever a command happens to print.
That value used to be built into the Agent tool's JSON schema.

Run against a fixed build it passes; run against the pre-fix commit every
request pair differs.

Usage:  prefix_stability.py            # exit 1 on drift
"""
import os, re, subprocess, sys, tempfile

CM = os.environ.get("CM_BIN", "cm2")

def parse(line):
    tools = dict(x.split(":") for x in re.search(r"\[([^]]*)\]", line).group(1).split(","))
    block = re.search(r"tools=(\w+)", line).group(1)
    sysmsg = line.split("] ", 1)[1].split(" ")[0]  # msg[0] is the system message
    return block, tools, sysmsg

def main():
    root = tempfile.mkdtemp(prefix="prefixtest-")
    log = os.path.join(root, "dump.log")
    # Deliberately English, so reading it swings the detected language away from
    # the Korean question that asked for it.
    with open(os.path.join(root, "notes.txt"), "w") as f:
        f.write("Deployment checklist\n- verify build\n- run tests\n- ship\n")
    env = dict(os.environ, CODEMASON_PREFIX_DUMP=log)
    subprocess.run([CM, "-p", "-dsp", "notes.txt 파일을 읽고 한 줄로 요약해줘"],
                   cwd=root, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=600)
    if not os.path.exists(log):
        print("FAIL — no prefix dump produced"); return 1
    rows = [parse(l) for l in open(log) if l.strip()]
    if len(rows) < 2:
        print(f"INCONCLUSIVE — the run made {len(rows)} request(s); the model "
              f"never called a tool, so no in-session flip could occur.")
        return 2
    base_block, base_tools, base_sys = rows[0]
    failures = []
    for i, (block, tools, sysmsg) in enumerate(rows[1:], start=2):
        if block != base_block:
            moved = [k for k in set(tools) | set(base_tools)
                     if tools.get(k) != base_tools.get(k)]
            failures.append(f"request {i}: tool block changed ({', '.join(moved)})")
        if sysmsg != base_sys:
            failures.append(f"request {i}: system prompt changed")
    print(f"{len(rows)} requests in one session")
    if failures:
        print("\nFAIL — the cached prefix moved mid-session:")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nPASS — tool block and system prompt held across every request.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
