# codemason 3.0

[한국어](README.md) | **English**

**codemason is maintained as one implementation in two languages.** This repository is the
**Rust edition**; the Python edition lives at [AmesianX/codemason-py](https://github.com/AmesianX/codemason-py).
A fix on one side is carried to the other — features and behavior are kept in step.

> ## ⚠️ Disclaimer
>
> This repository is published **as-is**, without warranty of any kind.
>
> - **All consequences of and responsibility for** using this code rest with **you**.
> - This tool **reads, edits and deletes files and runs shell commands.** The model decides
>   what to run. Do not use it on an important working tree — only under version control.
> - **You are responsible for complying with applicable law and third-party terms**
>   (including model, API and service licenses).
>
> See [LICENSE](LICENSE).

A coding-agent CLI that runs in the terminal against a **local LLM** (vLLM / llama.cpp),
with no cloud dependency.

```
153 .rs files / 74,068 lines · 617 tests · Rust 2021 edition
```

## Highlights

The matching server is [dspark-serve](https://github.com/AmesianX/dspark-serve)
(GLM-5.3-Flash on two DGX Sparks, TP2). Items marked ⓢ are server features the client uses.

| Feature | What it does |
|---|---|
| **Effectively unlimited prefix cache** ⓢ | The server keeps prefix KV evicted from the GPU on 200 GB of NVMe per node, and it survives a restart. Continuing a long session, or reopening it with `/resume`, picks up from the cache instead of re-reading the prompt. |
| **Per-request `/ab`** ⓢ | `/ab off\|on\|auto` — runtime refusal removal (a rank-24 adapter on o_proj). It rides on each request as `vllm_xargs.ab_scale`; the server has no global switch (`POST /ab` returns 410). Your setting has zero effect on anyone else's output on the same server — verified by bit-identical logprobs of co-batched neighbor requests. `auto` turns it on only for the first request of a user turn (the one that decides whether to accept), because abliteration removes not only refusal but also the ability to say "that doesn't exist", and fabrication rises (4/60 → 18–22/60). |
| **Parallel sub-agents + priority** ⓢ | Up to 3 at once. Sub-agent requests carry vLLM priority 1, so the main conversation is served first from the queue. Sub-agent context budget: 30% of the parent's threshold each, 90% combined. |
| **Deterministic output** ⓢ | At temperature 0 the output is the same regardless of batch composition, load or restarts (server batch invariance). |
| **Web search** | Self-hosted SearXNG first; falls back to duckduckgo, brave, mojeek and google directly. When results are thin or engines are blocked, the server escalates to an identity-bearing browser search on its own — the client cannot request it. |
| **Image input** ⓢ | Up to 5 images per request. Photos received over Telegram are passed through as-is. |
| **Encrypted gateway** ⓢ | For remote access the request body is encrypted with AES-GCM. The key is never sent — holding the key is the credential. |
| **Remote (Telegram)** | `/remote` — give instructions, approve tools and answer questions from a phone. Long results arrive as files. |
| **Unattended runs** | `/goal` (a goal injected every turn) + completion gates (`/goal gate add <cmd>`) + `/keepgoing`. |
| **Exact token counts** | Ships the served model's tokenizer, so auto-compaction (at 88% of the context), the window check and usage are real counts, not estimates. |
| **Per-model sampling** | `sampling_profile: "auto"` — GLM gets the vendor's recommended values (temperature 1.0, top_p 0.95, no top_k) and config.json's values are ignored. `"manual"` sends config.json as-is. |
| **Multilingual UI** | `/language`. English source strings are the translation keys. |

## Install

```bash
./setup.sh                 # Linux/macOS — builds, then installs the binary together with vendor/
setup.bat                  # Windows — adds the install path to the user PATH (new terminals only)
cargo build --release      # to build by hand
```

Copying only the binary elsewhere loses the bundled ripgrep and falls back to a slow native
walk. That is why `setup.sh` / `setup.bat` move the two together.

## Usage

```bash
codemason                  # REPL — typing is queued and accepted even while it works
codemason -p "..."         # print mode: ask once, print only the result
codemason --help
```

The model backend is chosen in the config (`vllm` recommended / `llamacpp`). It talks to an
OpenAI-compatible endpoint.

## Inference engine recommendation

> **Running on llama.cpp is implemented, but for performance and completeness we strongly
> recommend against it. This project recommends vLLM.**

This project's b12x kernels exist only for vLLM; in the project's environment they give a
comfortable prefill of 1,300+ tokens/s and decoding of 35–40+ tokens/s. llama.cpp, even after
heavy optimization, stayed around 450+ tokens/s prefill and 25–30 tokens/s decoding — a
clearly noticeable difference.

The llama.cpp setup we used also had no built-in two-node tensor parallelism; we implemented
it ourselves and optimized further, and still did not reach acceptable responsiveness.

So development and optimization now focus on vLLM. llama.cpp has only basic execution
implemented, without further features or sufficient testing, and is not recommended.

## Layout

```
src/
├── main.rs            entry point
├── cli.rs             REPL UI — input engine and output routing
├── cli_repl.rs        REPL loop
├── query.rs           agent loop — model calls, tool execution, auto-compaction,
│                      max-token recovery, repeated-call brake
├── print_runner.rs    -p mode runner
├── task.rs / tasks/   background tasks
├── tools/             Read · Write · Edit · Bash · Glob · Grep · Agent ·
│                      WebFetch · MCP · TodoWrite · AskUserQuestion …
├── services/          backend adapters (vllm, llamacpp), compaction, vision, A/B
├── remote/            remote bot pool
├── state/             sessions · resume · transcripts
├── constants/         prompts and schemas
├── utils/             config, files, formatting, privilege elevation, completion, crypto …
└── locales/           translations (English source string is the key)
```

### How it works

1. **Input** — the REPL queues input. Typing never blocks while the model runs; it goes in
   as steering on the next turn.
2. **Agent loop** (`query.rs`) — calls the model, executes the tool calls it returns, and
   feeds the results back. Long conversations are compacted automatically, and repeated
   identical calls are braked.
3. **Tools** (`tools/`) — each tool carries its schema and its executor. File edits require
   a prior read; the shell goes through permission checks.
4. **State** (`state/`) — sessions are kept as transcripts so they can be resumed.

## License

MIT. See [LICENSE](LICENSE).

Independently implemented with reference to the behavior of Claude Code; not affiliated
with the original product.
