# LLM Research Pipeline

Local-LLM-driven research pipeline. Takes a plain-text research brief, produces a comprehensive markdown report by generating search queries, fetching pages, extracting brief-relevant content, and synthesizing findings hierarchically.

Tested with Qwen3-35B-A3B on llama.cpp + Brave Search API. Pipeline assumes a local OpenAI-compatible server.

Used with the Brave search engine by default, but Brave is a paid option. DuckDuckGo Search (`ddgs`) is also supported, although `SEARCH_RESULT_COUNT` is limited with ddgs.

Page fetching with [Crawl4AI](https://github.com/unclecode/crawl4ai) - an open-source LLM-friendly web crawler that handles Playwright orchestration, anti-bot evasion, and clean markdown extraction out of the box. The pipeline wraps it in `fetch.py` with a small concurrency-limited cache.

## What it does

```
prompt.txt (your research brief)
        |
        v
[0] topic name -> run dir
[1] turn brief into "prompt" (LLM generates search queries)
[2] Brave/DDGS fetches results for each query
[4] Pass 1: LLM scans every unique URL: "does this page offer ANY value to the research?"
[5] Pass 2: LLM deep-extracts every prompt-relevant fact from the has_value pages
[6] LLM synthesizes pass-2 outputs in chunks (parallel)
[7] writes report.md (synthesis + source list)
[8] hierarchical pairwise merge of chunks -> report_final.md (THE result)
```

(Stage 3 was deleted in an early refactor - the numbering is intentional.)

## Requirements

- Python 3.10+
- A local OpenAI-compatible LLM server. Tested with `llama-server` from llama.cpp running Qwen3-35B-A3B.
- Brave Search API key (set `BRAVE_API_KEY` in `.env`) - OR install `ddgs[api]` and set `SEARCH_ENGINE = "ddgs"` in `config.py` (the sidecar autostarts; no separate terminal needed).
- Playwright Chromium (`crawl4ai` installs it).

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
echo BRAVE_API_KEY=... > .env
```

### DDGS (free alternative to Brave)

```bash
pip install ddgs[api]
```

Set `SEARCH_ENGINE = "ddgs"` in `config.py`. The `ddgs api` sidecar autostarts on first import of `search_ddg` (blocks until `http://127.0.0.1:8000/docs` responds) and is reaped at process exit. No separate terminal. If a sidecar is already running on port 8000, it gets reused as-is.

Process management is cross-platform:
- **Windows**: child is assigned to a Job Object with `KILL_ON_JOB_CLOSE`, so the sidecar dies even on Task Manager kills, segfaults, OOM, or `os._exit()`.
- **Linux**: child gets `prctl(PR_SET_PDEATHSIG, SIGTERM)`, so the kernel reaps it on any parent death (including `SIGKILL`).
- **macOS**: graceful teardown via `atexit` + `SIGTERM`. Hard parent crashes can leave a stale sidecar, but the next run reuses any sidecar already answering at `/docs`, so it self-heals.

Start llama-server (example, tune to your hardware):

```bash
llama-server.exe \
  -m <path-to>/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  -ngl 99 -c 262144 --jinja --parallel 2 \
  --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.5 \
  --host 0.0.0.0 --port 8080
```

## Thinking

The Qwen3 thinking-mode toggles in `research.py` (`PASS1_THINKING`, `PASS2_THINKING`, `STAGE8_THINKING`) materially affect output quality. Empirical recommendations:

- **Pass 1 (stage 4): thinking ON.** Better has_value verdicts when the model can reason through page-vs-brief relevance.
- **Pass 2 (stage 5): thinking OFF.** Pass 2 is mostly a mechanical extraction task with high enough context already in the prompt. Enabling thinking can dwarf the actual pass-2 output length, which is the bread and butter of the research.
- **Stage 8 pairwise merge: thinking OFF.** Way better results - mechanical merge task, and way longer outputs with Qwen3.6-35B-A3B specifically.

## Usage

Set locales in `config.py`, write your brief into `prompt.txt`, then:

```bash
python research.py
```

Creates `logs/{topic}-{timestamp}/`.

### Resume

Two resume modes that compose:

| invocation | behavior |
|---|---|
| `--continue` alone | auto-detect from checkpoint + md files, run only what's missing. Crash recovery. |
| `--resume-stage N` alone | **HARD** reset: delete stage-N-or-later artifacts, redo from N. |
| `--resume-stage N --continue` | force re-entry at stage N, but keep per-URL md files. Monolithic stage-N output is discarded so it redoes fresh. Mostly used for re-entering pass 1 or pass 2 (stages 4 and 5). |

Examples:

```bash
# crashed mid-stage-5, want to pick up where it left off
python research.py --run-dir logs/<run> --continue

# stage 6 produced garbage, want to redo it (keeping pass-2 work)
python research.py --run-dir logs/<run> --resume-stage 6 --continue

# nuke and redo just the final merge (stage 8)
python research.py --run-dir logs/<run> --resume-stage 8
```

## Configuration

### `config.py`

```python
SEARCH_ENGINE       = "brave"                # or "ddgs"
SEARCH_RESULT_COUNT = 30                     # per-query result count (ddgs has hard cap 10-20)
DDGS_BACKEND        = "google,duckduckgo"    # ddgs only; comma-separated, in order. e.g. "auto", "duckduckgo", "google,duckduckgo,bing"
AUTO_CRASH_ON_FAILED_SEARCH = True           # see "Search failure handling" below
LOCALE              = "fi-fi"                # DDGS region
LANGUAGE            = "Finnish"              # used in stage-1 query template
COUNTRY             = "FI"                   # Brave country (ISO 3166-1 alpha-2)
SEARCH_LANG         = "fi"                   # Brave search lang (ISO 639-1)
UI_LANG             = "fi-FI"                # Brave UI lang (BCP-47)
```

### `research.py` knobs (top of file)

| name | default | meaning |
|---|---|---|
| `URLS_PER_QUERY` | 30 | search results fetched per stage-1 query |
| `FETCH_CONCURRENCY` | 4 | parallel page fetches (Playwright is heavy) |
| `LLM_CONCURRENCY` | 2 | parallel LLM calls - match `llama-server --parallel` |
| `STAGE_6_CHUNK_SIZE` | 30 | pass-2 pages per stage-6 synthesis call. Higher = more pass-2 website results given to the LLM at once. 10 is low-ish, 100 is too high. |
| `PASS1_THINKING` | True | Qwen3 thinking mode for pass-1 scans |
| `PASS2_THINKING` | False | Qwen3 thinking mode for pass-2 deep extracts |
| `STAGE8_THINKING` | False | Qwen3 thinking mode for stage-8 merges |

### `prompt.txt`

Plain text. Contains the research brief - section structure, what to find, what to skip. The pipeline does NOT strip or alter it.

### Search failure handling

Non-200 responses (429 rate limit, 401-403 auth, 5xx, DDGS's 202 throttle) abort the run by default and print a one-line `--resume-stage 2 --continue` command. The semantic cache holds every query that already returned, so the resume only re-runs the failed ones. Set `AUTO_CRASH_ON_FAILED_SEARCH = False` to skip failures instead (partial coverage; degrades the report).

## Output structure

```
logs/{topic}-{timestamp}/
├── run.log                         # tee'd stdout/stderr
├── intermediates.json              # rolling checkpoint, written every stage
├── pages/{url-slug}.md             # raw fetched markdown per URL
├── stages/
│   ├── 01-queries.md               # stage 1: queries + reasoning
│   ├── 02-search_results.md        # stage 2: results per query
│   ├── 04-pass1/{url-slug}.md      # stage 4: per-URL has_value verdict + summary
│   ├── 05-pass2/{url-slug}.md      # stage 5: per-URL deep extract
│   ├── 06-synthesis.md             # stage 6: stitched chunks
│   ├── 06-synthesis/chunk-NN.md    # stage 6: per-chunk synthesis
│   └── 08-finalize/r{NN}p{NN}.md   # stage 8: per-merge cache
├── report.md                       # stage 7 output (synthesis + sources)
└── report_final.md                 # stage 8 output (the main result)
```

## Architecture

- **Independent LLM calls** - no shared chat history across stages. Every call gets a self-contained system prompt + user message.
- **Brief-global extraction** - pass-1 and pass-2 evaluate every page against the *entire* brief, not against the specific query that surfaced it. URL-deduped: a page that appears in multiple queries is processed once.
- **Chunked synthesis** - stage 6 splits pass-2 outputs into chunks of `STAGE_6_CHUNK_SIZE` so each call fits in a single llama-server slot. Each chunk is independent.
- **Hierarchical merge (stage 8)** - pairs of stage-6 chunks merge into super-chunks; super-chunks merge again; repeat until one report remains. Odd-out items carry through unchanged to the next round.
- **Per-stage checkpoints** - `intermediates.json` is rewritten atomically after every stage with the full pipeline state. Resume after a crash by parsing this + the per-stage md files.
- **Fetch deduplication** - URLs are fetched once per run (in-memory cache + disk cache under `pages/`). Resume preloads the disk cache, so stage 5 onwards don't re-hit the network.
- **DDGS sidecar autostart** - importing `search_ddg` while `SEARCH_ENGINE = "ddgs"` blocks until the sidecar is responding, then registers an `atexit` reaper. OS-level hardening (Windows Job Object, Linux `PR_SET_PDEATHSIG`) ensures the sidecar dies with the parent even on hard crashes.
- **Fail-loud search** - non-200 responses from Brave or DDGS abort by default (`AUTO_CRASH_ON_FAILED_SEARCH = True`). The pipeline prints a one-line resume command; the cache holds every query that succeeded, so only the failed ones re-run.

## Module map

| file | role |
|---|---|
| `research.py` | the pipeline (stages 0-8, resume logic, checkpoint plumbing) |
| `config.py` | locale + search-engine selection |
| `prompt.txt` | the research brief (input) |
| `search.py` | engine dispatcher with semantic cache, exact-match short-circuit, picker LLM |
| `search_brave.py` | Brave API client with offset auto-pagination |
| `search_ddg.py` | DDGS metasearch sidecar client |
| `fetch.py` | crawl4ai wrapper with anti-bot detection |
| `llm.py` | `ChatLlamaServer` langchain wrapper (json-schema strict mode) |

## Caveats

- Pass-1 has a hard char ceiling (`FETCH_CHAR_LIMIT = 50000` in `fetch.py`). Long pages get truncated.
- Stage 6 chunk size needs to fit in your llama-server's per-slot context budget (`-c <total> / --parallel <N>`). At `STAGE_6_CHUNK_SIZE=30` and ~2.5k-token pass-2 outputs, expect ~88k input tokens per chunk - fits comfortably in a 131k slot.
- Stage 8 with thinking ON tends to over-compress (the model spends output budget on the thinking trace). Default is OFF.
- Qwen3-MoE's hybrid SSM/recurrent layers invalidate llama.cpp's prompt cache across diverging prompts (PR #13194), so each chunk pays a full prefill. This is unavoidable on this model class.
- macOS has no `PR_SET_PDEATHSIG` equivalent and no Job Object - if the parent dies hard (`SIGKILL`, segfault, force-quit), the DDGS sidecar can outlive it. Mitigation: the next run reuses any sidecar already answering at `/docs:8000`, so no manual cleanup needed in practice.
