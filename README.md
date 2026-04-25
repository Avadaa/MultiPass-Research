# LLM Research Pipeline

Local-LLM-driven research pipeline. Takes a plain-text research brief, produces a comprehensive markdown report by generating search queries, fetching pages, extracting brief-relevant content, and synthesizing findings hierarchically.

Tested with Qwen3-35B-A3B on llama.cpp + Brave Search API. Pipeline assumes a local OpenAI-compatible server.

## What it does

```
prompt.txt (your research brief)
        |
        v
[0] topic name -> run dir
[1] LLM generates search queries from the brief
[2] Brave/DDGS fetches results for each query
[4] LLM scans every unique URL: "does this page offer ANY value to the brief?"
[5] LLM deep-extracts every brief-relevant fact from the has_value pages
[6] LLM synthesizes pass-2 outputs in chunks (parallel)
[7] writes report.md (synthesis + source list)
[8] hierarchical pairwise merge of chunks -> report_final.md (THE result)
```

(Stage 3 was deleted in an early refactor — the numbering is intentional.)

## Requirements

- Python 3.10+
- A local OpenAI-compatible LLM server. Tested with `llama-server` from llama.cpp running Qwen3-35B-A3B.
- Brave Search API key (set `BRAVE_API_KEY` in `.env`) — OR a local DDGS sidecar at `127.0.0.1:8000`.
- Playwright Chromium (`crawl4ai` installs it).

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
echo BRAVE_API_KEY=... > .env
```

Start llama-server (example, tune to your hardware):

```bash
llama-server.exe \
  -m <path-to>/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  -ngl 99 -c 262144 --jinja --parallel 2 \
  --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.5 \
  --host 0.0.0.0 --port 8080
```

Edit `prompt.txt` with your research brief and `config.py` with your locale.

## Usage

### Fresh run

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
| `--resume-stage N --continue` | force re-entry at stage N, but keep per-URL md files. Monolithic stage-N output is discarded so it redoes fresh. |

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
SEARCH_ENGINE       = "brave"     # or "ddgs"
SEARCH_RESULT_COUNT = 30          # default per-query result count
LOCALE              = "fi-fi"     # DDGS region
LANGUAGE            = "Finnish"   # used in stage-1 query template
COUNTRY             = "FI"        # Brave country (ISO 3166-1 alpha-2)
SEARCH_LANG         = "fi"        # Brave search lang (ISO 639-1)
UI_LANG             = "fi-FI"     # Brave UI lang (BCP-47)
```

### `research.py` knobs (top of file)

| name | default | meaning |
|---|---|---|
| `URLS_PER_QUERY` | 30 | search results fetched per stage-1 query |
| `FETCH_CONCURRENCY` | 4 | parallel page fetches (Playwright is heavy) |
| `LLM_CONCURRENCY` | 2 | parallel LLM calls — match `llama-server --parallel` |
| `STAGE_6_CHUNK_SIZE` | 30 | pass-2 pages per stage-6 synthesis call |
| `PASS1_THINKING` | False | Qwen3 thinking mode for pass-1 scans |
| `PASS2_THINKING` | False | Qwen3 thinking mode for pass-2 deep extracts |
| `STAGE8_THINKING` | False | Qwen3 thinking mode for stage-8 merges |

### `prompt.txt`

Plain text. Contains the research brief — section structure, what to find, what to skip. The pipeline does NOT strip or alter it.

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

- **Independent LLM calls** — no shared chat history across stages. Every call gets a self-contained system prompt + user message.
- **Brief-global extraction** — pass-1 and pass-2 evaluate every page against the *entire* brief, not against the specific query that surfaced it. URL-deduped: a page that appears in multiple queries is processed once.
- **Chunked synthesis** — stage 6 splits pass-2 outputs into chunks of `STAGE_6_CHUNK_SIZE` so each call fits in a single llama-server slot. Each chunk is independent.
- **Hierarchical merge (stage 8)** — pairs of stage-6 chunks merge into super-chunks; super-chunks merge again; repeat until one report remains. Odd-out items carry through unchanged to the next round.
- **Per-stage checkpoints** — `intermediates.json` is rewritten atomically after every stage with the full pipeline state. Resume after a crash by parsing this + the per-stage md files.
- **Fetch deduplication** — URLs are fetched once per run (in-memory cache + disk cache under `pages/`). Resume preloads the disk cache, so stage 5 onwards don't re-hit the network.

## Module map

| file | role |
|---|---|
| `research.py` | the pipeline (stages 0–8, resume logic, checkpoint plumbing) |
| `config.py` | locale + search-engine selection |
| `prompt.txt` | the research brief (input) |
| `search.py` | engine dispatcher with semantic cache, exact-match short-circuit, picker LLM |
| `search_brave.py` | Brave API client with offset auto-pagination |
| `search_ddg.py` | DDGS metasearch sidecar client |
| `fetch.py` | crawl4ai wrapper with anti-bot detection |
| `llm.py` | `ChatLlamaServer` langchain wrapper (json-schema strict mode) |

## Caveats

- Pass-1 has a hard char ceiling (`FETCH_CHAR_LIMIT = 50000` in `fetch.py`). Long pages get truncated.
- Stage 6 chunk size needs to fit in your llama-server's per-slot context budget (`-c <total> / --parallel <N>`). At `STAGE_6_CHUNK_SIZE=30` and ~2.5k-token pass-2 outputs, expect ~88k input tokens per chunk — fits comfortably in a 131k slot.
- Stage 8 with thinking ON tends to over-compress (the model spends output budget on the thinking trace). Default is OFF.
- Qwen3-MoE's hybrid SSM/recurrent layers invalidate llama.cpp's prompt cache across diverging prompts (PR #13194), so each chunk pays a full prefill. This is unavoidable on this model class.
