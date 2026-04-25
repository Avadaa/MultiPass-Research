"""research.py — independent-LLM-call research pipeline.

Pipeline (every LLM call is independent; no cross-stage context is accumulated):

  Stage 0: Topic namer (LLM, kebab-case slug for the log directory)
  Stage 1: Query generator (LLM, produces a list of search queries)
  Stage 2: Batch search via search.py (snapshot-before-batch semantics,
           with exact-match + semantic-cache short-circuits)
  Stage 4: Pass 1 — brief-global scan of each unique URL (fetch + LLM)
  Stage 5: Pass 2 — brief-global deep extract of has_value pages (LLM)
  Stage 6: Single blind synthesis across all pass-2 extracts (LLM)
  Stage 7: Write logs/{topic}-{ts}/report.md + intermediates.json

Rolling checkpoint: intermediates.json is written at the end of every stage
(1, 2, 4, 5, 6, 7). Resume a partial run with:
    python research.py --run-dir <path> --resume-stage N
which deletes artifacts for stage N and later, then re-runs from stage N.

Configuration knobs: URLS_PER_QUERY, FETCH_CONCURRENCY, LLM_CONCURRENCY.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
import threading
import time
import urllib.error
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

# crawl4ai banner chars + subprocess fd leaks on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore", category=ResourceWarning)

from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    AUTO_CRASH_ON_FAILED_SEARCH,
    AUTO_START_CHECKS,
    COUNTRY,
    LANGUAGE,
    LOCALE,
    SEARCH_ENGINE,
    SEARCH_LANG,
    UI_LANG,
)
from fetch import FETCH_CHAR_LIMIT, fetch_website
from llm import ChatLlamaServer
from search import search as cached_search


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

URLS_PER_QUERY = 30
FETCH_CONCURRENCY = 4        # Playwright subprocess is heavy
LLM_CONCURRENCY = 2          # matches llama-server --parallel 2
STAGE_6_CHUNK_SIZE = 30      # pass-2 extracts per stage-6 synthesis call
STAGE8_THINKING = False      # Qwen3 thinking mode for stage-8 pairwise merges
PASS1_THINKING = True        # Qwen3 thinking mode for pass-1 scans
PASS2_THINKING = False        # Qwen3 thinking mode for pass-2 deep extracts
PROMPT_FILE = Path(__file__).parent / "prompt.txt"
LOGS_DIR = Path(__file__).parent / "logs"


# ---------------------------------------------------------------------------
# Stdout/stderr tee into the run directory's run.log
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data: str) -> int:
        try:
            n = self._stream.write(data)
        except Exception:
            n = len(data)
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass
        return n
    def flush(self) -> None:
        for s in (self._stream, self._log):
            try: s.flush()
            except Exception: pass
    def __getattr__(self, name):
        return getattr(self._stream, name)


_RUN_DIR: Path | None = None


def install_logger(run_dir: Path, mode: str = "w") -> Path:
    global _RUN_DIR
    _RUN_DIR = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.log"
    f = open(path, mode, encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return path


def _write_artifact(subpath: str, content: str) -> None:
    """Write `content` to {run_dir}/{subpath}, creating parent dirs."""
    if _RUN_DIR is None:
        return
    p = _RUN_DIR / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _slug(s: str, maxlen: int = 80) -> str:
    """Filename-safe slug. Lowercase, hyphens only, truncated."""
    out = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return out[:maxlen] or "x"


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

# Maps a stage number to the artifacts (files/dirs) it owns relative to run_dir.
# Stage 0 owns the log dir name but not a specific artifact. Stage 3 is gone.
_STAGE_ARTIFACTS: dict[int, list[str]] = {
    1: ["stages/01-queries.md"],
    2: ["stages/02-search_results.md"],
    4: ["stages/04-pass1"],
    5: ["stages/05-pass2"],
    6: ["stages/06-synthesis.md"],
    7: ["report.md"],
    8: ["report_final.md", "stages/08-finalize"],
}

VALID_RESUME_STAGES = (1, 2, 4, 5, 6, 7, 8)


def _build_intermediates(
    topic: str,
    locale: str,
    language: str,
    country: str,
    search_lang: str,
    ui_lang: str,
    brief: str,
    stage1_reasoning: str | None,
    queries: list[str] | None,
    query_runs: list["QueryRun"] | None,
    pages: dict[str, "Page"] | None,
    synthesis: str | None,
    final_report: str | None,
    last_completed_stage: int,
) -> dict:
    """Serialize the full pipeline state for rolling intermediates.json."""
    return {
        "topic": topic,
        "locale": locale,
        "language": language,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
        "brief": brief,
        "stage1_reasoning": stage1_reasoning,
        "queries": queries,
        "query_runs": (
            [
                {
                    "query": qr.query,
                    "results": [asdict(r) for r in qr.results],
                }
                for qr in query_runs
            ]
            if query_runs is not None else None
        ),
        "pages": (
            {
                url: {
                    "url": p.url,
                    "title": p.title,
                    "surfaced_by": p.surfaced_by,
                    "pass1": asdict(p.pass1) if p.pass1 else None,
                    "pass2": asdict(p.pass2) if p.pass2 else None,
                }
                for url, p in pages.items()
            }
            if pages is not None else None
        ),
        "synthesis": synthesis,
        "final_report": final_report,
        "last_completed_stage": last_completed_stage,
    }


def _write_checkpoint(run_dir: Path, cp: dict) -> None:
    """Atomically write the rolling intermediates.json checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "intermediates.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(cp, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_checkpoint(run_dir: Path) -> dict:
    """Read and validate intermediates.json from a prior run."""
    path = run_dir / "intermediates.json"
    if not path.exists():
        sys.exit(
            f"no checkpoint at {path} — cannot resume. Either the run did "
            f"not reach the end of stage 1, or --run-dir points at the wrong "
            f"directory."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"checkpoint at {path} is not valid JSON: {e}")
    required = (
        "topic", "locale", "language", "country", "search_lang",
        "ui_lang", "brief", "last_completed_stage",
    )
    missing = [k for k in required if k not in data]
    if missing:
        sys.exit(
            f"checkpoint at {path} is missing required field(s): "
            f"{', '.join(missing)}"
        )
    return data


def _cleanup_from_stage(run_dir: Path, stage: int) -> None:
    """Delete artifacts for `stage` and every later stage. Preserves
    intermediates.json, run.log, and stages/pages/ fetch cache."""
    for s in sorted(_STAGE_ARTIFACTS):
        if s < stage:
            continue
        for rel in _STAGE_ARTIFACTS[s]:
            p = run_dir / rel
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                print(f"[cleanup] removed dir  {p}")
            elif p.exists():
                p.unlink()
                print(f"[cleanup] removed file {p}")


# ---------------------------------------------------------------------------
# Markdown artifact parsing — scavenger helpers for --continue
# ---------------------------------------------------------------------------

def _split_md_sections(text: str) -> dict[str, str]:
    """Split a markdown doc by level-2 (##) headers. Fence-aware: does not
    treat ## inside ``` code blocks as headers. Returns {header: body}."""
    out: dict[str, str] = {}
    current_key: str | None = None
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and line.startswith("## "):
            if current_key is not None:
                out[current_key] = "\n".join(current).strip()
            current_key = line[3:].strip()
            current = []
        else:
            current.append(line)
    if current_key is not None:
        out[current_key] = "\n".join(current).strip()
    return out


def _recover_queries_from_md(run_dir: Path) -> tuple[str | None, list[str] | None]:
    f = run_dir / "stages" / "01-queries.md"
    if not f.exists():
        return None, None
    sections = _split_md_sections(f.read_text(encoding="utf-8", errors="replace"))
    reasoning = sections.get("Reasoning") or None
    qbody = sections.get("Queries") or ""
    queries = re.findall(r"^\d+\.\s+(.+?)$", qbody, re.MULTILINE)
    return reasoning, (queries or None)


def _parse_stage2_results(body: str) -> list["SearchResult"]:
    """Parse '1. [title](url)\\n   description' entries into SearchResults."""
    results: list[SearchResult] = []
    entry_re = re.compile(r"^(\d+)\.\s+\[(.+?)\]\((.+?)\)\s*$")
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        m = entry_re.match(lines[i])
        if not m:
            i += 1
            continue
        title, url = m.group(2).strip(), m.group(3).strip()
        desc = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("   "):
            desc = lines[i + 1].strip()
            i += 2
        else:
            i += 1
        results.append(SearchResult(title=title, url=url, description=desc))
    return results


def _recover_query_runs_from_md(run_dir: Path) -> list["QueryRun"] | None:
    f = run_dir / "stages" / "02-search_results.md"
    if not f.exists():
        return None
    sections = _split_md_sections(f.read_text(encoding="utf-8", errors="replace"))
    per_num: list[tuple[int, str, str]] = []
    for key, body in sections.items():
        m = re.match(r"Query (\d+):\s*(.+)$", key)
        if m:
            per_num.append((int(m.group(1)), m.group(2).strip(), body))
    per_num.sort()
    runs = [
        QueryRun(query=query, results=_parse_stage2_results(body))
        for _, query, body in per_num
    ]
    return runs or None


def _recover_pass1_from_md(run_dir: Path, page: "Page") -> "Pass1 | None":
    f = run_dir / "stages" / "04-pass1" / f"{_slug(page.url)}.md"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    sections = _split_md_sections(text)
    resp = sections.get("Pass-1 response", "")
    m = re.search(r"- has_value:\s*\*\*(True|False)\*\*", resp)
    if not m:
        return None
    has_value = (m.group(1) == "True")
    sm = re.search(r"### Summary\s*\n\s*\n(.+?)\Z", resp, re.DOTALL)
    summary = sm.group(1).strip() if sm else ""
    return Pass1(url=page.url, has_value=has_value, summary=summary)


def _recover_pass2_from_md(run_dir: Path, page: "Page") -> "Pass2 | None":
    f = run_dir / "stages" / "05-pass2" / f"{_slug(page.url)}.md"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    sections = _split_md_sections(text)
    deep: str | None = None
    for key, body in sections.items():
        if key.startswith("Pass-2 deep summary"):
            deep = body.strip()
            break
    if not deep or deep == "(empty)" or deep.startswith("(no pass-2 result)"):
        return None
    return Pass2(url=page.url, deep_summary=deep)


def _recover_final_report_from_md(run_dir: Path) -> str | None:
    f = run_dir / "report_final.md"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _recover_synth_from_md(run_dir: Path) -> str | None:
    f = run_dir / "stages" / "06-synthesis.md"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"^_(?:synthesized|chunked) from [^_]+_\s*\n\s*\n(.+)",
        text, re.DOTALL | re.MULTILINE,
    )
    if m:
        body = m.group(1).rstrip()
        return body or None
    # Fallback: strip the first H1 line, return remainder
    parts = text.split("\n", 1)
    body = parts[1].strip() if len(parts) == 2 else text.strip()
    return body or None


def _preload_page_cache(run_dir: Path, pages: dict[str, "Page"]) -> int:
    """Populate _page_cache from run_dir/pages/*.md written by stage 4's
    fetcher. Lets stage-5-and-later resumes skip the re-fetch cost."""
    pages_dir = run_dir / "pages"
    if not pages_dir.exists():
        return 0
    loaded = 0
    for url in pages.keys():
        f = pages_dir / f"{_slug(url)}.md"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        # Format written by _get_page is: "# {url}\n\n{content-or-fetch-failed}"
        parts = text.split("\n", 2)
        content = parts[2] if len(parts) >= 3 else ""
        if content.strip() == "(fetch failed)":
            content = ""
        with _page_cache_lock:
            _page_cache[url] = content
        loaded += 1
    return loaded


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    description: str


@dataclass
class Pass1:
    url: str
    has_value: bool
    summary: str


@dataclass
class Pass2:
    url: str
    deep_summary: str


@dataclass
class Page:
    url: str
    title: str
    surfaced_by: list[str] = field(default_factory=list)
    pass1: Pass1 | None = None
    pass2: Pass2 | None = None


@dataclass
class QueryRun:
    query: str
    results: list[SearchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM builder (stage-specific config, uniform construction)
# ---------------------------------------------------------------------------

def _build_llm(
    temperature: float,
    max_tokens: int,
    thinking: bool,
    schema: dict,
    schema_name: str,
):
    base = ChatLlamaServer(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        model="local",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return base.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


# ---------------------------------------------------------------------------
# Stage 0 — topic namer
# ---------------------------------------------------------------------------

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}

TOPIC_SYSTEM = (
    "Summarize the user's research prompt in a very short kebab-case slug "
    "(e.g. 'cv-conventions-finland', 'renewable-energy-incentives-eu'). "
    "Max 60 characters, lowercase, hyphens only. Respond as JSON: "
    '{"topic": "<slug>"}'
)


def stage0_topic(prompt: str) -> str:
    llm = _build_llm(0.3, 512, False, TOPIC_SCHEMA, "topic")
    resp = llm.invoke([
        SystemMessage(content=TOPIC_SYSTEM),
        HumanMessage(content=prompt),
    ])
    topic = json.loads(resp.content)["topic"]
    topic = re.sub(r"[^a-z0-9-]+", "-", topic.lower().strip()).strip("-")
    return topic or "untitled-research"


# ---------------------------------------------------------------------------
# Stage 1 — query generator
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["reasoning", "queries"],
    "additionalProperties": False,
}

QUERY_SYSTEM_TEMPLATE = (
    "You are a search-query planner. Given a user task, produce the set of "
    "web-search queries that, together, would let a researcher answer the "
    "task.\n\n"
    "Budget and coverage:\n"
    "  - Aim for UNDER 20 queries total. A good plan with deliberate gaps "
    "beats an exhaustive plan with weak queries. Coverage completeness does "
    "NOT justify padding.\n"
    "  - If you cannot frame a good query for some part of the user's task, "
    "skip that part and note the skip in your reasoning. The rules below can "
    "force a skip — that is intended.\n\n"
    "Core rules:\n"
    "  - Each query covers a distinct information need not covered by the "
    "others. If the task is already a single concrete question, one query is "
    "correct; do not pad.\n"
    "  - Each query targets ONE entity or ONE information need. Do not pack "
    "multiple entities into a single query — this includes explicit "
    "comparisons ('X vs Y'), compounds joined by 'and'/'or', comma-separated "
    "lists, and listing two or more concrete named things (sectors, "
    "platforms, regions, languages) in one query. Split into one query per "
    "entity. If you genuinely want a comparison across N entities, generate "
    "N single-entity queries and let synthesis compare the findings later.\n"
    "  - Write queries in keyword form, not natural-language sentences. Drop "
    "question words, auxiliaries, and other low-signal filler. Aim for "
    "roughly 3–6 content terms per query.\n"
    "  - Do not add a year or other time anchor.\n"
    "  - If two queries would return largely overlapping pages, keep one.\n\n"
    "Language:\n"
    "  - Write EVERY word in {language} — including country names, generic "
    "filler words, and abstract nouns. Do not leave English words (e.g. "
    "'convention', 'Finland', 'recruitment') embedded in an otherwise "
    "local-language query; translate them to their local equivalents. "
    "Off-language queries collapse into globally-ranked pages that miss "
    "local practice.\n"
    "  - Use established {language} terms for concepts that have them: "
    "native word for the document type, proper name of a regulator or "
    "statute, a trade body, a major domestic platform. If the user's task "
    "names such a term, reuse that exact wording rather than substituting a "
    "coined synonym.\n"
    "  - Do not coin or guess proper names of laws, regulators, or "
    "institutions. If you are not confident a specific name exists in the "
    "local market, describe the thing functionally in the local language "
    "instead (e.g. generic phrasing for 'discrimination law in recruitment' "
    "or 'data protection authority guidance').\n\n"
    "Overrides — these prohibitions take PRIORITY over task coverage. If "
    "honoring one means skipping part of the user's task, skip it:\n"
    "  - **Foreign-market jargon.** If the user's task gives country-"
    "specific practice names from OTHER markets as examples of a broader "
    "category, do NOT include the foreign proper noun in your query, and do "
    "NOT calque, transliterate, or loan-translate it into the local "
    "language. Ask in local-language keywords whether the local market has "
    "an equivalent concept at all — but only if the concept is plausible "
    "locally. If it almost certainly doesn't exist in this market, skip the "
    "category.\n"
    "  - **Meta-queries and synthesis-output questions.** Drop queries that "
    "ask for 'comparisons between generic international advice and local "
    "practice', for causal effects of macroeconomic or demographic trends "
    "on local practice, or for 'recent shifts / trends' in the abstract. No "
    "one publishes direct articles answering these; search returns broad "
    "overviews that answer nothing. Such requests are OUTPUTS of synthesis "
    "across findings, not search queries — produce no query for them, even "
    "if the user's task explicitly asks for that section. If recency "
    "matters, anchor queries on a CONCRETE changed thing (a named statute, "
    "a named platform, a named reform) instead.\n"
    "  - **Abstract queries.** Every remaining query must pass this test: "
    "can you name a CONCRETE kind of page that would answer it — a named "
    "government guide, a trade publication, a career blog post, a specific "
    "statute page? If not, sharpen the query or drop it.\n\n"
    'Respond as JSON: {{"reasoning": "<one-paragraph plan, including which '
    'parts of the user\'s task you deliberately skipped and why>", '
    '"queries": ["...", "..."]}}'
)


def stage1_queries(brief: str, language: str) -> tuple[str, list[str]]:
    system = QUERY_SYSTEM_TEMPLATE.format(language=language)
    llm = _build_llm(1.0, 131072, True, QUERY_SCHEMA, "query_plan")
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"User task:\n{brief}"),
    ])
    parsed = json.loads(resp.content)
    return parsed["reasoning"], parsed["queries"]


# ---------------------------------------------------------------------------
# Stage 2 — batch search (cached)
# ---------------------------------------------------------------------------

def stage2_search(
    queries: list[str],
    locale: str,
    country: str,
    search_lang: str,
    ui_lang: str,
) -> list[list[SearchResult]]:
    per_query = cached_search(
        queries,
        count=URLS_PER_QUERY,
        region=locale,
        country=country,
        search_lang=search_lang,
        ui_lang=ui_lang,
    )
    out = []
    for rs in per_query:
        out.append([
            SearchResult(
                title=(r.get("title") or ""),
                url=(r.get("url") or ""),
                description=(r.get("description") or ""),
            )
            for r in rs
            if r.get("url")
        ])
    return out


# ---------------------------------------------------------------------------
# Page cache (shared by stages 4 and 5)
# ---------------------------------------------------------------------------

_page_cache: dict[str, str] = {}
_page_cache_lock = threading.Lock()
_fetch_locks: dict[str, threading.Lock] = {}
_fetch_locks_guard = threading.Lock()


def _get_page(url: str) -> str:
    """Fetch once per URL; subsequent calls hit the cache. Empty string on failure."""
    with _page_cache_lock:
        if url in _page_cache:
            return _page_cache[url]

    with _fetch_locks_guard:
        lock = _fetch_locks.setdefault(url, threading.Lock())

    with lock:
        with _page_cache_lock:
            if url in _page_cache:
                return _page_cache[url]
        try:
            print(f"[fetch] {url}")
            content = fetch_website(url)
        except Exception as e:
            print(f"[fetch failed] {url}: {e}")
            content = ""
        with _page_cache_lock:
            _page_cache[url] = content
        # Persist the raw fetched markdown once per URL (deduped).
        _write_artifact(
            f"pages/{_slug(url)}.md",
            f"# {url}\n\n{content if content else '(fetch failed)'}",
        )
        return content


# ---------------------------------------------------------------------------
# Stage 4 — pass 1 (scan) — brief-global, one call per unique URL
# ---------------------------------------------------------------------------

PASS1_SCHEMA = {
    "type": "object",
    "properties": {
        "has_value": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["has_value", "summary"],
    "additionalProperties": False,
}

PASS1_SYSTEM_TEMPLATE = (
    "You are scanning ONE web page against a research brief. Decide whether "
    "the page offers ANY value to the brief — any single category, any single "
    "data point, any fact, quote, date, number, named source, or useful "
    "reference that helps the researcher. Be inclusive: tangentially useful "
    "content still counts. A page only needs to answer ONE part of the brief "
    "to qualify as has_value=true.\n\n"
    "Reject (has_value=false) ONLY if the page is:\n"
    "  - a cookie banner, 404, paywall, login wall, or captcha\n"
    "  - navigation / search-result / category index with no substantive content\n"
    "  - entirely off-topic (nothing in it relates to any part of the brief)\n"
    "  - empty or effectively empty\n\n"
    "If has_value=true, write a medium-length summary of EVERYTHING on the "
    "page that relates to the brief — preserve specifics (names, dates, URLs, "
    "exact numbers, direct quotes). Don't filter by which part of the brief "
    "the content addresses; a downstream step will organize by topic.\n"
    "If has_value=false, briefly state why.\n\n"
    "--- BEGIN RESEARCH BRIEF ---\n"
    "{brief}\n"
    "--- END RESEARCH BRIEF ---\n\n"
    "--- BEGIN FETCHED PAGE CONTENT ---\n"
    "{content}\n"
    "--- END FETCHED PAGE CONTENT ---\n\n"
    'Respond as JSON: {{"has_value": <bool>, '
    '"summary": "<what the page offers, or why it was rejected>"}}'
)


def _scan_one(brief: str, url: str) -> Pass1:
    content = _get_page(url)
    if not content:
        return Pass1(url=url, has_value=False, summary="rejected (fetch failed)")
    system = PASS1_SYSTEM_TEMPLATE.format(
        brief=brief,
        content=content[:FETCH_CHAR_LIMIT],
    )
    llm = _build_llm(0.4, 65536, PASS1_THINKING, PASS1_SCHEMA, "pass1")
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Scan this page and decide: does it offer "
                                 f"any value to the brief above?"),
        ])
        parsed = json.loads(resp.content)
        return Pass1(
            url=url,
            has_value=bool(parsed.get("has_value", False)),
            summary=(parsed.get("summary") or ""),
        )
    except Exception as e:
        return Pass1(url=url, has_value=False, summary=f"rejected (llm error: {e!r})")


def _format_pass1_doc(page: Page, content: str) -> str:
    p1 = page.pass1
    parts = [
        f"# Pass-1 scan: {page.url}",
        "",
        f"- url: {page.url}",
        f"- title: {page.title}",
        f"- surfaced by {len(page.surfaced_by)} quer"
        f"{'y' if len(page.surfaced_by) == 1 else 'ies'}:",
        *(f"  - {q}" for q in page.surfaced_by),
        "",
        "## Pass-1 response",
        "",
        f"- has_value: **{p1.has_value if p1 else 'n/a'}**",
        "",
        "### Summary",
        "",
        (p1.summary if p1 else "(no pass-1 result)") or "(empty)",
        "",
        f"## Fetched content (first {FETCH_CHAR_LIMIT:,} chars)",
        "",
        "```",
        (content[:FETCH_CHAR_LIMIT] if content else "(fetch failed / empty)"),
        "```",
    ]
    return "\n".join(parts)


def stage4_pass1(pages: dict[str, Page], brief: str) -> None:
    """Mutates pages in place: fills page.pass1 for each URL.
    URLs whose page.pass1 is already populated (resume case) are skipped."""
    urls = [url for url, p in pages.items() if p.pass1 is None]
    skipped = len(pages) - len(urls)
    if skipped:
        print(f"[stage 4] skipping {skipped} URLs with cached pass-1 from "
              f"an earlier run")
    print(f"[stage 4] pass-1 on {len(urls)} unique URLs "
          f"(fetch_conc={FETCH_CONCURRENCY}, llm throttled server-side)")
    if not urls:
        return

    with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as pool:
        futs = {pool.submit(_scan_one, brief, url): url for url in urls}
        for fut in futs:
            url = futs[fut]
            try:
                p1 = fut.result()
            except Exception as e:
                p1 = Pass1(url=url, has_value=False,
                           summary=f"rejected (unexpected: {e!r})")
            pages[url].pass1 = p1
            verdict = "YES" if p1.has_value else "no "
            print(f"  [pass-1 {verdict}] {url}")
            content = _page_cache.get(url, "")
            _write_artifact(
                f"stages/04-pass1/{_slug(url)}.md",
                _format_pass1_doc(pages[url], content),
            )


# ---------------------------------------------------------------------------
# Stage 5 — pass 2 (elaborate on promising) — brief-global, per unique URL
# ---------------------------------------------------------------------------

PASS2_SCHEMA = {
    "type": "object",
    "properties": {"deep_summary": {"type": "string"}},
    "required": ["deep_summary"],
    "additionalProperties": False,
}

PASS2_SYSTEM_TEMPLATE = (
    "A previous pass flagged this page as relevant to the research brief "
    "below. Your job now is to EXTRACT DETAIL — pretend you are compiling "
    "source material for a research report.\n\n"
    "Pull out EVERY fact, quote, number, date, named source, URL, and "
    "concrete example on the page that relates to ANY part of the brief. "
    "Do not filter by topic — if it connects to any brief category, include "
    "it. Preserve direct quotes verbatim. Cite publication or update dates "
    "where the page shows them. Do not summarize at a high level; be "
    "exhaustive.\n\n"
    "Pass-1 summary (for context on what the previous pass spotted):\n"
    "{pass1_summary}\n\n"
    "--- BEGIN RESEARCH BRIEF ---\n"
    "{brief}\n"
    "--- END RESEARCH BRIEF ---\n\n"
    "--- BEGIN CACHED PAGE CONTENT ---\n"
    "{content}\n"
    "--- END CACHED PAGE CONTENT ---\n\n"
    'Respond as JSON: {{"deep_summary": "<long-form, detailed extraction>"}}'
)


def _elaborate_one(brief: str, page: Page) -> Pass2:
    # _get_page first checks the in-memory cache (populated by stage 4's
    # fetcher, or by _preload_page_cache on resume), and only re-fetches if
    # that's empty. Keeps resume-from-stage-5 cheap when pages/*.md exist.
    content = _get_page(page.url)
    if not content:
        return Pass2(url=page.url, deep_summary="(no cached content available)")
    p1_summary = page.pass1.summary if page.pass1 else ""
    system = PASS2_SYSTEM_TEMPLATE.format(
        pass1_summary=p1_summary,
        brief=brief,
        content=content[:FETCH_CHAR_LIMIT],
    )
    llm = _build_llm(0.6, 131072, PASS2_THINKING, PASS2_SCHEMA, "pass2")
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Extract everything on this page that relates "
                                 f"to any part of the brief. Source URL: {page.url}"),
        ])
        return Pass2(
            url=page.url,
            deep_summary=json.loads(resp.content)["deep_summary"],
        )
    except Exception as e:
        return Pass2(url=page.url, deep_summary=f"(LLM error: {e!r})")


def _format_pass2_doc(page: Page) -> str:
    content = _page_cache.get(page.url, "")
    p1 = page.pass1
    p2 = page.pass2
    parts = [
        f"# Pass-2 deep extract: {page.url}",
        "",
        f"- url: {page.url}",
        f"- title: {page.title}",
        f"- surfaced by {len(page.surfaced_by)} quer"
        f"{'y' if len(page.surfaced_by) == 1 else 'ies'}:",
        *(f"  - {q}" for q in page.surfaced_by),
        "",
        "## Pass-1 summary (input to pass 2)",
        "",
        (p1.summary if p1 else "(no pass-1 result)") or "(empty)",
        "",
        "## Pass-2 deep summary",
        "",
        (p2.deep_summary if p2 else "(no pass-2 result)") or "(empty)",
        "",
        f"## Cached page content (first {FETCH_CHAR_LIMIT:,} chars)",
        "",
        "```",
        (content[:FETCH_CHAR_LIMIT] if content else "(no cached content)"),
        "```",
    ]
    return "\n".join(parts)


def stage5_pass2(pages: dict[str, Page], brief: str) -> None:
    """Mutates pages in place: fills page.pass2 for URLs that passed pass 1.
    Pages whose page.pass2 is already populated (resume case) are skipped."""
    promising = [p for p in pages.values() if p.pass1 and p.pass1.has_value]
    todo = [p for p in promising if p.pass2 is None]
    skipped = len(promising) - len(todo)
    if skipped:
        print(f"[stage 5] skipping {skipped} pages with cached pass-2 from "
              f"an earlier run")
    print(f"[stage 5] pass-2 on {len(todo)} promising pages")
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futs = {pool.submit(_elaborate_one, brief, p): p for p in todo}
        for fut in futs:
            page = futs[fut]
            try:
                p2 = fut.result()
            except Exception as e:
                p2 = Pass2(url=page.url, deep_summary=f"(unexpected: {e!r})")
            page.pass2 = p2
            print(f"  [pass-2 done] {page.url}")
            _write_artifact(
                f"stages/05-pass2/{_slug(page.url)}.md",
                _format_pass2_doc(page),
            )


# ---------------------------------------------------------------------------
# Stage 6 — single blind synthesis across all deep extracts
# ---------------------------------------------------------------------------

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

SYNTHESIS_SYSTEM_TEMPLATE = (
    "You are synthesizing one CHUNK of research findings (pass-1 summaries "
    "and pass-2 deep extracts from multiple web pages) against the research "
    "brief below. Produce a comprehensive markdown report covering ONLY the "
    "pages in this chunk.\n\n"
    "Rules:\n"
    "  - Preserve specifics: direct quotes verbatim, exact numbers, named "
    "sources, dates, URL citations. Do not round off, paraphrase quotes, or "
    "drop attribution.\n"
    "  - Every factual claim MUST be attached to its source URL as a markdown "
    "link. Never cite by domain alone — always use the full page URL.\n"
    "  - Organize by topic. Let the structure emerge from what this chunk "
    "actually covers; do not force a predetermined outline.\n"
    "  - Flag contradictions between sources in this chunk explicitly.\n"
    "  - If the evidence is thin on a topic, say so plainly rather than "
    "padding or inferring.\n"
    "  - Do NOT speculate about pages outside this chunk. Other chunks "
    "will be merged later.\n"
    "  - Thoroughness beats brevity.\n\n"
    "--- BEGIN RESEARCH BRIEF ---\n"
    "{brief}\n"
    "--- END RESEARCH BRIEF ---\n\n"
    'Respond as JSON: {{"summary": "<long-form markdown report for this chunk>"}}'
)


def _synth_one_chunk(
    chunk_idx: int,
    n_chunks: int,
    brief: str,
    pages_in_chunk: list[Page],
) -> str:
    system = SYNTHESIS_SYSTEM_TEMPLATE.format(brief=brief)
    parts = []
    for p in pages_in_chunk:
        p1_text = (p.pass1.summary if p.pass1 else "").strip() or "(no pass-1)"
        p2_text = (p.pass2.deep_summary if p.pass2 else "").strip() or "(no pass-2)"
        parts.append(
            f"### Source: {p.url}\n"
            f"Title: {p.title}\n\n"
            f"**Pass-1 summary**\n\n{p1_text}\n\n"
            f"**Pass-2 deep extract**\n\n{p2_text}"
        )
    evidence = "\n\n---\n\n".join(parts)
    user = (
        f"Chunk {chunk_idx} of {n_chunks} — {len(pages_in_chunk)} pages.\n\n"
        f"{evidence}"
    )

    llm = _build_llm(0.4, 131072, True, SYNTHESIS_SCHEMA, "synthesis_chunk")
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        return json.loads(resp.content)["summary"]
    except Exception as e:
        return f"(chunk {chunk_idx} synthesis LLM error: {e!r})"


def stage6_synth(pages: dict[str, Page], brief: str) -> str:
    """Chunked synthesis: groups of STAGE_6_CHUNK_SIZE pass-2 extracts are
    synthesized in parallel; outputs are stitched into one markdown document
    with ### Chunk N/M subheaders. No second-level merge pass."""
    extracts = [p for p in pages.values() if p.pass2 and p.pass2.deep_summary]
    if not extracts:
        summary = "(no pass-2 extracts available — synthesis skipped)"
        _write_artifact("stages/06-synthesis.md",
                        "# Synthesis\n\n" + summary)
        return summary

    chunks = [
        extracts[i:i + STAGE_6_CHUNK_SIZE]
        for i in range(0, len(extracts), STAGE_6_CHUNK_SIZE)
    ]
    n_chunks = len(chunks)
    print(f"[stage 6] {len(extracts)} extracts → {n_chunks} chunk(s) of up to "
          f"{STAGE_6_CHUNK_SIZE} (parallel, llm_conc={LLM_CONCURRENCY})")

    chunk_outputs: list[str] = [""] * n_chunks
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        futs = {
            pool.submit(_synth_one_chunk, idx + 1, n_chunks, brief, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for fut in futs:
            idx = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                out = f"(chunk {idx + 1} fan-out error: {e!r})"
            chunk_outputs[idx] = out
            print(f"  [synth chunk {idx + 1}/{n_chunks} done] "
                  f"{len(out):,} chars")
            _write_artifact(
                f"stages/06-synthesis/chunk-{idx + 1:02d}.md",
                "\n".join([
                    f"# Synthesis chunk {idx + 1}/{n_chunks}",
                    "",
                    f"_{len(chunks[idx])} source pages_",
                    "",
                    out,
                ]),
            )

    # Stitch with demoted (### Chunk N/M) subheaders so the result drops
    # cleanly under report.md's ## Findings section.
    stitched_parts = []
    for i, out in enumerate(chunk_outputs, 1):
        stitched_parts.append(
            f"### Chunk {i}/{n_chunks} — {len(chunks[i - 1])} pages\n\n{out}"
        )
    combined = "\n\n---\n\n".join(stitched_parts)

    _write_artifact(
        "stages/06-synthesis.md",
        "\n".join([
            "# Synthesis (chunked)",
            "",
            f"_chunked from {len(extracts)} pass-2 deep extracts into "
            f"{n_chunks} chunk(s) of up to {STAGE_6_CHUNK_SIZE}_",
            "",
            combined,
        ]),
    )
    return combined


# ---------------------------------------------------------------------------
# Stage 8 — hierarchical pairwise finalize (report.md → report_final.md)
# ---------------------------------------------------------------------------

FINALIZE_SCHEMA = {
    "type": "object",
    "properties": {"report": {"type": "string"}},
    "required": ["report"],
    "additionalProperties": False,
}

FINALIZE_SYSTEM = (
    "You are merging two pieces of an in-progress research report into one "
    "comprehensive markdown report. Both pieces cover the same overall "
    "topic.\n\n"
    "PRESERVE EVERYTHING. You are unifying, not summarizing. Every distinct "
    "fact, claim, number, quote, date, named source, and example from "
    "either input must survive to the output. Do NOT condense, abridge, or "
    "shorten 'for brevity'. Information loss is worse than redundancy.\n\n"
    "KEEP FULL URL citations inline as markdown links. When an input has "
    "[text](https://full.url/path) or bare URLs, preserve them as proper "
    "markdown links in your output. Do NOT collapse to domain-only "
    "references like [example.com] — keep the full URL. Each source is "
    "mentioned briefly inline (a few words + the full-URL link), not as a "
    "long footnote or quote-block citation.\n\n"
    "PRESERVE direct quotes verbatim. PRESERVE exact numbers, dates, and "
    "named sources.\n\n"
    "DROP only:\n"
    "  - verbatim duplicate claims that appear in both inputs;\n"
    "  - claims attributed solely to commercial / advertisement-heavy / "
    "promotional vendor pages (CV-builder marketing, recruiter SEO blogs, "
    "AI-generated listicles, generic resume-template hubs) — AND only when "
    "the same claim also appears with a better source. A unique claim, "
    "even from a weak source, is kept.\n\n"
    "UNSOURCED claims: keep if plausible and well-stated. Drop only "
    "dubious-sounding unsourced claims.\n\n"
    "Do NOT produce a 'Sources', 'References', or 'Bibliography' section "
    "at the end. Inline citations only.\n\n"
    "Use markdown headers and structure for a research-report style. "
    "Reorganize where helpful, but do not drop topics.\n\n"
    'Respond as JSON: {"report": "<full markdown report>"}'
)


def _merge_pair(round_idx: int, pair_idx: int, left: str, right: str) -> str:
    user = (
        f"## Piece A\n\n{left}\n\n"
        f"---\n\n"
        f"## Piece B\n\n{right}"
    )
    llm = _build_llm(0.4, 131072, STAGE8_THINKING, FINALIZE_SCHEMA,
                     "finalize_merge")
    try:
        resp = llm.invoke([
            SystemMessage(content=FINALIZE_SYSTEM),
            HumanMessage(content=user),
        ])
        return json.loads(resp.content)["report"]
    except Exception as e:
        return (f"(merge LLM error round={round_idx} pair={pair_idx}: "
                f"{e!r})")


def _read_stage6_chunk_body(path: Path) -> str:
    """Strip the 4-line header that stage 6 prepends to chunk md files,
    returning just the synthesis body."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n", 4)
    return lines[4].strip() if len(lines) >= 5 else text.strip()


def stage8_finalize(run_dir: Path) -> str:
    """Hierarchical pairwise merge of stage-6 chunk md files into one final
    report. Algorithm: at each round, pair consecutive items; if the count
    is odd, the last item carries through to the next round unchanged.
    Continue until one item remains.

    Per-merge output is cached at stages/08-finalize/r{NN}p{NN}.md so that
    a crash mid-stage-8 can be resumed via --resume-stage 8 --continue
    (which skips merges whose output md already exists)."""
    chunk_dir = run_dir / "stages" / "06-synthesis"
    chunk_files = sorted(chunk_dir.glob("chunk-*.md"))
    if not chunk_files:
        msg = f"(stage 8 skipped — no chunk files in {chunk_dir})"
        print(f"[stage 8] {msg}")
        _write_artifact("stages/08-finalize/final.md", msg)
        return msg

    current: list[str] = [_read_stage6_chunk_body(f) for f in chunk_files]
    print(f"[stage 8] starting pairwise merge of {len(current)} chunks "
          f"(thinking={STAGE8_THINKING}, llm_conc={LLM_CONCURRENCY})")

    finalize_dir = run_dir / "stages" / "08-finalize"
    finalize_dir.mkdir(parents=True, exist_ok=True)

    round_idx = 1
    while len(current) > 1:
        n = len(current)
        n_pairs = n // 2
        has_carry = (n % 2 == 1)
        carry_label = " + 1 carry" if has_carry else ""
        print(f"[stage 8] round {round_idx}: {n} items → "
              f"{n_pairs} pair(s){carry_label}")

        next_round: list[str | None] = [None] * n_pairs
        merge_tasks: list[tuple[int, str, str, Path]] = []
        for pair_idx in range(n_pairs):
            left = current[pair_idx * 2]
            right = current[pair_idx * 2 + 1]
            cache_path = (finalize_dir
                          / f"r{round_idx:02d}p{pair_idx + 1:02d}.md")
            if cache_path.exists():
                cached = cache_path.read_text(
                    encoding="utf-8", errors="replace").strip()
                next_round[pair_idx] = cached
                print(f"  [r{round_idx:02d}p{pair_idx + 1:02d}] cached "
                      f"({len(cached):,} chars)")
            else:
                merge_tasks.append(
                    (pair_idx, left, right, cache_path))

        if merge_tasks:
            with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
                futs = {
                    pool.submit(_merge_pair, round_idx, pi + 1, l, r):
                        (pi, cp)
                    for pi, l, r, cp in merge_tasks
                }
                for fut in futs:
                    pi, cp = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        out = f"(merge fan-out error: {e!r})"
                    next_round[pi] = out
                    print(f"  [r{round_idx:02d}p{pi + 1:02d} done] "
                          f"{len(out):,} chars")
                    cp.write_text(out, encoding="utf-8")

        if has_carry:
            next_round.append(current[-1])

        current = [c for c in next_round if c is not None]
        round_idx += 1

    final = current[0]
    final_path = run_dir / "report_final.md"
    final_path.write_text(final, encoding="utf-8")
    print(f"[stage 8] wrote {final_path} ({len(final):,} chars)")
    return final


# ---------------------------------------------------------------------------
# Stage 7 — output
# ---------------------------------------------------------------------------

def stage7_output(
    run_dir: Path,
    topic: str,
    locale: str,
    language: str,
    country: str,
    search_lang: str,
    ui_lang: str,
    brief: str,
    stage1_reasoning: str | None,
    queries: list[str],
    query_runs: list[QueryRun],
    pages: dict[str, Page],
    synthesis: str,
    final_report: str | None,
) -> None:
    ts = _dt.datetime.now().isoformat(timespec="seconds")

    sources = [p for p in pages.values() if p.pass1 and p.pass1.has_value]
    sources.sort(key=lambda p: p.url)

    lines = [
        f"# {topic}",
        f"_generated: {ts}_",
        f"_locale: {locale} | language: {language}_",
        "",
        "## Original prompt",
        "",
        "```",
        brief.rstrip(),
        "```",
        "",
        "## Queries",
        "",
    ]
    for i, qr in enumerate(query_runs, 1):
        lines.append(f"{i}. {qr.query}")
    lines += ["", "## Findings", "", synthesis.rstrip(), "", "## Sources", ""]
    if sources:
        for p in sources:
            label = p.title or p.url
            lines.append(f"- [{label}]({p.url})")
    else:
        lines.append("_no pages passed pass-1 relevance gate_")
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stage 7] wrote {report_path}")

    # Per-query hit-rate courtesy log (since stage 6 is now blind to queries).
    print("[stage 7] per-query pass-1 hit rates:")
    for i, qr in enumerate(query_runs, 1):
        total = len(qr.results)
        hits = sum(
            1 for sr in qr.results
            if pages.get(sr.url) and pages[sr.url].pass1
            and pages[sr.url].pass1.has_value
        )
        print(f"  query {i:02d}: {hits}/{total} pages with value — {qr.query}")

    cp = _build_intermediates(
        topic, locale, language, country, search_lang, ui_lang, brief,
        stage1_reasoning, queries, query_runs, pages, synthesis,
        final_report, last_completed_stage=7,
    )
    _write_checkpoint(run_dir, cp)
    print(f"[stage 7] wrote {run_dir / 'intermediates.json'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_brief(path: Path) -> str:
    """Read the research brief from a plain-text file. Locale/language and
    Brave-localization fields live in config.py — only the brief body comes
    from this file."""
    if not path.exists():
        sys.exit(f"prompt file not found at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit(f"prompt file at {path} is empty")
    return text


def _build_pages_from_query_runs(query_runs: list[QueryRun]) -> dict[str, Page]:
    pages: dict[str, Page] = {}
    for qr in query_runs:
        for sr in qr.results:
            if sr.url in pages:
                pages[sr.url].surfaced_by.append(qr.query)
            else:
                pages[sr.url] = Page(
                    url=sr.url,
                    title=sr.title,
                    surfaced_by=[qr.query],
                )
    return pages


def _resume_state_from_disk(
    run_dir: Path,
    prompt_path: Path,
) -> tuple[
    str, str, str, str, str, str, str,
    str | None,
    list[str] | None,
    list[QueryRun] | None,
    dict[str, Page] | None,
    str | None,
    str | None,
]:
    """For --continue: reconstruct pipeline state from whatever's on disk.
    Prefers intermediates.json; falls back to parsing stage md files.
    Always backfills pass1/pass2/synth/final_report from md files on top of
    whatever the checkpoint has — handles mid-stage crashes where per-URL
    artifacts exist but the stage-boundary checkpoint wasn't written yet."""
    cp_path = run_dir / "intermediates.json"

    if cp_path.exists():
        cp = _load_checkpoint(run_dir)
        topic = cp["topic"]
        locale = cp["locale"]
        language = cp["language"]
        country = cp["country"]
        search_lang = cp["search_lang"]
        ui_lang = cp["ui_lang"]
        brief = cp["brief"]
        stage1_reasoning = cp.get("stage1_reasoning")
        queries = cp.get("queries")
        synthesis = cp.get("synthesis")
        final_report = cp.get("final_report")
        query_runs, pages = _rehydrate_from_checkpoint(cp)
    else:
        print(f"[continue] no intermediates.json — rebuilding state from md files")
        locale = LOCALE
        language = LANGUAGE
        country = COUNTRY
        search_lang = SEARCH_LANG
        ui_lang = UI_LANG
        brief = _load_brief(prompt_path)
        name = run_dir.name
        tm = re.match(r"^(.*?)-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name)
        topic = tm.group(1) if tm else name
        stage1_reasoning, queries = _recover_queries_from_md(run_dir)
        query_runs = _recover_query_runs_from_md(run_dir)
        pages = (
            _build_pages_from_query_runs(query_runs) if query_runs else None
        )
        synthesis = None
        final_report = None

    # Per-URL md backfill (pass1/pass2/synth/final_report) is done by the
    # caller in main(), after it has had a chance to null out state for
    # --resume-stage N.
    return (topic, locale, language, country, search_lang, ui_lang, brief,
            stage1_reasoning, queries, query_runs, pages, synthesis,
            final_report)


def _rehydrate_from_checkpoint(cp: dict) -> tuple[
    list[QueryRun] | None, dict[str, Page] | None,
]:
    """Reconstruct QueryRun / Page objects from a checkpoint dict."""
    query_runs: list[QueryRun] | None = None
    if cp.get("query_runs") is not None:
        query_runs = [
            QueryRun(
                query=qr["query"],
                results=[SearchResult(**r) for r in qr["results"]],
            )
            for qr in cp["query_runs"]
        ]

    pages: dict[str, Page] | None = None
    if cp.get("pages") is not None:
        pages = {}
        for url, pd in cp["pages"].items():
            p = Page(
                url=pd["url"],
                title=pd["title"],
                surfaced_by=list(pd.get("surfaced_by") or []),
            )
            if pd.get("pass1"):
                p.pass1 = Pass1(**pd["pass1"])
            if pd.get("pass2"):
                p.pass2 = Pass2(**pd["pass2"])
            pages[url] = p

    return query_runs, pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", default=str(PROMPT_FILE),
        help="path to plain-text prompt file (default: prompt.txt next to this script)",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="path to an existing run dir to resume (required with "
             "--resume-stage or --continue)",
    )
    parser.add_argument(
        "--resume-stage", type=int, default=None,
        choices=list(VALID_RESUME_STAGES),
        help="enter the pipeline at stage N. Without --continue, deletes "
             "stage-N-or-later artifacts first (hard reset).",
    )
    parser.add_argument(
        "--continue", dest="continue_mode", action="store_true",
        help="soft resume: keep every existing md file and scavenge "
             "pass1/pass2/synth state from disk. May be combined with "
             "--resume-stage N to force re-entry at stage N. Without "
             "--resume-stage, auto-detects which stages need to run.",
    )
    args = parser.parse_args()

    if args.resume_stage is not None and not args.run_dir:
        sys.exit("--resume-stage requires --run-dir <path>")
    if args.continue_mode and not args.run_dir:
        sys.exit("--continue requires --run-dir <path>")
    if args.run_dir and args.resume_stage is None and not args.continue_mode:
        sys.exit("--run-dir requires either --resume-stage <N> or --continue")

    if AUTO_START_CHECKS:
        from auto_start import run_checks
        run_checks()

    t0 = time.perf_counter()

    # ---- state to thread through the pipeline ----
    topic: str
    locale: str; language: str; country: str; search_lang: str; ui_lang: str
    brief: str
    stage1_reasoning: str | None = None
    queries: list[str] | None = None
    query_runs: list[QueryRun] | None = None
    pages: dict[str, Page] | None = None
    synthesis: str | None = None
    final_report: str | None = None

    if args.resume_stage is not None or args.continue_mode:
        # ---- RESUME (hard / soft / forced-entry-soft) ----
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            sys.exit(f"run dir not found: {run_dir}")
        install_logger(run_dir, mode="a")

        rs = args.resume_stage  # may be None (pure --continue)
        soft = args.continue_mode
        label = (
            f"hard-resume stage {rs}" if (rs is not None and not soft)
            else f"soft-resume stage {rs}" if (rs is not None and soft)
            else "continue (auto-detect)"
        )
        print(f"\n[resume] {run_dir}  [{label}]")

        # Hydrate state — prefer checkpoint, backfill from md files when soft
        if soft:
            (topic, locale, language, country, search_lang, ui_lang, brief,
             stage1_reasoning, queries, query_runs, pages, synthesis,
             final_report
            ) = _resume_state_from_disk(run_dir, Path(args.prompt))
        else:
            cp = _load_checkpoint(run_dir)
            print(f"[resume] last_completed_stage in checkpoint: "
                  f"{cp.get('last_completed_stage')}")
            topic = cp["topic"]
            locale = cp["locale"]
            language = cp["language"]
            country = cp["country"]
            search_lang = cp["search_lang"]
            ui_lang = cp["ui_lang"]
            brief = cp["brief"]
            stage1_reasoning = cp.get("stage1_reasoning")
            queries = cp.get("queries")
            synthesis = cp.get("synthesis")
            final_report = cp.get("final_report")
            query_runs, pages = _rehydrate_from_checkpoint(cp)

        # If --resume-stage N: null out in-memory state for stage N and later
        # so the condition-driven dispatch re-enters at N.
        if rs is not None:
            if rs <= 1:
                stage1_reasoning = None
                queries = None
            if rs <= 2:
                query_runs = None
                pages = None
            if rs <= 4 and pages is not None:
                for p in pages.values():
                    p.pass1 = None
                    p.pass2 = None
            if rs <= 5 and pages is not None:
                for p in pages.values():
                    p.pass2 = None
            if rs <= 6:
                synthesis = None
            if rs <= 8:
                final_report = None

        # Hard reset ONLY when --resume-stage given without --continue.
        if rs is not None and not soft:
            _cleanup_from_stage(run_dir, rs)

        # Soft mode: backfill per-URL state from md files AFTER the null-out,
        # so existing pass1/pass2 artifacts are trusted for pages that have
        # them, while missing ones stay None and will get re-run.
        if soft and pages is not None:
            recovered_p1 = recovered_p2 = 0
            for page in pages.values():
                if page.pass1 is None:
                    rec = _recover_pass1_from_md(run_dir, page)
                    if rec is not None:
                        page.pass1 = rec
                        recovered_p1 += 1
                if page.pass2 is None:
                    rec = _recover_pass2_from_md(run_dir, page)
                    if rec is not None:
                        page.pass2 = rec
                        recovered_p2 += 1
            if recovered_p1 or recovered_p2:
                print(f"[resume] backfilled pass-1:{recovered_p1} "
                      f"pass-2:{recovered_p2} from md files")
            # Synthesis is monolithic (one file, all-or-nothing), unlike
            # per-URL pass1/pass2 where partial state is meaningful. If the
            # user explicitly re-targeted stage 6 (or earlier), do NOT
            # backfill — otherwise a failed synthesis.md (e.g. an error
            # string from a prior crash) would be re-adopted and stage 6
            # would skip itself.
            if synthesis is None and (rs is None or rs > 6):
                synthesis = _recover_synth_from_md(run_dir)
                if synthesis:
                    print(f"[resume] backfilled synthesis from md "
                          f"({len(synthesis):,} chars)")
            # Same gate for final_report (stage 8): don't trust the on-disk
            # report_final.md when the user explicitly re-targeted stage 8.
            if final_report is None and (rs is None or rs > 8):
                final_report = _recover_final_report_from_md(run_dir)
                if final_report:
                    print(f"[resume] backfilled final_report from md "
                          f"({len(final_report):,} chars)")

        # Report state of play
        if queries is not None:
            print(f"[resume] {len(queries)} queries loaded")
        if pages is not None:
            n_p1 = sum(1 for p in pages.values() if p.pass1 is not None)
            n_has = sum(1 for p in pages.values()
                        if p.pass1 and p.pass1.has_value)
            n_p2 = sum(1 for p in pages.values() if p.pass2 is not None)
            print(f"[resume] {len(pages)} pages | pass1 done: {n_p1} "
                  f"({n_has} has_value) | pass2 done: {n_p2}")
        if synthesis:
            print(f"[resume] synthesis: {len(synthesis):,} chars")
        if final_report:
            print(f"[resume] final_report: {len(final_report):,} chars")

        # Pre-populate the in-memory page cache from pages/*.md so stage 4/5
        # can reuse content without re-fetching.
        if pages is not None:
            preloaded = _preload_page_cache(run_dir, pages)
            if preloaded:
                print(f"[resume] preloaded {preloaded} pages into fetch cache")

    else:
        # ---- FRESH RUN ----
        brief = _load_brief(Path(args.prompt))
        locale = LOCALE
        language = LANGUAGE
        country = COUNTRY
        search_lang = SEARCH_LANG
        ui_lang = UI_LANG

        # Stage 0 — topic (before the log dir exists, so console only)
        print("[stage 0] naming the topic ...")
        topic = stage0_topic(brief)

        # Create the run dir and install the log tee
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = LOGS_DIR / f"{topic}-{ts}"
        install_logger(run_dir)
        print(f"[log dir] {run_dir}")
        print(f"[topic]   {topic}")
        print(f"[locale]      {locale}")
        print(f"[language]    {language}")
        print(f"[country]     {country}")
        print(f"[search_lang] {search_lang}")
        print(f"[ui_lang]     {ui_lang}")

    def _checkpoint(stage: int) -> None:
        cp = _build_intermediates(
            topic, locale, language, country, search_lang, ui_lang, brief,
            stage1_reasoning, queries, query_runs, pages, synthesis,
            final_report, last_completed_stage=stage,
        )
        _write_checkpoint(run_dir, cp)
        print(f"[checkpoint] stage {stage} -> {run_dir / 'intermediates.json'}")

    # ---- Stage 1 ----
    if queries is None:
        print("\n[stage 1] planning queries ...")
        stage1_reasoning, queries = stage1_queries(brief, language)
        print(f"[stage 1] {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        _write_artifact(
            "stages/01-queries.md",
            "\n".join([
                "# Stage 1 — query plan",
                "",
                "## Reasoning",
                "",
                stage1_reasoning or "",
                "",
                "## Queries",
                "",
                *[f"{i}. {q}" for i, q in enumerate(queries, 1)],
            ]),
        )
        print("[stage 1] wrote stages/01-queries.md")
        _checkpoint(1)
    else:
        print(f"\n[stage 1] skipped — {len(queries)} queries loaded from checkpoint")

    # ---- Stage 2 ----
    if query_runs is None or pages is None:
        print("\n[stage 2] batch search (semantic cache aware) ...")
        try:
            per_query_results = stage2_search(
                queries, locale, country, search_lang, ui_lang,
            )
        except urllib.error.HTTPError as e:
            # AUTO_CRASH_ON_FAILED_SEARCH bubbled the failure up here. Print
            # a self-contained recovery hint and exit. The semantic cache has
            # already persisted every query that succeeded before the failure,
            # so the suggested resume command hits cache for those queries.
            cmd = f"python research.py --run-dir {run_dir} --resume-stage 2 --continue"
            print(
                f"\n[stage 2] FAILED.\n"
                f"Your {SEARCH_ENGINE!r} search backend failed to provide "
                f"results — this is usually a rate limit, an API quota / auth "
                f"issue, or an upstream block. The given error code was "
                f"{e.code} ({e.reason}).\n\n"
                f"You can continue the run with: {cmd}\n\n"
                f"(To proceed past failed queries instead of aborting, set "
                f"AUTO_CRASH_ON_FAILED_SEARCH = False in config.py.)",
                file=sys.stderr,
            )
            sys.exit(2)
        for i, rs_ in enumerate(per_query_results, 1):
            print(f"  query {i}: {len(rs_)} results")
        _write_artifact(
            "stages/02-search_results.md",
            "\n".join([
                "# Stage 2 — search results",
                "",
                *sum((
                    [f"## Query {i}: {q}", ""]
                    + [f"{j}. [{r.title or '(no title)'}]({r.url})\n   {r.description}"
                       for j, r in enumerate(rs_, 1)]
                    + [""]
                    for i, (q, rs_) in enumerate(zip(queries, per_query_results), 1)
                ), []),
            ]),
        )
        print("[stage 2] wrote stages/02-search_results.md")

        query_runs = [
            QueryRun(query=q, results=rs_)
            for q, rs_ in zip(queries, per_query_results)
        ]
        pages = {}
        for qr in query_runs:
            for sr in qr.results:
                if sr.url in pages:
                    pages[sr.url].surfaced_by.append(qr.query)
                else:
                    pages[sr.url] = Page(
                        url=sr.url,
                        title=sr.title,
                        surfaced_by=[qr.query],
                    )
        total_slots = sum(len(qr.results) for qr in query_runs)
        print(f"\n[pages] {len(pages)} unique URLs across "
              f"{total_slots} search-result slots")
        _checkpoint(2)
    else:
        total_slots = sum(len(qr.results) for qr in query_runs)
        print(f"\n[stage 2] skipped — {len(pages)} unique URLs across "
              f"{total_slots} slots loaded from checkpoint")

    # ---- Stage 4 ----
    if any(p.pass1 is None for p in pages.values()):
        print()
        stage4_pass1(pages, brief)
        hits = sum(1 for p in pages.values() if p.pass1 and p.pass1.has_value)
        print(f"[stage 4] {hits}/{len(pages)} URLs flagged has_value=True")
        _checkpoint(4)
    else:
        hits = sum(1 for p in pages.values() if p.pass1 and p.pass1.has_value)
        print(f"\n[stage 4] skipped — all {len(pages)} URLs already scanned "
              f"({hits} has_value=True)")

    # ---- Stage 5 ----
    promising = [p for p in pages.values() if p.pass1 and p.pass1.has_value]
    if any(p.pass2 is None for p in promising):
        print()
        stage5_pass2(pages, brief)
        _checkpoint(5)
    else:
        print(f"\n[stage 5] skipped — all {len(promising)} promising pages "
              f"already have pass-2 extracts")

    # ---- Stage 6 ----
    if synthesis is None:
        print()
        synthesis = stage6_synth(pages, brief)
        _checkpoint(6)
    else:
        print(f"\n[stage 6] skipped — synthesis already in checkpoint "
              f"({len(synthesis):,} chars)")

    # ---- Stage 7 ----
    print()
    stage7_output(
        run_dir, topic, locale, language, country, search_lang, ui_lang,
        brief, stage1_reasoning, queries, query_runs, pages, synthesis,
        final_report,
    )

    # ---- Stage 8 — pairwise finalize → report_final.md ----
    if final_report is None:
        print()
        final_report = stage8_finalize(run_dir)
        _checkpoint(8)
    else:
        print(f"\n[stage 8] skipped — final_report already in checkpoint "
              f"({len(final_report):,} chars)")

    elapsed = time.perf_counter() - t0
    print(f"\n[done] total elapsed {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
