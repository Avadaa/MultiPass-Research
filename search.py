"""Search orchestrator with a semantic cache.

Dispatches to one of the engine clients based on config.SEARCH_ENGINE:
  - search_brave.search(...) — Brave API (requires BRAVE_API_KEY)
  - search_ddg.search(...)   — DDGS metasearch via the local `ddgs api` sidecar

Each engine client returns the canonical shape
[{"title", "url", "description"}, ...] so the cache never sees engine-specific
payloads.

Usage:
    python search.py "your query"
    python search.py "q1" "q2" "q3"             # batch
    python search.py --no-cache "your query"    # bypass cache read, still store

Caching:
  - bge-m3 (BAAI/bge-m3) dense embeddings, loaded on every run.
  - SQLite at cache/search_cache.sqlite stores (timestamp, engine, query,
    query_embedding, results). Rows are partitioned by engine — only rows
    tagged with the current SEARCH_ENGINE are eligible for lookup.
  - Fast path: exact normalized-query match returns immediately.
  - Otherwise: cosine-rank cached queries from the last 30 days, pre-filter
    top1 >= 0.5, send top-5 to llama-server for a final yes/no.
  - LLM fail-open: if the local server is down, fetch fresh.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.error
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    AUTO_CRASH_ON_FAILED_SEARCH,
    LOCALE,
    SEARCH_ENGINE,
    SEARCH_RESULT_COUNT,
)
from llm import ChatLlamaServer
import search_brave
import search_ddg


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DB = CACHE_DIR / "search_cache.sqlite"
WINDOW_DAYS = 30
TOP_K = 5
SIM_FLOOR = 0.8
EMBED_DIM = 1024
_NORMALIZE_STRIP = ".,?-:;"


def _normalize(q: str) -> str:
    """Lowercase, replace .,?-:; with spaces, collapse whitespace."""
    s = q.lower()
    for ch in _NORMALIZE_STRIP:
        s = s.replace(ch, " ")
    return " ".join(s.split())


# -----------------------------------------------------------------------------
# Engine dispatcher
# -----------------------------------------------------------------------------

_KNOWN_ENGINES = ("brave", "ddgs")


def _engine_fetch(
    engine: str,
    query: str,
    count: int,
    region: str,
    country: str,
    search_lang: str,
    ui_lang: str,
) -> list[dict]:
    """Route a query to the configured engine. Engines return the canonical
    [{"title", "url", "description"}, ...] shape so the cache stores a
    uniform payload regardless of source.

    Localization fields are engine-specific:
      - DDGS uses `region` ("fi-fi") and ignores the rest.
      - Brave uses (country, search_lang, ui_lang) and ignores `region`.
    """
    if engine == "brave":
        return search_brave.search(
            query, count,
            country=country,
            search_lang=search_lang,
            ui_lang=ui_lang,
        )
    if engine == "ddgs":
        return search_ddg.search(query, count, region=region)
    raise RuntimeError(
        f"Unknown search engine {engine!r}. "
        f"Set config.SEARCH_ENGINE to one of: {list(_KNOWN_ENGINES)}"
    )


# -----------------------------------------------------------------------------
# Cache store
# -----------------------------------------------------------------------------

def _open_cache() -> sqlite3.Connection:
    CACHE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            search_engine TEXT NOT NULL,
            region TEXT,
            query TEXT NOT NULL,
            query_embedding BLOB NOT NULL,
            results TEXT NOT NULL
        )
    """)
    # Migrate existing DBs that predate the `region` column.
    try:
        conn.execute("ALTER TABLE search_cache ADD COLUMN region TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    # Purge pre-region ddgs rows — they were fetched without a known locality
    # and can't be safely matched against region-aware queries.
    conn.execute(
        "DELETE FROM search_cache "
        "WHERE search_engine = 'ddgs' AND region IS NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON search_cache(timestamp)"
    )
    conn.commit()
    return conn


def _store(
    conn: sqlite3.Connection,
    engine: str,
    region: str,
    query: str,
    embedding: np.ndarray,
    results: list[dict],
) -> None:
    conn.execute(
        "INSERT INTO search_cache "
        "(timestamp, search_engine, region, query, query_embedding, results) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            time.time(),
            engine,
            region,
            query,
            embedding.astype(np.float32).tobytes(),
            json.dumps(results),
        ),
    )
    conn.commit()


def _fetch_candidates(
    conn: sqlite3.Connection,
    window_days: int,
    engine: str,
    region: str,
) -> list[tuple[int, str, np.ndarray, str, float]]:
    """Return [(id, query, embedding, results_json, timestamp)] for rows newer
    than cutoff AND matching (`engine`, `region`). Deduplicated by normalized
    query — only the newest row per normalized query survives, so the LLM
    picker never sees the same query multiple times. Cross-engine AND cross-
    region rows are excluded: flipping LOCALE isolates past results
    fetched under a different locality."""
    cutoff = time.time() - window_days * 86400
    rows = conn.execute(
        "SELECT id, query, query_embedding, results, timestamp "
        "FROM search_cache "
        "WHERE timestamp > ? AND search_engine = ? AND region = ? "
        "ORDER BY timestamp DESC",
        (cutoff, engine, region),
    ).fetchall()
    seen: dict[str, tuple] = {}
    for r in rows:
        key = _normalize(r[1])
        if key not in seen:  # first = newest because ORDER BY timestamp DESC
            seen[key] = r
    return [
        (r[0], r[1], np.frombuffer(r[2], dtype=np.float32), r[3], r[4])
        for r in seen.values()
    ]


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------

def _load_embedder():
    """Load bge-m3. First run downloads ~2.3GB to HF cache."""
    print("[cache] loading bge-m3 (first run: ~2.3GB download) ...")
    from sentence_transformers import SentenceTransformer  # heavy import, defer
    model = SentenceTransformer("BAAI/bge-m3")
    print("[cache] embedder ready.")
    return model


def _embed(model, text: str) -> np.ndarray:
    v = model.encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


# -----------------------------------------------------------------------------
# LLM match decision
# -----------------------------------------------------------------------------

PICKER_SCHEMA = {
    "type": "object",
    "properties": {
        "match_idx": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["match_idx", "reason"],
    "additionalProperties": False,
}

PICKER_SYSTEM = (
    "You decide whether a cached search result is close enough to a new query "
    "that we can skip re-fetching. Favor a cache hit only when the cached "
    "query genuinely covers what the new query asks. Different entities, "
    "different time periods, or different intents = no match. "
    "Respond with match_idx = 1..N (the number of the matching candidate) or "
    "null if none is good enough.\n\n"
    "HARD RULE — language match. Search engines rank results by query "
    "language. Never pick a candidate written in a different language than "
    "the new query — its cached results are from a different result set and "
    "will not answer the new query. If every close-similarity candidate is "
    "in a different language than the new query, return match_idx = null.\n\n"
    "Note that even subtle differences in the prompts are a good enough "
    "reason to re-fetch. Your task is only to check if it's the same search "
    "but with different wording. See the semantics of the prompt and the "
    "proposed cache results; if it's even remotely possible that the user "
    "is searching something new, trigger the search.\n\n"
    "Examples (same language):\n"
    "'How long to train my puppy' and 'How long to the puppytrain' are different\n"
    "'How long to train my puppy' and 'How long to train my puppy german shepherd' are different\n"
    "'How long to train my puppy' and 'My puppy longs me' are different\n"
    "'How long to train my puppy' and 'How to train my dog' are different\n"
    "'How long to train my puppy' and 'What is the duration teach my young dog' are the same\n"
    "'How long to train my puppy' and 'I got a puppy, how long to train him' are the same\n"
    "'How long to train my puppy' and 'I got a puppy, how to train him' are different\n\n"
    "Examples (cross-language — always different, always null):\n"
    "'ansioluettelo pituus suositus' (Finnish) and 'CV length recommendation Finland' (English) — different\n"
    "'CV length recommendation' (English) and 'ansioluettelon sivumäärä' (Finnish) — different"
)


def _build_picker_llm():
    base = ChatLlamaServer(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        model="local",
        temperature=0.2,
        max_tokens=512,
    )
    return base.bind(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cache_pick",
                "strict": True,
                "schema": PICKER_SCHEMA,
            },
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def _llm_pick_match(new_query: str, candidates: list[dict]) -> tuple[int | None, str]:
    """candidates: list of {'query', 'sim', 'age_days'}. Returns (match_idx_1based | None, reason)."""
    lines = [f"New query: {new_query}", "", "Top candidates:"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f'{i}. "{c["query"]}" (sim={c["sim"]:.3f}, age={c["age_days"]:.1f}d)'
        )
    lines.append("")
    lines.append('Respond JSON: {"match_idx": <1-N or null>, "reason": "<brief>"}')
    user_msg = "\n".join(lines)

    llm = _build_picker_llm()
    resp = llm.invoke([
        SystemMessage(content=PICKER_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    parsed = json.loads(resp.content)
    idx = parsed.get("match_idx")
    reason = parsed.get("reason", "")
    if isinstance(idx, int) and 1 <= idx <= len(candidates):
        return idx, reason
    return None, reason


# -----------------------------------------------------------------------------
# Caching wrapper
# -----------------------------------------------------------------------------

def _resolve_one(
    query: str,
    q_emb: np.ndarray,
    snapshot: list[tuple[int, str, np.ndarray, str, float]],
    conn: sqlite3.Connection,
    count: int,
    use_cache: bool,
    region: str,
    country: str,
    search_lang: str,
    ui_lang: str,
) -> list[dict]:
    """Resolve a single query against a pre-taken snapshot. Stores fresh fetches."""

    def _fetch() -> list[dict]:
        return _engine_fetch(
            SEARCH_ENGINE, query, count,
            region, country, search_lang, ui_lang,
        )

    if not use_cache:
        print("[cache] --no-cache: bypassing cache read.")
        results = _fetch()
        _store(conn, SEARCH_ENGINE, region, query, q_emb, results)
        return results

    if not snapshot:
        print("[cache] empty within window; fetching fresh.")
        results = _fetch()
        _store(conn, SEARCH_ENGINE, region, query, q_emb, results)
        return results

    # Fast-path: normalized 1:1 match short-circuits embedding + LLM.
    # If multiple rows match, take the newest.
    q_norm = _normalize(query)
    exact = [r for r in snapshot if _normalize(r[1]) == q_norm]
    if exact:
        picked = max(exact, key=lambda r: r[4])  # newest by timestamp
        age_days = (time.time() - picked[4]) / 86400.0
        print(
            f"[cache hit] exact (normalized) match: "
            f'query={picked[1]!r}, age={age_days:.1f}d'
        )
        return json.loads(picked[3])

    embs = np.stack([r[2] for r in snapshot])  # (N, 1024)
    sims = embs @ q_emb                         # (N,)
    order = np.argsort(-sims)[:TOP_K]

    top_rows = [snapshot[i] for i in order]
    top_sims = [float(sims[i]) for i in order]

    if top_sims[0] < SIM_FLOOR:
        print(
            f"[cache] top sim {top_sims[0]:.3f} < floor {SIM_FLOOR}; "
            "no close candidate, fetching fresh."
        )
        results = _fetch()
        _store(conn, SEARCH_ENGINE, region, query, q_emb, results)
        return results

    now = time.time()
    candidates = [
        {
            "query": top_rows[i][1],
            "sim": top_sims[i],
            "age_days": (now - top_rows[i][4]) / 86400.0,
        }
        for i in range(len(top_rows))
    ]

    print("[cache] top candidates:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. \"{c['query']}\" (sim={c['sim']:.3f}, age={c['age_days']:.1f}d)")

    try:
        match_idx, reason = _llm_pick_match(query, candidates)
    except Exception as e:
        print(f"[cache] LLM picker failed ({e!r}); fetching fresh (fail-open).")
        results = _fetch()
        _store(conn, SEARCH_ENGINE, region, query, q_emb, results)
        return results

    if match_idx is None:
        print(f"[cache] LLM: no match ({reason!r}); fetching fresh.")
        results = _fetch()
        _store(conn, SEARCH_ENGINE, region, query, q_emb, results)
        return results

    picked = top_rows[match_idx - 1]
    print(
        f"[cache hit] via LLM: candidate {match_idx} "
        f'(query={picked[1]!r}, sim={top_sims[match_idx - 1]:.3f}) — {reason}'
    )
    return json.loads(picked[3])


def search(
    queries: list[str] | str,
    count: int = SEARCH_RESULT_COUNT,
    use_cache: bool = True,
    region: str | None = None,
    country: str = "US",
    search_lang: str = "en",
    ui_lang: str = "en-US",
) -> list[list[dict]]:
    """Resolve one or more queries. Cache snapshot is taken ONCE up front, so
    near-duplicate queries submitted in the same batch don't cache-hit each other.

    Accepts either a single string (returned as a single-element result list)
    or a list of strings. Always returns list[list[dict]] — one result list
    per query, in input order.

    Localization fields are forwarded to engines that honor them:
      - DDGS uses `region` (e.g. "fi-fi") and ignores country/search_lang/ui_lang.
      - Brave uses (country, search_lang, ui_lang) and ignores `region`.
    The cache partitions by (engine, region). Pass `region=None` to fall back
    to config.LOCALE.
    """
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return []

    region_eff = region if region is not None else LOCALE

    conn = _open_cache()
    embedder = _load_embedder()

    # Batch-embed all queries in one call (GPU-friendly).
    raw = embedder.encode(queries, normalize_embeddings=True)
    q_embs = np.asarray(raw, dtype=np.float32)
    if q_embs.ndim == 1:
        q_embs = q_embs.reshape(1, -1)

    # Snapshot taken ONCE before any query runs. Fresh fetches from this batch
    # are stored but don't re-enter the snapshot, so later queries in the same
    # batch see the pre-batch cache state.
    snapshot = _fetch_candidates(conn, WINDOW_DAYS, SEARCH_ENGINE, region_eff) if use_cache else []
    if use_cache:
        print(
            f"[cache] snapshot: {len(snapshot)} row(s) within {WINDOW_DAYS}d "
            f"window (engine={SEARCH_ENGINE}, region={region_eff})."
        )

    out: list[list[dict]] = []
    for i, q in enumerate(queries):
        if len(queries) > 1:
            print(f"\n=== query {i + 1}/{len(queries)}: {q!r} ===")
        try:
            results = _resolve_one(
                q, q_embs[i], snapshot, conn, count, use_cache,
                region_eff, country, search_lang, ui_lang,
            )
        except urllib.error.HTTPError as e:
            if AUTO_CRASH_ON_FAILED_SEARCH:
                raise
            # Soft mode: log the failure and continue with an empty result
            # for this query. Other queries in the batch still run.
            print(
                f"[search] query {i + 1} failed: HTTP {e.code} "
                f"({e.reason}); continuing with empty results "
                f"(AUTO_CRASH_ON_FAILED_SEARCH=False)"
            )
            results = []
        out.append(results)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _print_results(query: str, results: list[dict]) -> None:
    print(f"\n--- results for {query!r} ---")
    if not results:
        print("(no results)")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title')}")
        print(f"   {r.get('url')}")
        desc = (r.get("description") or "").strip()
        if desc:
            print(f"   {desc[:200]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Brave search with a local semantic cache. "
                    "Pass one or more quoted queries; each is resolved against "
                    "the cache snapshot taken at batch start.",
    )
    parser.add_argument("--no-cache", action="store_true", help="bypass cache read")
    parser.add_argument(
        "queries",
        nargs="*",
        help="one or more queries, each a separate (quoted) argument",
    )
    args = parser.parse_args()

    queries = args.queries or ["how to detect AI text in cover letters"]
    print(f"[queries] {queries}")

    all_results = search(queries, use_cache=not args.no_cache)

    for q, results in zip(queries, all_results):
        _print_results(q, results)


if __name__ == "__main__":
    main()
