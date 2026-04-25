"""Shared web fetch helper with a cross-run SQLite cache.

Successful fetches are stored at ./cache/page_cache.sqlite keyed by URL.
Subsequent calls for the same URL return the cached markdown without hitting
Playwright. Failed fetches are NOT cached (so future runs can retry).
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig


_CRAWLER_CFG = CrawlerRunConfig(
    excluded_tags=["nav", "header", "footer", "aside", "script", "style", "noscript"],
)


FETCH_CHAR_LIMIT = 50000
CACHE_DB = Path(__file__).parent / "cache" / "page_cache.sqlite"
_db_lock = threading.Lock()


def _open_db() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS page_cache (
            url TEXT PRIMARY KEY,
            fetched_at REAL NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _cache_get(url: str) -> tuple[str, float] | None:
    with _db_lock:
        conn = _open_db()
        try:
            row = conn.execute(
                "SELECT content, fetched_at FROM page_cache WHERE url = ?",
                (url,),
            ).fetchone()
        finally:
            conn.close()
    return (row[0], row[1]) if row else None


def _cache_put(url: str, content: str) -> None:
    if not content:
        return
    with _db_lock:
        conn = _open_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO page_cache "
                "(url, fetched_at, content) VALUES (?, ?, ?)",
                (url, time.time(), content),
            )
            conn.commit()
        finally:
            conn.close()


def fetch_website(url: str, force: bool = False) -> str:
    """Fetch URL, return markdown. Uses a SQLite page cache across runs.

    Raises on fetch failure (same as before). Pass force=True to bypass the
    cache for this call (the fresh result still goes into the cache on success).
    """
    if not force:
        cached = _cache_get(url)
        if cached is not None:
            content, ts = cached
            age_days = (time.time() - ts) / 86400.0
            print(f"[page-cache hit] {url} (age {age_days:.1f}d)")
            return content

    async def _run():
        async with AsyncWebCrawler() as c:
            result = await c.arun(url=url, config=_CRAWLER_CFG)
        if not result.success:
            raise RuntimeError(result.error_message or "fetch failed")
        md = result.markdown or ""
        return md if isinstance(md, str) else getattr(md, "raw_markdown", "")

    content = asyncio.run(_run())
    _cache_put(url, content)
    return content
