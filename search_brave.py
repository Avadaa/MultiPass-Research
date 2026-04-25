"""Brave Search HTTP client.

Exposes:
    search(query, count) -> list[{"title", "url", "description"}]

Auto-paginates: Brave caps `count` at 20 per request, so for count>20 this
issues multiple requests with `offset=0,1,2,...` and merges the pages,
deduped by URL. Brave also caps `offset` at 9, so the hard ceiling is 200
results per query (10 pages × 20). Caller sees a flat list, same shape as
a single-page response.

Requires BRAVE_API_KEY in env (or .env via python-dotenv).
Standalone CLI for smoke testing:  python search_brave.py "query"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config import SEARCH_RESULT_COUNT


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_PER_PAGE_MAX = 20   # Brave API cap on the `count` parameter
BRAVE_MAX_OFFSET = 9      # Brave API cap on the `offset` parameter
BRAVE_PAGE_DELAY_S = 1.1  # honor the 1-QPS free-tier rate limit


def _fetch_page(
    api_key: str, query: str, count: int, offset: int,
    country: str, search_lang: str, ui_lang: str,
) -> list[dict]:
    params = urllib.parse.urlencode({
        "q": query,
        "count": count,
        "offset": offset,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
    })
    req = urllib.request.Request(
        f"{BRAVE_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return data.get("web", {}).get("results", [])


def search(
    query: str,
    count: int = SEARCH_RESULT_COUNT,
    country: str = "US",
    search_lang: str = "en",
    ui_lang: str = "en-US",
) -> list[dict]:
    """Normalized to [{"title", "url", "description"}, ...], up to `count`
    items. Paginates transparently via Brave's `offset` when count > 20.

    Localization (Brave-specific):
      - country: ISO 3166-1 alpha-2, uppercase (e.g. "FI", "SE", "US").
      - search_lang: ISO 639-1, lowercase (e.g. "fi", "en").
      - ui_lang: BCP-47, language-COUNTRY (e.g. "fi-FI", "en-US").

    Raises on missing API key."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY not set. Get a key at "
            "https://api-dashboard.search.brave.com/"
        )
    if count <= 0:
        return []

    collected: list[dict] = []
    seen_urls: set[str] = set()
    offset = 0

    while len(collected) < count and offset <= BRAVE_MAX_OFFSET:
        if offset > 0:
            time.sleep(BRAVE_PAGE_DELAY_S)
        raw = _fetch_page(
            api_key, query, BRAVE_PER_PAGE_MAX, offset,
            country, search_lang, ui_lang,
        )
        if not raw:
            break  # Brave has no more results for this query

        added = 0
        for r in raw:
            url = r.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            collected.append({
                "title": r.get("title") or "",
                "url": url,
                "description": r.get("description") or "",
            })
            added += 1
            if len(collected) >= count:
                break

        # Page returned fewer unique items than a full page worth — likely
        # hit the tail of Brave's index for this query. Stop paginating.
        if len(raw) < BRAVE_PER_PAGE_MAX or added == 0:
            break
        offset += 1

    return collected[:count]


def main():
    parser = argparse.ArgumentParser(description="Brave Search smoke test")
    parser.add_argument("--count", type=int, default=SEARCH_RESULT_COUNT)
    parser.add_argument("--country", default="US")
    parser.add_argument("--search-lang", default="en")
    parser.add_argument("--ui-lang", default="en-US")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    query = " ".join(args.query) or "how to detect AI text in cover letters"
    print(f"[query]       {query}")
    print(f"[country]     {args.country}")
    print(f"[search_lang] {args.search_lang}")
    print(f"[ui_lang]     {args.ui_lang}\n")
    results = search(
        query, args.count,
        country=args.country,
        search_lang=args.search_lang,
        ui_lang=args.ui_lang,
    )
    print(f"({len(results)} result(s))\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title'] or '(no title)'}")
        print(f"   {r['url']}")
        if r["description"]:
            snippet = r["description"].strip().replace("\n", " ")
            print(f"   {snippet[:200]}")
        print()


if __name__ == "__main__":
    main()
