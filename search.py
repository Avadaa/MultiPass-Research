"""Brave Search smoke test.

Usage:
    set BRAVE_API_KEY=...        (cmd)
    $env:BRAVE_API_KEY = "..."   (PowerShell)
    python search.py "your query"
    python search.py             # default query

Get a key at https://api-dashboard.search.brave.com/ (free: 1 req/s, 2k/mo).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def search(query: str, count: int = 10) -> list[dict]:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY not set. Get a key at "
            "https://api-dashboard.search.brave.com/"
        )
    params = urllib.parse.urlencode({"q": query, "count": count})
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


def main():
    query = " ".join(sys.argv[1:]) or "how to detect AI text in cover letters"
    print(f"[query] {query}\n")
    results = search(query, count=10)
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


if __name__ == "__main__":
    main()
