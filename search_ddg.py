"""DDGS metasearch HTTP client via the localhost `ddgs api` sidecar.

Prereq:  in another terminal, run `ddgs api`. Default binding is
http://127.0.0.1:8000. Install once:  pip install ddgs[api].

Why a sidecar:  the HTTP call from this script to 127.0.0.1 is loopback
traffic and bypasses the VPN entirely. Whether ddgs's own outbound search
requests go through the VPN depends on the sidecar process's own split-
tunnel status — a single clean decision point instead of inheriting
python.exe's routing.

Exposes:
    search(query, count, backend, region, endpoint)
        -> list[{"title", "url", "description"}]

Non-200 responses (including 429 rate limits) raise immediately with a
full traceback — rate limits must be unmissable. Connection failures
(sidecar not running) exit cleanly with a hint.

Standalone CLI:  python search_ddg.py "query"
Browse live schema at http://127.0.0.1:8000/docs.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import LOCALE, SEARCH_RESULT_COUNT


DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
DEFAULT_BACKEND = "google,duckduckgo"


def search(
    query: str,
    count: int = SEARCH_RESULT_COUNT,
    backend: str = DEFAULT_BACKEND,
    region: str = LOCALE,
    endpoint: str = DEFAULT_ENDPOINT,
) -> list[dict]:
    """Normalized to [{"title", "url", "description"}, ...]."""
    params = urllib.parse.urlencode({
        "query": query,
        "max_results": count,
        "backend": backend,
        "region": region,
    })
    url = f"{endpoint.rstrip('/')}/search/text?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        if r.status != 200:
            raise RuntimeError(
                f"ddgs returned HTTP {r.status} (expected 200): "
                f"{body[:500].decode('utf-8', errors='replace')!r}"
            )
        raw = json.loads(body)
    # Server wraps the list in {"results": [...]}; older/other endpoints may
    # return a flat list. Accept both.
    if isinstance(raw, dict):
        for key in ("results", "items", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise RuntimeError(f"ddgs returned unexpected shape: {raw!r}")
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("href") or r.get("url") or "",
            "description": r.get("body") or r.get("description") or "",
        }
        for r in raw
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="ddgs backend(s). e.g. 'google,duckduckgo' or 'auto'")
    parser.add_argument("--count", type=int, default=SEARCH_RESULT_COUNT)
    parser.add_argument("--region", default=LOCALE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help=f"ddgs server base URL (default {DEFAULT_ENDPOINT})")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    query = " ".join(args.query) or "how to detect AI text in cover letters"
    print(f"[query]    {query}")
    print(f"[backend]  {args.backend}")
    print(f"[region]   {args.region}")
    print(f"[endpoint] {args.endpoint}\n")

    try:
        results = search(
            query,
            count=args.count,
            backend=args.backend,
            region=args.region,
            endpoint=args.endpoint,
        )
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        sys.exit(f"request failed: {e!r}. Is `ddgs api` running at {args.endpoint}?")

    if not results:
        print("(no results)")
        return
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
