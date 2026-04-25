"""Pre-flight environment checks for research.py.

When config.AUTO_START_CHECKS is True, research.py calls run_checks() at the
top of main(). Verifies that every library required by the chosen
SEARCH_ENGINE is importable, that the right credentials are in place,
prompt.txt exists, and three live probes succeed:

  1. search engine returns results for a known-good query ("python", 10)
  2. llama-server is reachable at http://localhost:8080
  3. crawl4ai can fetch https://github.com

Exits with a clear, actionable error if anything fails; prints a single OK
line on success.

Standalone CLI:  python auto_start.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import SEARCH_ENGINE, SEMANTIC_CACHE_MATCHING


_LLM_BASE_URL = "http://localhost:8080/v1"

# Module name (the name you'd `import`) -> friendly pip-install name.
# `playwright` is intentionally NOT here: crawl4ai pulls it in transitively
# and is the actual import-site, so the crawl4ai check covers it.
_ALWAYS_REQUIRED: dict[str, str] = {
    "numpy": "numpy",
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "crawl4ai": "crawl4ai",
}

_BRAVE_REQUIRED: dict[str, str] = {
    "dotenv": "python-dotenv",
}

_DDGS_REQUIRED: dict[str, str] = {
    "ddgs": "ddgs[api]",
}

# Only required when SEMANTIC_CACHE_MATCHING = True. Pulls in torch + the
# bge-m3 model (~2.3GB on first run); dropping these is the main reason to
# turn the toggle off.
_SEMANTIC_CACHE_REQUIRED: dict[str, str] = {
    "sentence_transformers": "sentence-transformers",
}


def _missing_modules(req: dict[str, str]) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for import_name, pip_name in req.items():
        try:
            spec = importlib.util.find_spec(import_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append((import_name, pip_name))
    return missing


def _check_brave_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    if not os.environ.get("BRAVE_API_KEY"):
        return (
            "BRAVE_API_KEY not set. Either:\n"
            "      - put `BRAVE_API_KEY=your-key` in .env (next to research.py),\n"
            "      - export BRAVE_API_KEY=your-key in your shell, or\n"
            "      - switch to ddgs by setting SEARCH_ENGINE = 'ddgs' in config.py."
        )
    return None


def _check_ddgs_binary() -> str | None:
    if shutil.which("ddgs") is None:
        return (
            "`ddgs` console-script not on PATH. Install with: "
            "`pip install ddgs[api]` (and ensure your scripts dir is on PATH)."
        )
    return None


def _check_prompt_file() -> str | None:
    p = Path(__file__).parent / "prompt.txt"
    if not p.exists():
        return f"prompt.txt not found at {p}. Write your research brief there."
    if not p.read_text(encoding="utf-8").strip():
        return f"prompt.txt at {p} is empty. Write your research brief there."
    return None


def _check_llm_reachable() -> str | None:
    """GET /v1/models — confirms llama-server is up and a model is loaded.
    Pure connectivity probe; no token generation."""
    url = f"{_LLM_BASE_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status != 200:
                return f"llama-server at {url} returned HTTP {r.status}"
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        return (
            f"llama-server unreachable at {url}: {e!r}. "
            f"Is `llama-server` running?"
        )
    except Exception as e:
        return f"llama-server probe failed: {e!r}"

    models = [m.get("id", "?") for m in data.get("data", [])]
    if not models:
        return f"llama-server at {url} reports no loaded models"
    print(f"  [llm]      OK - {url} (models: {models})")
    return None


def _check_search_query() -> str | None:
    """Hit the configured search engine with a known-good query and verify
    we got results back. Bypasses the semantic cache (this is a wire test)."""
    query = "python"
    target = 10
    t0 = time.perf_counter()
    try:
        if SEARCH_ENGINE == "brave":
            from config import COUNTRY, SEARCH_LANG, UI_LANG
            import search_brave
            results = search_brave.search(
                query, count=target,
                country=COUNTRY, search_lang=SEARCH_LANG, ui_lang=UI_LANG,
            )
        elif SEARCH_ENGINE == "ddgs":
            from config import DDGS_BACKEND, LOCALE
            import search_ddg  # autostarts the sidecar
            results = search_ddg.search(
                query, count=target, backend=DDGS_BACKEND, region=LOCALE,
            )
        else:
            return f"unknown SEARCH_ENGINE={SEARCH_ENGINE!r}"
    except urllib.error.HTTPError as e:
        return (
            f"{SEARCH_ENGINE!r} search test failed: HTTP {e.code} ({e.reason}). "
            f"Likely a rate limit, auth issue, or upstream block."
        )
    except Exception as e:
        return f"{SEARCH_ENGINE!r} search test failed: {e!r}"

    if not results:
        return f"{SEARCH_ENGINE!r} returned 0 results for {query!r}"

    elapsed = time.perf_counter() - t0
    suffix = "" if len(results) == target else f" (asked {target}, got fewer)"
    print(
        f"  [search]   OK - {SEARCH_ENGINE!r} returned {len(results)}/{target} "
        f"results for {query!r} in {elapsed:.1f}s{suffix}"
    )
    return None


def _check_crawl4ai_fetch() -> str | None:
    """Fetch a known-good URL via crawl4ai. Slow (Playwright spin-up): ~5-15s."""
    url = "https://github.com"
    t0 = time.perf_counter()
    try:
        from fetch import fetch_website
        content = fetch_website(url)
    except Exception as e:
        return f"crawl4ai fetch of {url} failed: {e!r}"

    if not content or len(content) < 100:
        return (
            f"crawl4ai fetch of {url} returned empty/tiny content "
            f"(len={len(content or '')}); Playwright may not be installed - "
            f"run `playwright install chromium`"
        )
    elapsed = time.perf_counter() - t0
    print(f"  [crawl4ai] OK - fetched {url} ({len(content):,} chars) in {elapsed:.1f}s")
    return None


def _exit_with_failures(failures: list[str]) -> None:
    print("\n[auto_start] pre-flight checks FAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    print(
        "\n(Set AUTO_START_CHECKS = False in config.py to skip these checks.)",
        file=sys.stderr,
    )
    sys.exit(2)


def run_checks() -> None:
    """Verify environment is ready. Two phases: static (libraries / creds /
    prompt), then live probes (search / llm / fetch). Static failures abort
    before live probes to avoid noisy follow-on errors."""
    print(f"[auto_start] running pre-flight checks (SEARCH_ENGINE={SEARCH_ENGINE!r})...")

    failures: list[str] = []

    # --- static checks ---
    for import_name, pip_name in _missing_modules(_ALWAYS_REQUIRED):
        failures.append(
            f"missing library `{import_name}` - install with `pip install {pip_name}`"
        )

    if SEMANTIC_CACHE_MATCHING:
        for import_name, pip_name in _missing_modules(_SEMANTIC_CACHE_REQUIRED):
            failures.append(
                f"missing library `{import_name}` (required by SEMANTIC_CACHE_MATCHING=True) "
                f"- install with `pip install {pip_name}`, or set "
                f"SEMANTIC_CACHE_MATCHING = False in config.py"
            )

    if SEARCH_ENGINE == "ddgs":
        for import_name, pip_name in _missing_modules(_DDGS_REQUIRED):
            failures.append(
                f"missing library `{import_name}` (required by SEARCH_ENGINE='ddgs') "
                f"- install with `pip install {pip_name}`"
            )
        err = _check_ddgs_binary()
        if err: failures.append(err)
    elif SEARCH_ENGINE == "brave":
        for import_name, pip_name in _missing_modules(_BRAVE_REQUIRED):
            failures.append(
                f"missing library `{import_name}` (required by SEARCH_ENGINE='brave') "
                f"- install with `pip install {pip_name}`"
            )
        err = _check_brave_key()
        if err: failures.append(err)
    else:
        failures.append(
            f"config.SEARCH_ENGINE = {SEARCH_ENGINE!r} is not one of: 'brave', 'ddgs'"
        )

    err = _check_prompt_file()
    if err: failures.append(err)

    if failures:
        _exit_with_failures(failures)

    # --- live probes (only if static passed) ---
    for probe in (_check_llm_reachable, _check_search_query, _check_crawl4ai_fetch):
        err = probe()
        if err: failures.append(err)

    if failures:
        _exit_with_failures(failures)

    print("[auto_start] all checks OK")


if __name__ == "__main__":
    run_checks()
