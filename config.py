"""Project configuration.

Behavior toggles live here. Secrets live in .env (loaded via python-dotenv).
"""
from __future__ import annotations

import json
from pathlib import Path

#ddgs api

# Which search engine search.py uses.
# "ddgs" — talks to a local `ddgs api` sidecar (DDGS metasearch: Google, Bing,
#          DuckDuckGo, Mojeek, Brave, etc. with automatic fallback).
# The semantic cache partitions by engine — when this changes, past results
# fetched from a different engine are NOT considered when looking for cache
# hits (they're still retained in the DB for later re-use).
SEARCH_ENGINE = "brave"

# Default number of results to fetch per query when no explicit count is passed.
SEARCH_RESULT_COUNT = 30

# DDGS backend(s) — comma-separated, in order of preference. Common values:
#   "auto"                    — let ddgs pick
#   "duckduckgo"              — DuckDuckGo only
#   "google,duckduckgo"       — Google first, fallback DuckDuckGo
#   "google,duckduckgo,bing"  — three-way
# Only consumed by the DDGS engine.
DDGS_BACKEND = "google,duckduckgo"

# Behavior when a search backend returns a non-200 HTTP response — typically
# a rate limit (429), API quota/auth problem (401/403), upstream block, or
# server error (5xx). DDGS additionally surfaces 2xx-non-200 codes (e.g. 202)
# when upstream DuckDuckGo throttles its scraper.
#
# True (default): abort the run immediately. Stage 2 is fragile — one failed
#   query means that query's URLs are absent from pass-1 onward, which
#   silently degrades the final report. The whole research becomes severely
#   handicapped. Crashing loud lets you fix the upstream issue (rotate keys,
#   wait out the rate limit, switch SEARCH_ENGINE) and resume cleanly with
#   --resume-stage 2 --continue. The semantic cache persists every query
#   that succeeded before the failure, so the resume hits cache for those.
#
# False: log the failure and continue with whatever queries did succeed.
#   Use only if partial coverage is acceptable.
AUTO_CRASH_ON_FAILED_SEARCH = True

# Pre-flight checks at research.py startup: required libraries, engine creds,
# prompt.txt, plus live probes (llama-server, search engine, crawl4ai fetch).
# True = fail fast with actionable errors. False = skip; trust the environment.
AUTO_START_CHECKS = True

# Semantic cache layer for the search cache. When True, queries are embedded
# with bge-m3 (sentence-transformers) and matched against past queries via
# cosine similarity + an LLM picker, so paraphrases hit cache. When False,
# only literal matching is used (lowercase + strip punctuation) and
# sentence-transformers / bge-m3 / torch can be dropped from the install.
SEMANTIC_CACHE_MATCHING = True

# ---------------------------------------------------------------------------
# Locale - the only two fields you set:
#
#   LANGUAGE : the language your search queries will be written in. Stage 1's
#              query-generator template asks the LLM to phrase every query in
#              this language ("Write EVERY word in {LANGUAGE}..."). Pick a
#              human-readable name from the keys of _LANGUAGE_TO_CODE below.
#   COUNTRY  : the geographic target for the search. ISO 3166-1 alpha-2,
#              uppercase (e.g. "FI", "DE", "US"). Biases the search engine to
#              surface more domestic pages from that country.
#
# Everything else (SEARCH_LANG, UI_LANG, LOCALE) is derived from these two.
# Don't hand-edit the derived fields - change LANGUAGE / COUNTRY instead.
#
# Mismatched combos are valid: LANGUAGE="Finnish" + COUNTRY="SE" gets you
# Finnish-language queries against Swedish-region search results.
# ---------------------------------------------------------------------------

LANGUAGE = "Finnish"
COUNTRY  = "FI" #GB, AU, NZ, DE...

# Human-readable language name -> ISO 639-1 lowercase code. Loaded from
# data/languages.json; edit that file to add or change entries. The code
# is what Brave's `search_lang` and DDG's `kl` consume.
_LANGUAGES_PATH = Path(__file__).parent / "data" / "languages.json"
with _LANGUAGES_PATH.open(encoding="utf-8") as _f:
    _LANGUAGE_TO_CODE: dict[str, str] = json.load(_f)

if LANGUAGE not in _LANGUAGE_TO_CODE:
    raise ValueError(
        f"config.LANGUAGE = {LANGUAGE!r} is not in _LANGUAGE_TO_CODE. "
        f"Add an entry, or pick one of: {sorted(_LANGUAGE_TO_CODE)}"
    )

# --- derived ---
SEARCH_LANG = _LANGUAGE_TO_CODE[LANGUAGE]              # ISO 639-1 lowercase, e.g. "fi"
UI_LANG     = f"{SEARCH_LANG}-{COUNTRY}"               # BCP-47, e.g. "fi-FI"
# DDG's `kl` param format is `<country>-<lang>` lowercased: "fi-fi", "us-en",
# "se-sv", "de-de". (Looks palindromic for Finland/German because their
# country and language codes match; not the case for e.g. Sweden -> "se-sv".)
LOCALE      = f"{COUNTRY.lower()}-{SEARCH_LANG}"
