"""Project configuration.

Behavior toggles live here. Secrets live in .env (loaded via python-dotenv).
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Locale — shapes search ranking AND fills in language placeholders in
# research.py's stage-1 query-generator system prompt.
# ---------------------------------------------------------------------------

# DDGS region / locality. Format is "<lang>-<country>" (e.g. "fi-fi", "us-en",
# "se-sv", "de-de"). Shapes result ranking and surfaces more domestic pages.
# Only consumed by the DDGS engine; Brave ignores it.
LOCALE = "fi-fi"

# Human-readable language name, used by research.py's stage-1 query-generator
# template ("Write EVERY word in {LANGUAGE}..."). E.g. "Finnish", "Swedish".
LANGUAGE = "Finnish"

# Brave Search localization. Brave ignores LOCALE and uses these instead.
#   COUNTRY     : ISO 3166-1 alpha-2, uppercase (e.g. "FI", "SE", "US").
#   SEARCH_LANG : ISO 639-1, lowercase (e.g. "fi", "en").
#   UI_LANG     : BCP-47 (e.g. "fi-FI", "en-US").
COUNTRY = "FI"
SEARCH_LANG = "fi"
UI_LANG = "fi-FI"
