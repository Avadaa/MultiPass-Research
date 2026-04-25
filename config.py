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
