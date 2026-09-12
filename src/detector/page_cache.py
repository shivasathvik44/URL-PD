"""Cache for Layer 4 (page fetch + LLM) results.

URL-level, unlike the domain-level Layer 2 cache: page content belongs to one
URL, and two URLs on the same domain can be completely different pages.

Caching Layer 4 is worth more than caching anything else here - each miss costs
a live HTTP fetch of a hostile page *and* a paid LLM call - but it is also the
one most dangerous to get wrong, so:

  * rows are keyed by a hash of the URL plus the analysis version, and a row
    written by a different version is ignored rather than reinterpreted;
  * the TTL is short by default (hours, not days): phishing pages are taken
    down, re-pointed and cloaked constantly, so a stale "safe" verdict is worse
    than a cache miss;
  * **only completed analyses are stored.** A blocked fetch, a network failure
    or an LLM outage is never cached - otherwise one transient failure would
    pin a "we couldn't check" result for the whole TTL.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .config import DEFAULT_CACHE_CONFIG, CacheConfig
from .llm_analyzer import ANALYSIS_VERSION, LLMAnalysis

_SCHEMA = """
CREATE TABLE IF NOT EXISTS page_analysis (
    url_hash         TEXT PRIMARY KEY,
    payload          TEXT NOT NULL,
    analysed_at      REAL NOT NULL,
    analysis_version INTEGER NOT NULL
);
"""

#: Fields restored from a cached row. Kept explicit so a schema change to
#: LLMAnalysis cannot silently resurrect a differently-shaped payload.
_RESTORED_FIELDS = (
    "performed", "brand_claimed", "brand_impersonation", "social_engineering",
    "social_engineering_tactics", "credential_request", "page_purpose",
    "suspicious_observations", "prompt_injection_attempt", "risk_level",
    "confidence", "llm_likelihood", "schema_violations", "provider", "model",
    "elapsed_ms", "error",
)
_TUPLE_FIELDS = frozenset(
    {"social_engineering_tactics", "suspicious_observations", "schema_violations"}
)


def url_key(url: str) -> str:
    """Stable, non-reversible key. Hashing keeps raw URLs (which can carry a
    victim's email or a session token) out of the cache file."""
    return hashlib.sha256((url or "").strip().encode("utf-8", "replace")).hexdigest()


class PageCache:
    """SQLite store of completed Layer 4 analyses, keyed by URL hash."""

    def __init__(self, path: str | Path | None = None, *, ttl_seconds: float = 6 * 3600.0):
        self.path = str(path or DEFAULT_CACHE_CONFIG.page_path)
        self.ttl_seconds = float(ttl_seconds)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0
        self.stale = 0

    def get(self, url: str) -> LLMAnalysis | None:
        row = self._conn.execute(
            "SELECT payload, analysed_at, analysis_version FROM page_analysis "
            "WHERE url_hash = ?",
            (url_key(url),),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None

        payload, analysed_at, version = row
        if version != ANALYSIS_VERSION:
            self.stale += 1
            self.misses += 1
            return None
        if self.ttl_seconds > 0 and (time.time() - float(analysed_at)) > self.ttl_seconds:
            self.stale += 1
            self.misses += 1
            return None

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:  # pragma: no cover - corrupt row
            self.misses += 1
            return None

        self.hits += 1
        kwargs = {}
        for name in _RESTORED_FIELDS:
            value = data.get(name)
            kwargs[name] = tuple(value or ()) if name in _TUPLE_FIELDS else value
        return LLMAnalysis(**kwargs)

    def put(self, url: str, analysis: LLMAnalysis) -> None:
        # Only completed analyses are worth keeping; a failure is transient.
        if not analysis.performed:
            return
        self._conn.execute(
            "INSERT INTO page_analysis (url_hash, payload, analysed_at, analysis_version) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(url_hash) DO UPDATE SET payload=excluded.payload, "
            "analysed_at=excluded.analysed_at, analysis_version=excluded.analysis_version",
            (url_key(url), json.dumps(analysis.as_dict(), default=str),
             time.time(), ANALYSIS_VERSION),
        )
        self._conn.commit()

    def row_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM page_analysis").fetchone()[0]

    def stats(self) -> dict:
        return {"rows": self.row_count(), "hits": self.hits, "misses": self.misses,
                "stale": self.stale, "path": self.path,
                "analysis_version": ANALYSIS_VERSION}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PageCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_page_cache(config: CacheConfig = DEFAULT_CACHE_CONFIG) -> PageCache | None:
    """Open the configured Layer 4 cache, or None when caching is disabled."""
    if not config.page_enabled:
        return None
    try:
        return PageCache(config.page_path, ttl_seconds=config.page_ttl_seconds)
    except sqlite3.Error:  # pragma: no cover - unwritable path
        return None
