"""Persistent cache for Layer 2 (DNS / RDAP / WHOIS) results.

Layer 2 does real network I/O per domain, and both training (thousands of URLs)
and live analysis (the same domain seen repeatedly) benefit from not repeating
it. This cache is keyed by **registered domain** - deliberately domain-level,
not URL-level, because DNS/RDAP facts belong to the domain and are shared by
every URL on it. Layer 4's page analysis is URL-level and lives in
`page_cache.py`.

Storage: one SQLite file (stdlib, no dependency, safe to interrupt).

Two correctness guards:
  * every row stores the `payload_version` that wrote it, and rows written by a
    different version are ignored - a cached payload can never be silently
    reinterpreted against a changed Layer 2 shape;
  * rows older than the configured TTL are ignored (TTL <= 0 means "never
    expire", which is what training wants for reproducibility).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .config import DEFAULT_CACHE_CONFIG, DEFAULT_CONFIG, CacheConfig, DetectorConfig
from .lexical_features import registered_domain_of
from .observability import get_logger

_log = get_logger("layer2.cache")

#: Bump when `DomainIntelligence.as_dict()` changes shape.
PAYLOAD_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS layer2 (
    domain          TEXT PRIMARY KEY,
    payload         TEXT NOT NULL,
    fetched_at      REAL NOT NULL,
    payload_version INTEGER NOT NULL DEFAULT 1,
    negative        INTEGER NOT NULL DEFAULT 0
);
"""


def _payload_resolves(payload: dict) -> bool:
    """Did this payload's domain resolve? Handles both stored shapes.

    `to_cache()` nests DNS under a "dns" key; the flat feature view from
    `as_dict()` has `dns_resolves` at the top level. Reading only one of them
    would mark every row of the other shape as a negative result.
    """
    nested = payload.get("dns")
    if isinstance(nested, dict) and "dns_resolves" in nested:
        return bool(nested["dns_resolves"])
    return bool(payload.get("dns_resolves", False))


def cache_key(url: str) -> str:
    """Registered domain for `url` (falls back to the raw string).

    Uses Layer 1's PSL logic via `registered_domain_of`, so cache keys can never
    disagree with the rest of the system about which domain a URL is on.
    """
    return registered_domain_of(url) or (url or "").strip().lower()


class Layer2Cache:
    """SQLite key/value store for Layer 2 payloads, keyed by registered domain."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        ttl_seconds: float = 0.0,
        negative_ttl_seconds: float | None = None,
    ) -> None:
        self.path = str(path or DEFAULT_CACHE_CONFIG.layer2_path)
        self.ttl_seconds = float(ttl_seconds)
        # A negative row (domain does not resolve) expires sooner than a
        # positive one; None means "same TTL as everything else".
        self.negative_ttl_seconds = (
            float(negative_ttl_seconds) if negative_ttl_seconds is not None
            else float(ttl_seconds)
        )
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._migrate()
        self._conn.commit()
        self.hits = 0
        self.misses = 0
        self.stale = 0

    def _migrate(self) -> None:
        """Add `payload_version` to a cache file written by an older build."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(layer2)")}
        if "payload_version" not in cols:
            self._conn.execute(
                "ALTER TABLE layer2 ADD COLUMN payload_version INTEGER NOT NULL DEFAULT 1"
            )
        if "negative" not in cols:
            self._conn.execute(
                "ALTER TABLE layer2 ADD COLUMN negative INTEGER NOT NULL DEFAULT 0"
            )

    # -- dict-ish API ----------------------------------------------------
    def get(self, url: str) -> dict | None:
        key = cache_key(url)
        row = self._conn.execute(
            "SELECT payload, fetched_at, payload_version, negative FROM layer2 "
            "WHERE domain = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None

        payload, fetched_at, version, negative = row
        if version != PAYLOAD_VERSION:
            self.stale += 1
            self.misses += 1
            return None

        ttl = self.negative_ttl_seconds if negative else self.ttl_seconds
        if ttl > 0 and (time.time() - float(fetched_at)) > ttl:
            self.stale += 1
            self.misses += 1
            return None

        self.hits += 1
        try:
            return json.loads(payload)
        except json.JSONDecodeError:  # pragma: no cover - corrupt row
            self.misses += 1
            return None

    def put(self, url: str, payload: dict, *, negative: bool | None = None) -> None:
        key = cache_key(url)
        if negative is None:
            negative = not _payload_resolves(payload)
        self._conn.execute(
            "INSERT INTO layer2 (domain, payload, fetched_at, payload_version, negative) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(domain) DO UPDATE SET payload=excluded.payload, "
            "fetched_at=excluded.fetched_at, payload_version=excluded.payload_version, "
            "negative=excluded.negative",
            (key, json.dumps(payload, default=str), time.time(), PAYLOAD_VERSION,
             1 if negative else 0),
        )
        self._conn.commit()

    def row_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM layer2").fetchone()[0]

    def purge_stale(self) -> int:
        """Delete rows from an older payload version. Returns rows removed."""
        cur = self._conn.execute(
            "DELETE FROM layer2 WHERE payload_version != ?", (PAYLOAD_VERSION,)
        )
        self._conn.commit()
        return cur.rowcount

    def stats(self) -> dict:
        return {"rows": self.row_count(), "hits": self.hits, "misses": self.misses,
                "stale": self.stale, "path": self.path,
                "payload_version": PAYLOAD_VERSION}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Layer2Cache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_layer2_cache(config: CacheConfig = DEFAULT_CACHE_CONFIG) -> Layer2Cache | None:
    """Open the configured Layer 2 cache, or None when caching is disabled."""
    if not config.layer2_enabled:
        return None
    try:
        return Layer2Cache(
            config.layer2_path,
            ttl_seconds=config.layer2_ttl_seconds,
            negative_ttl_seconds=config.layer2_negative_ttl_seconds,
        )
    except sqlite3.Error:  # pragma: no cover - unwritable path, read-only FS...
        return None


def get_layer2_intel(
    url: str,
    *,
    cache: Layer2Cache | None = None,
    config: DetectorConfig = DEFAULT_CONFIG,
    do_domain_age: bool = True,
    do_reverse_dns: bool = True,
    do_tls: bool = False,
    analyze_domain_fn=None,
):
    """Return a `DomainIntelligence` for `url`, using `cache` when possible.

    Cache hits are rehydrated into the real dataclass (via `from_cache`), so a
    cached result is indistinguishable from a fresh one to every caller.
    Returns None if Layer 2 itself failed - never raises.
    """
    from .domain_intelligence import DomainIntelligence

    if cache is not None:
        hit = cache.get(url)
        if hit is not None:
            try:
                return DomainIntelligence.from_cache(hit)
            except Exception as exc:  # noqa: BLE001 - a bad row is just a miss
                _log.debug("cache row could not be rehydrated",
                           extra={"fields": {"error": type(exc).__name__}})

    if analyze_domain_fn is None:
        from .domain_intelligence import analyze_domain as analyze_domain_fn

    try:
        intel = analyze_domain_fn(
            url, config, do_reverse_dns=do_reverse_dns,
            do_domain_age=do_domain_age, do_tls=do_tls,
        )
    except TypeError:
        # An injected test double may not accept do_tls.
        try:
            intel = analyze_domain_fn(
                url, config, do_reverse_dns=do_reverse_dns, do_domain_age=do_domain_age
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("layer2 lookup failed",
                         extra={"fields": {"error": type(exc).__name__}})
            return None
    except Exception as exc:  # noqa: BLE001
        _log.warning("layer2 lookup failed",
                     extra={"fields": {"error": type(exc).__name__}})
        return None

    # Never cache a failure - a transient DNS blip would otherwise be pinned
    # for the whole TTL.
    if cache is not None and getattr(intel, "network_performed", False):
        try:
            cache.put(url, intel.to_cache(),
                      negative=not intel.dns.dns_resolves)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            # A silently failing cache is invisible otherwise: every lookup
            # would quietly become a live query.
            _log.warning("layer2 cache write failed",
                         extra={"fields": {"error": type(exc).__name__}})
    return intel


def get_layer2_dict(
    url: str,
    *,
    cache: Layer2Cache | None = None,
    config: DetectorConfig = DEFAULT_CONFIG,
    do_domain_age: bool = True,
    do_reverse_dns: bool = True,
    analyze_domain_fn=None,
) -> dict | None:
    """Return the Layer 2 *feature* dict for `url` - what the ML pipeline wants.

    Thin wrapper over `get_layer2_intel`; a Layer 2 failure becomes a dict with
    `_layer2_error` rather than an exception, so feature extraction continues.
    """
    try:
        intel = get_layer2_intel(
            url, cache=cache, config=config, do_domain_age=do_domain_age,
            do_reverse_dns=do_reverse_dns, analyze_domain_fn=analyze_domain_fn,
        )
    except Exception as exc:  # noqa: BLE001
        return {"_layer2_error": f"{type(exc).__name__}: {exc}"}

    if intel is None:
        return {"_layer2_error": "layer2-unavailable"}
    return intel.as_dict()
