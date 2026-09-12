"""Both caches: versioning, TTL, and what must never be cached.

A stale cache is worse than no cache here - a cached "safe" verdict for a
domain that has since been compromised, or a cached payload reinterpreted
against a changed schema, both produce confident wrong answers.
"""

from __future__ import annotations

import time

from conftest import FakeResolver
from detector.domain_intelligence import DomainIntelligence, analyze_domain
from detector.layer2_cache import PAYLOAD_VERSION, Layer2Cache, cache_key, get_layer2_intel
from detector.llm_analyzer import LLMAnalysis
from detector.page_cache import PageCache, url_key


# ===========================================================================
# Layer 2 cache - domain level
# ===========================================================================
def test_cache_key_is_the_registered_domain():
    assert cache_key("https://a.example.com/one") == "example.com"
    assert cache_key("https://b.example.com/two") == "example.com"
    assert cache_key("https://other.org/") == "other.org"


def test_roundtrip_is_lossless():
    """`as_dict()` drops ip_details; the cache must not."""
    resolver = FakeResolver()
    intel = analyze_domain("example.com", resolver=resolver, do_domain_age=False)

    with Layer2Cache(":memory:") as cache:
        cache.put("https://example.com/x", intel.to_cache())
        restored = DomainIntelligence.from_cache(cache.get("https://example.com/y"))

    assert restored.registered_domain == intel.registered_domain
    assert restored.dns.queried == intel.dns.queried
    assert restored.dns.ip_details == intel.dns.ip_details
    assert restored.dns.errors == intel.dns.errors


def test_hit_and_miss_counters():
    with Layer2Cache(":memory:") as cache:
        assert cache.get("https://example.com/") is None
        cache.put("https://example.com/", {"registered_domain": "example.com"})
        assert cache.get("https://example.com/") is not None
        assert cache.hits == 1
        assert cache.misses == 1


def test_different_payload_version_is_not_reused():
    """A row written by an older Layer 2 must not be reinterpreted."""
    with Layer2Cache(":memory:") as cache:
        cache.put("https://example.com/", {"registered_domain": "example.com"})
        cache._conn.execute("UPDATE layer2 SET payload_version = 999")
        cache._conn.commit()

        assert cache.get("https://example.com/") is None
        assert cache.stale == 1


def test_expired_rows_are_not_reused():
    with Layer2Cache(":memory:", ttl_seconds=60) as cache:
        cache.put("https://example.com/", {"registered_domain": "example.com"})
        cache._conn.execute("UPDATE layer2 SET fetched_at = ?", (time.time() - 3600,))
        cache._conn.commit()

        assert cache.get("https://example.com/") is None
        assert cache.stale == 1


def test_zero_ttl_never_expires():
    with Layer2Cache(":memory:", ttl_seconds=0) as cache:
        cache.put("https://example.com/", {"registered_domain": "example.com"})
        cache._conn.execute("UPDATE layer2 SET fetched_at = ?", (time.time() - 10**7,))
        cache._conn.commit()
        assert cache.get("https://example.com/") is not None


def test_purge_stale_removes_old_versions():
    with Layer2Cache(":memory:") as cache:
        cache.put("https://a.com/", {"x": 1})
        cache.put("https://b.com/", {"x": 2})
        cache._conn.execute("UPDATE layer2 SET payload_version = 1 WHERE domain='a.com'")
        cache._conn.commit()

        removed = cache.purge_stale() if PAYLOAD_VERSION != 1 else 0
        assert removed == 1
        assert cache.row_count() == 1


def test_cache_avoids_a_second_lookup():
    calls = []

    def counting_layer2(url, config, **kwargs):
        calls.append(url)
        return analyze_domain(url, config, resolver=FakeResolver(), do_domain_age=False)

    with Layer2Cache(":memory:") as cache:
        first = get_layer2_intel("https://a.example.com/1", cache=cache,
                                 analyze_domain_fn=counting_layer2)
        second = get_layer2_intel("https://b.example.com/2", cache=cache,
                                  analyze_domain_fn=counting_layer2)

    assert first is not None and second is not None
    # Same registered domain -> one lookup, served twice.
    assert len(calls) == 1


def test_failed_lookup_is_not_cached():
    """A transient DNS failure must not be pinned for the TTL."""
    def failing_layer2(url, config, **kwargs):
        raise ConnectionError("resolver down")

    with Layer2Cache(":memory:") as cache:
        result = get_layer2_intel("https://example.com/", cache=cache,
                                  analyze_domain_fn=failing_layer2)
        assert result is None
        assert cache.row_count() == 0


# ===========================================================================
# Layer 4 cache - URL level
# ===========================================================================
def test_page_cache_key_is_hashed_not_the_raw_url():
    """Raw URLs can carry a victim's email or a session token."""
    key = url_key("https://phish.example/login?email=victim@corp.com")
    assert "victim" not in key
    assert "phish" not in key
    assert len(key) == 64


def test_page_cache_is_url_level_not_domain_level():
    with PageCache(":memory:") as cache:
        cache.put("https://site.com/a", LLMAnalysis(performed=True, risk_level="high"))
        assert cache.get("https://site.com/a") is not None
        assert cache.get("https://site.com/b") is None      # different page


def test_page_cache_roundtrip_preserves_fields():
    analysis = LLMAnalysis(
        performed=True, brand_claimed="PayPal", brand_impersonation=True,
        social_engineering_tactics=("urgency",), suspicious_observations=("odd form",),
        risk_level="high", confidence="high", llm_likelihood=0.95,
        prompt_injection_attempt=True, model="test-model",
    )
    with PageCache(":memory:") as cache:
        cache.put("https://site.com/x", analysis)
        restored = cache.get("https://site.com/x")

    assert restored.brand_claimed == "PayPal"
    assert restored.brand_impersonation is True
    assert restored.social_engineering_tactics == ("urgency",)
    assert restored.risk_level == "high"
    assert restored.llm_likelihood == 0.95
    assert restored.prompt_injection_attempt is True


def test_failed_analysis_is_never_cached():
    """One LLM outage must not pin 'we could not check' for hours."""
    with PageCache(":memory:") as cache:
        cache.put("https://site.com/x",
                  LLMAnalysis(performed=False, error="llm-unavailable"))
        assert cache.row_count() == 0
        assert cache.get("https://site.com/x") is None


def test_page_cache_ttl_expiry():
    with PageCache(":memory:", ttl_seconds=60) as cache:
        cache.put("https://site.com/x", LLMAnalysis(performed=True, risk_level="low"))
        cache._conn.execute("UPDATE page_analysis SET analysed_at = ?",
                            (time.time() - 3600,))
        cache._conn.commit()

        assert cache.get("https://site.com/x") is None
        assert cache.stale == 1


def test_page_cache_version_mismatch_ignored():
    with PageCache(":memory:") as cache:
        cache.put("https://site.com/x", LLMAnalysis(performed=True, risk_level="low"))
        cache._conn.execute("UPDATE page_analysis SET analysis_version = 999")
        cache._conn.commit()

        assert cache.get("https://site.com/x") is None
        assert cache.stale == 1


# ===========================================================================
# negative results expire sooner than positive ones
# ===========================================================================
def test_negative_rows_use_the_shorter_ttl():
    """A "does not resolve" answer is far less durable than a positive one.

    It is produced equally by an unregistered domain, a takedown, and our own
    resolver having a bad minute - so caching it for a full day would pin a
    transient outage as fact.
    """
    with Layer2Cache(":memory:", ttl_seconds=86400, negative_ttl_seconds=60) as cache:
        cache.put("https://gone.example/", {"dns_resolves": False})
        cache._conn.execute("UPDATE layer2 SET fetched_at = ?", (time.time() - 300,))
        cache._conn.commit()

        assert cache.get("https://gone.example/") is None      # expired at 60 s
        assert cache.stale == 1


def test_positive_rows_keep_the_long_ttl():
    with Layer2Cache(":memory:", ttl_seconds=86400, negative_ttl_seconds=60) as cache:
        cache.put("https://live.example/", {"dns_resolves": True})
        cache._conn.execute("UPDATE layer2 SET fetched_at = ?", (time.time() - 300,))
        cache._conn.commit()

        assert cache.get("https://live.example/") is not None  # still inside 86400 s


def test_negative_flag_is_derived_from_the_nested_cache_shape():
    """`to_cache()` nests dns_resolves; the flat feature view does not. Reading
    only one shape would mark every row of the other as negative."""
    with Layer2Cache(":memory:") as cache:
        cache.put("https://a.example/", {"dns": {"dns_resolves": True}})
        cache.put("https://b.example/", {"dns": {"dns_resolves": False}})
        rows = dict(cache._conn.execute("SELECT domain, negative FROM layer2"))

    assert rows["a.example"] == 0
    assert rows["b.example"] == 1
