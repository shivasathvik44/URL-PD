"""Layer 2 tests - domain age (RDAP/WHOIS) and the Layer 1 + Layer 2 pipeline.

All providers and resolvers are injected/mocked; no real network I/O.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from detector import analyze
from detector.config import DEFAULT_CONFIG, DetectorConfig
from detector.domain_age import DomainAgeResult, RawAge, lookup_domain_age
from detector.domain_intelligence import DomainIntelligence, analyze_dns, analyze_domain


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


# --------------------------------------------------------------------------
# domain age: provider chain
# --------------------------------------------------------------------------
def test_domain_age_from_rdap_old_domain():
    fake_rdap = lambda d, t: RawAge(
        creation_date=_days_ago(3200),
        expiration_date=_days_ago(-400),
        registrar="Example Registrar LLC",
    )
    res = lookup_domain_age("example.com", DEFAULT_CONFIG, rdap=fake_rdap, whois=None)

    assert res.available is True
    assert res.source == "rdap"
    assert res.registrar == "Example Registrar LLC"
    assert 3190 <= res.age_days <= 3210
    assert res.is_new_domain is False
    assert res.error is None


# --------------------------------------------------------------------------
# 7. a newly registered domain (safe synthetic fixture)
# --------------------------------------------------------------------------
def test_domain_age_new_domain_flagged():
    fake_rdap = lambda d, t: RawAge(creation_date=_days_ago(5))
    res = lookup_domain_age("brand-new.example", DEFAULT_CONFIG, rdap=fake_rdap, whois=None)

    assert res.available is True
    assert res.age_days == 5
    assert res.is_new_domain is True            # <= 30 day threshold


def test_new_domain_threshold_is_configurable():
    cfg = DetectorConfig(new_domain_max_age_days=7)
    fake_rdap = lambda d, t: RawAge(creation_date=_days_ago(10))
    assert lookup_domain_age("d.example", cfg, rdap=fake_rdap, whois=None).is_new_domain is False
    fake_rdap2 = lambda d, t: RawAge(creation_date=_days_ago(3))
    assert lookup_domain_age("d.example", cfg, rdap=fake_rdap2, whois=None).is_new_domain is True


# --------------------------------------------------------------------------
# WHOIS fallback when RDAP yields nothing
# --------------------------------------------------------------------------
def test_whois_fallback_when_rdap_empty():
    rdap = lambda d, t: None
    whois = lambda d, t: RawAge(creation_date=_days_ago(900), registrar="Fallback Registrar")
    res = lookup_domain_age("example.org", DEFAULT_CONFIG, rdap=rdap, whois=whois)
    assert res.source == "whois"
    assert res.registrar == "Fallback Registrar"
    assert res.available is True


# --------------------------------------------------------------------------
# 8. WHOIS/RDAP information unavailable -> feature, not verdict, not crash
# --------------------------------------------------------------------------
def test_domain_age_unavailable_is_safe():
    res = lookup_domain_age(
        "example.com", DEFAULT_CONFIG,
        rdap=lambda d, t: None,
        whois=lambda d, t: None,
    )
    assert res.available is False
    assert res.age_days is None
    assert res.is_new_domain is False           # unknown != "new", unknown != phishing
    assert res.source is None
    assert res.error is not None                # recorded, for transparency


def test_domain_age_provider_exception_is_contained():
    def boom(d, t):
        raise TimeoutError("rate limited")
    res = lookup_domain_age("example.com", DEFAULT_CONFIG, rdap=boom, whois=lambda d, t: None)
    assert res.available is False
    assert "rdap" in res.error


def test_domain_age_no_registered_domain():
    res = lookup_domain_age("localhost", DEFAULT_CONFIG)
    assert res.available is False
    assert res.is_new_domain is False


# --------------------------------------------------------------------------
# analyze_domain / analyze_dns orchestration (resolver injected)
# --------------------------------------------------------------------------
def test_analyze_domain_combines_dns_and_age(monkeypatch, make_resolver, fake_answer, fake_mx):
    r = make_resolver({
        "A": fake_answer(["93.184.216.34"], ttl=1200),
        "AAAA": fake_answer(["2606:2800:220:1:248:1893:25c8:1946"]),
        "NS": fake_answer(["a.iana-servers.net.", "b.iana-servers.net."]),
        "MX": fake_answer([fake_mx("mail.example.com.")]),
        "PTR": lambda q: fake_answer(["node.example-cdn.net."]),
    })
    monkeypatch.setattr(
        "detector.domain_intelligence.lookup_domain_age",
        lambda domain, config: DomainAgeResult(
            source="rdap", available=True, creation_date=_days_ago(4000),
            registrar="R", age_days=4000, is_new_domain=False,
        ),
    )

    intel = analyze_domain("https://example.com/login", resolver=r)
    d = intel.as_dict()

    assert isinstance(intel, DomainIntelligence)
    assert d["domain"] == "example.com"
    assert d["dns_resolves"] is True
    assert d["ipv4_count"] == 1 and d["ipv6_count"] == 1
    assert d["has_ns"] is True and d["ns_count"] == 2
    assert d["has_mx"] is True and d["mx_count"] == 1
    assert d["domain_age_days"] == 4000 and d["is_new_domain"] is False
    assert d["public_ip_count"] == 2
    assert d["reverse_dns_available"] is True
    assert d["ttl"] == 1200

    anomaly = d["anomaly_features"]
    for key in ["dns_resolves", "has_ns", "has_mx", "domain_age_days",
                "is_new_domain", "ipv4_count", "ipv6_count", "public_ip_count",
                "private_ip_count", "reverse_dns_available"]:
        assert key in anomaly


def test_analyze_dns_is_alias(monkeypatch, make_resolver, fake_answer):
    r = make_resolver({"A": fake_answer(["203.0.113.5"])})
    monkeypatch.setattr(
        "detector.domain_intelligence.lookup_domain_age",
        lambda domain, config: DomainAgeResult(source=None, available=False),
    )
    intel = analyze_dns("http://thing.example/path", resolver=r, do_reverse_dns=False)
    assert intel.dns.has_ipv4 is True


# --------------------------------------------------------------------------
# 9. malformed / invalid domain input to Layer 2
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "not a url", "http://", "ht!tp://%%%", None])
def test_layer2_malformed_input_does_not_crash(bad):
    intel = analyze_domain(bad, do_dns=False, do_domain_age=False)
    d = intel.as_dict()
    assert isinstance(d, dict)
    assert d["dns_resolves"] is False
    assert d["is_new_domain"] is False
    assert intel.network_performed is False


def test_layer2_ip_host_skips_domain_age(make_resolver):
    intel = analyze_domain("http://8.8.8.8/login", do_dns=False, do_domain_age=True)
    assert intel.is_ip_host is True
    assert intel.age.available is False           # no WHOIS for a bare IP
    assert intel.dns.dns_resolves is False


# --------------------------------------------------------------------------
# 2 (pipeline level). a non-resolving domain through the full pipeline
# --------------------------------------------------------------------------
def test_pipeline_non_resolving_domain(monkeypatch, make_resolver):
    import dns.resolver
    r = make_resolver({
        "A": dns.resolver.NXDOMAIN, "AAAA": dns.resolver.NXDOMAIN,
        "NS": dns.resolver.NXDOMAIN, "MX": dns.resolver.NXDOMAIN,
    })
    monkeypatch.setattr(
        "detector.domain_intelligence.lookup_domain_age",
        lambda domain, config: DomainAgeResult(source=None, available=False, error="no-data"),
    )
    result = analyze("http://definitely-not-real-4f9x.example/login", resolver=r)

    # Layer 1 still fully populated
    assert result.layer1.hostname == "definitely-not-real-4f9x.example"
    assert result.layer1.keyword_count == 1
    # Layer 2 recorded the failure without deciding anything
    assert result.layer2.dns.dns_resolves is False
    assert result.layer2.age.available is False
    d = result.as_dict()
    assert d["layer1_lexical"]["url_length"] > 0
    assert d["layer2_domain_intel"]["dns_resolves"] is False


# --------------------------------------------------------------------------
# Layer 1 must be untouched by Layer 2
# --------------------------------------------------------------------------
def test_layer1_unchanged_via_pipeline():
    result = analyze("https://paypal.security-check.xyz/login", run_layer2=False)
    l1 = result.layer1.as_dict()
    assert l1["brand_in_wrong_place"] is True
    assert l1["brand_in_wrong_place_name"] == "paypal"
    assert l1["is_suspicious_tld"] is True
    assert l1["keyword_login"] is True
    assert result.layer2 is None


def test_pipeline_layer1_only_does_no_network(make_resolver):
    # run_layer2=False -> resolver never consulted, no exceptions
    result = analyze("example.com", run_layer2=False)
    assert result.layer2 is None
    assert result.layer1.registered_domain == "example.com"
