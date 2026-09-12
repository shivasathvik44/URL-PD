"""Layer 2 must finish in bounded time, even when the network does not.

Observed before these fixes: a resolving domain took ~3 s, but an unresolvable
one took ~80 s. The probes each had their own timeout but ran in sequence, so
the individual limits multiplied instead of bounding the total - and the most
expensive probe (TLS) ran even when DNS had already reported no address.

Every test here uses a fake resolver that sleeps, so the timings are
deterministic and no real DNS is involved.
"""

from __future__ import annotations

import time
from dataclasses import replace

import dns.exception
import dns.resolver
import pytest

from conftest import FakeAnswer, FakeResolver
from detector.config import DEFAULT_CONFIG
from detector.dns_features import resolve_domain
from detector.domain_intelligence import analyze_domain
from detector.security import UnsafeTargetError, _resolve_host
from detector.tls_features import TLSResult


class SlowResolver:
    """A resolver that burns `delay` seconds per query, then raises `error`.

    Models the expensive case: a resolver that is reachable but not answering,
    so every query costs its full lifetime instead of failing fast.
    """

    def __init__(self, delay: float = 0.2, error=None):
        self.delay = delay
        self.error = error or dns.exception.Timeout()
        self.calls: list[tuple[str, str]] = []

    def resolve(self, qname, rdtype, *args, **kwargs):
        self.calls.append((str(qname), str(rdtype)))
        time.sleep(self.delay)
        raise self.error


class NxdomainResolver:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls: list[tuple[str, str]] = []

    def resolve(self, qname, rdtype, *args, **kwargs):
        self.calls.append((str(qname), str(rdtype)))
        time.sleep(self.delay)
        raise dns.resolver.NXDOMAIN


# ===========================================================================
# DNS: stop asking a resolver that is not answering
# ===========================================================================
def test_unresponsive_resolver_is_not_asked_four_times():
    """A resolver that failed A and AAAA will fail NS and MX identically.

    Two more queries would burn two more full lifetimes to learn the same
    thing, which is most of the worst-case Layer 2 latency.
    """
    resolver = SlowResolver(delay=0.1)
    result = resolve_domain("dead.example.com", "dead.example.com",
                            resolver=resolver, do_reverse_dns=False)

    rdtypes = [rdtype for _name, rdtype in resolver.calls]
    assert rdtypes == ["A", "AAAA"], f"expected to stop after A/AAAA, got {rdtypes}"
    assert result.dns_resolves is False
    assert "resolver unresponsive" in result.errors["NS"]
    assert "resolver unresponsive" in result.errors["MX"]


def test_nxdomain_short_circuits_ns_and_mx():
    """NXDOMAIN means the name does not exist - NS/MX cannot exist either."""
    resolver = NxdomainResolver()
    result = resolve_domain("gone.example.com", "gone.example.com",
                            resolver=resolver, do_reverse_dns=False)

    rdtypes = [rdtype for _name, rdtype in resolver.calls]
    assert rdtypes == ["A", "AAAA"]
    assert "NXDOMAIN" in result.errors["NS"]


def test_subdomain_nxdomain_still_queries_the_registered_domain():
    """NXDOMAIN on a subdomain says nothing about the registered domain, so
    the NS/MX queries must still happen - the short-circuit is narrow."""
    resolver = NxdomainResolver()
    result = resolve_domain("typo.example.com", "example.com",
                            resolver=resolver, do_reverse_dns=False)

    rdtypes = [rdtype for _name, rdtype in resolver.calls]
    assert rdtypes == ["A", "AAAA", "NS", "MX"]
    assert result.errors["NS"] == "NXDOMAIN"      # asked, and answered NXDOMAIN


def test_a_record_success_still_queries_everything(fake_answer):
    """The short-circuit must not fire when DNS is working normally."""
    resolver = FakeResolver({"A": FakeAnswer(["93.184.216.34"], ttl=300)})
    result = resolve_domain("example.com", "example.com",
                            resolver=resolver, do_reverse_dns=False)

    rdtypes = [rdtype for _name, rdtype in resolver.calls]
    assert rdtypes == ["A", "AAAA", "NS", "MX"]
    assert result.dns_resolves is True


def test_ipv6_only_domain_is_not_short_circuited(fake_answer):
    """AAAA succeeding must keep NS/MX alive even though A failed."""
    resolver = FakeResolver({"AAAA": FakeAnswer(["2606:2800:220:1:248:1893:25c8:1946"])})
    result = resolve_domain("v6.example.com", "v6.example.com",
                            resolver=resolver, do_reverse_dns=False)

    rdtypes = [rdtype for _name, rdtype in resolver.calls]
    assert rdtypes == ["A", "AAAA", "NS", "MX"]
    assert result.dns_resolves is True


# ===========================================================================
# TLS: never handshake with a host that has no address
# ===========================================================================
def test_tls_is_skipped_when_dns_found_no_address():
    """The regression that dominated the ~80 s path: TLS ran anyway, costing a
    name resolution plus up to three connection attempts, to learn nothing."""
    called = []

    def tls_fn(host, **kwargs):
        called.append(host)
        return TLSResult(host=host, checked=True)

    intel = analyze_domain(
        "dead.example.com", resolver=NxdomainResolver(),
        do_domain_age=False, do_tls=True, tls_fn=tls_fn,
    )

    assert called == [], "TLS must not run for a domain that does not resolve"
    assert intel.tls is not None
    assert "does not resolve" in intel.tls.tls_error
    assert any("tls-skipped" in n for n in intel.notes)


def test_tls_still_runs_when_dns_resolves(fake_answer):
    called = []

    def tls_fn(host, **kwargs):
        called.append(host)
        return TLSResult(host=host, checked=True, tls_available=True, cert_valid=True)

    intel = analyze_domain(
        "example.com",
        resolver=FakeResolver({"A": FakeAnswer(["93.184.216.34"])}),
        do_domain_age=False, do_tls=True, tls_fn=tls_fn,
    )

    assert called == ["example.com"]
    assert intel.tls.cert_valid is True


# ===========================================================================
# the overall Layer 2 budget
# ===========================================================================
def test_budget_skips_remaining_probes_instead_of_running_over():
    """Per-probe timeouts multiply; the budget is what bounds the total."""
    age_calls = []

    def slow_age(domain, config, **kwargs):
        age_calls.append(domain)
        raise AssertionError("domain age must not run once the budget is spent")

    cfg = replace(DEFAULT_CONFIG, layer2_budget=0.15)
    intel = analyze_domain(
        "slow.example.com", cfg,
        resolver=SlowResolver(delay=0.12), do_domain_age=True, do_tls=False,
    )

    assert age_calls == []
    assert "budget" in (intel.age.error or "")
    assert any("budget" in n for n in intel.notes)


def test_zero_budget_disables_the_deadline(fake_answer):
    """A budget of 0 means "no overall deadline" - used by training runs."""
    cfg = replace(DEFAULT_CONFIG, layer2_budget=0.0)
    intel = analyze_domain(
        "example.com", cfg,
        resolver=FakeResolver({"A": FakeAnswer(["93.184.216.34"])}),
        do_domain_age=False, do_tls=False,
    )
    assert intel.dns.dns_resolves is True
    assert not any("budget" in n for n in intel.notes)


def test_unresolvable_domain_analysis_is_bounded():
    """The headline regression: an unavailable domain must not take ~80 s.

    With a 0.1 s-per-query fake resolver the real bound is dominated by query
    count, so this asserts the *shape* of the fix (two queries, no TLS, no
    RDAP) rather than a wall-clock number that would be flaky in CI.
    """
    resolver = SlowResolver(delay=0.1)
    started = time.perf_counter()
    intel = analyze_domain(
        "unavailable.example.com", replace(DEFAULT_CONFIG, layer2_budget=2.0),
        resolver=resolver, do_domain_age=False, do_tls=True,
        tls_fn=lambda host, **kw: pytest.fail("TLS must not run"),
    )
    elapsed = time.perf_counter() - started

    assert len(resolver.calls) == 2          # not 4
    assert intel.dns.dns_resolves is False
    assert elapsed < 1.0


# ===========================================================================
# name resolution in the SSRF guard is bounded too
# ===========================================================================
def test_name_resolution_has_a_hard_timeout(monkeypatch):
    """`socket.getaddrinfo` is a blocking C call with no timeout parameter, so
    a hung resolver would otherwise stall the fetcher indefinitely."""
    import detector.security as security

    def hanging_getaddrinfo(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr(security.socket, "getaddrinfo", hanging_getaddrinfo)

    started = time.perf_counter()
    with pytest.raises(UnsafeTargetError) as exc:
        _resolve_host("slow.example.com", 443, timeout=0.3)
    elapsed = time.perf_counter() - started

    assert "exceeded" in str(exc.value)
    assert elapsed < 2.0


def test_resolution_failure_is_reported_not_hung(monkeypatch):
    import socket as socket_module

    import detector.security as security

    def failing(*args, **kwargs):
        raise socket_module.gaierror("Name or service not known")

    monkeypatch.setattr(security.socket, "getaddrinfo", failing)

    with pytest.raises(UnsafeTargetError) as exc:
        _resolve_host("nope.example.com", 443, timeout=1.0)
    assert "does not resolve" in str(exc.value)
