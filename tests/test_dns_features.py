"""Layer 2 tests - DNS features, IP intelligence, reverse DNS.

Every network-dependent test uses a FakeResolver (see conftest.py); nothing here
touches the real network, so the suite is deterministic.
"""

from __future__ import annotations

import dns.exception
import dns.resolver
import pytest

from detector.config import DetectorConfig
from detector.dns_features import DNSResult, classify_ip, resolve_domain


# --------------------------------------------------------------------------
# IP intelligence (local only)
# --------------------------------------------------------------------------
# Note: the stdlib treats loopback as a subset of "private", so a loopback
# address reports is_private=True as well.
@pytest.mark.parametrize("ip,version,public,private,loopback", [
    ("8.8.8.8", 4, True, False, False),
    ("1.1.1.1", 4, True, False, False),
    ("192.168.1.10", 4, False, True, False),
    ("10.0.0.5", 4, False, True, False),
    ("172.16.0.1", 4, False, True, False),
    ("127.0.0.1", 4, False, True, True),
    ("2606:4700:4700::1111", 6, True, False, False),
    ("fd00::1", 6, False, True, False),
    ("::1", 6, False, True, True),
])
def test_classify_ip(ip, version, public, private, loopback):
    info = classify_ip(ip)
    assert info["valid"] is True
    assert info["version"] == version
    assert info["is_public"] is public
    assert info["is_private"] is private
    assert info["is_loopback"] is loopback


def test_classify_ip_multicast_and_reserved():
    assert classify_ip("224.0.0.1")["is_multicast"] is True
    assert classify_ip("240.0.0.1")["is_reserved"] is True
    assert classify_ip("not-an-ip")["valid"] is False
    assert classify_ip("not-an-ip")["version"] is None


# --------------------------------------------------------------------------
# 1. a legitimate domain that resolves (A + AAAA + NS + MX)
# --------------------------------------------------------------------------
def test_legit_domain_resolves(make_resolver, fake_answer, fake_mx):
    r = make_resolver({
        "A": fake_answer(["93.184.216.34"], ttl=3600),
        "AAAA": fake_answer(["2606:2800:220:1:248:1893:25c8:1946"], ttl=3600),
        "NS": fake_answer(["a.iana-servers.net.", "b.iana-servers.net."]),
        "MX": fake_answer([fake_mx("mail.example.com.")]),
        "PTR": lambda q: fake_answer(["example-host.example.net."]),
    })
    res = resolve_domain("example.com", "example.com", resolver=r)

    assert isinstance(res, DNSResult)
    assert res.queried is True
    assert res.dns_resolves is True
    assert res.ipv4_count == 1 and res.has_ipv4 is True
    assert res.ipv6_count == 1 and res.has_ipv6 is True
    assert res.ip_count == 2
    assert res.has_ns is True and res.ns_count == 2
    assert res.nameservers == ("a.iana-servers.net", "b.iana-servers.net")
    assert res.has_mx is True and res.mx_count == 1
    assert res.mx_hosts == ("mail.example.com",)
    assert res.ttl == 3600 and res.ttl_available is True
    assert res.public_ip_count == 2 and res.private_ip_count == 0
    assert res.reverse_dns_available is True
    assert "example-host.example.net" in res.reverse_dns_names
    assert res.errors == {}


# --------------------------------------------------------------------------
# 2. a domain that does not resolve (NXDOMAIN)
# --------------------------------------------------------------------------
def test_domain_does_not_resolve(make_resolver):
    r = make_resolver({
        "A": dns.resolver.NXDOMAIN,
        "AAAA": dns.resolver.NXDOMAIN,
        "NS": dns.resolver.NXDOMAIN,
        "MX": dns.resolver.NXDOMAIN,
    })
    res = resolve_domain("no-such-domain-xyzzy.example", resolver=r)

    assert res.queried is True
    assert res.dns_resolves is False        # a feature, NOT "phishing"
    assert res.ipv4_count == 0 and res.ipv6_count == 0
    assert res.has_ns is False and res.has_mx is False
    assert set(res.errors) == {"A", "AAAA", "NS", "MX"}
    assert res.errors["A"] == "NXDOMAIN"


# --------------------------------------------------------------------------
# 3 & 4. IPv4-only and IPv6-only
# --------------------------------------------------------------------------
def test_ipv4_only(make_resolver, fake_answer):
    r = make_resolver({"A": fake_answer(["203.0.113.7", "203.0.113.8"])})
    res = resolve_domain("v4.example", resolver=r, do_reverse_dns=False)
    assert res.has_ipv4 is True and res.ipv4_count == 2
    assert res.has_ipv6 is False and res.ipv6_count == 0
    assert res.dns_resolves is True


def test_ipv6_only(make_resolver, fake_answer):
    r = make_resolver({"AAAA": fake_answer(["2001:db8::1"])})
    res = resolve_domain("v6.example", resolver=r, do_reverse_dns=False)
    assert res.has_ipv6 is True and res.ipv6_count == 1
    assert res.has_ipv4 is False
    assert res.dns_resolves is True
    # 2001:db8::/32 is documentation space -> not public
    assert res.public_ip_count == 0


# --------------------------------------------------------------------------
# 5 & 6. NS present / MX present
# --------------------------------------------------------------------------
def test_ns_records_present(make_resolver, fake_answer):
    r = make_resolver({"NS": fake_answer(["ns1.host.test.", "ns2.host.test."])})
    res = resolve_domain("hasns.example", resolver=r, do_reverse_dns=False)
    assert res.has_ns is True and res.ns_count == 2


def test_mx_records_present(make_resolver, fake_answer, fake_mx):
    r = make_resolver({"MX": fake_answer([fake_mx("alt1.mail.test."), fake_mx("alt2.mail.test.")])})
    res = resolve_domain("hasmx.example", resolver=r, do_reverse_dns=False)
    assert res.has_mx is True and res.mx_count == 2
    assert res.mx_hosts == ("alt1.mail.test", "alt2.mail.test")


def test_no_mx_is_just_a_feature(make_resolver, fake_answer):
    r = make_resolver({"A": fake_answer(["203.0.113.9"])})   # no MX key -> NoAnswer
    res = resolve_domain("nomx.example", resolver=r, do_reverse_dns=False)
    assert res.has_mx is False and res.mx_count == 0
    assert res.errors.get("MX") == "NoAnswer"


# --------------------------------------------------------------------------
# 10. DNS timeout / failure handling
# --------------------------------------------------------------------------
def test_dns_timeout_is_recorded_not_raised(make_resolver):
    r = make_resolver({
        "A": dns.exception.Timeout,
        "AAAA": dns.exception.Timeout,
        "NS": dns.exception.Timeout,
        "MX": dns.exception.Timeout,
    })
    res = resolve_domain("slow.example", resolver=r)     # must not raise
    assert res.dns_resolves is False
    assert res.errors["A"] == "Timeout"


def test_unexpected_resolver_error_is_contained(make_resolver):
    r = make_resolver({"A": RuntimeError("boom")})
    res = resolve_domain("weird.example", resolver=r, do_reverse_dns=False)
    assert res.dns_resolves is False
    assert "RuntimeError" in res.errors["A"]


# --------------------------------------------------------------------------
# reverse DNS is public-only and capped
# --------------------------------------------------------------------------
def test_reverse_dns_skips_private_ips(make_resolver, fake_answer):
    r = make_resolver({
        "A": fake_answer(["192.168.0.1", "10.0.0.1"]),
        "PTR": lambda q: pytest.fail("PTR must not be queried for private IPs"),
    })
    res = resolve_domain("intranet.example", resolver=r)
    assert res.reverse_dns_available is False
    assert res.private_ip_count == 2 and res.public_ip_count == 0


def test_reverse_dns_lookup_cap(make_resolver, fake_answer):
    cfg = DetectorConfig(reverse_dns_max_lookups=1)
    r = make_resolver({
        "A": fake_answer(["8.8.8.8", "8.8.4.4", "1.1.1.1"]),
        "PTR": lambda q: fake_answer(["ptr.example."]),
    })
    res = resolve_domain("many.example", "many.example", config=cfg, resolver=r)
    ptr_calls = [c for c in r.calls if c[1] == "PTR"]
    assert len(ptr_calls) == 1                       # capped
    assert res.reverse_dns_available is True


def test_reverse_dns_failure_is_safe(make_resolver, fake_answer):
    r = make_resolver({
        "A": fake_answer(["8.8.8.8"]),
        "PTR": dns.resolver.NXDOMAIN,
    })
    res = resolve_domain("norev.example", resolver=r)
    assert res.reverse_dns_available is False        # missing PTR != malicious
    assert res.errors.get("PTR") == "NXDOMAIN"


# --------------------------------------------------------------------------
# 9. invalid / malformed inputs to resolve_domain
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "localhost", "not a domain", "..."])
def test_resolve_domain_non_fqdn_inputs(bad):
    res = resolve_domain(bad)                        # no resolver, must not hit network
    assert res.queried is False
    assert res.dns_resolves is False
    assert "query" in res.errors


def test_resolve_domain_ip_literal_is_classified_not_resolved():
    res = resolve_domain("8.8.8.8")
    assert res.queried is False
    assert res.ipv4_addresses == ("8.8.8.8",)
    assert res.public_ip_count == 1
    assert "ip-literal" in res.errors["query"]


def test_feature_dict_shape(make_resolver, fake_answer):
    r = make_resolver({"A": fake_answer(["93.184.216.34"])})
    d = resolve_domain("example.com", "example.com", resolver=r, do_reverse_dns=False).feature_dict()
    for key in ["dns_resolves", "ipv4_count", "ipv6_count", "has_ns", "ns_count",
                "has_mx", "mx_count", "ttl", "ttl_available", "public_ip_count",
                "private_ip_count", "reserved_ip_count", "reverse_dns_available"]:
        assert key in d
