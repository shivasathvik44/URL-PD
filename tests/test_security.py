"""SSRF guard, output sanitisation and secret redaction.

The SSRF tests are the ones that matter most in this file: Layer 4 fetches a
URL an attacker chose, so every one of these is a live attack path if the guard
regresses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from detector.config import DEFAULT_NETWORK_CONFIG
from detector.security import (
    UnsafeTargetError,
    check_fetch_target,
    is_public_address,
    redact,
    redact_url,
    sanitize_str_list,
    sanitize_text,
)


# ---------------------------------------------------------------------------
# address classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("address", [
    "127.0.0.1",            # loopback
    "127.0.0.2",            # loopback
    "127.255.255.254",      # loopback
    "0.0.0.0",              # unspecified
    "0.0.0.1",              # current network
    "10.0.0.5",             # RFC1918
    "172.16.4.4",           # RFC1918
    "172.31.255.254",       # RFC1918
    "192.168.1.1",          # RFC1918
    "169.254.169.254",      # cloud metadata - the classic SSRF target
    "169.254.0.1",          # link-local
    "100.64.0.1",           # CGNAT
    "192.0.0.1",            # IETF protocol assignments
    "192.0.2.1",            # TEST-NET-1 documentation
    "198.51.100.1",         # TEST-NET-2 documentation
    "203.0.113.1",          # TEST-NET-3 documentation
    "198.18.0.1",           # benchmarking
    "192.88.99.1",          # 6to4 relay
    "224.0.0.1",            # multicast
    "239.255.255.255",      # multicast
    "240.0.0.1",            # reserved
    "255.255.255.255",      # broadcast
    "::",                   # IPv6 unspecified
    "::1",                  # IPv6 loopback
    "fe80::1",              # IPv6 link-local
    "fe80::dead:beef",      # IPv6 link-local
    "fc00::1",              # IPv6 unique-local
    "fd12:3456:789a::1",    # IPv6 unique-local
    "ff02::1",              # IPv6 multicast
    "ff01::1",              # IPv6 interface-local multicast
    "2001:db8::1",          # IPv6 documentation
    "100::1",               # IPv6 discard
    "::ffff:127.0.0.1",     # IPv4-mapped loopback
    "::ffff:10.0.0.1",      # IPv4-mapped RFC1918
    "::ffff:169.254.169.254", # IPv4-mapped metadata
    "::127.0.0.1",          # IPv4-compatible loopback
    "::10.0.0.1",           # IPv4-compatible private
    "64:ff9b::127.0.0.1",   # NAT64 loopback
    "64:ff9b::10.0.0.1",    # NAT64 private
    "64:ff9b:1::192.168.1.1", # local NAT64 private
    "2002:7f00:0001::",     # 6to4 loopback
    "2002:0a00:0001::",     # 6to4 private
    "2001:0::ffff:7f00:1",  # Teredo loopback
    "not-an-ip",
])
def test_non_public_addresses_rejected(address):
    assert is_public_address(address) is False


@pytest.mark.parametrize("address", [
    "8.8.8.8",
    "1.1.1.1",
    "93.184.216.34",
    "2606:4700::1111",
    "2001:4860:4860::8888",
    "::ffff:8.8.8.8",
    "2002:0808:0808::",
])
def test_public_addresses_accepted(address):
    assert is_public_address(address) is True


# ---------------------------------------------------------------------------
# scheme policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.com/",
    "ftp://example.com/x",
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "dict://localhost:11211/stat",
])
def test_non_http_schemes_refused(url):
    with pytest.raises(UnsafeTargetError) as exc:
        check_fetch_target(url, DEFAULT_NETWORK_CONFIG)
    assert "scheme" in str(exc.value).lower()


def test_empty_and_hostless_urls_refused():
    with pytest.raises(UnsafeTargetError):
        check_fetch_target("", DEFAULT_NETWORK_CONFIG)
    with pytest.raises(UnsafeTargetError):
        check_fetch_target("http:///nohost", DEFAULT_NETWORK_CONFIG)


# ---------------------------------------------------------------------------
# literal-IP targets (no DNS needed - these run offline)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/admin",
    "http://localhost/",                      # single-label internal name
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.1/internal",
    "http://192.168.1.1/router",
    "http://[::1]:9200/_cluster/health",
    "https://0.0.0.0/",
    "http://100.64.1.1/",
    "http://0177.0.0.1/",                     # octal loopback
    "http://0x7f.0.0.1/",                     # hex loopback
    "http://0x7f000001/",                     # hex dword loopback
    "http://2130706433/",                     # integer dword loopback
    "http://127.1/",                          # truncated IPv4
    "http://[::127.0.0.1]/",                  # IPv4-compatible loopback
    "http://[64:ff9b::127.0.0.1]/",           # NAT64 loopback
    "http://[2002:7f00:1::]/",                # 6to4 loopback
    "http://[::ffff:169.254.169.254]/",       # IPv4-mapped metadata
])
def test_internal_targets_refused(url):
    with pytest.raises(UnsafeTargetError):
        check_fetch_target(url, DEFAULT_NETWORK_CONFIG)


def test_public_literal_ip_allowed():
    target = check_fetch_target("https://8.8.8.8/", DEFAULT_NETWORK_CONFIG)
    assert target.host == "8.8.8.8"
    assert target.port == 443
    assert target.addresses == ("8.8.8.8",)
    assert target.pinned_address == "8.8.8.8"


def test_public_literal_ipv6_allowed():
    target = check_fetch_target("https://[2606:4700::1111]:8443/", DEFAULT_NETWORK_CONFIG)
    assert target.host == "2606:4700::1111"
    assert target.port == 8443
    assert target.addresses == ("2606:4700::1111",)
    assert target.pinned_address == "2606:4700::1111"


def test_alternative_numeric_public_ip_rejected_as_non_canonical():
    """Even if an alternative numeric IP encodes a public IP, it must be rejected."""
    with pytest.raises(UnsafeTargetError) as exc:
        check_fetch_target("http://0x08080808/", DEFAULT_NETWORK_CONFIG)
    assert "non-canonical" in str(exc.value) or "refusing" in str(exc.value)


def test_mixed_dns_answers_rejected():
    """A hostname resolving to both public and private addresses must be rejected."""
    from unittest.mock import patch

    with patch("detector.security._resolve_host", return_value=["93.184.216.34", "127.0.0.1"]):
        with pytest.raises(UnsafeTargetError) as exc:
            check_fetch_target("https://mixed-dns.example/", DEFAULT_NETWORK_CONFIG)
        assert "non-public" in str(exc.value)

    with patch("detector.security._resolve_host", return_value=["93.184.216.34", "::1"]):
        with pytest.raises(UnsafeTargetError) as exc:
            check_fetch_target("https://mixed-v6-dns.example/", DEFAULT_NETWORK_CONFIG)
        assert "non-public" in str(exc.value)


def test_idna_hostname_normalized():
    """Internationalized domain names must be normalized via IDNA to ASCII punycode."""
    from unittest.mock import patch

    with patch("detector.security._resolve_host", return_value=["93.184.216.34"]) as mock_resolve:
        target = check_fetch_target("https://exämple.com/", DEFAULT_NETWORK_CONFIG)
        assert target.host == "xn--exmple-cua.com"
        assert target.pinned_address == "93.184.216.34"
        mock_resolve.assert_called_once_with("xn--exmple-cua.com", 443)


@pytest.mark.parametrize("bad_port_url", [
    "http://example.com:0/",
    "http://example.com:65536/",
    "http://example.com:70000/",
])
def test_invalid_ports_refused(bad_port_url):
    with pytest.raises(UnsafeTargetError):
        check_fetch_target(bad_port_url, DEFAULT_NETWORK_CONFIG)


def test_explicitly_blocked_hostname_refused():
    with pytest.raises(UnsafeTargetError) as exc:
        check_fetch_target("http://metadata.google.internal/", DEFAULT_NETWORK_CONFIG)
    assert "blocked" in str(exc.value)


def test_single_label_hostnames_refused():
    """`http://jenkins/` resolves through internal search domains."""
    with pytest.raises(UnsafeTargetError) as exc:
        check_fetch_target("http://jenkins/", DEFAULT_NETWORK_CONFIG)
    assert "single-label" in str(exc.value)


def test_userinfo_does_not_smuggle_an_internal_host():
    """`http://example.com@127.0.0.1/` really addresses 127.0.0.1."""
    with pytest.raises(UnsafeTargetError):
        check_fetch_target("http://example.com@127.0.0.1/", DEFAULT_NETWORK_CONFIG)


def test_allow_private_targets_is_opt_in():
    permissive = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)
    target = check_fetch_target("http://127.0.0.1:8000/", permissive)
    assert target.host == "127.0.0.1"
    # ...but the scheme policy still applies even in permissive mode.
    with pytest.raises(UnsafeTargetError):
        check_fetch_target("file:///etc/passwd", permissive)


def test_resolution_failure_is_refused_not_fetched():
    with pytest.raises(UnsafeTargetError):
        check_fetch_target(
            "http://nonexistent-host-for-tests.invalid/", DEFAULT_NETWORK_CONFIG
        )


# ---------------------------------------------------------------------------
# sanitisation
# ---------------------------------------------------------------------------
def test_sanitize_strips_control_characters():
    hostile = "safe\x1b[31m text\x00with\x07controls\r\nfake log line"
    clean = sanitize_text(hostile)
    assert "\x1b" not in clean
    assert "\x00" not in clean
    assert "\n" not in clean and "\r" not in clean
    assert "safe" in clean


def test_sanitize_truncates_with_ellipsis():
    clean = sanitize_text("x" * 1000, max_chars=50)
    assert len(clean) == 50
    assert clean.endswith("…")


def test_sanitize_handles_none_and_non_strings():
    assert sanitize_text(None) == ""
    assert sanitize_text(12345) == "12345"


def test_sanitize_str_list_bounds_items_and_length():
    huge = [f"tactic number {i} " + "y" * 500 for i in range(500)]
    out = sanitize_str_list(huge, max_items=3, max_chars=20)
    assert len(out) == 3
    assert all(len(item) <= 20 for item in out)


def test_sanitize_str_list_drops_non_strings():
    out = sanitize_str_list(["ok", {"nested": "object"}, ["list"], None, 42],
                            max_items=10, max_chars=40)
    assert "ok" in out
    assert all(isinstance(item, str) for item in out)


def test_sanitize_str_list_rejects_non_list():
    assert sanitize_str_list("not a list", max_items=3, max_chars=10) == ()
    assert sanitize_str_list(None, max_items=3, max_chars=10) == ()


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
def test_redact_removes_api_keys():
    text = "using key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA for the call"
    assert "sk-ant" not in redact(text)
    assert "REDACTED" in redact(text)


def test_redact_removes_inline_secrets():
    assert "hunter2" not in redact("password=hunter2")
    assert "abc123xyz" not in redact("token: abc123xyz")


def test_redact_url_strips_credential_query_values():
    url = "https://phish.example/login?email=victim@corp.com&password=secret123&next=/home"
    out = redact_url(url)
    assert "secret123" not in out
    assert "REDACTED" in out
    assert "phish.example" in out          # the useful part survives


def test_redact_url_strips_userinfo():
    out = redact_url("https://user:pass@evil.example/path")
    assert "pass" not in out
    assert "evil.example" in out


def test_redact_url_survives_garbage():
    assert redact_url("not a url at all") != ""
    assert redact_url("") == ""
