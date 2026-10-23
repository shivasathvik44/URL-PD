"""Layer 4 SSRF and redirect security regression tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import FakeHttpClient, FakeResponse
from detector.config import DEFAULT_NETWORK_CONFIG
from detector.security import UnsafeTargetError
from detector.web_fetcher import fetch_page


@pytest.mark.parametrize(
    "target_redirect",
    [
        "http://127.0.0.1:8000/admin",
        "http://127.0.0.2:8000/status",
        "http://[::1]:8000/secret",
        "http://localhost:8000/internal",
    ],
)
def test_redirect_to_loopback_destinations_blocked(target_redirect):
    """A public hop redirecting to loopback IPv4/IPv6 must be blocked at the hop."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": target_redirect},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "blocked-by-ssrf-policy" in str(page.fetch_error)
    # Hop 1 was fetched; hop 2 was blocked before any request was issued
    assert client.urls == ["https://93.184.216.34/start"]


@pytest.mark.parametrize(
    "private_dest",
    [
        "http://10.0.0.1/admin",
        "http://10.254.0.1/config",
        "http://172.16.0.1/internal",
        "http://172.31.255.254/status",
        "http://192.168.1.1/router",
        "http://192.168.0.254/gateway",
        "http://100.64.0.1/cgnat",
    ],
)
def test_redirect_to_private_ipv4_destinations_blocked(private_dest):
    """A public hop redirecting to RFC1918 / CGNAT IPv4 addresses must be blocked."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": private_dest},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.urls == ["https://93.184.216.34/start"]


@pytest.mark.parametrize(
    "private_v6_dest",
    [
        "http://[fc00::1]/admin",
        "http://[fd12:3456:789a::1]/api",
        "http://[fe80::1]/local",
    ],
)
def test_redirect_to_private_and_link_local_ipv6_destinations_blocked(private_v6_dest):
    """A public hop redirecting to ULA or link-local IPv6 addresses must be blocked."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": private_v6_dest},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.urls == ["https://93.184.216.34/start"]


@pytest.mark.parametrize(
    "metadata_dest",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
    ],
)
def test_redirect_to_cloud_metadata_destinations_blocked(metadata_dest):
    """A public hop redirecting to cloud metadata endpoints must be blocked."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": metadata_dest},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.urls == ["https://93.184.216.34/start"]


def test_multi_hop_redirect_chain_public_to_private():
    """Multi-hop redirects must validate each step; an internal final target must be blocked."""
    client = FakeHttpClient({
        "https://93.184.216.34/hop1": FakeResponse(
            status_code=302,
            headers={"Location": "https://93.184.216.34/hop2"},
        ),
        "https://93.184.216.34/hop2": FakeResponse(
            status_code=302,
            headers={"Location": "http://192.168.1.1/secret"},
        ),
    })

    page = fetch_page("https://93.184.216.34/hop1", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    # Hop 1 and Hop 2 were requested; Hop 3 was blocked before sending
    assert client.urls == [
        "https://93.184.216.34/hop1",
        "https://93.184.216.34/hop2",
    ]


@pytest.mark.parametrize(
    "bad_redirect",
    [
        "gopher://127.0.0.1:70/",
        "file:///etc/shadow",
        "javascript:alert(1)",
        "http:///nohost",
    ],
)
def test_redirect_to_malformed_and_non_http_targets_blocked(bad_redirect):
    """Redirects to non-http(s) or malformed targets must fail cleanly."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": bad_redirect},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.urls == ["https://93.184.216.34/start"]


def test_dns_resolution_timeout_handled_safely():
    """DNS resolution exceeding the timeout must fail safely as an SSRF refusal."""
    with patch(
        "detector.security._resolve_host",
        side_effect=UnsafeTargetError("name resolution for slow.example exceeded 5s"),
    ):
        page = fetch_page(
            "https://slow.example/login",
            DEFAULT_NETWORK_CONFIG,
            client=FakeHttpClient(FakeResponse()),
        )

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "exceeded 5s" in str(page.blocked_reason)


def test_dns_resolution_failure_handled_safely():
    """Unresolvable hostnames must fail safely as an SSRF refusal."""
    with patch(
        "detector.security._resolve_host",
        side_effect=UnsafeTargetError("host does not resolve: nxdomain.example"),
    ):
        page = fetch_page(
            "https://nxdomain.example/login",
            DEFAULT_NETWORK_CONFIG,
            client=FakeHttpClient(FakeResponse()),
        )

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "host does not resolve" in str(page.blocked_reason)
