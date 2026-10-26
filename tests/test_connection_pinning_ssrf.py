"""Comprehensive regression tests for DNS rebinding, connection pinning, and SSRF (SEC-01).

Guarantees established by this suite:
  1. DNS rebinding prevention: outbound network sockets are bound directly to the IP
     address that passed SSRF validation, preventing TOCTOU races between validation and connect.
  2. The transport refuses unpinned connections and never silently falls back to ambient DNS.
  3. Intermediate redirect responses release socket resources cleanly.
  4. Redirect loops (self-referential or cyclical) terminate immediately.
  5. Per-hop re-validation ensures redirects to private, loopback, or metadata endpoints fail closed.
  6. TLS certificate verification, SNI, and HTTP Host virtual hosting are preserved under IP pinning.
  7. Environment and explicit HTTP proxies cannot bypass connection pinning.
"""

from __future__ import annotations

import os
import socket
import ssl
import subprocess
import tempfile
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
import requests

from conftest import FakeHttpClient, FakeResponse
from detector.config import DEFAULT_NETWORK_CONFIG
from detector.security import UnsafeTargetError
from detector.tls_features import inspect_tls
from detector.web_fetcher import (
    PINNED_IP_HEADER,
    PinnedHTTPAdapter,
    PinnedHTTPConnection,
    PinnedHTTPSConnection,
    _default_client,
    fetch_page,
)


# ---------------------------------------------------------------------------
# Test Fixtures: Local HTTP and HTTPS servers
# ---------------------------------------------------------------------------
class _RecordingHTTPHandler(BaseHTTPRequestHandler):
    """Local HTTP handler recording received requests, Host headers, and paths."""

    received: list[dict] = []

    def do_GET(self):
        type(self).received.append({
            "path": self.path,
            "headers": dict(self.headers),
            "client_address": self.client_address,
        })
        if self.path == "/redirect-to-target":
            self.send_response(302)
            self.send_header("Location", "/final-page")
            self.end_headers()
            return
        if self.path == "/redirect-loop-self":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop-self")
            self.end_headers()
            return
        if self.path == "/redirect-loop-a":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop-b")
            self.end_headers()
            return
        if self.path == "/redirect-loop-b":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop-a")
            self.end_headers()
            return
        if self.path == "/final-page":
            body = b"<html><head><title>Success</title></head><body>OK</body></html>"
        else:
            body = b"<html><head><title>Root</title></head><body>Hello</body></html>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def recording_http_server():
    _RecordingHTTPHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", _RecordingHTTPHandler.received
    server.shutdown()
    server.server_close()


@pytest.fixture
def recording_https_server():
    """Generates test certs and runs a local TLS server."""
    with tempfile.TemporaryDirectory() as td:
        key_path = os.path.join(td, "key.pem")
        cert_path = os.path.join(td, "cert.pem")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path, "-days", "1",
            "-nodes", "-subj", "/CN=test.example.com",
            "-addext", "subjectAltName=DNS:test.example.com",
        ]
        subprocess.run(  # noqa: S603, S607
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        received = []

        class _RecordingHTTPSHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append({
                    "path": self.path,
                    "headers": dict(self.headers),
                    "client_address": self.client_address,
                })
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                body = b"<html><head><title>Secure Page</title></head><body>OK</body></html>"
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_path, key_path)

        server = HTTPServer(("127.0.0.1", 0), _RecordingHTTPSHandler)
        server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)
        port = server.server_address[1]

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        yield {
            "origin": f"https://test.example.com:{port}",
            "port": port,
            "cert_path": cert_path,
            "received": received,
        }

        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# A. DNS Rebinding and Connection Pinning
# ---------------------------------------------------------------------------
def test_pinned_transport_connects_to_validated_ip(recording_http_server):
    """The transport binds the TCP socket directly to the validated IP."""
    origin, requests_received = recording_http_server
    port = int(origin.split(":")[-1])
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        page = fetch_page(f"http://example.org:{port}/final-page", config)

    assert page.fetched is True
    assert page.status_code == 200
    assert page.peer_address == "127.0.0.1"
    assert len(requests_received) == 1
    assert requests_received[0]["headers"]["Host"] == f"example.org:{port}"


def test_dns_rebinding_cannot_divert_connection_after_validation(recording_http_server):
    """Simulate DNS rebinding: host initially resolves to allowed IP, then changes to prohibited IP.

    The connection must connect only to the validated IP and cannot reach the changed destination.
    """
    origin, requests_received = recording_http_server
    port = int(origin.split(":")[-1])

    # Resolves to 127.0.0.1 on initial check, but would resolve to 192.168.1.1 on subsequent lookup
    dns_records = ["127.0.0.1"]

    def fake_resolve(host, p, timeout=5.0):
        if not dns_records:
            return ["192.168.1.1"]  # Rebound to private network
        return [dns_records.pop(0)]

    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    orig_gai = socket.getaddrinfo
    gai_queried_hosts: list[str] = []

    def fake_gai(host, p, *args, **kwargs):
        gai_queried_hosts.append(host)
        if host == "rebind-attack.example":
            # If connect-time DNS is attempted on the hostname, simulate rebinding to an unreachable address
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.168.1.1", p))
            ]
        return orig_gai(host, p, *args, **kwargs)

    with (
        patch("detector.security._resolve_host", side_effect=fake_resolve),
        patch("socket.getaddrinfo", side_effect=fake_gai),
    ):
        page = fetch_page(f"http://rebind-attack.example:{port}/final-page", config)

    assert page.fetched is True
    assert page.peer_address == "127.0.0.1"
    # The socket connected to the local test server at 127.0.0.1, NOT 192.168.1.1
    assert len(requests_received) == 1
    assert requests_received[0]["client_address"][0] == "127.0.0.1"
    # Ensure connect-time socket creation never queried the hostname through getaddrinfo
    assert "rebind-attack.example" not in gai_queried_hosts


def test_transport_fails_closed_without_pinned_ip():
    """PinnedHTTPConnection and PinnedHTTPSConnection refuse unpinned connections."""
    conn_http = PinnedHTTPConnection("example.com", 80)
    with pytest.raises(UnsafeTargetError) as exc_http:
        conn_http._new_conn()
    assert "Refusing unpinned connection" in str(exc_http.value)

    conn_https = PinnedHTTPSConnection("example.com", 443)
    with pytest.raises(UnsafeTargetError) as exc_https:
        conn_https._new_conn()
    assert "Refusing unpinned connection" in str(exc_https.value)


def test_transport_adapter_refuses_request_missing_pinned_ip():
    """PinnedHTTPAdapter refuses requests that lack pre-validated destination IP."""
    adapter = PinnedHTTPAdapter()
    req = requests.Request("GET", "http://example.com/").prepare()
    # Explicitly lacks PINNED_IP_HEADER
    with pytest.raises(UnsafeTargetError) as exc:
        adapter.get_connection_with_tls_context(req, verify=True)
    assert "Refusing unpinned connection" in str(exc.value)


def test_dns_resolution_failures_fail_closed():
    """Empty results, resolution failures, and timeouts fail closed."""
    with patch("detector.security._resolve_host", side_effect=UnsafeTargetError("host resolved to no addresses")):
        page = fetch_page("https://empty-dns.example/", DEFAULT_NETWORK_CONFIG)
    assert page.fetched is False
    assert "host resolved to no addresses" in str(page.blocked_reason)

    with patch("detector.security._resolve_host", side_effect=UnsafeTargetError("name resolution exceeded 5s")):
        page = fetch_page("https://slow-dns.example/", DEFAULT_NETWORK_CONFIG)
    assert page.fetched is False
    assert "exceeded 5s" in str(page.blocked_reason)


# ---------------------------------------------------------------------------
# B. IPv4 and IPv6 Address Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prohibited_url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.2:8080/",
        "http://127.255.255.254/",
        "http://0.0.0.0/",
        "http://0.0.0.1/",
        "http://10.0.0.1/",
        "http://10.255.255.255/",
        "http://172.16.0.1/",
        "http://172.31.255.254/",
        "http://192.168.0.1/",
        "http://192.168.1.254/",
        "http://169.254.169.254/",
        "http://169.254.1.1/",
        "http://100.64.0.1/",
        "http://192.0.0.1/",
        "http://192.0.2.1/",
        "http://198.18.0.1/",
        "http://198.51.100.1/",
        "http://203.0.113.1/",
        "http://224.0.0.1/",
        "http://240.0.0.1/",
        "http://[::]/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fd12:3456:789a::1]/",
        "http://[fe80::1]/",
        "http://[fe80::dead:beef]/",
        "http://[ff02::1]/",
        "http://[2001:db8::1]/",
        "http://[100::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://[::ffff:192.168.1.1]/",
        "http://[::127.0.0.1]/",
        "http://[::10.0.0.1]/",
        "http://[64:ff9b::127.0.0.1]/",
        "http://[64:ff9b::10.0.0.1]/",
        "http://[64:ff9b:1::192.168.1.1]/",
        "http://[2002:7f00:1::]/",
        "http://[2002:0a00:1::]/",
        "http://[2001:0::ffff:7f00:1]/",
        "http://0177.0.0.1/",
        "http://0x7f.0.0.1/",
        "http://0x7f000001/",
        "http://2130706433/",
        "http://127.1/",
    ],
)
def test_all_prohibited_destinations_blocked(prohibited_url):
    """Every non-global IPv4/IPv6 destination must be rejected before connecting."""
    client = FakeHttpClient(FakeResponse(body="OK"))
    page = fetch_page(prohibited_url, DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.requests == []


def test_mixed_ip_dns_resolution_blocked():
    """If a host resolves to a mixture of public and private addresses, it must fail closed."""
    client = FakeHttpClient(FakeResponse(body="OK"))
    with patch("detector.security._resolve_host", return_value=["93.184.216.34", "127.0.0.1"]):
        page = fetch_page("https://mixed-ip.example/", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "non-public" in str(page.blocked_reason)
    assert client.requests == []


# ---------------------------------------------------------------------------
# C. Redirect Validation
# ---------------------------------------------------------------------------
def test_redirects_between_permitted_destinations(recording_http_server):
    """Valid redirects between permitted destinations succeed and record peer IP."""
    origin, _ = recording_http_server
    port = int(origin.split(":")[-1])
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        page = fetch_page(f"http://example.org:{port}/redirect-to-target", config)

    assert page.fetched is True
    assert page.status_code == 200
    assert page.final_url == f"http://example.org:{port}/final-page"
    assert page.redirect_chain == (f"http://example.org:{port}/final-page",)
    assert page.peer_address == "127.0.0.1"


def test_redirect_to_loopback_and_metadata_blocked_at_hop():
    """A public hop redirecting to internal/loopback destinations is refused at the hop."""
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        ),
    })

    page = fetch_page("https://93.184.216.34/start", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "blocked-by-ssrf-policy" in str(page.fetch_error)
    # Hop 1 was requested; Hop 2 was blocked before being requested
    assert client.urls == ["https://93.184.216.34/start"]


def test_redirect_loop_self_terminates(recording_http_server):
    """An immediate self-redirect terminates as redirect-loop-detected."""
    origin, _ = recording_http_server
    port = int(origin.split(":")[-1])
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    page = fetch_page(f"http://127.0.0.1:{port}/redirect-loop-self", config)

    assert page.fetched is False
    assert "redirect-loop-detected" in str(page.fetch_error)


def test_redirect_cycle_terminates(recording_http_server):
    """A cyclical redirect (A -> B -> A) terminates as redirect-loop-detected."""
    origin, _ = recording_http_server
    port = int(origin.split(":")[-1])
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    page = fetch_page(f"http://127.0.0.1:{port}/redirect-loop-a", config)

    assert page.fetched is False
    assert "redirect-loop-detected" in str(page.fetch_error)


def test_redirect_to_disallowed_scheme_blocked():
    """A redirect to a non-allowed scheme (e.g. gopher://, file://) is refused."""
    client = FakeHttpClient({
        "https://93.184.216.34/hop1": FakeResponse(
            status_code=302,
            headers={"Location": "file:///etc/passwd"},
        ),
    })
    page = fetch_page("https://93.184.216.34/hop1", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert client.urls == ["https://93.184.216.34/hop1"]


def test_https_to_http_redirect_follows_scheme_policy():
    """When only HTTPS is allowed, an HTTP redirect hop is rejected."""
    https_only = replace(DEFAULT_NETWORK_CONFIG, allowed_schemes=("https",))
    client = FakeHttpClient({
        "https://93.184.216.34/start": FakeResponse(
            status_code=302,
            headers={"Location": "http://93.184.216.34/unencrypted"},
        ),
    })
    page = fetch_page("https://93.184.216.34/start", https_only, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "scheme 'http' not allowed" in str(page.blocked_reason)


def test_intermediate_responses_closed_on_redirects():
    """Each intermediate response in a redirect chain must be closed."""
    resp1 = FakeResponse(status_code=302, headers={"Location": "https://93.184.216.34/step2"})
    resp2 = FakeResponse(body="<html><body>Final</body></html>")
    client = FakeHttpClient({
        "https://93.184.216.34/step1": resp1,
        "https://93.184.216.34/step2": resp2,
    })

    page = fetch_page("https://93.184.216.34/step1", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is True
    assert resp1.closed is True
    assert resp2.closed is True


# ---------------------------------------------------------------------------
# D. TLS, SNI, and Host Header Correctness
# ---------------------------------------------------------------------------
def test_tls_sni_host_and_cert_verification_under_pinning(recording_https_server):
    """When pinned to an IP, TLS Server Name Indication, Host, and cert verification are correct."""
    server_info = recording_https_server
    origin = server_info["origin"]
    cert_path = server_info["cert_path"]
    received = server_info["received"]
    port = server_info["port"]

    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    client = _default_client()
    # verify points to the test certificate authority bundle
    client.verify = cert_path

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        page = fetch_page(origin, config, client=client)

    assert page.fetched is True
    assert page.status_code == 200
    assert page.title == "Secure Page"
    assert page.peer_address == "127.0.0.1"
    assert len(received) == 1
    assert received[0]["headers"]["Host"] == f"test.example.com:{port}"


def test_tls_wrong_hostname_fails_verification(recording_https_server):
    """A certificate presented for the wrong hostname causes verification to fail."""
    server_info = recording_https_server
    cert_path = server_info["cert_path"]
    port = server_info["port"]

    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    client = _default_client()
    client.verify = cert_path

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        # Cert is for test.example.com; connecting to wrong-host.example must fail
        page = fetch_page(f"https://wrong-host.example:{port}/", config, client=client)

    assert page.fetched is False
    assert "SSLError" in str(page.fetch_error) or "CERTIFICATE_VERIFY_FAILED" in str(page.fetch_error)


def test_inspect_tls_pinned_to_validated_ip(recording_https_server):
    """Layer 2 inspect_tls connects to the validated pinned IP while preserving server_hostname."""
    server_info = recording_https_server
    port = server_info["port"]
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        result = inspect_tls("test.example.com", port=port, network=config)

    assert result.checked is True
    assert result.tls_available is True
    assert result.tls_version is not None


# ---------------------------------------------------------------------------
# E. Proxy and Resource Limits
# ---------------------------------------------------------------------------
def test_environment_proxies_ignored_by_default():
    """Ambient HTTP_PROXY and HTTPS_PROXY are ignored via session.trust_env = False."""
    client = _default_client()
    assert client.trust_env is False


def test_explicit_proxies_rejected_by_pinned_adapter():
    """PinnedHTTPAdapter refuses requests configured with proxies to prevent pinning bypass."""
    adapter = PinnedHTTPAdapter()
    req = requests.Request("GET", "http://example.com/").prepare()
    req.headers[PINNED_IP_HEADER] = "93.184.216.34"

    with pytest.raises(UnsafeTargetError) as exc:
        adapter.get_connection_with_tls_context(req, verify=True, proxies={"http": "http://proxy.internal:8080"})
    assert "proxies are not permitted" in str(exc.value).lower()


def test_legacy_get_connection_fails_closed():
    """PinnedHTTPAdapter.get_connection refuses unpinned callers lacking PreparedRequest."""
    adapter = PinnedHTTPAdapter()
    with pytest.raises(UnsafeTargetError) as exc:
        adapter.get_connection("http://example.com/")
    assert "Refusing unpinned connection" in str(exc.value)


def test_ambient_proxy_env_vars_ignored(monkeypatch, recording_http_server):
    """Ambient HTTP_PROXY and HTTPS_PROXY environment variables are ignored."""
    origin, received = recording_http_server
    port = int(origin.split(":")[-1])
    config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

    # Point proxy to a non-existent port that would fail if accessed
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    with patch("detector.security._resolve_host", return_value=["127.0.0.1"]):
        page = fetch_page(f"http://example.org:{port}/final-page", config)

    assert page.fetched is True
    assert page.status_code == 200
    assert page.peer_address == "127.0.0.1"


def test_connection_pool_isolation_across_different_pinned_ips():
    """Different pinned IPs for the same host use separate connection pools."""
    adapter = PinnedHTTPAdapter()

    req1 = requests.Request("GET", "http://example.com/").prepare()
    req1.headers[PINNED_IP_HEADER] = "93.184.216.34"
    pool1 = adapter.get_connection_with_tls_context(req1, verify=True)

    req2 = requests.Request("GET", "http://example.com/").prepare()
    req2.headers[PINNED_IP_HEADER] = "93.184.216.35"
    pool2 = adapter.get_connection_with_tls_context(req2, verify=True)

    assert pool1 is not pool2
    assert pool1.pinned_ip == "93.184.216.34"
    assert pool2.pinned_ip == "93.184.216.35"
    assert len(adapter._pinned_pools) == 2


def test_streaming_response_size_limit_with_local_server():
    """Response body larger than max_response_bytes is truncated and socket closed cleanly."""
    big_body = b"A" * 150_000

    class BigHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(big_body)))
            self.end_headers()
            self.wfile.write(big_body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), BigHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        config = replace(
            DEFAULT_NETWORK_CONFIG,
            allow_private_targets=True,
            max_response_bytes=65536,
        )
        page = fetch_page(f"http://127.0.0.1:{port}/", config)
        assert page.fetched is True
        assert page.truncated is True
        assert len(page.text) <= 65536
    finally:
        server.shutdown()
        server.server_close()


def test_two_servers_dns_rebinding_never_hits_rebound_destination():
    """Verify with two live HTTP servers that DNS rebinding cannot divert to Server 2."""
    server1_hits: list[str] = []
    server2_hits: list[str] = []

    class Handler1(BaseHTTPRequestHandler):
        def do_GET(self):
            server1_hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            body = b"<html><head><title>Target 1</title></head><body>Server 1</body></html>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Handler2(BaseHTTPRequestHandler):
        def do_GET(self):
            server2_hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            body = b"<html><head><title>Target 2</title></head><body>Server 2 (Attack)</body></html>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server1 = HTTPServer(("127.0.0.1", 0), Handler1)
    port1 = server1.server_address[1]
    thread1 = threading.Thread(target=server1.serve_forever, daemon=True)
    thread1.start()

    server2 = HTTPServer(("127.0.0.1", 0), Handler2)
    port2 = server2.server_address[1]
    thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
    thread2.start()

    try:
        # Initial resolve points to server 1 (127.0.0.1)
        # But if connect-time getaddrinfo is queried, it returns port2
        orig_gai = socket.getaddrinfo

        def fake_gai(host, p, *args, **kwargs):
            if host == "rebind.example":
                # Simulated rebinding attempt: divert connection to Server 2
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port2))]
            return orig_gai(host, p, *args, **kwargs)

        config = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)

        with (
            patch("detector.security._resolve_host", return_value=["127.0.0.1"]),
            patch("socket.getaddrinfo", side_effect=fake_gai),
        ):
            page = fetch_page(f"http://rebind.example:{port1}/test", config)

        assert page.fetched is True
        assert page.title == "Target 1"
        assert len(server1_hits) == 1
        assert len(server2_hits) == 0  # Server 2 was NEVER touched!
    finally:
        server1.shutdown()
        server1.server_close()
        server2.shutdown()
        server2.server_close()
