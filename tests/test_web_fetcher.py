"""Layer 4a: the safe fetcher.

Every test injects a fake HTTP client, so nothing here touches the network.
The SSRF cases assert that the guard runs *before* any request is issued -
`client.requests` staying empty is the actual assertion.
"""

from __future__ import annotations

from dataclasses import replace

from conftest import FakeHttpClient, FakeResponse
from detector.config import DEFAULT_NETWORK_CONFIG
from detector.web_fetcher import fetch_page

HTML = """
<html><head><title>Sign in</title></head>
<body><h1>Welcome</h1>
<form action="/submit" method="post"><input type="password" name="pw"></form>
</body></html>
"""


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_fetch_success_parses_page(allow_private_network):
    client = FakeHttpClient(FakeResponse(body=HTML))
    page = fetch_page("http://127.0.0.1:8000/login", allow_private_network, client=client)

    assert page.fetched is True
    assert page.status_code == 200
    assert page.title == "Sign in"
    assert "Welcome" in page.text
    assert len(page.forms) == 1
    assert page.fetch_error is None


def test_fetch_issues_get_only_with_no_cookies(allow_private_network):
    client = FakeHttpClient(FakeResponse(body=HTML))
    fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)

    method, _url, kwargs = client.requests[0]
    assert method == "GET"
    assert kwargs["allow_redirects"] is False      # we follow them ourselves
    assert kwargs["verify"] is True                # TLS verification stays on
    assert "cookies" not in kwargs
    assert "auth" not in kwargs
    assert "data" not in kwargs and "json" not in kwargs   # never submits anything


def test_fetch_sends_identifying_user_agent(allow_private_network):
    client = FakeHttpClient(FakeResponse(body=HTML))
    fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)
    headers = client.requests[0][2]["headers"]
    assert "phishing-url-detector" in headers["User-Agent"]
    assert "Referer" not in headers


# ---------------------------------------------------------------------------
# SSRF: the guard must run before any request goes out
# ---------------------------------------------------------------------------
def test_internal_target_blocked_before_any_request():
    client = FakeHttpClient(FakeResponse(body=HTML))
    page = fetch_page("http://169.254.169.254/latest/meta-data/",
                      DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    assert "blocked-by-ssrf-policy" in page.fetch_error
    assert client.requests == []            # nothing was ever sent


def test_non_http_scheme_blocked_before_any_request():
    client = FakeHttpClient(FakeResponse(body=HTML))
    page = fetch_page("file:///etc/passwd", DEFAULT_NETWORK_CONFIG, client=client)

    assert page.fetched is False
    assert client.requests == []


def test_redirect_to_internal_host_is_blocked_at_the_hop(allow_private_network):
    """A public first hop must not launder a redirect to an internal address."""
    strict = replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=False)
    client = FakeHttpClient({
        "https://8.8.8.8/start": FakeResponse(
            status_code=302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        ),
    })
    page = fetch_page("https://8.8.8.8/start", strict, client=client)

    assert page.fetched is False
    assert page.blocked_reason is not None
    # hop 1 was requested; hop 2 was refused before being requested
    assert client.urls == ["https://8.8.8.8/start"]


def test_redirects_are_followed_and_recorded(allow_private_network):
    client = FakeHttpClient({
        "http://127.0.0.1:8000/a": FakeResponse(
            status_code=302, headers={"Location": "/b"}),
        "http://127.0.0.1:8000/b": FakeResponse(body=HTML),
    })
    page = fetch_page("http://127.0.0.1:8000/a", allow_private_network, client=client)

    assert page.fetched is True
    assert page.final_url == "http://127.0.0.1:8000/b"
    assert page.redirect_chain == ("http://127.0.0.1:8000/b",)


def test_redirect_limit_enforced(allow_private_network):
    limited = replace(allow_private_network, max_redirects=2)
    client = FakeHttpClient({
        f"http://127.0.0.1:8000/{i}": FakeResponse(
            status_code=302, headers={"Location": f"/{i + 1}"})
        for i in range(10)
    })
    page = fetch_page("http://127.0.0.1:8000/0", limited, client=client)

    assert page.fetched is False
    assert "too-many-redirects" in page.fetch_error
    assert len(client.requests) <= 3


# ---------------------------------------------------------------------------
# response limits
# ---------------------------------------------------------------------------
def test_oversized_body_is_truncated_not_swallowed(allow_private_network):
    small = replace(allow_private_network, max_response_bytes=200)
    body = "<html><body>" + ("A" * 50_000) + "</body></html>"
    client = FakeHttpClient(FakeResponse(body=body))

    page = fetch_page("http://127.0.0.1:8000/", small, client=client)

    assert page.fetched is True
    assert page.truncated is True


def test_declared_oversized_response_refused_without_reading(allow_private_network):
    small = replace(allow_private_network, max_response_bytes=1000)
    client = FakeHttpClient(FakeResponse(
        body="x", headers={"Content-Type": "text/html", "Content-Length": "999999999"},
    ))
    page = fetch_page("http://127.0.0.1:8000/", small, client=client)

    assert page.fetched is False
    assert "response-too-large" in page.fetch_error


def test_non_html_content_type_refused(allow_private_network):
    client = FakeHttpClient(FakeResponse(
        body="%PDF-1.4 binary", headers={"Content-Type": "application/pdf"},
    ))
    page = fetch_page("http://127.0.0.1:8000/doc.pdf", allow_private_network, client=client)

    assert page.fetched is False
    assert "content-type-not-analysable" in page.fetch_error


def test_missing_content_type_is_still_parsed(allow_private_network):
    client = FakeHttpClient(FakeResponse(body=HTML, headers={}))
    page = fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)
    assert page.fetched is True


def test_text_is_capped_independently_of_bytes(allow_private_network):
    tiny_text = replace(allow_private_network, max_text_chars=40)
    body = "<html><body>" + ("word " * 5000) + "</body></html>"
    client = FakeHttpClient(FakeResponse(body=body))

    page = fetch_page("http://127.0.0.1:8000/", tiny_text, client=client)

    assert page.fetched is True
    assert len(page.text) <= 40
    assert page.truncated is True


def test_chunked_body_reassembled(allow_private_network):
    """The reader must not lose data across iter_content boundaries."""
    client = FakeHttpClient(FakeResponse(body=HTML, chunk_size=7))
    page = fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)
    assert page.title == "Sign in"


# ---------------------------------------------------------------------------
# failure paths - all must be features, never exceptions
# ---------------------------------------------------------------------------
def test_connection_error_recorded(allow_private_network):
    client = FakeHttpClient(exc=ConnectionError("refused"))
    page = fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)

    assert page.fetched is False
    assert "ConnectionError" in page.fetch_error


def test_timeout_recorded(allow_private_network):
    client = FakeHttpClient(exc=TimeoutError("slow"))
    page = fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)

    assert page.fetched is False
    assert "TimeoutError" in page.fetch_error


def test_empty_url():
    page = fetch_page("", DEFAULT_NETWORK_CONFIG)
    assert page.fetched is False
    assert page.fetch_error == "empty-url"


def test_response_is_always_closed(allow_private_network):
    response = FakeResponse(body=HTML)
    client = FakeHttpClient(response)
    fetch_page("http://127.0.0.1:8000/", allow_private_network, client=client)
    assert response.closed is True


def test_as_dict_redacts_the_url(allow_private_network):
    client = FakeHttpClient(FakeResponse(body=HTML))
    page = fetch_page("http://127.0.0.1:8000/login?password=hunter2",
                      allow_private_network, client=client)
    assert "hunter2" not in str(page.as_dict())
