"""Layer 4a: fetching a hostile page, safely.

This is the component an attacker gets to point wherever they like, so every
default here is a restriction:

  * **GET only.** No other method is ever issued, no form is ever submitted.
  * **No session, no cookies, no credentials.** A fresh connection per fetch
    with cookies disabled - the detector never carries a user's identity to a
    phishing page, and a page cannot set state that leaks into the next fetch.
  * **SSRF-checked before every hop.** `security.check_fetch_target` validates
    the scheme and resolves the host, refusing loopback / RFC1918 / link-local
    (including cloud metadata) / CGNAT / reserved destinations. Redirects are
    followed manually so each hop is re-validated: hop 1 being public says
    nothing about hop 2.
  * **Bounded.** Connect and read timeouts, a wall-clock ceiling across all
    hops, a redirect limit, a decoded-body byte cap (which also bounds a
    decompression bomb), and a Content-Type allowlist.
  * **TLS verification on.** Never disabled; a verification failure is recorded
    as a feature, not worked around.
  * **Nothing is executed.** The body is parsed as text. No JavaScript engine,
    no subprocess, no file is written.

The response body is read incrementally through `iter_content`, which decodes
`Content-Encoding` as it goes, so the cap applies to *decompressed* bytes.
(Reading `raw` instead would cap the compressed stream - letting a small gzip
expand without limit, and producing binary garbage for any compressed page.)
"""

from __future__ import annotations

import http.cookiejar
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import DEFAULT_NETWORK_CONFIG, NetworkConfig
from .html_analyzer import PageStructure, analyze_html
from .observability import get_logger
from .security import UnsafeTargetError, check_fetch_target, redact_url

_log = get_logger("layer4.fetch")

#: Status codes we follow manually (each hop re-validated).
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class HttpClientLike(Protocol):
    """The slice of `requests`' API this module needs (so tests can fake it)."""

    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PageContent:
    """Everything Layer 4 needs from one fetched page."""

    url: str
    fetched: bool
    final_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    redirect_chain: tuple[str, ...] = ()
    structure: PageStructure | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0
    fetch_error: str | None = None
    blocked_reason: str | None = None      # set when the SSRF policy refused

    # --- convenience passthroughs to the parsed structure ---------------
    @property
    def title(self) -> str | None:
        return self.structure.title if self.structure else None

    @property
    def text(self) -> str:
        return self.structure.text if self.structure else ""

    @property
    def forms(self):
        return self.structure.forms if self.structure else ()

    def as_dict(self) -> dict:
        d = {
            "url": redact_url(self.url),
            "fetched": self.fetched,
            "final_url": redact_url(self.final_url) if self.final_url else None,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "redirect_chain": [redact_url(u) for u in self.redirect_chain],
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
            "fetch_error": self.fetch_error,
            "blocked_reason": self.blocked_reason,
        }
        d["structure"] = self.structure.as_dict() if self.structure else None
        return d


class BlockAllCookiePolicy(http.cookiejar.DefaultCookiePolicy):
    """A cookie policy that refuses to store or send anything.

    Subclasses `DefaultCookiePolicy` rather than duck-typing the interface.
    `http.cookiejar.CookieJar` reads policy *attributes* directly - `rfc2965`,
    `hide_cookie2`, `netscape`, `strict_ns_unverifiable` and friends - not just
    the four `*_ok` methods, so a partial stand-in raises `AttributeError`
    deep inside `extract_cookies()` on the first real response. Inheriting
    gives us the whole interface; overriding the two decision points is all
    that is needed to block cookies in both directions.
    """

    def set_ok(self, cookie, request) -> bool:      # never store
        return False

    def return_ok(self, cookie, request) -> bool:   # never send
        return False


def _default_client() -> HttpClientLike:
    import requests

    session = requests.Session()
    # No cookies in either direction: a hostile page must not be able to set
    # state that persists into the next request, and we never carry a user's
    # cookies outbound.
    session.cookies.set_policy(BlockAllCookiePolicy())
    session.trust_env = False          # ignore ambient proxy / netrc credentials
    # Deliberately NOT `session.max_redirects = 0`. Even with
    # `allow_redirects=False`, requests advances `resolve_redirects` once to
    # populate `response._next`, and that generator raises TooManyRedirects
    # when `len(history) >= max_redirects` - which is immediately true at 0. It
    # would turn every 3xx (http->https, apex->www: most of the web) into a
    # fetch failure. Our own redirect limit is enforced in `fetch_page`, which
    # is where each hop gets re-validated anyway.
    return session


def _content_type_allowed(content_type: str | None, config: NetworkConfig) -> bool:
    if not content_type:
        return True                    # absent header: let the parser decide
    base = content_type.split(";", 1)[0].strip().lower()
    return base in config.allowed_content_types


def _read_body(response, config: NetworkConfig, deadline: float) -> tuple[str, bool]:
    """Read at most `max_response_bytes` of *decoded* body. Returns (text, truncated)."""
    limit = config.max_response_bytes
    chunks: list[bytes] = []
    total = 0
    truncated = False

    for chunk in response.iter_content(chunk_size=16384, decode_unicode=False):
        if not chunk:
            continue
        if time.monotonic() > deadline:
            truncated = True
            break
        remaining = limit - total
        if len(chunk) >= remaining:
            chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)

    body = b"".join(chunks)
    encoding = response.encoding or response.apparent_encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace"), truncated
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace"), truncated


def fetch_page(
    url: str,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
    *,
    client: HttpClientLike | None = None,
) -> PageContent:
    """Fetch and parse `url` under the safety policy above. Never raises.

    A refusal by the SSRF guard is reported as `blocked_reason` with
    `fetched=False` - it is a *policy outcome*, not an error, and it is never
    treated as evidence that the URL is phishing.
    """
    started = time.perf_counter()
    deadline = time.monotonic() + config.total_timeout

    if not url or not url.strip():
        return PageContent(url=url or "", fetched=False, fetch_error="empty-url")

    try:
        # Named `session`, not `http`: a local called `http` would shadow the
        # `http.cookiejar` module imported above.
        session = client or _default_client()
    except Exception as exc:  # noqa: BLE001 - e.g. `requests` not installed
        return PageContent(url=url, fetched=False, fetch_error=f"no-http-client: {exc}")

    headers = {
        "User-Agent": config.user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
        "Accept-Language": "en",
        # Never send a referrer, and never reuse a connection across hosts.
        "Connection": "close",
    }

    current = url.strip()
    chain: list[str] = []
    response = None

    for hop in range(config.max_redirects + 1):
        try:
            check_fetch_target(current, config)
        except UnsafeTargetError as exc:
            _log.warning(
                "refused fetch target",
                extra={"fields": {"url": redact_url(current), "hop": hop,
                                  "reason": str(exc)}},
            )
            return PageContent(
                url=url, fetched=False, redirect_chain=tuple(chain),
                blocked_reason=str(exc),
                fetch_error=f"blocked-by-ssrf-policy: {exc}",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        if time.monotonic() > deadline:
            return PageContent(
                url=url, fetched=False, redirect_chain=tuple(chain),
                fetch_error="total-timeout-exceeded",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        try:
            response = session.request(
                "GET", current,
                headers=headers,
                timeout=(config.connect_timeout, config.read_timeout),
                allow_redirects=False,
                stream=True,
                verify=config.verify_tls,
            )
        except Exception as exc:  # noqa: BLE001 - DNS/TLS/connect/read failures
            return PageContent(
                url=url, fetched=False, redirect_chain=tuple(chain),
                fetch_error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        status = getattr(response, "status_code", None)
        location = (getattr(response, "headers", None) or {}).get("Location")
        if status in _REDIRECT_CODES and location:
            if hop >= config.max_redirects:
                return PageContent(
                    url=url, fetched=False, status_code=status,
                    redirect_chain=tuple(chain),
                    fetch_error=f"too-many-redirects (limit {config.max_redirects})",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            from urllib.parse import urljoin

            nxt = urljoin(current, location)
            chain.append(nxt)
            current = nxt
            continue
        break

    if response is None:  # pragma: no cover - loop always assigns or returns
        return PageContent(url=url, fetched=False, fetch_error="no-response")

    status = getattr(response, "status_code", None)
    headers_out = getattr(response, "headers", None) or {}
    content_type = headers_out.get("Content-Type") or headers_out.get("content-type")

    if not _content_type_allowed(content_type, config):
        return PageContent(
            url=url, fetched=False, final_url=current, status_code=status,
            content_type=content_type, redirect_chain=tuple(chain),
            fetch_error=f"content-type-not-analysable: {content_type}",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # Refuse an oversized body before reading it, when the server declares one.
    declared = headers_out.get("Content-Length")
    if declared and str(declared).isdigit() and int(declared) > config.max_response_bytes * 10:
        return PageContent(
            url=url, fetched=False, final_url=current, status_code=status,
            content_type=content_type, redirect_chain=tuple(chain),
            fetch_error=f"response-too-large: {declared} bytes declared",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    try:
        body, truncated = _read_body(response, config, deadline)
    except Exception as exc:  # noqa: BLE001
        return PageContent(
            url=url, fetched=False, final_url=current, status_code=status,
            content_type=content_type, redirect_chain=tuple(chain),
            fetch_error=f"body-read-failed: {type(exc).__name__}: {exc}",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    finally:
        try:
            response.close()
        except Exception as exc:  # noqa: BLE001 - the body is already read
            _log.debug("response close failed",
                       extra={"fields": {"error": type(exc).__name__}})

    structure = analyze_html(current, body, max_text_chars=config.max_text_chars)
    elapsed = round((time.perf_counter() - started) * 1000, 2)

    _log.info(
        "fetched page",
        extra={"fields": {"url": redact_url(url), "final_url": redact_url(current),
                          "status": status, "hops": len(chain),
                          "bytes_truncated": truncated, "elapsed_ms": elapsed}},
    )

    return PageContent(
        url=url,
        fetched=True,
        final_url=current,
        status_code=status,
        content_type=content_type,
        redirect_chain=tuple(chain),
        structure=structure,
        truncated=truncated or structure.text_truncated,
        elapsed_ms=elapsed,
    )
