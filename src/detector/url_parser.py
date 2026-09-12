"""Safe, scheme-tolerant URL parsing.

`parse_url` never raises on bad input: whatever cannot be understood comes back
as empty/`None` fields with `is_malformed=True`, so downstream feature
extraction can run unconditionally.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

# Matches "scheme://" at the very start (RFC 3986 scheme grammar).
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
# A bare "scheme:" with no slashes (mailto:, javascript:, tel: ...).
_OPAQUE_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:(?!//)")
# A syntactically plausible hostname: LDH + dots (+ ':' for IPv6 literals, '_'
# which shows up in the wild). Non-ASCII is allowed separately for IDN/unicode.
_PLAUSIBLE_HOST_RE = re.compile(r"^[a-z0-9._:\-]+$")


def _host_plausible(host: str) -> bool:
    if not host:
        return False
    return bool(_PLAUSIBLE_HOST_RE.match(host)) or any(ord(c) > 127 for c in host)


@dataclass(frozen=True)
class ParsedURL:
    """Structured, normalised view of a URL string."""

    raw: str
    scheme: str = ""
    scheme_present: bool = False
    username: str | None = None
    password: str | None = None
    hostname: str = ""          # lower-cased, trailing dot stripped, no brackets
    port: int | None = None
    path: str = ""
    query: str = ""
    fragment: str = ""
    is_ip: bool = False
    ip_version: int | None = None   # 4, 6, or None
    has_at_symbol: bool = False
    has_double_slash_in_path: bool = False
    is_malformed: bool = False

    @property
    def registered_host(self) -> str:
        """Hostname without a leading 'www.' - handy for comparisons."""
        h = self.hostname
        return h[4:] if h.startswith("www.") else h


def _strip_default_scheme(url: str) -> tuple[str, bool]:
    """Return (url_with_scheme, scheme_was_present).

    A URL like 'example.com/login' has no authority for urlsplit to find, so we
    prepend 'http://' for parsing while remembering the scheme was synthetic.
    """
    if _SCHEME_RE.match(url):
        return url, True
    if _OPAQUE_SCHEME_RE.match(url):
        # opaque scheme (mailto:, tel:) - leave as-is, there is no host
        return url, True
    if url.startswith("//"):
        return "http:" + url, False
    return "http://" + url, False


def _classify_ip(hostname: str) -> tuple[bool, int | None]:
    if not hostname:
        return False, None
    candidate = hostname
    # urlsplit already strips [] from IPv6 literals, but be defensive.
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return True, ipaddress.ip_address(candidate).version
    except ValueError:
        return False, None


def parse_url(url: str) -> ParsedURL:
    if not isinstance(url, str):
        return ParsedURL(raw="" if url is None else str(url), is_malformed=True)

    raw = url
    text = raw.strip()

    if not text:
        return ParsedURL(raw=raw, is_malformed=True)

    prepared, scheme_present = _strip_default_scheme(text)
    malformed = False

    try:
        parts = urlsplit(prepared)
    except ValueError:
        return ParsedURL(raw=raw, is_malformed=True)

    scheme = parts.scheme.lower()

    # .hostname / .port can each raise ValueError on malformed authorities.
    try:
        hostname = parts.hostname or ""
    except ValueError:
        hostname, malformed = "", True
    try:
        port = parts.port
    except ValueError:
        port, malformed = None, True

    try:
        username = parts.username
        password = parts.password
    except ValueError:
        username = password = None
        malformed = True

    hostname = hostname.lower().rstrip(".")
    is_ip, ip_version = _classify_ip(hostname)

    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""

    # '@' only counts when it is in the AUTHORITY, i.e. userinfo:
    # "http://paypal.com@evil.com/" really loads evil.com. An '@' in the path or
    # query ("?email=a@b.com") is ordinary and must not raise this flag - that
    # was a large false-positive source when the whole URL string was scanned.
    has_at = ("@" in (parts.netloc or "")) or (username is not None)

    # A '//' inside the path (decoded, to catch %2f%2f style) - the classic
    # "http://good.com//evil.com" or "/redirect//" trick.
    has_double_slash = "//" in unquote(path)

    if not hostname and not _OPAQUE_SCHEME_RE.match(text):
        malformed = True
    if hostname and not _host_plausible(hostname):
        malformed = True

    return ParsedURL(
        raw=raw,
        scheme=scheme,
        scheme_present=scheme_present,
        username=username,
        password=password,
        hostname=hostname,
        port=port,
        path=path,
        query=query,
        fragment=fragment,
        is_ip=is_ip,
        ip_version=ip_version,
        has_at_symbol=has_at,
        has_double_slash_in_path=has_double_slash,
        is_malformed=malformed,
    )
