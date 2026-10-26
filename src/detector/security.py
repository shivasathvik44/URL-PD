"""Security controls shared by every layer that touches untrusted input.

Three jobs:

1. **SSRF defence.** Layer 4 fetches a URL an attacker chose. Without a guard,
   "analyse this URL for me" becomes "make your server request
   http://169.254.169.254/latest/meta-data/iam/security-credentials/ and tell
   me what it says". `check_fetch_target()` resolves the host and refuses
   loopback, private, link-local, reserved, multicast and CGNAT destinations,
   non-http(s) schemes, and a small list of known metadata hostnames - and it
   is re-run on **every redirect hop**, because hop 1 being public says nothing
   about hop 2.

2. **Output sanitisation.** Anything derived from a fetched page - including an
   LLM's description of it - is attacker-influenced text. `sanitize_text()`
   strips control characters (ANSI escapes, CR/LF injection into logs) and
   truncates, so hostile page content cannot forge output structure.

3. **Secret redaction.** `redact()` keeps API keys and credential-ish query
   parameters out of logs and structured results.

What this module does: it validates target schemes, ports, hostnames
(including IDNA normalization and single-label rejection), and performs
pre-fetch DNS address classification (rejecting loopback, private, link-local,
multicast, CGNAT, reserved, and IPv6-encapsulated internal addresses).

To defeat DNS rebinding (TOCTOU between validation and socket connection),
`web_fetcher.py` enforces connection pinning at connect time via `PinnedHTTPAdapter`.
For hostile-input deployments, host-level network isolation (egress firewalls)
remains recommended as an additional defense-in-depth control.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import DEFAULT_NETWORK_CONFIG, NetworkConfig

# --- control characters we never allow into output or logs -----------------
# Everything in the Cc/Cf categories except ordinary whitespace we keep (\t\n
# are normalised to spaces by sanitize_text).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE_RE = re.compile(r"\s+")

# --- query parameters whose values must never be logged or echoed ----------
_SECRET_PARAM_RE = re.compile(
    r"(?i)\b(pass(word|wd)?|pwd|secret|token|api[_-]?key|auth|session|"
    r"otp|code|cvv|card|ssn|credential)s?\b"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)((?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|auth|session|otp)"
    r"s?\s*[=:]\s*)([^\s&;,'\"]+)"
)
#: Anthropic-style keys and generic long bearer-ish blobs.
_KEY_LIKE_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b")


class UnsafeTargetError(ValueError):
    """Raised when a URL must not be fetched (SSRF policy, bad scheme, ...)."""


@dataclass(frozen=True)
class FetchTarget:
    """A URL that passed the SSRF policy, with the addresses it resolved to."""

    url: str
    scheme: str
    host: str
    port: int
    addresses: tuple[str, ...]

    @property
    def primary_address(self) -> str:
        return self.addresses[0]

    @property
    def pinned_address(self) -> str:
        return self.addresses[0]


# ---------------------------------------------------------------------------
# IP classification
# ---------------------------------------------------------------------------
_IPV4_SPECIAL_NETWORKS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("0.0.0.0/8"),          # Current network / "this" host (RFC 1122)
    ipaddress.IPv4Network("10.0.0.0/8"),         # Private network (RFC 1918)
    ipaddress.IPv4Network("100.64.0.0/10"),      # Carrier-grade NAT (RFC 6598)
    ipaddress.IPv4Network("127.0.0.0/8"),        # Loopback (RFC 1122)
    ipaddress.IPv4Network("169.254.0.0/16"),     # Link-local / cloud metadata (RFC 3927)
    ipaddress.IPv4Network("172.16.0.0/12"),      # Private network (RFC 1918)
    ipaddress.IPv4Network("192.0.0.0/24"),       # IETF protocol assignments (RFC 6890)
    ipaddress.IPv4Network("192.0.2.0/24"),       # Documentation TEST-NET-1 (RFC 5737)
    ipaddress.IPv4Network("192.88.99.0/24"),     # 6to4 relay anycast (RFC 7526)
    ipaddress.IPv4Network("192.168.0.0/16"),     # Private network (RFC 1918)
    ipaddress.IPv4Network("198.18.0.0/15"),      # Benchmarking (RFC 2544)
    ipaddress.IPv4Network("198.51.100.0/24"),    # Documentation TEST-NET-2 (RFC 5737)
    ipaddress.IPv4Network("203.0.113.0/24"),     # Documentation TEST-NET-3 (RFC 5737)
    ipaddress.IPv4Network("224.0.0.0/4"),        # Multicast (RFC 5771)
    ipaddress.IPv4Network("240.0.0.0/4"),        # Reserved / Future use (RFC 1112)
    ipaddress.IPv4Network("255.255.255.255/32"), # Limited broadcast
)

_IPV6_SPECIAL_NETWORKS: tuple[ipaddress.IPv6Network, ...] = (
    ipaddress.IPv6Network("::/128"),             # Unspecified (RFC 4291)
    ipaddress.IPv6Network("::1/128"),           # Loopback (RFC 4291)
    ipaddress.IPv6Network("100::/64"),           # Discard prefix (RFC 6666)
    ipaddress.IPv6Network("2001:0::/32"),        # Teredo (RFC 4380, RFC 6890)
    ipaddress.IPv6Network("2001:2::/48"),        # Benchmarking (RFC 5180)
    ipaddress.IPv6Network("2001:20::/28"),       # ORCHIDv2 (RFC 7343)
    ipaddress.IPv6Network("2001:db8::/32"),      # Documentation (RFC 3849)
    ipaddress.IPv6Network("fc00::/7"),           # Unique-local addresses (ULA, RFC 4193)
    ipaddress.IPv6Network("fe80::/10"),          # Link-local unicast (RFC 4291)
    ipaddress.IPv6Network("ff00::/8"),           # Multicast (RFC 4291)
)


def _is_alternative_ip_format(host_str: str) -> bool:
    """Detect non-canonical or ambiguous numeric IPv4 representations (octal, hex, dword)."""
    if not host_str:
        return False
    if host_str.startswith(("0x", "0X")):
        return True
    if host_str.isdigit():
        return True
    parts = host_str.split(".")
    for part in parts:
        if len(part) > 1 and part.startswith("0") and part.isdigit():
            return True
        if part.startswith(("0x", "0X")):
            return True
    return len(parts) in (2, 3) and all(p.isdigit() for p in parts)


def is_public_address(address: str) -> bool:
    """True only for globally-routable unicast addresses.

    Everything else - loopback, RFC1918, link-local (including the cloud
    metadata 169.254.169.254), CGNAT, multicast, reserved, unspecified,
    documentation, test networks, and IPv6 encodings of internal IPv4
    destinations - is treated as internal and therefore off-limits.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    if ip.version == 6:
        # IPv4-mapped IPv6 (::ffff:0:0/96)
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            return is_public_address(str(mapped))

        # Deprecated IPv4-compatible IPv6 (::/96)
        if ip in ipaddress.ip_network("::/96"):
            if ip != ipaddress.ip_address("::") and ip != ipaddress.ip_address("::1"):
                embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
                return is_public_address(str(embedded))
            return False

        # NAT64 / IPv4-translated (64:ff9b::/96, 64:ff9b:1::/48)
        if ip in ipaddress.ip_network("64:ff9b::/96") or ip in ipaddress.ip_network("64:ff9b:1::/48"):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
            return is_public_address(str(embedded))

        # 6to4 (2002::/16)
        if ip in ipaddress.ip_network("2002::/16"):
            embedded = ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)
            return is_public_address(str(embedded))

        if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
        if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
        for net in _IPV6_SPECIAL_NETWORKS:
            if ip in net:
                return False

    elif ip.version == 4:
        if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
        if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
        for net in _IPV4_SPECIAL_NETWORKS:
            if ip in net:
                return False

    return True


#: Wall-clock ceiling for one name resolution, in seconds.
#:
#: `socket.getaddrinfo` is a blocking C call with NO timeout parameter - its
#: duration is entirely at the mercy of the OS resolver's retry schedule, which
#: on an unreachable or flaky upstream routinely runs for tens of seconds. That
#: is unbounded time inside an otherwise carefully bounded pipeline, so we run
#: it on a worker thread and stop waiting when the budget is spent. The thread
#: is a daemon: we cannot kill a blocked C call, but we can refuse to wait.
RESOLVE_TIMEOUT = 5.0


def _resolve_host(host: str, port: int, timeout: float = RESOLVE_TIMEOUT) -> list[str]:
    """All addresses `host` resolves to, within `timeout`.

    Raises `UnsafeTargetError` on failure or timeout - an unresolvable name is
    not fetchable, and a name we could not resolve in time is one we cannot
    validate, so refusing is the safe answer either way.

    A **daemon** thread, deliberately, not a `ThreadPoolExecutor`: the pool's
    workers are non-daemon and joined by an `atexit` hook, so a worker still
    stuck in `getaddrinfo` would hang interpreter shutdown - trading a slow
    call for a process that will not exit. We cannot kill a blocked C call, but
    a daemon thread lets us abandon it and let the process die normally.
    """
    result: dict = {}

    def _resolve() -> None:
        try:
            result["infos"] = socket.getaddrinfo(
                host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
            )
        except BaseException as exc:  # noqa: BLE001 - reported on the caller's thread
            result["error"] = exc

    worker = threading.Thread(
        target=_resolve, name=f"detector-resolve-{host[:32]}", daemon=True
    )
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        raise UnsafeTargetError(
            f"name resolution for {host} exceeded {timeout:g}s"
        )
    if "error" in result:
        exc = result["error"]
        raise UnsafeTargetError(f"host does not resolve: {host} ({exc})") from exc

    addresses = []
    for info in result.get("infos", ()):
        addr = info[4][0]
        if addr not in addresses:
            addresses.append(addr)
    if not addresses:
        raise UnsafeTargetError(f"host resolved to no addresses: {host}")
    return addresses


def check_fetch_target(
    url: str, config: NetworkConfig = DEFAULT_NETWORK_CONFIG
) -> FetchTarget:
    """Validate `url` against the SSRF policy. Raises `UnsafeTargetError`.

    Checks, in order: scheme allowlist, host present, port range, alternative IP
    representation, host not on the blocked list, host is not a bare internal name,
    IDNA normalization, and every resolved address is public.
    **All** addresses must be public - a host that resolves to one public and
    one private address is refused, since we cannot control which one a later
    connection would pick.
    """
    if not url or not url.strip():
        raise UnsafeTargetError("empty URL")

    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "").lower()
    if scheme not in config.allowed_schemes:
        raise UnsafeTargetError(
            f"scheme {scheme or '(none)'!r} not allowed "
            f"(allowed: {', '.join(config.allowed_schemes)})"
        )

    try:
        raw_host = (parts.hostname or "").strip().rstrip(".")
    except ValueError as exc:
        raise UnsafeTargetError(f"unparseable host: {exc}") from exc
    if not raw_host:
        raise UnsafeTargetError("URL has no host")

    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeTargetError(f"invalid port: {exc}") from exc
    if not (1 <= port <= 65535):
        raise UnsafeTargetError(f"port out of valid range (1-65535): {port}")

    # Check for alternative numeric IP formats (octal, hex, integer, truncated quad)
    if _is_alternative_ip_format(raw_host):
        try:
            raw_ip = socket.inet_aton(raw_host)
            canon_ip = str(ipaddress.IPv4Address(raw_ip))
        except (OSError, ValueError):
            raise UnsafeTargetError(
                f"ambiguous or malformed IP representation: {raw_host!r}"
            ) from None
        if not is_public_address(canon_ip):
            raise UnsafeTargetError(
                f"host {raw_host!r} resolves to non-public address(es): {canon_ip}"
            )
        raise UnsafeTargetError(
            f"refusing non-canonical numeric IP representation: {raw_host!r}"
        )

    # Check if host is a standard literal IP
    is_ip_literal = False
    try:
        ip = ipaddress.ip_address(raw_host)
        is_ip_literal = True
        host = str(ip)
    except ValueError:
        host = raw_host

    if not is_ip_literal:
        if "." not in raw_host:
            # Single-label names resolve through internal search domains
            # ("intranet", "jenkins") - never a legitimate public page.
            raise UnsafeTargetError(
                f"refusing single-label (internal) hostname: {raw_host}"
            )
        try:
            host = raw_host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError) as exc:
            raise UnsafeTargetError(f"invalid hostname {raw_host!r}: {exc}") from exc

    if host in {h.lower() for h in config.blocked_hosts}:
        raise UnsafeTargetError(f"host is explicitly blocked: {host}")

    addresses = [host] if is_ip_literal else _resolve_host(host, port)

    if not config.allow_private_targets:
        bad = [a for a in addresses if not is_public_address(a)]
        if bad:
            raise UnsafeTargetError(
                f"host {host} resolves to non-public address(es): {', '.join(bad)}"
            )

    return FetchTarget(
        url=url, scheme=scheme, host=host, port=port, addresses=tuple(addresses)
    )


# ---------------------------------------------------------------------------
# text sanitisation
# ---------------------------------------------------------------------------
def sanitize_text(value, max_chars: int = 500, *, collapse_whitespace: bool = True) -> str:
    """Make attacker-influenced text safe to log, print and return.

    Strips control characters (so a page cannot inject ANSI escapes or fake log
    lines), normalises Unicode to NFKC (so homoglyph tricks do not survive into
    output as invisible bytes), collapses whitespace, and truncates.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def sanitize_str_list(values, *, max_items: int, max_chars: int) -> tuple[str, ...]:
    """Sanitise a list of strings, dropping non-strings and empties."""
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    for v in values[: max_items * 4]:          # bounded scan even if huge
        if not isinstance(v, (str, int, float)):
            continue
        s = sanitize_text(v, max_chars)
        if s and s not in out:
            out.append(s)
        if len(out) >= max_items:
            break
    return tuple(out)


def redact(text: str, max_chars: int = 2000) -> str:
    """Remove secrets from a string destined for a log."""
    if not text:
        return ""
    out = _KEY_LIKE_RE.sub("[REDACTED-KEY]", str(text))
    out = _SECRET_VALUE_RE.sub(r"\1[REDACTED]", out)
    return sanitize_text(out, max_chars)


def redact_url(url: str, max_chars: int = 300) -> str:
    """A URL safe for logs: credential-ish query values replaced.

    Phishing URLs regularly carry the victim's email or a session token in the
    query string; those must not end up in our logs.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return sanitize_text(url, max_chars)

    netloc = parts.netloc
    if "@" in netloc:                       # strip userinfo entirely
        netloc = netloc.rsplit("@", 1)[-1]

    query = parts.query
    if query:
        kept = []
        for pair in query.split("&"):
            if not pair:
                continue
            key, sep, _value = pair.partition("=")
            if sep and _SECRET_PARAM_RE.search(key):
                kept.append(f"{key}=[REDACTED]")
            elif sep:
                kept.append(f"{key}={_value}" if len(_value) <= 40 else f"{key}=[TRUNCATED]")
            else:
                kept.append(pair)
        query = "&".join(kept)

    rebuilt = f"{parts.scheme}://{netloc}{parts.path}"
    if query:
        rebuilt += f"?{query}"
    return sanitize_text(redact(rebuilt), max_chars)
