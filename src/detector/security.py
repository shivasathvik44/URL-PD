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

What this module deliberately does NOT do: it does not make the network safe.
A determined attacker controlling DNS can still race the resolve/connect gap
(DNS rebinding). The mitigation for that is pinning the validated IP at connect
time, which `web_fetcher.py` does by connecting to the checked address.
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


# ---------------------------------------------------------------------------
# IP classification
# ---------------------------------------------------------------------------
def is_public_address(address: str) -> bool:
    """True only for globally-routable unicast addresses.

    Everything else - loopback, RFC1918, link-local (including the cloud
    metadata 169.254.169.254), CGNAT, multicast, reserved, unspecified - is
    treated as internal and therefore off-limits to the fetcher.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if not ip.is_global:
        return False
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return False
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    # 100.64.0.0/10 carrier-grade NAT reads as "global" on some Python versions.
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return False
    # IPv4-mapped / NAT64-style IPv6 that wraps a private v4 address.
    if ip.version == 6:
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None and not is_public_address(str(mapped)):
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
            result["infos"] = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
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

    Checks, in order: scheme allowlist, host present, host not on the blocked
    list, host is not a bare internal name, every resolved address is public.
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
        host = (parts.hostname or "").strip().lower().rstrip(".")
    except ValueError as exc:
        raise UnsafeTargetError(f"unparseable host: {exc}") from exc
    if not host:
        raise UnsafeTargetError("URL has no host")

    if host in {h.lower() for h in config.blocked_hosts}:
        raise UnsafeTargetError(f"host is explicitly blocked: {host}")

    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeTargetError(f"invalid port: {exc}") from exc

    if config.allow_private_targets:
        # Controlled test environments only: skip resolution entirely so that
        # loopback test servers work, but keep the scheme/host checks above.
        return FetchTarget(url=url, scheme=scheme, host=host, port=port, addresses=(host,))

    # A literal IP host is checked directly; a name is resolved first.
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        if "." not in host:
            # Single-label names resolve through internal search domains
            # ("intranet", "jenkins") - never a legitimate public page.
            raise UnsafeTargetError(
                f"refusing single-label (internal) hostname: {host}"
            ) from None
        addresses = _resolve_host(host, port)

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
