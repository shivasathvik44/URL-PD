"""Layer 2c: TLS certificate facts.

A short, bounded TLS handshake against port 443 to read what the certificate
says - issuer, validity window, whether the hostname actually matches, and how
recently it was issued. Free certificates are the norm for both phishing and
legitimate sites, so **none of this is a verdict**: a Let's Encrypt cert issued
yesterday is as common on a new small business site as on a phishing kit.

Three safety properties, matching the rest of Layer 2:
  * it goes through the same SSRF guard as Layer 4, so this cannot be used to
    probe internal TLS services;
  * an explicit timeout bounds the handshake;
  * every failure becomes a feature (`tls_error`), never an exception.

The handshake is deliberately done twice-over in one connection: once with
verification ON (does it validate?) and, if that fails, once with verification
OFF purely to *read* the certificate. Reading an invalid certificate is the
interesting case - we record `cert_valid=False` and still report the issuer.
"""

from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import DEFAULT_CONFIG, DEFAULT_NETWORK_CONFIG, DetectorConfig, NetworkConfig
from .security import UnsafeTargetError, check_fetch_target

_CERT_DATE_FORMAT = "%b %d %H:%M:%S %Y %Z"


@dataclass(frozen=True)
class TLSResult:
    """TLS/certificate features for one host."""

    host: str
    checked: bool = False
    tls_available: bool = False
    cert_valid: bool = False            # chain + hostname verified successfully
    issuer: str | None = None
    subject: str | None = None
    subject_alt_names: tuple[str, ...] = ()
    not_before: str | None = None
    not_after: str | None = None
    days_until_expiry: int | None = None
    cert_age_days: int | None = None
    hostname_matches: bool = False
    is_self_signed: bool = False
    tls_version: str | None = None
    tls_error: str | None = None

    def to_cache(self) -> dict:
        return {
            "host": self.host, "checked": self.checked,
            "tls_available": self.tls_available, "cert_valid": self.cert_valid,
            "issuer": self.issuer, "subject": self.subject,
            "subject_alt_names": list(self.subject_alt_names),
            "not_before": self.not_before, "not_after": self.not_after,
            "days_until_expiry": self.days_until_expiry,
            "cert_age_days": self.cert_age_days,
            "hostname_matches": self.hostname_matches,
            "is_self_signed": self.is_self_signed,
            "tls_version": self.tls_version, "tls_error": self.tls_error,
        }

    @classmethod
    def from_cache(cls, d: dict) -> TLSResult:
        return cls(
            host=d.get("host", ""), checked=bool(d.get("checked", False)),
            tls_available=bool(d.get("tls_available", False)),
            cert_valid=bool(d.get("cert_valid", False)),
            issuer=d.get("issuer"), subject=d.get("subject"),
            subject_alt_names=tuple(d.get("subject_alt_names") or ()),
            not_before=d.get("not_before"), not_after=d.get("not_after"),
            days_until_expiry=d.get("days_until_expiry"),
            cert_age_days=d.get("cert_age_days"),
            hostname_matches=bool(d.get("hostname_matches", False)),
            is_self_signed=bool(d.get("is_self_signed", False)),
            tls_version=d.get("tls_version"), tls_error=d.get("tls_error"),
        )

    def feature_dict(self) -> dict:
        return {
            "tls_checked": self.checked,
            "tls_available": self.tls_available,
            "tls_cert_valid": self.cert_valid,
            "tls_issuer": self.issuer,
            "tls_subject": self.subject,
            "tls_subject_alt_names": list(self.subject_alt_names),
            "tls_not_before": self.not_before,
            "tls_not_after": self.not_after,
            "tls_days_until_expiry": self.days_until_expiry,
            "tls_cert_age_days": self.cert_age_days,
            "tls_hostname_matches": self.hostname_matches,
            "tls_is_self_signed": self.is_self_signed,
            "tls_version": self.tls_version,
            "tls_error": self.tls_error,
        }


def _parse_cert_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, _CERT_DATE_FORMAT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _name_from(seq) -> str | None:
    """Flatten OpenSSL's ((('commonName', 'x'),),) name structure."""
    if not seq:
        return None
    parts = []
    for rdn in seq:
        for item in rdn:
            if len(item) == 2:
                parts.append(f"{item[0]}={item[1]}")
    return ", ".join(parts) or None


def _common_name(seq) -> str | None:
    if not seq:
        return None
    for rdn in seq:
        for item in rdn:
            if len(item) == 2 and item[0] in ("commonName", "CN"):
                return str(item[1])
    return None


def inspect_tls(
    host: str,
    *,
    port: int = 443,
    config: DetectorConfig = DEFAULT_CONFIG,
    network: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> TLSResult:
    """Read the TLS certificate presented by `host`. Never raises."""
    host = (host or "").strip().rstrip(".").lower()
    if not host:
        return TLSResult(host="", checked=False, tls_error="empty-host")

    # Same SSRF policy as Layer 4 - never open a TLS socket to an internal host.
    pinned_ip: str | None = None
    try:
        target = check_fetch_target(f"https://{host}:{port}/", network)
        pinned_ip = target.primary_address
    except UnsafeTargetError as exc:
        return TLSResult(host=host, checked=False, tls_error=f"blocked: {exc}")

    timeout = config.tls_timeout

    # Tier 1: full verification (valid chain AND matching hostname).
    strict = ssl.create_default_context()
    cert, tls_version, verify_error = _handshake(host, port, strict, timeout, pinned_ip=pinned_ip)
    if cert:
        return _build(host, cert, tls_version, verified=True, tls_error=None)

    # Tier 2: verify the chain but not the hostname. `getpeercert()` only
    # returns fields when the chain verified, so this is the one way to read a
    # certificate that is genuinely issued but presented for the wrong name -
    # a common phishing pattern worth reporting precisely.
    chain_only = ssl.create_default_context()
    chain_only.check_hostname = False
    cert, tls_version, chain_error = _handshake(host, port, chain_only, timeout, pinned_ip=pinned_ip)
    if cert:
        return _build(host, cert, tls_version, verified=False, tls_error=verify_error)

    # Tier 3: no verification at all. An expired or self-signed certificate
    # cannot be read via the stdlib, so we can only record that TLS answered.
    insecure = ssl.create_default_context()
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    _cert, tls_version, connect_error = _handshake(host, port, insecure, timeout, pinned_ip=pinned_ip)
    if tls_version is not None:
        return TLSResult(
            host=host, checked=True, tls_available=True, cert_valid=False,
            tls_version=tls_version,
            tls_error=verify_error or chain_error or "certificate not verifiable",
        )

    return TLSResult(
        host=host, checked=True, tls_available=False,
        tls_error=verify_error or chain_error or connect_error or "tls-unavailable",
    )


def _handshake(
    host: str, port: int, context: ssl.SSLContext, timeout: float, *, pinned_ip: str | None = None
):
    """Return (cert_dict_or_None, tls_version_or_None, error_string_or_None)."""
    try:
        connect_host = pinned_ip or host
        with socket.create_connection((connect_host, port), timeout=timeout) as raw:
            raw.settimeout(timeout)
            with context.wrap_socket(raw, server_hostname=host) as tls:
                return (tls.getpeercert() or None), tls.version(), None
    except ssl.SSLCertVerificationError as exc:
        return None, None, f"cert-verification-failed: {exc.verify_message or exc}"
    except ssl.SSLError as exc:
        return None, None, f"tls-error: {type(exc).__name__}: {exc}"
    except TimeoutError:
        return None, None, "timeout"
    except OSError as exc:
        return None, None, f"connect-failed: {type(exc).__name__}: {exc}"


def hostname_matches_cert(host: str, common_name: str | None,
                          sans: tuple[str, ...]) -> bool:
    """RFC 6125-style match of `host` against a cert's CN and DNS SANs.

    Only a leading `*.` wildcard is honoured, and it matches exactly one label -
    `*.example.com` covers `a.example.com` but not `a.b.example.com`.
    """
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    candidates = [c.lower().rstrip(".") for c in sans if c]
    if common_name:
        candidates.append(common_name.lower().rstrip("."))
    for pattern in candidates:
        if pattern == host:
            return True
        if pattern.startswith("*."):
            suffix = pattern[1:]           # ".example.com"
            if host.endswith(suffix) and "." not in host[: -len(suffix)]:
                return True
    return False


def _build(host: str, cert: dict, tls_version: str | None, *,
           verified: bool, tls_error: str | None) -> TLSResult:
    not_before = _parse_cert_time(cert.get("notBefore"))
    not_after = _parse_cert_time(cert.get("notAfter"))
    now = datetime.now(timezone.utc)

    issuer = _name_from(cert.get("issuer"))
    subject = _name_from(cert.get("subject"))
    sans = tuple(
        str(value).lower()
        for kind, value in (cert.get("subjectAltName") or ())
        if kind == "DNS"
    )

    return TLSResult(
        host=host,
        checked=True,
        tls_available=True,
        cert_valid=verified,
        issuer=issuer,
        subject=subject,
        subject_alt_names=sans,
        not_before=not_before.isoformat() if not_before else None,
        not_after=not_after.isoformat() if not_after else None,
        days_until_expiry=(not_after - now).days if not_after else None,
        cert_age_days=max((now - not_before).days, 0) if not_before else None,
        hostname_matches=hostname_matches_cert(host, _common_name(cert.get("subject")), sans),
        is_self_signed=bool(issuer and subject and issuer == subject),
        tls_version=tls_version,
        tls_error=tls_error,
    )
