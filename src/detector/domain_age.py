"""Layer 2b: domain registration age via RDAP (preferred) then WHOIS (fallback).

Design rules from the spec:
  * RDAP/WHOIS data is frequently missing, partial, rate-limited or slow.
  * A lookup failure is a *feature* ("age unknown"), never "phishing" and never
    an exception.
  * Modern RDAP is tried first; WHOIS is only a fallback.

Both providers are optional. If neither library is installed (or both fail), the
result is simply `available=False` with everything else `None`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_CONFIG, DetectorConfig

# Injection points - real implementations below, overridden in tests.
RdapLookup = Callable[[str, float], "RawAge | None"]
WhoisLookup = Callable[[str, float], "RawAge | None"]


@dataclass(frozen=True)
class RawAge:
    """Normalised subset of what a provider returned."""

    creation_date: datetime | None = None
    expiration_date: datetime | None = None
    registrar: str | None = None


@dataclass(frozen=True)
class DomainAgeResult:
    source: str | None            # "rdap" | "whois" | None
    available: bool
    creation_date: datetime | None = None
    expiration_date: datetime | None = None
    registrar: str | None = None
    age_days: int | None = None
    is_new_domain: bool = False   # False when unknown - never guess "new"
    error: str | None = None

    def to_cache(self) -> dict:
        iso = lambda d: d.isoformat() if isinstance(d, datetime) else None
        return {
            "source": self.source,
            "available": self.available,
            "creation_date": iso(self.creation_date),
            "expiration_date": iso(self.expiration_date),
            "registrar": self.registrar,
            "age_days": self.age_days,
            "is_new_domain": self.is_new_domain,
            "error": self.error,
        }

    @classmethod
    def from_cache(cls, d: dict) -> DomainAgeResult:
        parse = lambda v: _as_utc(v) if v else None
        creation = parse(d.get("creation_date"))
        # Recompute the age from the creation date rather than trusting the
        # stored number, which ages out while the row sits in the cache.
        # `is_new_domain` is kept as written because it depends on the config's
        # threshold, and the cache TTL bounds how far it can drift.
        age = _age_days(creation) if creation else d.get("age_days")
        return cls(
            source=d.get("source"),
            available=bool(d.get("available", False)),
            creation_date=creation,
            expiration_date=parse(d.get("expiration_date")),
            registrar=d.get("registrar"),
            age_days=age,
            is_new_domain=bool(d.get("is_new_domain", False)),
            error=d.get("error"),
        )

    def feature_dict(self) -> dict:
        iso = lambda d: d.isoformat() if isinstance(d, datetime) else None
        return {
            "domain_age_source": self.source,
            "domain_age_available": self.available,
            "creation_date": iso(self.creation_date),
            "expiration_date": iso(self.expiration_date),
            "registrar": self.registrar,
            "domain_age_days": self.age_days,
            "is_new_domain": self.is_new_domain,
            "domain_age_error": self.error,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_utc(value) -> datetime | None:
    """Coerce a datetime / list-of-datetime / ISO string into an aware UTC dt."""
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if v is not None), None)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_days(creation: datetime | None) -> int | None:
    if creation is None:
        return None
    delta = datetime.now(timezone.utc) - creation
    return max(delta.days, 0)


# --------------------------------------------------------------------------
# real providers (import lazily so a missing lib is not a hard error)
# --------------------------------------------------------------------------
def _bootstrap_path() -> Path:
    from .config import PROJECT_ROOT

    return PROJECT_ROOT / "data" / "processed" / "rdap_bootstrap.json"


def _ensure_bootstrapped(whoisit, max_age_days: int = 7) -> None:
    """Load IANA RDAP bootstrap data, from disk when we have a fresh copy.

    Without this, every process re-downloads the bootstrap registry before its
    first RDAP query - a network round trip that has nothing to do with the
    domain being analysed, repeated for every CLI invocation.
    """
    if whoisit.is_bootstrapped() and not whoisit.bootstrap_is_older_than(max_age_days):
        return

    path = _bootstrap_path()
    try:
        if path.exists():
            whoisit.load_bootstrap_data(json.loads(path.read_text()))
            if not whoisit.bootstrap_is_older_than(max_age_days):
                return
    except Exception:  # noqa: BLE001 - a bad cache file must not block lookups
        pass

    whoisit.bootstrap(overrides=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(whoisit.save_bootstrap_data()))
    except Exception:  # noqa: BLE001 - caching is best-effort
        pass


def _rdap_lookup(domain: str, timeout: float) -> RawAge | None:
    try:
        import whoisit
    except Exception:
        return None
    try:
        import requests

        class _TimeoutSession(requests.Session):
            def request(self, *a, **kw):
                kw.setdefault("timeout", timeout)
                return super().request(*a, **kw)

        _ensure_bootstrapped(whoisit)
        data = whoisit.domain(domain, session=_TimeoutSession())
    except Exception:
        return None

    registrar = None
    ents = (data.get("entities") or {}).get("registrar") or []
    if ents and isinstance(ents, list):
        registrar = ents[0].get("name") or ents[0].get("handle")
    return RawAge(
        creation_date=_as_utc(data.get("registration_date")),
        expiration_date=_as_utc(data.get("expiration_date")),
        registrar=registrar,
    )


def _whois_lookup(domain: str, timeout: float) -> RawAge | None:
    try:
        import whois  # python-whois
    except Exception:
        return None
    try:
        data = whois.whois(
            domain, timeout=int(max(timeout, 1)), ignore_socket_errors=True, quiet=True
        )
    except Exception:
        return None
    if not data:
        return None
    creation = _as_utc(data.get("creation_date"))
    expiration = _as_utc(data.get("expiration_date"))
    registrar = data.get("registrar")
    if isinstance(registrar, (list, tuple)):
        registrar = registrar[0] if registrar else None
    if creation is None and expiration is None and not registrar:
        return None
    return RawAge(creation_date=creation, expiration_date=expiration, registrar=registrar)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def lookup_domain_age(
    domain: str,
    config: DetectorConfig = DEFAULT_CONFIG,
    *,
    rdap: RdapLookup | None = _rdap_lookup,
    whois: WhoisLookup | None = _whois_lookup,
) -> DomainAgeResult:
    """Try RDAP, then WHOIS. Any failure yields `available=False`, not an error."""
    domain = (domain or "").strip().rstrip(".").lower()
    if not domain or "." not in domain:
        return DomainAgeResult(source=None, available=False, error="no-registered-domain")

    last_error: str | None = None
    for name, fn, timeout in (
        ("rdap", rdap, config.rdap_timeout),
        ("whois", whois, config.whois_timeout),
    ):
        if fn is None:
            continue
        try:
            raw = fn(domain, timeout)
        except Exception as exc:  # noqa: BLE001 - providers must never crash us
            last_error = f"{name}: {type(exc).__name__}: {exc}"
            continue
        if raw is None:
            last_error = last_error or f"{name}: no-data"
            continue

        age = _age_days(raw.creation_date)
        is_new = age is not None and age <= config.new_domain_max_age_days
        return DomainAgeResult(
            source=name,
            available=True,
            creation_date=raw.creation_date,
            expiration_date=raw.expiration_date,
            registrar=raw.registrar,
            age_days=age,
            is_new_domain=is_new,
            error=None,
        )

    return DomainAgeResult(
        source=None, available=False, error=last_error or "no-provider-available"
    )
