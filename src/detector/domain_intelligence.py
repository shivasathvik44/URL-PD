"""Layer 2: DNS / domain intelligence orchestration.

Ties together:
  * Layer 1's URL/domain parsing (`parse_url`, `extract_lexical_features`) - reused, not
    re-implemented, so hostname / registered-domain logic lives in one place;
  * `dns_features.resolve_domain` - A/AAAA/NS/MX/TTL/reverse-DNS/IP classes;
  * `domain_age.lookup_domain_age` - RDAP-then-WHOIS registration age.

Public API:
    analyze_domain("example.com")          -> DomainIntelligence
    analyze_dns("https://example.com/x")   -> DomainIntelligence   (URL form)

Layer 2 does network I/O but never crashes on failure and never emits a verdict.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .config import DEFAULT_CONFIG, DetectorConfig
from .dns_features import DNSResult, ResolverLike, resolve_domain
from .domain_age import DomainAgeResult, lookup_domain_age
from .lexical_features import LexicalFeatures, extract_lexical_features
from .observability import get_logger
from .tls_features import TLSResult, inspect_tls

_log = get_logger("layer2")


class _DomainRateLimiter:
    """Minimum spacing between outbound lookups for the same domain.

    Layer 2 is called once per URL, and a campaign can put hundreds of URLs on
    one domain. Without spacing, a bulk run hammers one registry's RDAP/WHOIS
    endpoint and gets rate-limited (or blocked) for everyone. The cache removes
    most of that traffic; this bounds what is left.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str, min_interval: float) -> float:
        if min_interval <= 0 or not domain:
            return 0.0
        with self._lock:
            now = time.monotonic()
            previous = self._last.get(domain)
            delay = 0.0
            if previous is not None:
                elapsed = now - previous
                if elapsed < min_interval:
                    delay = min_interval - elapsed
            self._last[domain] = now + delay
        if delay:
            time.sleep(delay)
        return delay


_RATE_LIMITER = _DomainRateLimiter()


@dataclass(frozen=True)
class DomainIntelligence:
    """All Layer 2 features for one URL / domain."""

    input: str
    host: str
    registered_domain: str
    is_ip_host: bool
    is_malformed: bool

    dns: DNSResult
    age: DomainAgeResult

    network_performed: bool
    tls: TLSResult | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ---- flat feature view --------------------------------------------
    def as_dict(self) -> dict:
        d: dict = {
            "input": self.input,
            "domain": self.registered_domain or self.host,
            "host": self.host,
            "registered_domain": self.registered_domain,
            "is_ip_host": self.is_ip_host,
            "is_malformed": self.is_malformed,
            "network_performed": self.network_performed,
            "notes": list(self.notes),
        }
        d.update(self.dns.feature_dict())
        d.update(self.age.feature_dict())
        if self.tls is not None:
            d.update(self.tls.feature_dict())
        d["anomaly_features"] = self.anomaly_features()
        return d

    # ---- cache serialisation (lossless; as_dict() is not) --------------
    def to_cache(self) -> dict:
        return {
            "input": self.input,
            "host": self.host,
            "registered_domain": self.registered_domain,
            "is_ip_host": self.is_ip_host,
            "is_malformed": self.is_malformed,
            "network_performed": self.network_performed,
            "notes": list(self.notes),
            "dns": self.dns.to_cache(),
            "age": self.age.to_cache(),
            "tls": self.tls.to_cache() if self.tls else None,
        }

    @classmethod
    def from_cache(cls, d: dict) -> DomainIntelligence:
        return cls(
            input=d.get("input", ""),
            host=d.get("host", ""),
            registered_domain=d.get("registered_domain", ""),
            is_ip_host=bool(d.get("is_ip_host", False)),
            is_malformed=bool(d.get("is_malformed", False)),
            dns=DNSResult.from_cache(d.get("dns") or {}),
            age=DomainAgeResult.from_cache(d.get("age") or {}),
            network_performed=bool(d.get("network_performed", False)),
            tls=TLSResult.from_cache(d["tls"]) if d.get("tls") else None,
            notes=tuple(d.get("notes") or ()),
        )

    def anomaly_features(self) -> dict:
        """The compact numeric/binary bundle named in the spec."""
        return {
            "dns_resolves": self.dns.dns_resolves,
            "has_ns": self.dns.has_ns,
            "has_mx": self.dns.has_mx,
            "domain_age_days": self.age.age_days,
            "is_new_domain": self.age.is_new_domain,
            "ipv4_count": self.dns.ipv4_count,
            "ipv6_count": self.dns.ipv6_count,
            "public_ip_count": self.dns.public_ip_count,
            "private_ip_count": self.dns.private_ip_count,
            "reserved_ip_count": self.dns.reserved_ip_count,
            "reverse_dns_available": self.dns.reverse_dns_available,
            "ttl": self.dns.ttl,
            "ttl_available": self.dns.ttl_available,
            "ns_count": self.dns.ns_count,
            "mx_count": self.dns.mx_count,
        }

    def __getitem__(self, key: str):
        return self.as_dict()[key]


# --------------------------------------------------------------------------
def _resolve_names(
    raw: str, lexical: LexicalFeatures | None = None
) -> tuple[str, str, bool, bool, list[str]]:
    """Return (host, registered_domain, is_ip_host, is_malformed, notes).

    Reuses Layer 1: `extract_lexical_features` already does PSL-aware extraction, so we do
    not parse hostnames here. A caller that already ran Layer 1 can pass the
    `LexicalFeatures` in to avoid re-parsing.
    """
    notes: list[str] = []
    text = (raw or "").strip()
    if not text and lexical is None:
        return "", "", False, True, ["empty-input"]

    # `extract_lexical_features` tolerates bare domains (it prepends a scheme internally).
    lex = lexical if lexical is not None else extract_lexical_features(text)
    host = lex.hostname
    registered = lex.registered_domain
    is_ip = lex.has_ip
    malformed = lex.is_malformed

    if malformed:
        notes.append("layer1-marked-malformed")
    if is_ip:
        notes.append("host-is-ip-literal")
    if host and not registered and not is_ip and not malformed:
        notes.append("no-registered-domain (single-label or unknown TLD)")

    return host, registered, is_ip, malformed, notes


def analyze_domain(
    domain: str,
    config: DetectorConfig = DEFAULT_CONFIG,
    *,
    resolver: ResolverLike | None = None,
    lexical: LexicalFeatures | None = None,
    do_dns: bool = True,
    do_reverse_dns: bool = True,
    do_domain_age: bool = True,
    do_tls: bool = False,
    tls_fn=inspect_tls,
) -> DomainIntelligence:
    """Layer 2 for a domain or URL string. Never raises; network is best-effort.

    `do_tls` is opt-in because a TLS handshake is the most expensive probe here
    and adds nothing for a domain that does not resolve.
    """
    host, registered, is_ip, malformed, notes = _resolve_names(domain, lexical)

    network_performed = False
    deadline = time.monotonic() + max(config.layer2_budget, 0.0)

    def budget_left() -> bool:
        """False once the overall Layer 2 deadline has passed."""
        return config.layer2_budget <= 0 or time.monotonic() < deadline

    # Space out repeated lookups for one domain (no-op when the interval is 0).
    if (do_dns or do_domain_age) and registered:
        waited = _RATE_LIMITER.wait(registered, config.domain_lookup_min_interval)
        if waited:
            notes.append(f"rate-limited {waited:.2f}s")

    if do_dns and (host and not malformed):
        dns_result = resolve_domain(
            host,
            registered or host,
            config=config,
            resolver=resolver,
            do_reverse_dns=do_reverse_dns,
        )
        network_performed = network_performed or dns_result.queried or bool(dns_result.errors)
    else:
        dns_result = DNSResult(
            host=host, registered_domain=registered or host, queried=False,
            errors={"dns": "skipped" if not do_dns else "no-usable-host"},
        )
        if not do_dns:
            notes.append("dns-skipped")

    age_wanted = do_domain_age and registered and not is_ip and not malformed
    if age_wanted and not budget_left():
        age_result = DomainAgeResult(
            source=None, available=False, error="skipped: layer 2 time budget spent")
        notes.append("domain-age-skipped (budget)")
    elif age_wanted:
        age_result = lookup_domain_age(registered, config)
        network_performed = network_performed or age_result.available or bool(age_result.error)
    else:
        reason = (
            "domain-age-skipped" if not do_domain_age
            else "no-registered-domain" if not registered
            else "ip-host" if is_ip else "malformed"
        )
        age_result = DomainAgeResult(source=None, available=False, error=reason)
        if not do_domain_age:
            notes.append("domain-age-skipped")

    tls_result: TLSResult | None = None
    tls_wanted = do_tls and host and not malformed and not is_ip
    if tls_wanted and not dns_result.dns_resolves:
        # A TLS handshake needs an address. When DNS produced none, the probe
        # can only fail - and failing takes a name resolution plus up to three
        # connection attempts, which is the single most expensive thing Layer 2
        # can do. Measured: it turned a 25 s analysis into 106 s.
        tls_result = TLSResult(
            host=host, checked=False, tls_error="skipped: domain does not resolve")
        notes.append("tls-skipped (no DNS resolution)")
    elif tls_wanted and not budget_left():
        tls_result = TLSResult(
            host=host, checked=False, tls_error="skipped: layer 2 time budget spent")
        notes.append("tls-skipped (budget)")
    elif tls_wanted:
        try:
            tls_result = tls_fn(host, config=config)
            network_performed = network_performed or tls_result.checked
        except Exception as exc:  # noqa: BLE001 - TLS must never crash Layer 2
            _log.debug("tls inspection failed", extra={"fields": {"error": str(exc)}})
            tls_result = TLSResult(host=host, checked=False, tls_error=str(exc))

    return DomainIntelligence(
        input=domain,
        host=host,
        registered_domain=registered,
        is_ip_host=is_ip,
        is_malformed=malformed,
        dns=dns_result,
        age=age_result,
        network_performed=network_performed,
        tls=tls_result,
        notes=tuple(notes),
    )


def analyze_dns(
    url: str,
    config: DetectorConfig = DEFAULT_CONFIG,
    **kwargs,
) -> DomainIntelligence:
    """Alias that reads as 'Layer 2 for a URL'. Identical behaviour to
    `analyze_domain` (which already accepts URLs)."""
    return analyze_domain(url, config, **kwargs)
