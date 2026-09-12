"""Layer 2a: DNS-level features.

Pure DNS: A / AAAA / NS / MX resolution, record TTL, local IP classification and
(optional) reverse DNS. Every network call is wrapped - a failure produces an
empty/`False` feature and an entry in `errors`, never an exception.

WHOIS / RDAP domain-age lookups live in `domain_age.py`; the orchestration that
ties DNS + age + Layer 1 together lives in `domain_intelligence.py`.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any, Protocol

try:  # dnspython is a Layer 2 dependency; degrade cleanly if it is missing
    import dns.exception
    import dns.resolver
    import dns.reversename
    _DNS_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - only hit on a broken install
    dns = None  # type: ignore
    _DNS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from .config import DEFAULT_CONFIG, DetectorConfig


class ResolverLike(Protocol):
    """The slice of dnspython's Resolver API this module needs (so tests can
    pass a fake)."""

    def resolve(self, qname: str, rdtype: str, *args: Any, **kwargs: Any) -> Any: ...


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DNSResult:
    """DNS features for one hostname / registered domain."""

    host: str                       # name resolved for A/AAAA/PTR
    registered_domain: str          # name queried for NS/MX
    queried: bool                   # False when there was nothing to resolve

    dns_resolves: bool = False
    ipv4_addresses: tuple[str, ...] = ()
    ipv6_addresses: tuple[str, ...] = ()

    nameservers: tuple[str, ...] = ()
    mx_hosts: tuple[str, ...] = ()

    ttl: int | None = None

    reverse_dns_names: tuple[str, ...] = ()
    reverse_dns: dict[str, tuple[str, ...]] = field(default_factory=dict)

    ip_details: tuple[dict, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)

    # --- derived scalar features -------------------------------------
    @property
    def ipv4_count(self) -> int:
        return len(self.ipv4_addresses)

    @property
    def ipv6_count(self) -> int:
        return len(self.ipv6_addresses)

    @property
    def ip_count(self) -> int:
        return self.ipv4_count + self.ipv6_count

    @property
    def has_ipv4(self) -> bool:
        return self.ipv4_count > 0

    @property
    def has_ipv6(self) -> bool:
        return self.ipv6_count > 0

    @property
    def ns_count(self) -> int:
        return len(self.nameservers)

    @property
    def has_ns(self) -> bool:
        return self.ns_count > 0

    @property
    def mx_count(self) -> int:
        return len(self.mx_hosts)

    @property
    def has_mx(self) -> bool:
        return self.mx_count > 0

    @property
    def ttl_available(self) -> bool:
        return self.ttl is not None

    @property
    def public_ip_count(self) -> int:
        return sum(1 for d in self.ip_details if d["is_public"])

    @property
    def private_ip_count(self) -> int:
        return sum(1 for d in self.ip_details if d["is_private"])

    @property
    def reserved_ip_count(self) -> int:
        return sum(1 for d in self.ip_details if d["is_reserved"])

    @property
    def loopback_ip_count(self) -> int:
        return sum(1 for d in self.ip_details if d["is_loopback"])

    @property
    def multicast_ip_count(self) -> int:
        return sum(1 for d in self.ip_details if d["is_multicast"])

    @property
    def reverse_dns_available(self) -> bool:
        return len(self.reverse_dns_names) > 0

    # --- cache serialisation (lossless, unlike feature_dict) -------------
    def to_cache(self) -> dict:
        """Full round-trippable form. `feature_dict()` drops `ip_details` and
        `queried`, so it cannot be used to rebuild this object."""
        return {
            "host": self.host,
            "registered_domain": self.registered_domain,
            "queried": self.queried,
            "dns_resolves": self.dns_resolves,
            "ipv4_addresses": list(self.ipv4_addresses),
            "ipv6_addresses": list(self.ipv6_addresses),
            "nameservers": list(self.nameservers),
            "mx_hosts": list(self.mx_hosts),
            "ttl": self.ttl,
            "reverse_dns_names": list(self.reverse_dns_names),
            "reverse_dns": {k: list(v) for k, v in self.reverse_dns.items()},
            "ip_details": [dict(d) for d in self.ip_details],
            "errors": dict(self.errors),
        }

    @classmethod
    def from_cache(cls, d: dict) -> DNSResult:
        return cls(
            host=d.get("host", ""),
            registered_domain=d.get("registered_domain", ""),
            queried=bool(d.get("queried", False)),
            dns_resolves=bool(d.get("dns_resolves", False)),
            ipv4_addresses=tuple(d.get("ipv4_addresses") or ()),
            ipv6_addresses=tuple(d.get("ipv6_addresses") or ()),
            nameservers=tuple(d.get("nameservers") or ()),
            mx_hosts=tuple(d.get("mx_hosts") or ()),
            ttl=d.get("ttl"),
            reverse_dns_names=tuple(d.get("reverse_dns_names") or ()),
            reverse_dns={k: tuple(v) for k, v in (d.get("reverse_dns") or {}).items()},
            ip_details=tuple(d.get("ip_details") or ()),
            errors=dict(d.get("errors") or {}),
        )

    def feature_dict(self) -> dict:
        return {
            "host": self.host,
            "registered_domain": self.registered_domain,
            "dns_resolves": self.dns_resolves,
            "ipv4_count": self.ipv4_count,
            "ipv6_count": self.ipv6_count,
            "ip_count": self.ip_count,
            "has_ipv4": self.has_ipv4,
            "has_ipv6": self.has_ipv6,
            "ipv4_addresses": list(self.ipv4_addresses),
            "ipv6_addresses": list(self.ipv6_addresses),
            "has_ns": self.has_ns,
            "ns_count": self.ns_count,
            "nameservers": list(self.nameservers),
            "has_mx": self.has_mx,
            "mx_count": self.mx_count,
            "mx_hosts": list(self.mx_hosts),
            "ttl": self.ttl,
            "ttl_available": self.ttl_available,
            "public_ip_count": self.public_ip_count,
            "private_ip_count": self.private_ip_count,
            "reserved_ip_count": self.reserved_ip_count,
            "loopback_ip_count": self.loopback_ip_count,
            "multicast_ip_count": self.multicast_ip_count,
            "reverse_dns_available": self.reverse_dns_available,
            "reverse_dns_names": list(self.reverse_dns_names),
            "dns_errors": dict(self.errors),
        }


# --------------------------------------------------------------------------
# IP classification (local only - no network, no reputation)
# --------------------------------------------------------------------------
def classify_ip(address: str) -> dict:
    """Local properties of an IP string. Unparseable input -> all False."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return {
            "address": address, "version": None, "is_public": False,
            "is_private": False, "is_loopback": False, "is_reserved": False,
            "is_multicast": False, "is_link_local": False, "valid": False,
        }
    return {
        "address": str(ip),
        "version": ip.version,
        "is_public": ip.is_global,
        "is_private": ip.is_private,
        "is_loopback": ip.is_loopback,
        "is_reserved": ip.is_reserved,
        "is_multicast": ip.is_multicast,
        "is_link_local": ip.is_link_local,
        "valid": True,
    }


# --------------------------------------------------------------------------
# resolver plumbing
# --------------------------------------------------------------------------
def make_resolver(config: DetectorConfig = DEFAULT_CONFIG) -> ResolverLike:
    """Build a dnspython resolver with this config's timeouts."""
    if dns is None:  # pragma: no cover
        raise RuntimeError(f"dnspython unavailable: {_DNS_IMPORT_ERROR}")
    resolver = dns.resolver.Resolver(configure=True)
    resolver.timeout = config.dns_timeout
    resolver.lifetime = config.dns_lifetime
    if config.dns_nameservers:
        resolver.nameservers = list(config.dns_nameservers)
    return resolver


# dnspython exception names we treat as "no data / lookup failed" (never fatal).
def _dns_exc_types() -> tuple[type[BaseException], ...]:
    if dns is None:  # pragma: no cover
        return (Exception,)
    return (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.resolver.NoResolverConfiguration,
        dns.exception.Timeout,
        dns.exception.DNSException,
    )


def _query(resolver: ResolverLike, qname: str, rdtype: str) -> tuple[list, int | None, str | None]:
    """Return (rdata_list, ttl, error_string). Never raises."""
    try:
        answer = resolver.resolve(qname, rdtype)
    except _dns_exc_types() as exc:
        return [], None, f"{type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - last-resort guard, must not crash
        return [], None, f"{type(exc).__name__}: {exc}"

    ttl = None
    rrset = getattr(answer, "rrset", None)
    if rrset is not None:
        ttl = getattr(rrset, "ttl", None)
    return list(answer), ttl, None


#: Error names meaning "the resolver did not answer", as opposed to "it
#: answered, and the answer was no".
_TIMEOUT_ERRORS = ("Timeout", "LifetimeTimeout", "NoNameservers",
                   "NoResolverConfiguration")


def _both_timed_out(errors: dict[str, str]) -> bool:
    a, aaaa = errors.get("A"), errors.get("AAAA")
    if not a or not aaaa:
        return False
    return (
        any(a.startswith(e) for e in _TIMEOUT_ERRORS)
        and any(aaaa.startswith(e) for e in _TIMEOUT_ERRORS)
    )


def _reverse_lookup(
    resolver: ResolverLike, ip: str
) -> tuple[tuple[str, ...], str | None]:
    if dns is None:  # pragma: no cover
        return (), "dnspython-unavailable"
    try:
        rev_name = dns.reversename.from_address(ip)
    except Exception as exc:  # noqa: BLE001
        return (), f"{type(exc).__name__}: {exc}"
    rdata, _ttl, err = _query(resolver, str(rev_name), "PTR")
    if err:
        return (), err
    names = tuple(str(r).rstrip(".") for r in rdata)
    return names, None


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def resolve_domain(
    host: str,
    registered_domain: str | None = None,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    resolver: ResolverLike | None = None,
    do_reverse_dns: bool = True,
) -> DNSResult:
    """Collect all DNS features for `host` (A/AAAA/PTR/TTL) and its
    `registered_domain` (NS/MX). Safe on any input; never raises.
    """
    host = (host or "").strip().rstrip(".").lower()
    registered_domain = (registered_domain or host or "").strip().rstrip(".").lower()

    # Nothing resolvable: empty input, an IP literal, or a single-label junk host.
    is_ip = False
    if host:
        try:
            ipaddress.ip_address(host)
            is_ip = True
        except ValueError:
            is_ip = False

    if not host or is_ip or "." not in host:
        result_errors = {}
        if is_ip:
            result_errors["query"] = "host-is-ip-literal (no DNS name to resolve)"
        elif not host:
            result_errors["query"] = "empty-host"
        else:
            result_errors["query"] = "not-a-fqdn"
        # Still classify the literal IP if that is what we were handed.
        details: tuple[dict, ...] = ()
        v4: tuple[str, ...] = ()
        v6: tuple[str, ...] = ()
        if is_ip:
            info = classify_ip(host)
            details = (info,)
            if info["version"] == 4:
                v4 = (info["address"],)
            elif info["version"] == 6:
                v6 = (info["address"],)
        return DNSResult(
            host=host, registered_domain=registered_domain, queried=False,
            ipv4_addresses=v4, ipv6_addresses=v6, ip_details=details,
            errors=result_errors,
        )

    if resolver is None:
        try:
            resolver = make_resolver(config)
        except Exception as exc:  # noqa: BLE001
            return DNSResult(
                host=host, registered_domain=registered_domain, queried=False,
                errors={"resolver": f"{type(exc).__name__}: {exc}"},
            )

    errors: dict[str, str] = {}

    a_rdata, a_ttl, a_err = _query(resolver, host, "A")
    if a_err:
        errors["A"] = a_err
    aaaa_rdata, aaaa_ttl, aaaa_err = _query(resolver, host, "AAAA")
    if aaaa_err:
        errors["AAAA"] = aaaa_err

    ipv4 = tuple(dict.fromkeys(str(r) for r in a_rdata))
    ipv6 = tuple(dict.fromkeys(str(r) for r in aaaa_rdata))

    # NXDOMAIN means "this name does not exist" - not "no records of this type".
    # When the name we would query for NS/MX is the SAME name that just came
    # back NXDOMAIN, those two queries cannot succeed either, and on a slow or
    # unreachable resolver each one burns another full `dns_lifetime`. Skipping
    # them loses no signal and removes two thirds of the worst-case DNS time.
    #
    # The guard is deliberately narrow: for a subdomain (`host` !=
    # `registered_domain`), NXDOMAIN on the host says nothing about the
    # registered domain, so NS/MX are still queried.
    name_is_nxdomain = (
        errors.get("A") == "NXDOMAIN" and errors.get("AAAA") == "NXDOMAIN"
    )
    skip_ns_mx = name_is_nxdomain and host == registered_domain
    skip_reason = "skipped: name is NXDOMAIN"

    # A resolver that failed to answer twice in a row is not going to answer the
    # next two either - it is unreachable, not merely quiet about this name. Two
    # more queries would just burn two more full lifetimes (12 s on the default
    # config) to learn the same thing, so stop asking. This applies whatever the
    # name, unlike the NXDOMAIN case above.
    if not skip_ns_mx and _both_timed_out(errors):
        skip_ns_mx = True
        skip_reason = "skipped: resolver unresponsive"

    if skip_ns_mx:
        errors["NS"] = skip_reason
        errors["MX"] = skip_reason
        nameservers: tuple[str, ...] = ()
        mx_hosts: tuple[str, ...] = ()
    else:
        ns_rdata, _ns_ttl, ns_err = _query(resolver, registered_domain, "NS")
        if ns_err:
            errors["NS"] = ns_err
        nameservers = tuple(
            sorted(dict.fromkeys(str(r).rstrip(".").lower() for r in ns_rdata))
        )

        mx_rdata, _mx_ttl, mx_err = _query(resolver, registered_domain, "MX")
        if mx_err:
            errors["MX"] = mx_err
        mx_hosts = tuple(
            sorted(dict.fromkeys(
                str(getattr(r, "exchange", r)).rstrip(".").lower() for r in mx_rdata
            ))
        )

    # TTL: prefer the A record's, fall back to AAAA's.
    ttl = a_ttl if a_ttl is not None else aaaa_ttl

    ip_details = tuple(classify_ip(a) for a in (*ipv4, *ipv6))

    reverse_map: dict[str, tuple[str, ...]] = {}
    all_reverse: list[str] = []
    if do_reverse_dns:
        public_ips = [d["address"] for d in ip_details if d["is_public"]]
        for ip in public_ips[: config.reverse_dns_max_lookups]:
            names, rerr = _reverse_lookup(resolver, ip)
            if names:
                reverse_map[ip] = names
                all_reverse.extend(names)
            elif rerr:
                errors.setdefault("PTR", rerr)

    dns_resolves = bool(ipv4 or ipv6)

    return DNSResult(
        host=host,
        registered_domain=registered_domain,
        queried=True,
        dns_resolves=dns_resolves,
        ipv4_addresses=ipv4,
        ipv6_addresses=ipv6,
        nameservers=nameservers,
        mx_hosts=mx_hosts,
        ttl=ttl,
        reverse_dns_names=tuple(dict.fromkeys(all_reverse)),
        reverse_dns=reverse_map,
        ip_details=ip_details,
        errors=errors,
    )
