"""Layer 1: lexical URL analysis.

`extract_lexical_features(url)` parses a URL and returns a `LexicalFeatures`
dataclass of deterministic, network-free features. It makes **no**
phishing/benign decision - that is for later layers. (The top-level
`detector.analyze_url()` runs *all* layers; this is just Layer 1.)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import tldextract

from .brands import brand_in_wrong_place, brand_proximity
from .config import DEFAULT_CONFIG, DetectorConfig
from .url_parser import ParsedURL, parse_url

# Offline extractor: use tldextract's bundled Public Suffix List snapshot only.
# No cache directory, no network fetch -> deterministic and dependency-free at
# run time (the PSL ships inside the tldextract package).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


@dataclass(frozen=True)
class LexicalFeatures:
    """Every Layer 1 feature for one URL."""

    # --- echo / context -------------------------------------------------
    url: str
    scheme: str
    scheme_present: bool
    hostname: str
    is_malformed: bool

    # --- shape --------------------------------------------------------
    url_length: int
    domain_length: int              # len(hostname)
    registered_domain: str
    registered_domain_length: int
    subdomain: str
    dot_count: int
    hyphen_count: int
    digit_count: int
    special_char_count: int
    subdomain_depth: int
    path_length: int
    query_length: int
    fragment_length: int

    # --- red flags --------------------------------------------------
    has_ip: bool
    ip_version: int | None
    has_at_symbol: bool
    has_double_slash: bool
    has_port: bool
    port: int | None
    is_suspicious_port: bool

    # --- keywords -------------------------------------------------
    keyword_count: int
    keywords_found: tuple[str, ...]
    keyword_flags: dict[str, bool]

    # --- brand impersonation -----------------------------------
    brand_in_wrong_place: bool
    brand_in_wrong_place_name: str | None

    # --- typosquatting ----------------------------------------
    levenshtein_distance: int | None
    closest_brand: str | None
    is_typosquat_suspicious: bool
    is_exact_brand_match: bool

    # --- homograph ------------------------------------------
    is_punycode: bool
    contains_non_ascii: bool

    # --- entropy ---------------------------------------------
    entropy: float

    # --- TLD risk ------------------------------------------
    tld: str
    is_suspicious_tld: bool

    def as_dict(self) -> dict:
        """Flat dict of every feature (keyword_flags expanded to keyword_<name>)."""
        data = asdict(self)
        flags = data.pop("keyword_flags")
        for name, present in flags.items():
            data[f"keyword_{name}"] = present
        return data

    # convenience so callers can treat it like a mapping
    def __getitem__(self, key: str):
        return self.as_dict()[key]


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits/character) of `text`. Empty string -> 0.0."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _count_chars(url: str, special: frozenset[str]) -> tuple[int, int, int, int]:
    dots = hyphens = digits = specials = 0
    for ch in url:
        if ch == ".":
            dots += 1
        elif ch == "-":
            hyphens += 1
        elif ch.isdigit():
            digits += 1
        elif ch in special:
            specials += 1
    return dots, hyphens, digits, specials


def _subdomain_depth(subdomain: str) -> int:
    return len([label for label in subdomain.split(".") if label])


def extract_lexical_features(
    url: str,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> LexicalFeatures:
    """Extract all Layer 1 lexical features from `url`. Never raises on bad input."""
    parsed: ParsedURL = parse_url(url)
    raw = parsed.raw

    # --- registered-domain analysis (PSL-aware, never split on '.') ------
    ext = _EXTRACT(parsed.hostname) if parsed.hostname else _EXTRACT("")
    domain_label = ext.domain or ""
    suffix = ext.suffix or ""
    subdomain = ext.subdomain or ""
    if domain_label and suffix:
        registered_domain = f"{domain_label}.{suffix}"
    elif domain_label:
        # No recognised public suffix (e.g. `.internal`, `.example`, a typo'd
        # TLD). Fall back to the WHOLE hostname, not the last label: collapsing
        # `a.internal` and `b.internal` to "internal" would make two unrelated
        # hosts share a Layer 2 cache entry and look same-origin to the
        # cross-domain form check.
        registered_domain = parsed.hostname
    else:
        registered_domain = ""

    # An IP host - or a hostname we could not trust - has no meaningful
    # registered domain / TLD.
    if parsed.is_ip or parsed.is_malformed:
        domain_label = suffix = subdomain = registered_domain = ""

    # --- shape ------------------------------------------------------
    dots, hyphens, digits, specials = _count_chars(raw, config.special_chars)

    # --- keywords --------------------------------------------------
    lowered = raw.lower()
    kw_flags = {kw: (kw in lowered) for kw in config.normalised_keywords()}
    keywords_found = tuple(kw for kw, hit in kw_flags.items() if hit)
    important = {
        kw: kw_flags.get(kw, kw in lowered)
        for kw in (k.lower() for k in config.important_keywords)
    }

    # --- brand impersonation ----------------------------------
    brands = config.normalised_brands()
    biwp, biwp_name = brand_in_wrong_place(
        subdomain=subdomain,
        path=parsed.path,
        domain_label=domain_label,
        brands=brands,
    )

    # --- typosquatting -------------------------------------
    prox = brand_proximity(
        domain_label,
        brands,
        max_distance=config.typosquat_max_distance,
        min_label_len=config.typosquat_min_label_len,
    )

    # --- homograph ---------------------------------------
    host = parsed.hostname
    is_punycode = "xn--" in host
    contains_non_ascii = any(ord(ch) > 127 for ch in host)

    # --- entropy (of the registered domain, per spec) ---
    entropy_source = registered_domain or host
    entropy = shannon_entropy(entropy_source)

    # --- TLD risk -------------------------------------
    tld_last = suffix.split(".")[-1] if suffix else ""
    is_suspicious_tld = tld_last in config.normalised_suspicious_tlds()

    # --- ports ----------------------------------------
    has_port = parsed.port is not None
    is_suspicious_port = has_port and parsed.port not in config.safe_ports

    return LexicalFeatures(
        url=raw,
        scheme=parsed.scheme,
        scheme_present=parsed.scheme_present,
        hostname=host,
        is_malformed=parsed.is_malformed,
        url_length=len(raw),
        domain_length=len(host),
        registered_domain=registered_domain,
        registered_domain_length=len(registered_domain),
        subdomain=subdomain,
        dot_count=dots,
        hyphen_count=hyphens,
        digit_count=digits,
        special_char_count=specials,
        subdomain_depth=_subdomain_depth(subdomain),
        path_length=len(parsed.path),
        query_length=len(parsed.query),
        fragment_length=len(parsed.fragment),
        has_ip=parsed.is_ip,
        ip_version=parsed.ip_version,
        has_at_symbol=parsed.has_at_symbol,
        has_double_slash=parsed.has_double_slash_in_path,
        has_port=has_port,
        port=parsed.port,
        is_suspicious_port=is_suspicious_port,
        keyword_count=len(keywords_found),
        keywords_found=keywords_found,
        keyword_flags=important,
        brand_in_wrong_place=biwp,
        brand_in_wrong_place_name=biwp_name,
        levenshtein_distance=prox.min_distance,
        closest_brand=prox.closest_brand,
        is_typosquat_suspicious=prox.is_typosquat_suspicious,
        is_exact_brand_match=prox.is_exact_brand,
        is_punycode=is_punycode,
        contains_non_ascii=contains_non_ascii,
        entropy=entropy,
        tld=suffix,
        is_suspicious_tld=is_suspicious_tld,
    )


def registered_domain_of(url: str) -> str:
    """PSL-aware *grouping key* for `url`: which domain does it live on?

    The one place that answers that question - used for Layer 2 cache keys,
    dataset grouping and cross-domain form comparison. It agrees with
    `LexicalFeatures.registered_domain` in every case but one, deliberately:
    for an **IP-literal host** Layer 1 reports `""` (an IP genuinely has no
    registered domain, and the feature should encode that), while this returns
    the IP itself - a key of `""` would collapse every IP-addressed URL in a
    corpus into one shared cache entry.

    Never raises.
    """
    try:
        parsed = parse_url(url)
        if not parsed.hostname:
            return ""
        if parsed.is_ip:
            return parsed.hostname
        ext = _EXTRACT(parsed.hostname)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        # No public suffix: use the full hostname (see the same fallback in
        # `extract_lexical_features` for why the last label alone is wrong).
        return parsed.hostname.lower()
    except Exception:  # noqa: BLE001 - callers rely on this never raising
        return ""
