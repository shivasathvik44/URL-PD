"""The authoritative Layer 3 feature schema.

One module owns the feature names, their order, their sentinels and the schema
hash. Training, validation, inference and the caches all import from here, so a
feature can never be added, renamed or reordered in one place and forgotten in
another.

Two guards hang off this module:

  * `SCHEMA_VERSION` is bumped by hand whenever the *meaning* of a feature
    changes (not just its name). A saved model or a cached payload carrying a
    different version is rejected rather than silently reinterpreted.
  * `schema_hash()` fingerprints the exact ordered name list, so an accidental
    reordering is caught even when the version was not bumped.

`feature_pipeline.py` computes the values; this module defines what they are.
"""

from __future__ import annotations

import hashlib
import json

#: Bump when a feature's *semantics* change (a renamed/reordered list is caught
#: by the hash; a changed meaning under the same name is not).
#:
#: v1 - initial 65-feature schema.
#: v2 - `l1_has_at_symbol` now means "userinfo/@ in the authority", not
#:      "@ anywhere in the URL"; `l1_levenshtein_distance` uses the MISSING
#:      sentinel for "unknown" instead of collapsing to 0.0 (which read as
#:      "exact brand match").
SCHEMA_VERSION = 2

#: Sentinel for "we have no number here" (unknown TTL, unknown domain age,
#: unknown edit distance). -1 is safe because every real value is >= 0, and it
#: always travels with a companion `*_available`-style flag where one exists.
MISSING = -1.0


# --- feature groups --------------------------------------------------------
# Layer 1 booleans -> 0/1
L1_BOOL: tuple[str, ...] = (
    "scheme_present", "is_malformed", "has_ip", "has_at_symbol",
    "has_double_slash", "has_port", "is_suspicious_port", "brand_in_wrong_place",
    "is_typosquat_suspicious", "is_exact_brand_match", "is_punycode",
    "contains_non_ascii", "is_suspicious_tld",
)
L1_NUM: tuple[str, ...] = (
    "url_length", "domain_length", "registered_domain_length", "dot_count",
    "hyphen_count", "digit_count", "special_char_count", "subdomain_depth",
    "path_length", "query_length", "fragment_length", "keyword_count",
    "levenshtein_distance", "entropy",
)
L1_KEYWORDS: tuple[str, ...] = (
    "keyword_login", "keyword_verify", "keyword_secure", "keyword_account",
    "keyword_update", "keyword_confirm", "keyword_wallet", "keyword_kyc",
    "keyword_signin", "keyword_password", "keyword_payment",
)

L2_BOOL: tuple[str, ...] = (
    "dns_resolves", "has_ipv4", "has_ipv6", "has_ns", "has_mx", "ttl_available",
    "reverse_dns_available", "is_ip_host",
)
L2_NUM: tuple[str, ...] = (
    "ipv4_count", "ipv6_count", "ip_count", "ns_count", "mx_count",
    "public_ip_count", "private_ip_count", "reserved_ip_count",
    "loopback_ip_count", "multicast_ip_count",
)

#: Domain-age features are isolated so (a) they can be excluded wholesale and
#: (b) the importance report can measure how much the model leans on them.
DOMAIN_AGE_FEATURES: tuple[str, ...] = (
    "l2_domain_age_available",
    "l2_domain_age_days",
    "l2_is_new_domain",
)

#: Features whose "unknown" value is MISSING rather than 0.0.
SENTINEL_FEATURES: frozenset[str] = frozenset(
    {"l2_ttl", "l2_domain_age_days", "l1_levenshtein_distance"}
)


def _canonical_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names += [f"l1_{n}" for n in L1_NUM]
    names += [f"l1_{n}" for n in L1_BOOL]
    names += [f"l1_{n}" for n in L1_KEYWORDS]
    names += ["l1_ip_version", "l1_port", "l1_scheme_is_https"]
    names += ["l2_present", "l2_dns_error_count"]
    names += [f"l2_{n}" for n in L2_NUM]
    names += [f"l2_{n}" for n in L2_BOOL]
    names += ["l2_ttl"]
    names += list(DOMAIN_AGE_FEATURES)
    return tuple(names)


#: Canonical, complete ordering. The *active* subset (see `active_feature_names`)
#: is what a given model is actually trained on.
FEATURE_NAMES: tuple[str, ...] = _canonical_feature_names()


def active_feature_names(include_domain_age: bool = True) -> list[str]:
    """The feature order fed to the model, optionally without domain age."""
    if include_domain_age:
        return list(FEATURE_NAMES)
    return [n for n in FEATURE_NAMES if n not in DOMAIN_AGE_FEATURES]


def schema_hash(feature_names, *, version: int = SCHEMA_VERSION) -> str:
    """Stable short fingerprint of an ordered feature list + schema version."""
    blob = json.dumps(
        {"version": version, "features": list(feature_names)}, sort_keys=False
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class SchemaMismatch(ValueError):
    """Raised when two feature schemas that must agree do not.

    Carrying this as a hard error (rather than a warning or a silent coercion)
    is deliberate: a model scoring a differently-ordered or differently-meaning
    feature vector produces confident nonsense.
    """


def describe(feature_names) -> dict:
    """Serialisable description of a schema, for model/cache metadata."""
    names = list(feature_names)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_features": len(names),
        "feature_names": names,
        "schema_hash": schema_hash(names),
    }


def validate(
    expected,
    actual,
    *,
    expected_version: int = SCHEMA_VERSION,
    actual_version: int | None = None,
    context: str = "feature schema",
) -> None:
    """Raise `SchemaMismatch` unless the two schemas agree exactly.

    Compares the schema version first (a changed version means changed
    *meaning*, so names matching is not enough), then the ordered names.
    """
    if actual_version is not None and actual_version != expected_version:
        raise SchemaMismatch(
            f"{context}: schema version mismatch - expected v{expected_version}, "
            f"got v{actual_version}. The feature definitions changed; retrain "
            f"the model (python -m detector train --sample) or clear the cache."
        )

    expected_names, actual_names = list(expected), list(actual)
    if expected_names == actual_names:
        return

    missing = [n for n in expected_names if n not in actual_names]
    extra = [n for n in actual_names if n not in expected_names]
    if missing or extra:
        raise SchemaMismatch(
            f"{context}: feature set differs - missing={missing[:10]} "
            f"extra={extra[:10]} (expected {len(expected_names)} features, "
            f"got {len(actual_names)})"
        )
    # Same set, different order - the dangerous silent case.
    first_diff = next(
        i for i, (a, b) in enumerate(zip(expected_names, actual_names)) if a != b
    )
    raise SchemaMismatch(
        f"{context}: feature ORDER differs at position {first_diff} - "
        f"expected {expected_names[first_diff]!r}, got {actual_names[first_diff]!r}. "
        f"Scoring with a reordered vector would silently produce wrong results."
    )
