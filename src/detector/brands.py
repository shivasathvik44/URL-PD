"""Brand-impersonation helpers: Levenshtein distance, nearest-brand lookup,
typosquat proximity, and "brand shown in the wrong part of the URL".

None of these functions decide "phishing". They return features; later layers
weigh them.
"""

from __future__ import annotations

from dataclasses import dataclass


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute, unit cost).

    Iterative two-row implementation - O(len(a) * len(b)) time, O(min) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # keep the shorter string on the inner axis to minimise memory
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class BrandProximity:
    """Result of comparing one registered-domain label against a brand list."""

    closest_brand: str | None
    min_distance: int | None          # None only when there are no brands
    is_typosquat_suspicious: bool
    is_exact_brand: bool              # label IS a brand (legit, not a typo)


def closest_brand(label: str, brands: tuple[str, ...]) -> tuple[str | None, int | None]:
    """Return (brand, distance) for the brand nearest `label` by edit distance."""
    best_brand: str | None = None
    best_distance: int | None = None
    for brand in brands:
        d = levenshtein(label, brand)
        if best_distance is None or d < best_distance:
            best_brand, best_distance = brand, d
    return best_brand, best_distance


def brand_proximity(
    label: str,
    brands: tuple[str, ...],
    *,
    max_distance: int,
    min_label_len: int,
) -> BrandProximity:
    label = (label or "").lower()
    brand, distance = closest_brand(label, brands)

    if not label or brand is None or distance is None:
        return BrandProximity(brand, distance, False, False)

    is_exact = distance == 0
    suspicious = (
        not is_exact
        and distance <= max_distance
        and len(label) >= min_label_len
        # length must be comparable - "git" vs "github" is not a typo
        and abs(len(label) - len(brand)) <= max_distance
    )
    return BrandProximity(brand, distance, suspicious, is_exact)


def brand_in_wrong_place(
    *,
    subdomain: str,
    path: str,
    domain_label: str,
    brands: tuple[str, ...],
) -> tuple[bool, str | None]:
    """True when a brand name appears in the subdomain or path but the actual
    registered-domain label is a *different* brand/word.

    e.g. paypal.security-check.xyz/login -> ('paypal' in subdomain, domain
    label is 'security-check')  =>  (True, 'paypal')
    """
    haystack = f"{subdomain} {path}".lower()
    domain_label = (domain_label or "").lower()
    for brand in brands:
        if brand in haystack and domain_label != brand:
            return True, brand
    return False, None
