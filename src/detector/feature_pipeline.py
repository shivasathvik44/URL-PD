"""Layer 3, part 1: turn Layer 1 + Layer 2 output into a numeric feature vector.

    URL -> Layer 1 (lexical) -> Layer 2 (DNS/domain) -> one feature vector -> model

Rules:
  * the model NEVER sees a raw URL string - only the numbers/booleans here;
  * training and inference use the EXACT same feature names in the EXACT same
    order, defined once in `feature_schema.py` (not here);
  * every field is deterministic; missing Layer 2 data becomes the MISSING
    sentinel plus an `*_available` flag, so "unknown" is distinguishable from a
    real value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import pandas as pd

from .config import DEFAULT_CONFIG, DEFAULT_ML_CONFIG, DetectorConfig, MLConfig
from .feature_schema import (
    DOMAIN_AGE_FEATURES,
    FEATURE_NAMES,
    L1_BOOL,
    L1_KEYWORDS,
    L1_NUM,
    L2_BOOL,
    L2_NUM,
    MISSING,
    SCHEMA_VERSION,
)
from .feature_schema import active_feature_names as _schema_active_names
from .layer2_cache import Layer2Cache, get_layer2_dict
from .lexical_features import extract_lexical_features

__all__ = [
    "MISSING",
    "FEATURE_NAMES",
    "DOMAIN_AGE_FEATURES",
    "SCHEMA_VERSION",
    "active_feature_names",
    "extract_layer_dicts",
    "features_from_dicts",
    "build_feature_vector",
    "build_dataset",
    "feature_matrix",
]


def active_feature_names(ml_config: MLConfig = DEFAULT_ML_CONFIG) -> list[str]:
    """The feature order actually fed to the model for this config."""
    return _schema_active_names(ml_config.include_domain_age_features)


# --- helpers -------------------------------------------------------------
def _num(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _bit(value) -> float:
    return 1.0 if bool(value) else 0.0


# --- extraction -------------------------------------------------------
def extract_layer_dicts(
    url: str,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    run_layer2: bool = True,
    cache: Layer2Cache | None = None,
    analyze_url_fn: Callable | None = None,
    layer2_fn: Callable | None = None,
    do_domain_age: bool = True,
    do_reverse_dns: bool = True,
) -> tuple[dict, dict | None]:
    """Return (layer1_as_dict, layer2_as_dict_or_None). Never raises."""
    extract = analyze_url_fn or extract_lexical_features
    try:
        l1 = extract(url, config).as_dict()
    except Exception as exc:  # noqa: BLE001 - Layer 1 should already be safe
        l1 = {"url": url, "is_malformed": True, "_layer1_error": str(exc)}

    l2: dict | None = None
    if run_layer2:
        l2 = get_layer2_dict(
            url, cache=cache, config=config, do_domain_age=do_domain_age,
            do_reverse_dns=do_reverse_dns, analyze_domain_fn=layer2_fn,
        )
    return l1, l2


def features_from_dicts(
    l1: dict,
    l2: dict | None,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
) -> dict[str, float]:
    """Flatten Layer 1 + Layer 2 dicts into the ordered feature mapping."""
    f: dict[str, float] = {}

    for n in L1_NUM:
        f[f"l1_{n}"] = _num(l1.get(n))
    for n in L1_BOOL:
        f[f"l1_{n}"] = _bit(l1.get(n))
    for n in L1_KEYWORDS:
        f[f"l1_{n}"] = _bit(l1.get(n))
    f["l1_ip_version"] = _num(l1.get("ip_version"), 0.0)
    f["l1_port"] = _num(l1.get("port"), 0.0)
    f["l1_scheme_is_https"] = _bit(str(l1.get("scheme", "")).lower() == "https")

    # Edit distance "unknown" (no brand list, or no domain label) must NOT
    # collapse to 0.0 - that is the value meaning "the label IS a brand".
    distance = l1.get("levenshtein_distance")
    f["l1_levenshtein_distance"] = MISSING if distance is None else _num(distance, MISSING)

    has_l2 = bool(l2) and "_layer2_error" not in (l2 or {})
    f["l2_present"] = _bit(has_l2)
    dns_errors = (l2 or {}).get("dns_errors") or {}
    f["l2_dns_error_count"] = _num(len(dns_errors) if isinstance(dns_errors, dict) else 0)

    for n in L2_NUM:
        f[f"l2_{n}"] = _num((l2 or {}).get(n)) if has_l2 else 0.0
    for n in L2_BOOL:
        f[f"l2_{n}"] = _bit((l2 or {}).get(n)) if has_l2 else 0.0

    ttl = (l2 or {}).get("ttl") if has_l2 else None
    f["l2_ttl"] = _num(ttl, MISSING) if ttl is not None else MISSING

    age_days = (l2 or {}).get("domain_age_days") if has_l2 else None
    f["l2_domain_age_available"] = _bit(has_l2 and (l2 or {}).get("domain_age_available"))
    f["l2_domain_age_days"] = _num(age_days, MISSING) if age_days is not None else MISSING
    f["l2_is_new_domain"] = _bit(has_l2 and (l2 or {}).get("is_new_domain"))

    names = active_feature_names(ml_config)
    return {k: f[k] for k in names}


def build_feature_vector(
    url: str,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    run_layer2: bool = True,
    cache: Layer2Cache | None = None,
    analyze_url_fn: Callable | None = None,
    layer2_fn: Callable | None = None,
    do_domain_age: bool = True,
    do_reverse_dns: bool = True,
    with_meta: bool = False,
) -> dict:
    """Single URL -> ordered {feature_name: float}. Deterministic; never raises.

    With `with_meta=True`, also returns the raw Layer 1/2 dicts under
    `_layer1` / `_layer2` (used by the explainer, never fed to the model).
    """
    l1, l2 = extract_layer_dicts(
        url, config=config, run_layer2=run_layer2, cache=cache,
        analyze_url_fn=analyze_url_fn, layer2_fn=layer2_fn,
        do_domain_age=do_domain_age, do_reverse_dns=do_reverse_dns,
    )
    feats = features_from_dicts(l1, l2, ml_config)
    if with_meta:
        return {**feats, "_layer1": l1, "_layer2": l2}
    return feats


def build_dataset(
    urls: Iterable[str],
    labels: Iterable[int] | None = None,
    *,
    sources: Iterable[str] | None = None,
    dates: Iterable | None = None,
    config: DetectorConfig = DEFAULT_CONFIG,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    run_layer2: bool = True,
    cache: Layer2Cache | None = None,
    analyze_url_fn: Callable | None = None,
    layer2_fn: Callable | None = None,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Many URLs -> feature DataFrame (+ label/source/date columns kept aligned).

    Returns (df, stats). Per-URL extraction failures are counted, not raised;
    the row is still emitted with a best-effort (mostly-sentinel) vector and
    `extraction_ok = 0`.
    """
    urls = list(urls)
    labels = list(labels) if labels is not None else [None] * len(urls)
    sources = list(sources) if sources is not None else [None] * len(urls)
    dates = list(dates) if dates is not None else [None] * len(urls)

    names = active_feature_names(ml_config)
    sentinel_row = {
        k: (MISSING if k in ("l2_ttl", "l2_domain_age_days", "l1_levenshtein_distance")
            else 0.0)
        for k in names
    }

    rows: list[dict] = []
    failures = 0

    for i, url in enumerate(urls):
        ok = 1
        try:
            feats = build_feature_vector(
                url, config=config, ml_config=ml_config, run_layer2=run_layer2,
                cache=cache, analyze_url_fn=analyze_url_fn, layer2_fn=layer2_fn,
            )
        except Exception:  # noqa: BLE001 - defensive; build_feature_vector is safe
            feats = dict(sentinel_row)
            ok = 0
            failures += 1

        row = dict(feats)
        row["url"] = url
        row["label"] = labels[i]
        row["source"] = sources[i]
        row["date"] = dates[i]
        row["extraction_ok"] = ok
        rows.append(row)

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... features: {i + 1}/{len(urls)}")

    df = pd.DataFrame(rows, columns=[*names, "url", "label", "source", "date",
                                     "extraction_ok"])
    stats = {
        "n_urls": len(urls),
        "feature_extraction_failures": failures,
        "n_features": len(names),
        "schema_version": SCHEMA_VERSION,
        "layer2_enabled": run_layer2,
    }
    if cache is not None:
        stats["layer2_cache"] = cache.stats()
    return df, stats


def feature_matrix(df: pd.DataFrame, ml_config: MLConfig = DEFAULT_ML_CONFIG) -> pd.DataFrame:
    """Select just the model-input columns, in canonical order, from `df`."""
    names = active_feature_names(ml_config)
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"feature frame is missing columns: {missing}")
    return df[names].astype("float64")
