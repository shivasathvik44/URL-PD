"""Layer 3 tests - deterministic feature-vector construction.

The model must never see raw strings, training and inference must agree on the
feature order, and missing Layer 2 data must not crash extraction (spec #3, #12).
"""

from __future__ import annotations

import pytest

from detector.config import MLConfig
from detector.feature_pipeline import (
    DOMAIN_AGE_FEATURES,
    FEATURE_NAMES,
    MISSING,
    active_feature_names,
    build_dataset,
    build_feature_vector,
    feature_matrix,
    features_from_dicts,
)


# --- a fake Layer 2 that never touches the network -----------------------
class _FakeIntel:
    def __init__(self, payload):
        self._p = payload

    def as_dict(self):
        return self._p


def fake_layer2(url, config, *, do_reverse_dns=True, do_domain_age=True):
    return _FakeIntel({
        "registered_domain": "example.com", "is_ip_host": False,
        "dns_resolves": True, "ipv4_count": 2, "ipv6_count": 1, "ip_count": 3,
        "has_ipv4": True, "has_ipv6": True, "has_ns": True, "ns_count": 4,
        "has_mx": True, "mx_count": 2, "ttl": 300, "ttl_available": True,
        "public_ip_count": 3, "private_ip_count": 0, "reserved_ip_count": 0,
        "loopback_ip_count": 0, "multicast_ip_count": 0,
        "reverse_dns_available": True, "dns_errors": {},
        "domain_age_available": True, "domain_age_days": 4000, "is_new_domain": False,
    })


# --------------------------------------------------------------------------
# 4 & 5. feature vector creation + stable ordering
# --------------------------------------------------------------------------
def test_feature_vector_is_ordered_numeric_and_stringfree():
    vec = build_feature_vector("https://paypal.security-check.xyz/login",
                               run_layer2=False)
    assert list(vec.keys()) == active_feature_names()
    assert all(isinstance(v, float) for v in vec.values())
    # not one raw-string field leaked in
    assert not any(isinstance(v, str) for v in vec.values())


def test_feature_order_is_deterministic_across_calls():
    a = build_feature_vector("http://a.example/x", run_layer2=False)
    b = build_feature_vector("http://a.example/x", run_layer2=False)
    assert list(a.keys()) == list(b.keys())
    assert a == b                                   # deterministic values too


def test_feature_names_match_schema_length():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))   # no dupes
    assert set(DOMAIN_AGE_FEATURES).issubset(FEATURE_NAMES)


def test_layer2_features_populated_when_present():
    vec = build_feature_vector("https://example.com/", layer2_fn=fake_layer2)
    assert vec["l2_present"] == 1.0
    assert vec["l2_dns_resolves"] == 1.0
    assert vec["l2_ns_count"] == 4.0
    assert vec["l2_domain_age_days"] == 4000.0
    assert vec["l2_ttl"] == 300.0


# --------------------------------------------------------------------------
# 12. missing Layer 2 features -> sentinels, no crash
# --------------------------------------------------------------------------
def test_missing_layer2_uses_sentinels_not_crash():
    vec = build_feature_vector("https://example.com/", run_layer2=False)
    assert vec["l2_present"] == 0.0
    assert vec["l2_ttl"] == MISSING
    assert vec["l2_domain_age_days"] == MISSING
    assert vec["l2_domain_age_available"] == 0.0
    assert vec["l2_dns_resolves"] == 0.0            # absence != "resolves"


def test_layer2_error_payload_is_treated_as_missing():
    err = lambda *a, **k: _FakeIntel({"_layer2_error": "Timeout"})
    vec = build_feature_vector("https://x.example/", layer2_fn=err)
    assert vec["l2_present"] == 0.0
    assert vec["l2_ttl"] == MISSING


# --------------------------------------------------------------------------
# 13. malformed URL handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "not a url", "http://", "ht!tp://%%%", None])
def test_malformed_urls_still_produce_full_vector(bad):
    vec = build_feature_vector(bad, run_layer2=False)
    assert list(vec.keys()) == active_feature_names()
    assert all(isinstance(v, float) for v in vec.values())


# --------------------------------------------------------------------------
# domain-age features can be excluded wholesale (leakage experiment, spec #2)
# --------------------------------------------------------------------------
def test_exclude_domain_age_features():
    cfg = MLConfig(include_domain_age_features=False)
    names = active_feature_names(cfg)
    assert all(f not in names for f in DOMAIN_AGE_FEATURES)
    vec = build_feature_vector("https://example.com/", ml_config=cfg, layer2_fn=fake_layer2)
    assert "l2_domain_age_days" not in vec
    assert list(vec.keys()) == names


# --------------------------------------------------------------------------
# build_dataset -> aligned feature frame
# --------------------------------------------------------------------------
def test_build_dataset_shape_and_alignment():
    urls = ["https://good.example/", "http://paypal.verify-account.tk/login",
            "not a url"]
    labels = [0, 1, 1]
    df, stats = build_dataset(urls, labels, sources=["s", "s", "s"],
                              dates=[None, None, None], run_layer2=False)
    assert len(df) == 3
    assert stats["n_features"] == len(active_feature_names())
    for col in (*active_feature_names(), "url", "label", "source", "date", "extraction_ok"):
        assert col in df.columns
    assert list(df["label"]) == labels
    # feature_matrix returns exactly the model columns, in order, as float64
    X = feature_matrix(df)
    assert list(X.columns) == active_feature_names()
    assert str(X.dtypes.iloc[0]) == "float64"


def test_features_from_dicts_handles_none_layer2():
    l1 = {"url": "x", "url_length": 5, "scheme": "http", "entropy": 1.2,
          "keyword_login": True}
    out = features_from_dicts(l1, None)
    assert out["l1_url_length"] == 5.0
    assert out["l1_keyword_login"] == 1.0
    assert out["l2_present"] == 0.0
