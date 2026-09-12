"""The authoritative feature schema and its guards.

The point of this module is that a model can never be scored against a feature
vector it was not trained on. These tests cover the three ways that could
silently happen: a changed feature set, a changed *order*, and a changed
*meaning* under unchanged names.
"""

from __future__ import annotations

import json

import pytest

from detector.feature_schema import (
    DOMAIN_AGE_FEATURES,
    FEATURE_NAMES,
    MISSING,
    SCHEMA_VERSION,
    SchemaMismatch,
    active_feature_names,
    describe,
    schema_hash,
    validate,
)


# ---------------------------------------------------------------------------
# the schema itself
# ---------------------------------------------------------------------------
def test_feature_names_are_unique_and_ordered():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert all(n.startswith(("l1_", "l2_")) for n in FEATURE_NAMES)


def test_domain_age_features_can_be_excluded_wholesale():
    with_age = active_feature_names(True)
    without = active_feature_names(False)

    assert len(with_age) - len(without) == len(DOMAIN_AGE_FEATURES)
    assert not any(n in without for n in DOMAIN_AGE_FEATURES)
    # Excluding them must not disturb the order of what remains.
    assert without == [n for n in with_age if n not in DOMAIN_AGE_FEATURES]


def test_schema_hash_is_stable_and_order_sensitive():
    names = list(FEATURE_NAMES)
    assert schema_hash(names) == schema_hash(list(FEATURE_NAMES))

    swapped = names[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert schema_hash(swapped) != schema_hash(names)


def test_schema_hash_changes_with_version():
    names = list(FEATURE_NAMES)
    assert schema_hash(names, version=1) != schema_hash(names, version=2)


def test_describe_is_json_serialisable():
    payload = describe(FEATURE_NAMES)
    json.dumps(payload)
    assert payload["n_features"] == len(FEATURE_NAMES)
    assert payload["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_identical_schemas_validate():
    validate(FEATURE_NAMES, list(FEATURE_NAMES))       # must not raise


def test_missing_feature_is_rejected():
    with pytest.raises(SchemaMismatch) as exc:
        validate(FEATURE_NAMES, list(FEATURE_NAMES)[:-1])
    assert "missing" in str(exc.value)


def test_extra_feature_is_rejected():
    with pytest.raises(SchemaMismatch) as exc:
        validate(FEATURE_NAMES, [*FEATURE_NAMES, "l1_brand_new_feature"])
    assert "extra" in str(exc.value)


def test_reordered_schema_is_rejected_with_the_position():
    """The dangerous silent case: same features, different order."""
    reordered = list(FEATURE_NAMES)
    reordered[0], reordered[5] = reordered[5], reordered[0]

    with pytest.raises(SchemaMismatch) as exc:
        validate(FEATURE_NAMES, reordered)
    message = str(exc.value)
    assert "ORDER" in message
    assert "position 0" in message


def test_version_mismatch_is_rejected_even_with_identical_names():
    """Same names can mean different things across schema versions."""
    with pytest.raises(SchemaMismatch) as exc:
        validate(FEATURE_NAMES, list(FEATURE_NAMES),
                 expected_version=2, actual_version=1)
    assert "version mismatch" in str(exc.value)
    assert "retrain" in str(exc.value)


def test_validation_message_names_the_context():
    with pytest.raises(SchemaMismatch) as exc:
        validate(FEATURE_NAMES, [], context="my cache")
    assert "my cache" in str(exc.value)


# ---------------------------------------------------------------------------
# sentinel semantics
# ---------------------------------------------------------------------------
def test_missing_sentinel_is_negative_so_it_cannot_collide():
    """Every real value of a sentinel-bearing feature is >= 0."""
    assert MISSING < 0


def test_unknown_edit_distance_is_not_encoded_as_zero():
    """Distance 0 means 'the label IS a brand' - the opposite of 'unknown'."""
    from detector.config import DetectorConfig
    from detector.feature_pipeline import features_from_dicts
    from detector.lexical_features import extract_lexical_features

    no_brands = DetectorConfig(brands=())
    l1 = extract_lexical_features("https://example.com/", no_brands)
    assert l1.levenshtein_distance is None

    feats = features_from_dicts(l1.as_dict(), None)
    assert feats["l1_levenshtein_distance"] == MISSING

    exact = extract_lexical_features("https://google.com/")
    exact_feats = features_from_dicts(exact.as_dict(), None)
    assert exact_feats["l1_levenshtein_distance"] == 0.0
