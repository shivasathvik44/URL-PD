"""Loading a model the current code cannot score must fail loudly.

This is the guard that stops the worst silent failure mode in the project: a
model trained on one feature vector being fed a different one. Scoring a
reordered or differently-meaning vector does not error - it produces confident
nonsense, which is far worse than a crash.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from detector.config import MLConfig
from detector.feature_schema import SCHEMA_VERSION, SchemaMismatch
from detector.model import PhishingModel
from detector.prediction import ModelNotFound, clear_model_cache, load_model


@pytest.fixture
def trained_model(tmp_path):
    """A small real model saved to a temp directory."""
    ml = MLConfig(n_estimators=5, max_depth=3, models_dir=str(tmp_path))
    model = PhishingModel(ml_config=ml)
    rng = np.random.default_rng(0)
    X = rng.random((40, len(model.feature_names)))
    y = (X[:, 0] > 0.5).astype(int)
    model.fit(X, y)
    model.save(tmp_path)
    clear_model_cache()
    return tmp_path, ml


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_matching_model_loads_and_scores(trained_model):
    path, ml = trained_model
    loaded = load_model(str(path), ml)

    assert loaded.info.schema_version == SCHEMA_VERSION
    probability = loaded.predict_proba([[0.0] * len(loaded.feature_names)])[0]
    assert 0.0 <= probability <= 1.0


def test_save_records_the_schema_version(trained_model):
    path, _ml = trained_model
    schema = json.loads((path / "feature_schema.json").read_text())
    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["n_features"] == len(schema["feature_names"])


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------
def test_older_schema_version_is_refused(trained_model):
    """A model whose features MEAN something different is not loadable."""
    path, ml = trained_model
    schema_path = path / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    schema["schema_version"] = SCHEMA_VERSION - 1
    schema_path.write_text(json.dumps(schema))
    clear_model_cache()

    with pytest.raises(SchemaMismatch) as exc:
        load_model(str(path), ml)
    assert "version mismatch" in str(exc.value)
    assert "retrain" in str(exc.value)


def test_sidecar_schema_disagreeing_with_the_model_is_refused(trained_model):
    path, ml = trained_model
    schema_path = path / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_names"] = schema["feature_names"][:-1]
    schema_path.write_text(json.dumps(schema))
    clear_model_cache()

    with pytest.raises(SchemaMismatch):
        load_model(str(path), ml)


def test_model_trained_on_a_different_feature_set_is_refused(tmp_path):
    """Simulates the pipeline gaining/losing a feature after training."""
    ml = MLConfig(n_estimators=5, max_depth=3, models_dir=str(tmp_path))
    short_names = PhishingModel(ml_config=ml).feature_names[:-3]
    model = PhishingModel(ml_config=ml, feature_names=short_names)
    rng = np.random.default_rng(0)
    X = rng.random((30, len(short_names)))
    model.fit(X, (X[:, 0] > 0.5).astype(int))
    model.save(tmp_path)
    clear_model_cache()

    with pytest.raises(SchemaMismatch) as exc:
        load_model(str(tmp_path), ml)
    assert "missing" in str(exc.value)


def test_reordered_features_are_refused(tmp_path):
    """The silent case: same feature set, different order."""
    ml = MLConfig(n_estimators=5, max_depth=3, models_dir=str(tmp_path))
    names = PhishingModel(ml_config=ml).feature_names[:]
    names[0], names[1] = names[1], names[0]

    model = PhishingModel(ml_config=ml, feature_names=names)
    rng = np.random.default_rng(0)
    X = rng.random((30, len(names)))
    model.fit(X, (X[:, 0] > 0.5).astype(int))
    model.save(tmp_path)
    clear_model_cache()

    with pytest.raises(SchemaMismatch) as exc:
        load_model(str(tmp_path), ml)
    assert "ORDER" in str(exc.value)


def test_archived_model_can_be_inspected_without_the_check(trained_model):
    """Escape hatch for inspection - never used on the scoring path."""
    path, ml = trained_model
    schema_path = path / "feature_schema.json"
    schema = json.loads(schema_path.read_text())
    schema["schema_version"] = SCHEMA_VERSION - 1
    schema_path.write_text(json.dumps(schema))

    # The sidecar check still applies; only the current-pipeline check is skipped.
    schema_path.unlink()
    loaded = PhishingModel.load(path, ml, check_current_schema=False)
    assert loaded.info is not None


def test_missing_model_raises_a_helpful_error(tmp_path):
    with pytest.raises(ModelNotFound) as exc:
        load_model(str(tmp_path / "nope"))
    assert "detector train" in str(exc.value)


# ---------------------------------------------------------------------------
# predict-time width check
# ---------------------------------------------------------------------------
def test_wrong_width_vector_is_rejected_at_predict(trained_model):
    path, ml = trained_model
    model = load_model(str(path), ml)

    with pytest.raises(SchemaMismatch):
        model.predict_proba([[0.0] * 3])


# ---------------------------------------------------------------------------
# model cache invalidation
# ---------------------------------------------------------------------------
def test_retraining_is_picked_up_not_served_from_cache(trained_model):
    """An lru_cache on the directory would serve the old model forever."""
    path, ml = trained_model
    first = load_model(str(path), ml)

    retrained = PhishingModel(ml_config=ml)
    rng = np.random.default_rng(7)
    X = rng.random((40, len(retrained.feature_names)))
    retrained.fit(X, (X[:, 1] > 0.5).astype(int))
    import os
    import time

    model_file = path / ml.model_filename
    retrained.save(path)
    # Ensure a distinct mtime even on a coarse-grained filesystem clock.
    os.utime(model_file, (time.time() + 1, time.time() + 1))

    second = load_model(str(path), ml)
    assert second.info.trained_at != first.info.trained_at


def test_same_model_is_served_from_cache(trained_model):
    path, ml = trained_model
    assert load_model(str(path), ml) is load_model(str(path), ml)
