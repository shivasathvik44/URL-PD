"""Layer 3 tests - model training, probability output, save/load, and the
inference API. Training uses a tiny in-memory synthetic frame (fast, offline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detector.config import MLConfig
from detector.feature_pipeline import active_feature_names
from detector.model import PhishingModel, SchemaMismatch
from detector.prediction import Predictor


@pytest.fixture
def tiny_ml_config():
    # small + fast, still a real RandomForest
    return MLConfig(n_estimators=40, max_depth=6, random_seed=0)


@pytest.fixture
def synthetic_frame():
    """Linearly-separable-ish frame over the real feature schema."""
    names = active_feature_names()
    rng = np.random.default_rng(0)
    n = 120
    rows = []
    for i in range(n):
        phishing = i % 2
        base = dict.fromkeys(names, 0.0)
        # phishing rows: high keyword count, suspicious tld, long url
        base["l1_keyword_count"] = rng.integers(2, 6) if phishing else 0
        base["l1_is_suspicious_tld"] = float(phishing)
        base["l1_url_length"] = rng.integers(60, 120) if phishing else rng.integers(15, 40)
        base["l1_entropy"] = rng.uniform(3.5, 4.5) if phishing else rng.uniform(2.0, 3.0)
        base["l2_domain_age_days"] = (rng.integers(0, 25) if phishing
                                      else rng.integers(400, 5000))
        base["l2_is_new_domain"] = float(phishing)
        base["label"] = phishing
        rows.append(base)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8, 9, 10. training + prediction + probability
# --------------------------------------------------------------------------
def test_fit_predict_proba_range(tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    model = PhishingModel(tiny_ml_config, names)
    X = synthetic_frame[names]
    y = synthetic_frame["label"]
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()

    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    # should learn the (obvious) synthetic signal well
    assert (preds == y.to_numpy()).mean() > 0.9


def test_threshold_changes_predictions(tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    model = PhishingModel(tiny_ml_config, names).fit(
        synthetic_frame[names], synthetic_frame["label"])
    p_low = model.predict(synthetic_frame[names], threshold=0.1).sum()
    p_high = model.predict(synthetic_frame[names], threshold=0.9).sum()
    assert p_low >= p_high            # lower threshold flags at least as many


def test_model_info_and_importances(tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    model = PhishingModel(tiny_ml_config, names).fit(
        synthetic_frame[names], synthetic_frame["label"])
    assert model.info is not None
    assert model.info.n_features == len(names)
    assert model.info.feature_names == names
    imp = model.feature_importances()
    assert len(imp) == len(names)
    assert abs(sum(v for _, v in imp) - 1.0) < 1e-6
    age = model.domain_age_reliance()
    assert "domain_age_importance_sum" in age


# --------------------------------------------------------------------------
# 5. feature ordering enforced at predict time
# --------------------------------------------------------------------------
def test_schema_mismatch_raises(tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    model = PhishingModel(tiny_ml_config, names).fit(
        synthetic_frame[names], synthetic_frame["label"])
    with pytest.raises(SchemaMismatch):
        model.predict_proba([[1.0, 2.0, 3.0]])                 # wrong width
    with pytest.raises(SchemaMismatch):
        model.predict_proba(synthetic_frame[names[:-1]])       # missing a column


# --------------------------------------------------------------------------
# 11. model save / load round-trip
# --------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    cfg = MLConfig(n_estimators=40, max_depth=6, random_seed=0,
                   models_dir=str(tmp_path))
    model = PhishingModel(cfg, names).fit(synthetic_frame[names], synthetic_frame["label"])
    model.save(str(tmp_path))
    assert (tmp_path / "random_forest.joblib").exists()
    assert (tmp_path / "feature_schema.json").exists()

    reloaded = PhishingModel.load(str(tmp_path), ml_config=cfg)
    assert reloaded.feature_names == names
    assert reloaded.info.model_version == model.info.model_version
    np.testing.assert_allclose(
        reloaded.predict_proba(synthetic_frame[names]),
        model.predict_proba(synthetic_frame[names]),
    )


def test_load_detects_tampered_schema(tmp_path, tiny_ml_config, synthetic_frame):
    names = active_feature_names()
    cfg = MLConfig(models_dir=str(tmp_path), n_estimators=20, random_seed=0)
    PhishingModel(cfg, names).fit(
        synthetic_frame[names], synthetic_frame["label"]).save(str(tmp_path))
    (tmp_path / "feature_schema.json").write_text('{"feature_names": ["nope"]}')
    with pytest.raises(SchemaMismatch):
        PhishingModel.load(str(tmp_path), ml_config=cfg)


# --------------------------------------------------------------------------
# inference API (Predictor) - offline, model injected
# --------------------------------------------------------------------------
@pytest.fixture
def trained_predictor(tmp_path, synthetic_frame):
    names = active_feature_names()
    cfg = MLConfig(models_dir=str(tmp_path), n_estimators=60, max_depth=8,
                   random_seed=0, default_threshold=0.5)
    model = PhishingModel(cfg, names).fit(synthetic_frame[names], synthetic_frame["label"])
    return Predictor(model=model, ml_config=cfg)


def test_predict_url_output_contract(trained_predictor):
    out = trained_predictor.predict_url("https://paypal.secure-login.tk/verify/account",
                                        run_layer2=False)
    assert out["url"].startswith("https://paypal")
    assert out["prediction"] in {"phishing", "benign"}
    assert 0.0 <= out["phishing_probability"] <= 1.0
    assert "threshold" in out and "model_version" in out
    assert isinstance(out["top_features"], list) and out["top_features"]
    assert "explainability_note" in out                # caveat is surfaced


def test_predict_url_threshold_override(trained_predictor):
    lo = trained_predictor.predict_url("http://x.example/", run_layer2=False, threshold=0.05)
    hi = trained_predictor.predict_url("http://x.example/", run_layer2=False, threshold=0.95)
    assert lo["phishing_probability"] == hi["phishing_probability"]   # same prob
    # label may differ; at minimum the higher threshold is not more aggressive
    assert not (hi["prediction"] == "phishing" and lo["prediction"] == "benign")


def test_predict_url_handles_malformed(trained_predictor):
    out = trained_predictor.predict_url("not a url", run_layer2=False)
    assert out["prediction"] in {"phishing", "benign"}
    assert 0.0 <= out["phishing_probability"] <= 1.0
