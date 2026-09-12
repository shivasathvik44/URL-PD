"""Layer 3 - end-to-end training-script test (offline).

Runs `detector.train.main` on a small local CSV with Layer 2 disabled, then
checks the artifacts exist and reload. No network, no live datasets.
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from detector.model import PhishingModel
from detector.train import main as train_main


@pytest.fixture
def mini_csv(tmp_path) -> Path:
    """~160 rows spanning 2025-2026 so the time split has all three buckets."""
    rng = random.Random(0)
    start = datetime(2025, 1, 1)
    rows = []
    kw = ["login", "verify", "secure", "account", "confirm", "wallet"]
    tld = ["xyz", "top", "tk", "click", "gq"]
    for i in range(160):
        d = (start + timedelta(days=rng.randint(0, 600))).isoformat()
        if i % 2:
            u = (f"http://{rng.choice(kw)}-{rng.choice(kw)}."
                 f"{rng.choice(tld)}/{rng.choice(kw)}/{i}")
            lbl = 1
        else:
            u = (f"https://{rng.choice(['nice','calm','open','tiny'])}-"
                 f"{rng.choice(['press','labs','shop','coop'])}{i}.com/"
                 f"{rng.choice(['about','blog','docs'])}")
            lbl = 0
        rows.append({"url": u, "label": lbl, "source": "fixture", "date": d})
    p = tmp_path / "mini.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "label", "source", "date"])
        w.writeheader()
        w.writerows(rows)
    return p


def test_train_main_produces_loadable_artifacts(mini_csv, tmp_path, capsys):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    rc = train_main([
        "--csv", str(mini_csv),
        "--no-layer2",
        "--models-dir", str(models_dir),
        "--reports-dir", str(reports_dir),
        "--seed", "0",
    ])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Precision" in out and "Recall" in out
    assert "Top 15 features by importance" in out
    assert "Domain-age importance share" in out

    # 11. artifacts
    assert (models_dir / "random_forest.joblib").exists()
    schema = json.loads((models_dir / "feature_schema.json").read_text())
    assert schema["n_features"] == len(schema["feature_names"])
    assert (reports_dir / "feature_importance.csv").exists()

    # 18. reproducibility metadata
    meta = json.loads((models_dir / "training_metadata.json").read_text())
    for key in ("trained_at", "random_seed", "model_params", "feature_names",
                "split", "class_imbalance_handling", "metrics",
                "domain_age_reliance"):
        assert key in meta
    assert meta["random_seed"] == 0
    assert meta["split"]["method"].startswith(("time_based", "grouped"))

    # reload + predict
    model = PhishingModel.load(str(models_dir))
    assert model.info.feature_names == schema["feature_names"]
    proba = model.predict_proba([[0.0] * model.info.n_features])
    assert 0.0 <= float(proba[0]) <= 1.0
