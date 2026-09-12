"""Layer 3, part 3: the classifier wrapper.

`PhishingModel` wraps a scikit-learn tree ensemble (Random Forest to start) and
adds the things a production model needs around it:

  * the ordered feature schema from `feature_schema.py`, re-checked on load AND
    on every predict;
  * probability output as a single P(phishing) in [0, 1];
  * a configurable decision threshold, never hard-wired to 0.5;
  * feature-importance extraction incl. a "how much is this leaning on domain
    age?" summary;
  * joblib save/load of model + schema + metadata.

The load-time check compares the saved schema against **the current code's**
schema, not merely against its own copy. A model trained on a different
feature set - or the same names in a different order, or the same names with
changed meaning (schema version) - is refused with `SchemaMismatch` rather than
scored silently, because a mismatched vector produces confident nonsense.

`_build_estimator` is the only place the algorithm is chosen, so XGBoost /
LightGBM can be slotted in later without touching the rest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import DEFAULT_ML_CONFIG, MLConfig
from .feature_pipeline import active_feature_names
from .feature_schema import (
    DOMAIN_AGE_FEATURES,
    SCHEMA_VERSION,
    SchemaMismatch,
    schema_hash,
    validate,
)

MODEL_FORMAT_VERSION = 2


def _build_estimator(ml_config: MLConfig):
    """Algorithm factory. Add elif branches for xgboost/lightgbm later."""
    kind = ml_config.model_kind.lower()
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=ml_config.n_estimators,
            max_depth=ml_config.max_depth,
            min_samples_leaf=ml_config.min_samples_leaf,
            min_samples_split=ml_config.min_samples_split,
            max_features=ml_config.max_features,
            class_weight=ml_config.class_weight,
            n_jobs=ml_config.n_jobs,
            random_state=ml_config.random_seed,
        )
    raise ValueError(f"unsupported model_kind {ml_config.model_kind!r}")


@dataclass
class ModelInfo:
    model_version: str
    model_kind: str
    format_version: int
    trained_at: str
    random_seed: int
    n_features: int
    feature_names: list[str]
    params: dict
    schema_hash: str
    #: Feature *semantics* version from feature_schema.SCHEMA_VERSION.
    schema_version: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


class PhishingModel:
    def __init__(self, ml_config: MLConfig = DEFAULT_ML_CONFIG,
                 feature_names: list[str] | None = None):
        self.ml_config = ml_config
        self.feature_names = feature_names or active_feature_names(ml_config)
        self.estimator = _build_estimator(ml_config)
        self.info: ModelInfo | None = None

    # -- helpers -------------------------------------------------------
    def _params(self) -> dict:
        return {
            "model_kind": self.ml_config.model_kind,
            "n_estimators": self.ml_config.n_estimators,
            "max_depth": self.ml_config.max_depth,
            "min_samples_leaf": self.ml_config.min_samples_leaf,
            "min_samples_split": self.ml_config.min_samples_split,
            "max_features": self.ml_config.max_features,
            "class_weight": self.ml_config.class_weight,
        }

    def _as_matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            extra = [c for c in X.columns if c not in self.feature_names
                     and not c.startswith("_") and c not in
                     ("url", "label", "source", "date", "extraction_ok",
                      "registered_domain")]
            if missing:
                raise SchemaMismatch(f"missing feature columns: {missing}")
            if extra:
                raise SchemaMismatch(f"unexpected feature columns: {extra}")
            X = X[self.feature_names]
            return X.to_numpy(dtype="float64")
        arr = np.asarray(X, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.feature_names):
            raise SchemaMismatch(
                f"expected {len(self.feature_names)} features, got {arr.shape[1]}"
            )
        return arr

    # -- train / predict --------------------------------------------
    def fit(self, X, y) -> PhishingModel:
        Xm = self._as_matrix(X)
        y = np.asarray(y).astype(int)
        self.estimator.fit(Xm, y)
        params = self._params()
        h = schema_hash(self.feature_names)
        self.info = ModelInfo(
            model_version=f"{self.ml_config.model_kind}-v{MODEL_FORMAT_VERSION}-{h}",
            model_kind=self.ml_config.model_kind,
            format_version=MODEL_FORMAT_VERSION,
            trained_at=datetime.now(timezone.utc).isoformat(),
            random_seed=self.ml_config.random_seed,
            n_features=len(self.feature_names),
            feature_names=list(self.feature_names),
            params=params,
            schema_hash=h,
            schema_version=SCHEMA_VERSION,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """1-D array of P(phishing) for each row."""
        Xm = self._as_matrix(X)
        proba = self.estimator.predict_proba(Xm)
        classes = list(self.estimator.classes_)
        idx = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
        return proba[:, idx]

    def predict(self, X, threshold: float | None = None) -> np.ndarray:
        thr = self.ml_config.default_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= thr).astype(int)

    # -- explainability -----------------------------------------
    def feature_importances(self) -> list[tuple[str, float]]:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            return []
        pairs = list(zip(self.feature_names, (float(v) for v in imp)))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs

    def domain_age_reliance(self) -> dict:
        """Sum of importance mass on domain-age features - the spec's
        'is the model just learning domain age?' check (#2, #10)."""
        imp = dict(self.feature_importances())
        age = {k: imp.get(k, 0.0) for k in DOMAIN_AGE_FEATURES if k in imp}
        return {
            "domain_age_features": age,
            "domain_age_importance_sum": float(sum(age.values())),
            "max_single_feature_importance": max(imp.values()) if imp else 0.0,
        }

    # -- persistence ------------------------------------------
    def save(self, models_dir: str | Path | None = None) -> dict:
        d = Path(models_dir or self.ml_config.models_dir)
        d.mkdir(parents=True, exist_ok=True)
        model_path = d / self.ml_config.model_filename
        schema_path = d / self.ml_config.schema_filename
        joblib.dump({"estimator": self.estimator, "info": self.info.as_dict()},
                    model_path)
        schema_path.write_text(json.dumps({
            "schema_version": self.info.schema_version,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "schema_hash": self.info.schema_hash,
            "model_version": self.info.model_version,
            "default_threshold": self.ml_config.default_threshold,
        }, indent=2))
        return {"model_path": str(model_path), "schema_path": str(schema_path)}

    @classmethod
    def load(cls, models_dir: str | Path | None = None,
             ml_config: MLConfig = DEFAULT_ML_CONFIG,
             *, check_current_schema: bool = True) -> PhishingModel:
        """Load a saved model, refusing anything the current code cannot score.

        Three checks, all fatal:
          1. the sidecar `feature_schema.json` must agree with the pickled info;
          2. the saved schema *version* must match `feature_schema.SCHEMA_VERSION`
             (same names can mean different things across versions);
          3. the saved feature list must equal what the current pipeline
             produces, in the same order.

        `check_current_schema=False` skips (2) and (3) - only for inspecting an
        archived model, never for scoring.
        """
        d = Path(models_dir or ml_config.models_dir)
        payload = joblib.load(d / ml_config.model_filename)
        info = ModelInfo(**payload["info"])
        obj = cls(ml_config=ml_config, feature_names=info.feature_names)
        obj.estimator = payload["estimator"]
        obj.info = info

        schema_path = d / ml_config.schema_filename
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
            validate(
                info.feature_names,
                schema.get("feature_names", []),
                expected_version=info.schema_version,
                actual_version=schema.get("schema_version"),
                context=f"{schema_path.name} vs saved model",
            )

        if check_current_schema:
            validate(
                active_feature_names(ml_config),
                info.feature_names,
                expected_version=SCHEMA_VERSION,
                actual_version=info.schema_version,
                context="current feature pipeline vs saved model",
            )
        return obj
