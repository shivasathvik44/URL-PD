"""Layer 3, part 5: inference.

    Layer 1 + Layer 2 output -> feature vector -> Random Forest -> P(phishing)

Two entry points, deliberately:

  * `Predictor.predict_from_layers(layer1, layer2)` scores layer results the
    caller already has. `pipeline.analyze_url` uses this, so one URL is parsed
    once and its domain looked up once.
  * `Predictor.predict_url(url)` is the standalone path that does its own
    Layer 1/2 extraction, for callers that only want a probability.

Explainability caveat: `top_features` is the model's *global*
`feature_importances_` intersected with this URL's non-default values. That is
a hint, NOT a faithful per-prediction attribution - Random Forest importances
are dataset-global and impurity-biased. Install the `shap` extra for real
per-prediction explanations.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_ML_CONFIG, DetectorConfig, MLConfig
from .feature_pipeline import build_feature_vector, features_from_dicts
from .feature_schema import MISSING
from .layer2_cache import Layer2Cache
from .model import PhishingModel


class ModelNotFound(FileNotFoundError):
    """No trained model at the configured location."""


# --- process-wide model cache ----------------------------------------------
# Keyed by (directory, mtime) so retraining is picked up automatically: an
# lru_cache on the directory alone would serve a stale model for the life of
# the process, which is exactly the kind of silent staleness this project
# tries to avoid elsewhere.
_MODEL_CACHE: dict[tuple[str, float], PhishingModel] = {}
_MODEL_LOCK = threading.Lock()


def load_model(
    models_dir: str | None = None,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    *,
    use_cache: bool = True,
) -> PhishingModel:
    """Load (and memoise) the trained model. Raises `ModelNotFound` if absent."""
    directory = Path(models_dir or ml_config.models_dir)
    model_path = directory / ml_config.model_filename
    if not model_path.exists():
        raise ModelNotFound(
            f"no trained model at {model_path}. Train one with:  "
            f"python -m detector train --sample"
        )

    mtime = model_path.stat().st_mtime
    key = (str(directory), mtime)

    if use_cache:
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    model = PhishingModel.load(directory, ml_config)

    if use_cache:
        with _MODEL_LOCK:
            _MODEL_CACHE.clear()      # only ever keep the current model
            _MODEL_CACHE[key] = model
    return model


def clear_model_cache() -> None:
    """Drop the memoised model (tests, and after retraining in-process)."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


class Predictor:
    """Holds one loaded model; `predict` is cheap to call repeatedly."""

    def __init__(
        self,
        model: PhishingModel | None = None,
        *,
        models_dir: str | None = None,
        config: DetectorConfig = DEFAULT_CONFIG,
        ml_config: MLConfig = DEFAULT_ML_CONFIG,
        cache: Layer2Cache | None = None,
        threshold: float | None = None,
    ):
        self.config = config
        self.ml_config = ml_config
        self.cache = cache
        self.model = model or load_model(models_dir, ml_config)
        self.threshold = (
            ml_config.default_threshold if threshold is None else float(threshold)
        )

    # ------------------------------------------------------------------
    def predict_from_layers(self, layer1, layer2=None) -> float:
        """P(phishing) from layer results the caller already computed.

        The path `pipeline.analyze_url` uses - no re-parsing, no second DNS
        lookup.
        """
        feats = features_from_dicts(
            layer1.as_dict(),
            layer2.as_dict() if layer2 is not None else None,
            self.ml_config,
        )
        return self._score(feats)

    def _score(self, feats: dict) -> float:
        ordered = [feats[name] for name in self.model.feature_names]
        return float(self.model.predict_proba([ordered])[0])

    # ------------------------------------------------------------------
    def predict_url(
        self,
        url: str,
        *,
        threshold: float | None = None,
        run_layer2: bool = True,
        explain: bool = True,
        top_k: int = 8,
    ) -> dict:
        """Standalone Layer 1-3 scoring for one URL string."""
        thr = self.threshold if threshold is None else float(threshold)
        vec = build_feature_vector(
            url, config=self.config, ml_config=self.ml_config,
            run_layer2=run_layer2, cache=self.cache, with_meta=True,
        )
        meta_l1 = vec.pop("_layer1", {})
        meta_l2 = vec.pop("_layer2", None)

        feats = {k: vec[k] for k in self.model.feature_names}
        prob = self._score(feats)
        label = int(prob >= thr)

        result = {
            "url": url,
            "prediction": "phishing" if label else "benign",
            "phishing_probability": round(prob, 4),
            "threshold": thr,
            "model_version": self.model.info.model_version if self.model.info else None,
            "model_trained_at": self.model.info.trained_at if self.model.info else None,
            "layer2_used": run_layer2 and (meta_l2 is not None)
            and "_layer2_error" not in (meta_l2 or {}),
            "registered_domain": meta_l1.get("registered_domain"),
        }
        if explain:
            result["top_features"] = self._contributing_features(feats, top_k)
            result["explainability_note"] = (
                "Global Random Forest importances weighted by this URL's active "
                "features - a hint, not a faithful per-prediction attribution."
            )
        return result

    __call__ = predict_url

    # ------------------------------------------------------------------
    def _contributing_features(self, feats: dict, top_k: int) -> list[dict]:
        importances = dict(self.model.feature_importances())
        scored = []
        for name, value in feats.items():
            imp = importances.get(name, 0.0)
            # "active" = the feature carries information for this URL: a set flag,
            # a non-zero count, or a known (non-sentinel) numeric value.
            active = value not in (0.0, MISSING)
            scored.append((name, value, imp, active))
        scored.sort(key=lambda t: (t[3], t[2]), reverse=True)
        return [
            {"feature": n, "value": v, "global_importance": round(imp, 5),
             "active_for_this_url": bool(active)}
            for n, v, imp, active in scored[:top_k]
        ]


# --- module-level convenience -------------------------------------------
def get_predictor(
    models_dir: str | None = None,
    *,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    **predictor_kwargs,
) -> Predictor:
    """Build a Predictor over the (memoised) model for `models_dir`.

    Unlike a cached singleton, this never lets a one-off call with a custom
    directory replace the default for every later caller - the *model* is
    cached, the Predictor wrapper is cheap.
    """
    model = load_model(models_dir, ml_config)
    return Predictor(model, ml_config=ml_config, **predictor_kwargs)


def predict_url(
    url: str,
    *,
    threshold: float | None = None,
    run_layer2: bool = True,
    explain: bool = True,
    top_k: int = 8,
    models_dir: str | None = None,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
) -> dict:
    """One-shot Layer 1-3 prediction. For the full four-layer verdict, use
    `detector.analyze_url()` instead."""
    return get_predictor(models_dir, ml_config=ml_config).predict_url(
        url, threshold=threshold, run_layer2=run_layer2, explain=explain,
        top_k=top_k,
    )
