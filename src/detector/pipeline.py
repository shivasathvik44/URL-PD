"""The one top-level API: `analyze_url(url)`.

    URL
     |
     v  Layer 1  lexical            (offline, deterministic)
     v  Layer 2  DNS / domain intel (network, cached, best-effort)
     v  Layer 3  Random Forest      (tabular classifier)
     |
     +--> triage: blocklist / allowlist / >= HIGH / <= LOW  -> done
     |
     v  Layer 4  page fetch + LLM   (uncertain band only, cached)
     v  decision engine             (explicit rules over all layers)
     v
    AnalysisResult

Everything below `analyze_url` is a building block it composes; nothing else
needs to orchestrate layers. One URL is parsed once, its domain looked up once
(then cached), and its page fetched at most once.

Every step is best-effort: a failure in any layer is recorded in `errors` and
degrades the result (usually to `"uncertain"`) rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    DEFAULT_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_ML_CONFIG,
    DecisionConfig,
    DetectorConfig,
    MLConfig,
)
from .decision_engine import Decision, decide, triage
from .domain_intelligence import DomainIntelligence, analyze_domain
from .explanations import Reason
from .feature_schema import SCHEMA_VERSION
from .layer2_cache import Layer2Cache, get_layer2_intel, open_layer2_cache
from .lexical_features import LexicalFeatures, extract_lexical_features
from .llm_analyzer import LLMAnalysis, analyze_page_content
from .observability import Timings, get_logger, new_analysis_id
from .page_cache import PageCache, open_page_cache
from .security import redact_url
from .web_fetcher import PageContent, fetch_page

_log = get_logger("pipeline")


@dataclass(frozen=True)
class DetectionResult:
    """Layers 1-2 only - the raw-feature building block."""

    url: str
    layer1: LexicalFeatures
    layer2: DomainIntelligence | None

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "layer1_lexical": self.layer1.as_dict(),
            "layer2_domain_intel": self.layer2.as_dict() if self.layer2 else None,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """The complete result of analysing one URL, from every layer that ran."""

    url: str
    analysis_id: str
    decision: Decision

    layer1: LexicalFeatures
    layer2: DomainIntelligence | None
    page: PageContent | None
    llm: LLMAnalysis | None

    timings_ms: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    # --- flattened accessors for the common questions --------------------
    @property
    def classification(self) -> str:
        return self.decision.classification

    @property
    def risk_score(self) -> float:
        return self.decision.risk_score

    @property
    def ml_probability(self) -> float | None:
        return self.decision.ml_probability

    @property
    def llm_likelihood(self) -> float | None:
        return self.decision.llm_likelihood

    @property
    def layer4_ran(self) -> bool:
        return self.decision.layer4_ran

    @property
    def reasons(self) -> tuple[Reason, ...]:
        return self.decision.reasons

    def reason_messages(self) -> list[str]:
        return [r.message for r in self.decision.reasons]

    def as_dict(self) -> dict:
        # Layer 1's own dict echoes the raw URL (it is the debugging view of the
        # parse). This is the *published* result, which gets logged and stored,
        # so the echo is redacted here too - phishing URLs routinely carry the
        # victim's email or a session token in the query string.
        layer1 = self.layer1.as_dict()
        layer1["url"] = redact_url(layer1.get("url", ""))

        return {
            "url": redact_url(self.url),
            "analysis_id": self.analysis_id,
            "classification": self.decision.classification,
            "risk_score": self.decision.risk_score,
            "risk_score_note": "ordinal severity, not a calibrated probability",
            "ml_probability": self.decision.ml_probability,
            "llm_likelihood": self.decision.llm_likelihood,
            "decision_path": self.decision.decision_path,
            "layer4_ran": self.decision.layer4_ran,
            "reasons": [r.as_dict() for r in self.decision.reasons],
            "layer1_lexical": layer1,
            "layer2_domain_intel": self.layer2.as_dict() if self.layer2 else None,
            "layer4_page": self.page.as_dict() if self.page else None,
            "layer4_llm": self.llm.as_dict() if self.llm else None,
            "timings_ms": self.timings_ms,
            "metadata": self.metadata,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Layers 1-2 building block
# ---------------------------------------------------------------------------
def analyze(
    url: str,
    config: DetectorConfig = DEFAULT_CONFIG,
    *,
    run_layer2: bool = True,
    do_reverse_dns: bool = True,
    do_domain_age: bool = True,
    do_tls: bool = False,
    resolver=None,
) -> DetectionResult:
    """Run Layer 1 (always) and Layer 2 (unless disabled). Never raises.

    A building block for `analyze_url`, and the entry point when you want raw
    features with no scoring and no decision at all.
    """
    layer1 = extract_lexical_features(url, config)

    layer2: DomainIntelligence | None = None
    if run_layer2:
        layer2 = analyze_domain(
            url,
            config,
            resolver=resolver,
            lexical=layer1,               # reuse Layer 1's parse - no re-parsing
            do_reverse_dns=do_reverse_dns,
            do_domain_age=do_domain_age,
            do_tls=do_tls,
        )

    return DetectionResult(url=url, layer1=layer1, layer2=layer2)


# ---------------------------------------------------------------------------
# the top-level API
# ---------------------------------------------------------------------------
def analyze_url(
    url: str,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    decision_config: DecisionConfig = DEFAULT_DECISION_CONFIG,
    run_layer2: bool = True,
    do_tls: bool = True,
    predictor=None,
    llm_client=None,
    fetcher=fetch_page,
    layer2_cache: Layer2Cache | None = None,
    page_cache: PageCache | None = None,
    use_caches: bool = True,
) -> AnalysisResult:
    """Analyse one URL through every layer and return the full result.

    Layer 4 runs only when Layers 1-3 land in the uncertain band and it is
    enabled. Never raises.

    The injection points (`predictor`, `llm_client`, `fetcher`, both caches)
    exist so tests - and callers bringing their own model or LLM - can supply
    implementations without monkeypatching.
    """
    analysis_id = new_analysis_id()
    timings = Timings()
    errors: list[str] = []
    should_run_layer4 = False

    _log.info("analysis started",
              extra={"fields": {"analysis_id": analysis_id, "url": redact_url(url)}})

    owns_l2_cache = False
    if layer2_cache is None and use_caches and run_layer2:
        layer2_cache = open_layer2_cache(decision_config.cache)
        owns_l2_cache = layer2_cache is not None

    try:
        # --- Layer 1 ------------------------------------------------------
        with timings.measure("layer1_ms"):
            layer1 = extract_lexical_features(url, config)

        # --- Layer 2 ------------------------------------------------------
        layer2: DomainIntelligence | None = None
        if run_layer2 and layer1.is_malformed:
            errors.append("layer2-skipped: URL did not parse")
        elif run_layer2:
            with timings.measure("layer2_ms"):
                layer2 = get_layer2_intel(
                    url, cache=layer2_cache, config=config, do_tls=do_tls,
                )
            if layer2 is None:
                errors.append("layer2-unavailable")

        # --- Layer 3 ------------------------------------------------------
        with timings.measure("layer3_ms"):
            ml_probability, model_version, ml_error = _predict(
                layer1, layer2, ml_config, predictor
            )
        if ml_error:
            errors.append(ml_error)

        # --- triage: can we answer without paying for Layer 4? ------------
        tri = triage(layer1.registered_domain, ml_probability, decision_config)

        page: PageContent | None = None
        llm: LLMAnalysis | None = None
        should_run_layer4 = (
            tri.needs_layer4
            and decision_config.enable_layer4
            and decision_config.llm.enabled
            and not layer1.is_malformed
        )

        # --- Layer 4 ------------------------------------------------------
        if should_run_layer4:
            page, llm = _run_layer4(
                url, layer1, layer2, config, decision_config,
                fetcher=fetcher, llm_client=llm_client, page_cache=page_cache,
                use_caches=use_caches, timings=timings, errors=errors,
            )

        # --- decision -----------------------------------------------------
        decision = decide(
            layer1=layer1, layer2=layer2, ml_probability=ml_probability,
            page=page, llm=llm, triage_result=tri, config=decision_config,
        )
    finally:
        if owns_l2_cache and layer2_cache is not None:
            layer2_cache.close()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "registered_domain": layer1.registered_domain,
        "layer2_ran": layer2 is not None,
        "layer4_attempted": should_run_layer4,
        "thresholds": {
            "low": decision_config.low_threshold,
            "high": decision_config.high_threshold,
        },
    }

    _log.info(
        "analysis complete",
        extra={"fields": {
            "analysis_id": analysis_id,
            "classification": decision.classification,
            "risk_score": decision.risk_score,
            "decision_path": decision.decision_path,
            "layer4_ran": decision.layer4_ran,
            "total_ms": timings.total_ms(),
        }},
    )

    return AnalysisResult(
        url=url, analysis_id=analysis_id, decision=decision,
        layer1=layer1, layer2=layer2, page=page, llm=llm,
        timings_ms=timings.as_dict(), metadata=metadata, errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _predict(layer1, layer2, ml_config, predictor):
    """Layer 3. Returns (probability, model_version, error_string)."""
    from .prediction import ModelNotFound, get_predictor

    try:
        pred = predictor or get_predictor(ml_config=ml_config)
    except ModelNotFound as exc:
        return None, None, f"model-unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 - a bad model file must not crash us
        return None, None, f"model-load-failed: {type(exc).__name__}: {exc}"

    try:
        probability = pred.predict_from_layers(layer1, layer2)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"prediction-failed: {type(exc).__name__}: {exc}"

    info = getattr(getattr(pred, "model", None), "info", None)
    return probability, getattr(info, "model_version", None), None


def _run_layer4(url, layer1, layer2, config, decision_config, *, fetcher,
                llm_client, page_cache, use_caches, timings, errors):
    """Fetch the page and run the LLM over it. Both cached, both best-effort."""
    owns_cache = False
    if page_cache is None and use_caches:
        page_cache = open_page_cache(decision_config.cache)
        owns_cache = page_cache is not None

    try:
        if page_cache is not None:
            cached = page_cache.get(url)
            if cached is not None:
                return None, cached

        with timings.measure("layer4_fetch_ms"):
            try:
                page = fetcher(url, decision_config.network)
            except Exception as exc:  # noqa: BLE001 - a fetcher must never raise
                errors.append(f"layer4-fetch-failed: {type(exc).__name__}: {exc}")
                return None, None

        if not page.fetched:
            errors.append(f"layer4-no-page: {page.blocked_reason or page.fetch_error}")
            return page, None

        with timings.measure("layer4_llm_ms"):
            llm = analyze_page_content(
                url, page, layer1=layer1, layer2=layer2, config=config,
                llm_config=decision_config.llm, llm_client=llm_client,
            )

        if not llm.performed and llm.error:
            errors.append(f"layer4-llm: {llm.error}")
        elif page_cache is not None:
            try:
                page_cache.put(url, llm)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"layer4-cache-write-failed: {type(exc).__name__}")

        return page, llm
    finally:
        if owns_cache and page_cache is not None:
            page_cache.close()
