"""The decision engine: turning four layers into one answer.

The flow, in order:

    1. blocklist      -> phishing          (policy, no further work)
    2. allowlist      -> safe              (policy, no further work)
    3. ML >= HIGH     -> phishing          (Layer 4 skipped: already confident)
    4. ML <= LOW      -> safe              (Layer 4 skipped: already confident)
    5. otherwise      -> run Layer 4, then combine by explicit rules

Two design decisions worth stating plainly.

**Scores are not averaged.** Averaging a model probability with an LLM's
self-rating produces a number that means nothing: the two are on different
scales, measure different things, and one of them is influenced by
attacker-controlled text. Instead, step 5 applies ordered rules over discrete
findings, and the final `risk_score` is derived from *which rule fired*,
clamped monotonically against the model's probability.

**`risk_score` is an ordinal severity, not a calibrated probability.** No
calibration (Platt scaling, isotonic regression, reliability-curve fitting) has
been performed on this model, so a 0.8 does not mean "80% of URLs scoring 0.8
are phishing". `ml_probability` is the Random Forest's raw output and
`llm_likelihood` is a fixed mapping from the LLM's risk/confidence enums;
`risk_score` orders severity and nothing more. The field names keep the three
apart on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    DEFAULT_DECISION_CONFIG,
    DecisionConfig,
)
from .explanations import (
    Reason,
    reason_from_ml,
    reason_from_policy,
    reasons_from_fetch,
    reasons_from_layer1,
    reasons_from_layer2,
    reasons_from_llm,
    reasons_from_page,
    sort_reasons,
)

PHISHING = "phishing"
SAFE = "safe"
UNCERTAIN = "uncertain"

#: Triage outcomes. NEEDS_LAYER4 is the only one that costs a fetch + LLM call.
TRIAGE_BLOCKLIST = "blocklist"
TRIAGE_ALLOWLIST = "allowlist"
TRIAGE_HIGH = "ml_high"
TRIAGE_LOW = "ml_low"
TRIAGE_NEEDS_LAYER4 = "needs_layer4"
TRIAGE_NO_MODEL = "no_model"

#: Floor/ceiling applied to `risk_score` when a step-5 rule fires. Chosen to
#: order severity sensibly; deliberately NOT presented as probabilities.
_SCORE_FLOOR_STATIC_PROOF = 0.90     # code-verified credential exfiltration
_SCORE_FLOOR_LLM_STRONG = 0.80       # LLM found impersonation + credential ask
_SCORE_FLOOR_LLM_HIGH = 0.70         # LLM rated high risk
_SCORE_CEILING_CLEARED = 0.20        # everything checked out


@dataclass(frozen=True)
class TriageResult:
    """Step 1-4: what we can decide before paying for Layer 4."""

    outcome: str                     # one of the TRIAGE_* constants
    classification: str | None       # set unless outcome is NEEDS_LAYER4/NO_MODEL
    reason: Reason | None = None

    @property
    def needs_layer4(self) -> bool:
        return self.outcome == TRIAGE_NEEDS_LAYER4


@dataclass(frozen=True)
class Decision:
    """The final verdict plus every number that went into it."""

    classification: str              # "phishing" | "safe" | "uncertain"
    risk_score: float                # ordinal severity 0..1, NOT calibrated
    ml_probability: float | None     # Layer 3 raw output (uncalibrated)
    llm_likelihood: float | None     # Layer 4 enum -> number mapping
    decision_path: str               # which rule produced the classification
    layer4_ran: bool
    reasons: tuple[Reason, ...] = ()

    def as_dict(self) -> dict:
        return {
            "classification": self.classification,
            "risk_score": self.risk_score,
            "risk_score_note": (
                "ordinal severity, not a calibrated probability - no calibration "
                "has been performed on this model"
            ),
            "ml_probability": self.ml_probability,
            "llm_likelihood": self.llm_likelihood,
            "decision_path": self.decision_path,
            "layer4_ran": self.layer4_ran,
            "reasons": [r.as_dict() for r in self.reasons],
        }


# ---------------------------------------------------------------------------
# steps 1-4: triage
# ---------------------------------------------------------------------------
def triage(
    registered_domain: str,
    ml_probability: float | None,
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> TriageResult:
    """Decide what we can before Layer 4. Pure function, no I/O."""
    domain = (registered_domain or "").strip().lower()

    if domain and domain in config.normalised_blocklist():
        return TriageResult(
            TRIAGE_BLOCKLIST, PHISHING,
            reason_from_policy(
                "policy_blocklist",
                f"Domain '{domain}' is on the configured blocklist"),
        )

    if domain and domain in config.normalised_allowlist():
        return TriageResult(
            TRIAGE_ALLOWLIST, SAFE,
            Reason("policy_allowlist",
                   f"Domain '{domain}' is on the configured allowlist",
                   "info", "policy", "policy"),
        )

    if ml_probability is None:
        return TriageResult(TRIAGE_NO_MODEL, None)

    if ml_probability >= config.high_threshold:
        return TriageResult(
            TRIAGE_HIGH, PHISHING,
            reason_from_ml(
                ml_probability,
                threshold_context=f"(at or above the {config.high_threshold:.2f} "
                                  f"high-confidence threshold)"),
        )

    if ml_probability <= config.low_threshold:
        return TriageResult(
            TRIAGE_LOW, SAFE,
            reason_from_ml(
                ml_probability,
                threshold_context=f"(at or below the {config.low_threshold:.2f} "
                                  f"low-risk threshold)"),
        )

    return TriageResult(TRIAGE_NEEDS_LAYER4, None)


# ---------------------------------------------------------------------------
# step 5: combine, by explicit rules
# ---------------------------------------------------------------------------
def _static_findings(structure) -> dict:
    """Code-computed page facts. An attacker cannot argue these away."""
    if structure is None:
        return {}
    return {
        "cross_domain_credential_post": structure.has_cross_domain_credential_post,
        "seed_phrase": any(f.has_seed_phrase_field for f in structure.forms),
        "insecure_credential_post": any(f.submits_insecurely for f in structure.forms),
        "otp": structure.has_otp_field,
        "credential_form": structure.has_credential_form,
        "obfuscated_script": bool(structure.scripts.obfuscation_markers),
        "hidden_text": len(structure.hidden_text) > 200,
    }


def combine(
    ml_probability: float,
    structure,
    llm,
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> tuple[str, float, str]:
    """Uncertain-band combination. Returns (classification, risk_score, path).

    Ordered rules, first match wins. Code-computed static evidence is checked
    *before* anything the LLM said, because static facts cannot be manipulated
    by the page's text while the LLM's judgement can be attempted.
    """
    static = _static_findings(structure)
    llm_ran = llm is not None and llm.performed

    # --- R1: code-verified credential exfiltration --------------------------
    if static.get("cross_domain_credential_post"):
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_STATIC_PROOF),
                "static_cross_domain_credential_post")

    if static.get("seed_phrase"):
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_STATIC_PROOF),
                "static_seed_phrase_request")

    if static.get("insecure_credential_post"):
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_LLM_STRONG),
                "static_insecure_credential_post")

    # --- R2: nothing more to go on -------------------------------------
    if not llm_ran:
        return (UNCERTAIN, ml_probability, "layer4_unavailable")

    # --- R3: a page that fights the analyser ---------------------------
    # Legitimate sites do not embed instructions aimed at automated analysers.
    if llm.prompt_injection_attempt:
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_LLM_STRONG),
                "prompt_injection_attempt")

    # --- R4: impersonation plus a credential ask ------------------
    asks_for_credentials = static.get("credential_form") or llm.credential_request
    if llm.brand_impersonation and asks_for_credentials:
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_LLM_STRONG),
                "brand_impersonation_with_credential_request")

    # --- R5: LLM rated it high risk ------------------------------
    if llm.risk_level == "high":
        return (PHISHING, max(ml_probability, _SCORE_FLOOR_LLM_HIGH),
                "llm_high_risk")

    # --- R6: impersonation without a credential ask --------------
    if llm.brand_impersonation:
        return (UNCERTAIN, max(ml_probability, 0.5), "brand_impersonation_only")

    # --- R7: cleared -------------------------------------------
    corroborating_flags = any(
        static.get(k) for k in ("otp", "obfuscated_script", "hidden_text")
    )
    if llm.risk_level == "low" and not corroborating_flags:
        return (SAFE, min(ml_probability, _SCORE_CEILING_CLEARED), "llm_low_risk_cleared")

    # --- R8: still ambiguous after everything -----------------
    return (UNCERTAIN, ml_probability, "inconclusive_after_layer4")


# ---------------------------------------------------------------------------
# assembling the final Decision
# ---------------------------------------------------------------------------
def decide(
    *,
    layer1=None,
    layer2=None,
    ml_probability: float | None,
    page=None,
    llm=None,
    triage_result: TriageResult | None = None,
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> Decision:
    """Produce the final `Decision` from whatever each layer managed to produce.

    Pure function: no network, no model loading, no I/O. `pipeline.analyze_url`
    gathers the inputs; this decides what they mean.
    """
    registered_domain = getattr(layer1, "registered_domain", "") or ""
    tri = triage_result or triage(registered_domain, ml_probability, config)

    reasons: list[Reason] = []
    reasons.extend(reasons_from_layer1(layer1))
    reasons.extend(reasons_from_layer2(layer2))
    reasons.extend(reasons_from_fetch(page))
    reasons.extend(reasons_from_page(getattr(page, "structure", None)))
    reasons.extend(reasons_from_llm(llm))

    llm_likelihood = getattr(llm, "llm_likelihood", None) if llm else None
    layer4_ran = bool(llm is not None and llm.performed)

    # --- terminal triage outcomes ---------------------------------------
    if tri.outcome in (TRIAGE_BLOCKLIST, TRIAGE_ALLOWLIST):
        if tri.reason:
            reasons.insert(0, tri.reason)
        score = 1.0 if tri.classification == PHISHING else 0.0
        return Decision(
            classification=tri.classification, risk_score=score,
            ml_probability=ml_probability, llm_likelihood=llm_likelihood,
            decision_path=tri.outcome, layer4_ran=layer4_ran,
            reasons=sort_reasons(reasons),
        )

    if tri.outcome == TRIAGE_NO_MODEL:
        return Decision(
            classification=UNCERTAIN, risk_score=0.5,
            ml_probability=None, llm_likelihood=llm_likelihood,
            decision_path=TRIAGE_NO_MODEL, layer4_ran=layer4_ran,
            reasons=sort_reasons(reasons),
        )

    if tri.reason:
        reasons.append(tri.reason)

    if tri.outcome in (TRIAGE_HIGH, TRIAGE_LOW):
        return Decision(
            classification=tri.classification, risk_score=float(ml_probability),
            ml_probability=ml_probability, llm_likelihood=llm_likelihood,
            decision_path=tri.outcome, layer4_ran=layer4_ran,
            reasons=sort_reasons(reasons),
        )

    # --- uncertain band -----------------------------------------------
    classification, score, path = combine(
        float(ml_probability), getattr(page, "structure", None), llm, config
    )
    return Decision(
        classification=classification, risk_score=round(float(score), 4),
        ml_probability=ml_probability, llm_likelihood=llm_likelihood,
        decision_path=path, layer4_ran=layer4_ran,
        reasons=sort_reasons(reasons),
    )
