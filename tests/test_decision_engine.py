"""The decision engine: triage, rule combination, and the final verdict.

These are pure functions over layer outputs, so every test here is offline and
deterministic - no model file, no network, no LLM.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from detector.config import DEFAULT_DECISION_CONFIG, ConfigError, DecisionConfig
from detector.decision_engine import (
    PHISHING,
    SAFE,
    TRIAGE_ALLOWLIST,
    TRIAGE_BLOCKLIST,
    TRIAGE_HIGH,
    TRIAGE_LOW,
    TRIAGE_NEEDS_LAYER4,
    TRIAGE_NO_MODEL,
    UNCERTAIN,
    combine,
    decide,
    triage,
)
from detector.html_analyzer import analyze_html
from detector.lexical_features import extract_lexical_features
from detector.llm_analyzer import LLMAnalysis

CFG = DEFAULT_DECISION_CONFIG

CLEAN_HTML = "<html><head><title>Blog</title></head><body><p>A post.</p></body></html>"
CREDENTIAL_SAME_DOMAIN = """
<html><body><form action="/login" method="post">
<input type="email" name="e"><input type="password" name="p"></form></body></html>
"""
CROSS_DOMAIN_HARVEST = """
<html><body><form action="https://collector-site.ru/x" method="post">
<input type="email" name="e"><input type="password" name="p"></form></body></html>
"""
SEED_PHRASE_PAGE = """
<html><body><form action="/restore" method="post">
<input name="recovery_phrase"></form></body></html>
"""


def _structure(html, url="https://site.example.com/login"):
    return analyze_html(url, html)


def _llm(**overrides) -> LLMAnalysis:
    base = {"performed": True, "risk_level": "medium", "confidence": "medium",
            "llm_likelihood": 0.5}
    base.update(overrides)
    return LLMAnalysis(**base)


# ===========================================================================
# configuration guards
# ===========================================================================
def test_inverted_thresholds_rejected():
    with pytest.raises(ConfigError):
        DecisionConfig(low_threshold=0.8, high_threshold=0.2)


def test_out_of_range_thresholds_rejected():
    with pytest.raises(ConfigError):
        DecisionConfig(low_threshold=-0.1)
    with pytest.raises(ConfigError):
        DecisionConfig(high_threshold=1.5)


# ===========================================================================
# steps 1-4: triage
# ===========================================================================
def test_blocklist_wins_over_a_low_ml_score():
    cfg = replace(CFG, blocklist_domains=("known-bad.com",))
    result = triage("known-bad.com", 0.01, cfg)

    assert result.outcome == TRIAGE_BLOCKLIST
    assert result.classification == PHISHING
    assert "blocklist" in result.reason.message.lower()


def test_allowlist_wins_over_a_high_ml_score():
    cfg = replace(CFG, allowlist_domains=("corp-intranet.com",))
    result = triage("corp-intranet.com", 0.99, cfg)

    assert result.outcome == TRIAGE_ALLOWLIST
    assert result.classification == SAFE


def test_blocklist_checked_before_allowlist():
    cfg = replace(CFG, blocklist_domains=("x.com",), allowlist_domains=("x.com",))
    assert triage("x.com", 0.5, cfg).classification == PHISHING


def test_high_score_decides_without_layer4():
    result = triage("site.com", 0.95, CFG)
    assert result.outcome == TRIAGE_HIGH
    assert result.classification == PHISHING
    assert result.needs_layer4 is False


def test_low_score_decides_without_layer4():
    result = triage("site.com", 0.02, CFG)
    assert result.outcome == TRIAGE_LOW
    assert result.classification == SAFE
    assert result.needs_layer4 is False


def test_uncertain_band_asks_for_layer4():
    result = triage("site.com", 0.5, CFG)
    assert result.outcome == TRIAGE_NEEDS_LAYER4
    assert result.needs_layer4 is True
    assert result.classification is None


@pytest.mark.parametrize("probability,expected", [
    (0.30, TRIAGE_LOW),        # boundary is inclusive
    (0.3001, TRIAGE_NEEDS_LAYER4),
    (0.6999, TRIAGE_NEEDS_LAYER4),
    (0.70, TRIAGE_HIGH),       # boundary is inclusive
])
def test_threshold_boundaries_are_inclusive(probability, expected):
    assert triage("site.com", probability, CFG).outcome == expected


def test_missing_model_is_its_own_outcome():
    assert triage("site.com", None, CFG).outcome == TRIAGE_NO_MODEL


def test_thresholds_are_configurable():
    strict = replace(CFG, low_threshold=0.1, high_threshold=0.4)
    assert triage("site.com", 0.5, strict).outcome == TRIAGE_HIGH
    assert triage("site.com", 0.5, CFG).outcome == TRIAGE_NEEDS_LAYER4


# ===========================================================================
# step 5: combining in the uncertain band
# ===========================================================================
def test_static_cross_domain_credential_post_is_conclusive():
    """Code-verified, and checked before anything the LLM said."""
    classification, score, path = combine(
        0.5, _structure(CROSS_DOMAIN_HARVEST),
        _llm(risk_level="low", confidence="high"),   # LLM says it's fine
        CFG,
    )
    assert classification == PHISHING
    assert path == "static_cross_domain_credential_post"
    assert score >= 0.9


def test_seed_phrase_request_is_conclusive():
    classification, _score, path = combine(
        0.5, _structure(SEED_PHRASE_PAGE), _llm(risk_level="low"), CFG)
    assert classification == PHISHING
    assert path == "static_seed_phrase_request"


def test_insecure_credential_post_is_conclusive():
    html = ('<form action="http://site.example.com/login" method="post">'
            '<input type="password" name="p"></form>')
    classification, _score, path = combine(
        0.5, _structure(html), _llm(risk_level="low"), CFG)
    assert classification == PHISHING
    assert path == "static_insecure_credential_post"


def test_prompt_injection_attempt_is_treated_as_evidence_against():
    classification, score, path = combine(
        0.5, _structure(CLEAN_HTML),
        _llm(prompt_injection_attempt=True, risk_level="low", confidence="high"),
        CFG,
    )
    assert classification == PHISHING
    assert path == "prompt_injection_attempt"
    assert score >= 0.8


def test_impersonation_plus_credential_request():
    classification, _score, path = combine(
        0.5, _structure(CREDENTIAL_SAME_DOMAIN),
        _llm(brand_impersonation=True, brand_claimed="PayPal", risk_level="medium"),
        CFG,
    )
    assert classification == PHISHING
    assert path == "brand_impersonation_with_credential_request"


def test_impersonation_without_credential_request_stays_uncertain():
    classification, _score, path = combine(
        0.5, _structure(CLEAN_HTML),
        _llm(brand_impersonation=True, credential_request=False, risk_level="medium"),
        CFG,
    )
    assert classification == UNCERTAIN
    assert path == "brand_impersonation_only"


def test_llm_high_risk_alone_is_enough():
    classification, score, path = combine(
        0.5, _structure(CLEAN_HTML), _llm(risk_level="high", confidence="high"), CFG)
    assert classification == PHISHING
    assert path == "llm_high_risk"
    assert score >= 0.7


def test_llm_low_risk_clears_a_clean_page():
    classification, score, path = combine(
        0.45, _structure(CLEAN_HTML), _llm(risk_level="low", confidence="high"), CFG)
    assert classification == SAFE
    assert path == "llm_low_risk_cleared"
    assert score <= 0.2


def test_llm_low_risk_does_not_clear_a_page_with_static_flags():
    """Corroborating static evidence outranks a reassuring LLM verdict."""
    html = ('<body><div style="display:none">' + "x" * 400 + '</div>'
            '<script>eval(atob("x"))</script></body>')
    classification, _score, path = combine(
        0.5, _structure(html), _llm(risk_level="low", confidence="high"), CFG)
    assert classification == UNCERTAIN
    assert path == "inconclusive_after_layer4"


def test_medium_risk_stays_uncertain():
    classification, score, path = combine(
        0.5, _structure(CLEAN_HTML), _llm(risk_level="medium"), CFG)
    assert classification == UNCERTAIN
    assert path == "inconclusive_after_layer4"
    assert score == 0.5              # unchanged - nothing new was learned


def test_layer4_unavailable_stays_uncertain():
    classification, score, path = combine(0.5, None, None, CFG)
    assert classification == UNCERTAIN
    assert path == "layer4_unavailable"
    assert score == 0.5


def test_failed_llm_is_not_treated_as_a_clean_result():
    classification, _score, path = combine(
        0.5, _structure(CLEAN_HTML),
        LLMAnalysis(performed=False, error="llm-unavailable"), CFG)
    assert classification == UNCERTAIN
    assert path == "layer4_unavailable"


def test_scores_are_never_averaged():
    """A high ML score must not be dragged down by a reassuring LLM."""
    _classification, score, _path = combine(
        0.69, _structure(CLEAN_HTML), _llm(risk_level="low", confidence="high"), CFG)
    # An average of 0.69 and 0.05 would be ~0.37; the clamp gives <= 0.2.
    assert score <= 0.2

    _classification, score, _path = combine(
        0.31, _structure(CROSS_DOMAIN_HARVEST), _llm(risk_level="low"), CFG)
    # An average would dilute conclusive static proof; the floor keeps it high.
    assert score >= 0.9


# ===========================================================================
# assembling the Decision
# ===========================================================================
def test_decision_carries_the_three_scores_separately():
    l1 = extract_lexical_features("https://site.example.com/login")
    decision = decide(
        layer1=l1, ml_probability=0.5, llm=_llm(risk_level="high", llm_likelihood=0.95),
        page=None, config=CFG,
    )
    assert decision.ml_probability == 0.5
    assert decision.llm_likelihood == 0.95
    assert decision.risk_score != 0.5           # derived, not copied
    assert "not a calibrated probability" in decision.as_dict()["risk_score_note"]


def test_decision_reasons_are_sorted_by_severity():
    l1 = extract_lexical_features("http://paypal.login-verify.tk/account")
    decision = decide(layer1=l1, ml_probability=0.95, config=CFG)

    severities = [r.severity for r in decision.reasons]
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    assert severities == sorted(severities, key=lambda s: order[s])
    assert decision.reasons[0].severity == "high"


def test_decision_reasons_reference_observed_values():
    l1 = extract_lexical_features("http://paypal.login-verify.tk/account")
    decision = decide(layer1=l1, ml_probability=0.95, config=CFG)
    messages = " ".join(r.message for r in decision.reasons)

    assert "paypal" in messages                 # the brand actually found
    assert ".tk" in messages                    # the TLD actually seen
    assert "0.95" in messages                   # the probability actually scored


def test_missing_model_decision_is_uncertain_not_safe():
    l1 = extract_lexical_features("https://site.example.com/")
    decision = decide(layer1=l1, ml_probability=None, config=CFG)

    assert decision.classification == UNCERTAIN
    assert decision.ml_probability is None
    assert decision.decision_path == TRIAGE_NO_MODEL


def test_blocklist_decision_scores_one():
    cfg = replace(CFG, blocklist_domains=("evil.com",))
    l1 = extract_lexical_features("https://evil.com/x")
    decision = decide(layer1=l1, ml_probability=0.1, config=cfg)

    assert decision.classification == PHISHING
    assert decision.risk_score == 1.0
    assert decision.reasons[0].code == "policy_blocklist"


def test_layer4_ran_flag_reflects_reality():
    l1 = extract_lexical_features("https://site.example.com/")
    assert decide(layer1=l1, ml_probability=0.5, config=CFG).layer4_ran is False
    assert decide(layer1=l1, ml_probability=0.5, llm=_llm(),
                  config=CFG).layer4_ran is True
