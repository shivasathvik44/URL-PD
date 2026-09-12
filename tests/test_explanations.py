"""Explanations: every reason must come from something actually observed.

The rule this file enforces is "never invent evidence" - a reason exists only
when the corresponding feature fired, and its message quotes the measured
value rather than paraphrasing it.
"""

from __future__ import annotations

from detector.explanations import (
    Reason,
    reason_from_ml,
    reasons_from_layer1,
    reasons_from_llm,
    reasons_from_page,
    sort_reasons,
)
from detector.html_analyzer import analyze_html
from detector.lexical_features import extract_lexical_features
from detector.llm_analyzer import LLMAnalysis


def _codes(reasons):
    return {r.code for r in reasons}


# ---------------------------------------------------------------------------
# nothing is invented
# ---------------------------------------------------------------------------
def test_clean_url_produces_no_alarming_reasons():
    l1 = extract_lexical_features("https://www.wikipedia.org/wiki/Main_Page")
    reasons = reasons_from_layer1(l1)
    assert not any(r.severity == "high" for r in reasons)


def test_clean_page_produces_no_reasons():
    structure = analyze_html(
        "https://example.com/blog",
        "<html><head><title>Blog</title></head><body><p>A post.</p></body></html>")
    assert reasons_from_page(structure) == []


def test_no_llm_reasons_when_the_llm_did_not_run():
    assert reasons_from_llm(None) == []
    assert reasons_from_llm(LLMAnalysis(performed=False, error="llm-unavailable")) == []


def test_no_ml_reason_without_a_probability():
    assert reason_from_ml(None) is None


# ---------------------------------------------------------------------------
# messages quote the observed value
# ---------------------------------------------------------------------------
def test_brand_reason_names_the_brand_and_the_real_domain():
    l1 = extract_lexical_features("https://paypal.security-check.xyz/login")
    reason = next(r for r in reasons_from_layer1(l1) if r.code == "l1_brand_in_wrong_place")

    assert "paypal" in reason.message
    assert "security-check.xyz" in reason.message
    assert reason.severity == "high"


def test_typosquat_reason_quotes_the_distance():
    l1 = extract_lexical_features("https://paypa1.com/login")
    reason = next(r for r in reasons_from_layer1(l1) if r.code == "l1_typosquat")

    assert "paypal" in reason.message
    assert str(l1.levenshtein_distance) in reason.message


def test_tld_reason_names_the_tld():
    l1 = extract_lexical_features("http://login.example.tk/")
    reason = next(r for r in reasons_from_layer1(l1) if r.code == "l1_suspicious_tld")
    assert ".tk" in reason.message


def test_at_symbol_reason_names_the_real_host():
    l1 = extract_lexical_features("http://paypal.com@evil-site.ru/login")
    reason = next(r for r in reasons_from_layer1(l1) if r.code == "l1_at_symbol")
    assert "evil-site.ru" in reason.message


def test_ml_reason_quotes_the_probability():
    reason = reason_from_ml(0.8765)
    assert "0.88" in reason.message
    assert reason.source == "model"


def test_cross_domain_form_reason_names_both_domains():
    structure = analyze_html(
        "https://paypal-login.com/signin",
        '<form action="https://collector-site.ru/x" method="post">'
        '<input type="password" name="p"></form>')
    reason = next(r for r in reasons_from_page(structure)
                  if r.code == "l4_cross_domain_credential_post")

    assert "collector-site.ru" in reason.message
    assert "paypal-login.com" in reason.message
    assert "password" in reason.message
    assert reason.severity == "high"


def test_otp_reason_explains_why_it_matters():
    structure = analyze_html(
        "https://site.com/verify",
        '<form action="/v"><input name="otp" maxlength="6" inputmode="numeric"></form>')
    reason = next(r for r in reasons_from_page(structure) if r.code == "l4_otp_field")
    assert "two-factor" in reason.message


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def test_llm_reasons_are_attributed_to_the_llm():
    llm = LLMAnalysis(performed=True, brand_claimed="PayPal", brand_impersonation=True,
                      social_engineering=True, social_engineering_tactics=("urgency",),
                      risk_level="high", confidence="high")
    reasons = reasons_from_llm(llm)

    assert all(r.source == "llm" for r in reasons)
    assert all(r.layer == "layer4" for r in reasons)


def test_static_page_reasons_are_attributed_to_rules_not_the_llm():
    structure = analyze_html(
        "https://site.com/x",
        '<form action="https://other-site.ru/x"><input type="password"></form>')
    assert all(r.source == "rule" for r in reasons_from_page(structure))


def test_prompt_injection_reason_says_it_was_not_obeyed():
    llm = LLMAnalysis(performed=True, prompt_injection_attempt=True, risk_level="medium")
    reason = next(r for r in reasons_from_llm(llm) if r.code == "l4_prompt_injection")

    assert "not followed" in reason.message
    assert reason.severity == "high"


# ---------------------------------------------------------------------------
# ordering and serialisation
# ---------------------------------------------------------------------------
def test_sorting_puts_the_most_serious_first():
    reasons = [
        Reason("a", "info msg", "info", "layer1", "rule"),
        Reason("b", "high msg", "high", "layer1", "rule"),
        Reason("c", "low msg", "low", "layer1", "rule"),
        Reason("d", "medium msg", "medium", "layer1", "rule"),
    ]
    assert [r.severity for r in sort_reasons(reasons)] == ["high", "medium", "low", "info"]


def test_reason_serialises_with_stable_machine_code():
    reason = Reason("l1_typosquat", "message", "high", "layer1", "rule")
    d = reason.as_dict()
    assert d["code"] == "l1_typosquat"
    assert set(d) == {"code", "message", "severity", "layer", "source"}
