"""Human-readable reasons, built only from observed values.

Every `Reason` is generated from a feature the system actually measured, and
its message quotes that measurement ("Domain was registered 5 days ago", not
"Domain is new"). Nothing here invents evidence, infers a motive, or repeats a
claim no layer made:

  * Layer 1/2/static-page reasons are derived from code-computed facts.
  * Layer 3 contributes one reason carrying the model's probability.
  * Layer 4 reasons are clearly attributed to the LLM and carry only its
    validated, sanitised, truncated output.

A `Reason` has a stable `code` for machine consumers and a `message` for
people, so downstream tooling never has to string-match prose.
"""

from __future__ import annotations

from dataclasses import dataclass

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass(frozen=True)
class Reason:
    """One piece of evidence, attributed to the layer that observed it."""

    code: str            # stable identifier, e.g. "l2_domain_new"
    message: str         # human-readable, quoting the observed value
    severity: str        # "high" | "medium" | "low" | "info"
    layer: str           # "layer1" | "layer2" | "layer3" | "layer4" | "policy"
    source: str          # "rule" | "model" | "llm" | "policy"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "layer": self.layer,
            "source": self.source,
        }


def sort_reasons(reasons) -> tuple[Reason, ...]:
    """Most serious first; stable within a severity."""
    return tuple(
        sorted(reasons, key=lambda r: SEVERITY_ORDER.get(r.severity, 9))
    )


# ---------------------------------------------------------------------------
# Layer 1 - lexical
# ---------------------------------------------------------------------------
def reasons_from_layer1(l1) -> list[Reason]:
    out: list[Reason] = []
    if l1 is None:
        return out

    if l1.is_malformed:
        out.append(Reason(
            "l1_malformed_url",
            "URL could not be parsed as a normal web address",
            "low", "layer1", "rule"))

    if l1.brand_in_wrong_place:
        out.append(Reason(
            "l1_brand_in_wrong_place",
            f"URL puts the brand '{l1.brand_in_wrong_place_name}' in the "
            f"subdomain or path, but the registered domain is "
            f"'{l1.registered_domain or l1.hostname}'",
            "high", "layer1", "rule"))

    if l1.is_typosquat_suspicious:
        out.append(Reason(
            "l1_typosquat",
            f"Registered domain '{l1.registered_domain}' is {l1.levenshtein_distance} "
            f"character edit(s) away from the brand '{l1.closest_brand}'",
            "high", "layer1", "rule"))

    if l1.has_at_symbol:
        out.append(Reason(
            "l1_at_symbol",
            "URL contains '@' before the host, so the real destination is "
            f"'{l1.hostname}', not what appears before the '@'",
            "high", "layer1", "rule"))

    if l1.is_punycode:
        out.append(Reason(
            "l1_punycode",
            f"Hostname '{l1.hostname}' uses punycode (xn--), which can render as "
            "look-alike characters",
            "medium", "layer1", "rule"))
    elif l1.contains_non_ascii:
        out.append(Reason(
            "l1_non_ascii_host",
            f"Hostname '{l1.hostname}' contains non-ASCII characters that can "
            "imitate Latin letters",
            "medium", "layer1", "rule"))

    if l1.has_ip:
        out.append(Reason(
            "l1_ip_host",
            f"URL addresses a raw IP ({l1.hostname}) instead of a domain name",
            "medium", "layer1", "rule"))

    if l1.is_suspicious_tld:
        out.append(Reason(
            "l1_suspicious_tld",
            f"Top-level domain '.{l1.tld}' is one commonly abused for "
            "disposable phishing domains",
            "low", "layer1", "rule"))

    if l1.is_suspicious_port:
        out.append(Reason(
            "l1_suspicious_port",
            f"URL uses the unusual port {l1.port} for web traffic",
            "low", "layer1", "rule"))

    if l1.keyword_count >= 2:
        out.append(Reason(
            "l1_keywords",
            "URL contains credential-related keywords: "
            + ", ".join(l1.keywords_found[:5]),
            "low", "layer1", "rule"))

    if l1.subdomain_depth >= 4:
        out.append(Reason(
            "l1_deep_subdomain",
            f"URL nests {l1.subdomain_depth} subdomain levels, which pushes the "
            "real domain out of sight in a browser bar",
            "low", "layer1", "rule"))

    return out


# ---------------------------------------------------------------------------
# Layer 2 - DNS / domain intelligence
# ---------------------------------------------------------------------------
def reasons_from_layer2(l2) -> list[Reason]:
    out: list[Reason] = []
    if l2 is None:
        return out

    if l2.age.available and l2.age.age_days is not None and l2.age.is_new_domain:
        out.append(Reason(
            "l2_domain_new",
            f"Domain was registered {l2.age.age_days} day(s) ago",
            "high", "layer2", "rule"))
    elif l2.age.available and l2.age.age_days is not None and l2.age.age_days > 730:
        out.append(Reason(
            "l2_domain_established",
            f"Domain has been registered for {l2.age.age_days} days "
            f"({l2.age.age_days // 365} years)",
            "info", "layer2", "rule"))

    if l2.dns.queried and not l2.dns.dns_resolves:
        out.append(Reason(
            "l2_no_dns",
            "Domain does not currently resolve to any IP address",
            "medium", "layer2", "rule"))

    if l2.dns.dns_resolves and not l2.dns.has_mx:
        out.append(Reason(
            "l2_no_mx",
            "Domain has no mail (MX) records, unusual for an organisation that "
            "emails its customers",
            "info", "layer2", "rule"))

    if l2.dns.private_ip_count:
        out.append(Reason(
            "l2_private_ip",
            f"Domain resolves to {l2.dns.private_ip_count} private/internal "
            "address(es)",
            "medium", "layer2", "rule"))

    tls = l2.tls
    if tls is not None and tls.checked:
        if tls.tls_available and not tls.cert_valid:
            detail = tls.tls_error or "certificate did not verify"
            out.append(Reason(
                "l2_tls_invalid",
                f"TLS certificate does not validate for this host ({detail})",
                "medium", "layer2", "rule"))
        elif tls.cert_valid and tls.cert_age_days is not None and tls.cert_age_days <= 7:
            out.append(Reason(
                "l2_tls_new_cert",
                f"TLS certificate was issued {tls.cert_age_days} day(s) ago",
                "low", "layer2", "rule"))

    return out


# ---------------------------------------------------------------------------
# Layer 4 - static page facts (code-computed, no LLM involved)
# ---------------------------------------------------------------------------
def reasons_from_page(structure) -> list[Reason]:
    out: list[Reason] = []
    if structure is None:
        return out

    for form in structure.forms:
        if form.is_credential_form and form.posts_cross_domain:
            out.append(Reason(
                "l4_cross_domain_credential_post",
                f"A form collecting {', '.join(form.collected_kinds())} submits to "
                f"'{form.action_registered_domain}', a different domain from the "
                f"page's own '{form.page_registered_domain}'",
                "high", "layer4", "rule"))
        elif form.submits_insecurely:
            out.append(Reason(
                "l4_insecure_credential_post",
                f"A form collecting {', '.join(form.collected_kinds())} submits "
                "over unencrypted HTTP",
                "high", "layer4", "rule"))

    collected = set()
    for form in structure.forms:
        collected.update(form.collected_kinds())
    if "one-time code" in collected:
        out.append(Reason(
            "l4_otp_field",
            "Page asks for a one-time verification code, which is how "
            "attackers defeat two-factor authentication in real time",
            "high", "layer4", "rule"))
    if "wallet seed phrase" in collected:
        out.append(Reason(
            "l4_seed_phrase_field",
            "Page asks for a crypto wallet seed phrase or private key",
            "high", "layer4", "rule"))
    if "payment card" in collected:
        out.append(Reason(
            "l4_payment_field",
            "Page collects payment card details",
            "medium", "layer4", "rule"))
    if "password" in collected:
        out.append(Reason(
            "l4_password_field",
            "Page contains a password field",
            "low", "layer4", "rule"))

    if structure.cross_site_iframes:
        out.append(Reason(
            "l4_cross_site_iframe",
            f"Page embeds {len(structure.cross_site_iframes)} iframe(s) from "
            "other domains",
            "low", "layer4", "rule"))

    if structure.hidden_text and len(structure.hidden_text) > 200:
        out.append(Reason(
            "l4_hidden_text",
            f"Page hides {len(structure.hidden_text)} characters of text from "
            "the visible rendering",
            "medium", "layer4", "rule"))

    markers = structure.scripts.obfuscation_markers
    if markers:
        out.append(Reason(
            "l4_obfuscated_script",
            "Page scripts use obfuscation techniques: " + ", ".join(markers[:4]),
            "medium", "layer4", "rule"))

    if structure.meta_refresh_to:
        out.append(Reason(
            "l4_meta_refresh",
            "Page immediately redirects visitors via a meta refresh tag",
            "low", "layer4", "rule"))

    return out


def reasons_from_fetch(page) -> list[Reason]:
    """Reasons about the fetch itself (redirects, blocks)."""
    out: list[Reason] = []
    if page is None:
        return out
    if page.blocked_reason:
        out.append(Reason(
            "l4_fetch_blocked",
            f"Page was not fetched: {page.blocked_reason}",
            "info", "layer4", "policy"))
    elif page.fetched and len(page.redirect_chain) >= 2:
        out.append(Reason(
            "l4_redirect_chain",
            f"URL redirected {len(page.redirect_chain)} times before reaching "
            "the final page",
            "low", "layer4", "rule"))
    return out


# ---------------------------------------------------------------------------
# Layer 4 - LLM findings (always attributed)
# ---------------------------------------------------------------------------
def reasons_from_llm(llm) -> list[Reason]:
    out: list[Reason] = []
    if llm is None or not llm.performed:
        return out

    if llm.prompt_injection_attempt:
        out.append(Reason(
            "l4_prompt_injection",
            "Page content contains text written to manipulate an automated "
            "analyser - the instructions were not followed, and the attempt "
            "itself is treated as suspicious",
            "high", "layer4", "llm"))

    if llm.brand_impersonation and llm.brand_claimed:
        out.append(Reason(
            "l4_brand_impersonation",
            f"Page presents itself as '{llm.brand_claimed}' but is not hosted "
            "on that organisation's domain",
            "high", "layer4", "llm"))
    elif llm.brand_claimed:
        out.append(Reason(
            "l4_brand_claimed",
            f"Page presents itself as '{llm.brand_claimed}'",
            "info", "layer4", "llm"))

    if llm.social_engineering and llm.social_engineering_tactics:
        out.append(Reason(
            "l4_social_engineering",
            "Page uses social-engineering language: "
            + "; ".join(llm.social_engineering_tactics[:3]),
            "medium", "layer4", "llm"))

    for observation in llm.suspicious_observations[:3]:
        out.append(Reason(
            "l4_llm_observation",
            f"Page analysis noted: {observation}",
            "low", "layer4", "llm"))

    if llm.risk_level:
        out.append(Reason(
            "l4_llm_risk",
            f"Page-content analysis rated this {llm.risk_level} risk "
            f"({llm.confidence or 'unstated'} confidence)",
            "info", "layer4", "llm"))

    return out


# ---------------------------------------------------------------------------
# Layer 3 - the model
# ---------------------------------------------------------------------------
def reason_from_ml(probability: float | None, *, threshold_context: str = "") -> Reason | None:
    if probability is None:
        return None
    suffix = f" {threshold_context}" if threshold_context else ""
    return Reason(
        "l3_model_probability",
        f"URL-feature model scored {probability:.2f} probability of phishing{suffix}",
        "info", "layer3", "model")


def reason_from_policy(code: str, message: str) -> Reason:
    return Reason(code, message, "high", "policy", "policy")
