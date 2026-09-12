"""Layer 4c: what a form collects, and where it sends it.

Everything here is computed in code from the HTML, never asked of the LLM. That
matters: "this login form posts to a different domain" is the single strongest
static phishing signal on a page, and a page full of text telling an LLM
otherwise cannot change an `action` attribute. The LLM is shown these facts and
asked to interpret them - it is not the source of them.

Field classification is intentionally conservative and multi-signal: a field
counts as a password/OTP/payment input based on its `type`, `name`, `id`,
`autocomplete`, `placeholder`, `inputmode`, `pattern` and ARIA label together,
because phishing kits routinely use `type="text"` with `name="pwd"` to dodge
naive `type="password"` checks.

**No field value is ever read, stored or logged** - only the shape of the form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexical_features import registered_domain_of

# --- field-name heuristics -------------------------------------------------
_PASSWORD_RE = re.compile(r"(?i)\b(pass(word|wd|phrase)?|pwd|senha|contrase|passcode)\b|pass")
_OTP_RE = re.compile(
    r"(?i)\b(otp|2fa|mfa|totp|one[\s_-]?time|verification[\s_-]?code|auth(entication)?[\s_-]?code"
    r"|security[\s_-]?code|sms[\s_-]?code|token|pin)\b"
)
_CARD_RE = re.compile(
    r"(?i)\b(card[\s_-]?(number|no|num)?|cc[\s_-]?(num|number)?|credit|debit|pan"
    r"|cvv|cvc|csc|security[\s_-]?code|exp(iry|iration)?[\s_-]?(date|month|year)?)\b"
)
_BANK_RE = re.compile(r"(?i)\b(iban|swift|bic|routing|account[\s_-]?number|sort[\s_-]?code)\b")
_IDENTITY_RE = re.compile(
    r"(?i)\b(ssn|social[\s_-]?security|national[\s_-]?id|passport|driver'?s?[\s_-]?licen[cs]e"
    r"|date[\s_-]?of[\s_-]?birth|dob)\b"
)
_USERNAME_RE = re.compile(
    r"(?i)\b(user(name|id|_name)?|login|email|e[\s_-]?mail|account|signin|phone|mobile|msisdn)\b"
)
_SEED_PHRASE_RE = re.compile(
    r"(?i)\b(seed|mnemonic|recovery[\s_-]?phrase|private[\s_-]?key|wallet[\s_-]?key)\b"
)

#: `autocomplete` tokens that state the purpose unambiguously.
_AUTOCOMPLETE_PASSWORD = frozenset({"current-password", "new-password"})
_AUTOCOMPLETE_OTP = frozenset({"one-time-code"})
_AUTOCOMPLETE_CARD = frozenset({
    "cc-number", "cc-csc", "cc-exp", "cc-exp-month", "cc-exp-year", "cc-name",
})
_AUTOCOMPLETE_USERNAME = frozenset({"username", "email", "tel"})


@dataclass(frozen=True)
class InputField:
    """One form control's *metadata*. Values are never captured."""

    name: str = ""
    input_type: str = "text"
    field_id: str = ""
    placeholder: str = ""
    autocomplete: str = ""
    label_hint: str = ""
    required: bool = False
    hidden: bool = False
    max_length: int | None = None
    inputmode: str = ""
    pattern: str = ""

    @property
    def _haystack(self) -> str:
        """Searchable text from every naming attribute.

        `_` and `-` become spaces first: they are word characters to `\\b`, so
        `\\bseed\\b` would not match `wallet_seed` and `\\blogin\\b` would not
        match `login_id` - and snake/kebab-case names are the norm in HTML
        forms, so that would miss most real fields.
        """
        joined = " ".join(
            p for p in (self.name, self.field_id, self.placeholder, self.label_hint)
            if p
        )
        return joined.replace("_", " ").replace("-", " ")

    @property
    def is_hidden(self) -> bool:
        return self.hidden or self.input_type == "hidden"

    @property
    def is_password(self) -> bool:
        if self.input_type == "password":
            return True
        if self.autocomplete.lower() in _AUTOCOMPLETE_PASSWORD:
            return True
        # A text input named "pwd" is a deliberate dodge of type="password".
        return bool(_PASSWORD_RE.search(self._haystack))

    @property
    def is_otp(self) -> bool:
        if self.autocomplete.lower() in _AUTOCOMPLETE_OTP:
            return True
        if _OTP_RE.search(self._haystack):
            return True
        # A short numeric-only field is the classic un-named OTP box.
        numeric = self.inputmode.lower() == "numeric" or "\\d" in self.pattern or "[0-9]" in self.pattern
        return bool(numeric and self.max_length is not None and 3 <= self.max_length <= 8)

    @property
    def is_payment(self) -> bool:
        if self.autocomplete.lower() in _AUTOCOMPLETE_CARD:
            return True
        return bool(_CARD_RE.search(self._haystack) or _BANK_RE.search(self._haystack))

    @property
    def is_identity(self) -> bool:
        return bool(_IDENTITY_RE.search(self._haystack))

    @property
    def is_seed_phrase(self) -> bool:
        return bool(_SEED_PHRASE_RE.search(self._haystack))

    @property
    def is_username(self) -> bool:
        if self.autocomplete.lower() in _AUTOCOMPLETE_USERNAME:
            return True
        if self.input_type in ("email", "tel"):
            return True
        return bool(_USERNAME_RE.search(self._haystack)) and not self.is_password

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.input_type,
            "autocomplete": self.autocomplete,
            "required": self.required,
            "hidden": self.is_hidden,
            "is_password": self.is_password,
            "is_otp": self.is_otp,
            "is_payment": self.is_payment,
            "is_identity": self.is_identity,
            "is_seed_phrase": self.is_seed_phrase,
            "is_username": self.is_username,
        }


@dataclass(frozen=True)
class FormInfo:
    """One `<form>`: what it collects and where it submits."""

    action: str
    declared_action: str
    method: str
    fields: tuple[InputField, ...]
    page_registered_domain: str
    action_registered_domain: str
    action_scheme: str

    # --- what it collects ------------------------------------------
    @property
    def has_password_field(self) -> bool:
        return any(f.is_password for f in self.fields)

    @property
    def has_otp_field(self) -> bool:
        return any(f.is_otp for f in self.fields)

    @property
    def has_payment_field(self) -> bool:
        return any(f.is_payment for f in self.fields)

    @property
    def has_identity_field(self) -> bool:
        return any(f.is_identity for f in self.fields)

    @property
    def has_seed_phrase_field(self) -> bool:
        return any(f.is_seed_phrase for f in self.fields)

    @property
    def has_username_field(self) -> bool:
        return any(f.is_username for f in self.fields)

    @property
    def hidden_field_count(self) -> int:
        return sum(1 for f in self.fields if f.is_hidden)

    @property
    def is_credential_form(self) -> bool:
        """Collects something worth stealing: password, OTP, card, or seed phrase."""
        return (
            self.has_password_field
            or self.has_otp_field
            or self.has_payment_field
            or self.has_seed_phrase_field
        )

    # --- where it sends it ---------------------------------------
    @property
    def posts_cross_domain(self) -> bool:
        """True when the form submits to a different registered domain.

        Compared at the *registered domain* level, so `login.example.com` ->
        `api.example.com` is correctly not a mismatch, while
        `paypal-login.com` -> `collector.ru` is.
        """
        if not self.action_registered_domain or not self.page_registered_domain:
            return False
        return self.action_registered_domain != self.page_registered_domain

    @property
    def submits_insecurely(self) -> bool:
        """Credentials sent over plain HTTP."""
        return self.is_credential_form and self.action_scheme == "http"

    def collected_kinds(self) -> tuple[str, ...]:
        kinds = []
        if self.has_password_field:
            kinds.append("password")
        if self.has_otp_field:
            kinds.append("one-time code")
        if self.has_payment_field:
            kinds.append("payment card")
        if self.has_seed_phrase_field:
            kinds.append("wallet seed phrase")
        if self.has_identity_field:
            kinds.append("identity document")
        if self.has_username_field:
            kinds.append("username/email")
        return tuple(kinds)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "method": self.method,
            "action_registered_domain": self.action_registered_domain,
            "field_count": len(self.fields),
            "fields": [f.as_dict() for f in self.fields],
            "hidden_field_count": self.hidden_field_count,
            "collects": list(self.collected_kinds()),
            "is_credential_form": self.is_credential_form,
            "posts_cross_domain": self.posts_cross_domain,
            "submits_insecurely": self.submits_insecurely,
        }

    def describe(self) -> str:
        """One line for the LLM prompt - facts only, no interpretation."""
        collects = ", ".join(self.collected_kinds()) or "nothing identifiable"
        where = self.action_registered_domain or "(same page)"
        cross = " CROSS-DOMAIN" if self.posts_cross_domain else " same-domain"
        insecure = " OVER-HTTP" if self.submits_insecurely else ""
        return (
            f"method={self.method.upper()} collects=[{collects}] "
            f"submits-to={where}{cross}{insecure} "
            f"fields={len(self.fields)} hidden={self.hidden_field_count}"
        )


def analyze_form(
    *,
    action: str,
    declared_action: str,
    method: str,
    fields: tuple[InputField, ...],
    page_registered_domain: str,
) -> FormInfo:
    """Build a `FormInfo` from raw parsed attributes. Never raises."""
    scheme = ""
    if action:
        head = action.split(":", 1)[0].lower()
        if head in ("http", "https", "javascript", "data", "mailto"):
            scheme = head

    return FormInfo(
        action=action,
        declared_action=declared_action,
        method=(method or "get").lower(),
        fields=tuple(fields),
        page_registered_domain=page_registered_domain,
        action_registered_domain=registered_domain_of(action) if action else "",
        action_scheme=scheme,
    )
