"""Layer 4b: static HTML analysis.

Parses fetched HTML with the stdlib `html.parser` - no extra dependency, and
crucially **nothing is executed**: no JavaScript engine, no DOM, no network
fetches for sub-resources. What comes out is a `PageStructure` of deterministic,
code-computed facts:

  * the `<title>` and the visible text (script/style/hidden content excluded);
  * every `<form>` with its action, method and input fields (see
    `form_analyzer.py` for what those fields mean);
  * `<iframe>` sources, split into same-site and cross-site;
  * hidden content - `display:none` / `visibility:hidden` / `hidden` /
    zero-size / off-screen elements, and the text inside them;
  * `<script>` inventory: how many, how much inline code, external sources, and
    a few obfuscation markers (`eval(`, `atob(`, `fromCharCode`, long
    percent-encoded or hex blobs);
  * password-manager-relevant metadata: `<meta>` refresh redirects, `<base>`
    overrides, and the favicon host.

These are facts, not verdicts, and they are computed *before* any LLM sees the
page - so they cannot be talked out of by text embedded in the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

from .form_analyzer import FormInfo, InputField, analyze_form
from .lexical_features import registered_domain_of
from .security import sanitize_text

#: Tags whose text content is never visible to a user.
_INVISIBLE_TAGS = frozenset({"script", "style", "noscript", "template", "head"})

#: Inline styles that hide an element.
_HIDDEN_STYLE_RE = re.compile(
    r"(?i)(display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?!\.)"
    r"|font-size\s*:\s*0|height\s*:\s*0(?:px)?\b|width\s*:\s*0(?:px)?\b"
    r"|left\s*:\s*-\d{3,}|text-indent\s*:\s*-\d{3,})"
)

#: Markers of deliberately obfuscated script content.
_OBFUSCATION_PATTERNS = (
    ("eval", re.compile(r"\beval\s*\(")),
    ("atob", re.compile(r"\batob\s*\(")),
    ("fromCharCode", re.compile(r"fromCharCode")),
    ("unescape", re.compile(r"\bunescape\s*\(")),
    ("document_write", re.compile(r"document\s*\.\s*write")),
    ("long_hex_blob", re.compile(r"(?:\\x[0-9a-fA-F]{2}){20,}")),
    ("long_percent_blob", re.compile(r"(?:%[0-9a-fA-F]{2}){30,}")),
    ("long_base64_blob", re.compile(r"['\"][A-Za-z0-9+/]{200,}={0,2}['\"]")),
)


@dataclass(frozen=True)
class ScriptInfo:
    """Inventory of the page's scripts (never executed, only counted)."""

    total: int = 0
    inline_count: int = 0
    external_count: int = 0
    inline_chars: int = 0
    external_sources: tuple[str, ...] = ()
    cross_site_sources: tuple[str, ...] = ()
    obfuscation_markers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "inline_count": self.inline_count,
            "external_count": self.external_count,
            "inline_chars": self.inline_chars,
            "external_sources": list(self.external_sources),
            "cross_site_sources": list(self.cross_site_sources),
            "obfuscation_markers": list(self.obfuscation_markers),
        }


@dataclass(frozen=True)
class PageStructure:
    """Everything the static analyzer extracted from one HTML document."""

    url: str
    registered_domain: str = ""
    title: str | None = None
    text: str = ""
    text_truncated: bool = False
    forms: tuple[FormInfo, ...] = ()
    iframes: tuple[str, ...] = ()
    cross_site_iframes: tuple[str, ...] = ()
    hidden_element_count: int = 0
    hidden_text: str = ""
    hidden_input_count: int = 0
    scripts: ScriptInfo = field(default_factory=ScriptInfo)
    meta_refresh_to: str | None = None
    base_href: str | None = None
    favicon_host: str | None = None
    parse_error: str | None = None

    # --- derived, code-computed signals (no LLM involved) ---------------
    @property
    def has_password_field(self) -> bool:
        return any(f.has_password_field for f in self.forms)

    @property
    def has_otp_field(self) -> bool:
        return any(f.has_otp_field for f in self.forms)

    @property
    def has_payment_field(self) -> bool:
        return any(f.has_payment_field for f in self.forms)

    @property
    def has_credential_form(self) -> bool:
        return any(f.is_credential_form for f in self.forms)

    @property
    def cross_domain_form_targets(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                f.action_registered_domain for f in self.forms if f.posts_cross_domain
            )
        )

    @property
    def has_cross_domain_credential_post(self) -> bool:
        """The single strongest static signal here: credentials leaving the site."""
        return any(f.is_credential_form and f.posts_cross_domain for f in self.forms)

    def as_dict(self) -> dict:
        return {
            "registered_domain": self.registered_domain,
            "title": self.title,
            "text": self.text,
            "text_truncated": self.text_truncated,
            "forms": [f.as_dict() for f in self.forms],
            "iframes": list(self.iframes),
            "cross_site_iframes": list(self.cross_site_iframes),
            "hidden_element_count": self.hidden_element_count,
            "hidden_text": self.hidden_text,
            "hidden_input_count": self.hidden_input_count,
            "scripts": self.scripts.as_dict(),
            "meta_refresh_to": self.meta_refresh_to,
            "base_href": self.base_href,
            "favicon_host": self.favicon_host,
            "parse_error": self.parse_error,
            "signals": {
                "has_password_field": self.has_password_field,
                "has_otp_field": self.has_otp_field,
                "has_payment_field": self.has_payment_field,
                "has_credential_form": self.has_credential_form,
                "cross_domain_form_targets": list(self.cross_domain_form_targets),
                "has_cross_domain_credential_post": self.has_cross_domain_credential_post,
            },
        }


def _is_hidden(attrs: dict) -> bool:
    if "hidden" in attrs:
        return True
    if attrs.get("type", "").lower() == "hidden":
        return True
    if attrs.get("aria-hidden", "").lower() == "true":
        return True
    style = attrs.get("style", "")
    if style and _HIDDEN_STYLE_RE.search(style):
        return True
    for dim in ("width", "height"):
        value = attrs.get(dim, "").strip()
        if value in ("0", "0px"):
            return True
    return False


class _StructureParser(HTMLParser):
    """Single pass over the document, collecting everything we need."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title: str | None = None
        self.base_href: str | None = None
        self.meta_refresh_to: str | None = None
        self.favicon: str | None = None

        self._invisible_depth = 0
        self._hidden_depth = 0
        self._in_title = False
        self._in_script = False

        self.text_parts: list[str] = []
        self.hidden_text_parts: list[str] = []

        self.forms: list[dict] = []
        self._current_form: dict | None = None
        self.hidden_element_count = 0
        self.hidden_input_count = 0

        self.iframes: list[str] = []
        self.script_srcs: list[str] = []
        self.inline_script: list[str] = []
        self.script_total = 0

    # -- helpers ------------------------------------------------------
    def _resolve(self, value: str) -> str:
        base = self.base_href or self.base_url
        try:
            return urljoin(base, value)
        except Exception:  # noqa: BLE001 - a hostile href must not crash parsing
            return value

    # -- tag handling -------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        hidden = _is_hidden(a)
        if hidden:
            self.hidden_element_count += 1

        if tag in _INVISIBLE_TAGS:
            self._invisible_depth += 1
        if hidden and tag not in ("input", "img", "br"):
            self._hidden_depth += 1

        if tag == "title":
            self._in_title = True
        elif tag == "base" and a.get("href"):
            self.base_href = self._resolve(a["href"])
        elif tag == "meta":
            if a.get("http-equiv", "").lower() == "refresh":
                content = a.get("content", "")
                if ";" in content.lower() and "url=" in content.lower():
                    target = content.lower().split("url=", 1)[1].strip().strip("'\"")
                    self.meta_refresh_to = self._resolve(target)
        elif tag == "link":
            rel = a.get("rel", "").lower()
            if "icon" in rel and a.get("href"):
                self.favicon = self._resolve(a["href"])
        elif tag == "iframe":
            src = a.get("src") or a.get("data-src")
            if src:
                self.iframes.append(self._resolve(src))
        elif tag == "script":
            self._in_script = True
            self.script_total += 1
            if a.get("src"):
                self.script_srcs.append(self._resolve(a["src"]))
        elif tag == "form":
            action = a.get("action", "")
            self._current_form = {
                "action": self._resolve(action) if action else (self.base_href or self.base_url),
                "declared_action": action,
                "method": (a.get("method") or "get").lower(),
                "fields": [],
            }
        elif tag in ("input", "textarea", "select", "button"):
            field_hidden = hidden or a.get("type", "").lower() == "hidden"
            if field_hidden:
                self.hidden_input_count += 1
            if self._current_form is not None:
                self._current_form["fields"].append(
                    InputField(
                        name=a.get("name", ""),
                        input_type=(a.get("type") or ("textarea" if tag == "textarea" else "text")).lower(),
                        field_id=a.get("id", ""),
                        placeholder=a.get("placeholder", ""),
                        autocomplete=a.get("autocomplete", ""),
                        label_hint=a.get("aria-label", "") or a.get("title", ""),
                        required="required" in a,
                        hidden=field_hidden,
                        max_length=_safe_int(a.get("maxlength")),
                        inputmode=a.get("inputmode", ""),
                        pattern=a.get("pattern", ""),
                    )
                )

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in ("input", "img", "br", "meta", "link", "base"):
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE_TAGS and self._invisible_depth > 0:
            self._invisible_depth -= 1
        if self._hidden_depth > 0 and tag not in ("input", "img", "br", "meta", "link"):
            self._hidden_depth -= 1

        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_script = False
        elif tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data
            return
        if self._in_script:
            self.inline_script.append(data)
            return
        if self._invisible_depth:
            return
        stripped = data.strip()
        if not stripped:
            return
        if self._hidden_depth:
            self.hidden_text_parts.append(stripped)
        else:
            self.text_parts.append(stripped)


def _safe_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _scripts_from(parser: _StructureParser, page_domain: str) -> ScriptInfo:
    inline = "".join(parser.inline_script)
    markers = tuple(
        name for name, pattern in _OBFUSCATION_PATTERNS if pattern.search(inline)
    )
    srcs = tuple(dict.fromkeys(parser.script_srcs))
    cross = tuple(
        s for s in srcs
        if page_domain and registered_domain_of(s) and registered_domain_of(s) != page_domain
    )
    return ScriptInfo(
        total=parser.script_total,
        inline_count=parser.script_total - len(srcs),
        external_count=len(srcs),
        inline_chars=len(inline),
        external_sources=srcs[:20],
        cross_site_sources=cross[:20],
        obfuscation_markers=markers,
    )


def analyze_html(
    url: str,
    html: str,
    *,
    max_text_chars: int = 20_000,
) -> PageStructure:
    """Parse `html` into a `PageStructure`. Pure function - no I/O, never raises."""
    page_domain = registered_domain_of(url)
    parser = _StructureParser(url)
    parse_error: str | None = None
    try:
        parser.feed(html or "")
        parser.close()
    except Exception as exc:  # noqa: BLE001 - malformed HTML is expected input
        parse_error = f"{type(exc).__name__}: {exc}"

    raw_text = " ".join(parser.text_parts)
    text = sanitize_text(raw_text, max_text_chars)
    text_truncated = len(raw_text) > max_text_chars

    hidden_text = sanitize_text(" ".join(parser.hidden_text_parts), 2000)

    forms = tuple(
        analyze_form(
            action=f["action"],
            declared_action=f["declared_action"],
            method=f["method"],
            fields=tuple(f["fields"]),
            page_registered_domain=page_domain,
        )
        for f in parser.forms
    )

    iframes = tuple(dict.fromkeys(parser.iframes))[:20]
    cross_iframes = tuple(
        s for s in iframes
        if page_domain and registered_domain_of(s) and registered_domain_of(s) != page_domain
    )

    favicon_host = None
    if parser.favicon:
        favicon_host = registered_domain_of(parser.favicon) or None

    return PageStructure(
        url=url,
        registered_domain=page_domain,
        title=sanitize_text(parser.title, 300) or None,
        text=text,
        text_truncated=text_truncated,
        forms=forms,
        iframes=iframes,
        cross_site_iframes=cross_iframes,
        hidden_element_count=parser.hidden_element_count,
        hidden_text=hidden_text,
        hidden_input_count=parser.hidden_input_count,
        scripts=_scripts_from(parser, page_domain),
        meta_refresh_to=parser.meta_refresh_to,
        base_href=parser.base_href,
        favicon_host=favicon_host,
        parse_error=parse_error,
    )
