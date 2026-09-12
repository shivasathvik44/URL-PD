"""Layer 4c: the local Ollama provider, and provider selection.

Everything here is mocked. No test in this file needs a running Ollama server,
a pulled model, an API key or a network - `FakeSession` stands in for the
`requests` module and scripts whatever the server would have done.

Two things are covered, and the second matters more than the first:

  1. that a well-formed `/api/chat` reply is turned into the assistant text;
  2. that EVERY failure mode - server down, connection refused, model missing,
     timeout, non-JSON body, unexpected envelope - becomes a graceful
     `llm-unavailable` / `llm-call-failed` instead of an exception escaping
     into the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from conftest import llm_response
from detector.config import DEFAULT_LLM_CONFIG, LLMConfig, decision_config_from_env
from detector.html_analyzer import analyze_html
from detector.llm_analyzer import analyze_page_content
from detector.llm_client import (
    AnthropicLLMClient,
    LLMClientError,
    LLMModelMissingError,
    LLMProviderUnavailableError,
    LLMResponseError,
    LLMTimeoutError,
    OllamaLLMClient,
    UnavailableLLMClient,
    create_llm_client,
)
from detector.web_fetcher import PageContent

PAGE_URL = "https://paypal-secure-login.com/signin"

PHISH_HTML = """
<html><head><title>Sign in to PayPal</title></head>
<body><h1>Your account has been limited</h1>
<p>Verify your identity within 24 hours or your account will be closed.</p>
<form action="https://collector-site.ru/steal" method="post">
  <input type="email" name="email"><input type="password" name="pw">
</form></body></html>
"""

OLLAMA_CONFIG = LLMConfig()  # the project default: provider=ollama, gemma3:4b


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------
class FakeOllamaResponse:
    """The slice of a `requests.Response` that OllamaLLMClient reads."""

    def __init__(self, payload, *, status_code: int = 200, body_is_json: bool = True):
        self.status_code = status_code
        self._payload = payload
        self._body_is_json = body_is_json

    def json(self):
        if not self._body_is_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class FakeSession:
    """Stand-in for `requests`: records the POST, returns or raises."""

    def __init__(self, response=None, *, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.posts: list[tuple[str, dict, float | None]] = []

    def post(self, url, *, json=None, timeout=None):  # noqa: A002 - requests' own name
        self.posts.append((url, json, timeout))
        if self._exc is not None:
            raise self._exc
        return self._response

    @property
    def last_payload(self) -> dict:
        return self.posts[-1][1]


class FakeConnectionError(ConnectionError):
    """Named like `requests.exceptions.ConnectionError` for the error mapper."""


class Timeout(Exception):
    """Named like `requests.exceptions.Timeout` for the error mapper."""


def chat_reply(content: str | dict, **extra) -> dict:
    """An `/api/chat` envelope as Ollama returns it (stream=False)."""
    if isinstance(content, dict):
        content = json.dumps(content)
    envelope = {
        "model": "gemma3:4b",
        "created_at": "2026-09-12T10:00:00.000000000Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }
    envelope.update(extra)
    return envelope


def ollama_client(response=None, *, exc: Exception | None = None,
                  config: LLMConfig = OLLAMA_CONFIG):
    session = FakeSession(response, exc=exc)
    return OllamaLLMClient(config, session=session), session


def _page(html: str = PHISH_HTML, url: str = PAGE_URL) -> PageContent:
    return PageContent(
        url=url, fetched=True, final_url=url, status_code=200,
        content_type="text/html", structure=analyze_html(url, html),
    )


# ---------------------------------------------------------------------------
# configuration defaults
# ---------------------------------------------------------------------------
def test_project_default_provider_is_local_ollama():
    assert DEFAULT_LLM_CONFIG.provider == "ollama"
    assert DEFAULT_LLM_CONFIG.model == "gemma3:4b"
    assert DEFAULT_LLM_CONFIG.ollama_base_url == "http://127.0.0.1:11434"


def test_ollama_needs_no_api_key():
    """No credential is read, and none is required for availability."""
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(llm_response())))
    assert client.available is True
    assert "api_key" not in repr(client).lower()


def test_switching_provider_switches_to_that_providers_default_model():
    anthropic = DEFAULT_LLM_CONFIG.for_provider("anthropic")
    assert anthropic.provider == "anthropic"
    assert anthropic.model.startswith("claude-")

    # ...and back again.
    assert anthropic.for_provider("ollama").model == "gemma3:4b"


def test_explicit_model_survives_a_provider_switch():
    cfg = DEFAULT_LLM_CONFIG.for_provider("anthropic", model="claude-opus-5")
    assert cfg.model == "claude-opus-5"


def test_staying_on_one_provider_keeps_a_customised_model():
    cfg = replace(DEFAULT_LLM_CONFIG, model="gemma3:12b")
    assert cfg.for_provider("ollama").model == "gemma3:12b"


def test_env_selects_provider_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:99999")
    monkeypatch.setenv("DETECTOR_LLM_MODEL", "gemma3:4b")
    llm = decision_config_from_env().llm
    assert (llm.provider, llm.model) == ("ollama", "gemma3:4b")
    assert llm.ollama_base_url == "http://127.0.0.1:99999"


def test_env_can_still_select_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    llm = decision_config_from_env().llm
    assert llm.provider == "anthropic"
    assert llm.model.startswith("claude-")


def test_env_provider_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("DETECTOR_LLM_PROVIDER", "  Ollama  ")
    assert decision_config_from_env().llm.provider == "ollama"


def test_no_secret_is_read_to_choose_a_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DETECTOR_LLM_PROVIDER", "ollama")
    assert decision_config_from_env().llm.api_key() is None


# ---------------------------------------------------------------------------
# provider selection
# ---------------------------------------------------------------------------
def test_provider_ollama_builds_the_ollama_client():
    client = create_llm_client(DEFAULT_LLM_CONFIG.for_provider("ollama"))
    assert isinstance(client, OllamaLLMClient)
    assert client.model == "gemma3:4b"


def test_provider_anthropic_still_builds_the_anthropic_client():
    """The hosted provider was not removed - only joined by a local one."""
    client = create_llm_client(DEFAULT_LLM_CONFIG.for_provider("anthropic"))
    assert isinstance(client, AnthropicLLMClient)
    assert client.model.startswith("claude-")


def test_anthropic_without_a_key_is_unavailable_not_an_exception(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = create_llm_client(DEFAULT_LLM_CONFIG.for_provider("anthropic"))
    assert client.available is False
    assert "ANTHROPIC_API_KEY" in client.unavailable_reason


def test_unknown_provider_degrades_instead_of_raising():
    client = create_llm_client(replace(DEFAULT_LLM_CONFIG, provider="gpt-9"))
    assert isinstance(client, UnavailableLLMClient)
    assert client.available is False
    assert "unknown LLM provider" in client.unavailable_reason


def test_analyzer_uses_the_configured_provider_when_no_client_is_injected(monkeypatch):
    """`llm_analyzer` asks the factory; it holds no provider knowledge itself."""
    built: list[LLMConfig] = []

    class Recorder:
        available = True

        def complete(self, system, prompt, *, max_tokens=1024):
            return json.dumps(llm_response(risk_level="medium"))

    def fake_factory(config, **kwargs):
        built.append(config)
        return Recorder()

    monkeypatch.setattr("detector.llm_client.create_llm_client", fake_factory)
    result = analyze_page_content(PAGE_URL, _page(), llm_config=OLLAMA_CONFIG)

    assert [c.provider for c in built] == ["ollama"]
    assert result.performed is True
    assert result.provider == "ollama"
    assert result.model == "gemma3:4b"


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------
def test_successful_call_returns_the_assistant_text():
    body = llm_response(risk_level="high", confidence="high", brand_claimed="PayPal")
    client, session = ollama_client(FakeOllamaResponse(chat_reply(body)))

    text = client.complete("SYSTEM", "USER", max_tokens=256)

    assert json.loads(text) == body
    url, payload, timeout = session.posts[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert timeout == OLLAMA_CONFIG.timeout


def test_request_shape_matches_the_ollama_chat_api():
    client, session = ollama_client(FakeOllamaResponse(chat_reply(llm_response())))
    client.complete("SYSTEM TEXT", "USER TEXT", max_tokens=512)
    payload = session.last_payload

    assert payload["model"] == "gemma3:4b"
    assert payload["stream"] is False           # one object back, not a token stream
    assert payload["format"] == "json"          # grammar-constrained JSON
    assert payload["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.0     # deterministic


def test_context_window_is_large_enough_to_hold_the_system_prompt():
    """Ollama drops the OLDEST tokens on overflow - which is the system prompt.

    An undersized context would silently delete the prompt-injection defences,
    so the requested num_ctx must exceed Ollama's 4096-token default.
    """
    client, session = ollama_client(FakeOllamaResponse(chat_reply(llm_response())))
    client.complete("SYSTEM", "USER", max_tokens=1024)
    assert session.last_payload["options"]["num_ctx"] >= 8192


def test_end_to_end_through_the_analyzer_with_a_mocked_server():
    body = llm_response(
        brand_claimed="PayPal", brand_impersonation=True, credential_request=True,
        social_engineering=True, risk_level="high", confidence="high",
        page_purpose="login",
    )
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(body)))
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )

    assert result.performed is True
    assert result.error is None
    assert result.risk_level == "high"
    assert result.llm_likelihood == 0.95
    assert result.provider == "ollama"
    assert result.model == "gemma3:4b"
    assert result.schema_violations == ()


def test_schema_is_shared_with_anthropic_not_forked():
    """The same JSON body validates identically whichever provider returned it."""
    body = llm_response(risk_level="medium", confidence="low", page_purpose="login")
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(body)))

    from conftest import FakeLLM

    local = analyze_page_content(PAGE_URL, _page(), llm_client=client,
                                 llm_config=OLLAMA_CONFIG)
    hosted = analyze_page_content(PAGE_URL, _page(), llm_client=FakeLLM(body),
                                  llm_config=OLLAMA_CONFIG.for_provider("anthropic"))

    for field in ("risk_level", "confidence", "page_purpose", "llm_likelihood",
                  "brand_impersonation", "schema_violations"):
        assert getattr(local, field) == getattr(hosted, field)


# ---------------------------------------------------------------------------
# failure modes - none of these may escape as an exception
# ---------------------------------------------------------------------------
def test_ollama_not_running_is_a_provider_unavailable_error():
    client, _session = ollama_client(exc=FakeConnectionError("refused"))
    with pytest.raises(LLMProviderUnavailableError) as excinfo:
        client.complete("s", "p", max_tokens=64)
    assert "is it running" in str(excinfo.value)


def test_connection_refused_degrades_the_analysis_gracefully():
    client, _session = ollama_client(exc=FakeConnectionError("[Errno 111] refused"))
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )

    assert result.performed is False
    assert result.error.startswith("llm-call-failed")
    assert result.provider == "ollama"
    # An outage is never a verdict.
    assert result.risk_level is None
    assert result.llm_likelihood is None


def test_request_timeout_is_reported_as_a_timeout():
    client, _session = ollama_client(exc=Timeout("read timed out"))
    with pytest.raises(LLMTimeoutError):
        client.complete("s", "p", max_tokens=64)


def test_timeout_degrades_the_analysis_gracefully():
    client, _session = ollama_client(exc=Timeout("read timed out"))
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )
    assert result.performed is False
    assert "llm-call-failed" in result.error
    assert result.llm_likelihood is None


def test_model_not_installed_names_the_pull_command():
    response = FakeOllamaResponse(
        {"error": 'model "gemma3:4b" not found, try pulling it first'},
        status_code=404,
    )
    client, _session = ollama_client(response)
    with pytest.raises(LLMModelMissingError) as excinfo:
        client.complete("s", "p", max_tokens=64)
    assert "ollama pull gemma3:4b" in str(excinfo.value)


def test_model_not_installed_degrades_the_analysis_gracefully():
    response = FakeOllamaResponse(
        {"error": 'model "gemma3:4b" not found, try pulling it first'},
        status_code=404,
    )
    client, _session = ollama_client(response)
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )
    assert result.performed is False
    assert "llm-call-failed" in result.error
    assert "ollama pull" in result.error


def test_server_error_status_is_a_response_error():
    client, _session = ollama_client(
        FakeOllamaResponse({"error": "something exploded"}, status_code=500)
    )
    with pytest.raises(LLMResponseError) as excinfo:
        client.complete("s", "p", max_tokens=64)
    assert "500" in str(excinfo.value)


def test_malformed_non_json_body_is_a_response_error():
    client, _session = ollama_client(
        FakeOllamaResponse("<html>proxy error</html>", body_is_json=False)
    )
    with pytest.raises(LLMResponseError):
        client.complete("s", "p", max_tokens=64)


@pytest.mark.parametrize("envelope", [
    {"done": True},                                    # no message at all
    {"message": {"role": "assistant"}},                # no content
    {"message": {"role": "assistant", "content": 42}},  # content not a string
    {"message": "just a string"},                      # message not an object
    ["not", "an", "object"],                           # not an object at all
])
def test_unexpected_response_shapes_are_rejected(envelope):
    client, _session = ollama_client(FakeOllamaResponse(envelope))
    with pytest.raises(LLMResponseError):
        client.complete("s", "p", max_tokens=64)


def test_error_field_in_a_200_body_is_still_an_error():
    client, _session = ollama_client(FakeOllamaResponse({"error": "out of memory"}))
    with pytest.raises(LLMResponseError):
        client.complete("s", "p", max_tokens=64)


def test_invalid_llm_json_is_not_guessed_at():
    """The transport worked; the model's own output is not our schema."""
    client, _session = ollama_client(
        FakeOllamaResponse(chat_reply("I think this page is probably fine!"))
    )
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )
    assert result.performed is False
    assert result.error.startswith("unparseable-response")
    assert result.risk_level is None


def test_out_of_schema_fields_from_a_local_model_are_rejected():
    body = llm_response(risk_level="low")
    body.update({"execute": "rm -rf /", "fetch_url": "http://169.254.169.254/"})
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(body)))
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )

    assert result.performed is True
    assert any("unexpected field" in v for v in result.schema_violations)
    for forbidden in ("execute", "fetch_url"):
        assert forbidden not in result.as_dict()


def test_a_truncated_reply_is_rejected_rather_than_half_parsed():
    truncated = '{"risk_level": "low", "confidence": "hi'
    client, _session = ollama_client(
        FakeOllamaResponse(chat_reply(truncated, done_reason="length"))
    )
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )
    assert result.performed is False
    assert "unparseable-response" in result.error


def test_every_client_failure_shares_one_base_class():
    """`llm_analyzer` catches broadly, but callers get one thing to catch."""
    for error in (LLMProviderUnavailableError, LLMModelMissingError,
                  LLMTimeoutError, LLMResponseError):
        assert issubclass(error, LLMClientError)


# ---------------------------------------------------------------------------
# the endpoint is operator configuration, never page-derived
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_url", [
    "file:///etc/passwd", "ftp://127.0.0.1", "127.0.0.1:11434", "", "   ",
])
def test_a_non_http_endpoint_is_unavailable_rather_than_requested(bad_url):
    client = OllamaLLMClient(replace(OLLAMA_CONFIG, ollama_base_url=bad_url))
    assert client.available is False
    assert "http(s)" in client.unavailable_reason

    with pytest.raises(LLMProviderUnavailableError):
        client.complete("s", "p", max_tokens=64)


def test_unavailable_client_reports_llm_unavailable_through_the_analyzer():
    client = OllamaLLMClient(replace(OLLAMA_CONFIG, ollama_base_url="file:///x"))
    result = analyze_page_content(
        PAGE_URL, _page(), llm_config=OLLAMA_CONFIG, llm_client=client
    )
    assert result.performed is False
    assert result.error == "llm-unavailable"
    assert result.provider == "ollama"


def test_base_url_trailing_slash_does_not_double_up():
    client = OllamaLLMClient(
        replace(OLLAMA_CONFIG, ollama_base_url="http://127.0.0.1:11434/")
    )
    assert client.chat_url == "http://127.0.0.1:11434/api/chat"


# ---------------------------------------------------------------------------
# prompt injection: the local provider inherits the same defences
# ---------------------------------------------------------------------------
def _injection_page():
    from pathlib import Path

    html = (Path(__file__).parent / "fixtures" / "prompt_injection.html").read_text()
    url = "https://paypal-account-verify.com/signin"
    return url, PageContent(
        url=url, fetched=True, final_url=url, status_code=200,
        content_type="text/html", structure=analyze_html(url, html),
    )


def test_hostile_page_text_stays_in_the_user_message_and_inside_the_fence():
    url, page = _injection_page()
    client, session = ollama_client(
        FakeOllamaResponse(chat_reply(llm_response(prompt_injection_attempt=True,
                                                   risk_level="high")))
    )
    result = analyze_page_content(url, page, llm_config=OLLAMA_CONFIG,
                                  llm_client=client)

    messages = {m["role"]: m["content"] for m in session.last_payload["messages"]}

    # 1. Two distinct roles reach the model: instructions, then data.
    assert set(messages) == {"system", "user"}

    # 2. The page's instructions never reach the system role.
    assert "Ignore all previous instructions" not in messages["system"]
    assert "NEW INSTRUCTIONS" not in messages["system"]
    assert "UNTRUSTED PAGE DATA" in messages["system"]

    # 3. They sit inside the nonce fence in the user message...
    user = messages["user"]
    assert user.index("Ignore all previous instructions") > user.index(
        "BEGIN_UNTRUSTED_PAGE_DATA:")
    # ...and the page's own fence markers were neutralised, so it cannot forge
    # a boundary and append trailing "real" instructions.
    assert user.count("BEGIN_UNTRUSTED_PAGE_DATA:") == 1
    assert user.count("END_UNTRUSTED_PAGE_DATA:") == 1
    assert "[fence-marker-removed]" in user

    assert result.prompt_injection_attempt is True


def test_page_content_cannot_influence_the_request_the_client_makes():
    """The endpoint, model and options are operator config, not page data."""
    url, page = _injection_page()
    client, session = ollama_client(FakeOllamaResponse(chat_reply(llm_response())))
    analyze_page_content(url, page, llm_config=OLLAMA_CONFIG, llm_client=client)

    request_url, payload, _timeout = session.posts[0]
    assert request_url == "http://127.0.0.1:11434/api/chat"      # not 169.254.169.254
    assert payload["model"] == "gemma3:4b"
    assert set(payload) == {"model", "messages", "stream", "format", "options"}
    # No tools, no functions: the model cannot call anything.
    assert "tools" not in payload


def test_a_local_model_talked_into_saying_safe_is_still_only_one_input():
    """If the small local model falls for the injection, code facts still hold."""
    url, page = _injection_page()
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(llm_response(
        risk_level="low", confidence="high", prompt_injection_attempt=False,
        brand_impersonation=False,
    ))))
    result = analyze_page_content(url, page, llm_config=OLLAMA_CONFIG,
                                  llm_client=client)

    assert result.risk_level == "low"
    # ...but the statically-computed facts the decision engine also weighs are
    # untouched by anything the model said.
    assert page.structure.has_cross_domain_credential_post is True
    assert page.structure.has_otp_field is True
    assert "credential-collector.ru" in page.structure.cross_domain_form_targets


# ---------------------------------------------------------------------------
# through the pipeline and the CLI
# ---------------------------------------------------------------------------
UNCERTAIN = 0.5          # inside the default 0.30-0.70 band, so Layer 4 runs

PIPELINE_HTML = """
<html><head><title>Verify your PayPal account</title></head>
<body><h1>Your account has been limited</h1>
<p>Verify within 24 hours or your account will be closed.</p>
<form action="https://collector-site.ru/x" method="post">
<input type="email" name="e"><input type="password" name="p">
<input type="text" name="otp" maxlength="6"></form></body></html>
"""


def _pipeline(probability=UNCERTAIN, *, client, html=PIPELINE_HTML,
              url="https://paypal-verify.example.com/login"):
    from conftest import FakePredictor
    from detector.config import DEFAULT_DECISION_CONFIG
    from detector.pipeline import analyze_url

    def fetch(target_url, network_config):
        assert network_config is not None
        return PageContent(
            url=target_url, fetched=True, final_url=url, status_code=200,
            content_type="text/html", structure=analyze_html(url, html),
        )

    return analyze_url(
        url,
        decision_config=replace(DEFAULT_DECISION_CONFIG, llm=OLLAMA_CONFIG),
        run_layer2=False,
        predictor=FakePredictor(probability),
        fetcher=fetch,
        llm_client=client,
        use_caches=False,
    )


def test_layer4_result_from_ollama_reaches_the_decision_engine():
    client, _session = ollama_client(FakeOllamaResponse(chat_reply(llm_response(
        brand_claimed="PayPal", brand_impersonation=True, credential_request=True,
        social_engineering=True, page_purpose="account_verification",
        risk_level="high", confidence="high",
    ))))
    result = _pipeline(client=client)

    assert result.metadata["layer4_attempted"] is True
    assert result.llm.performed is True
    assert result.llm.provider == "ollama"
    assert result.decision.llm_likelihood == 0.95       # the LLM reached the engine
    assert result.decision.classification == "phishing"
    assert any(r.source == "llm" for r in result.decision.reasons)


#: An ambiguous page with NO code-verified proof against it: a same-origin
#: login form. Whatever the verdict is, Layer 4 is what decides it.
AMBIGUOUS_HTML = """
<html><head><title>Member sign in</title></head>
<body><h1>Sign in to continue</h1>
<form action="/session" method="post">
<input type="email" name="e"><input type="password" name="p"></form></body></html>
"""


def test_ollama_offline_never_makes_a_page_phishing_on_its_own():
    """An outage is a missing opinion, not an accusation."""
    client, _session = ollama_client(exc=FakeConnectionError("refused"))
    result = _pipeline(client=client, html=AMBIGUOUS_HTML)

    assert result.llm.performed is False
    assert "llm-call-failed" in result.llm.error
    assert result.decision.llm_likelihood is None
    assert result.decision.classification == "uncertain"
    assert result.decision.decision_path == "layer4_unavailable"
    assert result.decision.risk_score == UNCERTAIN      # the ML score, untouched
    assert any("layer4-llm" in e for e in result.errors)
    # No reason may cite an LLM that never answered.
    assert all(r.source != "llm" for r in result.decision.reasons)


def test_thresholds_still_gate_layer4_with_the_local_provider():
    """A confident ML score must not start paying for a local inference run."""
    for probability in (0.95, 0.05):
        client, session = ollama_client(
            FakeOllamaResponse(chat_reply(llm_response()))
        )
        result = _pipeline(probability, client=client)
        assert result.metadata["layer4_attempted"] is False
        assert session.posts == []                 # Ollama was never contacted


def test_cli_reports_the_provider_and_model():
    from detector.cli import render_analysis

    client, _session = ollama_client(FakeOllamaResponse(chat_reply(llm_response(
        risk_level="high", confidence="medium", brand_claimed="PayPal",
    ))))
    rendered = render_analysis(_pipeline(client=client))

    assert "LAYER 4  page content + LLM" in rendered
    assert "LLM risk ........... high" in rendered
    assert "provider ........... ollama" in rendered
    assert "model .............. gemma3:4b" in rendered


def test_cli_reports_the_provider_when_ollama_is_down():
    from detector.cli import render_analysis

    client, _session = ollama_client(exc=FakeConnectionError("refused"))
    rendered = render_analysis(_pipeline(client=client))

    assert "LLM ................ unavailable" in rendered
    assert "is it running" in rendered            # the operator-facing hint
    assert "provider ........... ollama" in rendered


def test_the_local_call_ignores_ambient_proxy_configuration(monkeypatch):
    """An HTTP_PROXY must not route the page's text through a third party.

    The whole point of the local provider is that page content stays on this
    machine, so the session is built with `trust_env = False` - the same
    posture `web_fetcher` takes.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    client = OllamaLLMClient(OLLAMA_CONFIG)
    assert client.available is True
    assert client._session.trust_env is False
