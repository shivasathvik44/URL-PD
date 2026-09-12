"""Layer 4c: pluggable LLM client.

`LLMClient` is the only contract the analyzer needs: send a system prompt plus a
user prompt, get text back. Two providers implement it, and `create_llm_client`
picks between them from `LLMConfig.provider`:

  * **`ollama`** (the default) - `OllamaLLMClient` talks to an Ollama server
    running on this machine. No API key, no per-call cost, no page content
    leaving the host. The trade-off is a small local model and a much slower
    call, so the default timeout is correspondingly generous.
  * **`anthropic`** - `AnthropicLLMClient` calls the Anthropic API. The
    `anthropic` package is an *optional* extra (`pip install ".[llm]"`); if it
    is missing, or no API key is configured, `available` is False.

Availability is never an exception: a client that cannot run reports
`available = False` with an `unavailable_reason`, and Layer 4 records that as an
error rather than guessing at a verdict. Failures *during* a call raise
`LLMClientError`, which `llm_analyzer` catches - Layer 4 is best-effort, and no
provider outage may crash an analysis.

The Anthropic key is read from the environment named by `LLMConfig.api_key_env`
and is never stored in a field that could be printed: `repr()` of that client
shows the model and availability, never the credential.

Neither client is an agent. It sends two strings and returns one string; it
holds no tools, follows no instruction found in a model response, and never
lets a response choose a URL, a command or a configuration value.
"""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse

from .config import DEFAULT_LLM_CONFIG, LLMConfig
from .observability import get_logger

_log = get_logger("layer4.client")


class LLMClient(Protocol):
    """Minimal interface Layer 4 needs (so tests can inject a fake)."""

    @property
    def available(self) -> bool: ...

    def complete(self, system: str, prompt: str, *, max_tokens: int) -> str: ...


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class LLMClientError(RuntimeError):
    """A call to the provider failed. Layer 4 degrades; it never crashes."""


class LLMProviderUnavailableError(LLMClientError):
    """The provider could not be reached at all (e.g. Ollama is not running)."""


class LLMModelMissingError(LLMClientError):
    """The provider is up, but the requested model is not installed."""


class LLMTimeoutError(LLMClientError):
    """The provider did not answer within the configured timeout."""


class LLMResponseError(LLMClientError):
    """The provider answered, but not in the shape its API documents."""


# ---------------------------------------------------------------------------
# Anthropic (hosted)
# ---------------------------------------------------------------------------
class AnthropicLLMClient:
    """Calls the Anthropic Messages API, importing `anthropic` lazily.

    Deliberately not a dataclass: a generated `__repr__` would print every
    field, and this object holds an API key. The key lives in a private
    attribute and is excluded from `__repr__`.
    """

    def __init__(
        self,
        config: LLMConfig = DEFAULT_LLM_CONFIG,
        *,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._client = None
        self._unavailable_reason: str | None = None

        key = api_key or config.api_key()
        if not key:
            self._unavailable_reason = (
                f"no API key: set {config.api_key_env} in the environment"
            )
            return
        try:
            import anthropic
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = (
                f'anthropic package unavailable ({type(exc).__name__}); '
                f'install with: pip install ".[llm]"'
            )
            return
        try:
            self._client = anthropic.Anthropic(api_key=key, timeout=config.timeout)
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = f"client init failed: {type(exc).__name__}"

    def __repr__(self) -> str:  # never leak the key into logs or tracebacks
        return (
            f"AnthropicLLMClient(model={self.model!r}, available={self.available})"
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        if self._client is None:
            raise RuntimeError(self._unavailable_reason or "LLM client unavailable")
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------
#: Ollama replies to an unknown model with a 404 and a message like
#: `model "gemma3:4b" not found, try pulling it first`.
_MODEL_MISSING_MARKERS = ("not found", "no such model", "try pulling")


class OllamaLLMClient:
    """Calls a local Ollama server's `/api/chat` endpoint over plain HTTP.

    Deliberately thin: one POST, one JSON object back, the assistant text
    returned as a string. No `ollama` package is required - `requests` is
    already a core dependency for the Layer 4 page fetch, so the local provider
    adds no new dependency at all.

    Availability is decided without touching the network, so constructing a
    client (and running the test suite) never needs a server: `available` is
    False only when `requests` is missing or the configured endpoint is not a
    usable http(s) URL. A server that is down, or a model that is not pulled,
    surfaces at call time as an `LLMClientError` subclass, which Layer 4 turns
    into `llm-call-failed` rather than a crash or a verdict.

    Not a dataclass, for symmetry with `AnthropicLLMClient`: `repr()` here shows
    the endpoint and model only.
    """

    def __init__(
        self,
        config: LLMConfig = DEFAULT_LLM_CONFIG,
        *,
        session: Any | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self.base_url = (config.ollama_base_url or "").strip().rstrip("/")
        self._session = session
        self._unavailable_reason: str | None = None

        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            # The endpoint is operator configuration, never page-derived, but a
            # typo (or a file:// URL) should fail as "unavailable", not as a
            # surprising request.
            self._unavailable_reason = (
                f"invalid OLLAMA_BASE_URL {self.base_url!r}: expected an http(s) URL"
            )
            return

        if session is None:
            try:
                import requests
            except Exception as exc:  # noqa: BLE001 - requests is a core dep; be safe
                self._unavailable_reason = f"requests unavailable ({type(exc).__name__})"
                return
            self._session = requests.Session()
            # `trust_env=False` for the same reason web_fetcher sets it: an
            # ambient HTTP_PROXY must not silently route this request - which
            # carries the fetched page's text - through a third-party proxy.
            # The whole point of the local provider is that page content stays
            # on this machine.
            self._session.trust_env = False

    def __repr__(self) -> str:
        return (
            f"OllamaLLMClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"available={self.available})"
        )

    @property
    def available(self) -> bool:
        return self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send one system + one user message, return the assistant text.

        Raises an `LLMClientError` subclass for every failure mode - server
        down, model missing, timeout, non-JSON body, unexpected shape - so the
        caller has one thing to catch.
        """
        if self._unavailable_reason is not None:
            raise LLMProviderUnavailableError(self._unavailable_reason)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # One complete JSON object back, not a token stream.
            "stream": False,
            # Ollama's grammar-constrained JSON mode: the reply is guaranteed to
            # parse as JSON. It does NOT guarantee our schema - `llm_analyzer`
            # still validates every field, because a local model is no more
            # trusted than a hosted one.
            "format": "json",
            "options": {
                "temperature": self.config.ollama_temperature,
                "num_predict": max_tokens,
                "num_ctx": self.config.ollama_num_ctx,
            },
        }

        try:
            response = self._session.post(
                self.chat_url, json=payload, timeout=self.config.timeout
            )
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise self._transport_error(exc) from exc

        status = getattr(response, "status_code", None)
        if status != 200:
            raise self._http_error(status, response)

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - any JSON decoder error
            raise LLMResponseError(
                f"Ollama returned a non-JSON body ({type(exc).__name__})"
            ) from exc

        return self._text_from(body)

    # -- helpers ----------------------------------------------------------
    def _transport_error(self, exc: Exception) -> LLMClientError:
        """Map a requests exception onto our own vocabulary.

        Matched by class name rather than by `isinstance`, so an injected test
        double (or a urllib3 error that escaped requests) is classified the
        same way without importing `requests` here.
        """
        names = {type(exc).__name__} | {base.__name__ for base in type(exc).__mro__}
        if "Timeout" in names or "ReadTimeout" in names or "ConnectTimeout" in names:
            return LLMTimeoutError(
                f"Ollama did not respond within {self.config.timeout:g}s "
                f"(model {self.model!r} may still be loading)"
            )
        if "ConnectionError" in names or "ConnectionRefusedError" in names:
            return LLMProviderUnavailableError(
                f"cannot reach Ollama at {self.base_url} - is it running? "
                f"(start it with: ollama serve)"
            )
        return LLMClientError(f"Ollama request failed: {type(exc).__name__}")

    def _http_error(self, status, response: Any) -> LLMClientError:
        message = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                message = str(body.get("error") or "")
        except Exception:  # noqa: BLE001 - a body we cannot read is not fatal here
            message = ""
        lowered = message.lower()
        if status == 404 and any(m in lowered for m in _MODEL_MISSING_MARKERS):
            return LLMModelMissingError(
                f"model {self.model!r} is not installed - pull it with: "
                f"ollama pull {self.model}"
            )
        detail = f": {message[:200]}" if message else ""
        return LLMResponseError(f"Ollama returned HTTP {status}{detail}")

    def _text_from(self, body: Any) -> str:
        """Pull the assistant text out of an `/api/chat` envelope."""
        if not isinstance(body, dict):
            raise LLMResponseError(
                f"unexpected Ollama response: expected an object, got "
                f"{type(body).__name__}"
            )
        if body.get("error"):
            error = str(body["error"])
            if any(m in error.lower() for m in _MODEL_MISSING_MARKERS):
                raise LLMModelMissingError(
                    f"model {self.model!r} is not installed - pull it with: "
                    f"ollama pull {self.model}"
                )
            raise LLMResponseError(f"Ollama reported an error: {error[:200]}")

        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMResponseError(
                "unexpected Ollama response: no message.content string "
                f"(keys: {', '.join(sorted(map(str, body))[:8]) or 'none'})"
            )
        if body.get("done_reason") == "length":
            # The reply hit num_predict; it is almost certainly truncated JSON.
            # The analyzer will reject it - log why it looked malformed.
            _log.warning(
                "ollama reply truncated at num_predict",
                extra={"fields": {"model": self.model}},
            )
        return message["content"]


# ---------------------------------------------------------------------------
# provider selection
# ---------------------------------------------------------------------------
class UnavailableLLMClient:
    """Stands in for a provider we cannot build (e.g. an unknown name).

    Returning this instead of raising keeps a typo in `LLM_PROVIDER` from
    taking the whole analysis down: Layer 4 reports `llm-unavailable` and the
    decision engine falls back to Layers 1-3.
    """

    def __init__(self, reason: str, *, model: str | None = None) -> None:
        self._unavailable_reason = reason
        self.model = model

    def __repr__(self) -> str:
        return f"UnavailableLLMClient(reason={self._unavailable_reason!r})"

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        raise LLMProviderUnavailableError(self._unavailable_reason)


#: provider name -> constructor. One place to add a third provider.
_PROVIDERS = {
    "ollama": OllamaLLMClient,
    "anthropic": AnthropicLLMClient,
}


def create_llm_client(config: LLMConfig = DEFAULT_LLM_CONFIG, **kwargs: Any) -> LLMClient:
    """Build the client for `config.provider`.

    This is the only place that maps a provider name to a class; `llm_analyzer`
    holds no provider knowledge at all and just calls `complete()`. An unknown
    name yields an `UnavailableLLMClient` rather than an exception, so a
    misconfiguration degrades Layer 4 instead of failing the run.
    """
    provider = (config.provider or "").strip().lower()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        known = ", ".join(sorted(_PROVIDERS))
        return UnavailableLLMClient(
            f"unknown LLM provider {provider!r}: expected one of {known}"
        )
    try:
        return factory(config, **kwargs)
    except Exception as exc:  # noqa: BLE001 - construction must never crash Layer 4
        _log.warning(
            "llm client construction failed",
            extra={"fields": {"provider": provider, "error": type(exc).__name__}},
        )
        return UnavailableLLMClient(
            f"{provider} client could not be constructed: {type(exc).__name__}",
            model=config.model,
        )
