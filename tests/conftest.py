"""Shared test doubles.

Nothing in this suite touches the live network, a real LLM, or a real API key.
Every layer that does I/O has an injection point, and the fakes here are what
get injected:

    FakeResolver     dnspython stand-in                  (Layer 2)
    FakeHttpClient   requests.Session stand-in           (Layer 4 fetch)
    FakeLLM          LLMClient stand-in                  (Layer 4 analysis)
    FakeModel        PhishingModel stand-in              (Layer 3)
"""

from __future__ import annotations

import json

import dns.exception
import dns.resolver
import pytest


# ---------------------------------------------------------------------------
# Layer 2: DNS
# ---------------------------------------------------------------------------
class FakeRRset:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl


class FakeAnswer:
    """Mimics the slice of dnspython's Answer that dns_features touches."""

    def __init__(self, items, ttl: int = 300) -> None:
        self._items = list(items)
        self.rrset = FakeRRset(ttl)

    def __iter__(self):
        return iter(self._items)


class FakeMX:
    def __init__(self, exchange: str, preference: int = 10) -> None:
        self.exchange = exchange
        self.preference = preference

    def __str__(self) -> str:  # pragma: no cover - not used by the code path
        return f"{self.preference} {self.exchange}"


class FakeResolver:
    """Duck-typed stand-in for dns.resolver.Resolver.

    `records` maps an rdtype ("A", "AAAA", "NS", "MX", "PTR") to one of:
      * a FakeAnswer                -> returned
      * an Exception instance/class -> raised
      * a callable(qname)           -> called, result returned/raised
    A missing rdtype raises dns.resolver.NoAnswer.
    """

    def __init__(self, records: dict | None = None) -> None:
        self.records = records or {}
        self.calls: list[tuple[str, str]] = []

    def resolve(self, qname, rdtype, *args, **kwargs):
        self.calls.append((str(qname), str(rdtype)))
        value = self.records.get(str(rdtype))
        if value is None:
            raise dns.resolver.NoAnswer
        if callable(value) and not isinstance(value, BaseException):
            value = value(str(qname))
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, type) and issubclass(value, BaseException):
            raise value()
        return value


# ---------------------------------------------------------------------------
# Layer 4: HTTP
# ---------------------------------------------------------------------------
class FakeResponse:
    """The slice of a `requests.Response` that web_fetcher reads."""

    def __init__(self, *, status_code=200, body: str = "", headers=None,
                 encoding="utf-8", chunk_size=None):
        self.status_code = status_code
        self.headers = headers if headers is not None else {
            "Content-Type": "text/html; charset=utf-8"
        }
        self.encoding = encoding
        self.apparent_encoding = encoding
        self._body = body.encode(encoding or "utf-8", "replace") if isinstance(body, str) else body
        self._chunk_size = chunk_size
        self.closed = False

    def iter_content(self, chunk_size=16384, decode_unicode=False):
        size = self._chunk_size or chunk_size
        for i in range(0, len(self._body), size):
            yield self._body[i: i + size]

    def close(self):
        self.closed = True


class FakeHttpClient:
    """Stand-in for a requests.Session.

    `responses` is either a single FakeResponse (returned for every request) or
    a dict mapping URL -> FakeResponse | Exception, so redirect chains and
    per-host failures can be scripted.
    """

    def __init__(self, responses=None, *, exc: Exception | None = None):
        self._responses = responses
        self._exc = exc
        self.requests: list[tuple[str, str, dict]] = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if self._exc is not None:
            raise self._exc
        if isinstance(self._responses, dict):
            value = self._responses.get(url)
            if value is None:
                raise ConnectionError(f"no scripted response for {url}")
            if isinstance(value, BaseException):
                raise value
            return value
        return self._responses

    @property
    def urls(self) -> list[str]:
        return [u for _m, u, _k in self.requests]


# ---------------------------------------------------------------------------
# Layer 4: LLM
# ---------------------------------------------------------------------------
class FakeLLM:
    """Stand-in for an LLMClient. Records what it was asked."""

    def __init__(self, response=None, *, available: bool = True,
                 exc: Exception | None = None):
        if isinstance(response, dict):
            response = json.dumps(response)
        self._response = response
        self._available = available
        self._exc = exc
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls.append((system, prompt))
        if self._exc is not None:
            raise self._exc
        return self._response

    @property
    def last_prompt(self) -> str:
        return self.calls[-1][1] if self.calls else ""

    @property
    def last_system(self) -> str:
        return self.calls[-1][0] if self.calls else ""


def llm_response(**overrides) -> dict:
    """A schema-valid LLM response, with overrides."""
    base = {
        "brand_claimed": None,
        "brand_impersonation": False,
        "social_engineering": False,
        "social_engineering_tactics": [],
        "credential_request": False,
        "page_purpose": "content",
        "suspicious_observations": [],
        "prompt_injection_attempt": False,
        "risk_level": "low",
        "confidence": "high",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Layer 3: model
# ---------------------------------------------------------------------------
class FakeModel:
    """Stand-in for PhishingModel that returns a fixed probability."""

    def __init__(self, probability: float, feature_names=None):
        from detector.feature_schema import active_feature_names

        self.feature_names = list(feature_names or active_feature_names(True))
        self._probability = probability
        self.info = None
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return [self._probability for _ in X]

    def feature_importances(self):
        return [(n, 0.0) for n in self.feature_names]


class FakePredictor:
    """Stand-in for Predictor used by the pipeline."""

    def __init__(self, probability: float | None, *, exc: Exception | None = None):
        self._probability = probability
        self._exc = exc
        self.model = None
        self.calls = 0

    def predict_from_layers(self, layer1, layer2=None):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._probability


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_answer():
    return FakeAnswer


@pytest.fixture
def fake_mx():
    return FakeMX


@pytest.fixture
def make_resolver():
    return FakeResolver


@pytest.fixture
def make_http_client():
    return FakeHttpClient


@pytest.fixture
def make_response():
    return FakeResponse


@pytest.fixture
def make_llm():
    return FakeLLM


@pytest.fixture
def allow_private_network():
    """A NetworkConfig that permits loopback targets, for fetcher tests."""
    from dataclasses import replace

    from detector.config import DEFAULT_NETWORK_CONFIG

    return replace(DEFAULT_NETWORK_CONFIG, allow_private_targets=True)
