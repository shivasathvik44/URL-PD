"""Unit and integration tests for the detector FastAPI service and its security hardening."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from detector.decision_engine import Decision
from detector.explanations import Reason
from detector.lexical_features import extract_lexical_features
from detector.pipeline import AnalysisResult
from detector_service import ServiceConfig, create_app
from detector_service.config import ConfigError

TEST_TOKEN = "test-secret-token-1234567890abcdef"
DEFAULT_TEST_HEADERS = {
    "host": "127.0.0.1:8000",
    "x-detector-client": "firefox-extension",
    "authorization": f"Bearer {TEST_TOKEN}",
}


def _make_dummy_result(
    url: str = "https://example.com",
    classification: str = "safe",
    risk_score: float = 0.05,
    ml_probability: float | None = 0.03,
) -> AnalysisResult:
    """Helper to create a deterministic dummy AnalysisResult for testing."""
    reasons = (
        Reason(
            code="l1_clean",
            message="No suspicious patterns found in URL string",
            severity="info",
            layer="layer1",
            source="rule",
        ),
    )
    decision = Decision(
        classification=classification,
        risk_score=risk_score,
        ml_probability=ml_probability,
        llm_likelihood=None,
        decision_path="ml_low",
        layer4_ran=False,
        reasons=reasons,
    )
    l1 = extract_lexical_features(url)
    return AnalysisResult(
        url=url,
        analysis_id="test-analysis-1234",
        decision=decision,
        layer1=l1,
        layer2=None,
        page=None,
        llm=None,
        timings_ms={"layer1_ms": 0.5, "layer3_ms": 1.0, "total_ms": 1.5},
        metadata={"schema_version": 2, "test": True},
        errors=(),
    )


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_health_check_healthy():
    cfg = ServiceConfig(require_auth=False)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "phishing-url-detector"
        assert data["version"] == "0.5.0"
        assert "model_loaded" in data
        assert data["schema_version"] == 2


@pytest.mark.anyio
async def test_health_check_is_lightweight():
    """Ensure health check does not call predictor, fetcher, or llm."""
    mock_predictor = MagicMock()
    mock_fetcher = MagicMock()
    mock_llm = MagicMock()

    cfg = ServiceConfig(require_auth=False)
    app = create_app(
        service_config=cfg, predictor=mock_predictor, fetcher=mock_fetcher, llm_client=mock_llm
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        assert mock_predictor.predict_from_layers.call_count == 0
        assert mock_fetcher.call_count == 0
        assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------
# Host header validation (DNS Rebinding Defense)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
@pytest.mark.parametrize("valid_host", ["127.0.0.1:8000", "127.0.0.1", "localhost:8000", "[::1]:8000"])
async def test_host_header_validation_allows_loopback(valid_host):
    cfg = ServiceConfig(require_auth=False)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.get("/api/v1/health", headers={"host": valid_host})
        assert res.status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize("bad_host", ["attacker.com", "attacker.com:8000", "evil.org", "192.168.1.50:8000"])
async def test_host_header_validation_rejects_rebinding(bad_host):
    cfg = ServiceConfig(require_auth=False)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.get("/api/v1/health", headers={"host": bad_host})
        assert res.status_code == 400
        assert "invalid or untrusted host" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Client Header & Authentication Protection
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_missing_or_invalid_client_header_rejected():
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        # Missing client header
        res = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={"host": "127.0.0.1:8000", "authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert res.status_code == 403
        assert "client header" in res.json()["detail"].lower()

        # Invalid client header
        res2 = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={
                "host": "127.0.0.1:8000",
                "x-detector-client": "malicious-script",
                "authorization": f"Bearer {TEST_TOKEN}",
            },
        )
        assert res2.status_code == 403


@pytest.mark.anyio
async def test_unauthorized_token_rejected():
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        # Missing token
        res1 = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={"host": "127.0.0.1:8000", "x-detector-client": "firefox-extension"},
        )
        assert res1.status_code == 401
        assert "WWW-Authenticate" in res1.headers

        # Invalid token
        res2 = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={
                "host": "127.0.0.1:8000",
                "x-detector-client": "firefox-extension",
                "authorization": "Bearer wrong-token-xyz",
            },
        )
        assert res2.status_code == 401


@pytest.mark.anyio
@pytest.mark.parametrize(
    "auth_hdr",
    [
        {"authorization": f"Bearer {TEST_TOKEN}"},
        {"x-detector-token": TEST_TOKEN},
    ],
)
async def test_valid_authentication_accepted(auth_hdr):
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)
    headers = {"host": "127.0.0.1:8000", "x-detector-client": "firefox-extension", **auth_hdr}

    with patch("detector_service.app.analyze_url", return_value=_make_dummy_result()):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            res = await client.post(
                "/api/v1/analyze",
                json={"url": "https://example.com"},
                headers=headers,
            )
            assert res.status_code == 200


@pytest.mark.anyio
async def test_pre_slot_rejection_does_not_consume_scan_slot():
    """Verify unauthorized or malformed requests are rejected BEFORE acquiring concurrency slot."""
    cfg = ServiceConfig(max_concurrent_scans=1, scan_queue_timeout=0.05, require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        # 1. Unauthenticated request -> 401 immediately
        res_unauth = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={"host": "127.0.0.1:8000", "x-detector-client": "firefox-extension"},
        )
        assert res_unauth.status_code == 401

        # 2. Missing client header -> 403 immediately
        res_no_hdr = await client.post(
            "/api/v1/analyze",
            json={"url": "https://example.com"},
            headers={"host": "127.0.0.1:8000", "authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert res_no_hdr.status_code == 403

        # 3. Malformed URL -> 422 immediately
        res_bad_url = await client.post(
            "/api/v1/analyze",
            json={"url": "javascript:alert(1)"},
            headers=DEFAULT_TEST_HEADERS,
        )
        assert res_bad_url.status_code == 422

        # 4. Now verify that a valid authenticated request succeeds and finds the slot completely free
        with patch("detector_service.app.analyze_url", return_value=_make_dummy_result()):
            res_valid = await client.post(
                "/api/v1/analyze",
                json={"url": "https://example.com"},
                headers=DEFAULT_TEST_HEADERS,
            )
            assert res_valid.status_code == 200


# ---------------------------------------------------------------------------
# Request validation tests
# ---------------------------------------------------------------------------
@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_payload,expected_err_part",
    [
        ({}, "Field required"),
        ({"url": ""}, "URL cannot be empty"),
        ({"url": "   "}, "URL cannot be empty"),
        ({"url": "https://" + "a" * 2050 + ".com"}, "2048"),
        ({"url": "javascript:alert(1)"}, "Unsupported URL scheme 'javascript'"),
        ({"url": "file:///etc/passwd"}, "Unsupported URL scheme 'file'"),
        ({"url": "data:text/html,test"}, "Unsupported URL scheme 'data'"),
        ({"url": "ftp://files.example.com"}, "Unsupported URL scheme 'ftp'"),
        ({"url": "https://"}, "URL must contain a valid host"),
        ({"url": "http://user:pass@example.com/login"}, "Embedded credentials"),
        ({"url": "http://admin@example.com/dashboard"}, "Embedded credentials"),
        ({"url": "http://example.com:99999/test"}, "range"),
    ],
)
async def test_analyze_request_validation_rejections(invalid_payload, expected_err_part):
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.post("/api/v1/analyze", json=invalid_payload, headers=DEFAULT_TEST_HEADERS)
        assert res.status_code == 422
        body = res.json()
        assert expected_err_part.lower() in body["detail"].lower()


@pytest.mark.anyio
async def test_analyze_accepts_private_ips():
    """URL classification and outbound fetching are separate; API should accept private IP URLs."""
    dummy_result = _make_dummy_result("http://192.168.1.1/login")
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)

    with patch("detector_service.app.analyze_url", return_value=dummy_result) as mock_analyze:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            res = await client.post(
                "/api/v1/analyze",
                json={"url": "http://192.168.1.1/login"},
                headers=DEFAULT_TEST_HEADERS,
            )
            assert res.status_code == 200
            assert mock_analyze.call_count == 1
            call_url = mock_analyze.call_args[0][0]
            assert call_url == "http://192.168.1.1/login"


# ---------------------------------------------------------------------------
# Successful analysis serialization
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_analyze_successful_serialization():
    dummy = _make_dummy_result(
        url="https://secure-login.example.com/auth",
        classification="phishing",
        risk_score=0.92,
        ml_probability=0.915,
    )
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)

    with patch("detector_service.app.analyze_url", return_value=dummy):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            res = await client.post(
                "/api/v1/analyze",
                json={
                    "url": "https://secure-login.example.com/auth",
                    "run_layer2": True,
                    "run_layer4": False,
                },
                headers=DEFAULT_TEST_HEADERS,
            )
            assert res.status_code == 200
            data = res.json()
            assert data["url"] == "https://secure-login.example.com/auth"
            assert data["analysis_id"] == "test-analysis-1234"
            assert data["classification"] == "phishing"
            assert data["risk_score"] == 0.92
            assert data["ml_probability"] == 0.915
            assert data["decision_path"] == "ml_low"
            assert data["layer4_ran"] is False
            assert len(data["reasons"]) == 1
            assert data["reasons"][0]["code"] == "l1_clean"
            assert data["timings_ms"]["layer1_ms"] == 0.5
            assert data["metadata"]["schema_version"] == 2
            assert data["errors"] == []


# ---------------------------------------------------------------------------
# Concurrency limiting and queue saturation
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_analyze_concurrency_saturation():
    """Verify that when concurrency capacity is saturated, subsequent requests receive 503."""
    config = ServiceConfig(
        max_concurrent_scans=1,
        scan_queue_timeout=0.05,
        require_auth=True,
        auth_token=TEST_TOKEN,
    )
    app = create_app(service_config=config)

    in_call_event = threading.Event()
    release_event = threading.Event()

    def slow_analyze(*args, **kwargs):
        in_call_event.set()
        release_event.wait(timeout=5.0)
        return _make_dummy_result()

    with patch("detector_service.app.analyze_url", side_effect=slow_analyze):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            task1 = asyncio.create_task(
                client.post(
                    "/api/v1/analyze",
                    json={"url": "https://example.com/first"},
                    headers=DEFAULT_TEST_HEADERS,
                )
            )

            # Wait until task1 worker has entered analysis
            await asyncio.to_thread(in_call_event.wait, 2.0)

            # Second request should fail with 503 because capacity is 1 and timeout is 0.05s
            res2 = await client.post(
                "/api/v1/analyze",
                json={"url": "https://example.com/second"},
                headers=DEFAULT_TEST_HEADERS,
            )
            assert res2.status_code == 503
            assert "saturated" in res2.json()["detail"].lower()

            # Now release task1
            release_event.set()
            res1 = await task1
            assert res1.status_code == 200


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_analyze_internal_exception_handling():
    """Ensure unexpected exceptions return HTTP 500 without leaking stack trace."""
    cfg = ServiceConfig(require_auth=True, auth_token=TEST_TOKEN)
    app = create_app(service_config=cfg)

    with patch("detector_service.app.analyze_url", side_effect=RuntimeError("internal crash")):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as client:
            res = await client.post(
                "/api/v1/analyze",
                json={"url": "https://example.com/crash"},
                headers=DEFAULT_TEST_HEADERS,
            )
            assert res.status_code == 500
            data = res.json()
            assert "Analysis failed: RuntimeError" in data["detail"]


# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_cors_preflight_for_extension():
    cfg = ServiceConfig(require_auth=False)
    app = create_app(service_config=cfg)
    ext_origin = "moz-extension://12345678-abcd-ef01-2345-6789abcdef01"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.options(
            "/api/v1/analyze",
            headers={
                "host": "127.0.0.1:8000",
                "Origin": ext_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-detector-client,authorization",
            },
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == ext_origin


@pytest.mark.anyio
async def test_cors_rejects_untrusted_origin():
    cfg = ServiceConfig(require_auth=False)
    app = create_app(service_config=cfg)
    bad_origin = "http://evil-attacker.example.com"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as client:
        res = await client.options(
            "/api/v1/analyze",
            headers={
                "host": "127.0.0.1:8000",
                "Origin": bad_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------------
# Server and Config tests
# ---------------------------------------------------------------------------
def test_insecure_host_binding_raises_config_error():
    """Binding to a non-loopback host must fail unless explicitly permitted."""
    with pytest.raises(ConfigError) as exc:
        ServiceConfig(host="0.0.0.0", allow_external_binding=False)
    assert "Insecure host binding" in str(exc.value)

    with pytest.raises(ConfigError):
        ServiceConfig(host="192.168.1.100", allow_external_binding=False)

    # Allowed when explicitly overridden
    allowed_cfg = ServiceConfig(host="0.0.0.0", allow_external_binding=True)
    assert allowed_cfg.host == "0.0.0.0"


def test_service_config_from_env(monkeypatch):
    from detector_service.config import service_config_from_env

    monkeypatch.setenv("DETECTOR_SERVICE_HOST", "127.0.0.2")
    monkeypatch.setenv("DETECTOR_SERVICE_PORT", "9090")
    monkeypatch.setenv("DETECTOR_MAX_CONCURRENT_SCANS", "8")
    monkeypatch.setenv("DETECTOR_SCAN_QUEUE_TIMEOUT", "15.0")
    monkeypatch.setenv("DETECTOR_AUTH_TOKEN", "env-secret-token")

    cfg = service_config_from_env()
    assert cfg.host == "127.0.0.2"
    assert cfg.port == 9090
    assert cfg.max_concurrent_scans == 8
    assert cfg.scan_queue_timeout == 15.0
    assert cfg.auth_token == "env-secret-token"


def test_server_main_refuses_external_binding_without_flag():
    from detector_service.server import main

    code = main(["--host", "0.0.0.0"])
    assert code == 1


def test_server_main_invokes_uvicorn():
    from detector_service.server import main

    with patch("uvicorn.run") as mock_run:
        code = main(["--host", "127.0.0.1", "--port", "8888"])
        assert code == 0
        assert mock_run.call_count == 1
        kwargs = mock_run.call_args[1]
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8888
