"""FastAPI application factory and endpoints for phishing-url-detector."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import (
    DEFAULT_CONFIG,
    DEFAULT_DECISION_CONFIG,
    DEFAULT_ML_CONFIG,
    DecisionConfig,
    DetectorConfig,
    MLConfig,
    analyze_url,
)
from detector import __version__ as DETECTOR_VERSION
from detector.feature_schema import SCHEMA_VERSION
from detector.observability import get_logger
from detector.security import redact_url
from detector.web_fetcher import fetch_page

from .config import (
    DEFAULT_SERVICE_CONFIG,
    ServiceConfig,
    resolve_auth_token,
    validate_host_header,
)
from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

_log = get_logger("service")


def create_app(
    service_config: ServiceConfig = DEFAULT_SERVICE_CONFIG,
    detector_config: DetectorConfig = DEFAULT_CONFIG,
    ml_config: MLConfig = DEFAULT_ML_CONFIG,
    decision_config: DecisionConfig = DEFAULT_DECISION_CONFIG,
    *,
    predictor: Any = None,
    llm_client: Any = None,
    fetcher: Any = fetch_page,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    active_auth_token = resolve_auth_token(service_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.semaphore = asyncio.Semaphore(service_config.max_concurrent_scans)
        app.state.service_config = service_config
        app.state.detector_config = detector_config
        app.state.ml_config = ml_config
        app.state.decision_config = decision_config
        app.state.predictor = predictor
        app.state.llm_client = llm_client
        app.state.fetcher = fetcher
        app.state.active_auth_token = active_auth_token
        _log.info(
            "service started",
            extra={"fields": {
                "host": service_config.host,
                "port": service_config.port,
                "max_concurrent_scans": service_config.max_concurrent_scans,
                "require_auth": service_config.require_auth,
            }},
        )
        yield
        _log.info("service stopped")

    app = FastAPI(
        title="Phishing URL Detector Service",
        description="Local HTTP API exposing the multi-layer phishing URL detector.",
        version=DETECTOR_VERSION,
        lifespan=lifespan,
    )

    # Initialize app state immediately so it is available even before lifespan or in test runners
    app.state.semaphore = asyncio.Semaphore(service_config.max_concurrent_scans)
    app.state.service_config = service_config
    app.state.detector_config = detector_config
    app.state.ml_config = ml_config
    app.state.decision_config = decision_config
    app.state.predictor = predictor
    app.state.llm_client = llm_client
    app.state.fetcher = fetcher
    app.state.active_auth_token = active_auth_token

    # 1. Host header validation middleware (defeats DNS rebinding)
    @app.middleware("http")
    async def validate_host_middleware(request: Request, call_next):
        host = request.headers.get("host")
        if not validate_host_header(host, service_config.allowed_hosts):
            _log.warning(
                "rejected request with untrusted host header",
                extra={"fields": {"host": host or ""}},
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Bad Request: invalid or untrusted Host header"},
            )
        return await call_next(request)

    # 2. Configure CORS - restricted to local callers and Firefox extension origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(service_config.allow_origins),
        allow_origin_regex=service_config.allow_origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            msg = err.get("msg", "Validation error")
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            details.append(f"{loc}: {msg}" if loc else msg)
        return JSONResponse(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            content={"detail": "; ".join(details)},
        )

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        summary="Lightweight service health check",
    )
    async def health_check() -> HealthResponse:
        """Lightweight health check without triggering DNS lookups or inference."""
        model_file = Path(ml_config.models_dir) / ml_config.model_filename
        is_loaded = predictor is not None or model_file.is_file()
        return HealthResponse(
            status="ok",
            service="phishing-url-detector",
            version=DETECTOR_VERSION,
            model_loaded=is_loaded,
            schema_version=SCHEMA_VERSION,
        )

    @app.post(
        "/api/v1/analyze",
        response_model=AnalyzeResponse,
        summary="Analyze a URL for phishing indicators",
    )
    async def analyze_endpoint(request: AnalyzeRequest, raw_request: Request) -> AnalyzeResponse:
        """Analyze a URL through the four-layer detector pipeline."""
        # -------------------------------------------------------------------
        # Pre-slot Security Checks (run BEFORE acquiring concurrency slot)
        # -------------------------------------------------------------------
        # Check custom client header (cross-origin defense-in-depth)
        client_hdr_name = service_config.client_header_name.lower()
        client_hdr_val = raw_request.headers.get(client_hdr_name)
        if (
            client_hdr_val is None
            or client_hdr_val.strip().lower() != service_config.client_header_value.lower()
        ):
            _log.warning("rejected request: missing or invalid client header")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: missing or invalid client header",
            )

        # Check authentication token if authentication is enabled
        if service_config.require_auth:
            provided_token = None
            auth_header = raw_request.headers.get("authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                provided_token = auth_header[7:].strip()
            elif "x-detector-token" in raw_request.headers:
                provided_token = raw_request.headers.get("x-detector-token", "").strip()

            expected_token = app.state.active_auth_token or ""
            if not provided_token or not hmac.compare_digest(provided_token, expected_token):
                _log.warning("rejected request: unauthorized or invalid token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: invalid or missing authentication token",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )

        # -------------------------------------------------------------------
        # Acquire concurrency slot (only authenticated, valid requests reach here)
        # -------------------------------------------------------------------
        semaphore: asyncio.Semaphore = app.state.semaphore

        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=service_config.scan_queue_timeout,
            )
        except asyncio.TimeoutError:
            _log.warning(
                "scan rejected: service saturated",
                extra={"fields": {"url": redact_url(request.url)}},
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is saturated. Too many concurrent scans in progress.",
            ) from None

        try:
            # Build decision configuration, disabling Layer 4 if requested
            d_cfg = app.state.decision_config
            if not request.run_layer4 and d_cfg.enable_layer4:
                d_cfg = replace(d_cfg, enable_layer4=False)

            _log.info(
                "dispatching analysis to worker thread",
                extra={"fields": {"url": redact_url(request.url)}},
            )

            # Offload blocking synchronous analysis to a worker thread
            result = await asyncio.to_thread(
                analyze_url,
                request.url,
                config=app.state.detector_config,
                ml_config=app.state.ml_config,
                decision_config=d_cfg,
                run_layer2=request.run_layer2,
                predictor=app.state.predictor,
                llm_client=app.state.llm_client,
                fetcher=app.state.fetcher,
            )

            return AnalyzeResponse(**result.as_dict())
        except Exception as exc:
            _log.error(
                "analysis failed with unexpected exception",
                exc_info=exc,
                extra={"fields": {"url": redact_url(request.url)}},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {type(exc).__name__}",
            ) from exc
        finally:
            semaphore.release()

    return app
