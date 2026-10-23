"""Pydantic schemas for the detector FastAPI service."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """URL analysis request payload."""

    url: str = Field(
        ...,
        description="The URL to analyze (max 2048 characters).",
        max_length=2048,
    )
    run_layer2: bool = Field(
        default=True,
        description="Whether to perform DNS and domain intelligence lookups.",
    )
    run_layer4: bool = Field(
        default=True,
        description="Whether to allow Layer 4 content/LLM analysis if needed.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("URL cannot be empty")
        if len(s) > 2048:
            raise ValueError("URL exceeds maximum length of 2048 characters")
        try:
            parts = urlsplit(s)
        except Exception as exc:
            raise ValueError(f"Malformed URL: {exc}") from exc

        scheme = (parts.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported URL scheme '{scheme or '(none)'}'. Only 'http' and 'https' are supported."
            )

        # Reject embedded credentials in the URL authority
        if parts.username or parts.password:
            raise ValueError("Embedded credentials in URL authority are not allowed")

        try:
            hostname = parts.hostname
        except ValueError as exc:
            raise ValueError(f"Invalid URL host: {exc}") from exc

        if not hostname:
            raise ValueError("URL must contain a valid host")

        if " " in hostname or (hostname.startswith("[") and not hostname.endswith("]")):
            raise ValueError("Malformed URL hostname")

        # Validate port if explicitly present
        try:
            port = parts.port
            if port is not None and not (1 <= port <= 65535):
                raise ValueError(f"Port number {port} out of valid range (1-65535)")
        except ValueError as exc:
            raise ValueError(f"Invalid URL port or authority: {exc}") from exc

        return s


class ReasonResponse(BaseModel):
    """A single piece of evidence contributing to the classification."""

    code: str
    message: str
    severity: str
    layer: str
    source: str


class AnalyzeResponse(BaseModel):
    """Structured verdict and evidence for an analyzed URL."""

    url: str
    analysis_id: str
    classification: str
    risk_score: float
    risk_score_note: str = "ordinal severity, not a calibrated probability"
    ml_probability: float | None = None
    llm_likelihood: float | None = None
    decision_path: str
    layer4_ran: bool
    reasons: list[ReasonResponse] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    layer1_lexical: dict[str, Any] | None = None
    layer2_domain_intel: dict[str, Any] | None = None
    layer4_page: dict[str, Any] | None = None
    layer4_llm: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check status information."""

    status: str
    service: str = "phishing-url-detector"
    version: str
    model_loaded: bool
    schema_version: int
