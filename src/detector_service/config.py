"""Configuration settings for the detector FastAPI service."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from detector.config import PROJECT_ROOT


class ConfigError(ValueError):
    """Raised for invalid or insecure service configuration."""


def is_loopback_host(host: str) -> bool:
    """Check if a host string refers to a local loopback interface."""
    h = (host or "").strip().lower()
    return h in ("127.0.0.1", "localhost", "::1", "test", "testserver") or h.startswith("127.")



def validate_host_header(host_header: str | None, allowed_hosts: tuple[str, ...]) -> bool:
    """Validate that the Host header matches an allowed host (port-insensitive)."""
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):
        closing = host.find("]")
        if closing != -1:
            hostname = host[1:closing]
        else:
            return False
    else:
        hostname = host.split(":")[0]

    return hostname in {h.lower() for h in allowed_hosts}


@dataclass(frozen=True)
class ServiceConfig:
    """Service runtime configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    allow_external_binding: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "::1", "test", "testserver")
    max_concurrent_scans: int = 4
    scan_queue_timeout: float = 30.0

    # Custom header required on protected requests to defeat simple cross-origin requests
    client_header_name: str = "x-detector-client"
    client_header_value: str = "firefox-extension"

    # Token authentication
    require_auth: bool = True
    auth_token: str | None = None
    token_file_path: str = str(PROJECT_ROOT / "data" / "processed" / "service_token")

    # Strict CORS default: allow local callers and Firefox extension origins
    allow_origins: tuple[str, ...] = (
        "http://127.0.0.1",
        "http://localhost",
    )
    allow_origin_regex: str | None = r"^moz-extension://[a-zA-Z0-9\-]+$"

    def __post_init__(self) -> None:
        if not self.allow_external_binding and not is_loopback_host(self.host):
            raise ConfigError(
                f"Insecure host binding: refusing to bind to '{self.host}'. "
                "Service must bind to a loopback address unless allow_external_binding=True."
            )


def resolve_auth_token(config: ServiceConfig) -> str | None:
    """Resolve or generate the active authentication token for the service.

    1. If config.auth_token is explicitly set, use it.
    2. If config.require_auth is False, return None.
    3. If token_file_path exists and is non-empty, load the token.
    4. Otherwise, generate a 32-byte cryptographically secure hex token,
       persist it with 0600 file permissions, and return it.
    """
    if config.auth_token:
        return config.auth_token.strip()

    if not config.require_auth:
        return None

    path = Path(config.token_file_path)
    if path.is_file():
        try:
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass

    # Generate new cryptographically secure token
    token = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600  # Owner read/write only
        fd = os.open(path, flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
    except OSError:
        # Fall back to in-memory token if writing to disk is blocked
        pass

    return token


def service_config_from_env() -> ServiceConfig:
    """Construct ServiceConfig from environment variables."""
    host = os.environ.get("DETECTOR_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("DETECTOR_SERVICE_PORT", "8000"))
    allow_external = os.environ.get("DETECTOR_ALLOW_EXTERNAL_BINDING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    max_scans = int(os.environ.get("DETECTOR_MAX_CONCURRENT_SCANS", "4"))
    queue_timeout = float(os.environ.get("DETECTOR_SCAN_QUEUE_TIMEOUT", "30.0"))
    require_auth = os.environ.get("DETECTOR_REQUIRE_AUTH", "1").lower() not in ("0", "false", "no")
    auth_token = os.environ.get("DETECTOR_AUTH_TOKEN")
    token_file_path = os.environ.get(
        "DETECTOR_TOKEN_FILE_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "service_token"),
    )

    return ServiceConfig(
        host=host,
        port=port,
        allow_external_binding=allow_external,
        max_concurrent_scans=max_scans,
        scan_queue_timeout=queue_timeout,
        require_auth=require_auth,
        auth_token=auth_token,
        token_file_path=token_file_path,
    )


DEFAULT_SERVICE_CONFIG = ServiceConfig()
