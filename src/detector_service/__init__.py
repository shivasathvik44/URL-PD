"""FastAPI service package for phishing-url-detector."""

from .app import create_app
from .config import DEFAULT_SERVICE_CONFIG, ServiceConfig

__all__ = ["create_app", "ServiceConfig", "DEFAULT_SERVICE_CONFIG"]
