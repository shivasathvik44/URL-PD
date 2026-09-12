"""Structured logging and per-layer timing.

The detector is a library, so it never configures the root logger or prints to
stderr on import - it only emits records on the `detector.*` hierarchy and
leaves handling to the application (`configure_logging()` is there for the CLI).

Two rules, enforced here rather than left to each call site:

  * Every log record goes through a redaction filter, so a URL's credential-ish
    query parameters, API keys and form values cannot reach a log file even if
    a future call site forgets.
  * Every analysis carries an `analysis_id`, so the Layer 1/2/3/4 records for
    one URL can be correlated.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from .security import redact

LOGGER_NAME = "detector"

#: Keys whose values are never logged, whatever a call site passes.
_FORBIDDEN_KEYS = frozenset(
    {
        "password", "passwd", "pwd", "otp", "cookie", "cookies", "authorization",
        "auth", "api_key", "apikey", "token", "secret", "credentials",
        "form_values", "input_values", "session",
    }
)


class RedactionFilter(logging.Filter):
    """Scrubs secrets from the message and from structured `extra` fields."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            record.fields = {
                k: ("[REDACTED]" if k.lower() in _FORBIDDEN_KEYS else _scrub(v))
                for k, v in fields.items()
            }
        return True


def _scrub(value):
    if isinstance(value, str):
        return redact(value, 500)
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if k.lower() in _FORBIDDEN_KEYS else _scrub(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value][:20]
    return value


class JsonFormatter(logging.Formatter):
    """One JSON object per line - greppable and machine-readable."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "") -> logging.Logger:
    """A `detector.*` logger with the redaction filter attached."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
    if not any(isinstance(f, RedactionFilter) for f in logger.filters):
        logger.addFilter(RedactionFilter())
    return logger


def configure_logging(level: str = "WARNING", *, json_format: bool = False) -> None:
    """Attach a handler to the `detector` logger. For applications/CLIs only."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_format
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactionFilter())
    logger.addHandler(handler)
    logger.propagate = False


def new_analysis_id() -> str:
    """Short correlation id for one URL analysis."""
    return uuid.uuid4().hex[:12]


@dataclass
class Timings:
    """Wall-clock milliseconds per layer, plus the total."""

    layers: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @contextmanager
    def measure(self, layer: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.layers[layer] = round(self.layers.get(layer, 0.0) + elapsed, 2)

    def total_ms(self) -> float:
        return round(sum(self.layers.values()), 2)

    def as_dict(self) -> dict:
        return {**self.layers, "total_ms": self.total_ms()}
