"""Logging setup.

Development: readable one-line logs.
Production (``LOG_JSON=true``): one JSON object per line, which log tools
(Datadog, ELK, CloudWatch) can parse and search by field.
"""

import json
import logging
import sys
from datetime import UTC, datetime

from app.core.request_context import RequestIdFilter

# Attributes every LogRecord has, plus ones Formatter.format() adds while formatting.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via logger.info("...", extra={...}) becomes a field.
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ReadableFormatter(logging.Formatter):
    """Human-friendly one-line logs that also show ``extra=`` fields as key=value."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {
            k: v
            for k, v in vars(record).items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        if extras:
            line += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        return line


def ensure_utf8_console() -> None:
    """Let command-line tools print any character (e.g. the rupee sign).

    The Windows console often defaults to an old code page (cp1252) and
    crashes on characters outside it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger once at startup."""
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ReadableFormatter())
    handler.addFilter(RequestIdFilter())  # every line logged during a request gets its ID
    root = logging.getLogger()
    root.handlers.clear()  # avoid duplicate logs on reload
    root.addHandler(handler)
    root.setLevel(level)
