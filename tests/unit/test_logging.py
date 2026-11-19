"""Tests for the JSON log formatter."""

import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_outputs_parseable_json_with_extra_fields() -> None:
    record = logging.makeLogRecord(
        {"name": "test", "levelname": "INFO", "msg": "user asked", "role": "finance"}
    )

    line = JsonFormatter().format(record)
    data = json.loads(line)

    assert data["msg"] == "user asked"
    assert data["level"] == "INFO"
    assert data["role"] == "finance"  # extra fields become JSON keys


def test_readable_formatter_appends_extra_fields() -> None:
    from app.core.logging import ReadableFormatter

    record = logging.makeLogRecord(
        {"name": "security", "levelname": "INFO", "msg": "Login", "user": "kabir_fin"}
    )

    assert ReadableFormatter().format(record).endswith("| Login | user=kabir_fin")
