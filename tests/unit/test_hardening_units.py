"""Unit tests for the rate limiter, audit log and request context."""

import logging
from pathlib import Path

import pytest

from app.core.audit import MAX_QUESTION_CHARS, AuditLog
from app.core.rate_limit import RateLimiter
from app.core.request_context import (
    RequestIdFilter,
    new_request_id,
    reset_request_id,
    set_request_id,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_allows_up_to_limit_then_reports_wait() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=3, window_seconds=60, clock=clock)

    assert [limiter.hit("u") for _ in range(3)] == [None, None, None]
    clock.now += 10
    assert limiter.hit("u") == pytest.approx(50)  # oldest hit leaves the window in 50 s
    assert limiter.hit("other") is None  # keys are independent


def test_rate_limiter_window_slides_and_rejections_are_not_counted() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    limiter.hit("u")
    clock.now += 30
    limiter.hit("u")
    for _ in range(5):  # rejected attempts must not extend the wait
        assert limiter.hit("u") is not None
    clock.now += 30.5  # the first hit has left the window
    assert limiter.hit("u") is None
    assert limiter.hit("u") is not None


def test_rate_limiter_zero_disables() -> None:
    limiter = RateLimiter(limit=0)
    assert all(limiter.hit("u") is None for _ in range(1000))


def test_rate_limiter_prunes_idle_keys() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=1, clock=clock)
    for i in range(10_001):
        limiter.hit(f"k{i}")
    clock.now += 5
    limiter.hit("fresh")  # over the key cap: idle keys are dropped
    assert len(limiter._hits) == 1


def test_audit_log_writes_json_lines_with_request_id(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "nested" / "audit.log")
    token = set_request_id("req-123")
    try:
        audit.record("login", user="alice", role="employee")
    finally:
        reset_request_id(token)
    audit.record("chat", user="bob", question="x" * (MAX_QUESTION_CHARS + 50))

    first, second = audit.read()
    assert first["event"] == "login" and first["request_id"] == "req-123"
    assert first["user"] == "alice" and first["ts"].endswith("+00:00")
    assert len(second["question"]) == MAX_QUESTION_CHARS
    assert second["request_id"] is None


def test_audit_log_can_redact_questions(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.log", include_questions=False)
    audit.record("chat", user="bob", question="What is Isha's salary?")
    assert audit.read()[0]["question"] == "[redacted]"


def test_disabled_audit_log_writes_nothing() -> None:
    audit = AuditLog(None)
    audit.record("login", user="alice")
    assert audit.read() == []


def test_audit_write_failure_does_not_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audit = AuditLog(tmp_path / "audit.log")
    audit.path = tmp_path  # a directory: opening it for append fails
    audit.record("login", user="alice")
    assert "Could not write audit event" in caplog.text


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [("abc-123", True), ("ui.4f2e_9", True), (None, False), ("", False), ("bad id!", False)],
)
def test_new_request_id_only_reuses_safe_values(incoming: str | None, kept: bool) -> None:
    result = new_request_id(incoming)
    assert (result == incoming) is kept
    assert 1 <= len(result) <= 64


def test_new_request_id_rejects_overlong_values() -> None:
    assert new_request_id("a" * 65) != "a" * 65


def test_request_id_filter_adds_id_only_during_a_request() -> None:
    record = logging.makeLogRecord({"msg": "hi"})
    RequestIdFilter().filter(record)
    assert not hasattr(record, "request_id")

    token = set_request_id("req-9")
    try:
        record = logging.makeLogRecord({"msg": "hi"})
        assert RequestIdFilter().filter(record)
        assert record.request_id == "req-9"
    finally:
        reset_request_id(token)
