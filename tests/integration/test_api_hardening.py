"""API tests for Phase 8 hardening: request IDs, errors, readiness, limits, audit."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.audit import AuditLog
from app.core.config import Settings
from app.main import create_app
from tests.conftest import ROLE_USERS, TEST_PASSWORD, SessionServices, auth_header
from tests.fakes import FakeLLM


@pytest.fixture
def make_client(
    test_settings: Settings, services: SessionServices, tmp_path: Path
) -> Iterator[object]:
    """Build a client with overridden settings and its own audit file."""
    clients: list[TestClient] = []

    def factory(**overrides: object) -> TestClient:
        settings = test_settings.model_copy(
            update={"audit_log_path": tmp_path / "audit.log", **overrides}
        )
        app = create_app(settings, services_factory=lambda _: services)  # type: ignore[arg-type,return-value]
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


def audit_events(client: TestClient) -> list[dict[str, object]]:
    audit: AuditLog = client.app.state.audit  # type: ignore[attr-defined]
    return audit.read()


# ---------------------------------------------------------------- middleware


def test_every_response_has_request_id_and_security_headers(client: TestClient) -> None:
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) == 32
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_safe_incoming_request_id_is_reused(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "ui-42"}).headers["x-request-id"] == (
        "ui-42"
    )
    unsafe = client.get("/health", headers={"X-Request-ID": "x\r\nSet-Cookie: a=b"})
    assert unsafe.headers["x-request-id"] != "x\r\nSet-Cookie: a=b"


def test_unexpected_error_returns_safe_500_with_request_id(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    fake_llm.error = RuntimeError("db password=hunter2 at /srv/secret.py line 3")
    response = client.post(
        "/chat", json={"question": "revenue?"}, headers=auth_header(role_tokens["finance"])
    )

    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error", "request_id": body["request_id"]}
    assert body["request_id"] == response.headers["x-request-id"]
    assert "hunter2" not in response.text


def test_unexpected_error_during_stream_becomes_error_event(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    fake_llm.error = RuntimeError("internal detail")
    response = client.post(
        "/chat/stream", json={"question": "revenue?"}, headers=auth_header(role_tokens["finance"])
    )
    assert "event: error" in response.text
    assert "Internal server error" in response.text and "internal detail" not in response.text


# ---------------------------------------------------------------- readiness


def test_readiness_reports_ok_checks(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"vector_index", "user_store"}
    assert all(v.startswith("ok") for v in body["checks"].values())


def test_readiness_fails_without_users_or_unreachable_store(
    client: TestClient, services: SessionServices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services.users, "count", lambda: 0)

    def broken() -> int:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(services.store, "count", broken)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {
        "vector_index": "unreachable",
        "user_store": "no users: run seeding",
    }


# ---------------------------------------------------------------- rate limits


def test_chat_rate_limit_is_per_user_and_audited(
    make_client: object, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    client = make_client(rate_limit_chat_per_minute=2)  # type: ignore[operator]
    finance = auth_header(role_tokens["finance"])
    for _ in range(2):
        assert client.post("/chat", json={"question": "q"}, headers=finance).status_code == 200

    blocked = client.post("/chat/stream", json={"question": "q"}, headers=finance)
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert "Too many requests" in blocked.json()["detail"]
    # Another user has their own budget; search shares the chat budget.
    hr = auth_header(role_tokens["hr"])
    assert client.post("/chat", json={"question": "q"}, headers=hr).status_code == 200
    assert client.post("/search", json={"query": "q"}, headers=finance).status_code == 429

    limited = [e for e in audit_events(client) if e["event"] == "rate_limited"]
    assert {e["user"] for e in limited} == {ROLE_USERS["finance"]}


def test_login_rate_limit_applies_per_username(make_client: object) -> None:
    client = make_client(rate_limit_login_per_minute=2)  # type: ignore[operator]
    user = ROLE_USERS["finance"]
    for password in ("wrong-1", "wrong-2"):
        response = client.post("/auth/login", data={"username": user, "password": password})
        assert response.status_code == 401
    # Even the right password is refused until the window passes.
    response = client.post("/auth/login", data={"username": user, "password": TEST_PASSWORD})
    assert response.status_code == 429 and "retry-after" in response.headers
    # Other accounts are unaffected (e.g. everyone logging in through the UI).
    other = {"username": ROLE_USERS["hr"], "password": TEST_PASSWORD}
    assert client.post("/auth/login", data=other).status_code == 200

    events = [(e["event"], e["user"]) for e in audit_events(client)]
    assert events == [
        ("login_failed", user),
        ("login_failed", user),
        ("rate_limited", user),
        ("login", ROLE_USERS["hr"]),
    ]


# ---------------------------------------------------------------- audit


def test_chat_is_audited_with_documents_and_departments(
    make_client: object, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    client = make_client()  # type: ignore[operator]
    response = client.post(
        "/chat",
        json={"question": "What is the leave policy?"},
        headers={**auth_header(role_tokens["employee"]), "X-Request-ID": "audit-test-1"},
    )
    assert response.status_code == 200

    (event,) = [e for e in audit_events(client) if e["event"] == "chat"]
    assert event["request_id"] == "audit-test-1"
    assert event["user"] == ROLE_USERS["employee"] and event["role"] == "employee"
    assert event["question"] == "What is the leave policy?"
    assert event["outcome"] == "answered" and event["route"] == "docs"
    assert event["departments"] == ["general"]  # an employee only ever sees general docs
    assert event["documents"] and event["cited"] == [1, 2]


def test_stream_and_llm_errors_are_audited(
    make_client: object, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    from app.llm.base import LLMError

    client = make_client()  # type: ignore[operator]
    headers = auth_header(role_tokens["marketing"])
    client.post("/chat/stream", json={"question": "best campaign?"}, headers=headers)
    fake_llm.error = LLMError("LLM rate limit reached", retryable=True)
    assert client.post("/chat", json={"question": "again?"}, headers=headers).status_code == 503

    outcomes = [(e["question"], e["outcome"]) for e in audit_events(client)]
    assert outcomes == [("best campaign?", "answered"), ("again?", "llm_error")]


def test_search_is_audited(make_client: object, role_tokens: dict[str, str]) -> None:
    client = make_client()  # type: ignore[operator]
    response = client.post(
        "/search", json={"query": "revenue"}, headers=auth_header(role_tokens["finance"])
    )
    assert response.status_code == 200
    (event,) = audit_events(client)
    assert event["event"] == "search" and event["question"] == "revenue"
    assert set(event["departments"]) <= {"finance", "marketing", "general"}  # type: ignore[arg-type]


def test_audit_file_is_json_lines(make_client: object, role_tokens: dict[str, str]) -> None:
    client = make_client()  # type: ignore[operator]
    client.post("/auth/login", data={"username": "nobody", "password": "x"})
    path = client.app.state.audit.path  # type: ignore[attr-defined]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["login_failed"]
