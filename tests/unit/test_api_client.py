"""Tests for the UI's HTTP client, using httpx's mock transport (no server)."""

import json

import httpx
import pytest

from ui.api_client import ApiError, AuthError, Done, FinSolveClient, Token, iter_sse

ME = {
    "username": "kabir_fin",
    "full_name": "Kabir Shah",
    "role": "finance",
    "role_display_name": "Finance",
    "role_description": "Financial reports",
    "accessible_departments": [{"name": "finance", "description": "Money"}],
    "suggested_questions": ["Q1 revenue?"],
}
RESULT = {
    "answer": "Revenue was $2.4B [1].",
    "sources": [
        {
            "number": 1,
            "document": "finance/q.md",
            "title": "Quarterly",
            "section": "Q3",
            "department": "finance",
            "quarter": "Q3",
            "score": 0.8,
            "excerpt": "Revenue: $2.4B",
            "cited": True,
        }
    ],
    "model": "m",
    "latency_ms": 900,
    "role": "finance",
}


def sse(*events: tuple[str, dict[str, object]]) -> str:
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)


def make_client(handler: httpx.MockTransport, token: str | None = "tok") -> FinSolveClient:
    return FinSolveClient("http://api.test", token=token, transport=handler)


def test_login_stores_token_and_me_parses_profile() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            assert b"username=kabir_fin" in request.content
            return httpx.Response(200, json={"access_token": "abc"})
        assert request.headers["Authorization"] == "Bearer abc"
        return httpx.Response(200, json=ME)

    client = make_client(httpx.MockTransport(handler), token=None)
    assert client.login("kabir_fin", "pw") == "abc"
    profile = client.me()
    assert profile.role == "finance"
    assert profile.departments == [("finance", "Money")]
    assert profile.suggested_questions == ["Q1 revenue?"]


def test_wrong_password_raises_auth_error() -> None:
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(401, json={})), None)
    with pytest.raises(AuthError, match="Incorrect"):
        client.login("x", "y")


def test_expired_session_raises_auth_error() -> None:
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(401, json={})))
    with pytest.raises(AuthError, match="expired"):
        client.me()


def test_api_down_gives_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="Cannot reach"):
        make_client(httpx.MockTransport(handler)).me()


def test_503_passes_through_backend_message() -> None:
    client = make_client(
        httpx.MockTransport(lambda r: httpx.Response(503, json={"detail": "LLM rate limit"}))
    )
    with pytest.raises(ApiError, match="LLM rate limit"):
        client.chat("q", [])


def test_chat_stream_yields_tokens_then_done() -> None:
    body = sse(("token", {"text": "Revenue "}), ("token", {"text": "was"}), ("done", RESULT))

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"question": "q", "history": []}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(make_client(httpx.MockTransport(handler)).chat_stream("q", []))

    assert events[:2] == [Token("Revenue "), Token("was")]
    assert isinstance(events[2], Done)
    assert events[2].result.sources[0].cited is True


def test_chat_stream_error_event_raises() -> None:
    body = sse(("error", {"detail": "Could not reach the LLM service", "retryable": True}))
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(200, text=body)))
    with pytest.raises(ApiError, match="Could not reach") as info:
        list(client.chat_stream("q", []))
    assert info.value.retryable


def test_iter_sse_handles_multiline_data_and_missing_final_blank_line() -> None:
    lines = ["event: done", 'data: {"a":', "data: 1}", "", "event: token", 'data: {"text": "x"}']
    assert list(iter_sse(iter(lines))) == [("done", {"a": 1}), ("token", {"text": "x"})]
