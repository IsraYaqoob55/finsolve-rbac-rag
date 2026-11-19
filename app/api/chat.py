"""Chat endpoints.

POST /chat         -> the full answer as JSON
POST /chat/stream  -> Server-Sent Events: the answer token by token, then sources

SSE is a simple HTTP format for server->client streams:
    event: token
    data: {"text": "Revenue"}

    event: done
    data: {...full response with sources...}
"""

import json
import logging
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import Audit, ChatRateLimit, CurrentUser, Scope, ServicesDep
from app.auth.user_store import User
from app.core.audit import AuditLog
from app.llm.base import LLMError, Message
from app.services.chat_service import ChatAnswer, DoneEvent, TokenEvent

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class HistoryMessage(BaseModel):
    # Clients may only send user/assistant turns: a "system" message from the
    # client could override our instructions.
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)


class SourceOut(BaseModel):
    number: int
    document: str
    title: str
    section: str
    department: str
    quarter: str | None
    score: float
    excerpt: str
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str | None
    latency_ms: int
    role: str
    search_query: str = Field(description="What was searched (follow-ups are rewritten)")
    route: str = Field(description='"docs", or "sql+docs" when a SQL result was a source')


def _to_response(result: ChatAnswer, role: str) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                number=s.number,
                document=s.document,
                title=s.title,
                section=s.section,
                department=s.department,
                quarter=s.quarter,
                score=s.score,
                excerpt=s.content,
                cited=s.cited,
            )
            for s in result.sources
        ],
        model=result.model,
        latency_ms=result.latency_ms,
        role=role,
        search_query=result.search_query,
        route=result.route,
    )


def _audit_chat(
    audit: AuditLog,
    user: User,
    question: str,
    outcome: str,
    result: ChatAnswer | None = None,
    error: str | None = None,
) -> None:
    """One audit line per question: who asked, and which documents fed the answer."""
    fields: dict[str, object] = {
        "user": user.username,
        "role": user.role,
        "question": question,
        "outcome": outcome,
    }
    if result is not None:
        fields |= {
            "route": result.route,
            "search_query": result.search_query,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "documents": sorted({s.document for s in result.sources}),
            "departments": sorted({s.department for s in result.sources}),
            "cited": [s.number for s in result.cited_sources],
        }
    if error is not None:
        fields["error"] = error
    audit.record("chat", **fields)


def _history(request: ChatRequest) -> list[Message]:
    return [Message(m.role, m.content) for m in request.history]


@router.post("", response_model=ChatResponse, dependencies=[ChatRateLimit])
def chat(
    request: ChatRequest, user: CurrentUser, scope: Scope, services: ServicesDep, audit: Audit
) -> ChatResponse:
    """Ask a question; get a grounded answer with citations to your accessible documents."""
    try:
        result = services.chat.answer(request.question, user, scope, _history(request))
    except LLMError as exc:
        logger.warning("LLM failure", extra={"error": str(exc)})
        _audit_chat(audit, user, request.question, "llm_error", error=str(exc))
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    _audit_chat(audit, user, request.question, "answered", result)
    return _to_response(result, user.role)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream", dependencies=[ChatRateLimit])
def chat_stream(
    request: ChatRequest, user: CurrentUser, scope: Scope, services: ServicesDep, audit: Audit
) -> StreamingResponse:
    """Same as POST /chat, streamed as Server-Sent Events (token, done, error)."""
    history = _history(request)

    def events() -> Iterator[str]:
        try:
            for event in services.chat.stream(request.question, user, scope, history):
                if isinstance(event, TokenEvent):
                    yield _sse("token", {"text": event.text})
                elif isinstance(event, DoneEvent):
                    _audit_chat(audit, user, request.question, "answered", event.result)
                    yield _sse("done", _to_response(event.result, user.role).model_dump())
        except LLMError as exc:
            # Headers (200) are already sent once streaming starts, so errors
            # travel as an event the client can show.
            logger.warning("LLM failure during stream", extra={"error": str(exc)})
            _audit_chat(audit, user, request.question, "llm_error", error=str(exc))
            yield _sse("error", {"detail": str(exc), "retryable": exc.retryable})
        except Exception:
            # Same reason: the middleware can't turn this into a 500 any more.
            logger.exception("Unexpected error during stream")
            _audit_chat(audit, user, request.question, "error")
            yield _sse("error", {"detail": "Internal server error", "retryable": False})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
