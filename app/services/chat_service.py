"""Chat orchestration: the RAG loop.

    question ─▶ secure retrieve (role-filtered) ─▶ build grounded prompt
             ─▶ LLM ─▶ answer + the sources it actually cited

If retrieval finds nothing the user may see, we answer "not found" without
calling the LLM: no context means nothing to ground on, and it saves a call.
"""

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from app.auth.user_store import User
from app.core.config import Settings
from app.llm.base import LLMProvider, Message
from app.llm.prompts import (
    NO_CONTEXT_ANSWER,
    build_messages,
    build_system_prompt,
    extract_citations,
    format_departments,
    normalize_citations,
)
from app.rbac.policy import AccessScope
from app.retrieval.retriever import SecureRetriever
from app.retrieval.vector_store import SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    number: int  # the [n] used in the answer
    document: str
    title: str
    section: str
    department: str
    quarter: str | None
    score: float
    content: str
    cited: bool  # did the answer actually reference it?


@dataclass
class ChatAnswer:
    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str | None = None  # None when no LLM call was needed
    latency_ms: int = 0

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.cited]


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    result: ChatAnswer


ChatEvent = TokenEvent | DoneEvent


@dataclass(frozen=True)
class _Prepared:
    hits: list[SearchHit]
    messages: list[Message]


class ChatService:
    def __init__(self, retriever: SecureRetriever, llm: LLMProvider, settings: Settings) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    # ---------- public API ----------

    def answer(
        self,
        question: str,
        user: User,
        scope: AccessScope,
        history: Sequence[Message] = (),
    ) -> ChatAnswer:
        started = time.perf_counter()
        prepared = self._prepare(question, user, scope, history)
        if not prepared.hits:
            return self._no_context(scope, started)
        text = self._llm.complete(prepared.messages)
        return self._finish(text, prepared.hits, started, scope)

    def stream(
        self,
        question: str,
        user: User,
        scope: AccessScope,
        history: Sequence[Message] = (),
    ) -> Iterator[ChatEvent]:
        """Yield tokens as they arrive, then one DoneEvent with sources and timing."""
        started = time.perf_counter()
        prepared = self._prepare(question, user, scope, history)
        if not prepared.hits:
            result = self._no_context(scope, started)
            yield TokenEvent(result.answer)
            yield DoneEvent(result)
            return
        parts: list[str] = []
        for token in self._llm.stream(prepared.messages):
            token = normalize_citations(token)
            parts.append(token)
            yield TokenEvent(token)
        yield DoneEvent(self._finish("".join(parts), prepared.hits, started, scope))

    # ---------- steps ----------

    def _prepare(
        self, question: str, user: User, scope: AccessScope, history: Sequence[Message]
    ) -> _Prepared:
        hits = self._retriever.retrieve(question, scope, limit=self._settings.retrieval_top_k)
        system = build_system_prompt(user.full_name, scope.display_name, sorted(scope.departments))
        keep = self._settings.chat_history_messages
        recent = list(history)[-keep:] if keep else []  # note: [-0:] would keep everything
        return _Prepared(hits, build_messages(system, question, hits, recent))

    def _no_context(self, scope: AccessScope, started: float) -> ChatAnswer:
        departments = format_departments(sorted(scope.departments))
        return ChatAnswer(
            answer=NO_CONTEXT_ANSWER.format(departments=departments),
            latency_ms=_elapsed_ms(started),
        )

    def _finish(
        self, text: str, hits: list[SearchHit], started: float, scope: AccessScope
    ) -> ChatAnswer:
        text = normalize_citations(text)
        cited = set(extract_citations(text, len(hits)))
        sources = [
            Source(
                number=n,
                document=hit.chunk.source_path,
                title=hit.chunk.title,
                section=hit.chunk.section,
                department=hit.chunk.department,
                quarter=hit.chunk.quarter,
                score=round(hit.score, 4),
                content=hit.chunk.content,
                cited=n in cited,
            )
            for n, hit in enumerate(hits, start=1)
        ]
        result = ChatAnswer(
            answer=text.strip(),
            sources=sources,
            model=self._llm.model_name,
            latency_ms=_elapsed_ms(started),
        )
        logger.info(
            "chat_answer",
            extra={
                "role": scope.role,
                "retrieved": len(hits),
                "cited": sorted(cited),
                "latency_ms": result.latency_ms,
            },
        )
        return result


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
