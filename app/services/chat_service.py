"""Chat orchestration: the RAG loop.

    question ─▶ secure retrieve (role-filtered) ─▶ build grounded prompt
             ─▶ LLM ─▶ answer + the sources it actually cited

If retrieval finds nothing the user may see, we answer "not found" without
calling the LLM: no context means nothing to ground on, and it saves a call.
"""

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from app.auth.user_store import User
from app.core.config import Settings
from app.llm.base import LLMProvider, Message
from app.llm.prompts import (
    NO_CONTEXT_ANSWER,
    CitationStreamNormalizer,
    build_messages,
    build_system_prompt,
    extract_citations,
    format_departments,
    normalize_citations,
)
from app.nlp.rewriter import QueryRewriter
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
    search_query: str = ""  # what was actually searched (a rewritten follow-up)

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
    search_query: str


class ChatService:
    def __init__(
        self,
        retriever: SecureRetriever,
        llm: LLMProvider,
        settings: Settings,
        rewriter: QueryRewriter | None = None,
        department_catalog: Mapping[str, str] | None = None,
    ) -> None:
        """``department_catalog``: every department name -> description (from roles.yaml)."""
        self._retriever = retriever
        self._llm = llm
        self._settings = settings
        self._rewriter = rewriter
        self._catalog = dict(department_catalog or {})

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
            return self._no_context(scope, started, prepared.search_query)
        text = self._llm.complete(prepared.messages)
        return self._finish(text, prepared, started, scope)

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
            result = self._no_context(scope, started, prepared.search_query)
            yield TokenEvent(result.answer)
            yield DoneEvent(result)
            return
        parts: list[str] = []
        normalizer = CitationStreamNormalizer()
        for token in self._llm.stream(prepared.messages):
            text = normalizer.feed(token)
            if text:
                parts.append(text)
                yield TokenEvent(text)
        if tail := normalizer.flush():
            parts.append(tail)
            yield TokenEvent(tail)
        yield DoneEvent(self._finish("".join(parts), prepared, started, scope))

    # ---------- steps ----------

    def _prepare(
        self, question: str, user: User, scope: AccessScope, history: Sequence[Message]
    ) -> _Prepared:
        keep = self._settings.chat_history_messages
        recent = list(history)[-keep:] if keep else []  # note: [-0:] would keep everything
        # Follow-ups ("and Q4?") are rewritten into standalone queries for the
        # *search only*; the LLM still answers the user's own words + history.
        search_query = question
        if recent and self._rewriter is not None:
            search_query = self._rewriter.rewrite(question, recent)
        hits = self._retriever.retrieve(search_query, scope, limit=self._settings.retrieval_top_k)
        restricted = {d: desc for d, desc in self._catalog.items() if d not in scope.departments}
        system = build_system_prompt(
            user.full_name, scope.display_name, sorted(scope.departments), restricted
        )
        return _Prepared(hits, build_messages(system, question, hits, recent), search_query)

    def _no_context(self, scope: AccessScope, started: float, search_query: str) -> ChatAnswer:
        departments = format_departments(sorted(scope.departments))
        return ChatAnswer(
            answer=NO_CONTEXT_ANSWER.format(departments=departments),
            latency_ms=_elapsed_ms(started),
            search_query=search_query,
        )

    def _finish(
        self, text: str, prepared: _Prepared, started: float, scope: AccessScope
    ) -> ChatAnswer:
        hits = prepared.hits
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
            search_query=prepared.search_query,
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
