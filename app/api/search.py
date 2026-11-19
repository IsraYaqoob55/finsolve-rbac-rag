"""Document search endpoint (retrieval only, no LLM yet).

Useful on its own for debugging retrieval, and the chat endpoint in Phase 3
builds on exactly the same secure retriever.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import Audit, ChatRateLimit, CurrentUser, Scope, ServicesDep

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    quarter: str | None = Field(default=None, pattern=r"^Q[1-4]$")
    year: int | None = Field(default=None, ge=2000, le=2100)


class SourceReference(BaseModel):
    document: str
    title: str
    section: str
    department: str
    quarter: str | None


class SearchResult(BaseModel):
    content: str
    score: float
    source: SourceReference


class SearchResponse(BaseModel):
    query: str
    role: str
    results: list[SearchResult]


@router.post("/search", response_model=SearchResponse, dependencies=[ChatRateLimit])
def search(
    request: SearchRequest, user: CurrentUser, scope: Scope, services: ServicesDep, audit: Audit
) -> SearchResponse:
    """Semantic search over the documents the caller's role may access."""
    hits = services.retriever.retrieve(
        request.query, scope, limit=request.limit, quarter=request.quarter, year=request.year
    )
    # Search returns raw document text, so it is audited like chat.
    audit.record(
        "search",
        user=user.username,
        role=user.role,
        question=request.query,
        documents=sorted({hit.chunk.source_path for hit in hits}),
        departments=sorted({hit.chunk.department for hit in hits}),
    )
    return SearchResponse(
        query=request.query,
        role=user.role,
        results=[
            SearchResult(
                content=hit.chunk.content,
                score=round(hit.score, 4),
                source=SourceReference(
                    document=hit.chunk.source_path,
                    title=hit.chunk.title,
                    section=hit.chunk.section,
                    department=hit.chunk.department,
                    quarter=hit.chunk.quarter,
                ),
            )
            for hit in hits
        ],
    )
