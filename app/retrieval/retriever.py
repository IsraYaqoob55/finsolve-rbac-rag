"""Secure retriever: the only way user-facing code searches documents.

It takes an ``AccessScope`` (never a raw department list or None), applies it
as a database filter, and then re-checks the results with the RBAC guard.
Later phases (hybrid search, reranking) plug in here, behind the same API,
so the security guarantees stay in one place.
"""

import logging

from app.rbac.guards import enforce_scope
from app.rbac.policy import AccessScope
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import QdrantVectorStore, SearchHit

logger = logging.getLogger(__name__)


class SecureRetriever:
    def __init__(self, store: QdrantVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        scope: AccessScope,
        limit: int = 5,
        quarter: str | None = None,
        year: int | None = None,
    ) -> list[SearchHit]:
        if not scope.departments:
            return []  # deny by default: an empty scope sees nothing
        hits = self._store.search(
            self._embedder.embed_query(query),
            limit=limit,
            departments=sorted(scope.departments),
            quarter=quarter,
            year=year,
        )
        hits = enforce_scope(hits, scope)
        logger.info(
            "retrieve",
            extra={"role": scope.role, "hits": len(hits), "docs": [h.chunk.doc_id for h in hits]},
        )
        return hits
