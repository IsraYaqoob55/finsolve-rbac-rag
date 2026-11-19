"""Secure retriever: the only way user-facing code searches documents.

It takes an ``AccessScope`` (never a raw department list or None), applies it
as a database filter on every candidate list, and re-checks the results with
the RBAC guard. Quality features are layered behind the same API:

    question ─▶ detect quarters ─▶ dense (+ BM25) search, quarter-boosted, fused (RRF)
             ─▶ RBAC guard ─▶ cross-encoder rerank ─▶ top-k

Each layer can be switched off (``RetrievalOptions``), which is how the
evaluation script measures what each one contributes.
"""

import logging
from dataclasses import dataclass

from app.nlp.query_analysis import detect_quarters
from app.rbac.guards import enforce_scope
from app.rbac.policy import AccessScope
from app.retrieval.embeddings import Embedder, SparseEmbedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import QdrantVectorStore, SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalOptions:
    hybrid: bool = True
    quarter_boost: bool = True
    rerank: bool = True
    candidates: int = 10  # how many to fetch for fusion / reranking


class SecureRetriever:
    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Reranker | None = None,
        options: RetrievalOptions | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._sparse = sparse_embedder
        self._reranker = reranker
        self._options = options or RetrievalOptions()

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
        opts = self._options
        use_sparse = opts.hybrid and self._sparse is not None
        use_rerank = opts.rerank and self._reranker is not None
        boost = detect_quarters(query) if opts.quarter_boost and not quarter else ()

        hits = self._store.search(
            self._embedder.embed_query(query),
            sparse=self._sparse.embed_query(query) if use_sparse and self._sparse else None,
            limit=max(opts.candidates, limit) if use_rerank else limit,
            departments=sorted(scope.departments),
            quarter=quarter,
            year=year,
            boost_quarters=boost,
            candidates=max(opts.candidates, limit),
        )
        hits = enforce_scope(hits, scope)
        if use_rerank and self._reranker is not None and hits:
            hits = self._reranker.rerank(query, hits)
        hits = hits[:limit]
        logger.info(
            "retrieve",
            extra={
                "role": scope.role,
                "hits": len(hits),
                "boost": list(boost),
                "docs": [h.chunk.doc_id for h in hits],
            },
        )
        return hits
