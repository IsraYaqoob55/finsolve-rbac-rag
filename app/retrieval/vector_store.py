"""Qdrant vector store wrapper.

Every chunk is stored as a Qdrant *point*: an id, vectors, and a *payload*
(the chunk's metadata as JSON). Payload filters are what make RBAC safe:
the department filter is applied **inside the database search**, so chunks a
user may not see are never returned, never reach the LLM, and can't leak.

Each point carries two vectors:
- ``dense``: BGE meaning vector (cosine similarity)
- ``bm25``:  sparse keyword vector (Qdrant applies IDF at query time)
Hybrid search runs both and merges the rankings with Reciprocal Rank Fusion.
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.ingestion.models import Chunk
from app.retrieval.embeddings import SparseVector

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"

# Payload fields we filter on get an index, so filtering stays fast at scale.
_INDEXED_FIELDS: dict[str, models.PayloadSchemaType] = {
    "department": models.PayloadSchemaType.KEYWORD,
    "source_path": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "quarter": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
}


@dataclass(frozen=True)
class SearchHit:
    """One search result and its score.

    The score's scale depends on the stage that produced it: cosine similarity
    (dense only), a fused rank score (hybrid), or 0..1 relevance (reranked).
    Only compare scores produced by the same stage.
    """

    chunk: Chunk
    score: float


def build_client(settings: Settings) -> QdrantClient:
    """Server mode if QDRANT_URL is set, otherwise embedded on-disk mode."""
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key)
    path = settings.resolve(settings.qdrant_path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def build_filter(
    departments: Sequence[str] | None = None,
    quarter: str | None = None,
    year: int | None = None,
    any_quarter: Sequence[str] = (),
) -> models.Filter | None:
    """Translate our search options into a Qdrant filter (all conditions must hold)."""
    conditions: list[models.Condition] = []
    if departments is not None:
        conditions.append(
            models.FieldCondition(key="department", match=models.MatchAny(any=list(departments)))
        )
    if quarter:
        conditions.append(
            models.FieldCondition(key="quarter", match=models.MatchValue(value=quarter))
        )
    if any_quarter:
        conditions.append(
            models.FieldCondition(key="quarter", match=models.MatchAny(any=list(any_quarter)))
        )
    if year:
        conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    return models.Filter(must=conditions) if conditions else None


class QdrantVectorStore:
    """Stores and searches chunk vectors in one Qdrant collection."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    # ---------- collection lifecycle ----------

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes if they don't exist yet."""
        if self._client.collection_exists(self._collection):
            self._check_schema()
            return
        logger.info("Creating collection %s (dim=%d)", self._collection, self._dimension)
        self._client.create_collection(
            collection_name=self._collection,
            # Cosine similarity: compares the direction of vectors, i.e. meaning,
            # regardless of text length.
            vectors_config={
                DENSE: models.VectorParams(size=self._dimension, distance=models.Distance.COSINE)
            },
            # IDF modifier: Qdrant weights rare words higher, completing BM25.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Embedded (local) Qdrant ignores payload indexes and warns about it;
        # they matter on a Qdrant server, so we still create them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in _INDEXED_FIELDS.items():
                self._client.create_payload_index(self._collection, field, field_schema=schema)

    def _check_schema(self) -> None:
        """Fail with a clear message if the index was built by an older version."""
        params = self._client.get_collection(self._collection).config.params
        dense = params.vectors if isinstance(params.vectors, dict) else {}
        if DENSE not in dense or SPARSE not in (params.sparse_vectors or {}):
            raise RuntimeError(
                "The vector index uses an outdated schema. Rebuild it with: "
                "python -m app.ingestion.pipeline --rebuild"
            )

    def recreate(self) -> None:
        """Drop everything and start empty (used by --rebuild)."""
        if self._client.collection_exists(self._collection):
            # Delete the points explicitly first. Embedded Qdrant removes a
            # collection by deleting its folder, which can silently fail on
            # Windows while a file is open, and the "new" collection then
            # reloads the old points from disk.
            self._delete_all_points()
            self._client.delete_collection(self._collection)
        self.ensure_collection()
        if self.count():  # leftovers from a failed folder delete
            self._delete_all_points()
        if self.count():
            raise RuntimeError(f"Could not empty collection {self._collection!r}")

    def _delete_all_points(self) -> None:
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=models.Filter()),  # matches everything
            wait=True,
        )

    def close(self) -> None:
        """Release the client (embedded mode holds a file lock on the storage folder)."""
        self._client.close()

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    # ---------- writes ----------

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        """Insert or overwrite chunks (ids are deterministic, so this is idempotent)."""
        if not len(chunks) == len(dense) == len(sparse):
            raise ValueError("chunks, dense and sparse vectors must have the same length")
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE: list(d),
                    SPARSE: models.SparseVector(indices=s.indices, values=s.values),
                },
                payload=chunk.to_payload(),
            )
            for chunk, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self._client.upsert(self._collection, points=points, wait=True)

    def delete_source(self, source_path: str, keep_file_hash: str | None = None) -> None:
        """Delete a source file's points; optionally keep those from its current version.

        Upserting the new version first and then deleting only *stale* points
        means a document is never missing from search while it is re-indexed.
        """
        must: list[models.Condition] = [
            models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))
        ]
        must_not: list[models.Condition] = []
        if keep_file_hash:
            must_not.append(
                models.FieldCondition(
                    key="file_hash", match=models.MatchValue(value=keep_file_hash)
                )
            )
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=must, must_not=must_not)
            ),
            wait=True,
        )

    # ---------- reads ----------

    def indexed_sources(self) -> dict[str, str]:
        """Map of source_path -> file_hash for everything currently indexed."""
        sources: dict[str, str] = {}
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                limit=256,
                offset=offset,
                with_payload=["source_path", "file_hash"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                sources[str(payload["source_path"])] = str(payload["file_hash"])
            if offset is None:
                return sources

    def search(
        self,
        dense: Sequence[float],
        sparse: SparseVector | None = None,
        limit: int = 5,
        departments: Sequence[str] | None = None,
        quarter: str | None = None,
        year: int | None = None,
        boost_quarters: Sequence[str] = (),
        candidates: int = 30,
    ) -> list[SearchHit]:
        """Dense or hybrid search, always filtered inside Qdrant.

        - ``departments=None`` means no department filter. Code serving users
          must always pass the user's allowed departments (see SecureRetriever).
        - ``quarter``/``year`` are hard filters (only matching chunks).
        - ``boost_quarters`` is soft: extra candidate lists restricted to those
          quarters are fused in, so matching chunks collect more votes, but
          nothing else is excluded.
        Every candidate list carries the department filter.
        """
        base = build_filter(departments, quarter, year)
        if sparse is None and not boost_quarters:
            response = self._client.query_points(
                self._collection,
                query=list(dense),
                using=DENSE,
                query_filter=base,
                limit=limit,
                with_payload=True,
            )
        else:
            filters = [base]
            if boost_quarters:
                filters.append(build_filter(departments, quarter, year, any_quarter=boost_quarters))
            prefetch: list[models.Prefetch] = []
            for flt in filters:
                prefetch.append(
                    models.Prefetch(query=list(dense), using=DENSE, filter=flt, limit=candidates)
                )
                if sparse is not None:
                    prefetch.append(
                        models.Prefetch(
                            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                            using=SPARSE,
                            filter=flt,
                            limit=candidates,
                        )
                    )
            # Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).
            # Rank-based, so cosine and BM25 scores never need to be comparable.
            response = self._client.query_points(
                self._collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=base,
                limit=limit,
                with_payload=True,
            )
        return [
            SearchHit(
                chunk=Chunk(chunk_id=str(point.id), **(point.payload or {})), score=point.score
            )
            for point in response.points
        ]
