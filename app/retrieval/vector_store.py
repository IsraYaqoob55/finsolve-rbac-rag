"""Qdrant vector store wrapper.

Every chunk is stored as a Qdrant *point*: an id, a vector, and a *payload*
(the chunk's metadata as JSON). Payload filters are what make RBAC safe:
the department filter is applied **inside the database search**, so chunks a
user may not see are never returned, never reach the LLM, and can't leak.
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)

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
    """One search result: the chunk and how similar it is to the query (0..1)."""

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
            return
        logger.info("Creating collection %s (dim=%d)", self._collection, self._dimension)
        self._client.create_collection(
            collection_name=self._collection,
            # Cosine similarity: compares direction of vectors, i.e. meaning,
            # regardless of text length.
            vectors_config=models.VectorParams(
                size=self._dimension, distance=models.Distance.COSINE
            ),
        )
        # Embedded (local) Qdrant ignores payload indexes and warns about it;
        # they matter on a Qdrant server, so we still create them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in _INDEXED_FIELDS.items():
                self._client.create_payload_index(self._collection, field, field_schema=schema)

    def recreate(self) -> None:
        """Drop everything and start empty (used by --rebuild)."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self.ensure_collection()

    def close(self) -> None:
        """Release the client (embedded mode holds a file lock on the storage folder)."""
        self._client.close()

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    # ---------- writes ----------

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Insert or overwrite chunks (ids are deterministic, so this is idempotent)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            models.PointStruct(id=c.chunk_id, vector=list(v), payload=c.to_payload())
            for c, v in zip(chunks, vectors, strict=True)
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
        query_vector: Sequence[float],
        limit: int = 5,
        departments: Sequence[str] | None = None,
        quarter: str | None = None,
        year: int | None = None,
    ) -> list[SearchHit]:
        """Nearest-neighbour search, filtered inside Qdrant.

        ``departments=None`` means *no* department filter. Callers that serve
        users must always pass the user's allowed departments (Phase 2).
        """
        response = self._client.query_points(
            self._collection,
            query=list(query_vector),
            query_filter=build_filter(departments, quarter, year),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk=Chunk(chunk_id=str(point.id), **(point.payload or {})),
                score=point.score,
            )
            for point in response.points
        ]
