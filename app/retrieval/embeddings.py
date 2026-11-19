"""Text embeddings: turning text into vectors.

Two kinds, used together for hybrid search:
- **Dense** (BGE): a few hundred floats capturing *meaning*. "time off" and
  "leave" end up close together even though no word matches.
- **Sparse** (BM25): one weight per *word* present, like a search engine.
  Great at exact terms the dense model blurs: "DSO", "FINEMP1003", "Q3".

The rest of the code depends only on the small protocols below, so models can
be swapped (or faked in tests) without touching the pipeline or retriever.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Windows without Developer Mode can't create symlinks; the model cache still
# works (it copies files instead), so silence the noisy warning about it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass(frozen=True)
class SparseVector:
    """Word ids and their weights (all other words implicitly 0)."""

    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    """Anything that can embed documents and queries into same-sized dense vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model via fastembed (no GPU, no API key, no cost).

    The model (~130 MB for bge-small) is downloaded once into ``cache_dir``.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        # Imported here so modules that only *type* against Embedder don't pay
        # the cost of loading onnxruntime.
        from fastembed import TextEmbedding

        logger.info("Loading embedding model %s", model_name)
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._dimension = int(self._model.embedding_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self._batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        # Asymmetric retrieval: short questions and long passages are embedded
        # differently. BGE expects an instruction prefix on the query side only.
        vector = next(iter(self._model.embed([self._query_prefix + text])))
        return [float(x) for x in vector]


class FastEmbedSparseEmbedder:
    """BM25 term weights via fastembed (tokenise, stem, drop stopwords).

    Qdrant applies the IDF part of BM25 at search time (the collection's sparse
    vector uses the IDF modifier), so rare words like "DSO" weigh more.
    """

    def __init__(self, model_name: str = "Qdrant/bm25", cache_dir: Path | None = None) -> None:
        from fastembed import SparseTextEmbedding

        logger.info("Loading sparse model %s", model_name)
        self._model = SparseTextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(e.indices.tolist(), e.values.tolist()) for e in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        # query_embed gives every query term weight 1 (IDF is applied by Qdrant).
        embedding = next(iter(self._model.query_embed(text)))
        return SparseVector(embedding.indices.tolist(), embedding.values.tolist())
