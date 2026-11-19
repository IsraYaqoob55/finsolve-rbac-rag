"""Text embeddings: turning text into vectors whose distance reflects meaning.

The rest of the code depends only on the small ``Embedder`` protocol, so we
can swap the local fastembed model for OpenAI/Cohere embeddings, or a fake
in tests, without touching the pipeline or retriever.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that can embed documents and queries into same-sized vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


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
