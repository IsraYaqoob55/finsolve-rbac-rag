"""Cross-encoder reranking: the precision step of retrieval.

Embedding search (a "bi-encoder") embeds the question and each chunk
*separately*, then compares vectors: fast enough for millions of chunks, but
it never sees question and chunk together. A cross-encoder reads the pair
*jointly* ("does this passage answer this question?"): far more accurate,
too slow for the whole index. So: retrieve ~30 candidates fast, rerank them
precisely, keep the top few.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from app.retrieval.vector_store import SearchHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Return the hits re-scored (0..1) and sorted, best first."""
        ...


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("Loading reranker %s", model_name)
        self._model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )

    def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        # Score the embedded text (title + section + content) so the model can
        # also use the section path, e.g. "Q3 - July to September 2024".
        logits = list(self._model.rerank(query, [h.chunk.text for h in hits]))
        # Raw logits -> 0..1, so scores are readable in the UI.
        rescored = [
            replace(hit, score=round(_sigmoid(float(s)), 4))
            for hit, s in zip(hits, logits, strict=True)
        ]
        return sorted(rescored, key=lambda h: h.score, reverse=True)
