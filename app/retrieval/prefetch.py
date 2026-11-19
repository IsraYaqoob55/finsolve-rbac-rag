"""Download the embedding, BM25 and reranker models into the cache.

    python -m app.retrieval.prefetch

The Docker build runs this so the models are baked into the image: containers
start without downloading from Hugging Face, and every replica uses the same
files. It imports only the settings and fastembed, so the Docker layer that
runs it is not rebuilt when application code changes.
"""

import os

from app.core.config import get_settings


def main() -> None:  # pragma: no cover - downloads ~250 MB, exercised by the Docker build
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from fastembed import SparseTextEmbedding, TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings = get_settings()
    cache = str(settings.resolve(settings.embedding_cache_dir))
    TextEmbedding(model_name=settings.embedding_model, cache_dir=cache)
    SparseTextEmbedding(model_name=settings.sparse_model, cache_dir=cache)
    TextCrossEncoder(model_name=settings.reranker_model, cache_dir=cache)
    print(f"Models ready in {cache}")


if __name__ == "__main__":
    main()
