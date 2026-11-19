"""Ingestion pipeline: data/ folder -> chunks -> embeddings -> Qdrant.

Usage:
    python -m app.ingestion.pipeline              # incremental: only changed files
    python -m app.ingestion.pipeline --rebuild    # wipe and re-index everything
    python -m app.ingestion.pipeline --dry-run    # show chunking stats, no embedding
    python -m app.ingestion.pipeline --dry-run --show 3   # also print 3 sample chunks

Incremental by design: each file's SHA-256 is stored with its chunks. On the
next run unchanged files are skipped, changed files are re-indexed, and files
that were deleted from disk are removed from the index. With thousands of
documents this turns a full re-index into seconds of work.
"""

import argparse
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.logging import ensure_utf8_console, setup_logging
from app.ingestion.chunkers import chunk_document
from app.ingestion.loaders import discover_documents
from app.ingestion.models import Chunk, Document
from app.retrieval.embeddings import Embedder, SparseEmbedder
from app.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """What one pipeline run did."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def summary(self) -> str:
        return (
            f"added={len(self.added)} updated={len(self.updated)} "
            f"unchanged={len(self.unchanged)} deleted={len(self.deleted)} "
            f"chunks_written={self.chunks_written}"
        )


def chunk_documents(documents: list[Document], settings: Settings) -> list[Chunk]:
    return [
        chunk
        for doc in documents
        for chunk in chunk_document(
            doc,
            settings.chunk_max_chars,
            settings.chunk_overlap_chars,
            settings.chunk_min_section_chars,
        )
    ]


def run_ingestion(
    settings: Settings,
    store: QdrantVectorStore,
    embedder: Embedder,
    sparse_embedder: SparseEmbedder,
    data_dir: Path | None = None,
    rebuild: bool = False,
) -> IngestionReport:
    """Bring the vector store in sync with the files in ``data_dir``."""
    data_dir = data_dir or settings.resolve(settings.data_dir)
    if rebuild:
        store.recreate()
    else:
        store.ensure_collection()

    report = IngestionReport()
    on_disk = discover_documents(data_dir)
    indexed = store.indexed_sources()

    for source, documents in on_disk.items():
        file_hash = documents[0].file_hash
        if indexed.get(source) == file_hash:
            report.unchanged.append(source)
            continue

        chunks = chunk_documents(documents, settings)
        texts = [c.text for c in chunks]
        store.upsert(
            chunks, embedder.embed_documents(texts), sparse_embedder.embed_documents(texts)
        )
        # Remove points from the previous version (e.g. the doc got shorter).
        store.delete_source(source, keep_file_hash=file_hash)

        (report.updated if source in indexed else report.added).append(source)
        report.chunks_written += len(chunks)
        logger.info("Indexed %s: %d chunks", source, len(chunks))

    for source in sorted(indexed.keys() - on_disk.keys()):
        store.delete_source(source)
        report.deleted.append(source)
        logger.info("Removed %s (file no longer exists)", source)

    return report


def print_dry_run(settings: Settings, show: int) -> None:
    """Chunk everything and print statistics, without loading any model."""
    on_disk = discover_documents(settings.resolve(settings.data_dir))
    all_chunks: list[Chunk] = []
    print(f"{'source':45} {'dept':12} {'chunks':>6} {'avg chars':>9}")
    for source, documents in on_disk.items():
        chunks = chunk_documents(documents, settings)
        all_chunks.extend(chunks)
        avg = statistics.mean(len(c.content) for c in chunks) if chunks else 0
        print(f"{source:45} {documents[0].department:12} {len(chunks):>6} {avg:>9.0f}")

    sizes = [len(c.text) for c in all_chunks]
    print(
        f"\nTotal chunks: {len(all_chunks)}  |  text size min/median/max: "
        f"{min(sizes)}/{statistics.median(sizes):.0f}/{max(sizes)}"
    )
    print("By department:", dict(Counter(c.department for c in all_chunks)))
    print("By quarter:   ", dict(Counter(c.quarter or "-" for c in all_chunks)))

    for chunk in all_chunks[:: max(len(all_chunks) // max(show, 1), 1)][:show]:
        print("\n" + "=" * 80)
        print(f"{chunk.source_path} | {chunk.section or '(no section)'} | quarter={chunk.quarter}")
        print("-" * 80)
        print(chunk.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the data folder into Qdrant.")
    parser.add_argument("--rebuild", action="store_true", help="drop and re-create the index")
    parser.add_argument("--dry-run", action="store_true", help="only show chunking statistics")
    parser.add_argument("--show", type=int, default=0, help="with --dry-run: print N sample chunks")
    args = parser.parse_args()

    ensure_utf8_console()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    if args.dry_run:
        print_dry_run(settings, args.show)
        return

    # Heavy imports only when we really embed.
    from app.retrieval.vector_store import build_client
    from app.services.container import build_embedders

    embedder, sparse_embedder = build_embedders(settings)
    store = QdrantVectorStore(
        build_client(settings), settings.qdrant_collection, embedder.dimension
    )
    try:
        report = run_ingestion(settings, store, embedder, sparse_embedder, rebuild=args.rebuild)
        logger.info("Ingestion finished: %s | total points: %d", report.summary(), store.count())
    finally:
        store.close()


if __name__ == "__main__":
    main()
