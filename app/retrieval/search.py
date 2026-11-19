"""Command-line semantic search, for exploring what retrieval returns.

Usage:
    python -m app.retrieval.search "What was revenue in Q3?"
    python -m app.retrieval.search "leave policy" --departments general
    python -m app.retrieval.search "campaign ROI" --departments marketing --quarter Q2 -k 3
"""

import argparse
import textwrap

from app.core.config import get_settings
from app.core.logging import ensure_utf8_console, setup_logging
from app.retrieval.embeddings import FastEmbedEmbedder
from app.retrieval.vector_store import QdrantVectorStore, build_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--departments", nargs="+", help="only these departments (RBAC preview)")
    parser.add_argument("--quarter", help="e.g. Q3")
    parser.add_argument("--year", type=int)
    parser.add_argument("--full", action="store_true", help="print whole chunks")
    args = parser.parse_args()

    ensure_utf8_console()
    settings = get_settings()
    setup_logging("WARNING")
    embedder = FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.resolve(settings.embedding_cache_dir),
        query_prefix=settings.embedding_query_prefix,
    )
    store = QdrantVectorStore(
        build_client(settings), settings.qdrant_collection, embedder.dimension
    )

    try:
        hits = store.search(
            embedder.embed_query(args.query),
            limit=args.k,
            departments=args.departments,
            quarter=args.quarter,
            year=args.year,
        )
    finally:
        store.close()
    if not hits:
        print("No results.")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(f"\n#{rank}  score={hit.score:.3f}  [{c.department}]  {c.source_path}")
        print(f"    section: {c.section or '-'}   quarter: {c.quarter or '-'}")
        body = c.content if args.full else textwrap.shorten(c.content, 300, placeholder=" ...")
        print(textwrap.indent(body, "    "))


if __name__ == "__main__":
    main()
