"""Command-line search, for exploring what retrieval returns.

Usage:
    python -m app.retrieval.search "What was revenue in Q3?"
    python -m app.retrieval.search "leave policy" --departments general
    python -m app.retrieval.search "campaign ROI" --departments marketing --quarter Q2 -k 3
    python -m app.retrieval.search "net income Q3" --no-rerank --no-hybrid   # compare modes
"""

import argparse
import textwrap

from app.core.config import get_settings
from app.core.logging import ensure_utf8_console, setup_logging
from app.rbac.policy import AccessScope, RolePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--departments", nargs="+", help="only these departments (RBAC preview)")
    parser.add_argument("--quarter", help="hard filter, e.g. Q3")
    parser.add_argument("--year", type=int)
    parser.add_argument("--no-hybrid", action="store_true", help="dense search only")
    parser.add_argument("--no-boost", action="store_true", help="ignore quarters in the query")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--full", action="store_true", help="print whole chunks")
    args = parser.parse_args()

    ensure_utf8_console()
    setup_logging("WARNING")
    settings = get_settings().model_copy(
        update={
            "retrieval_hybrid": not args.no_hybrid,
            "retrieval_quarter_boost": not args.no_boost,
            "retrieval_rerank": not args.no_rerank,
        }
    )
    from app.services.container import build_retrieval_components

    policy = RolePolicy.from_yaml(settings.resolve(settings.roles_config))
    departments = frozenset(args.departments or policy.departments)
    scope = AccessScope("cli", "Command line", departments)
    components = build_retrieval_components(settings, with_reranker=not args.no_rerank)
    try:
        hits = components.retriever(settings).retrieve(
            args.query, scope, limit=args.k, quarter=args.quarter, year=args.year
        )
    finally:
        components.store.close()

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
