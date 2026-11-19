"""Measure retrieval quality against the golden set.

    python -m eval.retrieval_eval                 # current settings
    python -m eval.retrieval_eval --compare       # every retrieval mode side by side
    python -m eval.retrieval_eval --details       # per-question ranks

Metrics (k = how many chunks the chat actually sends to the LLM):
- Hit@k: share of questions where a correct chunk is in the top k. If it's not
  there, the LLM cannot answer, however good it is.
- MRR (mean reciprocal rank): 1/rank of the first correct chunk, averaged.
  1.0 = always first; 0.5 = typically second. Rewards ranking the answer high.
"""

import argparse
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.logging import ensure_utf8_console, setup_logging
from app.rbac.policy import AccessScope
from app.retrieval.retriever import SecureRetriever
from app.retrieval.vector_store import SearchHit

GOLDEN = PROJECT_ROOT / "eval" / "retrieval_golden.yaml"
ALL_DEPARTMENTS = frozenset({"general", "engineering", "finance", "marketing", "hr"})
C_LEVEL = AccessScope("c_level", "C-Level", ALL_DEPARTMENTS)


@dataclass(frozen=True)
class Case:
    question: str
    expected: list[dict[str, str]]


def load_cases(path: Path = GOLDEN) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Case(q["question"], q["expected"]) for q in raw["questions"]]


def is_relevant(hit: SearchHit, expected: list[dict[str, str]]) -> bool:
    chunk = hit.chunk
    for exp in expected:
        if chunk.source_path != exp["doc"]:
            continue
        if "section" in exp and exp["section"] in chunk.section:
            return True
        if "contains" in exp and exp["contains"] in chunk.content:
            return True
    return False


def first_relevant_rank(hits: list[SearchHit], expected: list[dict[str, str]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, expected):
            return rank
    return None


@dataclass
class Report:
    name: str
    ranks: list[int | None]
    k: int

    def hit_at(self, k: int) -> float:
        return sum(1 for r in self.ranks if r is not None and r <= k) / len(self.ranks)

    @property
    def mrr(self) -> float:
        return statistics.mean(1 / r if r else 0.0 for r in self.ranks)


Retrieve = Callable[[str], list[SearchHit]]


def evaluate(name: str, retrieve: Retrieve, cases: list[Case], k: int) -> Report:
    return Report(name, [first_relevant_rank(retrieve(c.question), c.expected) for c in cases], k)


def print_table(reports: list[Report]) -> None:
    k = reports[0].k
    print(f"\n{'mode':34} {'Hit@1':>6} {'Hit@3':>6} {f'Hit@{k}':>6} {'MRR':>6}")
    print("-" * 62)
    for r in reports:
        print(f"{r.name:34} {r.hit_at(1):6.0%} {r.hit_at(3):6.0%} {r.hit_at(k):6.0%} {r.mrr:6.3f}")


def print_details(report: Report, cases: list[Case]) -> None:
    print(f"\nPer-question rank of first correct chunk ({report.name}):")
    for case, rank in zip(cases, report.ranks, strict=True):
        mark = "MISS" if rank is None or rank > report.k else f"#{rank}"
        print(f"  {mark:>5}  {case.question}")


def as_c_level(retriever: SecureRetriever, k: int) -> Retrieve:
    def retrieve(question: str) -> list[SearchHit]:
        return retriever.retrieve(question, C_LEVEL, limit=k)

    return retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", action="store_true", help="compare retrieval modes")
    parser.add_argument("--details", action="store_true", help="show per-question ranks")
    args = parser.parse_args()

    ensure_utf8_console()
    setup_logging("WARNING")
    settings = get_settings()
    cases = load_cases()
    k = settings.retrieval_top_k

    from app.services.container import RETRIEVAL_MODES, build_retrieval_components

    # Models and index are loaded once and shared by every mode.
    components = build_retrieval_components(settings)
    modes: dict[str, dict[str, Any]] = RETRIEVAL_MODES if args.compare else {"current settings": {}}
    reports = []
    try:
        for name, overrides in modes.items():
            mode_settings: Settings = settings.model_copy(update=overrides)
            reports.append(
                evaluate(name, as_c_level(components.retriever(mode_settings), k), cases, k)
            )
            if args.details:
                print_details(reports[-1], cases)
    finally:
        components.store.close()
    print(f"\n{len(cases)} questions, C-level scope (all departments), k={k}")
    print_table(reports)


if __name__ == "__main__":
    main()
