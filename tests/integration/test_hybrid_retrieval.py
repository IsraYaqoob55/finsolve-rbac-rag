"""Hybrid search, quarter boost, reranking and index maintenance (in-memory Qdrant)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.ingestion.pipeline import run_ingestion
from app.rbac.policy import AccessScope
from app.retrieval.retriever import RetrievalOptions, SecureRetriever
from app.retrieval.vector_store import QdrantVectorStore
from tests.fakes import DIM, FakeEmbedder, FakeReranker, FakeSparseEmbedder

# Four look-alike quarterly sections plus other departments: the hard case.
QUARTERLY = "# Report 2024\n" + "".join(
    f"## {q} 2024\n### Campaign Highlights\nThe campaign results this quarter were strong "
    f"with growth across digital channels and events.\n"
    for q in ("Q1", "Q2", "Q3", "Q4")
)
FILES = {
    "marketing/report_2024.md": QUARTERLY,
    "finance/summary.md": "# Finance\n## Ratios\nDays sales outstanding (DSO) was 45 days.",
    "general/handbook.md": "# Handbook\n## Leave\nEmployees get 12 sick leave days per year.",
    "hr/staff.csv": "id,full_name,department\nFINEMP1003,Krishna Malhotra,Business\n",
}
ALL = AccessScope("c_level", "C-Level", frozenset({"marketing", "finance", "general", "hr"}))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QdrantVectorStore]:
    for name, body in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body, encoding="utf-8")
    vector_store = QdrantVectorStore(QdrantClient(":memory:"), "test", DIM)
    run_ingestion(
        Settings(_env_file=None), vector_store, FakeEmbedder(), FakeSparseEmbedder(), tmp_path
    )
    yield vector_store
    vector_store.close()


def retriever(
    store: QdrantVectorStore, reranker: FakeReranker | None = None, **opts: bool
) -> SecureRetriever:
    return SecureRetriever(
        store,
        FakeEmbedder(),
        FakeSparseEmbedder(),
        reranker,
        RetrievalOptions(**opts),  # type: ignore[arg-type]
    )


def test_quarter_boost_ranks_the_asked_quarter_first(store: QdrantVectorStore) -> None:
    plain = retriever(store, hybrid=True, quarter_boost=False, rerank=False)
    boosted = retriever(store, hybrid=True, quarter_boost=True, rerank=False)
    question = "Which campaigns performed best in Q3 2024?"

    assert boosted.retrieve(question, ALL, limit=1)[0].chunk.quarter == "Q3"
    # Without the boost the four look-alike sections are a coin toss.
    assert {h.chunk.quarter for h in plain.retrieve(question, ALL, limit=4)} == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }


def test_quarter_boost_is_soft_other_content_still_found(store: QdrantVectorStore) -> None:
    hits = retriever(store, quarter_boost=True, rerank=False).retrieve(
        "Q3 campaigns and sick leave days", ALL, limit=10
    )
    assert "general" in {h.chunk.department for h in hits}  # not filtered out


def test_hybrid_finds_exact_identifiers(store: QdrantVectorStore) -> None:
    hits = retriever(store, hybrid=True, rerank=False).retrieve("FINEMP1003", ALL, limit=1)
    assert "FINEMP1003" in hits[0].chunk.content


def test_reranker_reorders_and_trims_to_limit(store: QdrantVectorStore) -> None:
    reranker = FakeReranker()
    hits = retriever(store, reranker, rerank=True).retrieve("sick leave days", ALL, limit=2)
    assert reranker.calls == 1
    assert len(hits) == 2
    assert hits[0].chunk.source_path == "general/handbook.md"
    assert hits[0].score >= hits[1].score


@pytest.mark.parametrize(
    "opts", [{}, {"hybrid": False}, {"quarter_boost": False}, {"rerank": False}]
)
def test_every_mode_respects_department_scope(
    store: QdrantVectorStore, opts: dict[str, bool]
) -> None:
    """Each extra candidate list (BM25, quarter-boosted) must carry the RBAC filter."""
    scope = AccessScope("employee", "Employee", frozenset({"general"}))
    hits = retriever(store, FakeReranker(), **opts).retrieve(
        "Q3 campaign DSO FINEMP1003 sick leave", scope, limit=20
    )
    assert hits and {h.chunk.department for h in hits} == {"general"}


def test_hard_quarter_filter_still_works(store: QdrantVectorStore) -> None:
    hits = retriever(store, rerank=False).retrieve("campaign results", ALL, limit=10, quarter="Q2")
    assert hits and {h.chunk.quarter for h in hits} == {"Q2"}


def test_rebuild_really_empties_the_index(store: QdrantVectorStore) -> None:
    assert store.count() > 0
    store.recreate()
    assert store.count() == 0


def test_outdated_index_schema_gives_clear_error() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(  # the Phase 1-4 layout: one unnamed dense vector
        "old", vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    )
    with pytest.raises(RuntimeError, match="--rebuild"):
        QdrantVectorStore(client, "old", DIM).ensure_collection()
