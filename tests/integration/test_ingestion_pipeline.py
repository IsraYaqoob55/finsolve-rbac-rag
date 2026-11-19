"""End-to-end ingestion tests with an in-memory Qdrant and a fake embedder.

No model download, no disk storage: these run in milliseconds in CI.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.ingestion.pipeline import IngestionReport, run_ingestion
from app.retrieval.vector_store import QdrantVectorStore
from tests.fakes import DIM, FakeEmbedder, FakeSparseEmbedder


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for dept, name, body in [
        ("finance", "report_q3_2024.md", "# Finance Q3\n## Overview\nNet income was 300 million."),
        ("general", "handbook.md", "# Handbook\n## Leave\nEmployees get 12 sick leave days."),
        ("hr", "staff.csv", "id,full_name,department\nE1,Asha Rao,Finance\n"),
    ]:
        (tmp_path / dept).mkdir()
        (tmp_path / dept / name).write_text(body, encoding="utf-8")
    return tmp_path


@pytest.fixture
def store() -> Iterator[QdrantVectorStore]:
    vector_store = QdrantVectorStore(QdrantClient(":memory:"), "test", DIM)
    yield vector_store
    vector_store.close()


def ingest(data_dir: Path, store: QdrantVectorStore, rebuild: bool = False) -> IngestionReport:
    return run_ingestion(
        Settings(_env_file=None), store, FakeEmbedder(), FakeSparseEmbedder(), data_dir, rebuild
    )


def test_first_run_indexes_every_file(data_dir: Path, store: QdrantVectorStore) -> None:
    report = ingest(data_dir, store)

    assert sorted(report.added) == [
        "finance/report_q3_2024.md",
        "general/handbook.md",
        "hr/staff.csv",
    ]
    assert store.count() == 3


def test_second_run_skips_unchanged_files(data_dir: Path, store: QdrantVectorStore) -> None:
    ingest(data_dir, store)
    report = ingest(data_dir, store)

    assert report.added == [] and report.updated == []
    assert len(report.unchanged) == 3
    assert report.chunks_written == 0


def test_changed_file_is_reindexed_without_stale_chunks(
    data_dir: Path, store: QdrantVectorStore
) -> None:
    ingest(data_dir, store)
    (data_dir / "general" / "handbook.md").write_text(
        "# Handbook\n## Leave\nEmployees get 15 sick leave days.", encoding="utf-8"
    )

    report = ingest(data_dir, store)

    assert report.updated == ["general/handbook.md"]
    assert store.count() == 3
    hits = store.search(FakeEmbedder().embed_query("sick leave days"), departments=["general"])
    assert "15 sick leave days" in hits[0].chunk.content


def test_deleted_file_is_removed_from_index(data_dir: Path, store: QdrantVectorStore) -> None:
    ingest(data_dir, store)
    (data_dir / "hr" / "staff.csv").unlink()

    report = ingest(data_dir, store)

    assert report.deleted == ["hr/staff.csv"]
    assert store.count() == 2


def test_department_filter_never_returns_other_departments(
    data_dir: Path, store: QdrantVectorStore
) -> None:
    ingest(data_dir, store)
    query = FakeEmbedder().embed_query("net income finance Q3")

    hits = store.search(query, limit=10, departments=["general"])

    assert hits, "expected the general document to be returned"
    assert {h.chunk.department for h in hits} == {"general"}


def test_quarter_filter(data_dir: Path, store: QdrantVectorStore) -> None:
    ingest(data_dir, store)

    hits = store.search(FakeEmbedder().embed_query("net income"), limit=10, quarter="Q3")

    assert [h.chunk.source_path for h in hits] == ["finance/report_q3_2024.md"]
