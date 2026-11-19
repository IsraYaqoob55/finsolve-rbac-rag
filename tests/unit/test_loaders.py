"""Tests for file loaders."""

from pathlib import Path

from app.ingestion.loaders import discover_documents, load_csv_rows, load_markdown


def test_markdown_loader_reads_title_department_and_quarter(tmp_path: Path) -> None:
    (tmp_path / "marketing").mkdir()
    path = tmp_path / "marketing" / "marketing_report_q2_2024.md"
    path.write_text("# Marketing Report - Q2 2024\n\nBody", encoding="utf-8")

    [doc] = load_markdown(path, tmp_path)

    assert doc.doc_id == "marketing/marketing_report_q2_2024.md"
    assert doc.department == "marketing"
    assert doc.title == "Marketing Report - Q2 2024"
    assert (doc.quarter, doc.year) == ("Q2", 2024)
    assert len(doc.file_hash) == 64


def test_csv_loader_creates_one_readable_document_per_row(tmp_path: Path) -> None:
    (tmp_path / "hr").mkdir()
    path = tmp_path / "hr" / "hr_data.csv"
    path.write_text(
        "employee_id,full_name,department\nE1,Asha Rao,Finance\nE2,Ravi Kumar,Sales\n",
        encoding="utf-8",
    )

    docs = load_csv_rows(path, tmp_path)

    assert [d.doc_id for d in docs] == ["hr/hr_data.csv#E1", "hr/hr_data.csv#E2"]
    assert docs[0].title == "Hr data: Asha Rao (E1)"
    assert "Full name: Asha Rao" in docs[0].text
    assert "Department: Finance" in docs[0].text
    assert all(d.department == "hr" for d in docs)


def test_discover_skips_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "general").mkdir()
    (tmp_path / "general" / "handbook.md").write_text("# H\ntext", encoding="utf-8")
    (tmp_path / "general" / "image.png").write_bytes(b"\x89PNG")

    grouped = discover_documents(tmp_path)

    assert list(grouped) == ["general/handbook.md"]
