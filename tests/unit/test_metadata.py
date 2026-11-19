"""Tests for metadata helpers."""

from pathlib import Path

import pytest

from app.ingestion.metadata import department_from_path, detect_quarter, detect_year, humanise


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("marketing_report_q2_2024",), "Q2"),
        (("market_report_q4_2024",), "Q4"),
        (("Q3 - July to September 2024",), "Q3"),
        (("financial_summary", "Financial Report - 2024"), None),
        (("no quarter here", "but Q1 here"), "Q1"),
        (("Q5 is not a quarter",), None),
    ],
)
def test_detect_quarter(texts: tuple[str, ...], expected: str | None) -> None:
    assert detect_quarter(*texts) == expected


def test_detect_year() -> None:
    assert detect_year("marketing_report_q1_2024") == 2024
    assert detect_year("engineering_master_doc") is None


def test_department_from_path(tmp_path: Path) -> None:
    (tmp_path / "Finance").mkdir()
    assert department_from_path(tmp_path / "Finance" / "a.md", tmp_path) == "finance"
    assert department_from_path(tmp_path / "top_level.md", tmp_path) == "general"


def test_humanise() -> None:
    assert humanise("leave_balance") == "Leave balance"
