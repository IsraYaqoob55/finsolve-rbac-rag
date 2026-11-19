"""Tests for the structure-aware markdown chunker."""

from app.ingestion.chunkers import (
    chunk_document,
    merge_small_sections,
    pack_blocks,
    split_blocks,
    split_oversized_block,
    split_sections,
)
from app.ingestion.models import Document


def make_doc(text: str, title: str = "Report", quarter: str | None = None) -> Document:
    return Document(
        doc_id="finance/report.md",
        source_path="finance/report.md",
        department="finance",
        title=title,
        text=text,
        doc_type="markdown",
        file_hash="abc",
        quarter=quarter,
    )


# ---------- sections ----------


def test_atx_headings_build_a_nested_path() -> None:
    md = "# Report\n## Q1\n### Revenue\nUp 10%.\n## Q2\nFlat."
    sections = split_sections(md)
    assert [s.path for s in sections] == [("Report", "Q1", "Revenue"), ("Report", "Q2")]
    assert sections[0].body == "Up 10%."


def test_setext_headings_with_colon_are_detected() -> None:
    # The style used in financial_summary.md
    md = "# Report\n\nExecutive Summary:\n----------\nAll good.\n\nCash Flow:\n-----\nPositive."
    sections = split_sections(md)
    assert [s.path for s in sections] == [
        ("Report", "Executive Summary"),
        ("Report", "Cash Flow"),
    ]


def test_horizontal_rule_and_list_items_are_not_headings() -> None:
    md = "## A\nText.\n\n---\n\n- item\n---\nmore"
    sections = split_sections(md)
    assert len(sections) == 1
    assert "---" not in sections[0].body.splitlines()[1]  # rule removed
    assert "- item" in sections[0].body


def test_hash_inside_code_block_is_not_a_heading() -> None:
    md = "## Setup\n```bash\n# install deps\npip install x\n```"
    sections = split_sections(md)
    assert len(sections) == 1
    assert "# install deps" in sections[0].body


# ---------- small-section merging ----------


def test_small_sibling_sections_are_merged_under_parent() -> None:
    md = "## Exit Policy\n### Severance\n- As per law.\n### Exit Interview\n- Done by HR."
    merged = merge_small_sections(split_sections(md), min_chars=300, max_chars=1000)
    assert len(merged) == 1
    assert merged[0].path == ("Exit Policy",)
    assert "Severance:\n- As per law." in merged[0].body
    assert "Exit Interview:\n- Done by HR." in merged[0].body


def test_large_sections_are_not_merged() -> None:
    big = "x" * 400
    md = f"## Parent\n### A\n{big}\n### B\n{big}"
    merged = merge_small_sections(split_sections(md), min_chars=300, max_chars=1000)
    assert [s.path for s in merged] == [("Parent", "A"), ("Parent", "B")]


# ---------- blocks and packing ----------


def test_split_blocks_keeps_code_fences_with_blank_lines_together() -> None:
    blocks = split_blocks("para one\n\n```\nline1\n\nline2\n```\n\npara two")
    assert blocks == ["para one", "```\nline1\n\nline2\n```", "para two"]


def test_oversized_table_repeats_header_in_every_piece() -> None:
    header = "| Name | Value |\n|---|---|"
    rows = "\n".join(f"| row{i} | {i} |" for i in range(50))
    pieces = split_oversized_block(f"{header}\n{rows}", max_chars=200)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith(header)
        assert len(piece) <= 200


def test_long_paragraph_splits_at_sentence_boundaries() -> None:
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(40))
    pieces = split_oversized_block(paragraph, max_chars=150)
    assert all(len(p) <= 150 for p in pieces)
    assert all(p.endswith(".") for p in pieces)


def test_pack_blocks_respects_max_and_overlaps() -> None:
    blocks = [f"Block {i} " + "y" * 80 for i in range(6)]
    chunks = pack_blocks(blocks, max_chars=300, overlap_chars=100)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) > 1
    # The last block of chunk N is repeated at the start of chunk N+1.
    assert chunks[1].startswith(chunks[0].split("\n\n")[-1])


# ---------- full documents ----------


def test_chunks_get_context_header_and_section_quarter() -> None:
    md = (
        "# Quarterly Report\n"
        "## Q1 - Jan to Mar\n### Overview\nRevenue: $2.1B\n"
        "## Q3 - Jul to Sep\nRevenue: $2.4B"
    )
    chunks = chunk_document(make_doc(md, title="Quarterly Report"), 1500, 200, min_section_chars=0)
    assert [c.quarter for c in chunks] == ["Q1", "Q3"]
    assert chunks[0].section_path == ["Q1 - Jan to Mar", "Overview"]  # title not repeated
    assert chunks[0].text.startswith("Quarterly Report\nSection: Q1 - Jan to Mar > Overview")
    assert chunks[0].content == "Revenue: $2.1B"


def test_document_quarter_overrides_section_headings() -> None:
    # A Q2 report that mentions "Recommendations for Q3" is still a Q2 document.
    md = "# Q2 Report\n## Recommendations for Q3 2024\nDo more."
    chunks = chunk_document(make_doc(md, quarter="Q2"), 1500, 200)
    assert chunks[0].quarter == "Q2"


def test_chunk_ids_are_deterministic_and_unique() -> None:
    md = "## A\n" + "text " * 400 + "\n## B\nmore"
    first = chunk_document(make_doc(md), 500, 50)
    second = chunk_document(make_doc(md), 500, 50)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_no_chunk_exceeds_max_chars() -> None:
    md = "## Big\n" + "\n\n".join("word " * 120 for _ in range(10))
    chunks = chunk_document(make_doc(md), 800, 100)
    assert all(len(c.text) <= 800 for c in chunks)
