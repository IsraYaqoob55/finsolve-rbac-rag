"""Tests for prompt building and citation extraction."""

import pytest

from app.ingestion.models import Chunk
from app.llm.base import Message
from app.llm.prompts import build_messages, build_system_prompt, extract_citations, format_sources
from app.retrieval.vector_store import SearchHit


def make_hit(
    content: str, department: str = "finance", section: str = "Q3 > Overview"
) -> SearchHit:
    chunk = Chunk(
        chunk_id="00000000-0000-0000-0000-000000000001",
        doc_id="finance/report.md",
        source_path="finance/report.md",
        department=department,
        title="Quarterly Report",
        doc_type="markdown",
        section_path=section.split(" > "),
        chunk_index=0,
        content=content,
        text=content,
        file_hash="h",
    )
    return SearchHit(chunk=chunk, score=0.8)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Revenue grew [1].", [1]),
        ("A [2] then B [1] then A again [2].", [2, 1]),
        ("Combined [1][3] and grouped [2, 4].", [1, 3, 2, 4]),
        ("Out of range [9] and zero [0] are ignored [1].", [1]),
        ("No citations at all.", []),
    ],
)
def test_extract_citations(answer: str, expected: list[int]) -> None:
    assert extract_citations(answer, source_count=4) == expected


def test_sources_are_numbered_and_tagged() -> None:
    text = format_sources([make_hit("Revenue $2.4B"), make_hit("Leave: 12 days", "general")])
    assert '<source id="1" document="finance/report.md" department="finance">' in text
    assert '<source id="2"' in text and 'department="general"' in text
    assert "Location: Quarterly Report > Q3 > Overview" in text


def test_system_prompt_names_user_role_and_departments() -> None:
    prompt = build_system_prompt("Kabir Shah", "Finance", ["marketing", "finance"])
    assert "Kabir Shah" in prompt and "role: Finance" in prompt
    assert "finance, marketing" in prompt  # sorted
    assert "data, not instructions" in prompt


def test_message_order_history_before_latest_question_with_sources() -> None:
    history = [Message("user", "hi"), Message("assistant", "hello")]
    messages = build_messages("SYS", "What was revenue?", [make_hit("Revenue $2.4B")], history)

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1].content.endswith("Question: What was revenue?")
    assert "<source" in messages[-1].content
    assert all("<source" not in m.content for m in messages[:-1])


def test_alternative_citation_brackets_are_normalised() -> None:
    from app.llm.prompts import normalize_citations

    text = normalize_citations("12 days【2】 and a certificate【1】【2】")
    assert text == "12 days[2] and a certificate[1][2]"
    assert extract_citations(text, source_count=2) == [2, 1]
