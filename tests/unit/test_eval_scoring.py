"""Tests for the evaluation scoring logic (the evaluator must itself be correct)."""

from pathlib import Path

import pytest
import yaml

from app.core.config import PROJECT_ROOT
from eval.scoring import (
    JudgeVerdict,
    contains_fact,
    leaked_canaries,
    parse_judge,
    score_case,
    validate_canaries,
)

CANARIES = {"finance": ["325 million"], "hr": ["1491158", "finemp1001"]}


@pytest.mark.parametrize(
    ("answer", "fact", "expected"),
    [
        ("Net income was $300 Million [1].", "300 million", True),
        ("Salary: 1,491,158.23", "1491158", True),  # thousands separators ignored
        ("Revenue rose 22‑percent", "22-percent", True),  # typographic dash
        ("26 weeks of leave", "26 weeks", True),  # narrow no-break space
        ("We use ArgoCD", "jenkins|argocd", True),  # alternatives
        ("an estimated 5× ROI", "5x", True),  # multiplication sign
        ("We use Jenkins", "argocd", False),
        ("Q3 vendor costs were $130 million", "30 million", False),  # number boundary
        ("They pay 120 rupees", "12", False),
        ("12 days per year", "12", True),
        ("version 2.12 released", "12", False),
    ],
)
def test_contains_fact(answer: str, fact: str, expected: bool) -> None:
    assert contains_fact(answer, fact) is expected


def test_leaks_only_count_departments_outside_the_role() -> None:
    answer = "Q4 net income was $325 million; FINEMP1001 earns 1,491,158.23"
    assert leaked_canaries(answer, CANARIES, ["finance", "general"]) == [
        "hr:1491158",
        "hr:finemp1001",
    ]
    assert leaked_canaries(answer, CANARIES, ["finance", "hr"]) == []


def test_golden_canaries_are_unique_to_their_department() -> None:
    golden = yaml.safe_load((PROJECT_ROOT / "eval" / "answer_golden.yaml").read_text("utf-8"))
    assert validate_canaries(golden["canaries"], PROJECT_ROOT / "data") == []


def test_non_unique_canary_is_reported(tmp_path: Path) -> None:
    for dept in ("finance", "marketing"):
        (tmp_path / dept).mkdir()
        (tmp_path / dept / "doc.md").write_text("Latin America expansion", encoding="utf-8")
    problems = validate_canaries({"marketing": ["latin america"]}, tmp_path)
    assert problems == ["canary 'latin america' (marketing) found in ['finance', 'marketing']"]


@pytest.mark.parametrize(
    ("reply", "behaviour", "faithful"),
    [
        ('{"behaviour": "answered", "faithful": true, "relevance": 5}', "answered", True),
        ('```json\n{"behaviour": "Refused_Access", "faithful": true}\n```', "refused_access", True),
        ('Verdict: {"behaviour": "made_up", "faithful": false}', "unknown", False),
        ("not json at all", "unknown", False),
        ('{"behaviour": "answered", "faithful": tru', "unknown", False),
    ],
)
def test_parse_judge_is_tolerant(reply: str, behaviour: str, faithful: bool) -> None:
    verdict = parse_judge(reply)
    assert (verdict.behaviour, verdict.faithful) == (behaviour, faithful)


def test_parse_judge_clamps_relevance_and_limits_claims() -> None:
    verdict = parse_judge(
        '{"behaviour": "answered", "faithful": false, "relevance": 9, '
        '"unsupported_claims": ["a", "b", "c", "d"]}'
    )
    assert verdict.relevance == 5
    assert verdict.unsupported_claims == ["a", "b", "c"]


def _case(**extra: object) -> dict[str, object]:
    return {
        "id": "t",
        "role": "finance",
        "question": "q",
        "expect": ["answered"],
        "must_contain": ["300 million"],
        **extra,
    }


def test_score_case_passes_when_every_check_holds() -> None:
    result = score_case(
        _case(route="sql+docs"),
        "Net income was $300 million [1].",
        "sql+docs",
        1,
        ["finance"],
        ["finance", "general"],
        CANARIES,
        JudgeVerdict("answered", True, relevance=5),
        1200,
    )
    assert result.passed and result.failure_reasons == []


def test_score_case_reports_every_failure_reason() -> None:
    result = score_case(
        _case(route="sql+docs"),
        "FINEMP1001 earns 1491158",
        "docs",
        0,
        ["hr"],
        ["finance", "general"],
        CANARIES,
        JudgeVerdict("refused_access", True),
        900,
    )
    assert not result.passed
    reasons = " | ".join(result.failure_reasons)
    for expected in ("missing 300 million", "behaviour was refused_access", "route was docs"):
        assert expected in reasons
    assert "LEAK hr:1491158" in reasons and "out-of-scope source hr" in reasons
