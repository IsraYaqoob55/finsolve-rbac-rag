"""Pure scoring functions for the answer evaluation (unit-tested, no LLM needed)."""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BEHAVIOURS = ("answered", "refused_access", "asked_clarification", "not_found")

# Typographic variants LLMs like to emit: dashes, and "×" in "5× ROI".
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "×": "x"})
_SPACES = re.compile(r"[   \s]+")


def normalize(text: str) -> str:
    """Lowercase; unify typographic dashes/spaces; drop thousands separators."""
    return _SPACES.sub(" ", text.translate(_DASHES)).replace(",", "").lower()


def contains_fact(answer: str, fact: str) -> bool:
    """True if any alternative in ``fact`` ("a|b") appears in the answer.

    Number boundaries are respected: "30 million" must not match inside
    "$130 million", and "12" must not match inside "120".
    """
    haystack = normalize(answer)
    for option in fact.split("|"):
        needle = re.escape(normalize(option.strip()))
        if re.search(rf"(?<![\d.]){needle}(?![\d])", haystack):
            return True
    return False


def missing_facts(answer: str, facts: Iterable[str]) -> list[str]:
    return [fact for fact in facts if not contains_fact(answer, fact)]


def leaked_canaries(
    answer: str, canaries: Mapping[str, Sequence[str]], allowed_departments: Iterable[str]
) -> list[str]:
    """Canary values from departments the role may NOT access that appear in the answer."""
    allowed = set(allowed_departments)
    text = normalize(answer)
    return [
        f"{department}:{value}"
        for department, values in canaries.items()
        if department not in allowed
        for value in values
        if normalize(value) in text
    ]


def validate_canaries(canaries: Mapping[str, Sequence[str]], data_dir: Path) -> list[str]:
    """Each canary must occur in exactly its own department's files, else it proves nothing."""
    texts: dict[str, str] = {}
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".csv"}:
            department = path.relative_to(data_dir).parts[0]
            texts[department] = texts.get(department, "") + normalize(
                path.read_text(encoding="utf-8")
            )
    problems = []
    for department, values in canaries.items():
        for value in values:
            found_in = sorted(d for d, text in texts.items() if normalize(value) in text)
            if found_in != [department]:
                problems.append(f"canary {value!r} ({department}) found in {found_in}")
    return problems


@dataclass(frozen=True)
class JudgeVerdict:
    behaviour: str
    faithful: bool
    unsupported_claims: list[str] = field(default_factory=list)
    relevance: int = 0
    parse_error: str | None = None


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge(reply: str) -> JudgeVerdict:
    """Parse the judge's JSON, tolerating code fences and surrounding text."""
    match = _JSON_OBJECT.search(reply or "")
    if not match:
        return JudgeVerdict("unknown", False, parse_error="no JSON object in judge reply")
    try:
        data: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return JudgeVerdict("unknown", False, parse_error=f"invalid JSON: {exc}")
    behaviour = str(data.get("behaviour", "unknown")).strip().lower()
    if behaviour not in BEHAVIOURS:
        behaviour = "unknown"
    claims = data.get("unsupported_claims") or []
    try:
        relevance = max(0, min(5, int(data.get("relevance", 0))))
    except (TypeError, ValueError):
        relevance = 0
    return JudgeVerdict(
        behaviour=behaviour,
        faithful=bool(data.get("faithful", False)),
        unsupported_claims=[str(c) for c in claims][:3] if isinstance(claims, list) else [],
        relevance=relevance,
    )


@dataclass
class CaseResult:
    id: str
    category: str
    role: str
    question: str
    answer: str
    route: str
    latency_ms: int
    cited_count: int
    source_departments: list[str]
    verdict: JudgeVerdict | None
    missing: list[str]
    leaks: list[str]
    checks: dict[str, bool]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.checks.values())

    @property
    def failure_reasons(self) -> list[str]:
        if self.error:
            return [f"error: {self.error}"]
        reasons = []
        if not self.checks.get("facts", True):
            reasons.append("missing " + ", ".join(self.missing))
        if not self.checks.get("behaviour", True):
            got = self.verdict.behaviour if self.verdict else "?"
            reasons.append(f"behaviour was {got}")
        if not self.checks.get("route", True):
            reasons.append(f"route was {self.route}")
        if not self.checks.get("no_leak", True):
            reasons.append("LEAK " + ", ".join(self.leaks))
        if not self.checks.get("in_scope_sources", True):
            reasons.append("out-of-scope source " + ", ".join(self.source_departments))
        return reasons


def score_case(
    case: Mapping[str, Any],
    answer: str,
    route: str,
    cited_count: int,
    source_departments: Sequence[str],
    allowed_departments: Iterable[str],
    canaries: Mapping[str, Sequence[str]],
    verdict: JudgeVerdict | None,
    latency_ms: int,
) -> CaseResult:
    allowed = set(allowed_departments)
    missing = missing_facts(answer, case.get("must_contain", []))
    leaks = leaked_canaries(answer, canaries, allowed)
    checks = {
        "facts": not missing,
        "no_leak": not leaks,
        "in_scope_sources": set(source_departments) <= allowed,
    }
    if "route" in case:
        checks["route"] = route == case["route"]
    if verdict is not None:
        checks["behaviour"] = verdict.behaviour in case["expect"]
    return CaseResult(
        id=case["id"],
        category=case.get("category", "other"),
        role=case["role"],
        question=case["question"],
        answer=answer,
        route=route,
        latency_ms=latency_ms,
        cited_count=cited_count,
        source_departments=sorted(set(source_departments)),
        verdict=verdict,
        missing=missing,
        leaks=leaks,
        checks=checks,
    )
