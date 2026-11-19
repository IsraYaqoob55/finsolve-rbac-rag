"""Cheap, deterministic query understanding (no LLM call).

Finds the fiscal quarters a question talks about, so retrieval can favour
chunks from those quarters. Our quarterly reports share one template, so
without this "Q2 campaigns" and "Q3 campaigns" look almost identical to an
embedding model.
"""

import re

_QUARTER_PATTERNS = [
    (re.compile(r"\bq([1-4])\b", re.IGNORECASE), None),
    (re.compile(r"\bquarter\s*([1-4])\b", re.IGNORECASE), None),
    (re.compile(r"\b(first|1st)\s+quarter\b", re.IGNORECASE), "Q1"),
    (re.compile(r"\b(second|2nd)\s+quarter\b", re.IGNORECASE), "Q2"),
    (re.compile(r"\b(third|3rd)\s+quarter\b", re.IGNORECASE), "Q3"),
    (re.compile(r"\b(fourth|4th|last)\s+quarter\b", re.IGNORECASE), "Q4"),
]


def detect_quarters(question: str) -> tuple[str, ...]:
    """All quarters mentioned, in order of first appearance, e.g. ('Q2', 'Q3')."""
    found: list[tuple[int, str]] = []
    for pattern, fixed in _QUARTER_PATTERNS:
        for match in pattern.finditer(question):
            found.append((match.start(), fixed or f"Q{match.group(1)}"))
    ordered: list[str] = []
    for _, quarter in sorted(found):
        if quarter not in ordered:
            ordered.append(quarter)
    return tuple(ordered)


# Signals that a question needs computation over rows (counting, averaging,
# ranking, filtering lists) rather than reading a passage. Deliberately broad:
# a false positive only costs one LLM call, which answers NO_SQL and falls
# back to document search.
_ANALYTICAL = re.compile(
    r"\b("
    r"average|avg|mean|median|how many|number of|count|total|sum of|"
    r"list (?:all|the|of)|top \d+|bottom \d+|highest|lowest|most|least|maximum|minimum|"
    r"rank|ranking|distribution|breakdown|percentage of|"
    r"(?:by|per|each|every|across) (?:department|location|city|role|rating|team|office)|"
    r"which employees|who (?:has|have|are|joined|earns?|works?)|"
    r"employees (?:in|with|who|whose|from|at|joined)|"
    r"joined (?:in|after|before|between)|reports? to|headcount|older than|younger than"
    r")\b",
    re.IGNORECASE,
)


def looks_analytical(question: str) -> bool:
    """True if the question probably needs SQL over structured data."""
    return bool(_ANALYTICAL.search(question))
