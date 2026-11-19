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
