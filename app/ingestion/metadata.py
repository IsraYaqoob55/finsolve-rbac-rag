"""Small, pure helpers that derive metadata from paths and text.

Kept separate from loaders so they are trivial to unit-test.
"""

import hashlib
import re
from pathlib import Path

_QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _normalise(text: str) -> str:
    # "market_report_q4_2024" -> "market report q4 2024" so \b word boundaries work
    # (underscore counts as a word character in regex).
    return re.sub(r"[_\-.]+", " ", text)


def detect_quarter(*texts: str) -> str | None:
    """Return 'Q1'..'Q4' from the first text that mentions a quarter, else None."""
    for text in texts:
        match = _QUARTER_RE.search(_normalise(text))
        if match:
            return f"Q{match.group(1)}"
    return None


def detect_year(*texts: str) -> int | None:
    """Return the first 20xx year found in the given texts, else None."""
    for text in texts:
        match = _YEAR_RE.search(_normalise(text))
        if match:
            return int(match.group(1))
    return None


def department_from_path(path: Path, data_dir: Path) -> str:
    """The top-level folder under the data dir is the department.

    data/finance/report.md -> "finance". Files directly in data/ are "general".
    This convention means adding a department is just adding a folder.
    """
    relative = path.resolve().relative_to(data_dir.resolve())
    return relative.parts[0].lower() if len(relative.parts) > 1 else "general"


def relative_posix(path: Path, data_dir: Path) -> str:
    """Path relative to the data dir with forward slashes (same id on every OS)."""
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def file_sha256(path: Path) -> str:
    """Content hash of a file. Changes whenever a single byte changes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def humanise(name: str) -> str:
    """'leave_balance' -> 'Leave balance'."""
    return name.replace("_", " ").strip().capitalize()
