"""Loaders turn files on disk into ``Document`` objects.

Each file type has one loader function, registered in ``LOADERS`` by file
extension. Supporting a new format (PDF, DOCX...) means writing one function
and adding one line to the registry; nothing else in the pipeline changes.
"""

import csv
import logging
from collections.abc import Callable
from pathlib import Path

from app.ingestion.metadata import (
    department_from_path,
    detect_quarter,
    detect_year,
    file_sha256,
    humanise,
    relative_posix,
)
from app.ingestion.models import Document

logger = logging.getLogger(__name__)

Loader = Callable[[Path, Path], list[Document]]


def _markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_markdown(path: Path, data_dir: Path) -> list[Document]:
    """Load a markdown file as a single Document."""
    text = path.read_text(encoding="utf-8")
    source = relative_posix(path, data_dir)
    title = _markdown_title(text, fallback=humanise(path.stem))
    return [
        Document(
            doc_id=source,
            source_path=source,
            department=department_from_path(path, data_dir),
            title=title,
            text=text,
            doc_type="markdown",
            file_hash=file_sha256(path),
            # Filename/title decide the quarter of the *whole* document, e.g.
            # marketing_report_q2_2024.md -> Q2. Section-level quarters are
            # handled by the chunker for multi-quarter reports.
            quarter=detect_quarter(path.stem, title),
            year=detect_year(path.stem, title),
        )
    ]


def load_csv_rows(path: Path, data_dir: Path) -> list[Document]:
    """Load each CSV row as its own Document, rendered as readable text.

    Embedding models understand "Department: Finance" far better than a raw
    line like "FINEMP1001,Isha Chowdhury,...,Finance,...". The first column is
    used as the row id; a ``full_name``/``name`` column, if present, goes in the title.
    """
    source = relative_posix(path, data_dir)
    department = department_from_path(path, data_dir)
    file_hash = file_sha256(path)
    label = humanise(path.stem)

    documents: list[Document] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        id_field = reader.fieldnames[0]
        for row in reader:
            row_id = (row.get(id_field) or "").strip()
            if not row_id:
                continue
            name = row.get("full_name") or row.get("name")
            title = f"{label}: {name} ({row_id})" if name else f"{label}: {row_id}"
            body = "\n".join(f"{humanise(k)}: {v}" for k, v in row.items() if k)
            documents.append(
                Document(
                    doc_id=f"{source}#{row_id}",
                    source_path=source,
                    department=department,
                    title=title,
                    text=body,
                    doc_type="csv_row",
                    file_hash=file_hash,
                )
            )
    return documents


LOADERS: dict[str, Loader] = {
    ".md": load_markdown,
    ".csv": load_csv_rows,
}


def discover_documents(data_dir: Path) -> dict[str, list[Document]]:
    """Load every supported file under ``data_dir``.

    Returns documents grouped by source file, because change detection and
    deletion happen per file.
    """
    grouped: dict[str, list[Document]] = {}
    for path in sorted(data_dir.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if not path.is_file() or loader is None:
            continue
        documents = loader(path, data_dir)
        if documents:
            grouped[relative_posix(path, data_dir)] = documents
            logger.debug("Loaded %s (%d documents)", path.name, len(documents))
    return grouped
