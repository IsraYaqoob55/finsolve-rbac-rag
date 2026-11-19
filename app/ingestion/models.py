"""Data models that flow through the ingestion pipeline.

    file on disk ──load──▶ Document ──chunk──▶ Chunk(s) ──embed──▶ vector store

A *Document* is one logical source (a markdown file, or one CSV row).
A *Chunk* is a retrievable piece of a document plus all the metadata we
need later: department (for RBAC filtering), section path (for citations),
quarter/year (for time-aware filtering) and file hash (for incremental updates).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal["markdown", "csv_row", "sql_result"]


class Document(BaseModel):
    """One logical source document, before chunking."""

    doc_id: str = Field(description="Stable unique id, e.g. 'finance/financial_summary.md'")
    source_path: str = Field(description="File path relative to the data dir (posix style)")
    department: str = Field(description="Owning department; drives access control")
    title: str
    text: str
    doc_type: DocType
    file_hash: str = Field(description="SHA-256 of the source file, for change detection")
    quarter: str | None = None  # "Q1".."Q4" when the whole document is about one quarter
    year: int | None = None


class Chunk(BaseModel):
    """A retrievable piece of a document, stored as one point in the vector DB."""

    chunk_id: str  # deterministic UUID, so re-ingesting overwrites instead of duplicating
    doc_id: str
    source_path: str
    department: str
    title: str
    doc_type: DocType
    section_path: list[str] = Field(default_factory=list)
    chunk_index: int
    content: str = Field(description="The raw chunk text, shown to users and the LLM")
    text: str = Field(description="What gets embedded: content plus title/section context")
    quarter: str | None = None
    year: int | None = None
    file_hash: str

    @property
    def section(self) -> str:
        """Human-readable section path, e.g. 'Q3 - July to September 2024 > Cash Flow'."""
        return " > ".join(self.section_path)

    def to_payload(self) -> dict[str, Any]:
        """Everything except the id, as a JSON-serialisable dict for the vector store."""
        return self.model_dump(exclude={"chunk_id"})
