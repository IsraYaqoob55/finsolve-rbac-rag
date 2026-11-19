"""Structure-aware chunking for markdown documents.

Why not just cut every N characters? Because a fixed-size cut can split a
table in half, glue the end of "Q1 Cash Flow" to the start of "Q2 Overview",
and lose which section a sentence belongs to. Retrieval then returns
confusing fragments and citations become vague.

Our approach, in three steps:

1. **Sections.** Split the document at headings. Both heading styles in our
   data are supported: ``## Heading`` (ATX) and ``Heading:`` underlined with
   ``-----`` (setext). Each section remembers its heading path, e.g.
   ``["Q3 - July to September 2024", "Cash Flow Analysis"]``.
2. **Blocks.** Inside a section, split on blank lines into paragraphs, lists,
   tables and code blocks. A block is never cut unless it alone is too big.
3. **Packing.** Greedily pack blocks into chunks up to ``max_chars``, carrying
   the last block(s) into the next chunk as overlap so context isn't lost at
   the boundary. Oversized tables are split by rows, repeating the header row.

Finally each chunk's *embedded text* is prefixed with the document title and
section path ("contextual chunk headers"), so a chunk that just says
"Revenue: $2.1 billion" is still findable by "Q1 2024 revenue".
"""

import re
import uuid
from dataclasses import dataclass

from app.ingestion.metadata import detect_quarter
from app.ingestion.models import Chunk, Document

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-+)\s*$")
_HORIZONTAL_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_LIST_OR_QUOTE = re.compile(r"^\s*([-*+>]|\d+[.)])\s")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Namespace for deterministic chunk ids: same doc + same index => same UUID.
_CHUNK_NAMESPACE = uuid.UUID("6f1c6c2e-3d7a-4a5e-9b8f-2a1d0c9e7b41")


@dataclass(frozen=True)
class Section:
    """A run of text under one heading path."""

    path: tuple[str, ...]
    body: str


def _clean_heading(text: str) -> str:
    return text.replace("**", "").strip().rstrip(":").strip()


def split_sections(markdown: str) -> list[Section]:
    """Split markdown into sections at ATX (``#``) and setext (underlined) headings."""
    lines = markdown.splitlines()
    stack: list[tuple[int, str]] = []  # (level, heading) from outermost to innermost
    sections: list[Section] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body_lines = [ln for ln in buffer if not _HORIZONTAL_RULE.match(ln)]
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(Section(tuple(h for _, h in stack), body))
        buffer.clear()

    def push(level: int, heading: str) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, _clean_heading(heading)))

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code:
            atx = _ATX_HEADING.match(line)
            if atx:
                flush()
                push(len(atx.group(1)), atx.group(2))
                i += 1
                continue
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            is_setext = (
                line.strip()
                and _SETEXT_UNDERLINE.match(next_line)
                and not _TABLE_ROW.match(line)
                and not _LIST_OR_QUOTE.match(line)
            )
            if is_setext:
                flush()
                push(1 if next_line.strip().startswith("=") else 2, line)
                i += 2  # skip the heading line and its underline
                continue
        buffer.append(line)
        i += 1
    flush()
    return sections


def merge_small_sections(sections: list[Section], min_chars: int, max_chars: int) -> list[Section]:
    """Merge runs of small sibling sections into one section under their parent.

    Handbooks are full of tiny sections ("Severance Pay: as per policy."). Alone,
    each is a near-useless chunk; together under their parent ("Exit Policy")
    they form one coherent, well-sized chunk. Each merged member keeps its
    heading as a label so nothing is lost. A section's short intro text is also
    merged with its children (e.g. "4.1 Agile" intro + "4.1.1 Sprints").
    """
    groups: list[tuple[tuple[str, ...], list[Section]]] = []

    def size(members: list[Section]) -> int:
        return sum(len(m.body) + len(m.path[-1] if m.path else "") + 4 for m in members)

    for section in sections:
        parent = section.path[:-1]
        if groups and len(section.body) < min_chars and section.path:
            group_path, members = groups[-1]
            all_small = all(len(m.body) < min_chars for m in members)
            is_sibling = group_path == parent
            is_child_of_lone_intro = len(members) == 1 and members[0].path == parent
            if (
                all_small
                and (is_sibling or is_child_of_lone_intro)
                and (size([*members, section]) <= max_chars)
            ):
                members.append(section)
                if is_child_of_lone_intro:
                    groups[-1] = (members[0].path, members)
                continue
        groups.append((parent if section.path else (), [section]))

    merged: list[Section] = []
    for group_path, members in groups:
        if len(members) == 1:
            merged.append(members[0])  # nothing merged: keep its precise path
            continue
        parts = []
        for member in members:
            label = member.path[-1] if len(member.path) > len(group_path) else None
            parts.append(f"{label}:\n{member.body}" if label else member.body)
        merged.append(Section(group_path, "\n\n".join(parts)))
    return merged


def split_blocks(text: str) -> list[str]:
    """Split on blank lines, but never inside a fenced code block."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack_units(units: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join small units (each <= max_chars) into pieces <= max_chars."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{joiner}{unit}" if current else unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_sentences(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph at sentence ends; cut mid-sentence only as a last resort."""
    units: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        units.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
    return _pack_units(units, max_chars, " ")


def split_oversized_block(block: str, max_chars: int) -> list[str]:
    """Break one block that is bigger than max_chars into smaller pieces."""
    lines = block.splitlines()
    if len(lines) >= 3 and all(_TABLE_ROW.match(ln) for ln in lines[:2]):
        # Markdown table: keep header + separator on top of every piece.
        header = "\n".join(lines[:2])
        pieces: list[str] = []
        rows: list[str] = []
        for row in lines[2:]:
            if rows and len(header) + sum(len(r) + 1 for r in rows) + len(row) + 1 > max_chars:
                pieces.append("\n".join([header, *rows]))
                rows = []
            rows.append(row)
        if rows:
            pieces.append("\n".join([header, *rows]))
        return pieces
    # Lists and paragraphs: split between lines first (keeps list items whole),
    # and split any single over-long line at sentence boundaries.
    units: list[str] = []
    for line in lines:
        units.extend([line] if len(line) <= max_chars else _split_sentences(line, max_chars))
    return _pack_units(units, max_chars, "\n")


def pack_blocks(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack blocks into chunks of at most max_chars, with block-level overlap."""
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(
            [block] if len(block) <= max_chars else split_oversized_block(block, max_chars)
        )

    chunks: list[str] = []
    current: list[str] = []

    def size(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0)

    for piece in pieces:
        if current and size([*current, piece]) > max_chars:
            chunks.append("\n\n".join(current))
            # Overlap: carry trailing blocks (whole blocks only, so we never
            # repeat half a sentence) up to overlap_chars into the next chunk.
            tail: list[str] = []
            for previous in reversed(current):
                if size([previous, *tail]) > overlap_chars:
                    break
                tail.insert(0, previous)
            current = tail if size([*tail, piece]) <= max_chars else []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _context_header(title: str, section_path: list[str]) -> str:
    header = title
    if section_path:
        header += "\nSection: " + " > ".join(section_path)
    return header + "\n\n"


def chunk_id_for(doc_id: str, index: int) -> str:
    """Deterministic id so re-ingesting a document overwrites its old points."""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_id}#{index}"))


def chunk_document(
    document: Document, max_chars: int, overlap_chars: int, min_section_chars: int = 300
) -> list[Chunk]:
    """Turn one Document into retrieval-ready Chunks."""
    if document.doc_type == "csv_row":
        sections = [Section((), document.text)]
    else:
        sections = merge_small_sections(
            split_sections(document.text), min_section_chars, max_chars - 200
        )

    chunks: list[Chunk] = []
    for section in sections:
        # The H1 heading is usually the document title; don't repeat it in the path.
        path = [h for h in section.path if h != document.title]
        header = _context_header(document.title, path)
        budget = max(max_chars - len(header), 200)
        # A whole-document quarter (from the filename) wins; otherwise look at
        # the headings, which handles one report covering Q1..Q4 in sections.
        quarter = document.quarter or detect_quarter(*path)
        for content in pack_blocks(split_blocks(section.body), budget, overlap_chars):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document.doc_id, index),
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    department=document.department,
                    title=document.title,
                    doc_type=document.doc_type,
                    section_path=path,
                    chunk_index=index,
                    content=content,
                    text=header + content,
                    quarter=quarter,
                    year=document.year,
                    file_hash=document.file_hash,
                )
            )
    return chunks
