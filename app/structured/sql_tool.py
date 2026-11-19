"""Text-to-SQL: answer analytical questions over structured data exactly.

Retrieval sees only the top few chunks, so it can't count, average or rank
across 100 employee rows. For those questions the LLM writes a SQL query,
the locked-down engine runs it, and the result becomes a *source* in the
normal answer flow (so citations, streaming and the sources panel just work,
and users can see the exact query that produced the numbers).

    question ─▶ LLM writes SQL (schema + rules) ─▶ SafeSQLDatabase
             └─ on error: send the error back, LLM fixes the query (one retry)
             ─▶ result table as a <source> ─▶ grounded answer
"""

import logging
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from app.ingestion.models import Chunk
from app.llm.base import LLMError, LLMProvider, Message
from app.rbac.policy import AccessScope
from app.retrieval.vector_store import SearchHit
from app.structured.sql_engine import (
    QueryResult,
    SafeSQLDatabase,
    SQLExecutionError,
    SQLSafetyError,
)

logger = logging.getLogger(__name__)

SQL_PROMPT = """You translate questions about company data into ONE SQLite query.

{schema}

Rules:
- Reply with only the SQL inside a ```sql code block.
- If the question cannot be answered from these tables (e.g. it asks about \
policies, procedures, documents or opinions), reply with exactly: NO_SQL
- A single read-only SELECT statement (WITH ... SELECT is fine). Never modify data.
- Use only the tables and columns listed. Match categorical values exactly as listed.
- Names are not unique: match them with lower(full_name) LIKE lower('%name%') and \
always return employee_id with full_name.
- For lists and rankings add ORDER BY and LIMIT 50. Round averages and percentages to 2 decimals.
- Give computed columns short readable aliases (e.g. avg_attendance_pct, employee_count)."""

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)
_NAMESPACE = uuid.UUID("2f0c1e9a-7b6d-4c1a-9e2f-5a8b3c4d6e70")


def extract_sql(reply: str) -> str | None:
    """SQL from a ```sql block (or the bare reply); None if the model said NO_SQL."""
    text = reply.strip()
    if not text or text.upper().startswith("NO_SQL"):
        return None
    match = _SQL_BLOCK.search(text)
    return (match.group(1) if match else text).strip() or None


@dataclass(frozen=True)
class StructuredAnswer:
    table: str
    sql: str
    result: QueryResult
    department: str
    source_path: str
    attempts: int


class SQLTool:
    def __init__(
        self,
        database: SafeSQLDatabase,
        llm: LLMProvider,
        table_departments: Mapping[str, str],
        table_sources: Mapping[str, str],
    ) -> None:
        """``table_departments``: table -> owning department (RBAC).
        ``table_sources``: table -> source file shown in citations (e.g. hr/hr_data.csv)."""
        self._db = database
        self._llm = llm
        self._departments = dict(table_departments)
        self._sources = dict(table_sources)

    def tables_for(self, scope: AccessScope) -> list[str]:
        """Tables this role may query. Empty -> the SQL path is never used."""
        return [t for t, dept in self._departments.items() if scope.can_access(dept)]

    def answer(self, question: str, scope: AccessScope) -> StructuredAnswer | None:
        """Run the question as SQL, or return None to fall back to document search."""
        tables = self.tables_for(scope)
        if not tables:
            return None  # RBAC: never even generate SQL for roles without access
        messages = [
            Message("system", SQL_PROMPT.format(schema=self._db.schema_prompt(tables))),
            Message("user", question),
        ]
        for attempt in (1, 2):
            try:
                reply = self._llm.complete(messages)
            except LLMError as exc:
                logger.warning("SQL generation failed", extra={"error": str(exc)})
                return None
            sql = extract_sql(reply)
            if sql is None:
                logger.info("sql_skipped", extra={"question": question})
                return None
            try:
                result = self._db.execute(sql, tables)
            except (SQLSafetyError, SQLExecutionError) as exc:
                logger.warning(
                    "sql_failed", extra={"attempt": attempt, "sql": sql, "error": str(exc)}
                )
                # Self-correction: show the model its query and the error once.
                messages += [
                    Message("assistant", f"```sql\n{sql}\n```"),
                    Message(
                        "user",
                        f"That query failed: {exc}. Reply with a corrected query "
                        "(same rules), or NO_SQL.",
                    ),
                ]
                continue
            table = next((t for t in tables if t in sql.lower()), tables[0])
            logger.info(
                "sql_answer",
                extra={
                    "role": scope.role,
                    "sql": sql,
                    "rows": len(result.rows),
                    "attempt": attempt,
                },
            )
            return StructuredAnswer(
                table=table,
                sql=sql,
                result=result,
                department=self._departments[table],
                source_path=self._sources[table],
                attempts=attempt,
            )
        return None


def as_search_hit(answer: StructuredAnswer) -> SearchHit:
    """Wrap a SQL result as a retrieval hit so the normal answer flow can cite it."""
    rows = len(answer.result.rows)
    content = (
        f"Computed with a live SQL query over the {answer.table} table.\n\n"
        f"```sql\n{answer.sql}\n```\n\n"
        f"Result ({rows} row{'s' if rows != 1 else ''}):\n{answer.result.to_markdown()}"
    )
    chunk = Chunk(
        chunk_id=str(uuid.uuid5(_NAMESPACE, answer.sql)),
        doc_id=f"sql:{answer.table}",
        source_path=answer.source_path,
        department=answer.department,
        title=f"{answer.table} table (SQL query)",
        doc_type="sql_result",
        section_path=["Computed result"],
        chunk_index=0,
        content=content,
        text=content,
        file_hash="",
    )
    return SearchHit(chunk=chunk, score=1.0)
