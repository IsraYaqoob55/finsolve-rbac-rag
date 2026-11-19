"""A locked-down, read-only SQL engine for LLM-generated queries.

Running SQL written by a language model is risky, so safety is layered and
enforced by the database itself, not by inspecting query text:

1. Data lives in a private **in-memory** SQLite copy built from the CSV. No
   query can reach the real file, the app's user database, or the disk.
2. A SQLite **authorizer** callback approves every operation while a query is
   compiled: only SELECT, reading the allowed tables, and allow-listed
   functions pass. INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA, system tables and
   unknown functions (e.g. ``randomblob`` for memory exhaustion) are refused.
3. ``PRAGMA query_only`` makes the connection read-only as a second barrier.
4. One statement per call, a **time limit** (progress handler), and a **row cap**.
"""

import csv
import logging
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.structured.datasets import DatasetSpec

logger = logging.getLogger(__name__)

# Functions an analytical query legitimately needs. Anything else is denied.
ALLOWED_FUNCTIONS = frozenset(
    {
        "count", "sum", "avg", "min", "max", "total", "group_concat",
        "round", "abs", "lower", "upper", "length", "substr", "substring", "trim",
        "ltrim", "rtrim", "replace", "instr", "coalesce", "ifnull", "nullif", "iif",
        "like", "glob", "printf", "format", "cast", "typeof",
        "date", "time", "datetime", "julianday", "strftime", "unixepoch",
        "row_number", "rank", "dense_rank", "percent_rank", "ntile", "lag", "lead",
        "first_value", "last_value",
    }
)  # fmt: skip

_SQL_TYPES = {"TEXT": "TEXT", "INTEGER": "INTEGER", "REAL": "REAL", "DATE": "TEXT"}


class SQLSafetyError(Exception):
    """The query was rejected before or during compilation (not allowed)."""


class SQLExecutionError(Exception):
    """The query was allowed but failed (syntax error, unknown column, timeout...)."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[object, ...]]
    truncated: bool = False  # more rows existed than the cap
    elapsed_ms: int = 0

    def to_markdown(self, max_rows: int = 50) -> str:
        """Render as a markdown table (what the LLM reads, and users see as the source)."""
        if not self.rows:
            return "(no rows)"
        header = "| " + " | ".join(self.columns) + " |"
        divider = "|" + "---|" * len(self.columns)
        body = [
            "| " + " | ".join(_fmt(value) for value in row) + " |" for row in self.rows[:max_rows]
        ]
        extra = len(self.rows) - max_rows
        footer = [f"... {extra} more rows"] if extra > 0 else []
        if self.truncated:
            footer.append("(result truncated by the row limit)")
        return "\n".join([header, divider, *body, *footer])


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value).replace("|", "/")


def _convert(raw: str, column_type: str) -> object:
    if raw == "":
        return None
    if column_type == "INTEGER":
        return int(float(raw))
    if column_type == "REAL":
        return float(raw)
    return raw


def basic_checks(sql: str) -> str:
    """Normalise and reject obviously wrong input before it reaches SQLite."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SQLSafetyError("empty query")
    first_word = cleaned.split(None, 1)[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        raise SQLSafetyError(f"only SELECT queries are allowed, got {first_word}")
    if ";" in cleaned:
        raise SQLSafetyError("only one statement is allowed")
    return cleaned


@dataclass
class _Guard:
    """Authorizer state for the query currently being compiled."""

    allowed_tables: frozenset[str] = frozenset()
    denied: list[str] = field(default_factory=list)


class SafeSQLDatabase:
    def __init__(
        self,
        datasets: dict[str, DatasetSpec],
        base_dir: Path,
        max_rows: int = 200,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._datasets = datasets
        self._max_rows = max_rows
        self._timeout = timeout_seconds
        self._lock = threading.Lock()  # one query at a time on the shared connection
        self._guard = _Guard()
        self._deadline = 0.0

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        for table, spec in datasets.items():
            self._load(table, spec, base_dir)
        # Lock it down only after loading: from now on every query is checked.
        self._conn.execute("PRAGMA query_only = ON")
        self._conn.set_authorizer(self._authorize)
        self._conn.set_progress_handler(self._check_deadline, 1000)

    # ---------- loading ----------

    def _load(self, table: str, spec: DatasetSpec, base_dir: Path) -> None:
        columns = list(spec.columns)
        ddl = ", ".join(f"{c} {_SQL_TYPES[spec.columns[c].type]}" for c in columns)
        self._conn.execute(f"CREATE TABLE {table} ({ddl})")  # names validated in datasets.py
        path = spec.source if spec.source.is_absolute() else base_dir / spec.source
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [
                tuple(_convert(row.get(c, ""), spec.columns[c].type) for c in columns)
                for row in csv.DictReader(handle)
            ]
        placeholders = ", ".join("?" for _ in columns)
        self._conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)  # noqa: S608
        self._conn.commit()
        logger.info("Loaded table %s (%d rows)", table, len(rows))

    # ---------- guards ----------

    def _authorize(
        self, action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None
    ) -> int:
        guard = self._guard
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 in guard.allowed_tables:
                return sqlite3.SQLITE_OK
            guard.denied.append(f"read of table {arg1!r}")
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            if arg2 and arg2.lower() in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            guard.denied.append(f"function {arg2!r}")
            return sqlite3.SQLITE_DENY
        guard.denied.append(f"operation code {action}")
        return sqlite3.SQLITE_DENY

    def _check_deadline(self) -> int:
        return 1 if time.monotonic() > self._deadline else 0  # non-zero aborts the query

    # ---------- public API ----------

    @property
    def tables(self) -> list[str]:
        return list(self._datasets)

    def execute(self, sql: str, allowed_tables: Sequence[str]) -> QueryResult:
        cleaned = basic_checks(sql)
        with self._lock:
            self._guard = _Guard(allowed_tables=frozenset(allowed_tables))
            self._deadline = time.monotonic() + self._timeout
            started = time.perf_counter()
            try:
                cursor = self._conn.execute(cleaned)
                rows = cursor.fetchmany(self._max_rows + 1)
            except sqlite3.DatabaseError as exc:
                if self._guard.denied:
                    raise SQLSafetyError("not allowed: " + ", ".join(self._guard.denied)) from exc
                if "interrupted" in str(exc).lower():
                    raise SQLExecutionError(f"query exceeded {self._timeout}s time limit") from exc
                raise SQLExecutionError(str(exc)) from exc
            except sqlite3.Warning as exc:  # e.g. "You can only execute one statement at a time"
                raise SQLSafetyError(str(exc)) from exc
            columns = [d[0] for d in cursor.description or []]
        elapsed = int((time.perf_counter() - started) * 1000)
        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows[: self._max_rows]],
            truncated=len(rows) > self._max_rows,
            elapsed_ms=elapsed,
        )

    def schema_prompt(self, tables: Sequence[str]) -> str:
        """Describe tables for the LLM: columns, meanings, categorical values, samples."""
        parts = []
        for table in tables:
            spec = self._datasets[table]
            lines = [f"TABLE {table} -- {spec.description}"]
            for name, col in spec.columns.items():
                line = f"  {name} {col.type}: {col.description}"
                if col.categorical:
                    values = self._distinct(table, name)
                    line += f" | values: {', '.join(map(str, values))}"
                lines.append(line)
            lines.append("Sample rows:")
            lines.append(self._sample(table))
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _internal(self, sql: str, table: str) -> QueryResult:
        return self.execute(sql, [table])

    def _distinct(self, table: str, column: str) -> list[object]:
        result = self._internal(f"SELECT DISTINCT {column} FROM {table} ORDER BY 1", table)  # noqa: S608
        return [row[0] for row in result.rows]

    def _sample(self, table: str) -> str:
        return self._internal(f"SELECT * FROM {table} LIMIT 3", table).to_markdown()  # noqa: S608

    def close(self) -> None:
        self._conn.close()
