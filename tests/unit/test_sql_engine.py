"""Security and behaviour tests for the locked-down SQL engine (real HR data)."""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.structured.datasets import load_datasets
from app.structured.sql_engine import (
    SafeSQLDatabase,
    SQLExecutionError,
    SQLSafetyError,
    basic_checks,
)

TABLES = ["employees"]


@pytest.fixture(scope="module")
def db() -> Iterator[SafeSQLDatabase]:
    datasets = load_datasets(PROJECT_ROOT / "config" / "datasets.yaml")
    database = SafeSQLDatabase(datasets, PROJECT_ROOT, max_rows=50, timeout_seconds=0.5)
    yield database
    database.close()


def test_loads_all_rows_with_proper_types(db: SafeSQLDatabase) -> None:
    result = db.execute(
        "SELECT COUNT(*), typeof(salary), typeof(performance_rating) FROM employees", TABLES
    )
    assert result.rows == [(100, "real", "integer")]


def test_aggregates_and_self_join_work(db: SafeSQLDatabase) -> None:
    by_dept = db.execute(
        "SELECT department, COUNT(*) n FROM employees GROUP BY department ORDER BY n DESC",
        TABLES,
    )
    assert by_dept.columns == ["department", "n"]
    assert sum(row[1] for row in by_dept.rows) == 100  # type: ignore[misc]

    managers = db.execute(
        "SELECT e.employee_id, m.full_name FROM employees e "
        "JOIN employees m ON e.manager_id = m.employee_id LIMIT 3",
        TABLES,
    )
    assert len(managers.rows) == 3


ATTACKS = {
    "drop table": "DROP TABLE employees",
    "delete rows": "DELETE FROM employees",
    "update hidden behind WITH": "WITH x AS (SELECT 1) UPDATE employees SET salary = 0",
    "insert via WITH": "WITH x AS (SELECT 1) INSERT INTO employees (employee_id) VALUES ('X')",
    "second statement": "SELECT 1; DROP TABLE employees",
    "attach a file": "ATTACH DATABASE 'stolen.db' AS s",
    "pragma": "PRAGMA writable_schema = 1",
    "read schema table": "SELECT sql FROM sqlite_master",
    "pragma table function": "SELECT * FROM pragma_table_info('employees')",
    "memory bomb": "SELECT randomblob(1000000000)",
    "zeroblob bomb": "SELECT zeroblob(1000000000)",
    "load extension": "SELECT load_extension('evil.dll')",
    "recursive cte": (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT * FROM c"
    ),
    "create table": "CREATE TABLE t (x)",
    "empty": "   ",
}


@pytest.mark.parametrize("sql", ATTACKS.values(), ids=ATTACKS.keys())
def test_attacks_are_rejected(db: SafeSQLDatabase, sql: str) -> None:
    with pytest.raises(SQLSafetyError):
        db.execute(sql, TABLES)


def test_data_is_untouched_after_attacks(db: SafeSQLDatabase) -> None:
    for sql in ATTACKS.values():
        with contextlib.suppress(SQLSafetyError, SQLExecutionError):
            db.execute(sql, TABLES)
    result = db.execute("SELECT COUNT(*), MIN(salary) > 0 FROM employees", TABLES)
    assert result.rows == [(100, 1)]


def test_table_not_in_allowed_list_is_denied(db: SafeSQLDatabase) -> None:
    with pytest.raises(SQLSafetyError, match="employees"):
        db.execute("SELECT * FROM employees", allowed_tables=[])


def test_slow_query_is_stopped_by_time_limit(db: SafeSQLDatabase) -> None:
    cartesian = "SELECT COUNT(*) FROM employees a, employees b, employees c, employees d"
    with pytest.raises(SQLExecutionError, match="time limit"):
        db.execute(cartesian, TABLES)


def test_row_cap_truncates(db: SafeSQLDatabase) -> None:
    result = db.execute("SELECT employee_id FROM employees", TABLES)
    assert len(result.rows) == 50 and result.truncated


def test_syntax_error_is_an_execution_error_not_a_crash(db: SafeSQLDatabase) -> None:
    with pytest.raises(SQLExecutionError, match="no such column"):
        db.execute("SELECT salaryy FROM employees", TABLES)


def test_schema_prompt_lists_exact_categorical_values(db: SafeSQLDatabase) -> None:
    prompt = db.schema_prompt(TABLES)
    assert "TABLE employees" in prompt
    assert "values: Business, Compliance, Data, Design, Finance" in prompt
    assert "Sample rows:" in prompt


def test_markdown_rendering() -> None:
    from app.structured.sql_engine import QueryResult

    table = QueryResult(["dept", "avg"], [("Finance", 91.234), ("HR|Ops", 88.0)]).to_markdown()
    assert table.splitlines() == [
        "| dept | avg |",
        "|---|---|",
        "| Finance | 91.23 |",
        "| HR/Ops | 88.00 |",
    ]


def test_basic_checks_strip_trailing_semicolon() -> None:
    assert basic_checks("  SELECT 1 ;  ") == "SELECT 1"


def test_invalid_identifiers_in_config_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "datasets.yaml"
    bad.write_text(
        "datasets:\n  t:\n    department: hr\n    source: x.csv\n    description: d\n"
        "    columns: {'name; DROP TABLE x': {type: TEXT}}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid column names"):
        load_datasets(bad)
