"""Tests for text-to-SQL generation, self-correction, routing and RBAC."""

from collections.abc import Iterator

import pytest

from app.core.config import PROJECT_ROOT
from app.llm.base import LLMError
from app.nlp.query_analysis import looks_analytical
from app.rbac.policy import AccessScope
from app.structured.datasets import load_datasets
from app.structured.sql_engine import SafeSQLDatabase
from app.structured.sql_tool import SQLTool, as_search_hit, extract_sql
from tests.fakes import FakeLLM

HR = AccessScope("hr", "Human Resources", frozenset({"hr", "general"}))
MARKETING = AccessScope("marketing", "Marketing", frozenset({"marketing", "general"}))
GOOD_SQL = (
    "```sql\nSELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department\n```"
)


@pytest.fixture(scope="module")
def db() -> Iterator[SafeSQLDatabase]:
    database = SafeSQLDatabase(load_datasets(PROJECT_ROOT / "config/datasets.yaml"), PROJECT_ROOT)
    yield database
    database.close()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def tool(db: SafeSQLDatabase, llm: FakeLLM) -> SQLTool:
    return SQLTool(db, llm, {"employees": "hr"}, {"employees": "hr/hr_data.csv"})


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            GOOD_SQL,
            "SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department",
        ),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 2", "SELECT 2"),
        ("NO_SQL", None),
        ("no_sql - this is about policies", None),
        ("", None),
    ],
)
def test_extract_sql(reply: str, expected: str | None) -> None:
    assert extract_sql(reply) == expected


def test_answers_with_sql_on_first_try(tool: SQLTool, llm: FakeLLM) -> None:
    llm.script = [GOOD_SQL]
    answer = tool.answer("How many employees are in each department?", HR)

    assert answer is not None and answer.attempts == 1
    assert answer.result.columns == ["department", "employee_count"]
    assert sum(r[1] for r in answer.result.rows) == 100  # type: ignore[misc]
    system_prompt = llm.calls[0][0].content
    assert "TABLE employees" in system_prompt and "NO_SQL" in system_prompt


def test_self_corrects_after_an_error(tool: SQLTool, llm: FakeLLM) -> None:
    llm.script = ["```sql\nSELECT dept, COUNT(*) FROM employees GROUP BY dept\n```", GOOD_SQL]
    answer = tool.answer("Headcount per department?", HR)

    assert answer is not None and answer.attempts == 2
    feedback = llm.calls[1][-1].content
    assert "failed" in feedback and "no such column: dept" in feedback


def test_unsafe_query_is_not_executed_and_gives_up_after_retry(tool: SQLTool, llm: FakeLLM) -> None:
    llm.script = ["DROP TABLE employees", "DELETE FROM employees"]
    assert tool.answer("Remove everyone", HR) is None
    assert len(llm.calls) == 2


def test_no_sql_reply_falls_back(tool: SQLTool, llm: FakeLLM) -> None:
    llm.script = ["NO_SQL"]
    assert tool.answer("What is the leave policy?", HR) is None


def test_llm_failure_falls_back(tool: SQLTool, llm: FakeLLM) -> None:
    llm.error = LLMError("timeout")
    assert tool.answer("How many employees?", HR) is None


def test_roles_without_access_never_reach_the_llm(tool: SQLTool, llm: FakeLLM) -> None:
    assert tool.tables_for(MARKETING) == []
    assert tool.answer("What is the average salary by department?", MARKETING) is None
    assert llm.calls == []  # the schema was never even shown to the model


def test_result_becomes_a_citable_source(tool: SQLTool, llm: FakeLLM) -> None:
    llm.script = [GOOD_SQL]
    answer = tool.answer("Headcount per department?", HR)
    assert answer is not None

    hit = as_search_hit(answer)

    assert hit.chunk.department == "hr"
    assert hit.chunk.source_path == "hr/hr_data.csv"
    assert hit.chunk.doc_type == "sql_result"
    assert "(SQL query)" in hit.chunk.title
    assert "```sql" in hit.chunk.content and "| department | employee_count |" in hit.chunk.content


@pytest.mark.parametrize(
    "question",
    [
        "What is the average attendance by department?",
        "How many employees have a rating of 5?",
        "Top 5 highest paid employees in Finance",
        "Which employees joined in 2024?",
        "List all employees in Pune",
        "Who reports to FINEMP1006?",
        "headcount per location",
    ],
)
def test_analytical_questions_are_detected(question: str) -> None:
    assert looks_analytical(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is the leave policy?",
        "How does our CI/CD deployment work?",
        "What is Aadhya Patel's performance rating?",
        "Summarise Q3 2024 marketing results",
    ],
)
def test_document_questions_are_not_routed_to_sql(question: str) -> None:
    assert not looks_analytical(question)
