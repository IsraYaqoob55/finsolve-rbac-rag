"""Structured (SQL) answers through the chat API, with RBAC."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import auth_header
from tests.fakes import FakeLLM

SQL = (
    "```sql\nSELECT department, ROUND(AVG(attendance_pct), 2) AS avg_attendance_pct "
    "FROM employees GROUP BY department ORDER BY avg_attendance_pct DESC\n```"
)
QUESTION = "What is the average attendance by department?"


def ask(client: TestClient, token: str, question: str) -> dict:  # type: ignore[type-arg]
    response = client.post("/chat", json={"question": question}, headers=auth_header(token))
    assert response.status_code == 200, response.text
    return response.json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("role", ["hr", "c_level"])
def test_hr_data_questions_are_computed_with_sql(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM, role: str
) -> None:
    fake_llm.script = [SQL, "Design leads with 96.34% [1]."]
    body = ask(client, role_tokens[role], QUESTION)

    assert body["route"] == "sql+docs"
    first = body["sources"][0]
    assert first["number"] == 1 and first["cited"]
    assert first["document"] == "hr/hr_data.csv" and first["department"] == "hr"
    assert "avg_attendance_pct" in first["excerpt"] and "```sql" in first["excerpt"]
    answer_prompt = fake_llm.calls[-1][-1].content
    assert "| department | avg_attendance_pct |" in answer_prompt  # result reached the LLM


@pytest.mark.parametrize("role", ["employee", "engineering", "marketing", "finance"])
def test_other_roles_never_trigger_sql_or_see_the_schema(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM, role: str
) -> None:
    body = ask(client, role_tokens[role], "What is the average salary by department?")

    assert body["route"] == "docs"
    assert len(fake_llm.calls) == 1  # only the answer call: no SQL generation
    prompt = "\n".join(m.content for m in fake_llm.calls[0])
    assert "TABLE employees" not in prompt and "salary REAL" not in prompt
    assert all(s["department"] != "hr" for s in body["sources"])


def test_no_sql_reply_falls_back_to_documents(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    fake_llm.script = ["NO_SQL", "Here is the policy [1]."]
    body = ask(client, role_tokens["hr"], "How many days of maternity leave are allowed?")
    assert body["route"] == "docs"
    assert body["sources"]


def test_document_questions_skip_the_sql_step(
    client: TestClient, role_tokens: dict[str, str], fake_llm: FakeLLM
) -> None:
    body = ask(client, role_tokens["hr"], "What is the leave encashment policy?")
    assert body["route"] == "docs"
    assert len(fake_llm.calls) == 1
