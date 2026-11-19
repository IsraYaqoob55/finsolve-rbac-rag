"""Headless tests of the Streamlit UI with Streamlit's AppTest.

AppTest runs the real script in-process; we patch the HTTP client so no
backend is needed.
"""

from collections.abc import Iterator

import pytest
from streamlit.testing.v1 import AppTest

from app.core.config import PROJECT_ROOT
from ui import api_client
from ui.api_client import ChatResult, Done, Profile, Source, Token

APP = str(PROJECT_ROOT / "ui" / "streamlit_app.py")  # AppTest needs an absolute path

PROFILE = Profile(
    username="kabir_fin",
    full_name="Kabir Shah",
    role="finance",
    role_display_name="Finance",
    role_description="Financial reports plus general information",
    departments=[("finance", "Money"), ("general", "Policies")],
    suggested_questions=["What was Q3 revenue?", "Biggest expenses?"],
)
RESULT = ChatResult(
    answer="Revenue was $2.4 billion [1].",
    sources=[
        Source(1, "finance/q.md", "Quarterly", "Q3", "finance", "Q3", 0.8, "Revenue $2.4B", True),
        Source(2, "general/h.md", "Handbook", "Leave", "general", None, 0.5, "Leave...", False),
    ],
    model="fake-model",
    latency_ms=1200,
)


def logged_in_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["token"] = "tok"
    at.session_state["profile"] = PROFILE
    at.session_state["messages"] = []
    return at


@pytest.fixture
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[dict[str, str]]]]:
    calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat_stream(
        self: api_client.FinSolveClient, question: str, history: list[dict[str, str]]
    ) -> Iterator[api_client.StreamEvent]:
        calls.append((question, history))
        yield Token("Revenue was ")
        yield Token("$2.4 billion [1].")
        yield Done(RESULT)

    monkeypatch.setattr(api_client.FinSolveClient, "chat_stream", chat_stream)
    return calls


def test_login_page_shown_when_not_logged_in() -> None:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Username", "Password"]
    assert at.button[0].label == "Log in"


def test_failed_login_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_login(self: api_client.FinSolveClient, username: str, password: str) -> str:
        raise api_client.AuthError("Incorrect username or password.")

    monkeypatch.setattr(api_client.FinSolveClient, "login", bad_login)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("kabir_fin")
    at.text_input[1].input("wrong")
    at.button[0].click().run()

    assert at.error[0].value == "Incorrect username or password."
    assert "token" not in at.session_state


def test_logged_in_view_shows_profile_and_suggestions() -> None:
    at = logged_in_app().run()
    assert not at.exception
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "Kabir Shah" in sidebar_text
    assert at.chat_input[0].placeholder.startswith("Ask a question (Finance access)")
    assert "Hi Kabir" in " ".join(m.value for m in at.markdown)


def test_asking_streams_answer_and_stores_sources(
    fake_stream: list[tuple[str, list[dict[str, str]]]],
) -> None:
    at = logged_in_app().run()
    at.chat_input[0].set_value("What was Q3 revenue?").run()

    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["result"].answer == RESULT.answer
    assert fake_stream == [("What was Q3 revenue?", [])]
    assert any("Sources (1 cited)" in e.label for e in at.expander)


def test_follow_up_sends_history(fake_stream: list[tuple[str, list[dict[str, str]]]]) -> None:
    at = logged_in_app().run()
    at.chat_input[0].set_value("What was Q3 revenue?").run()
    at.chat_input[0].set_value("And Q4?").run()

    question, history = fake_stream[-1]
    assert question == "And Q4?"
    assert [h["role"] for h in history] == ["user", "assistant"]


def test_expired_session_returns_to_login(monkeypatch: pytest.MonkeyPatch) -> None:
    def expired(
        self: api_client.FinSolveClient, question: str, history: list[dict[str, str]]
    ) -> Iterator[api_client.StreamEvent]:
        raise api_client.AuthError("Your session has expired. Please log in again.")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(api_client.FinSolveClient, "chat_stream", expired)
    at = logged_in_app().run()
    at.chat_input[0].set_value("hello").run()

    assert "token" not in at.session_state
    assert at.info[0].value == "Your session has expired. Please log in again."


def test_logout_clears_session() -> None:
    at = logged_in_app().run()
    logout = next(b for b in at.sidebar.button if b.label == "Log out")
    logout.click().run()
    assert "token" not in at.session_state
    assert at.info[0].value == "You have been logged out."
