"""Tests for quarter detection and follow-up rewriting."""

import pytest

from app.llm.base import LLMError, Message
from app.nlp.query_analysis import detect_quarters
from app.nlp.rewriter import QueryRewriter
from tests.fakes import FakeLLM


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What was net income in Q3 2024?", ("Q3",)),
        ("compare q2 and Q4 revenue", ("Q2", "Q4")),
        ("Revenue in the third quarter", ("Q3",)),
        ("results for the 1st quarter and quarter 2", ("Q1", "Q2")),
        ("Q3 vs Q3 again", ("Q3",)),
        ("What is the leave policy?", ()),
        ("FAQ: Q5 is not a quarter", ()),
        ("SQL query optimisation", ()),
    ],
)
def test_detect_quarters(question: str, expected: tuple[str, ...]) -> None:
    assert detect_quarters(question) == expected


HISTORY = [
    Message("user", "What was net income in Q3 2024?"),
    Message("assistant", "Net income was $300 million [1]."),
]


def test_no_history_means_no_llm_call() -> None:
    llm = FakeLLM()
    assert (
        QueryRewriter(llm).rewrite("What is the leave policy?", []) == "What is the leave policy?"
    )
    assert llm.calls == []


def test_follow_up_is_rewritten_with_conversation_context() -> None:
    llm = FakeLLM()
    llm.reply = '"What was net income in Q4 2024?"\n'

    result = QueryRewriter(llm).rewrite("And Q4?", HISTORY)

    assert result == "What was net income in Q4 2024?"  # quotes and newline stripped
    prompt = llm.calls[0][-1].content
    assert "USER: What was net income in Q3 2024?" in prompt
    assert prompt.endswith("Latest message: And Q4?")


@pytest.mark.parametrize("reply", ["", "   ", "x" * 600])
def test_unusable_rewrite_falls_back_to_original(reply: str) -> None:
    llm = FakeLLM()
    llm.reply = reply
    assert QueryRewriter(llm).rewrite("And Q4?", HISTORY) == "And Q4?"


def test_llm_failure_falls_back_to_original() -> None:
    llm = FakeLLM()
    llm.error = LLMError("rate limit")
    assert QueryRewriter(llm).rewrite("And Q4?", HISTORY) == "And Q4?"
