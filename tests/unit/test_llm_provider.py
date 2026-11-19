"""Tests for the OpenAI-compatible provider, without any network calls."""

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from app.core.config import Settings
from app.llm.base import LLMError, Message
from app.llm.openai_compatible import DEFAULT_BASE_URLS, OpenAICompatibleProvider, _to_llm_error

REQUEST = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")


def status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return cls("boom", response=httpx.Response(code, request=REQUEST), body=None)


@pytest.mark.parametrize(
    ("exc", "retryable", "text"),
    [
        (status_error(openai.AuthenticationError, 401), False, "LLM_API_KEY"),
        (status_error(openai.RateLimitError, 429), True, "rate limit"),
        (openai.APITimeoutError(request=REQUEST), True, "timed out"),
        (openai.APIConnectionError(request=REQUEST), True, "reach"),
        (status_error(openai.InternalServerError, 503), True, "503"),
        (status_error(openai.BadRequestError, 400), False, "400"),
    ],
)
def test_sdk_errors_map_to_llm_error(exc: Exception, retryable: bool, text: str) -> None:
    error = _to_llm_error(exc)
    assert isinstance(error, LLMError)
    assert error.retryable is retryable
    assert text in str(error)


def test_key_required_for_hosted_providers() -> None:
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        OpenAICompatibleProvider.from_settings(Settings(_env_file=None, llm_provider="groq"))


def test_ollama_needs_no_key_and_uses_local_url() -> None:
    provider = OpenAICompatibleProvider.from_settings(
        Settings(_env_file=None, llm_provider="ollama", llm_model="llama3.2")
    )
    assert provider.model_name == "llama3.2"
    assert str(provider._client.base_url).rstrip("/") == DEFAULT_BASE_URLS["ollama"]


class _StubCompletions:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result, self.error, self.kwargs = result, error, {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def provider_with(stub: _StubCompletions) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider("m", "https://llm.invalid/v1", "key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))  # type: ignore[assignment]
    return provider


def test_complete_sends_messages_and_returns_text() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [1]"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )
    stub = _StubCompletions(result=response)

    text = provider_with(stub).complete([Message("system", "S"), Message("user", "Q")])

    assert text == "Answer [1]"
    assert stub.kwargs["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
    ]
    assert stub.kwargs["stream"] is False


def test_stream_yields_only_content_deltas() -> None:
    def chunk(text: str | None) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    stub = _StubCompletions(result=iter([chunk("Hel"), chunk(None), chunk("lo")]))

    assert list(provider_with(stub).stream([Message("user", "Q")])) == ["Hel", "lo"]


def test_complete_wraps_sdk_errors() -> None:
    stub = _StubCompletions(error=openai.APITimeoutError(request=REQUEST))
    with pytest.raises(LLMError, match="timed out"):
        provider_with(stub).complete([Message("user", "Q")])


def test_reasoning_effort_sent_only_when_configured() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None
    )
    plain = _StubCompletions(result=response)
    provider_with(plain).complete([Message("user", "Q")])
    assert "reasoning_effort" not in plain.kwargs

    reasoning = _StubCompletions(result=response)
    provider = provider_with(reasoning)
    provider._reasoning_effort = "low"
    provider.complete([Message("user", "Q")])
    assert reasoning.kwargs["reasoning_effort"] == "low"
