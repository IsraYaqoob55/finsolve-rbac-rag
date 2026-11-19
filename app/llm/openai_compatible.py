"""One client for every OpenAI-compatible chat API: Groq, OpenAI, Ollama, ...

Most LLM vendors expose the same HTTP API shape that OpenAI defined, so a
single implementation with a different ``base_url`` covers all of them.
"""

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import openai

from app.core.config import Settings
from app.llm.base import LLMError, Message

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",  # local models, no key needed
}


def _to_llm_error(exc: Exception) -> LLMError:
    """Map SDK exceptions to our own error type (callers never see vendor types)."""
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("LLM authentication failed: check LLM_API_KEY", retryable=False)
    if isinstance(exc, openai.RateLimitError):
        return LLMError("LLM rate limit reached, please retry shortly")
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("LLM request timed out")
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("Could not reach the LLM service")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"LLM service error ({exc.status_code})", retryable=exc.status_code >= 500)
    return LLMError(f"Unexpected LLM error: {exc}")


class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        timeout: float = 30,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
    ) -> None:
        # The SDK retries rate limits / 5xx errors with exponential backoff.
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed",  # Ollama ignores the key
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleProvider":
        api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        if api_key is None and settings.llm_provider != "ollama":
            raise RuntimeError(f"LLM_API_KEY is required for provider '{settings.llm_provider}'")
        return cls(
            model=settings.llm_model,
            base_url=settings.llm_base_url or DEFAULT_BASE_URLS[settings.llm_provider],
            api_key=api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            reasoning_effort=settings.llm_reasoning_effort,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def _request(self, messages: Sequence[Message], stream: bool) -> Any:
        # Only reasoning models accept reasoning_effort, so send it only when configured.
        extra: dict[str, Any] = (
            {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}
        )
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=stream,
            **extra,
        )

    def complete(self, messages: Sequence[Message]) -> str:
        try:
            response = self._request(messages, stream=False)
        except openai.OpenAIError as exc:
            raise _to_llm_error(exc) from exc
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "llm_call",
                extra={
                    "model": self._model,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            )
        return str(response.choices[0].message.content or "")

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        try:
            for event in self._request(messages, stream=True):
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except openai.OpenAIError as exc:
            raise _to_llm_error(exc) from exc
