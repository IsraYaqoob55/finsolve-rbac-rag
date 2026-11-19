"""The LLM interface the rest of the app depends on.

Chat code never imports a vendor SDK. It talks to ``LLMProvider``, so the
model or vendor can change with configuration alone, and tests can use a
fake that returns canned answers.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class LLMError(Exception):
    """The LLM call failed (network, timeout, rate limit, auth...).

    ``retryable`` tells the API layer whether "try again later" makes sense.
    """

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def complete(self, messages: Sequence[Message]) -> str:
        """Return the full response text."""
        ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield the response piece by piece, as the model generates it."""
        ...
