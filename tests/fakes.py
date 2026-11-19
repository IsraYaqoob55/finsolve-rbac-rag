"""Test doubles shared across the test suite."""

import hashlib
import math
import re
from collections.abc import Iterator, Sequence

from app.llm.base import Message

DIM = 64


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors.

    Deterministic, instant, and needs no model download, so retrieval tests
    can run in CI. It is NOT a good embedder; tests only rely on word overlap.
    """

    dimension = DIM

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM  # noqa: S324
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Scripted LLM that records what it was sent.

    Tests set ``reply`` (or ``error``) and then inspect ``calls`` to see the
    exact prompt, e.g. to prove no out-of-scope document reached the model.
    """

    model_name = "fake-llm"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.reply = "Revenue was $2.4 billion [1]. Net income rose [2][1]."
        self.error: Exception | None = None
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return self.reply

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        for word in self.reply.split(" "):
            yield word + " "
