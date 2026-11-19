"""Rewrite follow-up questions into standalone search queries.

"And Q4?" or "what about her attendance?" retrieve nothing useful on their
own: the meaning lives in the earlier turns. One short LLM call rewrites the
latest question using the conversation, and *only the search* uses the
rewrite. The answer step still sees the user's own words plus the history.
Any failure falls back to the original question (rewriting must never break chat).
"""

import logging
from collections.abc import Sequence

from app.llm.base import LLMError, LLMProvider, Message

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You rewrite the user's latest message into a standalone search query \
for a company document search engine.

Rules:
- Resolve pronouns and references ("it", "her", "that quarter", "and Q4?") using the conversation.
- Keep names, numbers, quarters, years and document terms exactly.
- If the latest message is already standalone, return it unchanged.
- Output ONLY the rewritten query on one line: no quotes, no explanation."""

_MAX_HISTORY_CHARS = 600  # per message; long answers only need their gist


class QueryRewriter:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def rewrite(self, question: str, history: Sequence[Message]) -> str:
        if not history:
            return question
        conversation = "\n".join(
            f"{m.role.upper()}: {m.content[:_MAX_HISTORY_CHARS]}" for m in history
        )
        messages = [
            Message("system", REWRITE_PROMPT),
            Message("user", f"Conversation:\n{conversation}\n\nLatest message: {question}"),
        ]
        try:
            rewritten = self._llm.complete(messages)
        except LLMError as exc:
            logger.warning("Query rewrite failed, using original", extra={"error": str(exc)})
            return question
        lines = rewritten.strip().splitlines()  # empty when the model returned only whitespace
        cleaned = lines[0].strip().strip('"').strip() if lines else ""
        if not cleaned or len(cleaned) > 500:
            return question
        logger.info("query_rewrite", extra={"original": question, "rewritten": cleaned})
        return cleaned
