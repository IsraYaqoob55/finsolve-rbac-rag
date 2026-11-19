"""Per-request context: a request ID that follows every log line.

When something goes wrong, a user (or the UI) can report the ``X-Request-ID``
they got back, and we can find every log and audit line for that request.
A ``ContextVar`` holds it: like a global, but separate for each request,
including in the worker threads FastAPI runs sync endpoints in.
"""

import logging
import re
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Accept a caller's ID only if it is short and plain (it ends up in logs).
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id(incoming: str | None = None) -> str:
    """Reuse a well-formed incoming ID (from a proxy or the UI), else make one."""
    if incoming and _VALID_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(value: str | None) -> object:
    """Set the current ID; returns a token for ``reset_request_id``."""
    return _request_id.set(value)


def reset_request_id(token: object) -> None:
    _request_id.reset(token)  # type: ignore[arg-type]


class RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to log records emitted while handling a request."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = _request_id.get()
        if request_id is not None and not hasattr(record, "request_id"):
            record.request_id = request_id
        return True
