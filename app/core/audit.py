"""Audit trail: who asked what, which data was used, and who was refused.

Application logs are for debugging and get rotated, filtered and sampled. An
audit log answers compliance questions ("did anyone outside HR see salary
data?"), so it is a separate, append-only file with one JSON object per line
and a fixed set of fields. Each line carries the request ID, which links it to
the application logs.

Production systems ship these lines to write-once storage (e.g. S3 Object Lock
or a SIEM); a local JSON-lines file keeps the same format for the demo.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.request_context import get_request_id

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500


class AuditLog:
    """Thread-safe JSON-lines writer. ``path=None`` disables auditing."""

    def __init__(self, path: Path | None, include_questions: bool = True) -> None:
        self.path = path
        self.include_questions = include_questions
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        """Append one event. Auditing must never break a request, so errors are logged."""
        if self.path is None:
            return
        if "question" in fields:
            question = str(fields["question"])
            fields["question"] = (
                question[:MAX_QUESTION_CHARS] if self.include_questions else "[redacted]"
            )
        entry = {
            "ts": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "request_id": get_request_id(),
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            logger.exception("Could not write audit event", extra={"event": event})

    def read(self) -> list[dict[str, Any]]:
        """All events (for tests and small admin tools)."""
        if self.path is None or not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
