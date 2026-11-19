"""In-memory sliding-window rate limiter.

Why rate limits here:
- Every chat request costs LLM tokens (money, or free-tier quota shared by
  everyone), so one user or a runaway script must not use it all.
- Login attempts are limited per username and client, which slows down
  password guessing.

Sliding window: we keep the timestamps of recent requests per key and allow a
new one only if fewer than ``limit`` happened in the last ``window`` seconds.
Unlike a fixed "per calendar minute" counter, it can't be gamed by sending a
burst at the end of one minute and another at the start of the next.

The state lives in this process. With several API replicas you would keep it
in Redis instead, behind the same ``hit()`` interface.
"""

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``limit=0`` disables the limiter."""
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record a request for ``key``.

        Returns ``None`` if it is allowed, otherwise the seconds until the
        oldest request leaves the window (for the ``Retry-After`` header).
        Rejected requests are not recorded, so waiting always helps.
        """
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(hits[0] + self.window - now, 0.0)
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop keys with no recent hits
                self._prune(now)
            return None

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window]:
            del self._hits[key]
