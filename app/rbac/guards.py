"""Second line of defence: re-check every retrieved chunk against the user's scope.

The vector-store filter should already guarantee this. The guard exists so
that a future bug (a forgotten filter, a new retrieval path such as keyword
search, a cache) can't leak data: anything out of scope is dropped and logged
as a security event, so the bug gets noticed.
"""

import logging
from collections.abc import Sequence
from typing import Protocol, TypeVar

from app.rbac.policy import AccessScope

security_log = logging.getLogger("security")


class HasDepartment(Protocol):
    @property
    def department(self) -> str: ...


class HasChunk(Protocol):
    @property
    def chunk(self) -> HasDepartment: ...


T = TypeVar("T", bound=HasChunk)


def enforce_scope(hits: Sequence[T], scope: AccessScope) -> list[T]:
    """Return only hits whose chunk department is inside ``scope``."""
    allowed = [hit for hit in hits if scope.can_access(hit.chunk.department)]
    blocked = len(hits) - len(allowed)
    if blocked:
        security_log.error(
            "RBAC guard blocked out-of-scope chunks",
            extra={"role": scope.role, "blocked": blocked},
        )
    return allowed
