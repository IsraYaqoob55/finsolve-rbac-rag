"""Health checks.

- ``GET /health``        liveness: the process is up (cheap, no dependencies).
- ``GET /health/ready``  readiness: it can actually serve questions (index built and
  users present; a missing LLM key already stops startup). Docker/Kubernetes send
  traffic only once this returns 200, and compose waits on it before starting the UI.

Readiness never reveals data: only check names, ok/failed and counts.
"""

import logging

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app import __version__
from app.api.deps import ServicesDep
from app.core.config import get_settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Shape of the /health response."""

    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API is up, with its version and environment."""
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, environment=settings.app_env)


@router.get("/health/ready", response_model=ReadinessResponse)
def ready(services: ServicesDep, response: Response) -> ReadinessResponse:
    """Report whether every dependency needed to answer questions is in place."""
    checks: dict[str, str] = {}
    try:
        chunks = services.store.count()
        checks["vector_index"] = f"ok ({chunks} chunks)" if chunks else "empty: run ingestion"
    except Exception:  # any failure means "not ready", whatever its type
        logger.exception("Readiness: vector store check failed")
        checks["vector_index"] = "unreachable"
    try:
        users = services.users.count()
        checks["user_store"] = f"ok ({users} users)" if users else "no users: run seeding"
    except Exception:
        logger.exception("Readiness: user store check failed")
        checks["user_store"] = "unreachable"

    ok = all(value.startswith("ok") for value in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ok else "not_ready", checks=checks)
