"""Health check endpoint.

Load balancers, Docker and Kubernetes call this to know the service is alive.
Later phases will add dependency checks (vector store reachable, etc.).
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Shape of the /health response."""

    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API is up, with its version and environment."""
    settings = get_settings()
    return HealthResponse(status="ok", version=__version__, environment=settings.app_env)
