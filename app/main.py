"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app import __version__
from app.api import health
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI app (the "application factory" pattern).

    A factory instead of a module-level global makes the app easy to
    create fresh in tests with different settings.
    """
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Internal chatbot with role-based access control over company documents.",
    )
    app.include_router(health.router)

    logger.info("App created", extra={"env": settings.app_env})
    return app


app = create_app()
