"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import auth, health, search
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.container import ServicesFactory, build_services

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, services_factory: ServicesFactory = build_services
) -> FastAPI:
    """Build and configure the FastAPI app (the "application factory" pattern).

    A factory instead of a module-level global makes the app easy to create
    fresh in tests, with different settings and fake services.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once at startup (before the first request) and once at shutdown.
        app.state.services = services_factory(settings)
        logger.info("Services ready", extra={"env": settings.app_env})
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Internal chatbot with role-based access control over company documents.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(search.router)
    return app


app = create_app()
