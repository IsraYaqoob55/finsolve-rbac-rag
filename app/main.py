"""FastAPI application entry point.

Run locally with:
    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import auth, chat, health, search
from app.api.middleware import RequestContextMiddleware
from app.core.audit import AuditLog
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.core.rate_limit import RateLimiter
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
    # Cross-cutting web concerns live on app.state, next to the services.
    audit_path = settings.resolve(settings.audit_log_path) if settings.audit_log_path else None
    app.state.audit = AuditLog(audit_path, settings.audit_include_questions)
    app.state.chat_limiter = RateLimiter(settings.rate_limit_chat_per_minute)
    app.state.login_limiter = RateLimiter(settings.rate_limit_login_per_minute)

    # Request IDs, access logs, security headers and safe 500 responses.
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    return app


app = create_app()
