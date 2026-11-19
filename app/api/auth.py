"""Authentication endpoints: log in, and ask "who am I and what can I see?"."""

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.api.deps import (
    Audit,
    CurrentUser,
    Scope,
    ServicesDep,
    client_address,
    too_many_requests,
)
from app.auth.tokens import create_access_token
from app.core.rate_limit import RateLimiter

router = APIRouter(prefix="/auth", tags=["auth"])
security_log = logging.getLogger("security")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_at: datetime
    role: str


class DepartmentInfo(BaseModel):
    name: str
    description: str


class MeResponse(BaseModel):
    username: str
    full_name: str
    role: str
    role_display_name: str
    role_description: str
    accessible_departments: list[DepartmentInfo]
    suggested_questions: list[str]


@router.post("/login", response_model=TokenResponse)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    services: ServicesDep,
    audit: Audit,
) -> TokenResponse:
    """Exchange username + password for a bearer token (OAuth2 password flow)."""
    address = client_address(request)
    # Keyed by client *and* username: slows password guessing against one
    # account without locking out everyone behind the same proxy or the UI.
    limiter: RateLimiter = request.app.state.login_limiter
    retry_after = limiter.hit(f"{address}|{form.username.lower()}")
    if retry_after is not None:
        security_log.warning("Login rate limit hit", extra={"user": form.username})
        audit.record("rate_limited", user=form.username, endpoint="login", client=address)
        raise too_many_requests(retry_after)

    user = services.users.authenticate(form.username, form.password)
    if user is None:
        security_log.warning("Failed login", extra={"user": form.username})
        audit.record("login_failed", user=form.username, client=address)
        # Same message for "no such user" and "wrong password": don't help attackers.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_at = create_access_token(user.username, user.role, services.settings)
    security_log.info("Login", extra={"user": user.username, "role": user.role})
    audit.record("login", user=user.username, role=user.role, client=address)
    return TokenResponse(access_token=token, expires_at=expires_at, role=user.role)


@router.get("/me", response_model=MeResponse)
def me(user: CurrentUser, scope: Scope, services: ServicesDep) -> MeResponse:
    """The logged-in user's profile and the departments they may query."""
    role = services.policy.describe(user.role)
    descriptions = services.policy.departments
    return MeResponse(
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        role_display_name=role.display_name,
        role_description=role.description,
        accessible_departments=[
            DepartmentInfo(name=d, description=descriptions[d]) for d in sorted(scope.departments)
        ],
        suggested_questions=role.suggested_questions,
    )
