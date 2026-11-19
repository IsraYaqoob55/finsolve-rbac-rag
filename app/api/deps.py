"""FastAPI dependencies shared by routers.

A dependency is a function FastAPI runs before your endpoint and whose result
it passes in. Putting auth here means every protected endpoint just declares
``user: CurrentUser`` and can't forget to check the token.
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.tokens import InvalidTokenError, decode_access_token
from app.auth.user_store import User
from app.rbac.policy import AccessScope, UnknownRoleError
from app.services.container import Services

security_log = logging.getLogger("security")

# tokenUrl makes the "Authorize" button in /docs work out of the box.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], services: ServicesDep) -> User:
    """Validate the bearer token and load the *current* user record.

    We re-read the user from the database on every request instead of
    trusting the role inside the token, so disabling a user or changing
    their role takes effect immediately, even for tokens already issued.
    """
    try:
        token_data = decode_access_token(token, services.settings)
    except InvalidTokenError as exc:
        security_log.warning("Rejected token", extra={"reason": str(exc)})
        raise _UNAUTHORIZED from exc
    user = services.users.get(token_data.username)
    if user is None or user.disabled:
        security_log.warning("Token for missing/disabled user", extra={"user": token_data.username})
        raise _UNAUTHORIZED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_access_scope(user: CurrentUser, services: ServicesDep) -> AccessScope:
    try:
        return services.policy.scope_for(user.role)
    except UnknownRoleError as exc:
        # A user with a role the policy doesn't know gets nothing, loudly.
        security_log.error(
            "User has unknown role", extra={"user": user.username, "role": user.role}
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Your role has no access policy") from exc


Scope = Annotated[AccessScope, Depends(get_access_scope)]
