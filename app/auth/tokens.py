"""JSON Web Tokens (JWT) for stateless authentication.

After login the client receives a signed token and sends it on every request
in the ``Authorization: Bearer <token>`` header. The signature proves we
issued it; the ``exp`` claim makes it expire. Anyone can *read* a JWT
(it's only base64), so it must never contain secrets.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from app.core.config import Settings


class InvalidTokenError(Exception):
    """The token is missing, malformed, tampered with or expired."""


@dataclass(frozen=True)
class TokenData:
    username: str
    role: str
    expires_at: datetime


def create_access_token(username: str, role: str, settings: Settings) -> tuple[str, datetime]:
    """Return a signed token and its expiry time."""
    now = datetime.now(tz=UTC)
    expires_at = now + timedelta(minutes=settings.jwt_expire_minutes)
    claims = {
        "sub": username,  # "subject": who the token is about
        "role": role,  # informational; the server re-checks the role on each request
        "iat": now,  # issued at
        "exp": expires_at,  # expiry, checked automatically by PyJWT
    }
    token = jwt.encode(
        claims, settings.jwt_secret_key.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return token, expires_at


def decode_access_token(token: str, settings: Settings) -> TokenData:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            # Pin the algorithm list: never let the token choose (e.g. "none").
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
    return TokenData(
        username=str(claims["sub"]),
        role=str(claims.get("role", "")),
        expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
    )
