"""Tests for password hashing, JWT tokens and the user store."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.auth.passwords import hash_password, verify_password
from app.auth.tokens import InvalidTokenError, create_access_token, decode_access_token
from app.auth.user_store import UserAlreadyExistsError, UserStore
from app.core.config import Settings

SECRET = "unit-test-secret-" + "y" * 40


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, jwt_secret_key=SECRET)  # type: ignore[arg-type]


# ---------- passwords ----------


def test_hash_is_salted_and_verifiable() -> None:
    first, second = hash_password("correct horse"), hash_password("correct horse")
    assert first != second  # random salt each time
    assert verify_password("correct horse", first)
    assert not verify_password("wrong horse", first)


def test_short_password_rejected() -> None:
    with pytest.raises(ValueError):
        hash_password("short")


def test_malformed_hash_does_not_crash() -> None:
    assert verify_password("anything1", "not-a-bcrypt-hash") is False


# ---------- tokens ----------


def test_token_round_trip(settings: Settings) -> None:
    token, expires_at = create_access_token("kabir_fin", "finance", settings)
    data = decode_access_token(token, settings)
    assert (data.username, data.role) == ("kabir_fin", "finance")
    assert data.expires_at > datetime.now(tz=UTC)
    assert abs((data.expires_at - expires_at).total_seconds()) < 1


def test_expired_token_rejected(settings: Settings) -> None:
    past = datetime.now(tz=UTC) - timedelta(minutes=5)
    token = jwt.encode({"sub": "u", "iat": past, "exp": past}, SECRET, algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, settings)


def test_tampered_token_rejected(settings: Settings) -> None:
    token, _ = create_access_token("alice_emp", "employee", settings)
    header, payload, signature = token.split(".")
    forged_claims = {"sub": "alice_emp", "role": "c_level", "iat": 1, "exp": 4102444800}
    forged = jwt.encode(forged_claims, "attacker-secret-" + "z" * 40, algorithm="HS256")
    with pytest.raises(InvalidTokenError):
        decode_access_token(forged, settings)
    with pytest.raises(InvalidTokenError):
        decode_access_token(f"{header}.{payload}x.{signature}", settings)


def test_unsigned_alg_none_token_rejected(settings: Settings) -> None:
    now = datetime.now(tz=UTC)
    token = jwt.encode(
        {"sub": "x", "iat": now, "exp": now + timedelta(hours=1)}, key=None, algorithm="none"
    )
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, settings)


def test_production_refuses_default_secret() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        Settings(_env_file=None, app_env="production")


# ---------- user store ----------


@pytest.fixture
def store() -> UserStore:
    return UserStore(":memory:")


def test_authenticate(store: UserStore) -> None:
    store.create("sara_hr", "Sara Khan", "hr", "password-123")
    user = store.authenticate("sara_hr", "password-123")
    assert user is not None and user.role == "hr"
    assert store.authenticate("sara_hr", "wrong-password") is None
    assert store.authenticate("nobody", "password-123") is None


def test_disabled_user_cannot_authenticate(store: UserStore) -> None:
    store.create("u1", "U One", "employee", "password-123")
    store.set_disabled("u1", True)
    assert store.authenticate("u1", "password-123") is None


def test_duplicate_username_rejected(store: UserStore) -> None:
    store.create("u1", "U One", "employee", "password-123")
    with pytest.raises(UserAlreadyExistsError):
        store.create("u1", "Other", "finance", "password-456")


def test_sql_injection_in_username_is_harmless(store: UserStore) -> None:
    store.create("u1", "U One", "employee", "password-123")
    assert store.authenticate("u1' OR '1'='1", "password-123") is None
    assert len(store.list_users()) == 1
