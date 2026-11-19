"""Tests for settings loading."""

import pytest

from app.core.config import PROJECT_ROOT, Settings


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("JWT_EXPIRE_MINUTES", "15")

    settings = Settings(_env_file=None)

    assert settings.app_env == "staging"
    assert settings.jwt_expire_minutes == 15


def test_invalid_env_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "not-a-real-env")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_secrets_are_hidden_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key")

    settings = Settings(_env_file=None)

    assert "super-secret-key" not in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "super-secret-key"


def test_relative_paths_resolve_to_project_root() -> None:
    settings = Settings(_env_file=None)

    assert settings.resolve(settings.data_dir) == PROJECT_ROOT / "data"
    assert (PROJECT_ROOT / "data").is_dir()
