"""Application settings.

All configuration comes from environment variables (or a local ``.env`` file),
validated by Pydantic. Code never reads ``os.environ`` directly: it asks for
``get_settings()``. This gives us one typed, documented place for config and
makes tests easy (override env vars, clear the cache).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/core/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JWT_SECRET = "change-me-to-a-long-random-string"  # noqa: S105 - placeholder, rejected in prod


class Settings(BaseSettings):
    """Typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown env vars are ignored, not errors
    )

    # --- App ---
    app_name: str = "FinSolve RBAC RAG"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # --- Paths (relative paths are resolved against the project root) ---
    data_dir: Path = Path("data")
    roles_config: Path = Path("config/roles.yaml")

    # --- Vector store (Qdrant) ---
    # If qdrant_url is set we talk to a Qdrant server (Docker/cloud);
    # otherwise Qdrant runs embedded, storing data in qdrant_path.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("qdrant_storage")
    qdrant_collection: str = "finsolve_docs"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Downloaded model files are kept here (default temp folders can get wiped).
    embedding_cache_dir: Path = Path(".cache/fastembed")
    # BGE models are trained to see this instruction before *queries* (not documents).
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Chunking (sizes in characters; ~4 characters = 1 token) ---
    chunk_max_chars: int = Field(default=1500, gt=200)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    # Sections shorter than this are merged with their small siblings.
    chunk_min_section_chars: int = Field(default=300, ge=0)

    # --- LLM ---
    # Any OpenAI-compatible API works; the provider name picks a default base URL.
    llm_provider: Literal["groq", "openai", "ollama"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    # For reasoning models only (e.g. gpt-oss): "low" | "medium" | "high". None = don't send.
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = None
    llm_api_key: SecretStr | None = None  # SecretStr hides the value in logs/reprs
    llm_base_url: str | None = None  # override the provider's default endpoint
    llm_temperature: float = Field(default=0.1, ge=0, le=2)  # low = factual, repeatable
    # Reasoning models spend part of this budget "thinking" before answering.
    llm_max_tokens: int = Field(default=1500, gt=0)
    llm_timeout_seconds: float = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # --- Chat / RAG ---
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    chat_history_messages: int = Field(default=6, ge=0)  # previous turns sent to the LLM

    # --- Auth ---
    jwt_secret_key: SecretStr = SecretStr(DEFAULT_JWT_SECRET)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_expire_minutes: int = Field(default=60, gt=0)
    user_db_path: Path = Path("var/users.db")
    seed_users_config: Path = Path("config/users.seed.yaml")
    # Password given to demo users by `python -m app.auth.manage seed`.
    seed_user_password: SecretStr | None = None

    @field_validator(
        "qdrant_url",
        "qdrant_api_key",
        "llm_api_key",
        "llm_base_url",
        "llm_reasoning_effort",
        "seed_user_password",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        """Treat ``KEY=`` (empty value in .env) as "not set"."""
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _no_default_secret_in_production(self) -> "Settings":
        """Fail fast: never run production with the placeholder JWT secret."""
        if self.app_env == "production" and (
            self.jwt_secret_key.get_secret_value() == DEFAULT_JWT_SECRET
            or len(self.jwt_secret_key.get_secret_value()) < 32
        ):
            raise ValueError("JWT_SECRET_KEY must be set to a random value of 32+ characters")
        return self

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as an absolute path anchored at the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton (cached so .env is parsed only once)."""
    return Settings()
