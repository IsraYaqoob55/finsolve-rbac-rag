"""Application settings.

All configuration comes from environment variables (or a local ``.env`` file),
validated by Pydantic. Code never reads ``os.environ`` directly: it asks for
``get_settings()``. This gives us one typed, documented place for config and
makes tests easy (override env vars, clear the cache).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (app/core/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    llm_provider: Literal["groq", "openai", "anthropic", "ollama"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: SecretStr | None = None  # SecretStr hides the value in logs/reprs

    # --- Auth ---
    jwt_secret_key: SecretStr = SecretStr("change-me-to-a-long-random-string")
    jwt_expire_minutes: int = Field(default=60, gt=0)

    @field_validator("qdrant_url", "qdrant_api_key", "llm_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        """Treat ``KEY=`` (empty value in .env) as "not set"."""
        return None if isinstance(value, str) and not value.strip() else value

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as an absolute path anchored at the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton (cached so .env is parsed only once)."""
    return Settings()
