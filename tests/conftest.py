"""Shared pytest fixtures.

The API tests run the real app with *fake* heavy parts:
- real config/roles.yaml (so RBAC tests check the actual policy)
- real data/ documents, indexed into an in-memory Qdrant with FakeEmbedder
- an in-memory SQLite user store with one user per role
Built once per test session because bcrypt hashing and indexing take time.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.auth.user_store import UserStore
from app.core.config import Settings, get_settings
from app.ingestion.pipeline import run_ingestion
from app.main import create_app
from app.nlp.rewriter import QueryRewriter
from app.rbac.policy import RolePolicy
from app.retrieval.retriever import SecureRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.services.chat_service import ChatService
from app.services.container import build_sql_tool
from tests.fakes import DIM, FakeEmbedder, FakeLLM, FakeReranker, FakeSparseEmbedder

TEST_PASSWORD = "test-password-123"  # noqa: S105 - test-only credential
ROLE_USERS = {
    "employee": "t_employee",
    "engineering": "t_engineering",
    "marketing": "t_marketing",
    "finance": "t_finance",
    "hr": "t_hr",
    "c_level": "t_c_level",
}


class SessionServices:
    """Services whose close() is a no-op, so one app shutdown doesn't break later tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.policy = RolePolicy.from_yaml(settings.resolve(settings.roles_config))
        self.users = UserStore(":memory:")
        for role, username in ROLE_USERS.items():
            self.users.create(username, f"Test {role}", role, TEST_PASSWORD)
        embedder, sparse = FakeEmbedder(), FakeSparseEmbedder()
        self.store = QdrantVectorStore(QdrantClient(":memory:"), "test", DIM)
        run_ingestion(settings, self.store, embedder, sparse)  # real data/ folder
        # The full production pipeline (hybrid + quarter boost + rerank), with fakes.
        self.reranker = FakeReranker()
        self.retriever = SecureRetriever(self.store, embedder, sparse, self.reranker)
        self.llm = FakeLLM()
        self.sql_tool = build_sql_tool(settings, self.llm, self.policy)  # real HR data
        self.chat = ChatService(
            self.retriever,
            self.llm,
            settings,
            QueryRewriter(self.llm),
            self.policy.departments,
            self.sql_tool,
        )

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Make each test read settings fresh, so env overrides take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        jwt_secret_key="test-secret-" + "x" * 40,  # type: ignore[arg-type]
        audit_log_path=tmp_path_factory.mktemp("audit") / "audit.log",
    )


@pytest.fixture(scope="session")
def services(test_settings: Settings) -> SessionServices:
    return SessionServices(test_settings)


@pytest.fixture
def client(test_settings: Settings, services: SessionServices) -> Iterator[TestClient]:
    """HTTP client for the app, running in-process with the session services."""
    app = create_app(test_settings, services_factory=lambda _: services)  # type: ignore[arg-type,return-value]
    with TestClient(app) as test_client:  # `with` runs startup/shutdown (lifespan)
        yield test_client


def login(client: TestClient, username: str, password: str = TEST_PASSWORD) -> str:
    response = client.post("/auth/login", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    token: str = response.json()["access_token"]
    return token


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def fake_llm(services: SessionServices) -> FakeLLM:
    """The session's fake LLM, reset to its default reply for each test."""
    services.llm.reset()
    return services.llm


@pytest.fixture(scope="session")
def role_tokens(test_settings: Settings, services: SessionServices) -> dict[str, str]:
    """A valid token for each role, created once."""
    from app.auth.tokens import create_access_token

    return {
        role: create_access_token(username, role, test_settings)[0]
        for role, username in ROLE_USERS.items()
    }
