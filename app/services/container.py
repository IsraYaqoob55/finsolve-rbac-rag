"""The service container: long-lived objects the API needs, built once at startup.

Loading the embedding model or opening the database per request would be
slow, so we build them once (in FastAPI's lifespan) and share them. Tests
swap in fakes by passing a different factory to ``create_app``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.auth.user_store import UserStore
from app.core.config import Settings
from app.rbac.policy import RolePolicy
from app.retrieval.retriever import SecureRetriever
from app.retrieval.vector_store import QdrantVectorStore


@dataclass
class Services:
    settings: Settings
    policy: RolePolicy
    users: UserStore
    store: QdrantVectorStore
    retriever: SecureRetriever

    def close(self) -> None:
        self.store.close()
        self.users.close()


ServicesFactory = Callable[[Settings], Services]


def build_services(settings: Settings) -> Services:
    """Build the real services (embedding model, Qdrant, SQLite)."""
    from app.retrieval.embeddings import FastEmbedEmbedder
    from app.retrieval.vector_store import build_client

    policy = RolePolicy.from_yaml(settings.resolve(settings.roles_config))
    embedder = FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.resolve(settings.embedding_cache_dir),
        query_prefix=settings.embedding_query_prefix,
    )
    store = QdrantVectorStore(
        build_client(settings), settings.qdrant_collection, embedder.dimension
    )
    return Services(
        settings=settings,
        policy=policy,
        users=UserStore(settings.resolve(settings.user_db_path)),
        store=store,
        retriever=SecureRetriever(store, embedder),
    )
