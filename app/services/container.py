"""The service container: long-lived objects the API needs, built once at startup.

Loading models or opening databases per request would be slow, so we build
them once (in FastAPI's lifespan) and share them. Tests swap in fakes by
passing a different factory to ``create_app``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.auth.user_store import UserStore
from app.core.config import Settings
from app.llm.base import LLMProvider
from app.nlp.rewriter import QueryRewriter
from app.rbac.policy import RolePolicy
from app.retrieval.embeddings import Embedder, SparseEmbedder
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalOptions, SecureRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.services.chat_service import ChatService
from app.structured.datasets import load_datasets
from app.structured.sql_engine import SafeSQLDatabase
from app.structured.sql_tool import SQLTool

# Named retrieval configurations, used by the evaluation script to measure
# what each layer contributes (each entry overrides Settings fields).
RETRIEVAL_MODES: dict[str, dict[str, Any]] = {
    "dense only (baseline)": {
        "retrieval_hybrid": False,
        "retrieval_quarter_boost": False,
        "retrieval_rerank": False,
    },
    "dense + quarter boost": {
        "retrieval_hybrid": False,
        "retrieval_quarter_boost": True,
        "retrieval_rerank": False,
    },
    "hybrid (dense + BM25)": {
        "retrieval_hybrid": True,
        "retrieval_quarter_boost": False,
        "retrieval_rerank": False,
    },
    "hybrid + quarter boost": {
        "retrieval_hybrid": True,
        "retrieval_quarter_boost": True,
        "retrieval_rerank": False,
    },
    "hybrid + boost + rerank (default)": {
        "retrieval_hybrid": True,
        "retrieval_quarter_boost": True,
        "retrieval_rerank": True,
    },
}


@dataclass
class Services:
    settings: Settings
    policy: RolePolicy
    users: UserStore
    store: QdrantVectorStore
    retriever: SecureRetriever
    chat: ChatService

    def close(self) -> None:
        self.store.close()
        self.users.close()


ServicesFactory = Callable[[Settings], Services]


@dataclass
class RetrievalComponents:
    """The heavy, shareable parts of retrieval (models + index)."""

    store: QdrantVectorStore
    embedder: Embedder
    sparse: SparseEmbedder
    reranker: Reranker | None

    def retriever(self, settings: Settings) -> SecureRetriever:
        """A retriever whose behaviour follows ``settings`` (cheap to create)."""
        return SecureRetriever(
            self.store, self.embedder, self.sparse, self.reranker, retrieval_options(settings)
        )


def retrieval_options(settings: Settings) -> RetrievalOptions:
    return RetrievalOptions(
        hybrid=settings.retrieval_hybrid,
        quarter_boost=settings.retrieval_quarter_boost,
        rerank=settings.retrieval_rerank,
        candidates=settings.retrieval_candidates,
    )


def build_embedders(settings: Settings) -> tuple[Embedder, SparseEmbedder]:
    from app.retrieval.embeddings import FastEmbedEmbedder, FastEmbedSparseEmbedder

    cache = settings.resolve(settings.embedding_cache_dir)
    dense = FastEmbedEmbedder(
        settings.embedding_model, cache_dir=cache, query_prefix=settings.embedding_query_prefix
    )
    return dense, FastEmbedSparseEmbedder(settings.sparse_model, cache_dir=cache)


def build_retrieval_components(
    settings: Settings, with_reranker: bool = True
) -> RetrievalComponents:
    from app.retrieval.reranker import CrossEncoderReranker
    from app.retrieval.vector_store import build_client

    embedder, sparse = build_embedders(settings)
    store = QdrantVectorStore(
        build_client(settings), settings.qdrant_collection, embedder.dimension
    )
    store.ensure_collection()  # fails fast with a clear message if the index is outdated
    reranker = (
        CrossEncoderReranker(
            settings.reranker_model, cache_dir=settings.resolve(settings.embedding_cache_dir)
        )
        if with_reranker
        else None
    )
    return RetrievalComponents(store, embedder, sparse, reranker)


def build_services(settings: Settings) -> Services:
    """Build the real services (models, Qdrant, SQLite, LLM client)."""
    from app.llm.openai_compatible import OpenAICompatibleProvider

    policy = RolePolicy.from_yaml(settings.resolve(settings.roles_config))
    components = build_retrieval_components(settings, with_reranker=settings.retrieval_rerank)
    retriever = components.retriever(settings)
    llm = OpenAICompatibleProvider.from_settings(settings)
    # Follow-up rewriting is a short, easy task: a smaller model is fast enough.
    helper_llm = (
        OpenAICompatibleProvider.from_settings(
            settings.model_copy(update={"llm_model": settings.llm_helper_model})
        )
        if settings.llm_helper_model
        else llm
    )
    rewriter = QueryRewriter(helper_llm) if settings.query_rewrite else None
    sql_tool = build_sql_tool(settings, llm, policy) if settings.sql_enabled else None
    return Services(
        settings=settings,
        policy=policy,
        users=UserStore(settings.resolve(settings.user_db_path)),
        store=components.store,
        retriever=retriever,
        chat=ChatService(retriever, llm, settings, rewriter, policy.departments, sql_tool),
    )


def build_sql_tool(settings: Settings, llm: LLMProvider, policy: RolePolicy) -> SQLTool:
    """Load the structured datasets into the locked-down SQL engine."""
    datasets = load_datasets(settings.resolve(settings.datasets_config))
    unknown = {s.department for s in datasets.values()} - set(policy.departments)
    if unknown:  # a dataset nobody could access, or a typo: fail fast
        raise RuntimeError(f"datasets.yaml uses departments not in roles.yaml: {sorted(unknown)}")
    database = SafeSQLDatabase(
        datasets,
        base_dir=settings.resolve(Path(".")),
        max_rows=settings.sql_max_rows,
        timeout_seconds=settings.sql_timeout_seconds,
    )
    data_dir = settings.resolve(settings.data_dir)
    sources = {
        name: settings.resolve(spec.source).relative_to(data_dir).as_posix()
        for name, spec in datasets.items()
    }
    return SQLTool(
        database, llm, {name: spec.department for name, spec in datasets.items()}, sources
    )
