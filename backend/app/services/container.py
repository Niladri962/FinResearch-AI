"""Composition root: builds every component once and wires dependencies.

Routes and background tasks receive this container; nothing else constructs
infrastructure. Tests build a container from test settings to get an isolated,
offline stack.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from app.agents.state import AgentDeps
from app.agents.supervisor import Supervisor
from app.config import Settings
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.models.database import build_engine, build_session_factory
from app.models.db import Base
from app.rag.bm25 import BM25Index
from app.rag.embeddings import build_embedder
from app.rag.ingestion import IngestionPipeline
from app.rag.reranker import build_reranker
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import build_vector_store
from app.services.analysis import AnalysisService
from app.services.chat import ChatService
from app.services.comparison import ComparisonService
from app.services.dashboard import DashboardService
from app.services.documents import CompanyService, DocumentService
from app.services.financials import FinancialDataService
from app.services.reports import ReportService
from app.utils.cache import build_cache
from app.utils.logging import get_logger
from app.utils.rate_limit import RateLimiter

logger = get_logger(__name__)


class AppContainer:
    def __init__(self, settings: Settings, *, llm: LLMClient | None = None) -> None:
        """``llm`` overrides the configured provider (used by tests to inject a scripted model)."""
        settings.ensure_directories()
        self.settings = settings

        # Infrastructure
        self.engine = build_engine(settings.resolved_database_url)
        Base.metadata.create_all(self.engine)
        self.session_factory = build_session_factory(self.engine)
        self.cache = build_cache(settings.redis_url)
        self.rate_limiter = RateLimiter(
            self.cache, limit_per_minute=settings.rate_limit_per_minute, enabled=settings.rate_limit_enabled
        )

        # RAG
        self.embedder = build_embedder(settings)
        self.vector_store = build_vector_store(settings)
        self.bm25 = BM25Index(self.session_factory)
        self.reranker = build_reranker(settings)
        self.retriever = HybridRetriever(
            settings, self.session_factory, self.embedder, self.vector_store, self.bm25, self.reranker, self.cache
        )
        self.ingestion = IngestionPipeline(settings, self.session_factory, self.embedder, self.vector_store)
        self.llm = llm or build_llm(settings)

        # Domain services
        self.financials = FinancialDataService(self.session_factory)
        self.comparison = ComparisonService(self.financials)
        self.documents = DocumentService(settings, self.session_factory, self.vector_store)
        self.companies = CompanyService(self.session_factory, self.financials)
        self.supervisor = Supervisor(
            AgentDeps(
                settings=settings, session_factory=self.session_factory, retriever=self.retriever,
                financials=self.financials, comparison=self.comparison, llm=self.llm,
            )
        )
        self.chat = ChatService(settings, self.session_factory, self.supervisor)
        self.analysis = AnalysisService(self.financials, self.comparison, self.companies, self.chat)
        self.reports = ReportService(settings, self.session_factory, self.retriever, self.financials, self.llm)
        self.dashboard = DashboardService(self.session_factory, self.analysis)

    def health(self) -> dict[str, dict[str, Any]]:
        """Component status for the readiness probe. Never includes secrets or URLs."""
        components: dict[str, dict[str, Any]] = {}
        try:
            with self.session_factory() as session:
                session.execute(text("SELECT 1"))
            components["database"] = {"status": "ok", "backend": self.engine.dialect.name}
        except Exception as exc:
            components["database"] = {"status": "error", "error": type(exc).__name__}
        try:
            components["vector_store"] = {
                "status": "ok", "backend": self.vector_store.backend, "vectors": self.vector_store.count()
            }
        except Exception as exc:
            components["vector_store"] = {"status": "error", "backend": self.vector_store.backend, "error": type(exc).__name__}
        components["cache"] = {"status": "ok" if self.cache.healthy() else "degraded", "backend": self.cache.backend}
        components["llm"] = {
            "status": "ok" if self.llm.available else "not_configured",
            "provider": self.llm.provider, "model": self.llm.model or None,
        }
        components["embeddings"] = {"status": "ok", "provider": self.settings.embedding_provider, "model": self.embedder.name}
        components["reranker"] = {"status": "ok", "provider": self.settings.reranker_provider, "model": self.reranker.name}
        return components

    async def aclose(self) -> None:
        await self.llm.aclose()
        self.engine.dispose()
