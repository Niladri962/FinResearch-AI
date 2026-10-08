from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.deps import AdminDep, ContainerDep, ViewerDep
from app.config import ALLOWED_EXTENSIONS
from app.guardrails.output_guard import DISCLAIMER
from app.models.schemas import DashboardResponse, HealthResponse, SystemInfo

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(container: ContainerDep) -> HealthResponse:
    """Liveness/readiness probe. Unauthenticated; exposes no secrets or connection strings."""
    components = container.health()
    critical = ("database", "vector_store")
    degraded = any(components[name]["status"] != "ok" for name in critical)
    return HealthResponse(
        status="degraded" if degraded else "ok", version=container.settings.app_version, components=components
    )


@router.get("/system", response_model=SystemInfo)
def system_info(container: ContainerDep, _principal: ViewerDep) -> SystemInfo:
    """Non-secret runtime configuration, shown on the Settings page."""
    s = container.settings
    return SystemInfo(
        app_name=s.app_name, version=s.app_version, environment=s.environment,
        llm={
            "provider": container.llm.provider, "model": container.llm.model or None,
            "configured": container.llm.available, "temperature": s.llm_temperature,
            "mode": "generative" if container.llm.available else "extractive",
        },
        embeddings={"provider": s.embedding_provider, "model": container.embedder.name},
        reranker={"provider": s.reranker_provider, "model": container.reranker.name},
        retrieval={
            "semantic_top_k": s.semantic_top_k, "bm25_top_k": s.bm25_top_k, "semantic_weight": s.semantic_weight,
            "keyword_weight": s.keyword_weight, "fusion_method": s.fusion_method, "rerank_top_n": s.rerank_top_n,
            "chunk_target_tokens": s.chunk_target_tokens, "chunk_overlap_tokens": s.chunk_overlap_tokens,
        },
        vector_store={"backend": container.vector_store.backend},
        database=container.engine.dialect.name,
        cache=container.cache.backend,
        auth_enabled=s.auth_enabled,
        limits={
            "max_upload_mb": s.max_upload_mb, "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            "rate_limit_per_minute": s.rate_limit_per_minute if s.rate_limit_enabled else None,
            "max_query_chars": s.max_query_chars, "disclaimer": DISCLAIMER,
        },
    )


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(container: ContainerDep, principal: ViewerDep) -> DashboardResponse:
    return await container.dashboard.overview(principal.subject)


@router.get("/observability/traces")
def traces(
    container: ContainerDep, _principal: AdminDep, limit: int = Query(50, ge=1, le=200)
) -> list[dict[str, Any]]:
    """Recent query traces: intent, stage latencies, token usage, source ids. No document text."""
    return container.dashboard.traces(limit)
