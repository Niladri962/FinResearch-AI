"""Application configuration.

All runtime behaviour is driven by environment variables (see ``.env.example``).
Nothing in the codebase reads ``os.environ`` directly; components receive a
``Settings`` instance so they can be constructed with test overrides.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

_PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "local": "http://localhost:11434/v1",
}

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".xlsx", ".txt", ".md"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "FinResearch AI"
    app_subtitle: str = "AI-Powered Financial Research & Analysis Platform"
    app_version: str = "1.0.0"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000"
    # Optional regex for origins that change per deployment, e.g. Vercel previews:
    # https://my-app(-[a-z0-9-]+)?\.vercel\.app
    cors_origin_regex: str = ""

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["auto", "groq", "openai", "local", "none"] = "auto"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_provider: Literal["fastembed", "sentence_transformers", "openai", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_batch_size: int = 8   # small batches are faster on CPU and keep memory low
    hash_embedding_dim: int = 384

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_provider: Literal["fastembed", "sentence_transformers", "lexical", "none"] = "fastembed"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # ── Retrieval ────────────────────────────────────────────────────────
    semantic_top_k: int = Field(20, ge=1, le=200)
    bm25_top_k: int = Field(20, ge=1, le=200)
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0)
    fusion_method: Literal["weighted", "rrf"] = "weighted"
    rerank_top_n: int = Field(6, ge=1, le=20)
    min_semantic_score: float = 0.2

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 48
    chunk_max_table_tokens: int = 900

    # ── Vector store ─────────────────────────────────────────────────────
    # auto → Qdrant when QDRANT_URL is set, otherwise the on-disk local store.
    vector_store: Literal["auto", "qdrant", "local", "memory"] = "auto"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "finresearch_chunks"

    # ── Persistence ──────────────────────────────────────────────────────
    database_url: str = ""
    redis_url: str = ""
    data_dir: Path = REPO_ROOT / "data"
    model_cache: str = ""          # MODEL_CACHE: where embedding/reranker models are stored (default DATA_DIR/models)
    max_upload_mb: int = Field(50, ge=1, le=500)

    # ── Security ─────────────────────────────────────────────────────────
    auth_enabled: bool = False
    api_keys: str = ""
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(120, ge=1)
    max_query_chars: int = 2000

    # ── Observability ────────────────────────────────────────────────────
    trace_query_text: bool = False
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "finresearch-ai"
    otel_exporter_otlp_endpoint: str = ""

    @field_validator("data_dir", mode="before")
    @classmethod
    def _default_data_dir(cls, value: object) -> object:
        # An empty DATA_DIR in .env means "use the default".
        return REPO_ROOT / "data" if value in ("", None) else value

    # ── Derived values ───────────────────────────────────────────────────
    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if not self.llm_api_key:
            return "local" if self.llm_base_url else "none"
        return "groq" if self.llm_api_key.startswith("gsk_") else "openai"

    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        return _PROVIDER_BASE_URLS.get(self.resolved_llm_provider, "")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            # Accept the common `postgres://` / `postgresql://` forms from hosting providers.
            url = self.database_url
            if url.startswith("postgres://"):
                url = "postgresql://" + url[len("postgres://"):]
            if url.startswith("postgresql://"):
                url = "postgresql+psycopg://" + url[len("postgresql://"):]
            return url
        return f"sqlite:///{(self.data_dir / 'finresearch.db').as_posix()}"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def model_cache_dir(self) -> Path:
        return Path(self.model_cache) if self.model_cache else self.data_dir / "models"

    @property
    def resolved_vector_store(self) -> str:
        if self.vector_store != "auto":
            return self.vector_store
        return "qdrant" if self.qdrant_url else "local"

    @property
    def local_vector_path(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.model_cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
