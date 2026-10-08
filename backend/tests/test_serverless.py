"""Serverless (Vercel) mode: no persistent disk, no background work, slim dependencies."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.database import build_engine, build_session_factory
from app.models.db import Base
from app.rag.embeddings import HashEmbedder, OpenAICompatibleEmbedder, build_embedder
from app.rag.reranker import LexicalReranker, build_reranker
from app.rag.vector_store import SearchFilter, SqlVectorStore, VectorPoint, build_vector_store
from app.utils.errors import VectorStoreError
from tests.conftest import make_settings

AR25 = "aurora_industries_annual_report_fy2025.pdf"


def _points() -> list[VectorPoint]:
    return [
        VectorPoint("a", [1.0, 0.0, 0.0], {"document_id": "d1", "company_id": 1, "fiscal_year": 2025}),
        VectorPoint("b", [0.8, 0.6, 0.0], {"document_id": "d1", "company_id": 1, "fiscal_year": 2025}),
        VectorPoint("c", [0.0, 1.0, 0.0], {"document_id": "d2", "company_id": 2, "fiscal_year": 2024}),
    ]


@pytest.fixture
def session_factory(tmp_path: Path):
    engine = build_engine(f"sqlite:///{(tmp_path / 'vectors.db').as_posix()}")
    Base.metadata.create_all(engine)
    return build_session_factory(engine)


class TestSqlVectorStore:
    def test_search_filter_and_delete(self, session_factory):
        store = SqlVectorStore(session_factory)
        store.upsert(_points())
        assert store.count() == 3
        hits = store.search([1.0, 0.0, 0.0], 3)
        assert [h.id for h in hits] == ["a", "b", "c"] and hits[1].score == pytest.approx(0.8)
        assert [h.id for h in store.search([1, 0, 0], 5, SearchFilter(company_ids=[2]))] == ["c"]
        store.delete_document("d1")
        assert store.count() == 1 and [h.id for h in store.search([1, 0, 0], 5)] == ["c"]

    def test_a_second_instance_sees_writes_from_the_first(self, session_factory):
        """Separate function instances share nothing but the database."""
        writer, reader = SqlVectorStore(session_factory), SqlVectorStore(session_factory)
        assert reader.search([1, 0, 0], 5) == []
        writer.upsert(_points())
        assert [h.id for h in reader.search([0, 1, 0], 1)] == ["c"]
        writer.delete_document("d2")
        assert "c" not in [h.id for h in reader.search([0, 1, 0], 5)]

    def test_upsert_replaces_and_dimension_is_checked(self, session_factory):
        store = SqlVectorStore(session_factory)
        store.upsert(_points())
        store.upsert([VectorPoint("a", [0.0, 0.0, 1.0], {"document_id": "d1"})])
        assert store.count() == 3 and store.search([0, 0, 1], 1)[0].id == "a"
        with pytest.raises(VectorStoreError, match="3-dimensional"):
            store.upsert([VectorPoint("z", [1.0, 0.0], {"document_id": "d9"})])


class TestServerlessSettings:
    def test_detected_from_the_platform(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("VERCEL", "1")
        from app.config import Settings

        settings = Settings(_env_file=None)
        assert settings.serverless is True
        assert settings.data_dir == Path("/tmp/finresearch")
        assert settings.resolved_vector_store == "database"
        assert settings.effective_max_upload_mb == 4 and settings.max_upload_bytes == 4 * 1024 * 1024
        assert settings.persistent is False                       # no DATABASE_URL yet
        with_db = Settings(_env_file=None, database_url="postgres://u:p@host/db", qdrant_url="https://q.example")
        assert with_db.persistent and with_db.resolved_vector_store == "qdrant"

    def test_off_by_default(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("VERCEL", raising=False)
        settings = make_settings(tmp_path, vector_store="auto")
        assert not settings.serverless and settings.resolved_vector_store == "local"
        assert settings.effective_max_upload_mb == 50

    def test_factory_builds_the_database_store(self, tmp_path: Path, session_factory):
        settings = make_settings(tmp_path, serverless=True, vector_store="auto")
        assert build_vector_store(settings, session_factory).backend == "database"
        with pytest.raises(VectorStoreError):
            build_vector_store(settings)

    def test_postgres_uses_no_connection_pool_when_serverless(self):
        from sqlalchemy.pool import NullPool

        try:
            engine = build_engine("postgresql+psycopg://u:p@localhost:5432/db", serverless=True)
        except ImportError:
            pytest.skip("PostgreSQL driver is not loadable on this machine")
        assert isinstance(engine.pool, NullPool)


class TestSlimInstall:
    """Vercel installs requirements.txt only: no fastembed, no pdfplumber."""

    @pytest.fixture
    def no_fastembed(self, monkeypatch):
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "fastembed" else real(name, *a, **k))

    def test_embedder_falls_back(self, tmp_path: Path, no_fastembed):
        offline = build_embedder(make_settings(tmp_path, embedding_provider="fastembed"))
        assert isinstance(offline, HashEmbedder)
        api = build_embedder(make_settings(tmp_path, embedding_provider="fastembed", embedding_api_key="sk-test"))
        assert isinstance(api, OpenAICompatibleEmbedder) and api.name == "text-embedding-3-small"

    def test_reranker_falls_back(self, tmp_path: Path, no_fastembed):
        assert isinstance(build_reranker(make_settings(tmp_path, reranker_provider="fastembed")), LexicalReranker)

    def test_core_requirements_exclude_heavy_packages(self):
        backend = Path(__file__).resolve().parents[1]
        core = (backend / "requirements.txt").read_text(encoding="utf-8")
        packages = {line.split(">=")[0].split("[")[0].strip() for line in core.splitlines() if line and not line.startswith("#")}
        assert not packages & {"fastembed", "pdfplumber", "sentence-transformers", "torch", "onnxruntime"}
        assert {"fastapi", "sqlalchemy", "psycopg", "pymupdf", "langgraph"} <= packages
        assert "fastembed" in (backend / "requirements-local.txt").read_text(encoding="utf-8")


@pytest.mark.integration
class TestServerlessApp:
    @pytest.fixture
    def app(self, tmp_path: Path):
        settings = make_settings(tmp_path / "data", serverless=True, vector_store="auto")
        return create_app(settings)

    def test_works_without_lifespan_events(self, app):
        """Some serverless runtimes never send ASGI startup; the first request builds the container."""
        client = TestClient(app)                                  # no `with`: lifespan is not run
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["components"]["vector_store"]["backend"] == "database"
        assert body["components"]["storage"] == {
            "status": "ephemeral", "mode": "serverless",
            "detail": "Set DATABASE_URL: data is lost between invocations.",
        }
        assert client.get("/api/system").json()["limits"]["max_upload_mb"] == 4

    def test_upload_is_processed_within_the_request_and_is_searchable(self, app, sample_docs):
        client = TestClient(app)
        with sample_docs[AR25].open("rb") as handle:
            response = client.post("/api/documents/upload", files={"files": (AR25, handle)})   # no ?wait=true
        document = response.json()["results"][0]["document"]
        assert response.status_code == 202 and document["status"] == "ready" and document["fact_count"] == 72

        container = app.state.container
        assert container.vector_store.backend == "database" and container.vector_store.count() == document["chunk_count"]
        answer = client.post("/api/chat", json={"message": "What is Aurora's debt-to-equity ratio?", "stream": False}).json()
        assert answer["calculations"][0]["display"] == "0.64x" and answer["sources"]

        assert client.delete(f"/api/documents/{document['id']}").status_code == 204
        assert container.vector_store.count() == 0

    def test_oversized_upload_is_rejected_with_the_serverless_limit(self, app):
        client = TestClient(app)
        response = client.post("/api/documents/upload", files={"files": ("big.txt", b"revenue " * 700_000)})
        assert response.status_code == 413 and "4 MB" in response.json()["error"]["message"]


def test_vercel_entry_point_exports_the_asgi_app():
    backend = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vercel_entry", backend / "api" / "index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vercel_entry"] = module
    spec.loader.exec_module(module)
    assert module.app.title == "FinResearch AI"
    import json

    config = json.loads((backend / "vercel.json").read_text(encoding="utf-8"))
    assert config["rewrites"] == [{"source": "/(.*)", "destination": "/api/index"}]
    assert config["functions"]["api/index.py"]["maxDuration"] == 300
