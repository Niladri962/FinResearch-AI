"""Embeddings, vector stores, BM25, hybrid fusion, reranking and the full retriever."""
from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest

from app.rag.embeddings import HashEmbedder, build_embedder
from app.rag.hybrid_search import fuse
from app.rag.reranker import FastEmbedReranker, LexicalReranker, NoopReranker, build_reranker
from app.rag.vector_store import LocalVectorStore, QdrantVectorStore, SearchFilter, VectorPoint, build_vector_store
from app.utils.errors import VectorStoreError
from tests.conftest import make_settings


class TestEmbeddings:
    def test_hash_embedder_is_deterministic_and_normalised(self):
        embedder = HashEmbedder(128)
        a, b = embedder.embed_query("debt to equity ratio"), embedder.embed_query("debt to equity ratio")
        assert a == b and len(a) == embedder.dimension == 128
        assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)

    def test_similar_text_scores_higher(self):
        embedder = HashEmbedder(384)
        query = embedder.embed_query("interest rate risk on borrowings")
        related, unrelated = embedder.embed_documents([
            "Borrowings carry floating interest rates, exposing us to interest rate risk.",
            "The cafeteria menu was refreshed with seasonal vegetables.",
        ])
        dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
        assert dot(query, related) > dot(query, unrelated) + 0.1

    def test_factory(self, tmp_path: Path):
        assert build_embedder(make_settings(tmp_path)).name == "hash-384"
        fast = build_embedder(make_settings(tmp_path, embedding_provider="fastembed"))
        assert fast.name == "BAAI/bge-small-en-v1.5"   # constructed lazily, no download until first use


def _points() -> list[VectorPoint]:
    return [
        VectorPoint("a", [1.0, 0.0, 0.0], {"document_id": "d1", "company_id": 1, "fiscal_year": 2025, "document_type": "annual_report"}),
        VectorPoint("b", [0.8, 0.6, 0.0], {"document_id": "d1", "company_id": 1, "fiscal_year": 2025, "document_type": "annual_report"}),
        VectorPoint("c", [0.0, 1.0, 0.0], {"document_id": "d2", "company_id": 2, "fiscal_year": 2024, "document_type": "earnings_call_transcript"}),
    ]


class TestLocalVectorStore:
    def test_search_orders_by_cosine(self):
        store = LocalVectorStore()
        store.upsert(_points())
        hits = store.search([1.0, 0.0, 0.0], limit=3)
        assert [h.id for h in hits] == ["a", "b", "c"]
        assert hits[0].score == pytest.approx(1.0) and hits[1].score == pytest.approx(0.8)

    def test_metadata_filters(self):
        store = LocalVectorStore()
        store.upsert(_points())
        assert [h.id for h in store.search([1, 0, 0], 5, SearchFilter(company_ids=[2]))] == ["c"]
        assert [h.id for h in store.search([1, 0, 0], 5, SearchFilter(fiscal_years=[2025], document_ids=["d1"]))] == ["a", "b"]
        assert store.search([1, 0, 0], 5, SearchFilter(document_types=["research_report"])) == []

    def test_upsert_replaces_and_delete_removes(self):
        store = LocalVectorStore()
        store.upsert(_points())
        store.upsert([VectorPoint("a", [0.0, 0.0, 1.0], {"document_id": "d1", "company_id": 1})])
        assert store.count() == 3
        assert store.search([0, 0, 1], 1)[0].id == "a"
        store.delete_document("d1")
        assert store.count() == 1 and store.search([1, 0, 0], 5)[0].id == "c"

    def test_persists_to_disk(self, tmp_path: Path):
        LocalVectorStore(tmp_path / "vectors").upsert(_points())
        reopened = LocalVectorStore(tmp_path / "vectors")
        assert reopened.count() == 3
        assert reopened.search([0, 1, 0], 1)[0].id == "c"

    def test_dimension_mismatch_is_a_clear_error(self):
        store = LocalVectorStore()
        store.upsert(_points())
        with pytest.raises(VectorStoreError, match="Re-index"):
            store.ensure_collection(8)

    def test_factory_selects_backend(self, tmp_path: Path):
        assert build_vector_store(make_settings(tmp_path)).backend == "memory"
        assert build_vector_store(make_settings(tmp_path, vector_store="auto")).backend == "local"
        assert build_vector_store(make_settings(tmp_path, vector_store="auto", qdrant_url="http://qdrant:6333")).backend == "qdrant"
        with pytest.raises(VectorStoreError):
            build_vector_store(make_settings(tmp_path, vector_store="qdrant"))


class TestQdrantVectorStore:
    """Exercises the REST contract against a stubbed transport."""

    def _store(self, handler) -> QdrantVectorStore:  # noqa: ANN001
        store = QdrantVectorStore("http://qdrant:6333", api_key="secret", collection="test")
        store._client = httpx.Client(base_url="http://qdrant:6333", transport=httpx.MockTransport(handler), headers={"api-key": "secret"})
        return store

    def test_creates_collection_indexes_and_upserts(self):
        calls: list[tuple[str, str, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            calls.append((request.method, request.url.path, body))
            assert request.headers["api-key"] == "secret"
            if request.method == "GET":
                return httpx.Response(404, json={"status": "not found"})
            return httpx.Response(200, json={"result": {}, "status": "ok"})

        self._store(handler).upsert(_points())
        methods = [(m, p) for m, p, _ in calls]
        assert methods[0] == ("GET", "/collections/test")
        assert ("PUT", "/collections/test") in methods
        assert methods.count(("PUT", "/collections/test/index")) == 4
        create = next(b for m, p, b in calls if (m, p) == ("PUT", "/collections/test"))
        assert create == {"vectors": {"size": 3, "distance": "Cosine"}}
        upsert = next(b for m, p, b in calls if p == "/collections/test/points")
        assert [p["id"] for p in upsert["points"]] == ["a", "b", "c"]

    def test_search_builds_filter_and_parses_hits(self):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"result": {"config": {"params": {"vectors": {"size": 3}}}}})
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"result": {"points": [{"id": "a", "score": 0.91, "payload": {"company_id": 1}}]}})

        hits = self._store(handler).search([1, 0, 0], 5, SearchFilter(company_ids=[1, 2], fiscal_years=[2025]))
        assert [(h.id, h.score, h.payload) for h in hits] == [("a", 0.91, {"company_id": 1})]
        assert seen["limit"] == 5 and seen["with_payload"] is True
        assert seen["filter"] == {"must": [
            {"key": "company_id", "match": {"any": [1, 2]}},
            {"key": "fiscal_year", "match": {"any": [2025]}},
        ]}

    def test_errors_are_mapped_without_leaking_details(self):
        def unauthorized(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden: key=secret")

        with pytest.raises(VectorStoreError) as exc:
            self._store(unauthorized).upsert(_points())
        assert "secret" not in exc.value.message

        def down(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(VectorStoreError, match="unreachable"):
            self._store(down).search([1, 0, 0], 3)

    def test_existing_collection_with_other_dimension(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"result": {"config": {"params": {"vectors": {"size": 768}}}}})

        with pytest.raises(VectorStoreError, match="768"):
            self._store(handler).ensure_collection(384)


class TestHybridFusion:
    SEMANTIC = [("a", 0.90), ("b", 0.70), ("c", 0.50)]
    KEYWORD = [("c", 12.0), ("d", 6.0), ("a", 3.0)]

    def test_weighted_fusion_matches_the_formula(self):
        fused = {c.chunk_id: c for c in fuse(self.SEMANTIC, self.KEYWORD, semantic_weight=0.7, keyword_weight=0.3)}
        # min-max normalised: semantic a=1, b=.5, c=0; keyword c=1, d=1/3, a=0
        assert fused["a"].hybrid_score == pytest.approx(0.7)
        assert fused["b"].hybrid_score == pytest.approx(0.35)
        assert fused["c"].hybrid_score == pytest.approx(0.3)
        assert fused["d"].hybrid_score == pytest.approx(0.1)
        assert fused["d"].semantic_score == 0.0 and fused["d"].keyword_score == 6.0

    def test_weights_change_the_ranking(self):
        semantic_first = [c.chunk_id for c in fuse(self.SEMANTIC, self.KEYWORD, semantic_weight=0.9, keyword_weight=0.1)]
        keyword_first = [c.chunk_id for c in fuse(self.SEMANTIC, self.KEYWORD, semantic_weight=0.1, keyword_weight=0.9)]
        assert semantic_first[0] == "a" and keyword_first[0] == "c"

    def test_rrf(self):
        fused = fuse(self.SEMANTIC, self.KEYWORD, semantic_weight=0.5, keyword_weight=0.5, method="rrf")
        assert {c.chunk_id for c in fused} == {"a", "b", "c", "d"}
        assert fused[0].chunk_id in ("a", "c")              # present in both lists
        assert fused[-1].chunk_id in ("b", "d")

    def test_one_sided_and_empty(self):
        assert [c.chunk_id for c in fuse(self.SEMANTIC, [])] == ["a", "b", "c"]
        assert [c.chunk_id for c in fuse([], self.KEYWORD)] == ["c", "d", "a"]
        assert fuse([], []) == []


class TestReranker:
    DOCS = [
        "The cafeteria introduced a new seasonal menu for employees.",
        "Borrowings increased and a significant portion carries floating interest rates.",
        "Interest rate risk: a rise in interest rates would reduce interest coverage on our borrowings.",
    ]

    def test_lexical_reranker_orders_by_relevance(self):
        scores = LexicalReranker().score("interest rate risk on borrowings", self.DOCS)
        assert scores[2] > scores[1] > scores[0]

    def test_noop_keeps_order(self):
        assert NoopReranker().score("anything", self.DOCS) is None

    def test_model_reranker_degrades_to_lexical_when_model_cannot_load(self, monkeypatch):
        reranker = FastEmbedReranker("some/missing-model", "unused")
        monkeypatch.setattr(reranker, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
        scores = reranker.score("interest rate risk on borrowings", self.DOCS)
        assert scores[2] > scores[0]
        assert reranker.name.startswith("lexical (fallback")

    def test_factory(self, tmp_path: Path):
        assert build_reranker(make_settings(tmp_path)).name == "lexical"
        assert build_reranker(make_settings(tmp_path, reranker_provider="none")).name == "none"
        assert isinstance(build_reranker(make_settings(tmp_path, reranker_provider="fastembed")), FastEmbedReranker)


@pytest.mark.integration
class TestRetrieverEndToEnd:
    def _retrieve(self, client, query, **filter_kwargs):
        container = client.app.state.container
        return container.retriever.retrieve(query, SearchFilter(**filter_kwargs) if filter_kwargs else None)

    def test_bm25_index_tracks_the_chunk_table(self, client):
        container = client.app.state.container
        hits = container.bm25.search("customer concentration", 5)
        assert hits and container.bm25.size == container.vector_store.count()

    def test_returns_relevant_passage_with_scores_and_metadata(self, client):
        result = self._retrieve(client, "What are the capital expenditure plans for the Gujarat complex?")
        assert 1 <= len(result.chunks) <= client.app.state.container.settings.rerank_top_n
        top = result.chunks[0]
        assert "Gujarat" in top.text and top.company_name == "Aurora Industries Limited"
        assert top.document_title and top.page_start >= 1 and top.section
        assert top.rerank_score is not None and top.hybrid_score > 0
        assert result.semantic_hits > 0 and result.keyword_hits > 0
        assert {"embed_query", "vector_search", "bm25_search", "rerank"} <= set(result.timings_ms)
        scores = [c.rerank_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_statement_tables_are_guaranteed_for_metric_questions(self, client, company_ids):
        result = self._retrieve(client, "What is the debt-to-equity ratio for FY2025?", company_ids=[company_ids["aurora"]])
        tables = [c for c in result.chunks if c.chunk_type == "financial_statement"]
        assert result.statement_hits >= 1 and len(result.chunks) <= 6
        balance_sheet = next(c for c in tables if c.section == "Consolidated Balance Sheet")
        assert balance_sheet.document_title == "Annual Report FY2025" and "| Total equity | 5,500 | 4,950 |" in balance_sheet.text
        # Ordering still follows the reranker; the lane only reserves slots.
        scores = [c.rerank_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_statement_lane_respects_filters_and_is_skipped_for_qualitative_questions(self, client, company_ids):
        borealis = self._retrieve(client, "What was net income?", company_ids=[company_ids["borealis"]])
        assert borealis.statement_hits >= 1 and {c.company_name for c in borealis.chunks} == {"Borealis Technologies Inc."}
        qualitative = self._retrieve(client, "What did management say about customer concentration?")
        assert qualitative.statement_hits == 0

    def test_company_filter_is_enforced(self, client, company_ids):
        result = self._retrieve(client, "revenue growth drivers", company_ids=[company_ids["borealis"]])
        assert result.chunks and {c.company_name for c in result.chunks} == {"Borealis Technologies Inc."}

    def test_document_type_and_year_filters(self, client):
        transcripts = self._retrieve(client, "margin recovery", document_types=["earnings_call_transcript"])
        assert transcripts.chunks and {c.document_type for c in transcripts.chunks} == {"earnings_call_transcript"}
        fy24 = self._retrieve(client, "revenue from operations", fiscal_years=[2024])
        assert fy24.chunks and {c.fiscal_year for c in fy24.chunks} == {2024}

    def test_keyword_search_survives_a_vector_outage(self, client, monkeypatch):
        container = client.app.state.container

        def boom(*_args, **_kwargs):
            raise VectorStoreError("down")

        monkeypatch.setattr(container.vector_store, "search", boom)
        result = container.retriever.retrieve("customer concentration risk")
        assert result.chunks and result.semantic_hits == 0
        assert result.degraded == ["semantic_search_unavailable:vector_store_unavailable"]

    def test_no_match_returns_nothing(self, client):
        result = self._retrieve(client, "zxqv plmokn wertyu")
        assert result.chunks == [] and not result.sufficient
