"""Hybrid retriever: semantic + BM25 → fusion → rerank → top-N contexts.

A third, structured lane handles the weak spot of text retrieval in finance:
statement tables. When a query names a metric or ratio, the tables those
figures were extracted from are looked up through ``financial_facts`` and
guaranteed a small number of evidence slots, so the numbers behind an answer
are always citeable even when prose that merely mentions the metric ranks higher.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import select

from app.config import Settings
from app.financial.periods import find_periods
from app.financial.ratios import query_metric_keys
from app.models.database import SessionFactory
from app.models.db import Chunk, Company, Document, FinancialFact
from app.models.schemas import RetrievedChunk
from app.rag.bm25 import BM25Index
from app.rag.embeddings import Embedder
from app.rag.hybrid_search import FusedCandidate, fuse
from app.rag.reranker import Reranker
from app.rag.vector_store import SearchFilter, VectorStore
from app.utils.cache import Cache
from app.utils.errors import AppError
from app.utils.logging import get_logger
from app.utils.text import sha256_text

logger = get_logger(__name__)

STATEMENT_SLOTS = 2          # evidence slots reserved for statement tables
STATEMENT_CANDIDATES = 6     # tables considered per query


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    candidates: int = 0
    semantic_hits: int = 0
    keyword_hits: int = 0
    statement_hits: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return bool(self.chunks)


def rerank_text(chunk: RetrievedChunk) -> str:
    """What the reranker reads: provenance first, then the passage."""
    header = " | ".join(p for p in (chunk.company_name, chunk.document_title, chunk.section) if p)
    return f"{header}\n{chunk.text}" if header else chunk.text


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        session_factory: SessionFactory,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        cache: Cache,
    ) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._embedder = embedder
        self._vectors = vector_store
        self._bm25 = bm25
        self._reranker = reranker
        self._cache = cache

    def _embed_query(self, query: str) -> list[float]:
        key = f"qemb:{self._embedder.name}:{sha256_text(query)}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vector = self._embedder.embed_query(query)
        self._cache.set(key, vector, ttl_seconds=3600)
        return vector

    def _statement_lane(self, query: str, flt: SearchFilter | None) -> dict[str, tuple[int, int]]:
        """Statement-table chunks holding the metrics a query asks about.

        Returns ``{chunk_id: priority}`` (lower sorts first): tables covering more of
        the requested line items, then tables for the period asked about (or the
        latest period when none is named).
        """
        keys = query_metric_keys(query)
        if not keys:
            return {}
        statement = (
            select(FinancialFact.chunk_id, FinancialFact.metric, FinancialFact.fiscal_year)
            .join(Document, Document.id == FinancialFact.document_id)
            .where(
                FinancialFact.metric.in_(keys), FinancialFact.period_kind == "annual",
                FinancialFact.chunk_id.is_not(None),
            )
        )
        if flt is not None:
            if flt.company_ids:
                statement = statement.where(FinancialFact.company_id.in_(flt.company_ids))
            if flt.document_ids:
                statement = statement.where(FinancialFact.document_id.in_(flt.document_ids))
            if flt.fiscal_years:
                statement = statement.where(Document.fiscal_year.in_(flt.fiscal_years))
            if flt.document_types:
                statement = statement.where(Document.document_type.in_(flt.document_types))
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        if not rows:
            return {}

        metrics: dict[str, set[str]] = {}
        years: dict[str, set[int]] = {}
        for chunk_id, metric, fiscal_year in rows:
            metrics.setdefault(chunk_id, set()).add(metric)
            years.setdefault(chunk_id, set()).add(fiscal_year)
        wanted_years = {p.fiscal_year for p in find_periods(query)} or {max(y for ys in years.values() for y in ys)}
        priority = {
            chunk_id: (-len(found), 0 if years[chunk_id] & wanted_years else 1)
            for chunk_id, found in metrics.items()
        }
        best = sorted(priority, key=lambda cid: (priority[cid], cid))[:STATEMENT_CANDIDATES]
        return {cid: priority[cid] for cid in best}

    def _hydrate(self, candidates: list[FusedCandidate]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        by_id = {c.chunk_id: c for c in candidates}
        with self._session_factory() as session:
            rows = session.execute(
                select(Chunk, Document, Company.name)
                .join(Document, Document.id == Chunk.document_id)
                .outerjoin(Company, Company.id == Document.company_id)
                .where(Chunk.id.in_(list(by_id)))
            ).all()
        hydrated = []
        for chunk, document, company_name in rows:
            candidate = by_id[chunk.id]
            hydrated.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title or document.filename,
                    filename=document.filename,
                    company_id=document.company_id,
                    company_name=company_name,
                    fiscal_year=document.fiscal_year,
                    document_type=document.document_type,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    chunk_type=chunk.chunk_type,
                    semantic_score=round(candidate.semantic_score, 4),
                    keyword_score=round(candidate.keyword_score, 4),
                    hybrid_score=round(candidate.hybrid_score, 4),
                )
            )
        # Chunks deleted between search and hydration simply drop out.
        return sorted(hydrated, key=lambda c: -c.hybrid_score)

    def retrieve(self, query: str, flt: SearchFilter | None = None, top_n: int | None = None) -> RetrievalResult:
        s = self.settings
        result = RetrievalResult()
        top_n = top_n or s.rerank_top_n

        # 1. Semantic search
        semantic: list[tuple[str, float]] = []
        start = time.perf_counter()
        try:
            vector = self._embed_query(query)
            result.timings_ms["embed_query"] = round((time.perf_counter() - start) * 1000, 1)
            start = time.perf_counter()
            semantic = [(hit.id, hit.score) for hit in self._vectors.search(vector, s.semantic_top_k, flt)]
            result.timings_ms["vector_search"] = round((time.perf_counter() - start) * 1000, 1)
        except AppError as exc:
            # Keyword search still works when embeddings or the vector DB are down.
            result.degraded.append(f"semantic_search_unavailable:{exc.code}")
            logger.warning("Semantic search failed; continuing with keyword search", extra={"code": exc.code})

        # 2. Keyword search
        start = time.perf_counter()
        keyword = self._bm25.search(query, s.bm25_top_k, flt)
        result.timings_ms["bm25_search"] = round((time.perf_counter() - start) * 1000, 1)
        result.semantic_hits, result.keyword_hits = len(semantic), len(keyword)

        # 3. Fuse, and add statement tables for any metric the query names
        fused = fuse(
            semantic, keyword,
            semantic_weight=s.semantic_weight, keyword_weight=s.keyword_weight, method=s.fusion_method,
        )
        lane = self._statement_lane(query, flt)
        known = {c.chunk_id for c in fused}
        fused.extend(FusedCandidate(cid) for cid in lane if cid not in known)
        chunks = self._hydrate(fused)
        result.candidates = len(chunks)

        # Drop candidates with no real signal: nothing lexical and a weak embedding match.
        chunks = [
            c for c in chunks
            if c.chunk_id in lane or c.keyword_score > 0 or c.semantic_score >= s.min_semantic_score
        ]

        # 4. Rerank
        start = time.perf_counter()
        if chunks:
            scores = self._reranker.score(query, [rerank_text(c) for c in chunks])
            if scores is not None:
                for chunk, score in zip(chunks, scores):
                    chunk.rerank_score = round(float(score), 4)
                chunks.sort(key=lambda c: (-(c.rerank_score or 0.0), -c.hybrid_score))
        result.timings_ms["rerank"] = round((time.perf_counter() - start) * 1000, 1)

        # 5. Select: best by rerank, with reserved slots for the statement tables
        selected = chunks[:top_n]
        if lane:
            slots = STATEMENT_SLOTS if top_n >= 5 else 1
            order = {c.chunk_id: i for i, c in enumerate(chunks)}
            tables = sorted((c for c in chunks if c.chunk_id in lane), key=lambda c: (lane[c.chunk_id], order[c.chunk_id]))
            reserved = tables[:slots]
            missing = [c for c in reserved if c not in selected]
            if missing:
                keep = [c for c in selected if c not in reserved][: top_n - len(reserved)]
                selected = sorted([*keep, *reserved], key=lambda c: order[c.chunk_id])
            result.statement_hits = len(reserved)
        result.chunks = selected
        return result
