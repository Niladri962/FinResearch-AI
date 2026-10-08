"""BM25 keyword index.

Built from the ``chunks`` table and held in memory. The index rebuilds itself
whenever the chunk set changes (detected by a cheap count/latest-timestamp
signature), so it stays correct across uploads, deletions and multiple workers
without any explicit invalidation calls.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select

from app.models.database import SessionFactory
from app.models.db import Chunk, Company, Document
from app.rag.vector_store import SearchFilter
from app.utils.logging import get_logger
from app.utils.text import tokenize

logger = get_logger(__name__)


@dataclass
class _Entry:
    chunk_id: str
    payload: dict


class BM25Index:
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._bm25: BM25Okapi | None = None
        self._entries: list[_Entry] = []

    def _current_signature(self, session) -> tuple:  # noqa: ANN001
        count, latest = session.execute(select(func.count(Chunk.id), func.max(Chunk.created_at))).one()
        return (count, str(latest))

    def _ensure(self) -> None:
        with self._session_factory() as session:
            signature = self._current_signature(session)
            if signature == self._signature:
                return
            with self._lock:
                if signature == self._signature:
                    return
                rows = session.execute(
                    select(
                        Chunk.id, Chunk.text, Chunk.section, Chunk.document_id, Chunk.company_id,
                        Document.fiscal_year, Document.document_type, Document.title, Company.name,
                    )
                    .join(Document, Document.id == Chunk.document_id)
                    .outerjoin(Company, Company.id == Document.company_id)
                ).all()
                corpus, entries = [], []
                for chunk_id, text, section, document_id, company_id, fiscal_year, document_type, title, company in rows:
                    # Provenance is indexed with the body: a statement table never repeats the company
                    # name or the document period, yet both are how users refer to it.
                    corpus.append(tokenize(f"{company or ''} {title or ''} {section} {text}"))
                    entries.append(
                        _Entry(chunk_id, {
                            "document_id": document_id, "company_id": company_id,
                            "fiscal_year": fiscal_year, "document_type": document_type,
                        })
                    )
                self._bm25 = BM25Okapi(corpus) if corpus else None
                self._entries = entries
                self._signature = signature
                logger.info("bm25_index_built", extra={"chunks": len(entries)})

    def search(self, query: str, limit: int, flt: SearchFilter | None = None) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, bm25_score)]`` for chunks with a positive score."""
        self._ensure()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        if flt and flt.clauses():
            mask = np.fromiter((flt.matches(e.payload) for e in self._entries), dtype=bool, count=len(self._entries))
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores)[:limit]
        return [(self._entries[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]

    @property
    def size(self) -> int:
        return len(self._entries)
