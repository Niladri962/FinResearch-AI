"""Research agent: gathers evidence with hybrid retrieval + reranking."""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import select

from app.agents.state import AgentDeps, AgentState, RunContext
from app.guardrails.input_guard import is_suspicious_context
from app.models.db import Document
from app.models.enums import DocumentStatus, Intent
from app.models.schemas import RetrievedChunk
from app.rag.retriever import RetrievalResult
from app.rag.vector_store import SearchFilter

# Extra retrieval terms per intent. They steer both the embedding and BM25 towards
# the sections where that kind of answer usually lives.
_EXPANSIONS: dict[Intent, str] = {
    Intent.RISK_ANALYSIS: "risk factors key risks uncertainties threats",
    Intent.MANAGEMENT_ANALYSIS: "management commentary outlook guidance expectations",
    Intent.SUMMARY: "key highlights performance results summary",
    Intent.TREND_ANALYSIS: "growth year over year change reasons",
}
_BROAD_INTENTS = {Intent.SUMMARY, Intent.RISK_ANALYSIS, Intent.COMPANY_COMPARISON, Intent.PERIOD_COMPARISON}


class ResearchAgent:
    name = "research"

    def __init__(self, deps: AgentDeps) -> None:
        self._deps = deps

    def _available_types(self, company_ids: list[int], wanted: list[str]) -> list[str]:
        """Only filter by document type when such documents actually exist in scope."""
        if not wanted:
            return []
        with self._deps.session_factory() as session:
            query = select(Document.document_type).where(
                Document.status == DocumentStatus.READY.value, Document.document_type.in_(wanted)
            )
            if company_ids:
                query = query.where(Document.company_id.in_(company_ids))
            return sorted(set(session.execute(query).scalars()))

    async def run(self, state: AgentState, run: RunContext) -> dict[str, Any]:
        understanding = state["understanding"]
        filters = state["filters"]
        settings = self._deps.settings
        intent = understanding.intent

        company_ids = [c.id for c in understanding.companies] or list(filters.company_ids)
        doc_types = list(filters.document_types) or self._available_types(company_ids, understanding.document_types)
        query = understanding.query
        if intent in _EXPANSIONS:
            query = f"{query} {_EXPANSIONS[intent]}"
        top_n = settings.rerank_top_n + (2 if intent in _BROAD_INTENTS else 0)

        def make_filter(companies: list[int]) -> SearchFilter:
            return SearchFilter(
                company_ids=companies, document_ids=list(filters.document_ids),
                fiscal_years=list(filters.fiscal_years), document_types=doc_types,
            )

        # For a company comparison, retrieve per company so each side is represented.
        if intent == Intent.COMPANY_COMPARISON and len(company_ids) >= 2:
            per_company = max(3, top_n // len(company_ids))
            searches = [(make_filter([cid]), per_company) for cid in company_ids]
        else:
            searches = [(make_filter(company_ids), top_n)]

        run.status("retrieval", "Searching documents (semantic + keyword)…")
        with run.trace.span("retrieval"):
            results: list[RetrievalResult] = await asyncio.gather(
                *(asyncio.to_thread(self._deps.retriever.retrieve, query, flt, n) for flt, n in searches)
            )

        evidence: list[RetrievedChunk] = list(state.get("evidence", []))
        seen = {c.chunk_id for c in evidence}
        notes = list(state.get("notes", []))
        dropped = 0
        # Interleave so every company keeps its best hits when the list is trimmed.
        for rank in range(max((len(r.chunks) for r in results), default=0)):
            for result in results:
                if rank >= len(result.chunks):
                    continue
                chunk = result.chunks[rank]
                if chunk.chunk_id in seen:
                    continue
                if is_suspicious_context(chunk.text):
                    dropped += 1  # indirect prompt injection planted in a document
                    continue
                seen.add(chunk.chunk_id)
                evidence.append(chunk)

        for result in results:
            for name, ms in result.timings_ms.items():
                run.trace.add_timing(name, ms)
            for issue in result.degraded:
                if "Semantic search was unavailable; results are keyword-only." not in notes:
                    notes.append("Semantic search was unavailable; results are keyword-only.")
                run.trace.set(degraded=issue)
        if dropped:
            notes.append(f"{dropped} passage(s) were excluded because they contained embedded instructions.")
        names = sorted({c.company_name for c in evidence if c.company_name})
        if not company_ids and len(names) > 1:
            notes.append(
                "The question does not name a company, so evidence was drawn from all uploaded companies: "
                + ", ".join(names) + "."
            )
        run.trace.set(
            retrieved_chunks=len(evidence),
            candidates=sum(r.candidates for r in results),
            source_chunk_ids=[c.chunk_id for c in evidence],
            source_document_ids=sorted({c.document_id for c in evidence}),
        )
        return {"evidence": evidence, "notes": notes}
