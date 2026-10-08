"""Dashboard aggregates and recent-activity feeds."""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.models.database import SessionFactory
from app.models.db import Chunk, Company, Conversation, Document, FinancialFact, Message, QueryTrace, Report
from app.models.enums import DocumentStatus
from app.models.schemas import AnalyzeRequest, ConversationOut, DashboardResponse
from app.services.analysis import AnalysisService
from app.services.documents import document_out
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DashboardService:
    def __init__(self, session_factory: SessionFactory, analysis: AnalysisService) -> None:
        self._session_factory = session_factory
        self._analysis = analysis

    def _performance(self, session) -> dict[str, Any]:  # noqa: ANN001
        traces = session.execute(
            select(QueryTrace).order_by(QueryTrace.created_at.desc()).limit(50)
        ).scalars().all()
        if not traces:
            return {"queries": 0}
        ok = [t for t in traces if t.status == "ok"]

        def avg(key: str) -> float | None:
            values = [t.data.get("timings_ms", {}).get(key) for t in ok]
            values = [v for v in values if v is not None]
            return round(sum(values) / len(values), 1) if values else None

        scores = [t.data.get("grounding_score") for t in ok if t.data.get("grounding_score") is not None]
        intents: dict[str, int] = {}
        for trace in traces:
            if trace.intent:
                intents[trace.intent] = intents.get(trace.intent, 0) + 1
        return {
            "queries": len(traces),
            "errors": len(traces) - len(ok),
            "avg_total_ms": round(sum(t.total_ms for t in ok) / len(ok), 1) if ok else None,
            "avg_retrieval_ms": avg("retrieval"),
            "avg_rerank_ms": avg("rerank"),
            "avg_llm_ms": avg("llm"),
            "avg_grounding_score": round(sum(scores) / len(scores), 3) if scores else None,
            "total_tokens": sum(t.data.get("usage", {}).get("total_tokens", 0) for t in traces),
            "intents": intents,
        }

    async def overview(self, owner: str = "anonymous") -> DashboardResponse:
        with self._session_factory() as session:
            def count(model, *where) -> int:  # noqa: ANN001
                return session.execute(select(func.count()).select_from(model).where(*where)).scalar_one()

            status_counts = dict(session.execute(select(Document.status, func.count()).group_by(Document.status)).all())
            questions = session.execute(
                select(Message, Conversation.title)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(Message.role == "user", Conversation.owner == owner)
                .order_by(Message.created_at.desc()).limit(8)
            ).all()
            conversations = session.execute(
                select(Conversation, func.count(Message.id))
                .outerjoin(Message, Message.conversation_id == Conversation.id)
                .where(Conversation.owner == owner)
                .group_by(Conversation.id).order_by(Conversation.updated_at.desc()).limit(6)
            ).all()
            documents = session.execute(
                select(Document, Company.name).outerjoin(Company, Company.id == Document.company_id)
                .order_by(Document.uploaded_at.desc()).limit(6)
            ).all()
            # Spotlight the company with the most structured data, newest upload first on ties.
            spotlight_id = session.execute(
                select(FinancialFact.company_id)
                .where(FinancialFact.company_id.is_not(None))
                .group_by(FinancialFact.company_id)
                .order_by(func.count(func.distinct(FinancialFact.fiscal_year)).desc(), func.max(FinancialFact.id).desc())
                .limit(1)
            ).scalar_one_or_none()

            recent_research = []
            for conversation, message_count in conversations:
                item = ConversationOut.model_validate(conversation)
                item.message_count = message_count
                recent_research.append(item)

            response = DashboardResponse(
                companies=count(Company),
                documents=sum(status_counts.values()),
                documents_ready=status_counts.get(DocumentStatus.READY.value, 0),
                documents_processing=status_counts.get(DocumentStatus.PROCESSING.value, 0)
                + status_counts.get(DocumentStatus.QUEUED.value, 0),
                documents_failed=status_counts.get(DocumentStatus.FAILED.value, 0),
                chunks=count(Chunk),
                facts=count(FinancialFact),
                conversations=count(Conversation, Conversation.owner == owner),
                reports=count(Report),
                recent_questions=[
                    {"conversation_id": m.conversation_id, "question": m.content[:200], "asked_at": m.created_at.isoformat()}
                    for m, _title in questions
                ],
                recent_research=recent_research,
                recent_documents=[document_out(doc, name) for doc, name in documents],
                performance=self._performance(session),
            )
        if spotlight_id is not None:
            try:
                response.spotlight = await self._analysis.analyze(AnalyzeRequest(company_id=spotlight_id))
            except Exception:
                logger.exception("Dashboard spotlight failed")
        return response

    def traces(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = session.execute(
                select(QueryTrace).order_by(QueryTrace.created_at.desc()).limit(limit)
            ).scalars().all()
            return [
                {
                    "id": t.id, "created_at": t.created_at.isoformat(), "intent": t.intent, "status": t.status,
                    "error": t.error, "total_ms": t.total_ms, "query_hash": t.query_hash,
                    "query_preview": t.query_preview, **{k: v for k, v in t.data.items() if k in (
                        "timings_ms", "usage", "plan", "retrieved_chunks", "candidates", "validation_status",
                        "grounding_score", "mode", "llm_model", "source_document_ids", "guard_action",
                    )},
                }
                for t in rows
            ]
