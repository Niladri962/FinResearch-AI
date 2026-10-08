"""Research chat: conversation persistence around the agent graph."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from sqlalchemy import func, select

from app.agents.state import AgentState
from app.agents.supervisor import Supervisor
from app.config import Settings
from app.guardrails.output_guard import DISCLAIMER
from app.models.database import SessionFactory, session_scope
from app.models.db import Conversation, Message, QueryTrace
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationOut,
    MessageOut,
    QueryFilters,
    ValidationReport,
)
from app.utils.errors import AppError, NotFoundError
from app.utils.logging import get_logger
from app.utils.observability import Trace
from app.utils.text import truncate

logger = get_logger(__name__)

HISTORY_TURNS = 6
GENERIC_ERROR = "Something went wrong while answering. Please try again."


def persist_trace(session_factory: SessionFactory, trace: Trace) -> None:
    summary = trace.summary()
    try:
        with session_scope(session_factory) as session:
            session.add(QueryTrace(
                id=trace.id, conversation_id=trace.conversation_id, query_hash=trace.query_hash,
                query_preview=trace.query_preview, intent=trace.attributes.get("intent"),
                status=trace.status, error=trace.error, total_ms=summary["total_ms"], data=summary,
            ))
    except Exception:  # observability must never break a response
        logger.exception("Could not persist query trace")
    trace.log()


def response_from_state(state: AgentState, *, conversation_id: str, message_id: str, trace: Trace) -> ChatResponse:
    understanding = state["understanding"]
    return ChatResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        answer=state.get("answer", ""),
        intent=understanding.intent.value,
        intent_confidence=understanding.confidence,
        mode=state.get("mode", "generative"),  # type: ignore[arg-type]
        citations=state.get("citations", []),
        sources=state.get("sources", []),
        calculations=state.get("calculations", []),
        tables=state.get("tables", []),
        charts=state.get("charts", []),
        validation=state.get("validation", ValidationReport()),
        followups=state.get("followups", []),
        notes=state.get("notes", []),
        disclaimer=DISCLAIMER,
        trace={
            "trace_id": trace.id, "total_ms": trace.total_ms, "timings_ms": dict(trace.timings_ms),
            "usage": dict(trace.usage), "plan": trace.attributes.get("plan", []),
            "retrieved_chunks": trace.attributes.get("retrieved_chunks", 0),
            "llm_model": trace.attributes.get("llm_model"),
        },
    )


class ChatService:
    def __init__(self, settings: Settings, session_factory: SessionFactory, supervisor: Supervisor) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._supervisor = supervisor

    # ── Conversations ────────────────────────────────────────────────────
    def _open(self, request: ChatRequest, owner: str) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
        """Load or create the conversation, store the user turn, return history and context."""
        with session_scope(self._session_factory) as session:
            conversation = None
            if request.conversation_id:
                conversation = session.get(Conversation, request.conversation_id)
                if conversation is None or conversation.owner != owner:
                    raise NotFoundError("Conversation not found.")
            if conversation is None:
                conversation = Conversation(title=truncate(request.message, 80), owner=owner, context={})
                session.add(conversation)
                session.flush()
            recent = session.execute(
                select(Message).where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.desc()).limit(HISTORY_TURNS)
            ).scalars().all()
            history = [{"role": m.role, "content": m.content} for m in reversed(recent)]
            session.add(Message(conversation_id=conversation.id, role="user", content=request.message, payload={}))
            return conversation.id, history, dict(conversation.context or {})

    def _close(self, conversation_id: str, state: AgentState, trace: Trace) -> ChatResponse:
        with session_scope(self._session_factory) as session:
            message = Message(conversation_id=conversation_id, role="assistant", content=state.get("answer", ""), payload={})
            session.add(message)
            session.flush()
            response = response_from_state(state, conversation_id=conversation_id, message_id=message.id, trace=trace)
            message.payload = response.model_dump(mode="json", exclude={"answer", "conversation_id", "message_id"})
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                companies = [c.id for c in state["understanding"].companies]
                if companies:
                    conversation.context = {**(conversation.context or {}), "company_ids": companies}
                conversation.updated_at = message.created_at
            return response

    def list_conversations(self, owner: str, limit: int = 50) -> list[ConversationOut]:
        with self._session_factory() as session:
            rows = session.execute(
                select(Conversation, func.count(Message.id))
                .outerjoin(Message, Message.conversation_id == Conversation.id)
                .where(Conversation.owner == owner)
                .group_by(Conversation.id).order_by(Conversation.updated_at.desc()).limit(limit)
            ).all()
            out = []
            for conversation, count in rows:
                item = ConversationOut.model_validate(conversation)
                item.message_count = count
                out.append(item)
            return out

    def get_conversation(self, conversation_id: str, owner: str) -> ConversationDetail:
        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None or conversation.owner != owner:
                raise NotFoundError("Conversation not found.")
            return ConversationDetail(
                id=conversation.id, title=conversation.title, created_at=conversation.created_at,
                updated_at=conversation.updated_at, message_count=len(conversation.messages),
                messages=[MessageOut.model_validate(m) for m in conversation.messages],
            )

    def delete_conversation(self, conversation_id: str, owner: str) -> None:
        with session_scope(self._session_factory) as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None or conversation.owner != owner:
                raise NotFoundError("Conversation not found.")
            session.delete(conversation)

    # ── Answering ────────────────────────────────────────────────────────
    async def stream(self, request: ChatRequest, owner: str = "anonymous") -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event, data)`` pairs for SSE. Always ends with ``final`` or ``error``."""
        trace = Trace(request.message, store_query_text=self.settings.trace_query_text)
        try:
            conversation_id, history, context = await asyncio.to_thread(self._open, request, owner)
            trace.conversation_id = conversation_id
            yield "conversation", {"conversation_id": conversation_id}
            async for event, data in self._supervisor.stream(
                query=request.message, trace=trace, history=history, filters=request.filters, context=context,
            ):
                if event == "result":
                    response = await asyncio.to_thread(self._close, conversation_id, data["state"], trace)
                    yield "final", response.model_dump(mode="json")
                elif event == "error":
                    raise data["exception"]
                else:
                    yield event, data
        except AppError as exc:
            trace.fail(exc)
            yield "error", {"code": exc.code, "message": exc.message}
        except asyncio.CancelledError:
            trace.status, trace.error = "cancelled", "ClientDisconnected"
            raise
        except Exception as exc:
            trace.fail(exc)
            logger.exception("chat_failed", extra={"trace_id": trace.id})
            yield "error", {"code": "internal_error", "message": GENERIC_ERROR}
        finally:
            await asyncio.shield(asyncio.to_thread(persist_trace, self._session_factory, trace))

    async def respond(self, request: ChatRequest, owner: str = "anonymous") -> ChatResponse:
        async for event, data in self.stream(request, owner):
            if event == "final":
                return ChatResponse.model_validate(data)
            if event == "error":
                raise AppError(data["message"]) if data["code"] == "internal_error" else _rebuild_error(data)
        raise AppError(GENERIC_ERROR)  # pragma: no cover

    async def ask(self, question: str, filters: QueryFilters | None = None) -> AgentState:
        """Run the pipeline without creating a conversation (used by analysis and reports)."""
        trace = Trace(question, store_query_text=False)
        try:
            return await self._supervisor.run(query=question, trace=trace, filters=filters or QueryFilters())
        except Exception as exc:
            trace.fail(exc)
            raise
        finally:
            await asyncio.to_thread(persist_trace, self._session_factory, trace)


def _rebuild_error(data: dict[str, Any]) -> AppError:
    error = AppError(data["message"])
    error.code = data["code"]
    error.status_code = {
        "llm_timeout": 504, "llm_error": 502, "llm_not_configured": 503, "vector_store_unavailable": 503,
        "embedding_failed": 502, "not_found": 404, "validation_error": 400,
    }.get(data["code"], 500)
    return error
