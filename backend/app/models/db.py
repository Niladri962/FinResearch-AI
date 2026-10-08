"""SQLAlchemy ORM entities."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.models.enums import DocumentStatus


def new_id() -> str:
    # Canonical UUID string — also a valid Qdrant point id.
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON, list[str]: JSON}


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True)  # normalised name
    aliases: Mapped[list[str]] = mapped_column(default=list)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    documents: Mapped[list[Document]] = relationship(back_populates="company")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))            # sanitised original name
    stored_name: Mapped[str] = mapped_column(String(80))           # server-generated name on disk
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(255), default="")
    document_type: Mapped[str] = mapped_column(String(40), default="other")
    fiscal_year: Mapped[int | None] = mapped_column(Integer, index=True)
    quarter: Mapped[int | None] = mapped_column(Integer)
    reporting_period: Mapped[str | None] = mapped_column(String(40))
    currency: Mapped[str | None] = mapped_column(String(8))
    unit: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.QUEUED.value, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(default=list)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    fact_count: Mapped[int] = mapped_column(Integer, default=0)
    # User-supplied overrides captured at upload; they win over auto-detection.
    overrides: Mapped[dict[str, Any]] = mapped_column(default=dict)
    uploaded_at: Mapped[datetime] = mapped_column(default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column()

    company: Mapped[Company | None] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")
    facts: Mapped[list[FinancialFact]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    company_id: Mapped[int | None] = mapped_column(Integer, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(255), default="")
    chunk_type: Mapped[str] = mapped_column(String(40), default="narrative")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ChunkVector(Base):
    """Embeddings kept in the SQL database (``VECTOR_STORE=database``).

    Used where there is no persistent disk and no dedicated vector database.
    Search is exact and done in memory, which suits thousands of passages.
    """

    __tablename__ = "chunk_vectors"

    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    dimension: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)            # float32, little-endian
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class FinancialFact(Base):
    """One reported value: metric × period, traceable to a page and chunk."""

    __tablename__ = "financial_facts"
    __table_args__ = (Index("ix_fact_lookup", "company_id", "metric", "fiscal_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int | None] = mapped_column(Integer, index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[str | None] = mapped_column(String(36))
    metric: Mapped[str] = mapped_column(String(60))               # canonical key, e.g. "revenue"
    label: Mapped[str] = mapped_column(String(255))                # row label as reported
    period_label: Mapped[str] = mapped_column(String(30))          # "FY2025", "Q1 FY2025"
    period_kind: Mapped[str] = mapped_column(String(10), default="annual")
    fiscal_year: Mapped[int] = mapped_column(Integer)
    quarter: Mapped[int | None] = mapped_column(Integer)
    value: Mapped[float] = mapped_column(Float)                    # as reported
    scale: Mapped[float] = mapped_column(Float, default=1.0)       # multiplier to absolute units
    unit: Mapped[str] = mapped_column(String(20), default="")      # "crore", "million", …
    currency: Mapped[str] = mapped_column(String(8), default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    document: Mapped[Document] = relationship(back_populates="facts")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(200), default="New research thread")
    owner: Mapped[str] = mapped_column(String(80), default="anonymous", index=True)
    context: Mapped[dict[str, Any]] = mapped_column(default=dict)  # carried company ids etc.
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(12))
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)  # citations, artifacts, validation
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[Any]] = mapped_column(default=list)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class QueryTrace(Base):
    __tablename__ = "query_traces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), index=True)
    query_hash: Mapped[str] = mapped_column(String(16))
    query_preview: Mapped[str | None] = mapped_column(String(220))
    intent: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(12), default="ok")
    error: Mapped[str | None] = mapped_column(String(120))
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict)     # timings, usage, source ids
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
