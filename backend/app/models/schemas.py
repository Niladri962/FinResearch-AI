"""Pydantic request/response schemas and the value objects passed between layers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── Errors ───────────────────────────────────────────────────────────────────
class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


# ── Companies & documents ────────────────────────────────────────────────────
class CompanyOut(APIModel):
    id: int
    name: str
    document_count: int = 0
    fiscal_years: list[int] = Field(default_factory=list)
    periods_with_data: list[str] = Field(default_factory=list)
    currency: str | None = None
    unit: str | None = None


class DocumentOut(APIModel):
    id: str
    filename: str
    title: str
    company_id: int | None
    company_name: str | None = None
    document_type: str
    document_type_label: str = ""
    fiscal_year: int | None
    quarter: int | None
    reporting_period: str | None
    currency: str | None
    unit: str | None
    status: str
    progress: str | None = None      # live stage while processing, e.g. "Embedding passages 640 of 2072"
    error: str | None
    warnings: list[str] = Field(default_factory=list)
    page_count: int
    chunk_count: int
    fact_count: int
    size_bytes: int
    uploaded_at: datetime
    processed_at: datetime | None


class UploadResult(BaseModel):
    filename: str
    document: DocumentOut | None = None
    error: ErrorBody | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
    accepted: int
    rejected: int


class ChunkOut(APIModel):
    id: str
    document_id: str
    document_title: str = ""
    company_name: str | None = None
    text: str
    page_start: int
    page_end: int
    section: str
    chunk_type: str


class FactOut(APIModel):
    metric: str
    metric_name: str = ""
    label: str
    period_label: str
    fiscal_year: int
    quarter: int | None
    value: float
    unit: str
    currency: str
    page: int
    chunk_id: str | None
    confidence: float


# ── Retrieval / citations ────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    filename: str
    company_id: int | None = None
    company_name: str | None = None
    fiscal_year: int | None = None
    document_type: str = "other"
    text: str
    page_start: int
    page_end: int
    section: str = ""
    chunk_type: str = "narrative"
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    id: str                       # "S1"
    chunk_id: str
    document_id: str
    document_title: str
    filename: str
    company_name: str | None = None
    page: int
    page_end: int | None = None
    section: str = ""
    chunk_type: str = "narrative"
    snippet: str = ""
    score: float | None = None

    @property
    def label(self) -> str:
        section = f', Section "{self.section}"' if self.section else ""
        return f"{self.document_title}, Page {self.page}{section}"


# ── Computed artifacts ───────────────────────────────────────────────────────
class SourceRef(BaseModel):
    document_id: str
    document_title: str
    page: int | None = None
    chunk_id: str | None = None


class CalcInput(BaseModel):
    key: str
    name: str
    value: float | None
    display: str
    derived: bool = False
    formula: str | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class Calculation(BaseModel):
    id: str = ""
    kind: Literal["ratio", "growth", "metric", "cagr", "flag", "projection"] = "ratio"
    key: str
    name: str
    category: str = ""
    company: str = ""
    period: str = ""
    formula: str = ""
    value: float | None = None
    unit: str = ""
    display: str = "Not available"
    inputs: list[CalcInput] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    note: str | None = None
    severity: Literal["info", "low", "medium", "high"] | None = None
    change: float | None = None          # vs. the prior period
    change_unit: str = "%"               # "%" for amounts, "pp" for percentage ratios, "x" for multiples


class DataTable(BaseModel):
    id: str = ""
    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str | None = None


class ChartSeries(BaseModel):
    name: str
    data: list[float | None]


class ChartSpec(BaseModel):
    id: str = ""
    title: str
    kind: Literal["line", "bar"] = "line"
    x: list[str]
    series: list[ChartSeries]
    unit: str = ""


class ValidationReport(BaseModel):
    status: Literal["grounded", "partially_grounded", "ungrounded", "not_applicable"] = "not_applicable"
    grounding_score: float | None = None
    cited_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    uncited_numeric_sentences: int = 0
    warnings: list[str] = Field(default_factory=list)


# ── Chat ─────────────────────────────────────────────────────────────────────
class QueryFilters(BaseModel):
    company_ids: list[int] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    filters: QueryFilters = Field(default_factory=QueryFilters)
    stream: bool = True

    @field_validator("message")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    intent: str
    intent_confidence: float = 0.0
    mode: Literal["generative", "extractive", "blocked", "insufficient_evidence", "general_knowledge"]
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    tables: list[DataTable] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    followups: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    trace: dict[str, Any] = Field(default_factory=dict)


class MessageOut(APIModel):
    id: str
    role: str
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationOut(APIModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


# ── Analysis ─────────────────────────────────────────────────────────────────
class RatioRequest(BaseModel):
    """Either reference stored data (``company_id``) or pass raw ``values``."""

    company_id: int | None = None
    period: str | None = None
    ratios: list[str] = Field(default_factory=list)
    values: dict[str, float] = Field(default_factory=dict)
    previous_values: dict[str, float] = Field(default_factory=dict)


class RatioResponse(BaseModel):
    company: str | None = None
    period: str | None = None
    available_periods: list[str] = Field(default_factory=list)
    calculations: list[Calculation]


class AnalyzeRequest(BaseModel):
    company_id: int
    analysis_type: Literal["overview", "trend", "risk"] = "overview"
    metrics: list[str] = Field(default_factory=list)
    include_projection: bool = False
    include_narrative: bool = False


class PeriodSnapshot(BaseModel):
    period: str
    fiscal_year: int
    metrics: dict[str, float | None]
    ratios: dict[str, float | None]


class AnalyzeResponse(BaseModel):
    company: CompanyOut
    analysis_type: str
    periods: list[PeriodSnapshot] = Field(default_factory=list)
    kpis: list[Calculation] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    tables: list[DataTable] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    risk_flags: list[Calculation] = Field(default_factory=list)
    narrative: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""


class CompareRequest(BaseModel):
    """Two or more companies → company comparison; one company → period comparison."""

    company_ids: list[int] = Field(min_length=1, max_length=5)
    periods: list[str] = Field(default_factory=list)
    include_narrative: bool = True


class CompareResponse(BaseModel):
    mode: Literal["company", "period"]
    table: DataTable
    charts: list[ChartSpec] = Field(default_factory=list)
    narrative: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = ""


# ── Reports ──────────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    company_id: int
    title: str | None = Field(default=None, max_length=200)


class ReportOut(APIModel):
    id: str
    company_id: int | None
    company_name: str | None = None
    title: str
    created_at: datetime
    meta: dict[str, Any] = Field(default_factory=dict)


class ReportDetail(ReportOut):
    content: str
    sources: list[Citation] = Field(default_factory=list)


# ── Dashboard / system ───────────────────────────────────────────────────────
class DashboardResponse(BaseModel):
    companies: int
    documents: int
    documents_ready: int
    documents_processing: int
    documents_failed: int
    chunks: int
    facts: int
    conversations: int
    reports: int
    recent_questions: list[dict[str, Any]] = Field(default_factory=list)
    recent_research: list[ConversationOut] = Field(default_factory=list)
    recent_documents: list[DocumentOut] = Field(default_factory=list)
    spotlight: AnalyzeResponse | None = None
    performance: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SystemInfo(BaseModel):
    app_name: str
    version: str
    environment: str
    llm: dict[str, Any]
    embeddings: dict[str, Any]
    reranker: dict[str, Any]
    retrieval: dict[str, Any]
    vector_store: dict[str, Any]
    database: str
    cache: str
    auth_enabled: bool
    limits: dict[str, Any]
