"""Shared state and dependencies for the agent graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.base import LLMClient
from app.models.database import SessionFactory
from app.models.enums import Intent
from app.models.schemas import (
    Calculation,
    ChartSpec,
    Citation,
    DataTable,
    QueryFilters,
    RetrievedChunk,
    ValidationReport,
)
from app.rag.retriever import HybridRetriever
from app.services.comparison import ComparisonService
from app.services.financials import FinancialDataService
from app.utils.observability import Trace


class CompanyRef(BaseModel):
    id: int
    name: str


class QueryUnderstanding(BaseModel):
    intent: Intent = Intent.DOCUMENT_QA
    confidence: float = 0.0
    method: str = "rules"                      # rules | llm
    query: str = ""                            # standalone question used for retrieval
    companies: list[CompanyRef] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    ratios: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    is_followup: bool = False
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def needs_numbers(self) -> bool:
        return bool(self.ratios or self.metrics)


class AgentState(TypedDict, total=False):
    # Input
    query: str
    history: list[dict[str, str]]
    filters: QueryFilters
    context: dict[str, Any]                    # carried conversation context (company ids)
    # Control
    blocked: bool
    guard_notice: str
    guard_categories: list[str]
    understanding: QueryUnderstanding
    plan: list[str]
    cursor: int
    next: str
    # Working memory
    evidence: list[RetrievedChunk]
    calculations: list[Calculation]
    tables: list[DataTable]
    charts: list[ChartSpec]
    notes: list[str]
    # Output
    answer: str
    mode: str
    citations: list[Citation]
    sources: list[Citation]
    validation: ValidationReport
    followups: list[str]


@dataclass
class AgentDeps:
    settings: Settings
    session_factory: SessionFactory
    retriever: HybridRetriever
    financials: FinancialDataService
    comparison: ComparisonService
    llm: LLMClient


@dataclass
class RunContext:
    """Per-request handles passed to every agent."""

    trace: Trace
    emit: Callable[[str, dict[str, Any]], None]

    def status(self, stage: str, message: str) -> None:
        self.emit("status", {"stage": stage, "message": message})


def assign_ids(state: AgentState, calculations: list[Calculation], tables: list[DataTable], charts: list[ChartSpec]) -> dict[str, Any]:
    """Append new artifacts to the state, giving each a stable citation id (C1, T1, …)."""
    existing_calcs = list(state.get("calculations", []))
    existing_tables = list(state.get("tables", []))
    existing_charts = list(state.get("charts", []))
    for calc in calculations:
        calc.id = f"C{len(existing_calcs) + 1}"
        existing_calcs.append(calc)
    for table in tables:
        table.id = f"T{len(existing_tables) + 1}"
        existing_tables.append(table)
    for chart in charts:
        chart.id = f"chart-{len(existing_charts) + 1}"
        existing_charts.append(chart)
    return {"calculations": existing_calcs, "tables": existing_tables, "charts": existing_charts}
