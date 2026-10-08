"""Research report generation.

A report is assembled section by section. Tables and ratios are inserted
programmatically from the financial engine; narrative paragraphs are written
from retrieved passages and pass through the same citation validation as chat
answers. Source ids are numbered once across the whole report so every claim
can be traced through the final "Sources" section.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select

from app.agents.prompts import SYSTEM_PROMPT, build_context
from app.agents.summarization_agent import _best_sentences
from app.config import Settings
from app.documents.table_extractor import table_to_markdown
from app.financial.ratios import CATEGORIES, RATIOS
from app.guardrails.citation_validator import build_citations, validate_answer
from app.guardrails.input_guard import is_suspicious_context
from app.guardrails.output_guard import DISCLAIMER, check_output
from app.llm.base import LLMClient
from app.models.database import SessionFactory, session_scope
from app.models.db import Company, Report
from app.models.schemas import Calculation, Citation, DataTable, ReportDetail, ReportOut, RetrievedChunk
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import SearchFilter
from app.services.financials import FinancialDataService, PeriodData, display_name
from app.utils.errors import LLMError, MissingFinancialDataError, NotFoundError
from app.utils.formatting import NOT_AVAILABLE, fmt_number, fmt_ratio
from app.utils.logging import get_logger

logger = get_logger(__name__)

_MARKER = re.compile(r"\[((?:[SCT]\d+)(?:\s*,\s*(?:[SCT]\d+))*)\]")


@dataclass
class SectionSpec:
    title: str
    query: str = ""                         # retrieval query; empty → no narrative
    instruction: str = ""
    metric_keys: tuple[str, ...] = ()       # figures table shown in the section
    ratio_categories: tuple[str, ...] = ()
    include_flags: bool = False


SECTIONS: list[SectionSpec] = [
    SectionSpec("Company Overview", "company overview business segments products operations customers",
                "Describe what the company does, its main businesses and where its revenue comes from."),
    SectionSpec("Revenue Analysis", "revenue growth drivers segment performance volume price",
                "Explain how revenue changed and what drove the change.", ("revenue",), ("growth",)),
    SectionSpec("Profitability Analysis", "operating margin profitability costs EBITDA net profit drivers",
                "Explain the level and direction of profitability and the reasons given for changes in margins and profit.",
                ("gross_profit", "ebitda", "ebit", "net_income", "eps"), ("profitability",)),
    SectionSpec("Balance Sheet Analysis", "balance sheet borrowings debt leverage equity liquidity working capital",
                "Assess leverage, liquidity and working capital using the figures provided and what the documents say about them.",
                ("total_assets", "equity", "total_debt", "cash", "current_assets", "current_liabilities"),
                ("liquidity", "solvency")),
    SectionSpec("Cash Flow Analysis", "cash flow from operating activities capital expenditure free cash flow",
                "Explain operating cash flow, capital expenditure and free cash flow, and the reasons for changes.",
                ("cfo", "capex", "free_cash_flow", "cfi", "cff")),
    SectionSpec("Management Commentary", "management commentary outlook guidance expectations strategy priorities",
                "Summarise what management said about performance, strategy and expectations. Attribute statements to management and keep forward-looking statements framed as expectations."),
    SectionSpec("Risk Analysis", "risk factors key risks uncertainties threats concerns",
                "List the main risks disclosed, each with a one-line explanation, and relate them to the quantitative signals provided.",
                include_flags=True),
    SectionSpec("Growth Opportunities", "growth opportunities expansion new capacity new products markets investment plans",
                "Describe the growth opportunities and investment plans the documents mention. Do not speculate beyond them."),
]


@dataclass
class _SourceRegistry:
    """Assigns one global S-number per chunk across the whole report."""

    order: list[RetrievedChunk] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)

    def register(self, chunk: RetrievedChunk) -> int:
        if chunk.chunk_id not in self.index:
            self.order.append(chunk)
            self.index[chunk.chunk_id] = len(self.order)
        return self.index[chunk.chunk_id]

    def remap(self, text: str, local: list[RetrievedChunk]) -> str:
        """Rewrite section-local ids to global ones; computed-figure markers become "[computed]"."""
        def replace(match: re.Match[str]) -> str:
            out: list[str] = []
            for cid in re.findall(r"[SCT]\d+", match.group(1)):
                if cid[0] == "S":
                    position = int(cid[1:])
                    if 1 <= position <= len(local):
                        label = f"S{self.register(local[position - 1])}"
                    else:
                        continue
                else:
                    label = "computed"
                if label not in out:
                    out.append(label)
            return f"[{', '.join(out)}]" if out else ""
        return _MARKER.sub(replace, text)


def _figures_rows(periods: list[PeriodData], keys: tuple[str, ...]) -> list[list[str]]:
    """Header row plus one row per metric that has data; empty when nothing is available."""
    rows = [["Metric", *[p.label for p in periods]]]
    for key in keys:
        if any(key in p.values for p in periods):
            rows.append([display_name(key), *[fmt_number(p.values[key].reported) if key in p.values else NOT_AVAILABLE for p in periods]])
    return rows if len(rows) > 1 else []


def _ratio_rows(fin: FinancialDataService, periods: list[PeriodData], categories: tuple[str, ...]) -> list[list[str]]:
    rows = [["Ratio", "Formula", *[p.label for p in periods]]]
    for ratio in RATIOS:
        if ratio.category not in categories:
            continue
        _, unit, points = fin.series(periods, ratio.key)
        if any(v is not None for _, _, v in points):
            rows.append([ratio.name, ratio.formula, *[fmt_ratio(v, unit) for _, _, v in points]])
    return rows if len(rows) > 1 else []


class ReportService:
    def __init__(
        self, settings: Settings, session_factory: SessionFactory, retriever: HybridRetriever,
        financials: FinancialDataService, llm: LLMClient,
    ) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._retriever = retriever
        self._fin = financials
        self._llm = llm

    # ── Narrative ────────────────────────────────────────────────────────
    async def _narrative(
        self, company: str, title: str, instruction: str, evidence: list[RetrievedChunk],
        calculations: list[Calculation], tables: list[DataTable], registry: _SourceRegistry,
    ) -> str:
        if not evidence and not calculations and not tables:
            return "_No relevant passages were found in the uploaded documents for this section._"
        if self._llm.available:
            user = (
                build_context(evidence, calculations, tables)
                + f"\n\nTASK: Write the \"{title}\" section of a financial research report on {company}. {instruction} "
                "Write 1–2 short paragraphs or up to 5 bullet points. Do not add a heading. Do not repeat tables."
            )
            try:
                response = await self._llm.complete(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], max_tokens=600
                )
                text, _report = validate_answer(
                    response.text.strip(), evidence=evidence, calculations=calculations, tables=tables, charts=[],
                )
                return registry.remap(check_output(text).text, evidence)
            except LLMError as exc:
                logger.warning("Report section fell back to extractive text", extra={"section": title, "code": exc.code})
        if not evidence:
            return ""
        return "\n".join(
            f"- {_best_sentences(instruction, chunk)} [S{registry.register(chunk)}]" for chunk in evidence[:4]
        )

    async def _build_section(
        self, spec: SectionSpec, company: Company, periods: list[PeriodData], flags: list[Calculation],
        registry: _SourceRegistry, semaphore: asyncio.Semaphore,
    ) -> str:
        parts: list[str] = [f"## {spec.title}"]
        tables: list[DataTable] = []
        if spec.metric_keys and periods:
            unit = periods[-1].unit_label
            rows = _figures_rows(periods, spec.metric_keys)
            if rows:
                parts.append(f"**Reported figures{f' ({unit})' if unit else ''}**\n\n{table_to_markdown(rows)}")
                tables.append(DataTable(id="T1", title=f"{spec.title} — reported figures ({unit})", columns=rows[0], rows=rows[1:]))
        if spec.ratio_categories and periods:
            rows = _ratio_rows(self._fin, periods, spec.ratio_categories)
            if rows:
                parts.append(f"**Ratios (calculated)**\n\n{table_to_markdown(rows)}")
                tables.append(DataTable(id=f"T{len(tables) + 1}", title=f"{spec.title} — ratios", columns=rows[0], rows=rows[1:]))
        section_flags = flags if spec.include_flags else []
        if section_flags:
            parts.append("**Quantitative risk signals (calculated)**\n\n" + "\n".join(
                f"- **{f.name}** ({f.severity}): {f.display}" for f in section_flags
            ))

        evidence: list[RetrievedChunk] = []
        if spec.query:
            async with semaphore:
                result = await asyncio.to_thread(
                    self._retriever.retrieve, f"{company.name} {spec.query}", SearchFilter(company_ids=[company.id]), 5
                )
                evidence = [c for c in result.chunks if not is_suspicious_context(c.text)]
                calcs = [f.model_copy(update={"id": f"C{i}"}) for i, f in enumerate(section_flags, start=1)]
                narrative = await self._narrative(company.name, spec.title, spec.instruction, evidence, calcs, tables, registry)
            if narrative:
                parts.append(narrative)
        return "\n\n".join(parts)

    async def _framing(self, company: str, body: str, registry: _SourceRegistry) -> tuple[str, str]:
        """Executive summary and conclusion, written from the finished sections."""
        if not self._llm.available:
            note = "_A language model is not configured, so this section is omitted. The sections below contain the computed figures and the supporting passages._"
            return note, note
        valid = {f"S{i}" for i in range(1, len(registry.order) + 1)}

        async def write(task: str) -> str:
            try:
                response = await self._llm.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"REPORT BODY (already validated; its [S#] ids are the only ids you may cite):\n{body[:14000]}\n\n"
                            f"TASK: {task} Base it only on the report body. Keep every [S#] citation attached to the "
                            "claim it supports, and mark calculated figures with [computed]. Do not add a heading."
                        )},
                    ],
                    max_tokens=500,
                )
            except LLMError:
                return "_This section could not be generated because the language model was unavailable._"

            def keep(match: re.Match[str]) -> str:
                ids = [i for i in re.findall(r"[SCT]\d+", match.group(1)) if i in valid]
                return f"[{', '.join(ids)}]" if ids else ""
            return check_output(_MARKER.sub(keep, response.text.strip())).text

        summary, conclusion = await asyncio.gather(
            write(f"Write an executive summary of this research report on {company} in 4–6 bullet points."),
            write(f"Write a balanced conclusion for this research report on {company} in one short paragraph: "
                  "the main strengths, the main concerns and what to monitor. No investment recommendation."),
        )
        return summary, conclusion

    # ── Public API ───────────────────────────────────────────────────────
    async def generate(self, company_id: int, title: str | None = None) -> ReportDetail:
        company = self._fin.get_company(company_id)
        periods = await asyncio.to_thread(self._fin.periods, company_id)
        flags = self._fin.risk_flags(company.name, periods)
        registry = _SourceRegistry()
        semaphore = asyncio.Semaphore(3)

        sections = await asyncio.gather(
            *(self._build_section(spec, company, periods, flags, registry, semaphore) for spec in SECTIONS)
        )
        if not registry.order and not periods:
            raise MissingFinancialDataError(
                f"No indexed content was found for {company.name}. Upload documents and wait for processing to finish."
            )

        ratios_md = table_to_markdown(_ratio_rows(self._fin, periods, CATEGORIES)) if periods else ""
        trend_tables, _charts, cagr = self._fin.trend_artifacts(
            company.name, periods, ["revenue", "ebitda", "net_income", "eps", "net_margin", "debt_to_equity"]
        ) if periods else ([], [], [])
        trends_md = "\n\n".join(
            f"**{t.title.split(' — ')[-1].replace(' trend', '')}** — {t.note}\n\n{table_to_markdown([t.columns, *t.rows])}"
            for t in trend_tables
        )
        if cagr:
            trends_md += "\n\n**Compound annual growth (calculated)**\n\n" + "\n".join(
                f"- {c.name} ({c.period}): {c.display}" for c in cagr
            )

        body = "\n\n".join(sections)
        summary, conclusion = await self._framing(company.name, body, registry)
        citations = build_citations(registry.order)
        sources_md = "\n".join(
            f"- **[{c.id}]** {c.document_title} — page {c.page}" + (f', section "{c.section}"' if c.section else "")
            + f" ({c.filename})"
            for c in citations
        ) or "_No document passages were cited._"

        generated = datetime.now(timezone.utc)
        coverage = f"{periods[0].label}–{periods[-1].label}" if periods else "no structured financial data"
        report_title = title or f"{company.name} — Financial Research Report"
        content = "\n\n".join([
            f"# {report_title}",
            f"_Generated {generated:%d %B %Y} · Coverage: {coverage} · "
            f"Basis: {periods[-1].unit_label if periods and periods[-1].unit_label else 'as reported'}_",
            f"> {DISCLAIMER}",
            "> **Reading this report:** `[S#]` cites a passage listed under Sources. `[computed]` marks figures "
            "calculated by the financial engine from the extracted statements. Tables are generated from extracted "
            "data, not written by a language model.",
            "## Executive Summary", summary,
            body,
            "## Key Financial Ratios", ratios_md or "_Not available — no structured financial statements were extracted._",
            "## Historical Trends", trends_md or "_Not available — at least two fiscal years of data are required._",
            "## Conclusion", conclusion,
            "## Sources", sources_md,
        ])

        with session_scope(self._session_factory) as session:
            report = Report(
                company_id=company.id, title=report_title, content=content,
                sources=[c.model_dump() for c in citations],
                meta={
                    "periods": [p.label for p in periods], "source_count": len(citations),
                    "mode": "generative" if self._llm.available else "extractive", "llm_model": self._llm.model or None,
                },
            )
            session.add(report)
            session.flush()
            return ReportDetail(
                id=report.id, company_id=company.id, company_name=company.name, title=report.title,
                created_at=report.created_at, meta=report.meta, content=content, sources=citations,
            )

    def list(self) -> list[ReportOut]:
        with self._session_factory() as session:
            rows = session.execute(
                select(Report, Company.name).outerjoin(Company, Company.id == Report.company_id)
                .order_by(Report.created_at.desc())
            ).all()
            out = []
            for report, company_name in rows:
                item = ReportOut.model_validate(report)
                item.company_name = company_name
                out.append(item)
            return out

    def get(self, report_id: str) -> ReportDetail:
        with self._session_factory() as session:
            row = session.execute(
                select(Report, Company.name).outerjoin(Company, Company.id == Report.company_id)
                .where(Report.id == report_id)
            ).first()
            if row is None:
                raise NotFoundError("Report not found.")
            report, company_name = row
            return ReportDetail(
                id=report.id, company_id=report.company_id, company_name=company_name, title=report.title,
                created_at=report.created_at, meta=report.meta, content=report.content,
                sources=[Citation.model_validate(s) for s in report.sources],
            )

    def delete(self, report_id: str) -> None:
        with session_scope(self._session_factory) as session:
            report = session.get(Report, report_id)
            if report is None:
                raise NotFoundError("Report not found.")
            session.delete(report)
