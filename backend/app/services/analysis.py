"""Financial analysis use-cases behind /api/analyze, /api/compare and /api/financial-ratios."""
from __future__ import annotations

from app.financial.forecasting import PROJECTION_NOTE, project_cagr
from app.financial.metrics import METRIC_BY_KEY
from app.financial.ratios import RATIO_BY_KEY, RATIOS, compute_ratios, derive_metrics
from app.guardrails.output_guard import DISCLAIMER
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CalcInput,
    Calculation,
    ChartSeries,
    ChartSpec,
    CompareRequest,
    CompareResponse,
    DataTable,
    PeriodSnapshot,
    QueryFilters,
    RatioRequest,
    RatioResponse,
)
from app.services.chat import ChatService
from app.services.comparison import ComparisonService
from app.services.documents import CompanyService
from app.services.financials import FinancialDataService, PeriodData, display_name
from app.utils.errors import MissingFinancialDataError, ValidationAppError
from app.utils.formatting import NOT_AVAILABLE, fmt_number, fmt_ratio

# Dashboard chart definitions: (title, kind, keys, unit override)
_TREND_CHARTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("Revenue trend", "bar", ("revenue",)),
    ("Profit trend", "bar", ("ebitda", "net_income")),
    ("Margin trend", "line", ("gross_margin", "ebitda_margin", "operating_margin", "net_margin")),
    ("Debt trend", "bar", ("total_debt", "equity")),
    ("Leverage (debt-to-equity)", "line", ("debt_to_equity",)),
    ("Cash flow trend", "bar", ("cfo", "free_cash_flow")),
]
_SNAPSHOT_METRICS = (
    "revenue", "gross_profit", "ebitda", "ebit", "net_income", "eps", "total_assets", "equity",
    "total_debt", "cash", "cfo", "capex", "free_cash_flow",
)
_STATEMENT_ROWS = [*_SNAPSHOT_METRICS]


class AnalysisService:
    def __init__(
        self,
        financials: FinancialDataService,
        comparison: ComparisonService,
        companies: CompanyService,
        chat: ChatService,
    ) -> None:
        self._fin = financials
        self._comparison = comparison
        self._companies = companies
        self._chat = chat

    # ── Ratios ───────────────────────────────────────────────────────────
    def ratios(self, request: RatioRequest) -> RatioResponse:
        unknown = [k for k in request.ratios if k not in RATIO_BY_KEY]
        if unknown:
            raise ValidationAppError(f"Unknown ratio(s): {', '.join(unknown)}.", details={"available": sorted(RATIO_BY_KEY)})
        keys = request.ratios or [r.key for r in RATIOS]

        if request.company_id is None:
            if not request.values:
                raise ValidationAppError("Provide either company_id or a non-empty 'values' object.")
            bad = [k for k in {*request.values, *request.previous_values} if k not in METRIC_BY_KEY]
            if bad:
                raise ValidationAppError(f"Unknown metric(s): {', '.join(sorted(bad))}.", details={"available": sorted(METRIC_BY_KEY)})
            values, derivations = derive_metrics(request.values)
            previous, _ = derive_metrics(request.previous_values) if request.previous_values else ({}, {})
            calcs = []
            for result in compute_ratios(values, previous or None, keys):
                d = result.definition
                calcs.append(Calculation(
                    kind="growth" if d.needs_previous else "ratio", key=d.key, name=d.name, category=d.category,
                    formula=d.formula, value=result.value, unit=d.unit, display=fmt_ratio(result.value, d.unit),
                    inputs=[
                        CalcInput(
                            key=name, name=display_name(name), value=value,
                            display=fmt_number(value) if value is not None else NOT_AVAILABLE,
                            derived=name in derivations, formula=derivations.get(name),
                        )
                        for name, value in result.inputs.items()
                    ],
                    missing=[display_name(m) for m in result.missing],
                    note=("Cannot be calculated: missing " + ", ".join(display_name(m) for m in result.missing)) if result.missing else None,
                ))
            return RatioResponse(calculations=calcs)

        company = self._fin.get_company(request.company_id)
        periods = self._fin.periods(company.id)
        if not periods:
            raise MissingFinancialDataError(
                f"No structured financial statements have been extracted for {company.name}. "
                "Upload an annual report or financial statements with statement tables."
            )
        period, previous = self._fin.pick(periods, request.period)
        if period is None:
            raise MissingFinancialDataError(
                f"No data for period '{request.period}'. Available: {', '.join(p.label for p in periods)}."
            )
        return RatioResponse(
            company=company.name, period=period.label, available_periods=[p.label for p in periods],
            calculations=[self._fin.ratio(company.name, key, period, previous) for key in keys],
        )

    # ── Company analysis ─────────────────────────────────────────────────
    def _snapshots(self, periods: list[PeriodData]) -> list[PeriodSnapshot]:
        snapshots = []
        for index, period in enumerate(periods):
            previous = periods[index - 1] if index and periods[index - 1].fiscal_year == period.fiscal_year - 1 else None
            ratio_values = {
                r.definition.key: r.value
                for r in compute_ratios(period.numbers, previous.numbers if previous else None)
            }
            snapshots.append(PeriodSnapshot(
                period=period.label, fiscal_year=period.fiscal_year,
                metrics={k: (period.values[k].reported if k in period.values else None) for k in _SNAPSHOT_METRICS},
                ratios=ratio_values,
            ))
        return snapshots

    def _trend_charts(self, company: str, periods: list[PeriodData]) -> list[ChartSpec]:
        charts = []
        for title, kind, keys in _TREND_CHARTS:
            series, unit = [], ""
            for key in keys:
                name, key_unit, points = self._fin.series(periods, key)
                if any(v is not None for _, _, v in points):
                    series.append(ChartSeries(name=name, data=[v for _, _, v in points]))
                    unit = unit or key_unit
            if series:
                charts.append(ChartSpec(
                    id=title.lower().replace(" ", "-").replace("(", "").replace(")", ""),
                    title=title, kind=kind, x=[p.label for p in periods], series=series, unit=unit,  # type: ignore[arg-type]
                ))
        return charts

    def _statement_table(self, company: str, periods: list[PeriodData]) -> DataTable:
        unit = periods[-1].unit_label if periods else ""
        rows = []
        for key in _STATEMENT_ROWS:
            if not any(key in p.values for p in periods):
                continue
            rows.append([
                display_name(key),
                *[fmt_number(p.values[key].reported) if key in p.values else NOT_AVAILABLE for p in periods],
            ])
        return DataTable(
            id="key-figures", title=f"{company} — key figures" + (f" ({unit})" if unit else ""),
            columns=["Metric", *[p.label for p in periods]], rows=rows,
            note="Figures as extracted from the uploaded statements; derived items are computed from reported lines.",
        )

    def _ratio_table(self, company: str, periods: list[PeriodData]) -> DataTable:
        rows = []
        for ratio in RATIOS:
            _, unit, points = self._fin.series(periods, ratio.key)
            if all(v is None for _, _, v in points):
                continue
            rows.append([ratio.name, *[fmt_ratio(v, unit) for _, _, v in points]])
        return DataTable(id="ratios", title=f"{company} — financial ratios", columns=["Ratio", *[p.label for p in periods]], rows=rows)

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        company_out = self._companies.get(request.company_id)
        periods = self._fin.periods(request.company_id)
        response = AnalyzeResponse(company=company_out, analysis_type=request.analysis_type, disclaimer=DISCLAIMER)
        if not periods:
            response.notes.append(
                "No structured financial statements were extracted from this company's documents, so metrics "
                "and charts are not available. Qualitative questions can still be asked in Research Chat."
            )
            return response

        name = company_out.name
        response.periods = self._snapshots(periods)
        response.kpis = self._fin.kpis(name, periods)
        response.risk_flags = self._fin.risk_flags(name, periods)
        if len(periods) < 2:
            response.notes.append("Only one fiscal year of data is available; trends need at least two.")

        if request.analysis_type == "trend":
            keys = [k for k in request.metrics if k in RATIO_BY_KEY or k in METRIC_BY_KEY]
            unknown = [k for k in request.metrics if k not in keys]
            if unknown:
                raise ValidationAppError(f"Unknown metric(s): {', '.join(unknown)}.")
            keys = keys or ["revenue", "net_income", "operating_margin", "net_margin"]
            tables, charts, calcs = self._fin.trend_artifacts(name, periods, keys)
            response.tables, response.charts, response.calculations = tables, charts, calcs
        else:
            response.charts = self._trend_charts(name, periods)
            response.tables = [self._statement_table(name, periods), self._ratio_table(name, periods)]
            period, previous = self._fin.pick(periods)
            response.calculations = [self._fin.ratio(name, r.key, period, previous) for r in RATIOS] if period else []

        if request.include_projection:
            history = [(p.fiscal_year, p.values["revenue"].reported) for p in periods if "revenue" in p.values]
            projection = project_cagr(history)
            if projection is not None:
                response.charts.append(ChartSpec(
                    id="revenue-projection", title="Revenue — illustrative extrapolation", kind="line",
                    x=[*[f"FY{y}" for y, _ in history], *projection.labels],
                    series=[
                        ChartSeries(name="Reported", data=[*[v for _, v in history], *[None] * len(projection.values)]),
                        ChartSeries(name="Extrapolated at historical CAGR",
                                    data=[*[None] * (len(history) - 1), history[-1][1], *projection.values]),
                    ],
                    unit=periods[-1].unit_label,
                ))
                response.notes.append(PROJECTION_NOTE)

        if request.include_narrative:
            question = {
                "risk": f"What are the key risks for {name}?",
                "trend": f"How has {name}'s financial performance changed over time?",
            }.get(request.analysis_type, f"Summarize {name}'s financial performance.")
            state = await self._chat.ask(question, QueryFilters(company_ids=[request.company_id]))
            response.narrative = state.get("answer")
            response.citations = state.get("sources", [])
        return response

    # ── Comparison ───────────────────────────────────────────────────────
    async def compare(self, request: CompareRequest) -> CompareResponse:
        ids = list(dict.fromkeys(request.company_ids))
        companies = [(cid, self._fin.get_company(cid).name) for cid in ids]
        years = []
        for label in request.periods:
            digits = "".join(ch for ch in label if ch.isdigit())
            if len(digits) != 4:
                raise ValidationAppError(f"Invalid period '{label}'. Use the form FY2025.")
            years.append(int(digits))

        if len(companies) >= 2:
            result = self._comparison.compare_companies(companies, years[-1] if years else None)
            question = "Compare " + " and ".join(name for _, name in companies) + "."
        else:
            company_id, name = companies[0]
            result = self._comparison.compare_periods(company_id, name, years if len(years) >= 2 else None)
            question = f"Compare {name}'s financial performance between its latest reporting periods."
            if len(years) >= 2:
                question = f"Compare {' and '.join(f'FY{y}' for y in years)} for {name}."

        response = CompareResponse(
            mode=result.mode, table=result.table, charts=result.charts, notes=result.notes, disclaimer=DISCLAIMER,  # type: ignore[arg-type]
        )
        if request.include_narrative and result.has_data:
            state = await self._chat.ask(question, QueryFilters(company_ids=ids))
            response.narrative = state.get("answer")
            response.citations = state.get("sources", [])
        return response
