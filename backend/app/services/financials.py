"""Structured financial data service.

Bridges stored ``financial_facts`` and the pure calculation engine. Everything
numeric the product shows — ratios, trends, comparisons, risk flags — is built
here, and every value keeps a reference to the document page it came from.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import select

from app.financial.calculations import difference, pct_change
from app.financial.metrics import METRIC_BY_KEY, is_per_share, metric_name
from app.financial.ratios import DERIVED_COMPONENTS, RATIO_BY_KEY, compute_ratio, derive_metrics
from app.financial.risk_signals import detect_risk_signals
from app.financial.trends import DIRECTION_LABELS, TrendResult, analyze_series
from app.models.database import SessionFactory
from app.models.db import Company, Document, FinancialFact
from app.models.enums import DocumentType
from app.models.schemas import CalcInput, Calculation, ChartSeries, ChartSpec, DataTable, SourceRef
from app.utils.errors import NotFoundError
from app.utils.formatting import NOT_AVAILABLE, fmt_amount, fmt_change, fmt_number, fmt_ratio

KPI_KEYS: tuple[str, ...] = ("revenue", "ebitda", "net_income", "eps", "roe", "debt_to_equity", "free_cash_flow")


@dataclass
class MetricValue:
    key: str
    value: float                  # absolute (reported × scale); per-share values are unscaled
    scale: float = 1.0
    unit: str = ""
    currency: str = ""
    derived: bool = False
    formula: str | None = None
    sources: list[SourceRef] = field(default_factory=list)

    @property
    def name(self) -> str:
        return metric_name(self.key)

    @property
    def reported(self) -> float:
        return self.value / (self.scale or 1.0)

    @property
    def display(self) -> str:
        if is_per_share(self.key):
            return " ".join(p for p in (self.currency, fmt_number(self.value)) if p)
        return fmt_amount(self.value, self.scale, self.unit, self.currency)


@dataclass
class PeriodData:
    label: str
    fiscal_year: int
    values: dict[str, MetricValue] = field(default_factory=dict)
    scale: float = 1.0
    unit: str = ""
    currency: str = ""

    @property
    def numbers(self) -> dict[str, float]:
        return {key: mv.value for key, mv in self.values.items()}

    def amount(self, value: float) -> str:
        return fmt_amount(value, self.scale, self.unit, self.currency)

    @property
    def unit_label(self) -> str:
        return " ".join(p for p in (self.currency, self.unit) if p)


def display_name(key: str) -> str:
    if key in RATIO_BY_KEY:
        return RATIO_BY_KEY[key].name
    if key.startswith("prev_"):
        return f"Prior-period {metric_name(key[5:])}"
    return metric_name(key)


class FinancialDataService:
    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    # ── Loading ──────────────────────────────────────────────────────────
    def get_company(self, company_id: int) -> Company:
        with self._session_factory() as session:
            company = session.get(Company, company_id)
            if company is None:
                raise NotFoundError(f"Company {company_id} was not found.")
            return company

    def periods(self, company_id: int) -> list[PeriodData]:
        """Annual periods for a company, oldest first, with derived metrics filled in."""
        with self._session_factory() as session:
            rows = session.execute(
                select(FinancialFact, Document)
                .join(Document, Document.id == FinancialFact.document_id)
                .where(FinancialFact.company_id == company_id, FinancialFact.period_kind == "annual")
            ).all()

        # When several documents report the same line item, prefer the filing for that
        # year, then extraction confidence, then annual reports, then the newest upload.
        best: dict[tuple[int, str], tuple[tuple, FinancialFact, Document]] = {}
        for fact, document in rows:
            rank = (
                document.fiscal_year == fact.fiscal_year,
                fact.confidence,
                document.document_type == DocumentType.ANNUAL_REPORT.value,
                document.uploaded_at,
            )
            key = (fact.fiscal_year, fact.metric)
            if key not in best or rank > best[key][0]:
                best[key] = (rank, fact, document)

        by_year: dict[int, PeriodData] = {}
        for (year, metric), (_rank, fact, document) in best.items():
            period = by_year.setdefault(year, PeriodData(label=f"FY{year}", fiscal_year=year))
            period.values[metric] = MetricValue(
                key=metric, value=fact.value * (fact.scale or 1.0), scale=fact.scale or 1.0,
                unit=fact.unit, currency=fact.currency,
                sources=[SourceRef(
                    document_id=document.id, document_title=document.title or document.filename,
                    page=fact.page, chunk_id=fact.chunk_id,
                )],
            )

        for period in by_year.values():
            scaled = [mv for mv in period.values.values() if not is_per_share(mv.key)]
            if scaled:
                period.scale, period.unit = Counter((mv.scale, mv.unit) for mv in scaled).most_common(1)[0][0]
            currencies = Counter(mv.currency for mv in period.values.values() if mv.currency)
            period.currency = currencies.most_common(1)[0][0] if currencies else ""
            numbers, derivations = derive_metrics(period.numbers)
            for key, formula in derivations.items():
                sources: list[SourceRef] = []
                for component in DERIVED_COMPONENTS.get(key, ()):
                    if component in period.values:
                        sources.extend(period.values[component].sources)
                period.values[key] = MetricValue(
                    key=key, value=numbers[key], scale=period.scale, unit=period.unit,
                    currency=period.currency, derived=True, formula=formula, sources=sources,
                )
        return [by_year[y] for y in sorted(by_year)]

    @staticmethod
    def pick(periods: list[PeriodData], label_or_year: str | int | None = None) -> tuple[PeriodData | None, PeriodData | None]:
        """Return ``(period, previous)``; the latest period when nothing is specified."""
        if not periods:
            return None, None
        index = len(periods) - 1
        if label_or_year is not None:
            wanted = str(label_or_year).upper().replace(" ", "")
            wanted = wanted if wanted.startswith("FY") else f"FY{wanted}"
            index = next((i for i, p in enumerate(periods) if p.label == wanted), -1)
            if index < 0:
                return None, None
        previous = periods[index - 1] if index > 0 and periods[index - 1].fiscal_year == periods[index].fiscal_year - 1 else None
        return periods[index], previous

    # ── Calculations ─────────────────────────────────────────────────────
    @staticmethod
    def _input(key: str, period: PeriodData | None) -> CalcInput:
        base = key[5:] if key.startswith("prev_") else key
        mv = period.values.get(base) if period else None
        return CalcInput(
            key=key, name=display_name(key),
            value=mv.reported if mv else None,
            display=mv.display if mv else NOT_AVAILABLE,
            derived=bool(mv and mv.derived), formula=mv.formula if mv else None,
            sources=mv.sources if mv else [],
        )

    def ratio(self, company: str, key: str, period: PeriodData, previous: PeriodData | None) -> Calculation:
        definition = RATIO_BY_KEY[key]
        result = compute_ratio(key, period.numbers, previous.numbers if previous else None)
        inputs = [self._input(name, previous if name.startswith("prev_") else period) for name in definition.inputs]
        change = None
        if previous is not None and not definition.needs_previous and result.value is not None:
            earlier = compute_ratio(key, previous.numbers).value
            change = difference(result.value, earlier)
        note = None
        if result.missing:
            note = "Cannot be calculated: " + ", ".join(display_name(m) for m in result.missing) + " not found in the documents."
        elif any(i.derived for i in inputs):
            note = "Uses derived inputs: " + "; ".join(f"{i.name} = {i.formula}" for i in inputs if i.derived)
        return Calculation(
            kind="growth" if definition.needs_previous else "ratio",
            key=key, name=definition.name, category=definition.category, company=company, period=period.label,
            formula=definition.formula, value=result.value, unit=definition.unit,
            display=fmt_ratio(result.value, definition.unit), inputs=inputs,
            missing=[display_name(m) for m in result.missing], note=note,
            change=change, change_unit="pp" if definition.unit == "%" else "x",
        )

    def metric(self, company: str, key: str, period: PeriodData, previous: PeriodData | None) -> Calculation:
        mv = period.values.get(key)
        earlier = previous.values.get(key) if previous else None
        return Calculation(
            kind="metric", key=key, name=display_name(key), category=METRIC_BY_KEY[key].statement if key in METRIC_BY_KEY else "",
            company=company, period=period.label,
            formula=(mv.formula or "Derived") if mv and mv.derived else "As reported",
            value=mv.reported if mv else None,
            unit="" if is_per_share(key) else period.unit_label,
            display=mv.display if mv else NOT_AVAILABLE,
            inputs=[self._input(key, period)] if mv else [],
            missing=[] if mv else [display_name(key)],
            note=None if mv else f"{display_name(key)} was not found in the documents for {period.label}.",
            change=pct_change(mv.value, earlier.value) if mv and earlier else None,
        )

    def calculate(self, company: str, key: str, period: PeriodData, previous: PeriodData | None) -> Calculation:
        return self.ratio(company, key, period, previous) if key in RATIO_BY_KEY else self.metric(company, key, period, previous)

    def kpis(self, company: str, periods: list[PeriodData]) -> list[Calculation]:
        period, previous = self.pick(periods)
        return [self.calculate(company, key, period, previous) for key in KPI_KEYS] if period else []

    # ── Trends ───────────────────────────────────────────────────────────
    def series(self, periods: list[PeriodData], key: str) -> tuple[str, str, list[tuple[str, int, float | None]]]:
        """``(name, unit, [(label, fiscal_year, value)])`` for a metric or ratio key."""
        if key in RATIO_BY_KEY:
            definition = RATIO_BY_KEY[key]
            points = []
            for index, period in enumerate(periods):
                previous = periods[index - 1] if index and periods[index - 1].fiscal_year == period.fiscal_year - 1 else None
                value = compute_ratio(key, period.numbers, previous.numbers if previous else None).value
                points.append((period.label, period.fiscal_year, value))
            return definition.name, definition.unit, points
        latest = periods[-1] if periods else None
        scale = 1.0 if is_per_share(key) or latest is None else (latest.scale or 1.0)
        unit = "" if latest is None else (latest.currency if is_per_share(key) else latest.unit_label)
        return (
            display_name(key), unit,
            [(p.label, p.fiscal_year, p.values[key].value / scale if key in p.values else None) for p in periods],
        )

    def trend(self, periods: list[PeriodData], key: str) -> TrendResult:
        name, unit, points = self.series(periods, key)
        return analyze_series(key, name, points, unit=unit)

    def trend_artifacts(
        self, company: str, periods: list[PeriodData], keys: list[str]
    ) -> tuple[list[DataTable], list[ChartSpec], list[Calculation]]:
        """Tables, charts and CAGR calculations for a set of metric/ratio keys."""
        tables: list[DataTable] = []
        charts: list[ChartSpec] = []
        calcs: list[Calculation] = []
        for key in keys:
            trend = self.trend(periods, key)
            if not trend.has_data:
                continue
            is_ratio = key in RATIO_BY_KEY
            value_header = f"{trend.name} ({trend.unit})" if trend.unit else trend.name
            change_header = "YoY change (pp)" if trend.change_unit == "pp" else "YoY change (%)"
            rows = [
                [
                    p.label,
                    NOT_AVAILABLE if p.value is None else (fmt_ratio(p.value, trend.unit) if is_ratio else fmt_number(p.value)),
                    NOT_AVAILABLE if p.change is None else fmt_change(p.change, " pp" if trend.change_unit == "pp" else "%"),
                ]
                for p in trend.points
            ]
            note = DIRECTION_LABELS.get(trend.direction, "")
            if trend.significant:
                note += ". Notable moves: " + "; ".join(trend.significant)
            tables.append(DataTable(title=f"{company} — {trend.name} trend", columns=["Period", value_header, change_header], rows=rows, note=note))
            charts.append(ChartSpec(
                title=f"{company} — {trend.name}", kind="line" if is_ratio else "bar",
                x=[p.label for p in trend.points],
                series=[ChartSeries(name=trend.name, data=[p.value for p in trend.points])],
                unit=trend.unit,
            ))
            if trend.cagr is not None:
                present = [p for p in trend.points if p.value is not None]
                first, last = present[0], present[-1]
                calcs.append(Calculation(
                    kind="cagr", key=f"{key}_cagr", name=f"{trend.name} CAGR", category="growth",
                    company=company, period=f"{first.label}–{last.label}",
                    formula=f"(Ending / Beginning)^(1 / {trend.cagr_years}) − 1",
                    value=trend.cagr, unit="%", display=fmt_ratio(trend.cagr, "%"),
                    inputs=[
                        CalcInput(key=f"{key}_start", name=f"{trend.name} {first.label}", value=first.value, display=fmt_number(first.value)),
                        CalcInput(key=f"{key}_end", name=f"{trend.name} {last.label}", value=last.value, display=fmt_number(last.value)),
                    ],
                    note=f"Over {trend.cagr_years} year(s). Values in {trend.unit}." if trend.unit else f"Over {trend.cagr_years} year(s).",
                ))
        return tables, charts, calcs

    # ── Risk ─────────────────────────────────────────────────────────────
    def risk_flags(self, company: str, periods: list[PeriodData]) -> list[Calculation]:
        period, previous = self.pick(periods)
        if period is None:
            return []
        signals = detect_risk_signals(
            period.label, period.numbers,
            previous.label if previous else None, previous.numbers if previous else None,
            fmt=period.amount,
        )
        flags = []
        for signal in signals:
            flags.append(Calculation(
                kind="flag", key=signal.key, name=signal.title, category="risk", company=company,
                period=period.label, formula="Rule-based signal computed from reported figures",
                value=signal.value, unit=signal.unit, display=signal.detail, severity=signal.severity,  # type: ignore[arg-type]
                inputs=[self._input(m, period) for m in signal.metrics if m in period.values],
            ))
        return flags

    # ── Company summary ──────────────────────────────────────────────────
    def reporting_basis(self, periods: list[PeriodData]) -> tuple[str | None, str | None]:
        latest = periods[-1] if periods else None
        return (latest.currency or None, latest.unit or None) if latest else (None, None)
