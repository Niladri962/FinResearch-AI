"""Company-vs-company and period-vs-period comparison tables.

Missing values are shown as "Not available" — never estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.financial.calculations import difference, pct_change
from app.financial.metrics import is_per_share
from app.financial.ratios import RATIO_BY_KEY
from app.models.schemas import Calculation, ChartSeries, ChartSpec, DataTable
from app.services.financials import FinancialDataService, PeriodData, display_name
from app.utils.formatting import NOT_AVAILABLE, fmt_change, fmt_number, fmt_ratio

COMPARISON_KEYS: tuple[str, ...] = (
    "revenue", "revenue_growth", "ebitda", "ebitda_margin", "net_income", "net_margin",
    "total_debt", "debt_to_equity", "roe", "roa", "cfo", "free_cash_flow", "eps",
)
_CHART_KEYS: tuple[str, ...] = ("ebitda_margin", "net_margin", "roe", "roa", "revenue_growth")


@dataclass
class ComparisonResult:
    mode: str                                   # "company" | "period"
    table: DataTable
    charts: list[ChartSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    has_data: bool = False


def _cell(calc: Calculation) -> str:
    if calc.value is None:
        return NOT_AVAILABLE
    return calc.display if calc.key in RATIO_BY_KEY else fmt_number(calc.value)


def _row_label(key: str, unit_label: str = "") -> str:
    name = display_name(key)
    if key in RATIO_BY_KEY or not unit_label or is_per_share(key):
        return name
    return f"{name} ({unit_label})"


class ComparisonService:
    def __init__(self, financials: FinancialDataService) -> None:
        self._fin = financials

    def compare_companies(self, companies: list[tuple[int, str]], fiscal_year: int | None = None) -> ComparisonResult:
        """``companies`` is ``[(id, name)]``. Uses the latest year all have in common when possible."""
        data = {cid: self._fin.periods(cid) for cid, _ in companies}
        notes: list[str] = []

        common = None
        year_sets = [{p.fiscal_year for p in periods} for periods in data.values() if periods]
        if fiscal_year is not None:
            common = fiscal_year
        elif year_sets and len(year_sets) == len(companies):
            shared = set.intersection(*year_sets)
            common = max(shared) if shared else None

        selected: dict[int, tuple[PeriodData | None, PeriodData | None]] = {}
        for cid, name in companies:
            period, previous = self._fin.pick(data[cid], common) if common else self._fin.pick(data[cid])
            if period is None and common:
                period, previous = self._fin.pick(data[cid])  # fall back to that company's latest year
            selected[cid] = (period, previous)
            if period is None:
                notes.append(f"No structured financial data is available for {name}.")
        if len({p.label for p, _ in selected.values() if p}) > 1:
            notes.append("The companies do not share a common fiscal year in the uploaded documents; each column shows that company's latest available year.")
        bases = {p.unit_label for p, _ in selected.values() if p and p.unit_label}
        if len(bases) > 1:
            notes.append(
                "Absolute amounts are shown in each company's own reporting currency and unit and are not "
                "currency-converted. Compare ratios, margins and growth rates rather than absolute amounts."
            )

        columns = ["Metric"]
        for cid, name in companies:
            period = selected[cid][0]
            suffix = f"{period.label}, {period.unit_label}" if period and period.unit_label else (period.label if period else "no data")
            columns.append(f"{name} ({suffix})")

        calcs: dict[str, dict[int, Calculation | None]] = {}
        rows: list[list[str]] = []
        for key in COMPARISON_KEYS:
            row = [display_name(key)]
            calcs[key] = {}
            for cid, name in companies:
                period, previous = selected[cid]
                calc = self._fin.calculate(name, key, period, previous) if period else None
                calcs[key][cid] = calc
                row.append(_cell(calc) if calc else NOT_AVAILABLE)
            rows.append(row)

        chart_keys = [k for k in _CHART_KEYS if any(c and c.value is not None for c in calcs[k].values())]
        charts = []
        if chart_keys:
            charts.append(ChartSpec(
                title="Profitability and growth comparison", kind="bar",
                x=[display_name(k) for k in chart_keys],
                series=[
                    ChartSeries(name=name, data=[calcs[k][cid].value if calcs[k][cid] else None for k in chart_keys])
                    for cid, name in companies
                ],
                unit="%",
            ))
        names = [name for _, name in companies]
        return ComparisonResult(
            mode="company",
            table=DataTable(title="Company comparison: " + " vs ".join(names), columns=columns, rows=rows, note=" ".join(notes) or None),
            charts=charts, notes=notes, companies=names,
            has_data=any(p is not None for p, _ in selected.values()),
        )

    def compare_periods(self, company_id: int, company: str, fiscal_years: list[int] | None = None) -> ComparisonResult:
        """Compare the requested fiscal years, or the latest two when none are given."""
        periods = self._fin.periods(company_id)
        notes: list[str] = []
        if fiscal_years:
            chosen = [p for p in periods if p.fiscal_year in set(fiscal_years)]
            missing = sorted(set(fiscal_years) - {p.fiscal_year for p in chosen})
            if missing:
                notes.append("No structured data for " + ", ".join(f"FY{y}" for y in missing) + ".")
        else:
            chosen = periods[-2:]
        if not periods:
            notes.append(f"No structured financial data is available for {company}.")
        elif len(chosen) < 2:
            notes.append("At least two fiscal years with data are needed for a period comparison.")

        index = {p.fiscal_year: i for i, p in enumerate(periods)}
        unit_label = chosen[-1].unit_label if chosen else ""
        columns = ["Metric", *[p.label for p in chosen]]
        show_change = len(chosen) >= 2
        if show_change:
            columns.append(f"Change ({chosen[0].label} → {chosen[-1].label})")

        rows: list[list[str]] = []
        values: dict[str, list[float | None]] = {}
        for key in COMPARISON_KEYS:
            row = [_row_label(key, unit_label)]
            series: list[float | None] = []
            for period in chosen:
                i = index[period.fiscal_year]
                previous = periods[i - 1] if i > 0 and periods[i - 1].fiscal_year == period.fiscal_year - 1 else None
                calc = self._fin.calculate(company, key, period, previous)
                series.append(calc.value)
                row.append(_cell(calc))
            values[key] = series
            if show_change:
                first, last = series[0], series[-1]
                if key in RATIO_BY_KEY:
                    unit = RATIO_BY_KEY[key].unit
                    row.append(fmt_change(difference(last, first), " pp" if unit == "%" else "x"))
                else:
                    row.append(fmt_change(pct_change(last, first), "%"))
            rows.append(row)

        chart_keys = [k for k in _CHART_KEYS if any(v is not None for v in values.get(k, []))]
        charts = []
        if chart_keys and chosen:
            charts.append(ChartSpec(
                title=f"{company} — margins and returns by period", kind="bar",
                x=[display_name(k) for k in chart_keys],
                series=[ChartSeries(name=p.label, data=[values[k][i] for k in chart_keys]) for i, p in enumerate(chosen)],
                unit="%",
            ))
        return ComparisonResult(
            mode="period",
            table=DataTable(
                title=f"{company} — period comparison" + (f" ({' vs '.join(p.label for p in chosen)})" if chosen else ""),
                columns=columns, rows=rows, note=" ".join(notes) or None,
            ),
            charts=charts, notes=notes, companies=[company], has_data=len(chosen) >= 1,
        )
