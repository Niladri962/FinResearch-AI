"""Multi-period trend analysis: YoY changes, CAGR, direction and notable moves."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.financial.calculations import cagr, difference, pct_change

SIGNIFICANT_PCT = 15.0   # |YoY %| at or above this is flagged for amounts
SIGNIFICANT_PP = 2.0     # percentage-point move flagged for ratios expressed in %
STABLE_PCT = 2.0


@dataclass
class TrendPoint:
    label: str
    fiscal_year: int
    value: float | None
    change: float | None = None   # YoY % for amounts, percentage points for % ratios


@dataclass
class TrendResult:
    key: str
    name: str
    unit: str                      # "%" → changes are in percentage points
    points: list[TrendPoint] = field(default_factory=list)
    cagr: float | None = None
    cagr_years: int = 0
    direction: str = "insufficient_data"
    significant: list[str] = field(default_factory=list)

    @property
    def change_unit(self) -> str:
        return "pp" if self.unit == "%" else "%"

    @property
    def has_data(self) -> bool:
        return sum(1 for p in self.points if p.value is not None) >= 2


def analyze_series(
    key: str, name: str, series: list[tuple[str, int, float | None]], *, unit: str = ""
) -> TrendResult:
    """``series`` is ``[(period_label, fiscal_year, value)]`` in any order."""
    ordered = sorted(series, key=lambda item: item[1])
    is_pct = unit == "%"
    result = TrendResult(key=key, name=name, unit=unit)

    previous: tuple[int, float] | None = None
    for label, year, value in ordered:
        change = None
        # Only compare consecutive fiscal years; a gap is not a year-over-year change.
        if value is not None and previous is not None and year - previous[0] == 1:
            change = difference(value, previous[1]) if is_pct else pct_change(value, previous[1])
        result.points.append(TrendPoint(label=label, fiscal_year=year, value=value, change=change))
        if value is not None:
            previous = (year, value)

    present = [p for p in result.points if p.value is not None]
    if len(present) < 2:
        return result

    first, last = present[0], present[-1]
    result.cagr_years = last.fiscal_year - first.fiscal_year
    if not is_pct:
        result.cagr = cagr(first.value, last.value, result.cagr_years)

    changes = [p.change for p in result.points if p.change is not None]
    threshold = SIGNIFICANT_PP if is_pct else SIGNIFICANT_PCT
    stable = 0.5 if is_pct else STABLE_PCT
    if changes and all(abs(c) < stable for c in changes):
        result.direction = "stable"
    elif changes and all(c > 0 for c in changes):
        result.direction = "increasing"
    elif changes and all(c < 0 for c in changes):
        result.direction = "decreasing"
    elif last.value > first.value:  # type: ignore[operator]
        result.direction = "mixed_up"
    elif last.value < first.value:  # type: ignore[operator]
        result.direction = "mixed_down"
    else:
        result.direction = "mixed"

    suffix = "pp" if is_pct else "%"
    for point in result.points:
        if point.change is not None and abs(point.change) >= threshold:
            verb = "rose" if point.change > 0 else "fell"
            result.significant.append(f"{name} {verb} {abs(point.change):.2f}{suffix} in {point.label}")
    return result


DIRECTION_LABELS = {
    "increasing": "Consistently increasing",
    "decreasing": "Consistently decreasing",
    "stable": "Broadly stable",
    "mixed_up": "Volatile, higher overall",
    "mixed_down": "Volatile, lower overall",
    "mixed": "Volatile, flat overall",
    "insufficient_data": "Insufficient data",
}
