"""Illustrative projections.

These are mechanical extrapolations of reported history, not forecasts. They
are always labelled as such in the API and never presented as predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.financial.calculations import cagr

PROJECTION_NOTE = (
    "Illustrative extrapolation of historical figures only. It ignores business, market and "
    "macro factors and is not a forecast or investment guidance."
)


@dataclass
class Projection:
    method: str
    labels: list[str]
    values: list[float]
    growth_rate: float | None = None


def project_cagr(history: list[tuple[int, float]], periods: int = 2) -> Projection | None:
    """Extend the series at its historical CAGR. Needs ≥2 positive points."""
    points = sorted((y, v) for y, v in history if v is not None)
    if len(points) < 2:
        return None
    (first_year, first), (last_year, last) = points[0], points[-1]
    rate = cagr(first, last, last_year - first_year)
    if rate is None:
        return None
    values, current = [], last
    for _ in range(periods):
        current = current * (1 + rate / 100.0)
        values.append(current)
    labels = [f"FY{last_year + i}E" for i in range(1, periods + 1)]
    return Projection(method="historical_cagr", labels=labels, values=values, growth_rate=rate)


def project_linear(history: list[tuple[int, float]], periods: int = 2) -> Projection | None:
    """Least-squares straight line through the series. Needs ≥3 points."""
    points = sorted((y, v) for y, v in history if v is not None)
    if len(points) < 3:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x
    last_year = points[-1][0]
    years = [last_year + i for i in range(1, periods + 1)]
    return Projection(
        method="linear_trend",
        labels=[f"FY{y}E" for y in years],
        values=[slope * y + intercept for y in years],
    )
