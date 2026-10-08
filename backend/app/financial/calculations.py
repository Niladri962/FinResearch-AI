"""Primitive financial arithmetic.

Every function is pure and returns ``None`` when an input is missing or the
calculation is undefined (e.g. division by zero). Callers surface ``None`` as
"Not available" — nothing is ever guessed or defaulted.
"""
from __future__ import annotations

import math
from typing import Iterable

from app.utils.errors import CalculationError

Number = float | int | None


def _check(*values: Number) -> bool:
    """True when all values are present; raises on NaN/inf which indicate corrupt input."""
    for value in values:
        if value is None:
            return False
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculationError(f"Non-numeric value in calculation: {value!r}")
        if math.isnan(value) or math.isinf(value):
            raise CalculationError("Calculation input is NaN or infinite.")
    return True


def safe_div(numerator: Number, denominator: Number) -> float | None:
    if not _check(numerator, denominator) or denominator == 0:
        return None
    return float(numerator) / float(denominator)  # type: ignore[arg-type]


def percentage(numerator: Number, denominator: Number) -> float | None:
    result = safe_div(numerator, denominator)
    return None if result is None else result * 100.0


def pct_change(current: Number, previous: Number) -> float | None:
    """Percentage change from ``previous`` to ``current`` (relative to |previous|)."""
    if not _check(current, previous) or previous == 0:
        return None
    return (float(current) - float(previous)) / abs(float(previous)) * 100.0  # type: ignore[arg-type]


def difference(current: Number, previous: Number) -> float | None:
    if not _check(current, previous):
        return None
    return float(current) - float(previous)  # type: ignore[arg-type]


def cagr(first: Number, last: Number, years: Number) -> float | None:
    """Compound annual growth rate in percent. Undefined for non-positive endpoints."""
    if not _check(first, last, years) or years <= 0 or first <= 0 or last <= 0:  # type: ignore[operator]
        return None
    return ((float(last) / float(first)) ** (1.0 / float(years)) - 1.0) * 100.0  # type: ignore[arg-type]


def average(values: Iterable[Number]) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    _check(*present)
    return sum(present) / len(present)
