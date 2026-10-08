"""Human-readable number formatting used in calculations, tables and reports."""
from __future__ import annotations

NOT_AVAILABLE = "Not available"


def fmt_number(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return NOT_AVAILABLE
    if abs(value) >= 100 and float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def fmt_amount(value_abs: float | None, scale: float = 1.0, unit: str = "", currency: str = "") -> str:
    """Format an absolute amount back into its reported unit, e.g. ``INR 1,250.0 crore``."""
    if value_abs is None:
        return NOT_AVAILABLE
    reported = value_abs / (scale or 1.0)
    body = f"{reported:,.2f}".rstrip("0").rstrip(".") if not float(reported).is_integer() else f"{reported:,.0f}"
    parts = [p for p in (currency, body, unit) if p]
    return " ".join(parts)


def fmt_ratio(value: float | None, unit: str) -> str:
    """``unit`` is ``%`` for percentages, ``x`` for multiples, anything else is appended."""
    if value is None:
        return NOT_AVAILABLE
    if unit == "%":
        return f"{value:,.2f}%"
    if unit == "x":
        return f"{value:,.2f}x"
    return f"{value:,.2f}{(' ' + unit) if unit else ''}"


def fmt_change(value: float | None, unit: str = "%") -> str:
    if value is None:
        return NOT_AVAILABLE
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.2f}{unit}"
