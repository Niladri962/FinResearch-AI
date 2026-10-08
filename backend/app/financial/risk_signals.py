"""Quantitative risk signals derived from reported figures.

Each rule compares the latest period with the one before it and emits a signal
only when the underlying numbers are present. Thresholds are deliberately simple
and stated in the output so a reader can judge them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from app.financial.calculations import pct_change, percentage, safe_div

Values = Mapping[str, float | None]
AmountFormatter = Callable[[float], str]


@dataclass
class RiskSignal:
    key: str
    title: str
    severity: str            # low | medium | high
    detail: str
    metrics: list[str] = field(default_factory=list)   # metric keys the signal relies on
    value: float | None = None
    unit: str = ""


def _ratio(values: Values, num: str, den: str) -> float | None:
    return safe_div(values.get(num), values.get(den))


def detect_risk_signals(
    current_label: str,
    current: Values,
    previous_label: str | None,
    previous: Values | None,
    fmt: AmountFormatter = lambda v: f"{v:,.0f}",
) -> list[RiskSignal]:
    signals: list[RiskSignal] = []
    prev = previous or {}
    vs = f" versus {previous_label}" if previous_label else ""

    def chg(key: str) -> float | None:
        return pct_change(current.get(key), prev.get(key))

    # ── Leverage ────────────────────────────────────────────────────────
    de_now, de_prev = _ratio(current, "total_debt", "equity"), _ratio(prev, "total_debt", "equity")
    debt_chg = chg("total_debt")
    if debt_chg is not None and debt_chg >= 10 and de_now is not None and de_prev is not None and de_now > de_prev:
        signals.append(RiskSignal(
            "rising_leverage", "Rising leverage", "high" if de_now > 1.5 or debt_chg >= 40 else "medium",
            f"Total debt increased {debt_chg:.2f}% to {fmt(current['total_debt'])} in {current_label}{vs}; "  # type: ignore[arg-type]
            f"debt-to-equity moved from {de_prev:.2f}x to {de_now:.2f}x.",
            ["total_debt", "equity"], debt_chg, "%",
        ))
    elif de_now is not None and de_now > 1.0:
        signals.append(RiskSignal(
            "high_leverage", "High leverage", "high" if de_now > 2.0 else "medium",
            f"Debt-to-equity is {de_now:.2f}x in {current_label} (above 1.0x).",
            ["total_debt", "equity"], de_now, "x",
        ))

    cover_now, cover_prev = _ratio(current, "ebit", "finance_costs"), _ratio(prev, "ebit", "finance_costs")
    if cover_now is not None and cover_now < 3.0:
        signals.append(RiskSignal(
            "weak_interest_coverage", "Weak interest coverage", "high" if cover_now < 1.5 else "medium",
            f"EBIT covers finance costs {cover_now:.2f}x in {current_label} (below 3.0x).",
            ["ebit", "finance_costs"], cover_now, "x",
        ))
    elif cover_now is not None and cover_prev is not None and cover_now <= cover_prev * 0.75:
        signals.append(RiskSignal(
            "declining_interest_coverage", "Declining interest coverage", "medium",
            f"Interest coverage fell from {cover_prev:.2f}x to {cover_now:.2f}x in {current_label}.",
            ["ebit", "finance_costs"], cover_now, "x",
        ))

    # ── Cash flow ───────────────────────────────────────────────────────
    cfo, cfo_chg = current.get("cfo"), chg("cfo")
    if cfo is not None and cfo < 0:
        signals.append(RiskSignal(
            "negative_operating_cash_flow", "Negative operating cash flow", "high",
            f"Cash flow from operations was {fmt(cfo)} in {current_label}.", ["cfo"], cfo,
        ))
    elif cfo_chg is not None and cfo_chg <= -10:
        signals.append(RiskSignal(
            "declining_operating_cash_flow", "Declining operating cash flow", "medium",
            f"Cash flow from operations fell {abs(cfo_chg):.2f}% to {fmt(cfo)} in {current_label}{vs}.",  # type: ignore[arg-type]
            ["cfo"], cfo_chg, "%",
        ))
    fcf = current.get("free_cash_flow")
    if fcf is not None and fcf < 0:
        signals.append(RiskSignal(
            "negative_free_cash_flow", "Negative free cash flow", "medium",
            f"Free cash flow was {fmt(fcf)} in {current_label}: capital expenditure exceeded operating cash flow.",
            ["cfo", "capex"], fcf,
        ))

    # ── Profitability ───────────────────────────────────────────────────
    for key, num, name in (("operating_margin", "ebit", "Operating margin"), ("net_margin", "net_income", "Net profit margin")):
        now, before = percentage(current.get(num), current.get("revenue")), percentage(prev.get(num), prev.get("revenue"))
        if now is not None and before is not None and before - now >= 1.0:
            signals.append(RiskSignal(
                f"{key}_compression", f"{name} compression", "high" if before - now >= 5.0 else "medium",
                f"{name} fell from {before:.2f}% to {now:.2f}% in {current_label} "
                f"(down {before - now:.2f} percentage points).",
                [num, "revenue"], now - before, "pp",
            ))
    rev_chg, ni_chg = chg("revenue"), chg("net_income")
    if rev_chg is not None and rev_chg < 0:
        signals.append(RiskSignal(
            "revenue_decline", "Revenue decline", "high" if rev_chg <= -10 else "medium",
            f"Revenue fell {abs(rev_chg):.2f}% in {current_label}{vs}.", ["revenue"], rev_chg, "%",
        ))
    if ni_chg is not None and ni_chg <= -5 and rev_chg is not None and rev_chg > 0:
        signals.append(RiskSignal(
            "profit_decline_despite_growth", "Profit fell despite revenue growth", "medium",
            f"Net profit fell {abs(ni_chg):.2f}% in {current_label} while revenue grew {rev_chg:.2f}%.",
            ["net_income", "revenue"], ni_chg, "%",
        ))
    ni = current.get("net_income")
    if ni is not None and ni < 0:
        signals.append(RiskSignal(
            "net_loss", "Net loss", "high", f"The company reported a net loss of {fmt(ni)} in {current_label}.",
            ["net_income"], ni,
        ))

    # ── Liquidity & working capital ─────────────────────────────────────
    current_ratio = _ratio(current, "current_assets", "current_liabilities")
    if current_ratio is not None and current_ratio < 1.2:
        signals.append(RiskSignal(
            "tight_liquidity", "Tight liquidity", "high" if current_ratio < 1.0 else "medium",
            f"Current ratio is {current_ratio:.2f}x in {current_label} (below 1.2x).",
            ["current_assets", "current_liabilities"], current_ratio, "x",
        ))
    if rev_chg is not None:
        for key, name in (("receivables", "Trade receivables"), ("inventory", "Inventories")):
            item_chg = chg(key)
            if item_chg is not None and item_chg - rev_chg >= 10:
                signals.append(RiskSignal(
                    f"{key}_build_up", f"{name} growing faster than revenue", "low",
                    f"{name} grew {item_chg:.2f}% in {current_label} against revenue growth of {rev_chg:.2f}%, "
                    "tying up working capital.",
                    [key, "revenue"], item_chg, "%",
                ))

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(signals, key=lambda s: order[s.severity])
