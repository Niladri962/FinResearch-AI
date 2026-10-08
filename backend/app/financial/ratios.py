"""Financial ratio engine.

Ratios are declared once in ``RATIOS`` (formula text, required inputs, compute
function, query aliases). The same registry powers calculations, the glossary
used for general-finance questions, and query routing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

from app.financial.calculations import cagr as _cagr
from app.financial.calculations import pct_change, percentage, safe_div
from app.financial.metrics import detect_metrics

Values = Mapping[str, float | None]


@dataclass(frozen=True)
class RatioDefinition:
    key: str
    name: str
    category: str                 # liquidity | solvency | profitability | efficiency | growth
    formula: str
    inputs: tuple[str, ...]
    unit: str                     # "x" (multiple) or "%"
    compute: Callable[[Values], float | None]
    aliases: tuple[str, ...]
    description: str
    needs_previous: bool = False


@dataclass
class RatioResult:
    definition: RatioDefinition
    value: float | None
    inputs: dict[str, float | None] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _g(values: Values, key: str) -> float | None:
    return values.get(key)


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


RATIOS: tuple[RatioDefinition, ...] = (
    # ── Liquidity ───────────────────────────────────────────────────────
    RatioDefinition(
        "current_ratio", "Current Ratio", "liquidity",
        "Current Assets / Current Liabilities", ("current_assets", "current_liabilities"), "x",
        lambda v: safe_div(_g(v, "current_assets"), _g(v, "current_liabilities")),
        ("current ratio",),
        "Ability to cover short-term obligations with short-term assets.",
    ),
    RatioDefinition(
        "quick_ratio", "Quick Ratio", "liquidity",
        "(Current Assets − Inventories) / Current Liabilities",
        ("current_assets", "inventory", "current_liabilities"), "x",
        lambda v: safe_div(_sub(_g(v, "current_assets"), _g(v, "inventory")), _g(v, "current_liabilities")),
        ("quick ratio", "acid test", "acid-test ratio"),
        "Liquidity excluding inventory, which may be slow to convert to cash.",
    ),
    RatioDefinition(
        "cash_ratio", "Cash Ratio", "liquidity",
        "Cash & Cash Equivalents / Current Liabilities", ("cash", "current_liabilities"), "x",
        lambda v: safe_div(_g(v, "cash"), _g(v, "current_liabilities")),
        ("cash ratio",),
        "The most conservative liquidity measure: cash alone against current liabilities.",
    ),
    # ── Solvency ────────────────────────────────────────────────────────
    RatioDefinition(
        "debt_to_equity", "Debt-to-Equity", "solvency",
        "Total Debt / Shareholders' Equity", ("total_debt", "equity"), "x",
        lambda v: safe_div(_g(v, "total_debt"), _g(v, "equity")),
        ("debt-to-equity", "debt to equity", "debt/equity", "d/e ratio", "d/e", "gearing"),
        "Financial leverage: how much debt funds the business relative to shareholders' capital.",
    ),
    RatioDefinition(
        "debt_ratio", "Debt Ratio", "solvency",
        "Total Debt / Total Assets", ("total_debt", "total_assets"), "x",
        lambda v: safe_div(_g(v, "total_debt"), _g(v, "total_assets")),
        ("debt ratio", "debt to assets", "debt-to-assets"),
        "Share of the asset base that is financed with debt.",
    ),
    RatioDefinition(
        "interest_coverage", "Interest Coverage", "solvency",
        "EBIT / Finance Costs", ("ebit", "finance_costs"), "x",
        lambda v: safe_div(_g(v, "ebit"), _g(v, "finance_costs")),
        ("interest coverage", "interest cover", "times interest earned"),
        "How many times operating profit covers interest obligations.",
    ),
    # ── Profitability ───────────────────────────────────────────────────
    RatioDefinition(
        "gross_margin", "Gross Margin", "profitability",
        "Gross Profit / Revenue × 100", ("gross_profit", "revenue"), "%",
        lambda v: percentage(_g(v, "gross_profit"), _g(v, "revenue")),
        ("gross margin", "gross profit margin"),
        "Revenue retained after direct production costs.",
    ),
    RatioDefinition(
        "operating_margin", "Operating Margin", "profitability",
        "EBIT / Revenue × 100", ("ebit", "revenue"), "%",
        lambda v: percentage(_g(v, "ebit"), _g(v, "revenue")),
        ("operating margin", "operating margins", "ebit margin", "operating profit margin"),
        "Revenue retained after operating costs, before interest and tax.",
    ),
    RatioDefinition(
        "net_margin", "Net Profit Margin", "profitability",
        "Net Profit / Revenue × 100", ("net_income", "revenue"), "%",
        lambda v: percentage(_g(v, "net_income"), _g(v, "revenue")),
        ("net profit margin", "net margin", "profit margin", "pat margin", "net income margin"),
        "Revenue that ends up as profit for shareholders.",
    ),
    RatioDefinition(
        "ebitda_margin", "EBITDA Margin", "profitability",
        "EBITDA / Revenue × 100", ("ebitda", "revenue"), "%",
        lambda v: percentage(_g(v, "ebitda"), _g(v, "revenue")),
        ("ebitda margin",),
        "Operating profitability before depreciation and amortisation.",
    ),
    RatioDefinition(
        "roa", "Return on Assets (ROA)", "profitability",
        "Net Profit / Total Assets × 100 (period-end assets)", ("net_income", "total_assets"), "%",
        lambda v: percentage(_g(v, "net_income"), _g(v, "total_assets")),
        ("return on assets", "roa"),
        "Profit generated per unit of assets.",
    ),
    RatioDefinition(
        "roe", "Return on Equity (ROE)", "profitability",
        "Net Profit / Shareholders' Equity × 100 (period-end equity)", ("net_income", "equity"), "%",
        lambda v: percentage(_g(v, "net_income"), _g(v, "equity")),
        ("return on equity", "roe"),
        "Profit generated per unit of shareholders' capital.",
    ),
    # ── Efficiency ──────────────────────────────────────────────────────
    RatioDefinition(
        "asset_turnover", "Asset Turnover", "efficiency",
        "Revenue / Total Assets (period-end assets)", ("revenue", "total_assets"), "x",
        lambda v: safe_div(_g(v, "revenue"), _g(v, "total_assets")),
        ("asset turnover", "assets turnover"),
        "Revenue generated per unit of assets.",
    ),
    RatioDefinition(
        "inventory_turnover", "Inventory Turnover", "efficiency",
        "Cost of Goods Sold / Inventories (period-end inventory)", ("cogs", "inventory"), "x",
        lambda v: safe_div(_g(v, "cogs"), _g(v, "inventory")),
        ("inventory turnover", "stock turnover"),
        "How many times inventory is sold and replaced in a period.",
    ),
    RatioDefinition(
        "receivables_turnover", "Receivables Turnover", "efficiency",
        "Revenue / Trade Receivables (period-end receivables)", ("revenue", "receivables"), "x",
        lambda v: safe_div(_g(v, "revenue"), _g(v, "receivables")),
        ("receivables turnover", "receivable turnover", "debtor turnover", "debtors turnover"),
        "How quickly credit sales are collected.",
    ),
    # ── Growth (need the prior period) ──────────────────────────────────
    RatioDefinition(
        "revenue_growth", "Revenue Growth", "growth",
        "(Revenue − Prior Revenue) / |Prior Revenue| × 100", ("revenue", "prev_revenue"), "%",
        lambda v: pct_change(_g(v, "revenue"), _g(v, "prev_revenue")),
        ("revenue growth", "sales growth", "top line growth", "topline growth"),
        "Year-over-year change in revenue.", needs_previous=True,
    ),
    RatioDefinition(
        "profit_growth", "Net Profit Growth", "growth",
        "(Net Profit − Prior Net Profit) / |Prior Net Profit| × 100", ("net_income", "prev_net_income"), "%",
        lambda v: pct_change(_g(v, "net_income"), _g(v, "prev_net_income")),
        ("profit growth", "net profit growth", "earnings growth", "net income growth", "pat growth"),
        "Year-over-year change in net profit.", needs_previous=True,
    ),
    RatioDefinition(
        "eps_growth", "EPS Growth", "growth",
        "(EPS − Prior EPS) / |Prior EPS| × 100", ("eps", "prev_eps"), "%",
        lambda v: pct_change(_g(v, "eps"), _g(v, "prev_eps")),
        ("eps growth", "earnings per share growth"),
        "Year-over-year change in earnings per share.", needs_previous=True,
    ),
)

RATIO_BY_KEY: dict[str, RatioDefinition] = {r.key: r for r in RATIOS}
CATEGORIES: tuple[str, ...] = ("liquidity", "solvency", "profitability", "efficiency", "growth")

# Broad topics → the ratios that describe them.
TOPIC_RATIOS: dict[str, tuple[str, ...]] = {
    "profitability": ("gross_margin", "operating_margin", "ebitda_margin", "net_margin", "roe", "roa"),
    "liquidity": ("current_ratio", "quick_ratio", "cash_ratio"),
    "solvency": ("debt_to_equity", "debt_ratio", "interest_coverage"),
    "leverage": ("debt_to_equity", "debt_ratio", "interest_coverage"),
    "efficiency": ("asset_turnover", "inventory_turnover", "receivables_turnover"),
    "growth": ("revenue_growth", "profit_growth", "eps_growth"),
    "margins": ("gross_margin", "operating_margin", "ebitda_margin", "net_margin"),
}

_ALIAS_RES = [
    (re.compile(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", re.I), ratio.key)
    for ratio in RATIOS
    for alias in ratio.aliases
]
_ALIAS_RES.sort(key=lambda item: len(item[0].pattern), reverse=True)
_TOPIC_RES = [(re.compile(rf"\b{topic}\b", re.I), keys) for topic, keys in TOPIC_RATIOS.items()]


def detect_ratios(query: str, *, include_topics: bool = True) -> list[str]:
    """Ratio keys referenced by a query, by alias first and broad topic second."""
    found: list[str] = []
    consumed: list[tuple[int, int]] = []
    for pattern, key in _ALIAS_RES:
        for match in pattern.finditer(query):
            if any(match.start() < e and match.end() > s for s, e in consumed):
                continue
            consumed.append(match.span())
            if key not in found:
                found.append(key)
    if include_topics and not found:
        for pattern, keys in _TOPIC_RES:
            if pattern.search(query):
                found.extend(k for k in keys if k not in found)
    return found


# Reported line items behind each metric that can be derived when it is not reported directly.
DERIVED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "total_debt": ("short_term_debt", "long_term_debt"),
    "cogs": ("materials_consumed", "purchases_stock_in_trade", "inventory_change"),
    "gross_profit": ("revenue", "cogs"),
    "ebit": ("pbt", "finance_costs", "total_income", "total_expenses"),
    "ebitda": ("ebit", "depreciation"),
    "free_cash_flow": ("cfo", "capex"),
    "total_liabilities": ("total_assets", "equity"),
}


def query_metric_keys(query: str) -> list[str]:
    """Every reported line item a query depends on.

    "debt-to-equity" → total_debt, equity and, because total debt is often derived,
    short_term_debt and long_term_debt. Used to locate the statements behind a question.
    """
    keys: list[str] = []
    residual = query
    for ratio_key in detect_ratios(query, include_topics=False):
        definition = RATIO_BY_KEY[ratio_key]
        keys.extend(name[5:] if name.startswith("prev_") else name for name in definition.inputs)
        for alias in definition.aliases:
            residual = re.sub(re.escape(alias), " ", residual, flags=re.I)
    keys.extend(detect_metrics(residual))
    for key in list(keys):
        keys.extend(DERIVED_COMPONENTS.get(key, ()))
    return list(dict.fromkeys(keys))


def derive_metrics(values: Values) -> tuple[dict[str, float], dict[str, str]]:
    """Fill in metrics that follow arithmetically from reported ones.

    Returns ``(values, derivations)`` where ``derivations`` maps each derived key
    to the formula used, so the UI can show that the figure was not reported directly.
    """
    out: dict[str, float] = {k: float(v) for k, v in values.items() if v is not None}
    derived: dict[str, str] = {}

    def put(key: str, value: float | None, formula: str) -> None:
        if key not in out and value is not None:
            out[key] = value
            derived[key] = formula

    st, lt = out.get("short_term_debt"), out.get("long_term_debt")
    if st is not None or lt is not None:
        put("total_debt", (st or 0.0) + (lt or 0.0), "Short-Term Borrowings + Long-Term Borrowings")
    materials = out.get("materials_consumed")
    if materials is not None:
        put(
            "cogs",
            materials + out.get("purchases_stock_in_trade", 0.0) + out.get("inventory_change", 0.0),
            "Cost of Materials Consumed + Purchases of Stock-in-Trade + Changes in Inventories",
        )
    put("gross_profit", _sub(out.get("revenue"), out.get("cogs")), "Revenue − Cost of Goods Sold")
    pbt, fin = out.get("pbt"), out.get("finance_costs")
    if pbt is not None and fin is not None:
        put("ebit", pbt + abs(fin), "Profit Before Tax + Finance Costs")
    income, expenses = out.get("total_income"), out.get("total_expenses")
    if income is not None and expenses is not None and fin is not None:
        # Statements with exceptional items or discontinued operations print no single
        # profit-before-tax line; total income less total expenses is the pre-exceptional figure.
        put("ebit", income - expenses + abs(fin),
            "Total Income − Total Expenses + Finance Costs (before exceptional items; includes other income)")
    ebit, dep = out.get("ebit"), out.get("depreciation")
    if ebit is not None and dep is not None:
        put("ebitda", ebit + abs(dep), "EBIT + Depreciation & Amortisation")
    cfo, capex = out.get("cfo"), out.get("capex")
    if cfo is not None and capex is not None:
        put("free_cash_flow", cfo - abs(capex), "Cash Flow from Operations − Capital Expenditure")
    put("total_liabilities", _sub(out.get("total_assets"), out.get("equity")), "Total Assets − Shareholders' Equity")
    return out, derived


def compute_ratio(key: str, values: Values, previous: Values | None = None) -> RatioResult:
    definition = RATIO_BY_KEY[key]
    merged: dict[str, float | None] = dict(values)
    if definition.needs_previous:
        for name in definition.inputs:
            if name.startswith("prev_"):
                merged[name] = (previous or {}).get(name[5:])
    inputs = {name: merged.get(name) for name in definition.inputs}
    missing = [name for name, value in inputs.items() if value is None]
    value = None if missing else definition.compute(merged)
    return RatioResult(definition=definition, value=value, inputs=inputs, missing=missing)


def compute_ratios(
    values: Values, previous: Values | None = None, keys: list[str] | None = None
) -> list[RatioResult]:
    selected = keys if keys else [r.key for r in RATIOS]
    return [compute_ratio(k, values, previous) for k in selected if k in RATIO_BY_KEY]


def compute_cagr(first: float | None, last: float | None, years: int) -> float | None:
    return _cagr(first, last, years)
