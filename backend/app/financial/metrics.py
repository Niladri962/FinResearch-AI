"""Canonical financial-metric taxonomy and label matching.

Filings name the same line item many ways ("Revenue from operations", "Net
sales", "Turnover"). Each canonical metric lists its accepted labels; matching
is deliberately conservative — an unmapped row is better than a wrong number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    name: str
    statement: str            # income | balance | cashflow | per_share
    synonyms: tuple[str, ...]
    per_share: bool = False   # per-share values are never scaled by the reporting unit


def _m(key: str, name: str, statement: str, *synonyms: str, per_share: bool = False) -> MetricDefinition:
    return MetricDefinition(key, name, statement, tuple(synonyms), per_share)


METRICS: tuple[MetricDefinition, ...] = (
    # ── Income statement ────────────────────────────────────────────────
    _m("revenue", "Revenue", "income",
       "revenue from operations", "total revenue from operations", "net revenue from operations",
       "revenue", "revenues", "total revenue", "total revenues", "net revenue", "net revenues",
       "net sales", "sales", "total net sales", "turnover", "income from operations",
       "total income from operations", "revenue from contracts with customers"),
    _m("other_income", "Other Income", "income", "other income", "other income net"),
    _m("total_income", "Total Income", "income", "total income"),
    _m("cogs", "Cost of Goods Sold", "income",
       "cost of goods sold", "cost of revenue", "cost of revenues", "cost of sales",
       "total cost of revenue", "cost of products sold"),
    # Ind AS / Schedule III statements report cost of goods sold as three separate lines.
    _m("materials_consumed", "Cost of Materials Consumed", "income",
       "cost of materials consumed", "cost of raw materials consumed", "raw materials consumed",
       "cost of raw materials and components consumed"),
    _m("purchases_stock_in_trade", "Purchases of Stock-in-Trade", "income",
       "purchases of stock in trade", "purchase of stock in trade", "purchases of traded goods",
       "purchase of traded goods"),
    _m("inventory_change", "Changes in Inventories", "income",
       "changes in inventories", "changes in inventories of finished goods work in progress and stock in trade",
       "changes in inventories of finished goods and work in progress"),
    _m("gross_profit", "Gross Profit", "income", "gross profit", "gross margin", "gross income"),
    _m("operating_expenses", "Operating Expenses", "income",
       "total operating expenses", "operating expenses"),
    _m("total_expenses", "Total Expenses", "income", "total expenses", "total costs and expenses"),
    _m("ebitda", "EBITDA", "income",
       "ebitda", "operating ebitda", "adjusted ebitda",
       "earnings before interest tax depreciation and amortisation",
       "earnings before interest tax depreciation and amortization",
       "earnings before interest taxes depreciation and amortization"),
    _m("depreciation", "Depreciation & Amortisation", "income",
       "depreciation and amortisation expense", "depreciation and amortization expense",
       "depreciation and amortisation", "depreciation and amortization",
       "depreciation amortisation and impairment", "depreciation"),
    _m("ebit", "Operating Profit (EBIT)", "income",
       "operating profit", "operating income", "ebit", "profit from operations",
       "income from operations before tax", "earnings before interest and tax",
       "earnings before interest and taxes", "operating profit ebit"),
    _m("finance_costs", "Finance Costs", "income",
       "finance costs", "finance cost", "interest expense", "interest and finance charges",
       "interest expenses", "finance charges", "interest cost"),
    _m("pbt", "Profit Before Tax", "income",
       "profit before tax", "profit before taxation", "income before income taxes",
       "profit before income tax", "earnings before tax", "income before taxes",
       "profit loss before tax"),
    _m("tax_expense", "Tax Expense", "income",
       "tax expense", "total tax expense", "income tax expense", "provision for income taxes",
       "tax expenses", "income taxes"),
    _m("net_income", "Net Profit", "income",
       "profit for the year", "profit for the period", "net profit", "net income",
       "profit after tax", "net profit after tax", "pat", "net earnings",
       "profit loss for the year", "profit loss for the period", "net profit for the year",
       "net income attributable to shareholders", "profit attributable to owners of the company",
       "profit attributable to shareholders"),
    # ── Per share ───────────────────────────────────────────────────────
    _m("eps", "Earnings Per Share (Basic)", "per_share",
       "basic eps", "basic earnings per share", "earnings per share", "earnings per share basic",
       "eps", "eps basic", "basic", "earnings per equity share", "earnings per equity share basic",
       per_share=True),
    _m("eps_diluted", "Earnings Per Share (Diluted)", "per_share",
       "diluted eps", "diluted earnings per share", "earnings per share diluted", "eps diluted",
       "diluted", per_share=True),
    # ── Balance sheet ───────────────────────────────────────────────────
    _m("cash", "Cash & Cash Equivalents", "balance",
       "cash and cash equivalents", "cash and equivalents", "cash and bank balances",
       "cash and short term investments"),
    _m("inventory", "Inventories", "balance", "inventories", "inventory"),
    _m("receivables", "Trade Receivables", "balance",
       "trade receivables", "accounts receivable", "accounts receivable net", "receivables",
       "trade and other receivables"),
    _m("current_assets", "Current Assets", "balance", "total current assets", "current assets"),
    _m("non_current_assets", "Non-Current Assets", "balance",
       "total non current assets", "non current assets"),
    _m("total_assets", "Total Assets", "balance", "total assets"),
    _m("short_term_debt", "Short-Term Borrowings", "balance",
       "short term borrowings", "current borrowings", "short term debt",
       "current portion of long term debt", "current maturities of long term debt",
       "borrowings current"),
    _m("long_term_debt", "Long-Term Borrowings", "balance",
       "long term borrowings", "non current borrowings", "long term debt",
       "long term debt net of current portion", "borrowings non current"),
    _m("total_debt", "Total Debt", "balance",
       "total borrowings", "total debt", "borrowings", "gross debt"),
    _m("current_liabilities", "Current Liabilities", "balance",
       "total current liabilities", "current liabilities"),
    _m("non_current_liabilities", "Non-Current Liabilities", "balance",
       "total non current liabilities", "non current liabilities"),
    _m("total_liabilities", "Total Liabilities", "balance", "total liabilities"),
    _m("equity", "Shareholders' Equity", "balance",
       "total equity", "shareholders equity", "total shareholders equity", "stockholders equity",
       "total stockholders equity", "shareholders funds", "total shareholders funds", "net worth",
       "equity attributable to owners of the company", "total equity attributable to owners",
       "shareholder equity"),
    # ── Cash flow ───────────────────────────────────────────────────────
    _m("cfo", "Cash Flow from Operations", "cashflow",
       "net cash from operating activities", "net cash generated from operating activities",
       "net cash provided by operating activities", "net cash flow from operating activities",
       "net cash inflow from operating activities", "cash flow from operating activities",
       "cash flow from operations", "operating cash flow", "cash generated from operations",
       "net cash generated from operations", "net cash from operations"),
    _m("cfi", "Cash Flow from Investing", "cashflow",
       "net cash used in investing activities", "net cash from investing activities",
       "net cash flow from investing activities", "net cash provided by investing activities",
       "cash flow from investing activities"),
    _m("cff", "Cash Flow from Financing", "cashflow",
       "net cash used in financing activities", "net cash from financing activities",
       "net cash flow from financing activities", "net cash provided by financing activities",
       "cash flow from financing activities"),
    _m("capex", "Capital Expenditure", "cashflow",
       "capital expenditure", "capital expenditures", "capex",
       "purchase of property plant and equipment", "purchases of property plant and equipment",
       "purchase of property and equipment", "purchases of property and equipment",
       "purchase of fixed assets", "purchase of tangible and intangible assets",
       "payments for property plant and equipment", "additions to property plant and equipment",
       "purchase of property plant and equipment and intangible assets",
       "payment for purchase of property plant and equipment"),
    _m("free_cash_flow", "Free Cash Flow", "cashflow", "free cash flow", "fcf"),
)

METRIC_BY_KEY: dict[str, MetricDefinition] = {m.key: m for m in METRICS}

# Rows we must never map, even though they contain a known synonym.
_BLOCKLIST = re.compile(
    r"\b(margin|growth|per cent|percent|percentage|as a % of|% of|ratio|days|number of|"
    r"weighted average|deferred|comprehensive|non controlling|minority|diluted shares|"
    r"equity and liabilities|liabilities and equity|liabilities and shareholders|"
    r"liabilities and stockholders|before exceptional|exceptional items|"
    r"discontinued|per employee|change in|increase in|decrease in|movement in)\b"
)
# Trailing qualifiers that do not change which line item a label refers to.
_ALLOWED_SUFFIX = re.compile(
    r"^(net|total|gross|for the (year|period|quarter)|after tax|attributable to .*|"
    r"as at .*|a|b|c|i|ii|iii|iv|v|in .*|rs|inr|usd|refer note.*|note.*)$"
)
_ALLOWED_PREFIX = re.compile(r"^(total|net|consolidated|standalone|less|add|a|b|c|i|ii|iii|iv|v)$")

_PUNCT = re.compile(r"[^a-z0-9% ]+")
_NET_CASH = re.compile(
    r"^net cash (?:flows? |inflows? |outflows? )?(?:(?:used in|generated from|generated by|provided by|from) ?)+"
)
_FORMULA_TAG = re.compile(r"\(\s*[a-z]{1,3}(?:\s*[=+\-]\s*[a-z]{1,3})*\s*\)")
_NOTE_REF = re.compile(r"\((?:refer )?notes?\s*[\d.a-z, &-]*\)|\bnotes?\s*\d+[a-z.]*", re.I)
_LEADING_ENUM = re.compile(r"^\s*(?:\(?[ivxlc]{1,5}\)|\(?[a-h]\)|\d{1,2}[.)]|[ivxlc]{1,5}\.)\s+", re.I)


def normalize_label(label: str) -> str:
    label = _NOTE_REF.sub(" ", label)
    label = _LEADING_ENUM.sub("", label)
    label = label.lower().replace("&", " and ").replace("’", "").replace("'", "")
    label = label.replace("/(loss)", " loss").replace("/ (loss)", " loss")
    label = _FORMULA_TAG.sub(" ", label)   # "(A)", "(C=A+B)", "(I+II)"
    label = label.replace("-", " ").replace("/", " ")
    label = _PUNCT.sub(" ", label)
    label = re.sub(r"\s+", " ", label).strip()
    # "Net cash flows (used in) / generated from investing activities" → "net cash from investing activities"
    return _NET_CASH.sub("net cash from ", label)


_SYNONYM_INDEX: dict[str, str] = {}
for _metric in METRICS:
    for _syn in _metric.synonyms:
        _SYNONYM_INDEX.setdefault(normalize_label(_syn), _metric.key)
# Longest synonyms first so "total current assets" wins over "current assets".
_MULTIWORD = sorted((s for s in _SYNONYM_INDEX if " " in s), key=len, reverse=True)


def match_metric(label: str) -> tuple[str, float] | None:
    """Map a reported row label to ``(metric_key, confidence)`` or ``None``."""
    norm = normalize_label(label)
    if not norm or len(norm) > 90:
        return None
    if norm in _SYNONYM_INDEX:
        return _SYNONYM_INDEX[norm], 1.0
    # This label is long and usually wraps, so only its opening words are reliable.
    if norm.startswith("changes in inventories"):
        return "inventory_change", 0.9
    if _BLOCKLIST.search(norm):
        return None
    for syn in _MULTIWORD:
        if norm.startswith(syn + " "):
            if _ALLOWED_SUFFIX.match(norm[len(syn) + 1:]):
                return _SYNONYM_INDEX[syn], 0.85
        elif norm.endswith(" " + syn):
            if _ALLOWED_PREFIX.match(norm[: -len(syn) - 1]):
                return _SYNONYM_INDEX[syn], 0.85
    return None


def metric_name(key: str) -> str:
    definition = METRIC_BY_KEY.get(key)
    return definition.name if definition else key.replace("_", " ").title()


def is_per_share(key: str) -> bool:
    definition = METRIC_BY_KEY.get(key)
    return bool(definition and definition.per_share)


# ── Query-side detection ─────────────────────────────────────────────────────
# Phrases a user might type → canonical metric keys (checked longest-first).
_QUERY_TERMS: dict[str, str] = {
    "revenue": "revenue", "revenues": "revenue", "sales": "revenue", "turnover": "revenue",
    "top line": "revenue", "topline": "revenue",
    "ebitda": "ebitda", "ebit": "ebit", "operating profit": "ebit", "operating income": "ebit",
    "gross profit": "gross_profit",
    "net profit": "net_income", "net income": "net_income", "profit after tax": "net_income",
    "pat": "net_income", "bottom line": "net_income", "profit": "net_income",
    "profits": "net_income",
    "eps": "eps", "earnings per share": "eps",
    "total assets": "total_assets", "current assets": "current_assets",
    "current liabilities": "current_liabilities", "total liabilities": "total_liabilities",
    "debt": "total_debt", "borrowings": "total_debt", "leverage": "total_debt",
    "equity": "equity", "net worth": "equity",
    "cash flow from operations": "cfo", "operating cash flow": "cfo", "cash flow": "cfo",
    "cash flows": "cfo", "free cash flow": "free_cash_flow", "fcf": "free_cash_flow",
    "capex": "capex", "capital expenditure": "capex", "capital expenditures": "capex",
    "finance costs": "finance_costs", "interest expense": "finance_costs",
    "inventory": "inventory", "inventories": "inventory", "receivables": "receivables",
    "cash and cash equivalents": "cash", "cash balance": "cash",
    "depreciation": "depreciation", "tax expense": "tax_expense",
}
_QUERY_TERM_RES = [
    (re.compile(rf"\b{re.escape(term)}\b", re.I), key)
    for term, key in sorted(_QUERY_TERMS.items(), key=lambda kv: len(kv[0]), reverse=True)
]


def detect_metrics(query: str) -> list[str]:
    """Canonical metrics mentioned in a natural-language query (longest phrase wins)."""
    found: list[str] = []
    consumed: list[tuple[int, int]] = []
    for pattern, key in _QUERY_TERM_RES:
        for match in pattern.finditer(query):
            if any(match.start() < e and match.end() > s for s, e in consumed):
                continue
            consumed.append(match.span())
            if key not in found:
                found.append(key)
    return found
