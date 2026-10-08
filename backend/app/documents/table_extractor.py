"""Financial table understanding.

Two jobs:

1. ``detect_text_tables`` finds tables that exist only as aligned text (no ruling
   lines) so the chunker can keep them intact.
2. ``extract_structured_table`` turns any table grid into structured facts —
   canonical metric × reporting period × value — with unit and currency.

Extraction is conservative by design: a row is mapped only when its label is a
recognised line item and its values align with detected period columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.financial.metrics import is_per_share, match_metric, normalize_label
from app.financial.periods import Period, find_periods, strip_periods

# Many Indian filings embed a rupee-symbol font whose glyph extracts as "H" or "`".
_RUPEE_GLYPH = r"(?<![A-Za-z])[H`](?=\s?[\d(])"
_CURRENCY_WORDS = re.compile(rf"(?i:\b(rs\.?|inr|usd|us\$|eur|gbp)\b)|[$₹€£]|{_RUPEE_GLYPH}")
_PLAIN_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_PLACEHOLDER = re.compile(r"(?i)^(-+|–|—|nil|n/?a|n\.a\.)$")
_HEADER_WORDS = re.compile(
    r"(?i)\b(particulars?|description|for the|year(s)? ended|quarter(s)? ended|months? ended|"
    r"as (at|of|on)|year|years|quarter|period|ended|notes?|no\.?|amounts?|audited|unaudited|"
    r"consolidated|standalone|restated|in|rs\.?|inr|usd|eur|gbp|crores?|cr|lakhs?|lacs?|"
    r"millions?|mn|billions?|bn|thousands?|except per share data|three|six|nine|twelve|half)\b"
)
_INTERIM_CONTEXT = re.compile(r"(?i)\b(quarter(s)? ended|quarterly|(three|six|nine) months|half[- ]year)\b")
_BS_SECTION = re.compile(r"^(non current|current) (assets|liabilities)\b")

_UNIT_RE = re.compile(
    r"(?i)\b(?:in|of)\s+(?:rs\.?|inr|usd|us\$|eur|gbp|₹|\$|€|£|h\b|`)?\s*"
    r"(crores?|cr|lakhs?|lacs?|millions?|mn|billions?|bn|thousands?)\b|"
    r"(?:₹|rs\.?|inr|usd|us\$|\$|€|£)\s*(crores?|cr|lakhs?|lacs?|millions?|mn|billions?|bn|thousands?)\b|"
    r"\(\s*(crores?|lakhs?|millions?|billions?|thousands?)\s*\)|('000s?)"
)
_UNIT_SCALES = {
    "crore": ("crore", 1e7), "cr": ("crore", 1e7),
    "lakh": ("lakh", 1e5), "lac": ("lakh", 1e5),
    "million": ("million", 1e6), "mn": ("million", 1e6),
    "billion": ("billion", 1e9), "bn": ("billion", 1e9),
    "thousand": ("thousand", 1e3), "'000": ("thousand", 1e3),
}
_CURRENCY_RES = [
    ("INR", re.compile(r"(?i:₹|\bINR\b|\bRs\.?(?=[\s\d]|$)|\brupees\b)|(?:\bin|\() ?[H`](?![A-Za-z])|\bH ?(?:crores?|lakhs?)\b")),
    ("USD", re.compile(r"US\$|\bUSD\b|\$|\bUS dollars?\b", re.I)),
    ("EUR", re.compile(r"€|\bEUR\b|\beuros?\b", re.I)),
    ("GBP", re.compile(r"£|\bGBP\b|\bpounds? sterling\b", re.I)),
]


# ── Number / unit parsing ────────────────────────────────────────────────────
def parse_number(raw: str | None) -> float | None:
    """Parse a reported figure. Parentheses and leading minus mean negative.

    Returns ``None`` for blanks, placeholders ("-", "nil") and percentages.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.endswith("%"):
        return None
    s = _CURRENCY_WORDS.sub("", s).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    if s[:1] in "-−–":
        negative, s = True, s[1:].strip()
    if not _PLAIN_NUMBER.fullmatch(s):
        return None
    value = float(s.replace(",", ""))
    return -value if negative else value


def is_numeric_cell(raw: str) -> bool:
    s = raw.strip()
    if not s:
        return False
    if _PLACEHOLDER.match(s):
        return True
    return parse_number(s.rstrip("%")) is not None


def detect_unit(text: str) -> tuple[str, float] | None:
    match = _UNIT_RE.search(text)
    if not match:
        return None
    word = next(g for g in match.groups() if g).lower().rstrip("s")
    if word.startswith("'000"):
        word = "'000"
    return _UNIT_SCALES.get(word)


def detect_currency(text: str) -> str | None:
    hits = [(m.start(), code) for code, rx in _CURRENCY_RES if (m := rx.search(text))]
    return min(hits)[1] if hits else None


# ── Text-table detection ─────────────────────────────────────────────────────
def split_row(line: str) -> tuple[str, list[str]] | None:
    """Split ``Label  1,234  (567)`` into label and trailing numeric cells."""
    tokens = line.replace("|", " ").split()
    numbers: list[str] = []
    while len(tokens) > 1 and is_numeric_cell(tokens[-1]):
        numbers.insert(0, tokens.pop())
    label = " ".join(tokens)
    if not numbers or not re.search(r"[A-Za-z]", label):
        return None
    if len(label) > 90 or len(tokens) > 14 or label.rstrip()[-1:] in ".;,":
        return None
    return label, numbers


def _values_only(line: str) -> list[str] | None:
    """Cells of a line made of figures only, e.g. ``(395)   (78)``."""
    tokens = line.replace("|", " ").split()
    if not tokens or len(tokens) > 12 or not all(is_numeric_cell(t) for t in tokens):
        return None
    return tokens if any(parse_number(t) is not None for t in tokens) else None


def header_periods(text: str) -> list[Period]:
    periods = find_periods(text)
    if _INTERIM_CONTEXT.search(text):
        periods = [p.as_interim() for p in periods]
    return periods


def is_header_line(line: str) -> bool:
    """A row made of period labels and boilerplate only ("Particulars  FY2025  FY2024")."""
    if not find_periods(line):
        return False
    residual = _HEADER_WORDS.sub(" ", strip_periods(line))
    residual = _CURRENCY_WORDS.sub(" ", residual)
    return len(re.findall(r"[A-Za-z]", residual)) <= 3


def _is_sublabel(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) <= 70 and len(s.split()) <= 8 and s[-1] not in ".;," and bool(
        re.search(r"[A-Za-z]{3}", s)
    )


# Statements stack section labels ("Liabilities" / "Non-current liabilities" / "Financial liabilities").
MAX_LABEL_RUN = 4


def _is_wrapped_label(line: str, next_line: str) -> bool:
    """A long row label whose figures sit alone on the following line."""
    s = line.strip()
    return bool(s) and len(s) <= 160 and bool(re.search(r"[A-Za-z]{3}", s)) and _values_only(next_line) is not None


def is_header_fragment(line: str) -> bool:
    """Part of a multi-line column header ("Particulars   Note", "As at    As at")."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    residual = _CURRENCY_WORDS.sub(" ", _HEADER_WORDS.sub(" ", strip_periods(s)))
    return len(re.findall(r"[A-Za-z]", residual)) <= 3


@dataclass
class TextTable:
    start: int   # index of first line (inclusive)
    end: int     # index after last line (exclusive)
    rows: list[list[str]]


def detect_text_tables(lines: list[str]) -> list[TextTable]:
    """Locate runs of lines that form a table (≥3 numeric rows, or ≥2 under a period header)."""
    tables: list[TextTable] = []
    n, i = len(lines), 0
    while i < n:
        if not (is_header_line(lines[i]) or split_row(lines[i])):
            i += 1
            continue
        rows: list[list[str]] = []
        numeric = gap = 0
        has_header = False
        j = i
        while j < n:
            line = lines[j]
            if is_header_line(line):
                if numeric:
                    break  # a new header after data rows starts the next table
                rows.append(["Particulars", *[p.label for p in header_periods(line)]])
                has_header, gap = True, 0
            elif (parsed := split_row(line)) is not None:
                rows.append([parsed[0], *parsed[1]])
                numeric, gap = numeric + 1, 0
            elif rows and (has_header or numeric) and (values := _values_only(line)) is not None:
                # A long label wraps and its figures sit on a line of their own:
                # they belong to the row above.
                if len(rows[-1]) == 1:
                    numeric += 1
                if not (has_header and len(rows) == 1):
                    rows[-1] = [*rows[-1], *values]
                gap = 0
            elif (numeric or has_header) and gap < MAX_LABEL_RUN and (
                _is_sublabel(line) or _is_wrapped_label(line, lines[j + 1] if j + 1 < n else "")
            ):
                rows.append([line.strip()])
                gap += 1
            else:
                break
            j += 1
        while rows and len(rows[-1]) == 1:  # trailing sub-labels belong to what follows
            rows.pop()
            j -= 1
        if numeric >= 3 or (has_header and numeric >= 2):
            tables.append(TextTable(start=i, end=j, rows=rows))
            i = j
        else:
            i += 1
    return tables


# ── Structured extraction ────────────────────────────────────────────────────
@dataclass
class ExtractedFact:
    metric: str
    label: str
    period: Period
    value: float
    confidence: float


@dataclass
class StructuredRow:
    label: str
    metric: str | None
    values: dict[str, float | None]


@dataclass
class StructuredTable:
    title: str
    periods: list[str]
    period_objects: list[Period] = field(default_factory=list)
    rows: list[StructuredRow] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    currency: str | None = None
    unit: str | None = None
    scale: float = 1.0

    def as_records(self) -> list[dict[str, object]]:
        """``[{"metric": "Revenue", "FY2024": 1000, "FY2025": 1250, "currency": "INR"}]``"""
        return [
            {"metric": r.metric or r.label, "label": r.label, **r.values,
             "currency": self.currency, "unit": self.unit}
            for r in self.rows
        ]


def _clean(cell: object) -> str:
    return re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""


def _find_header(rows: list[list[str]]) -> tuple[int, list[Period]] | None:
    """The leading row naming the most periods. A caption such as "for the year ended
    31 March 2026" also names a period, so the first match is not always the header."""
    best: tuple[int, list[Period]] | None = None
    for idx, row in enumerate(rows[:6]):
        joined = " ".join(row)
        if not is_header_line(joined):
            continue
        periods: list[Period] = []
        interim = bool(_INTERIM_CONTEXT.search(joined))
        for cell in row:
            for period in find_periods(cell):
                periods.append(period.as_interim() if interim else period)
        if periods and (best is None or len(periods) > len(best[1])):
            best = (idx, periods)
    return best


def _resolve_metric(label: str, context: str, bs_section: str) -> tuple[str, float] | None:
    norm = normalize_label(label)
    # "Basic"/"Diluted" are EPS only under an earnings-per-share heading.
    if norm in ("basic", "diluted") or norm.startswith(("basic in", "diluted in", "basic rs", "diluted rs")):
        if "earnings per" not in context and "eps" not in context.split():
            return None
        # Where EPS is split by continuing/discontinued operations, the combined figure wins.
        if "discontinued" in context and "continuing" not in context:
            return None
        combined = "continuing and discontinued" in context
        return ("eps" if norm.startswith("basic") else "eps_diluted"), (0.95 if combined else 0.9)
    # A bare "Borrowings" line is long- or short-term depending on its balance-sheet section.
    if norm in ("borrowings", "financial liabilities borrowings"):
        if bs_section.startswith("non current"):
            return "long_term_debt", 0.9
        if bs_section.startswith("current"):
            return "short_term_debt", 0.9
        return "total_debt", 0.8
    return match_metric(label)


def extract_structured_table(
    rows: list[list[str]],
    *,
    title: str = "",
    context: str = "",
    default_unit: tuple[str, float] | None = None,
    default_currency: str | None = None,
    fallback_periods: list[Period] | None = None,
) -> StructuredTable | None:
    """Convert a table grid into structured rows and facts.

    Returns ``None`` when no reporting-period header can be identified — values
    are never assigned to a period by guesswork. ``fallback_periods`` are the
    columns of the table this one continues (same page, same statement), used
    when a statement is interrupted and resumes without repeating its header.
    """
    grid = [[_clean(c) for c in row] for row in rows if row and any(_clean(c) for c in row)]
    header = _find_header(grid)
    if header is None:
        if not fallback_periods:
            return None
        header = (-1, list(fallback_periods))
    header_idx, periods = header

    head_text = " ".join([title, *(" ".join(r) for r in grid[: header_idx + 1])])
    unit = detect_unit(head_text) or detect_unit(context) or default_unit
    currency = detect_currency(head_text) or detect_currency(context) or default_currency
    table = StructuredTable(
        title=title,
        period_objects=list(periods),
        periods=[p.label for p in periods],
        currency=currency,
        unit=unit[0] if unit else None,
        scale=unit[1] if unit else 1.0,
    )

    seen: dict[tuple[str, str], float] = {}
    sub_context, bs_section, eps_scope = "", "", ""
    for row in grid[header_idx + 1:]:
        cells = [c for c in row]
        label_idx = next((i for i, c in enumerate(cells) if re.search(r"[A-Za-z]", c)), None)
        if label_idx is None:
            continue
        label = cells[label_idx]
        value_cells: list[str] = []
        for cell in cells[label_idx + 1:]:
            # A single cell can hold several space-separated figures in poorly ruled tables.
            parts = cell.split() if cell and all(is_numeric_cell(p) for p in cell.split()) else [cell]
            value_cells.extend(parts)
        if "earnings per" in label.lower():
            eps_scope = "earnings per share"   # the heading row often carries a note number
        if not any(is_numeric_cell(c) for c in value_cells):
            sub_context = normalize_label(label)
            if _BS_SECTION.match(sub_context):
                bs_section = sub_context
            continue
        if len(value_cells) < len(periods):
            continue
        aligned = value_cells[-len(periods):]
        if any(c.strip().endswith("%") for c in aligned):
            continue  # percentage rows are recomputed, never ingested
        values = [parse_number(c) for c in aligned]
        resolved = _resolve_metric(label, f"{eps_scope} {sub_context}".strip(), bs_section)
        metric, confidence = resolved if resolved else (None, 0.0)
        table.rows.append(
            StructuredRow(label=label, metric=metric, values={p.label: v for p, v in zip(periods, values)})
        )
        if metric is None:
            continue
        for period, value in zip(periods, values):
            if value is None:
                continue
            key = (metric, period.label)
            if seen.get(key, -1.0) >= confidence:
                continue  # keep the first, highest-confidence occurrence
            if key in seen:
                table.facts = [f for f in table.facts if (f.metric, f.period.label) != key]
            seen[key] = confidence
            table.facts.append(ExtractedFact(metric, label, period, value, confidence))
    return table if table.rows else None


def fact_scale(metric: str, table: StructuredTable) -> float:
    """Per-share figures are reported in currency units, not the table's scale."""
    return 1.0 if is_per_share(metric) else table.scale


def table_to_markdown(rows: list[list[str]]) -> str:
    grid = [[_clean(c).replace("|", "/") for c in row] for row in rows if row]
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join([" --- "] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)
