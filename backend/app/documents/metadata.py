"""Document-level metadata extraction: type, company, fiscal period, currency and unit.

Everything here is heuristic and deterministic. Values supplied by the user at
upload time always take precedence over what is detected.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.documents.parser import ParsedDocument
from app.documents.table_extractor import detect_currency, detect_unit
from app.financial.periods import find_periods
from app.models.enums import DOCUMENT_TYPE_LABELS, DocumentType

# (type, strong phrases, weak phrases). Earlier entries win ties.
_TYPE_RULES: list[tuple[DocumentType, tuple[str, ...], tuple[str, ...]]] = [
    (DocumentType.EARNINGS_CALL,
     ("earnings call", "earnings conference call", "conference call transcript", "call transcript", "concall"),
     ("operator", "moderator", "question-and-answer", "q&a session", "prepared remarks")),
    (DocumentType.ANNUAL_REPORT,
     ("annual report", "integrated report", "form 10-k", "10-k"),
     ("directors' report", "chairman", "corporate governance", "auditor's report")),
    (DocumentType.QUARTERLY_REPORT,
     ("quarterly report", "form 10-q", "10-q", "quarterly results", "unaudited financial results",
      "interim report", "results for the quarter"),
     ("quarter ended", "three months ended")),
    (DocumentType.INVESTOR_PRESENTATION,
     ("investor presentation", "investor day", "earnings presentation", "results presentation", "analyst presentation"),
     ("safe harbour", "safe harbor", "forward-looking statements")),
    (DocumentType.RESEARCH_REPORT,
     ("equity research", "initiating coverage", "research report", "target price", "price target"),
     ("we rate", "overweight", "underweight", "valuation")),
    (DocumentType.MDA,
     ("management discussion and analysis", "management's discussion and analysis", "md&a"),
     ("results of operations", "liquidity and capital resources")),
    (DocumentType.FINANCIAL_STATEMENT,
     ("financial statements", "balance sheet", "statement of profit and loss", "statement of cash flows",
      "income statement"),
     ("notes to the financial statements", "total assets")),
    (DocumentType.COMPANY_FILING,
     ("form 8-k", "prospectus", "offer document", "regulatory filing", "stock exchange filing"),
     ("securities and exchange", "sebi", "listing regulations")),
]

_COMPANY_SUFFIX = (
    r"(?:Limited|Ltd\.?|Incorporated|Inc\.?|Corporation|Corp\.?|PLC|plc|LLC|L\.L\.C\.|"
    r"Company|Co\.|Holdings|Group|AG|S\.A\.|SA|N\.V\.|NV|SE|Bank|Industries|Technologies|Enterprises)"
)
# Name words may contain inner dots ("J.P.Morgan") but not end with one, so a name never
# runs across a sentence boundary ("… Globex Corporation. Globex Corporation is …").
_NAME_WORD = r"[A-Z][\w&'\-]*(?:\.[\w&'\-]+)*"
_COMPANY_RE = re.compile(rf"\b((?:{_NAME_WORD}\s+){{1,5}}{_COMPANY_SUFFIX})(?=[\s,.;:)\n]|$)")
_COMPANY_NOISE = re.compile(r"(?i)^(the|a|an|of|for|and|to|in|annual|report|our|your|this)\s+")
_LEGAL_SUFFIX = re.compile(
    r"(?i)[\s,]+(limited|ltd\.?|incorporated|inc\.?|corporation|corp\.?|plc|llc|l\.l\.c\.|company|co\.|"
    r"ag|s\.a\.|sa|n\.v\.|nv|se)\s*$"
)
_FILENAME_NOISE = re.compile(
    r"(?i)\b(annual|report|quarterly|integrated|earnings|call|transcript|concall|investor|presentation|"
    r"financial|statements?|results?|final|draft|copy|signed|consolidated|standalone|fy|q[1-4]|10[- ]?[kq]|"
    r"ar|pdf|docx|xlsx|v\d+)\b"
)


@dataclass
class DocumentMetadata:
    company: str | None = None
    document_type: str = DocumentType.OTHER.value
    fiscal_year: int | None = None
    quarter: int | None = None
    reporting_period: str | None = None
    currency: str | None = None
    unit: tuple[str, float] | None = None
    detected: dict[str, object] = field(default_factory=dict)

    @property
    def title(self) -> str:
        label = DOCUMENT_TYPE_LABELS.get(self.document_type, "Document")
        return f"{label} {self.reporting_period}" if self.reporting_period else label


def company_key(name: str) -> str:
    """Normalised identity used to de-duplicate companies ("Acme Ltd." == "ACME Limited")."""
    name = re.sub(r"\((?:fictitious|the company|formerly[^)]*)\)", " ", name, flags=re.I)
    previous = None
    while previous != name:
        previous = name
        name = _LEGAL_SUFFIX.sub("", name.strip())
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def classify_document(text: str, filename: str) -> str:
    head = text[:1500].lower()   # cover page: what the document calls itself
    body = text[:40000].lower()
    name = re.sub(r"[_\-.]+", " ", filename.lower())
    scores: Counter[str] = Counter()
    for doc_type, strong, weak in _TYPE_RULES:
        for phrase in strong:
            if phrase in name:
                scores[doc_type.value] += 10
            if phrase in head:
                scores[doc_type.value] += 6
            scores[doc_type.value] += min(body.count(phrase), 3)
        for phrase in weak:
            if phrase in body:
                scores[doc_type.value] += 1
    if not scores or max(scores.values()) < 3:
        return DocumentType.OTHER.value
    order = [t.value for t, _, _ in _TYPE_RULES]
    return max(scores, key=lambda t: (scores[t], -order.index(t)))


def detect_company(text: str, filename: str) -> str | None:
    candidates: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    display: dict[str, str] = {}
    for match in _COMPANY_RE.finditer(text[:8000]):
        name = _COMPANY_NOISE.sub("", re.sub(r"\s+", " ", match.group(1)).strip())
        if len(name) < 5 or len(name.split()) < 2:
            continue
        key = company_key(name)
        if not key:
            continue
        candidates[key] += 1
        first_seen.setdefault(key, match.start())
        display.setdefault(key, name)
    if candidates:
        # Most frequently named entity; ties go to the one mentioned first.
        best = max(candidates, key=lambda k: (candidates[k], -first_seen[k]))
        return display[best]
    stem = re.sub(r"[_\-.]+", " ", Path(filename).stem)
    stem = re.sub(r"(?i)\bfy\s*\d{2,4}\b|\b(19|20)\d{2}\b|\b\d{2}\b", " ", stem)
    stem = re.sub(r"\s+", " ", _FILENAME_NOISE.sub(" ", stem)).strip()
    return stem.title() if len(stem) >= 2 else None


def detect_period(text: str, filename: str, document_type: str) -> tuple[int | None, int | None, str | None]:
    """Return ``(fiscal_year, quarter, reporting_period)``."""
    quarterly_type = document_type in (
        DocumentType.QUARTERLY_REPORT.value, DocumentType.EARNINGS_CALL.value, DocumentType.INVESTOR_PRESENTATION.value
    )
    for source in (re.sub(r"[_\-.]+", " ", filename), text[:1500], text[:8000]):
        periods = find_periods(source)
        if not periods:
            continue
        quarters = [p for p in periods if p.kind == "quarter"]
        if quarters and quarterly_type:
            first = quarters[0]
            return first.fiscal_year, first.quarter, first.label
        counts = Counter(p.fiscal_year for p in periods)
        top = max(counts.values())
        year = max(y for y, c in counts.items() if c == top)  # ties → the latest year
        return year, None, f"FY{year}"
    return None, None, None


def extract_metadata(parsed: ParsedDocument, filename: str, overrides: dict[str, object] | None = None) -> DocumentMetadata:
    overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
    head = parsed.text(max_pages=5)
    document_type = str(overrides.get("document_type") or classify_document(head, filename))
    year, quarter, period = detect_period(head, filename, document_type)
    detected_company = detect_company(head, filename)

    if overrides.get("fiscal_year"):
        year = int(overrides["fiscal_year"])  # type: ignore[arg-type]
        quarter = int(overrides["quarter"]) if overrides.get("quarter") else None  # type: ignore[arg-type]
    elif overrides.get("quarter"):
        quarter = int(overrides["quarter"])  # type: ignore[arg-type]
    if year is not None:
        period = f"Q{quarter} FY{year}" if quarter else f"FY{year}"

    # Currency and unit: the most common declaration across the document.
    units: Counter[tuple[str, float]] = Counter()
    currencies: Counter[str] = Counter()
    for page in parsed.pages[:60]:
        text = page.text
        if (unit := detect_unit(text)) is not None:
            units[unit] += 1
        if (currency := detect_currency(text)) is not None:
            currencies[currency] += 1

    return DocumentMetadata(
        company=str(overrides.get("company") or detected_company or "") or None,
        document_type=document_type,
        fiscal_year=year,
        quarter=quarter,
        reporting_period=period,
        currency=currencies.most_common(1)[0][0] if currencies else None,
        unit=units.most_common(1)[0][0] if units else None,
        detected={"company": detected_company, "document_type": document_type},
    )
