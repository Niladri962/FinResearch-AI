"""Citation and numeric-grounding verification.

After generation the answer is checked against what the model was actually given:

  * citation markers that reference a non-existent source are removed;
  * every figure in the answer must appear in a retrieved source or in a value
    computed by the financial engine — anything else is reported as unsupported;
  * sentences that state figures without citing anything are counted.

The result drives the "grounded / partially grounded" badge in the UI.
"""
from __future__ import annotations

import re
from typing import Iterable

from app.models.schemas import Calculation, ChartSpec, Citation, DataTable, RetrievedChunk, ValidationReport
from app.utils.text import truncate

_MARKER = re.compile(r"\[((?:[SCT]\d+)(?:\s*[,;]\s*(?:[SCT]\d+))*)\]")
_ID = re.compile(r"[SCT]\d+")
_NUMBER = re.compile(r"(?<![A-Za-z\d])(\d[\d,]*(?:\.\d+)?)")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_PERIOD_TOKEN = re.compile(r"(?i)\b(?:FY|Q[1-4]\s*FY|CY|H[12]\s*FY)\s*'?\d{2,4}(?:\s*[-–/]\s*\d{2,4})?")
_LIST_PREFIX = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?\d{1,2}[.)]\s")


def extract_citation_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in _MARKER.finditer(text):
        for cid in _ID.findall(match.group(1)):
            if cid not in ids:
                ids.append(cid)
    return ids


def strip_invalid_citations(text: str, valid: set[str]) -> tuple[str, list[str]]:
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        ids = _ID.findall(match.group(1))
        kept = [i for i in ids if i in valid]
        invalid.extend(i for i in ids if i not in valid and i not in invalid)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _MARKER.sub(replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned), invalid


def _numbers(text: str) -> list[tuple[str, float]]:
    """Figures stated in ``text`` as ``(as_written, value)``, ignoring non-claims.

    Skipped: citation markers, period labels (FY2025), bare years, list numbering
    and small standalone integers ("three factors", "5 years").
    """
    text = _MARKER.sub(" ", text)
    text = _PERIOD_TOKEN.sub(" ", text)
    text = _LIST_PREFIX.sub(" ", text)
    found: list[tuple[str, float]] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1).rstrip(",")
        tail = text[match.end():match.end() + 1]
        is_percent = tail == "%"
        if _YEAR.match(raw) and not is_percent:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if "." not in raw and "," not in raw and value <= 12 and not is_percent:
            continue
        found.append((raw, value))
    return found


def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def _supported(raw: str, value: float, allowed: list[float]) -> bool:
    """A stated figure is supported when it equals a known value at the precision it was written."""
    tolerance = 0.5 * 10 ** (-_decimals(raw)) + 1e-9
    return any(abs(abs(value) - abs(known)) <= tolerance for known in allowed)


def _allowed_numbers(
    evidence: Iterable[RetrievedChunk], calculations: Iterable[Calculation],
    tables: Iterable[DataTable], charts: Iterable[ChartSpec], query: str,
) -> list[float]:
    values: list[float] = []

    def add_text(text: str) -> None:
        for match in _NUMBER.finditer(text):
            try:
                values.append(float(match.group(1).rstrip(",").replace(",", "")))
            except ValueError:
                continue

    for chunk in evidence:
        add_text(chunk.text)
    for calc in calculations:
        if calc.value is not None:
            values.append(calc.value)
        if calc.change is not None:
            values.append(calc.change)
        add_text(f"{calc.display} {calc.note or ''} {calc.formula}")
        for item in calc.inputs:
            if item.value is not None:
                values.append(item.value)
            add_text(item.display)
    for table in tables:
        add_text(table.note or "")
        for row in table.rows:
            add_text(" ".join(row))
    for chart in charts:
        for series in chart.series:
            values.extend(v for v in series.data if v is not None)
    add_text(query)  # the user's own figures are not model claims
    return values


def validate_answer(
    answer: str,
    *,
    evidence: list[RetrievedChunk],
    calculations: list[Calculation],
    tables: list[DataTable],
    charts: list[ChartSpec],
    query: str = "",
    require_citations: bool = True,
) -> tuple[str, ValidationReport]:
    """Return the cleaned answer and a validation report."""
    valid_ids = (
        {f"S{i}" for i in range(1, len(evidence) + 1)}
        | {c.id for c in calculations if c.id}
        | {t.id for t in tables if t.id}
    )
    cleaned, invalid = strip_invalid_citations(answer, valid_ids)
    cited = extract_citation_ids(cleaned)
    report = ValidationReport(cited_ids=cited, invalid_citations=invalid)

    if not require_citations:
        report.status = "not_applicable"
        return cleaned, report

    allowed = _allowed_numbers(evidence, calculations, tables, charts, query)
    stated = _numbers(cleaned)
    unsupported: list[str] = []
    for raw, value in stated:
        if not _supported(raw, value, allowed) and raw not in unsupported:
            unsupported.append(raw)
    report.unsupported_numbers = unsupported[:20]

    # Checked per paragraph/bullet: a citation at the end of a bullet covers the whole bullet.
    for block in cleaned.split("\n"):
        stripped = block.strip()
        if not stripped or stripped.startswith(("|", "#", ">", "_")):
            continue  # table rows, headings and notes carry no claims of their own
        if _numbers(stripped) and not _MARKER.search(stripped):
            report.uncited_numeric_sentences += 1

    supported = len(stated) - sum(1 for raw, _ in stated if raw in unsupported)
    report.grounding_score = round(supported / len(stated), 3) if stated else (1.0 if cited else None)

    if invalid:
        report.warnings.append(f"Removed {len(invalid)} citation(s) that did not match any provided source.")
    if unsupported:
        report.warnings.append(
            f"{len(unsupported)} figure(s) could not be matched to the sources or computed metrics: "
            + ", ".join(unsupported[:8])
        )
    if report.uncited_numeric_sentences:
        report.warnings.append(f"{report.uncited_numeric_sentences} passage(s) state figures without a citation.")

    if not cited:
        report.status = "ungrounded"
        report.warnings.append("The answer does not cite any source.")
    elif unsupported or invalid:
        report.status = "partially_grounded"
    else:
        report.status = "grounded"
    return cleaned, report


def build_citations(evidence: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            id=f"S{index}",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            filename=chunk.filename,
            company_name=chunk.company_name,
            page=chunk.page_start,
            page_end=chunk.page_end if chunk.page_end != chunk.page_start else None,
            section=chunk.section,
            chunk_type=chunk.chunk_type,
            snippet=truncate(chunk.text, 420),
            score=chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score,
        )
        for index, chunk in enumerate(evidence, start=1)
    ]
