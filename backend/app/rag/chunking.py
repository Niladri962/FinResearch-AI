"""Financial-aware chunking.

Rules
  * A table is one chunk. Oversized tables are split on row boundaries and the
    header rows are repeated in every part, so no part loses its column meaning.
  * Headings open a new section; narrative never crosses a section boundary.
  * Narrative is packed paragraph-by-paragraph up to a token budget with a
    sentence-level overlap, never cut mid-sentence.
  * Every chunk records page range, section and a semantic ``chunk_type``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.documents.parser import Element, ParsedDocument
from app.documents.table_extractor import is_numeric_cell, table_to_markdown
from app.financial.metrics import match_metric
from app.models.enums import ChunkType, DocumentType
from app.utils.text import count_tokens, split_sentences

_STATEMENT = re.compile(
    r"(?i)balance sheet|profit (and|&) loss|income statement|statement of (operations|cash flows?|financial position|"
    r"comprehensive income)|cash flow statement|financial (highlights|summary|results)|results of operations"
)
_SECTION_TYPES: list[tuple[re.Pattern[str], ChunkType]] = [
    (re.compile(r"(?i)\brisks?\b|threats|uncertaint|concerns"), ChunkType.RISK_FACTOR),
    (re.compile(r"(?i)accounting polic|basis of preparation|critical accounting"), ChunkType.ACCOUNTING_POLICY),
    (re.compile(r"(?i)^notes?\b|notes to (the )?(consolidated |standalone )?financial"), ChunkType.NOTES),
    (re.compile(r"(?i)question|q\s*&\s*a|prepared remarks|opening remarks|closing remarks"), ChunkType.EARNINGS_COMMENTARY),
    (re.compile(
        r"(?i)management|md&a|discussion and analysis|outlook|guidance|strategy|strategic|chairman|"
        r"letter to shareholders|ceo|cfo|review of operations|business overview|opportunit"
    ), ChunkType.MANAGEMENT_COMMENTARY),
]
MIN_NARRATIVE_TOKENS = 12
_RISK_WORDS = re.compile(
    r"(?i)\b(risk|adverse(ly)?|uncertain(ty|ties)?|volatil(e|ity)|exposure|headwinds?|litigation|"
    r"default|impairment|downturn|disruption)\b"
)


@dataclass
class ChunkData:
    text: str
    page_start: int
    page_end: int
    section: str
    chunk_type: str
    token_count: int
    table_rows: list[list[str]] | None = None   # full grid, set on the first part of a table
    table_title: str = ""


def _narrative_type(section: str, text: str, document_type: str) -> str:
    for pattern, chunk_type in _SECTION_TYPES:
        if pattern.search(section):
            return chunk_type.value
    if document_type == DocumentType.EARNINGS_CALL.value:
        return ChunkType.EARNINGS_COMMENTARY.value
    words = max(count_tokens(text), 1)
    if len(_RISK_WORDS.findall(text)) / words > 0.025:
        return ChunkType.RISK_FACTOR.value
    if document_type == DocumentType.MDA.value:
        return ChunkType.MANAGEMENT_COMMENTARY.value
    return ChunkType.NARRATIVE.value


def _table_type(rows: list[list[str]], title: str, section: str) -> str:
    if _STATEMENT.search(f"{title} {section}"):
        return ChunkType.FINANCIAL_STATEMENT.value
    labels = [next((c for c in row if re.search(r"[A-Za-z]", c)), "") for row in rows]
    if sum(1 for label in labels if label and match_metric(label)) >= 2:
        return ChunkType.FINANCIAL_STATEMENT.value
    return ChunkType.TABLE.value


_TOP_LEVEL = re.compile(r"(?i)^(notes?\s+(to|forming|on)\b|independent auditor|directors'? report|corporate governance)")
_PRIMARY = re.compile(
    r"(?i)balance sheet|statement of (profit (and|&) loss|operations|cash flows?|financial position|income)|"
    r"income statement|cash flow statement|financial highlights"
)


def is_primary_statement(section: str, title: str = "") -> bool:
    """True for the face of a balance sheet, income statement or cash flow statement."""
    return bool(_PRIMARY.search(f"{section} {title}"))


def _header_row_count(rows: list[list[str]]) -> int:
    """Leading rows without figures (other than period labels) form the header."""
    count = 0
    for row in rows[:3]:
        body = [c for c in row[1:] if c]
        plain_numbers = [c for c in body if is_numeric_cell(c) and not re.fullmatch(r"(19|20)\d{2}", c.strip())]
        if plain_numbers:
            break
        count += 1
    return max(1, min(count, len(rows) - 1)) if len(rows) > 1 else 1


class FinancialChunker:
    def __init__(self, target_tokens: int = 320, overlap_tokens: int = 48, max_table_tokens: int = 900) -> None:
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.max_table = max_table_tokens

    # ── Tables ───────────────────────────────────────────────────────────
    def _table_chunks(
        self, element: Element, page: int, section: str, title: str, caption: str = ""
    ) -> list[ChunkData]:
        rows = element.rows or []
        if not rows:
            return []
        chunk_type = _table_type(rows, title, section)
        # The caption usually states the unit ("INR in crore"), so it travels with the table.
        prefix = "".join(f"{line}\n" for line in (title, caption) if line)
        title = " ".join(p for p in (title, caption) if p)
        whole = prefix + table_to_markdown(rows)
        if count_tokens(whole) <= self.max_table:
            return [ChunkData(whole, page, page, section, chunk_type, count_tokens(whole), rows, title)]

        header_n = _header_row_count(rows)
        header, body = rows[:header_n], rows[header_n:]
        budget = self.max_table - count_tokens(prefix + table_to_markdown(header))
        chunks: list[ChunkData] = []
        part: list[list[str]] = []
        used = 0

        def emit() -> None:
            if not part:
                return
            label = f"{title} (part {len(chunks) + 1})" if title else f"Table (part {len(chunks) + 1})"
            text = f"{label}\n{table_to_markdown(header + part)}"
            chunks.append(
                ChunkData(text, page, page, section, chunk_type, count_tokens(text),
                          rows if not chunks else None, title)
            )

        for row in body:
            cost = count_tokens(" | ".join(row)) + 2
            if part and used + cost > max(budget, 60):
                emit()
                part, used = [], 0
            part.append(row)
            used += cost
        emit()
        return chunks

    # ── Narrative ────────────────────────────────────────────────────────
    def _split_long(self, text: str) -> list[str]:
        """Break a paragraph that alone exceeds the budget, on sentence boundaries."""
        if count_tokens(text) <= self.target:
            return [text]
        pieces: list[str] = []
        current: list[str] = []
        size = 0
        for sentence in split_sentences(text) or [text]:
            tokens = count_tokens(sentence)
            if current and size + tokens > self.target:
                pieces.append(" ".join(current))
                current, size = [], 0
            current.append(sentence)
            size += tokens
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _overlap_tail(self, text: str) -> str:
        if self.overlap <= 0:
            return ""
        tail: list[str] = []
        size = 0
        for sentence in reversed(split_sentences(text)):
            tokens = count_tokens(sentence)
            if size + tokens > self.overlap:
                break
            tail.insert(0, sentence)
            size += tokens
        return " ".join(tail)

    def chunk(self, parsed: ParsedDocument, document_type: str = "other") -> list[ChunkData]:
        chunks: list[ChunkData] = []
        section = ""
        parent = ""                     # enclosing top-level heading, for "Parent › Sub" section paths
        heading: str | None = None      # heading waiting to be attached to the next content
        last_heading = ""
        buffer: list[tuple[str, int]] = []
        fresh = 0                       # tokens in the buffer that are not overlap carry-over
        carried = False                 # buffer starts with overlap from the previous chunk

        def flush(carry: bool) -> None:
            nonlocal buffer, fresh, carried
            carried = False
            if buffer and fresh > 0:
                text = "\n\n".join(t for t, _ in buffer)
                # Stray titles, running headers and page furniture are not worth indexing.
                if fresh >= MIN_NARRATIVE_TOKENS:
                    chunks.append(
                        ChunkData(
                            text=text,
                            page_start=buffer[0][1],
                            page_end=buffer[-1][1],
                            section=section,
                            chunk_type=_narrative_type(section, text, document_type),
                            token_count=count_tokens(text),
                        )
                    )
                tail = self._overlap_tail(buffer[-1][0]) if carry else ""
                buffer = [(tail, buffer[-1][1])] if tail else []
                carried = bool(tail)
            else:
                buffer = []
            fresh = 0

        for page in parsed.pages:
            for element in page.elements:
                if element.kind == "heading":
                    flush(carry=False)
                    # Notes are typeset smaller than statement titles but start a new top-level part.
                    if element.level <= 1 or not parent or _TOP_LEVEL.match(element.text):
                        parent, section = element.text, element.text[:250]
                    else:
                        section = f"{parent} › {element.text}"[:250]
                    heading = last_heading = element.text
                elif element.kind == "table":
                    # A short line right before a table is its caption, not a chunk of its own.
                    caption = ""
                    if buffer and not carried and 0 < fresh <= 30:
                        pieces = [t for t, _ in buffer if t != last_heading]
                        caption, buffer, fresh = " ".join(pieces), [], 0
                    flush(carry=False)
                    chunks.extend(
                        self._table_chunks(element, page.number, section, last_heading[:200], caption[:200])
                    )
                    heading = None
                else:
                    text = element.text.strip()
                    if not text:
                        continue
                    for piece in self._split_long(text):
                        tokens = count_tokens(piece)
                        if fresh and fresh + tokens > self.target:
                            flush(carry=True)
                        if heading is not None:
                            buffer.insert(0, (heading, page.number))
                            heading = None
                        buffer.append((piece, page.number))
                        fresh += tokens
        flush(carry=False)
        return chunks


def embedding_text(chunk_text: str, *, company: str | None, title: str, section: str) -> str:
    """Prefix the chunk with its provenance so the embedding carries document context."""
    header = " | ".join(p for p in (company, title, section) if p)
    return f"{header}\n{chunk_text}" if header else chunk_text
