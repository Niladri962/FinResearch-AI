"""Document ingestion pipeline.

    validate → parse → metadata → chunk → extract facts → embed → index → ready

The file is validated at upload time; this module runs the rest, usually in a
background task. Any failure marks the document ``failed`` with a user-safe
message and removes partial index entries so retrieval never sees half a document.
"""
from __future__ import annotations

import time

from sqlalchemy import delete, select

from app.config import Settings
from app.documents.metadata import company_key, extract_metadata
from app.documents.parser import parse_document
from app.documents.table_extractor import extract_structured_table, fact_scale
from app.models.database import SessionFactory, session_scope
from app.models.db import Chunk, Company, Document, FinancialFact, new_id, utcnow
from app.models.enums import DocumentStatus
from app.rag.chunking import FinancialChunker, embedding_text, is_primary_statement
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorPoint, VectorStore
from app.utils.errors import AppError, EmptyDocumentError
from app.utils.logging import get_logger
from app.utils.security import safe_join

logger = get_logger(__name__)

EMBED_PROGRESS_STEP = 64
STATEMENT_MAX_PAGES = 2     # a statement face rarely runs past three PDF pages


def get_or_create_company(session, name: str) -> Company:  # noqa: ANN001
    key = company_key(name) or name.strip().lower()
    company = session.execute(select(Company).where(Company.key == key)).scalar_one_or_none()
    if company is None:
        company = Company(name=name.strip(), key=key, aliases=[])
        session.add(company)
        session.flush()
    elif name.strip() != company.name and name.strip() not in company.aliases:
        company.aliases = [*company.aliases, name.strip()]
    return company


class IngestionPipeline:
    def __init__(
        self, settings: Settings, session_factory: SessionFactory, embedder: Embedder, vector_store: VectorStore
    ) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._embedder = embedder
        self._vectors = vector_store
        self._chunker = FinancialChunker(
            settings.chunk_target_tokens, settings.chunk_overlap_tokens, settings.chunk_max_table_tokens
        )
        # Live stage per document, shown in the UI while a large file is processed.
        # In-process only: with several API containers, other instances simply show "Processing".
        self.progress: dict[str, str] = {}

    def run(self, document_id: str) -> None:
        """Ingest one stored document. Never raises: failures are recorded on the document."""
        started = time.perf_counter()
        try:
            stats = self._ingest(document_id)
            logger.info(
                "document_ingested",
                extra={"document_id": document_id, "ms": round((time.perf_counter() - started) * 1000), **stats},
            )
        except AppError as exc:
            self._fail(document_id, exc.message, exc.code)
        except Exception as exc:  # unexpected: log the type, show a generic message
            logger.exception("ingestion_crashed", extra={"document_id": document_id})
            self._fail(document_id, "Unexpected error while processing the document.", type(exc).__name__)
        finally:
            self.progress.pop(document_id, None)

    def _fail(self, document_id: str, message: str, code: str) -> None:
        logger.warning("document_ingestion_failed", extra={"document_id": document_id, "code": code})
        try:
            self._vectors.delete_document(document_id)
        except Exception:
            pass
        with session_scope(self._session_factory) as session:
            document = session.get(Document, document_id)
            if document is None:
                return
            session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            session.execute(delete(FinancialFact).where(FinancialFact.document_id == document_id))
            document.status = DocumentStatus.FAILED.value
            document.error = message
            document.chunk_count = document.fact_count = 0
            document.processed_at = utcnow()

    def _ingest(self, document_id: str) -> dict[str, int]:
        with session_scope(self._session_factory) as session:
            document = session.get(Document, document_id)
            if document is None:
                raise AppError("Document record not found.")
            document.status = DocumentStatus.PROCESSING.value
            document.error = None
            stored_name, filename, overrides = document.stored_name, document.filename, dict(document.overrides or {})

        path = safe_join(self.settings.upload_dir, stored_name)

        # 1. Parse + metadata
        self.progress[document_id] = "Reading pages and tables"
        parsed = parse_document(path)
        meta = extract_metadata(parsed, filename, overrides)

        # 2. Chunk
        self.progress[document_id] = f"Splitting {parsed.page_count} pages into passages"
        chunks = self._chunker.chunk(parsed, meta.document_type)
        if not chunks:
            raise EmptyDocumentError("No usable content was found in the document.")

        with session_scope(self._session_factory) as session:
            company = get_or_create_company(session, meta.company) if meta.company else None
            company_id, company_name = (company.id, company.name) if company else (None, None)

        # 3. Structured facts from tables
        chunk_ids = [new_id() for _ in chunks]
        facts: list[FinancialFact] = []
        # Reports usually carry standalone and consolidated statements with the same line
        # items. The basis is read from section headings as we go; consolidated figures get
        # the higher confidence so they are the ones used when both exist.
        basis_weight = 1.0
        previous: tuple[int, str, list] | None = None   # (page, part, periods) of the last table read
        statement: tuple[str, int] = ("", 0)            # (part, page where it starts)
        for chunk_id, chunk in zip(chunk_ids, chunks):
            heading = f"{chunk.section} {chunk.table_title}".lower()
            if "consolidated" in heading:
                basis_weight = 1.0
            elif "standalone" in heading:
                basis_weight = 0.9
            part = chunk.section.split(" › ")[0]
            if part != statement[0]:
                statement = (part, chunk.page_start)
            if not chunk.table_rows:
                continue
            page_text = parsed.pages[chunk.page_start - 1].text if chunk.page_start <= parsed.page_count else ""
            # A statement interrupted mid-page resumes without repeating its column header.
            inherited = previous[2] if previous and previous[:2] == (chunk.page_start, part) else None
            table = extract_structured_table(
                chunk.table_rows,
                title=chunk.table_title,
                context=page_text[:800],
                default_unit=meta.unit,
                default_currency=meta.currency,
                fallback_periods=inherited,
            )
            if table is None:
                continue
            previous = (chunk.page_start, part, table.period_objects)
            # The face of a statement (its first pages) outranks note tables, which often
            # repeat a line item for a subsidiary, a segment or a reconciliation.
            primary = is_primary_statement(part) and chunk.page_start - statement[1] <= STATEMENT_MAX_PAGES
            weight = basis_weight * (1.0 if primary else 0.8)
            for fact in table.facts:
                facts.append(
                    FinancialFact(
                        company_id=company_id, document_id=document_id, chunk_id=chunk_id,
                        metric=fact.metric, label=fact.label[:255],
                        period_label=fact.period.label, period_kind=fact.period.kind,
                        fiscal_year=fact.period.fiscal_year, quarter=fact.period.quarter,
                        value=fact.value, scale=fact_scale(fact.metric, table),
                        unit="" if fact_scale(fact.metric, table) == 1.0 and table.scale != 1.0 else (table.unit or ""),
                        currency=table.currency or "", page=chunk.page_start,
                        confidence=round(fact.confidence * weight, 3),
                    )
                )

        # 4. Embed + index
        texts = [
            embedding_text(c.text, company=company_name, title=meta.title, section=c.section) for c in chunks
        ]
        # Embed shortest-first: a batch is padded to its longest text, so mixing a 700-token
        # table with short paragraphs wastes most of the compute (measured ~2.7x slower).
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors: list[list[float]] = [[] for _ in texts]
        for start in range(0, len(order), EMBED_PROGRESS_STEP):
            self.progress[document_id] = f"Embedding passages {start} of {len(texts)}"
            batch = order[start:start + EMBED_PROGRESS_STEP]
            for index, vector in zip(batch, self._embedder.embed_documents([texts[i] for i in batch])):
                vectors[index] = vector
        self.progress[document_id] = "Saving to the index"
        self._vectors.ensure_collection(len(vectors[0]))
        self._vectors.delete_document(document_id)  # idempotent re-ingestion
        self._vectors.upsert([
            VectorPoint(
                id=chunk_id,
                vector=vector,
                payload={
                    "document_id": document_id, "company_id": company_id, "fiscal_year": meta.fiscal_year,
                    "document_type": meta.document_type, "chunk_type": chunk.chunk_type, "page": chunk.page_start,
                },
            )
            for chunk_id, chunk, vector in zip(chunk_ids, chunks, vectors)
        ])

        # 5. Persist (BM25 picks the new chunks up from here on its next query)
        with session_scope(self._session_factory) as session:
            document = session.get(Document, document_id)
            if document is None:  # deleted while processing
                self._vectors.delete_document(document_id)
                return {"chunks": 0, "facts": 0}
            session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            session.execute(delete(FinancialFact).where(FinancialFact.document_id == document_id))
            session.add_all(
                Chunk(
                    id=chunk_id, document_id=document_id, company_id=company_id, chunk_index=index,
                    text=chunk.text, page_start=chunk.page_start, page_end=chunk.page_end,
                    section=chunk.section, chunk_type=chunk.chunk_type, token_count=chunk.token_count,
                )
                for index, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks))
            )
            session.add_all(facts)
            document.company_id = company_id
            document.title = meta.title
            document.document_type = meta.document_type
            document.fiscal_year = meta.fiscal_year
            document.quarter = meta.quarter
            document.reporting_period = meta.reporting_period
            document.currency = meta.currency
            document.unit = meta.unit[0] if meta.unit else None
            document.page_count = parsed.page_count
            document.chunk_count = len(chunks)
            document.fact_count = len(facts)
            document.warnings = parsed.warnings
            document.status = DocumentStatus.READY.value
            document.processed_at = utcnow()
        return {"pages": parsed.page_count, "chunks": len(chunks), "facts": len(facts)}
