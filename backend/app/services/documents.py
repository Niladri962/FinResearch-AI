"""Document and company use-cases: secure upload, listing, deletion, previews."""
from __future__ import annotations

import hashlib
import uuid
from datetime import timedelta
from typing import BinaryIO

from sqlalchemy import delete, func, select

from app.config import Settings
from app.financial.metrics import metric_name
from app.models.database import SessionFactory, session_scope
from app.models.db import Chunk, Company, Document, FinancialFact, Report, utcnow
from app.models.enums import DOCUMENT_TYPE_LABELS, DocumentStatus, DocumentType
from app.models.schemas import ChunkOut, CompanyOut, DocumentOut, FactOut
from app.rag.vector_store import VectorStore
from app.services.financials import FinancialDataService
from app.utils.errors import ConflictError, FileTooLargeError, NotFoundError, ValidationAppError
from app.utils.logging import get_logger
from app.utils.security import safe_join, sanitize_filename, validate_extension, validate_magic_bytes

logger = get_logger(__name__)

_CHUNK = 1024 * 1024
_VALID_TYPES = {t.value for t in DocumentType}


def document_out(document: Document, company_name: str | None) -> DocumentOut:
    out = DocumentOut.model_validate(document)
    out.company_name = company_name
    out.document_type_label = DOCUMENT_TYPE_LABELS.get(document.document_type, "Document")
    return out


def clean_overrides(
    company: str | None, document_type: str | None, fiscal_year: int | None, quarter: int | None
) -> dict[str, object]:
    """Validate the optional metadata a user supplies with an upload."""
    overrides: dict[str, object] = {}
    if company and company.strip():
        name = " ".join(company.split())
        if len(name) > 120:
            raise ValidationAppError("Company name is too long (max 120 characters).")
        overrides["company"] = name
    if document_type and document_type.strip():
        if document_type not in _VALID_TYPES:
            raise ValidationAppError(f"Unknown document_type. Allowed: {', '.join(sorted(_VALID_TYPES))}.")
        overrides["document_type"] = document_type
    if fiscal_year is not None:
        if not 1980 <= fiscal_year <= 2100:
            raise ValidationAppError("fiscal_year must be between 1980 and 2100.")
        overrides["fiscal_year"] = fiscal_year
    if quarter is not None:
        if quarter not in (1, 2, 3, 4):
            raise ValidationAppError("quarter must be 1, 2, 3 or 4.")
        overrides["quarter"] = quarter
    return overrides


class DocumentService:
    def __init__(self, settings: Settings, session_factory: SessionFactory, vector_store: VectorStore) -> None:
        self.settings = settings
        self._session_factory = session_factory
        self._vectors = vector_store

    # ── Upload ───────────────────────────────────────────────────────────
    def save_upload(self, filename: str | None, stream: BinaryIO, overrides: dict[str, object]) -> DocumentOut:
        """Validate and store an uploaded file, returning the queued document.

        The file is written under a server-generated name; the client's filename
        is kept only as a sanitised display label, so it can never steer a path.
        """
        display_name = sanitize_filename(filename or "document")
        extension = validate_extension(display_name)
        stored_name = f"{uuid.uuid4().hex}{extension}"
        target = safe_join(self.settings.upload_dir, stored_name)
        target.parent.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha256()
        size = 0
        head = b""
        try:
            with target.open("wb") as handle:
                while True:
                    block = stream.read(_CHUNK)
                    if not block:
                        break
                    size += len(block)
                    if size > self.settings.max_upload_bytes:
                        raise FileTooLargeError(
                            f"'{display_name}' exceeds the {self.settings.effective_max_upload_mb} MB upload limit."
                        )
                    if not head:
                        head = block[:2048]
                    digest.update(block)
                    handle.write(block)
            validate_magic_bytes(extension, head, target)

            sha256 = digest.hexdigest()
            with session_scope(self._session_factory) as session:
                duplicate = session.execute(
                    select(Document).where(Document.sha256 == sha256, Document.status != DocumentStatus.FAILED.value)
                ).scalars().first()
                if duplicate is not None:
                    raise ConflictError(
                        f"This file has already been uploaded as '{duplicate.filename}'.",
                        details={"document_id": duplicate.id},
                    )
                document = Document(
                    filename=display_name, stored_name=stored_name, sha256=sha256, size_bytes=size,
                    title=display_name, status=DocumentStatus.QUEUED.value, overrides=overrides,
                )
                session.add(document)
                session.flush()
                return document_out(document, None)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    # ── Queries ──────────────────────────────────────────────────────────
    def list(self, company_id: int | None = None) -> list[DocumentOut]:
        with self._session_factory() as session:
            query = select(Document, Company.name).outerjoin(Company, Company.id == Document.company_id)
            if company_id is not None:
                query = query.where(Document.company_id == company_id)
            rows = session.execute(query.order_by(Document.uploaded_at.desc())).all()
            return [document_out(doc, name) for doc, name in rows]

    def get(self, document_id: str) -> DocumentOut:
        with self._session_factory() as session:
            row = session.execute(
                select(Document, Company.name).outerjoin(Company, Company.id == Document.company_id)
                .where(Document.id == document_id)
            ).first()
            if row is None:
                raise NotFoundError("Document not found.")
            return document_out(row[0], row[1])

    def chunk(self, chunk_id: str) -> ChunkOut:
        with self._session_factory() as session:
            row = session.execute(
                select(Chunk, Document, Company.name)
                .join(Document, Document.id == Chunk.document_id)
                .outerjoin(Company, Company.id == Document.company_id)
                .where(Chunk.id == chunk_id)
            ).first()
            if row is None:
                raise NotFoundError("Source passage not found. The document may have been deleted.")
            chunk, document, company_name = row
            out = ChunkOut.model_validate(chunk)
            out.document_title, out.company_name = document.title or document.filename, company_name
            return out

    def facts(self, document_id: str) -> list[FactOut]:
        self.get(document_id)
        with self._session_factory() as session:
            facts = session.execute(
                select(FinancialFact).where(FinancialFact.document_id == document_id)
                .order_by(FinancialFact.page, FinancialFact.id)
            ).scalars().all()
            out = []
            for fact in facts:
                item = FactOut.model_validate(fact)
                item.metric_name = metric_name(fact.metric)
                out.append(item)
            return out

    # ── Deletion ─────────────────────────────────────────────────────────
    def delete(self, document_id: str) -> None:
        with session_scope(self._session_factory) as session:
            document = session.get(Document, document_id)
            if document is None:
                raise NotFoundError("Document not found.")
            stored_name, company_id = document.stored_name, document.company_id
            # Remove vectors first: if this fails the document stays intact and can be retried.
            self._vectors.delete_document(document_id)
            session.delete(document)
            session.flush()
            if company_id is not None:
                remaining = session.execute(
                    select(func.count(Document.id)).where(Document.company_id == company_id)
                ).scalar_one()
                if remaining == 0:
                    session.execute(delete(Report).where(Report.company_id == company_id))
                    session.execute(delete(Company).where(Company.id == company_id))
        try:
            safe_join(self.settings.upload_dir, stored_name).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored file", extra={"document_id": document_id})

    def recover_interrupted(self, older_than_minutes: int = 0) -> int:
        """Mark documents left mid-ingestion by a previous process as failed.

        ``older_than_minutes`` limits this to work abandoned at least that long ago,
        for deployments where another instance may still be processing.
        """
        with session_scope(self._session_factory) as session:
            query = select(Document).where(
                Document.status.in_([DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value])
            )
            if older_than_minutes:
                query = query.where(Document.uploaded_at < utcnow() - timedelta(minutes=older_than_minutes))
            stuck = session.execute(query).scalars().all()
            for document in stuck:
                document.status = DocumentStatus.FAILED.value
                document.error = "Processing was interrupted by a restart. Delete and upload the file again."
            return len(stuck)


class CompanyService:
    def __init__(self, session_factory: SessionFactory, financials: FinancialDataService) -> None:
        self._session_factory = session_factory
        self._financials = financials

    def _out(self, session, company: Company) -> CompanyOut:  # noqa: ANN001
        docs = session.execute(
            select(Document.fiscal_year).where(
                Document.company_id == company.id, Document.status == DocumentStatus.READY.value
            )
        ).scalars().all()
        periods = self._financials.periods(company.id)
        currency, unit = self._financials.reporting_basis(periods)
        return CompanyOut(
            id=company.id, name=company.name, document_count=len(docs),
            fiscal_years=sorted({y for y in docs if y}),
            periods_with_data=[p.label for p in periods], currency=currency, unit=unit,
        )

    def list(self) -> list[CompanyOut]:
        with self._session_factory() as session:
            companies = session.execute(select(Company).order_by(Company.name)).scalars().all()
            return [self._out(session, c) for c in companies]

    def get(self, company_id: int) -> CompanyOut:
        with self._session_factory() as session:
            company = session.get(Company, company_id)
            if company is None:
                raise NotFoundError("Company not found.")
            return self._out(session, company)
