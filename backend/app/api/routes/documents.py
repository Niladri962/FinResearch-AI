from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Response, UploadFile, status

from app.api.deps import AdminDep, AnalystDep, ContainerDep, ViewerDep
from app.models.schemas import ChunkOut, DocumentOut, ErrorBody, FactOut, UploadResponse, UploadResult
from app.services.documents import clean_overrides
from app.utils.errors import AppError, ValidationAppError

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILES_PER_REQUEST = 10


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
def upload_documents(
    container: ContainerDep,
    _principal: AnalystDep,
    background: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, XLSX, TXT or MD files")],
    company: Annotated[str | None, Form(description="Overrides the detected company name")] = None,
    document_type: Annotated[str | None, Form()] = None,
    fiscal_year: Annotated[int | None, Form()] = None,
    quarter: Annotated[int | None, Form()] = None,
    wait: Annotated[bool, Query(description="Process synchronously and return the final status")] = False,
) -> UploadResponse:
    """Upload one or more financial documents. Processing continues in the background;
    poll ``GET /api/documents/{id}`` until ``status`` is ``ready`` or ``failed``."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise ValidationAppError(f"Upload at most {MAX_FILES_PER_REQUEST} files per request.")
    overrides = clean_overrides(company, document_type, fiscal_year, quarter)

    results: list[UploadResult] = []
    first_error: AppError | None = None
    for upload in files:
        name = upload.filename or "document"
        try:
            document = container.documents.save_upload(name, upload.file, overrides)
        except AppError as exc:
            first_error = first_error or exc
            results.append(UploadResult(filename=name, error=ErrorBody(code=exc.code, message=exc.message, details=exc.details)))
            continue
        finally:
            upload.file.close()
        if wait:
            container.ingestion.run(document.id)
            document = container.documents.get(document.id)
        else:
            background.add_task(container.ingestion.run, document.id)
        results.append(UploadResult(filename=name, document=document))

    accepted = sum(1 for r in results if r.document is not None)
    if accepted == 0 and first_error is not None:
        raise first_error  # nothing was accepted: answer with the precise error status
    return UploadResponse(results=results, accepted=accepted, rejected=len(results) - accepted)


@router.get("", response_model=list[DocumentOut])
def list_documents(container: ContainerDep, _principal: ViewerDep, company_id: int | None = None) -> list[DocumentOut]:
    documents = container.documents.list(company_id)
    for document in documents:
        document.progress = container.ingestion.progress.get(document.id)
    return documents


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str, container: ContainerDep, _principal: ViewerDep) -> ChunkOut:
    """Full text of a cited passage — powers the source preview in the UI."""
    return container.documents.chunk(chunk_id)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, container: ContainerDep, _principal: ViewerDep) -> DocumentOut:
    document = container.documents.get(document_id)
    document.progress = container.ingestion.progress.get(document_id)
    return document


@router.get("/{document_id}/facts", response_model=list[FactOut])
def get_document_facts(document_id: str, container: ContainerDep, _principal: ViewerDep) -> list[FactOut]:
    """Structured financial facts extracted from the document's tables."""
    return container.documents.facts(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, container: ContainerDep, _principal: AdminDep) -> Response:
    container.documents.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
