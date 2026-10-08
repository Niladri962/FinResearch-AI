from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ContainerDep, ViewerDep
from app.models.schemas import CompanyOut, DocumentOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=list[CompanyOut])
def list_companies(container: ContainerDep, _principal: ViewerDep) -> list[CompanyOut]:
    return container.companies.list()


@router.get("/{company_id}", response_model=CompanyOut)
def get_company(company_id: int, container: ContainerDep, _principal: ViewerDep) -> CompanyOut:
    return container.companies.get(company_id)


@router.get("/{company_id}/documents", response_model=list[DocumentOut])
def company_documents(company_id: int, container: ContainerDep, _principal: ViewerDep) -> list[DocumentOut]:
    container.companies.get(company_id)
    return container.documents.list(company_id)
