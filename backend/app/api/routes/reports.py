from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import AnalystDep, ContainerDep, ViewerDep
from app.models.schemas import ReportDetail, ReportOut, ReportRequest

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/generate", response_model=ReportDetail, status_code=status.HTTP_201_CREATED)
async def generate_report(request: ReportRequest, container: ContainerDep, _principal: AnalystDep) -> ReportDetail:
    """Generate a full research report (Markdown) with traceable sources."""
    return await container.reports.generate(request.company_id, request.title)


@router.get("", response_model=list[ReportOut])
def list_reports(container: ContainerDep, _principal: ViewerDep) -> list[ReportOut]:
    return container.reports.list()


@router.get("/{report_id}", response_model=ReportDetail)
def get_report(report_id: str, container: ContainerDep, _principal: ViewerDep) -> ReportDetail:
    return container.reports.get(report_id)


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_report(report_id: str, container: ContainerDep, _principal: AnalystDep) -> Response:
    container.reports.delete(report_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
