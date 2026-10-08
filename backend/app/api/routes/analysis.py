from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import ContainerDep, ViewerDep
from app.financial.metrics import METRICS
from app.financial.ratios import RATIOS
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CompareRequest,
    CompareResponse,
    RatioRequest,
    RatioResponse,
)

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, container: ContainerDep, _principal: ViewerDep) -> AnalyzeResponse:
    """KPIs, period snapshots, trend charts and risk signals for a company."""
    return await container.analysis.analyze(request)


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest, container: ContainerDep, _principal: ViewerDep) -> CompareResponse:
    """Two or more ``company_ids`` → company comparison; one → period comparison."""
    return await container.analysis.compare(request)


@router.post("/financial-ratios", response_model=RatioResponse)
def financial_ratios(request: RatioRequest, container: ContainerDep, _principal: ViewerDep) -> RatioResponse:
    """Compute ratios from stored statements (``company_id``) or from raw ``values``."""
    return container.analysis.ratios(request)


@router.get("/financial-ratios/definitions")
def ratio_definitions(_principal: ViewerDep) -> dict[str, list[dict[str, Any]]]:
    """The ratio and metric catalogue the engine understands."""
    return {
        "ratios": [
            {"key": r.key, "name": r.name, "category": r.category, "formula": r.formula, "unit": r.unit,
             "inputs": list(r.inputs), "description": r.description}
            for r in RATIOS
        ],
        "metrics": [{"key": m.key, "name": m.name, "statement": m.statement} for m in METRICS],
    }
