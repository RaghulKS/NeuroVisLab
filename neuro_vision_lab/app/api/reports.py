from __future__ import annotations

from app.api._compat import APIRouter
from app.schemas import CaseReportRequest
from app.services.report_generator import generate_case_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/case")
def create_case_report(request: CaseReportRequest) -> dict[str, object]:
    return generate_case_report(
        case_id=request.case_id,
        metadata=request.metadata,
        prediction=request.prediction,
        heatmap_path=request.heatmap_path,
        similar_cases=request.similar_cases,
    )

