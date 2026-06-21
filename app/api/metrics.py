from __future__ import annotations

from app.api._compat import APIRouter
from app.config import settings
from app.services.model_evaluation import model_comparison_table

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/models")
def metrics_models() -> dict[str, object]:
    return {"models": model_comparison_table(), "disclaimer": settings.disclaimer}

