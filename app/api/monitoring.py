from __future__ import annotations

from app.api._compat import APIRouter
from app.config import settings
from app.services.model_evaluation import model_comparison_table
from app.services.monitoring import drift_report, load_registry

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/drift")
def get_drift(reference_dataset_path: str | None = None, current_dataset_path: str | None = None) -> dict[str, object]:
    return drift_report(reference_dataset_path=reference_dataset_path, current_dataset_path=current_dataset_path)


@router.get("/registry")
def get_registry() -> dict[str, object]:
    return {"registry": load_registry(), "disclaimer": settings.disclaimer}


@router.get("/models")
def get_models() -> dict[str, object]:
    return {"models": model_comparison_table(), "disclaimer": settings.disclaimer}

