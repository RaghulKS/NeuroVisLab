from __future__ import annotations

import importlib.util

from app.api._compat import APIRouter
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, object]:
    optional_packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ["torch", "torchvision", "sklearn", "faiss", "transformers", "mlflow", "cv2"]
    }
    return {
        "status": "ok",
        "project": settings.project_name,
        "environment": settings.environment,
        "optional_packages": optional_packages,
        "disclaimer": settings.disclaimer,
    }

