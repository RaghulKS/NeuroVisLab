from __future__ import annotations

from app.api._compat import APIRouter
from app.config import settings
from app.mlops.monitoring import prometheus_metrics
from app.services.model_evaluation import model_comparison_table

try:
    from fastapi.responses import PlainTextResponse
except ImportError:  # pragma: no cover
    PlainTextResponse = None  # type: ignore[assignment]

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", include_in_schema=False)
def prometheus() -> object:
    payload = prometheus_metrics()
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4") if PlainTextResponse else payload


@router.get("/models")
def metrics_models() -> dict[str, object]:
    return {"models": model_comparison_table(), "disclaimer": settings.disclaimer}
