from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - lets local import checks run without FastAPI.
    class FastAPI:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.routes: list[Any] = []

        def include_router(self, router: Any) -> None:
            self.routes.append(router)

        def get(self, *_: Any, **__: Any):
            def decorator(func):
                return func

            return decorator

        def add_middleware(self, *_: Any, **__: Any) -> None:
            return None

    class CORSMiddleware:  # noqa: D401
        """Placeholder when FastAPI is not installed."""

from app.api import data, explain, health, metrics, monitoring, predict, reports, retrieve, train
from app.config import settings
from app.database import init_db


app = FastAPI(title=settings.project_name, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    health.router,
    data.router,
    train.router,
    predict.router,
    explain.router,
    retrieve.router,
    reports.router,
    metrics.router,
    monitoring.router,
):
    app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": settings.project_name,
        "message": "Medical AI research demo API. Not for clinical use.",
        "docs": "/docs",
    }


init_db()
