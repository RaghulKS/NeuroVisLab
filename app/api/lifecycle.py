from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.mlops.config import GateConfig
from app.mlops.gates import evaluate_gates
from app.mlops.monitoring import submit_label, summary
from app.mlops.registry import audit_trail, list_models, promote, register_run, rollback, set_shadow

router = APIRouter(prefix="/lifecycle", tags=["mlops-lifecycle"])


class RegisterRequest(BaseModel):
    run_manifest: str
    actor: str = "api"


class PromoteRequest(BaseModel):
    model_id: str
    target: str = Field(pattern="^(staging|production)$")
    traffic_percent: int = Field(default=100, ge=1, le=100)
    actor: str = "api"
    reason: str = "controlled promotion"


class RollbackRequest(BaseModel):
    model_id: str
    actor: str = "api"
    reason: str = "rollback requested"


class ShadowRequest(BaseModel):
    model_id: str | None = None
    actor: str = "api"


class LabelRequest(BaseModel):
    request_id: str
    true_label: str


@router.post("/register")
def register(request: RegisterRequest) -> dict[str, Any]:
    try:
        run = json.loads(Path(request.run_manifest).read_text(encoding="utf-8"))
        run["run_manifest"] = str(Path(request.run_manifest).resolve())
        config_gates = GateConfig(**run["config"].get("gates", {}))
        return {"model": register_run(run, evaluate_gates(run, config_gates), request.actor)}
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models")
def models() -> dict[str, Any]:
    return {"models": list_models()}


@router.get("/audit")
def audit() -> dict[str, Any]:
    return {"events": audit_trail()}


@router.post("/promote")
def promote_endpoint(request: PromoteRequest) -> dict[str, Any]:
    try:
        return {"model": promote(request.model_id, request.target, request.traffic_percent, request.actor, request.reason)}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rollback")
def rollback_endpoint(request: RollbackRequest) -> dict[str, Any]:
    try:
        return {"restored_model": rollback(request.model_id, request.actor, request.reason)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/shadow")
def shadow(request: ShadowRequest) -> dict[str, Any]:
    set_shadow(request.model_id, request.actor)
    return {"shadow_model_id": request.model_id}


@router.post("/feedback")
def feedback(request: LabelRequest) -> dict[str, Any]:
    try:
        return submit_label(request.request_id, request.true_label)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/monitoring")
def monitoring() -> dict[str, Any]:
    return summary()
