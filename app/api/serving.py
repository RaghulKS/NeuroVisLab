from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.mlops.serving import InferenceService

router = APIRouter(prefix="/infer", tags=["mlops-serving"])
service = InferenceService()


@router.post("")
async def infer(file: Annotated[UploadFile, File(...)], request_id: str | None = None) -> dict[str, Any]:
    try:
        return service.infer(await file.read(), request_id)
    except (LookupError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/batch")
async def batch_infer(files: Annotated[list[UploadFile], File(...)]) -> dict[str, Any]:
    try:
        results = service.batch_infer([await file.read() for file in files])
        return {"count": len(results), "results": results}
    except (LookupError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
