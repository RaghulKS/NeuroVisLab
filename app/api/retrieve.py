from __future__ import annotations

import json
from typing import Any

from app.api._compat import APIRouter, File, Form, UploadFile
from app.api._uploads import save_upload
from app.config import settings
from app.services.case_retrieval import retrieve_similar_cases

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("/similar")
async def retrieve_similar_endpoint(
    file: UploadFile = File(...),
    clinical_note: str | None = Form(None),
    metadata_json: str | None = Form(None),
    top_k: int = Form(5),
) -> dict[str, Any]:
    image_path = await save_upload(file, "retrieve")
    metadata = json.loads(metadata_json) if metadata_json else {}
    similar_cases = retrieve_similar_cases(image_path, clinical_note=clinical_note, metadata=metadata, top_k=top_k)
    return {"similar_cases": similar_cases, "disclaimer": settings.disclaimer}

