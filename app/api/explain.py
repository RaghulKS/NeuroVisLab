from __future__ import annotations

import json
from typing import Any

from app.api._compat import APIRouter, File, Form, UploadFile
from app.api._uploads import save_upload
from app.ml.inference import latest_model_path
from app.services.explainability import explain_image

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/image")
async def explain_image_endpoint(
    file: UploadFile = File(...),
    clinical_note: str | None = Form(None),
    metadata_json: str | None = Form(None),
    model_type: str = Form("image"),
) -> dict[str, Any]:
    image_path = await save_upload(file, "explain")
    metadata = json.loads(metadata_json) if metadata_json else {}
    model_path = latest_model_path(model_type) or latest_model_path("image")
    return explain_image(image_path, model_path=model_path, clinical_note=clinical_note, metadata=metadata)

