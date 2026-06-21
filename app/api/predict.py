from __future__ import annotations

import json
from typing import Any

from app.api._compat import APIRouter, File, Form, UploadFile
from app.api._uploads import save_upload
from app.config import settings
from app.database import log_case_prediction
from app.ml.inference import latest_model_path, predict_image
from app.services.calibration import uncertainty_flags
from app.services.multimodal_fusion import evidence_summary_fields
from app.services.dataset_loader import CaseRecord

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    clinical_note: str | None = Form(None),
    metadata_json: str | None = Form(None),
    model_type: str = Form("image"),
) -> dict[str, Any]:
    image_path = await save_upload(file, "predict")
    metadata = json.loads(metadata_json) if metadata_json else {}
    model_path = latest_model_path(model_type) or latest_model_path("image")
    prediction = predict_image(image_path, model_path=model_path, clinical_note=clinical_note, metadata=metadata)
    prediction["uncertainty"] = uncertainty_flags(prediction["probabilities"])
    prediction["evidence_summary"] = evidence_summary_fields(
        CaseRecord(
            image_path=str(image_path),
            label=prediction["prediction_label"],
            clinical_note=clinical_note,
            age=metadata.get("age"),
            sex=metadata.get("sex"),
            view_position=metadata.get("view_position"),
        ),
        prediction["probabilities"],
    )
    prediction["disclaimer"] = settings.disclaimer
    log_case_prediction(prediction)
    return prediction

