from __future__ import annotations

from app.api._compat import APIRouter
from app.schemas import TrainRequest
from app.services.model_training import train_image_model, train_multimodal_model

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/image")
def train_image(request: TrainRequest) -> dict[str, object]:
    return train_image_model(
        dataset_path=request.dataset_path,
        epochs=request.epochs,
        image_size=request.image_size,
        model_name=request.model_name,
    )


@router.post("/multimodal")
def train_multimodal(request: TrainRequest) -> dict[str, object]:
    return train_multimodal_model(
        dataset_path=request.dataset_path,
        epochs=request.epochs,
        image_size=request.image_size,
        model_name=request.model_name,
    )

