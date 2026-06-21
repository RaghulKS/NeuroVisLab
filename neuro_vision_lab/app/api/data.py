from __future__ import annotations

from app.api._compat import APIRouter
from app.config import settings
from app.schemas import PrepareDataRequest
from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/prepare")
def prepare_data(request: PrepareDataRequest) -> dict[str, object]:
    if request.create_demo_if_missing:
        try:
            splits = load_any_dataset(request.dataset_path or settings.data_dir, csv_path=request.csv_path, image_root=request.image_root)
            if not splits.all_records():
                raise FileNotFoundError("Dataset exists but contains no supported images.")
        except FileNotFoundError:
            demo_dir = create_synthetic_demo_dataset()
            splits = load_any_dataset(demo_dir)
            return {
                "dataset_path": str(demo_dir),
                "created_demo_dataset": True,
                "n_train": len(splits.train),
                "n_validation": len(splits.validation),
                "n_test": len(splits.test),
                "classes": splits.classes,
                "disclaimer": settings.disclaimer,
            }
    else:
        splits = load_any_dataset(request.dataset_path or settings.data_dir, csv_path=request.csv_path, image_root=request.image_root)
    return {
        "dataset_path": str(request.dataset_path or settings.data_dir),
        "created_demo_dataset": False,
        "n_train": len(splits.train),
        "n_validation": len(splits.validation),
        "n_test": len(splits.test),
        "classes": splits.classes,
        "disclaimer": settings.disclaimer,
    }
