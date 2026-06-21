from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.case_retrieval import build_retrieval_index  # noqa: E402
from app.services.dataset_loader import create_synthetic_demo_dataset  # noqa: E402
from app.services.model_training import train_image_model, train_multimodal_model  # noqa: E402


def main() -> None:
    dataset_dir = settings.data_dir / "demo_medical_images"
    if not dataset_dir.exists():
        create_synthetic_demo_dataset(dataset_dir)
    image_result = train_image_model(dataset_dir, epochs=2, image_size=settings.default_image_size)
    multimodal_result = train_multimodal_model(dataset_dir, epochs=2, image_size=settings.default_image_size)
    retrieval_result = build_retrieval_index(dataset_dir)
    print(
        json.dumps(
            {
                "image_model": image_result["model_path"],
                "multimodal_model": multimodal_result["model_path"],
                "retrieval_index": retrieval_result,
                "image_metrics": image_result["metrics"],
                "multimodal_metrics": multimodal_result["metrics"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

