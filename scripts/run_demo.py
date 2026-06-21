from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ml.inference import predict_image  # noqa: E402
from app.services.calibration import uncertainty_flags  # noqa: E402
from app.services.case_retrieval import build_retrieval_index, retrieve_similar_cases  # noqa: E402
from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset  # noqa: E402
from app.services.explainability import explain_image  # noqa: E402
from app.services.model_training import train_image_model, train_multimodal_model  # noqa: E402
from app.services.report_generator import generate_case_report  # noqa: E402


def main() -> None:
    dataset_dir = settings.data_dir / "demo_medical_images"
    create_synthetic_demo_dataset(dataset_dir)
    splits = load_any_dataset(dataset_dir)
    image_result = train_image_model(dataset_dir)
    train_multimodal_model(dataset_dir)
    build_retrieval_index(dataset_dir)
    sample = (splits.test or splits.validation or splits.train)[0]
    prediction = predict_image(sample.image_path, model_path=image_result["model_path"], clinical_note=sample.clinical_note)
    prediction["uncertainty"] = uncertainty_flags(prediction["probabilities"])
    explanation = explain_image(sample.image_path, model_path=image_result["model_path"], clinical_note=sample.clinical_note)
    similar = retrieve_similar_cases(sample.image_path, clinical_note=sample.clinical_note, top_k=3)
    report = generate_case_report(
        case_id=sample.patient_id,
        metadata=sample.to_dict(),
        prediction=prediction,
        heatmap_path=explanation["heatmap_path"],
        similar_cases=similar,
    )
    print(
        json.dumps(
            {
                "sample_image": sample.image_path,
                "prediction": prediction,
                "explanation": explanation,
                "similar_cases": similar,
                "report_path": report["report_path"],
                "disclaimer": settings.disclaimer,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

