from __future__ import annotations

from app.ml.inference import predict_image
from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset
from app.services.model_training import train_image_model


def test_train_and_predict_fallback_model(tmp_path, monkeypatch):
    dataset_dir = create_synthetic_demo_dataset(tmp_path / "demo", samples_per_class=5, image_size=64)
    splits = load_any_dataset(dataset_dir)
    result = train_image_model(dataset_dir, image_size=64, model_name="pytest_image_model")
    sample = splits.all_records()[0]
    prediction = predict_image(sample.image_path, model_path=result["model_path"])
    assert prediction["prediction_label"] in splits.classes
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert set(prediction["probabilities"]) == set(splits.classes)

