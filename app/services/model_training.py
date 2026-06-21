from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.ml.training_loop import train_numpy_centroid_classifier
from app.models import ModelRegistryEntry
from app.services.dataset_loader import DatasetSplits, CaseRecord, create_synthetic_demo_dataset, load_any_dataset
from app.services.model_evaluation import evaluate_model_outputs, save_metrics
from app.services.multimodal_fusion import build_multimodal_features
from app.services.preprocessing import feature_matrix


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _model_dir(model_type: str, model_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in model_name)
    path = settings.artifacts_dir / "models" / f"{model_type}_{safe_name}_{_timestamp()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_dataset(dataset_path: str | Path | None = None) -> DatasetSplits:
    try:
        splits = load_any_dataset(dataset_path or settings.data_dir)
        if splits.all_records():
            return splits
    except FileNotFoundError:
        pass
    demo_dir = create_synthetic_demo_dataset()
    return load_any_dataset(demo_dir)


def _eval_records(splits: DatasetSplits) -> list[CaseRecord]:
    return splits.test or splits.validation or splits.train


def _write_predictions(
    records: list[CaseRecord],
    predictions: list[str],
    probabilities: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> Path:
    path = output_dir / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image_path", "patient_id", "label", "prediction", "confidence", *[f"prob_{label}" for label in labels]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, pred, probs in zip(records, predictions, probabilities):
            row: dict[str, Any] = {
                "image_path": record.image_path,
                "patient_id": record.patient_id,
                "label": record.label,
                "prediction": pred,
                "confidence": float(np.max(probs)),
            }
            row.update({f"prob_{label}": float(probs[i]) for i, label in enumerate(labels)})
            writer.writerow(row)
    return path


def _write_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "confusion_matrix.csv"
    labels = metrics.get("labels", [])
    matrix = metrics.get("confusion_matrix", [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\prediction", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
    return path


def _write_dataset_snapshot(splits: DatasetSplits, output_dir: Path) -> Path:
    path = output_dir / "dataset_snapshot.json"
    path.write_text(json.dumps(splits.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def _register_model(entry: ModelRegistryEntry, output_dir: Path) -> Path:
    registry_path = settings.artifacts_dir / "model_registry.json"
    existing: list[dict[str, Any]] = []
    if registry_path.exists():
        try:
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    payload = entry.to_dict()
    payload["artifact_dir"] = str(output_dir)
    existing.append(payload)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    (output_dir / "model_card.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return registry_path


def _finalize_training(
    model,
    eval_records: list[CaseRecord],
    eval_features: np.ndarray,
    splits: DatasetSplits,
    output_dir: Path,
    model_type: str,
    model_name: str,
) -> dict[str, Any]:
    probabilities = model.predict_proba(eval_features)
    predictions = model.predict(eval_features)
    metrics = evaluate_model_outputs(eval_records, predictions, probabilities, model.classes)
    metrics.update(
        {
            "model_name": model_name,
            "model_type": model_type,
            "model_version": model.metadata.get("model_version", "v0"),
            "classes": model.classes,
            "train_size": len(splits.train),
            "validation_size": len(splits.validation),
            "test_size": len(splits.test),
            "disclaimer": settings.disclaimer,
        }
    )
    model_path = model.save(output_dir / "model.npz")
    metrics_path = save_metrics(metrics, output_dir)
    predictions_path = _write_predictions(eval_records, predictions, probabilities, model.classes, output_dir)
    confusion_path = _write_confusion_matrix(metrics, output_dir)
    dataset_snapshot = _write_dataset_snapshot(splits, output_dir)
    entry = ModelRegistryEntry(
        model_name=model_name,
        model_version=model.metadata.get("model_version", "v0"),
        model_type=model_type,
        dataset_size=len(splits.all_records()),
        classes=model.classes,
        metrics={
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "expected_calibration_error": metrics.get("calibration", {}).get("expected_calibration_error"),
        },
        thresholds={
            "low_confidence": 0.55,
            "high_uncertainty_margin": 0.20,
            "ood_distance": 0.65,
        },
        known_limitations=[
            "Synthetic demo data is not clinically representative.",
            "Fallback classifier uses engineered image/text features, not diagnostic image understanding.",
            "Model outputs require expert review and are not for clinical use.",
        ],
    )
    registry_path = _register_model(entry, output_dir)
    return {
        "model_path": str(model_path),
        "artifact_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "confusion_matrix_path": str(confusion_path),
        "dataset_snapshot_path": str(dataset_snapshot),
        "registry_path": str(registry_path),
        "metrics": metrics,
        "disclaimer": settings.disclaimer,
    }


def train_image_model(
    dataset_path: str | Path | None = None,
    epochs: int = 2,
    image_size: int = 128,
    model_name: str = "numpy_centroid_image",
) -> dict[str, Any]:
    splits = _ensure_dataset(dataset_path)
    if not splits.train:
        raise ValueError("No training records found.")
    output_dir = _model_dir("image", model_name)
    train_features = feature_matrix(splits.train, image_size=image_size)
    train_labels = [record.label for record in splits.train]
    model = train_numpy_centroid_classifier(
        train_features,
        train_labels,
        metadata={
            "model_name": model_name,
            "model_version": output_dir.name,
            "model_type": "image",
            "image_size": image_size,
            "epochs_requested": epochs,
            "training_backend": "numpy_centroid_fallback",
        },
    )
    eval_records = _eval_records(splits)
    eval_features = feature_matrix(eval_records, image_size=image_size)
    return _finalize_training(model, eval_records, eval_features, splits, output_dir, "image", model_name)


def train_multimodal_model(
    dataset_path: str | Path | None = None,
    epochs: int = 2,
    image_size: int = 128,
    model_name: str = "numpy_centroid_multimodal",
) -> dict[str, Any]:
    splits = _ensure_dataset(dataset_path)
    if not splits.train:
        raise ValueError("No training records found.")
    output_dir = _model_dir("multimodal", model_name)
    train_features, text_encoder = build_multimodal_features(splits.train, image_size=image_size, fit_text=True)
    train_labels = [record.label for record in splits.train]
    model = train_numpy_centroid_classifier(
        train_features,
        train_labels,
        metadata={
            "model_name": model_name,
            "model_version": output_dir.name,
            "model_type": "multimodal",
            "image_size": image_size,
            "text_encoder": "simple_bow_fallback",
            "epochs_requested": epochs,
            "training_backend": "numpy_centroid_fallback",
        },
        text_vocabulary=text_encoder.vocabulary,
    )
    eval_records = _eval_records(splits)
    eval_features, _ = build_multimodal_features(
        eval_records,
        image_size=image_size,
        text_encoder=text_encoder,
        fit_text=False,
    )
    return _finalize_training(model, eval_records, eval_features, splits, output_dir, "multimodal", model_name)
