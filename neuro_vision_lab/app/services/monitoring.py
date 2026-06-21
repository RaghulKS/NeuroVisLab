from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.services.dataset_loader import CaseRecord, class_counts, create_synthetic_demo_dataset, load_any_dataset
from app.services.model_evaluation import model_comparison_table
from app.services.preprocessing import compute_image_features, load_image


def _image_profile(records: list[CaseRecord], image_size: int = 128) -> dict[str, Any]:
    if not records:
        return {
            "n_cases": 0,
            "class_distribution": {},
            "brightness_mean": 0.0,
            "brightness_std": 0.0,
            "width_mean": 0.0,
            "height_mean": 0.0,
            "missing_metadata_rate": 1.0,
        }
    brightness: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    missing_fields = 0
    total_fields = len(records) * 3
    embedding_norms: list[float] = []
    for record in records:
        image = load_image(record.image_path)
        widths.append(image.width)
        heights.append(image.height)
        arr_features = compute_image_features(record.image_path, image_size=image_size)
        brightness.append(float(arr_features[22]))
        embedding_norms.append(float(np.linalg.norm(arr_features)))
        missing_fields += int(record.age is None) + int(record.sex is None) + int(record.view_position is None)
    return {
        "n_cases": len(records),
        "class_distribution": class_counts(records),
        "brightness_mean": float(np.mean(brightness)),
        "brightness_std": float(np.std(brightness)),
        "width_mean": float(np.mean(widths)),
        "height_mean": float(np.mean(heights)),
        "embedding_norm_mean": float(np.mean(embedding_norms)),
        "embedding_norm_std": float(np.std(embedding_norms)),
        "missing_metadata_rate": float(missing_fields / max(total_fields, 1)),
    }


def _relative_change(current: float, reference: float) -> float:
    return float(abs(current - reference) / max(abs(reference), 1e-6))


def compare_profiles(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    ref_classes = reference.get("class_distribution", {})
    cur_classes = current.get("class_distribution", {})
    all_labels = sorted(set(ref_classes) | set(cur_classes))
    ref_total = max(sum(ref_classes.values()), 1)
    cur_total = max(sum(cur_classes.values()), 1)
    class_shift = {
        label: abs((cur_classes.get(label, 0) / cur_total) - (ref_classes.get(label, 0) / ref_total))
        for label in all_labels
    }
    checks = {
        "image_brightness_distribution_shift": _relative_change(current.get("brightness_mean", 0.0), reference.get("brightness_mean", 0.0)),
        "image_size_distribution_shift": max(
            _relative_change(current.get("width_mean", 0.0), reference.get("width_mean", 0.0)),
            _relative_change(current.get("height_mean", 0.0), reference.get("height_mean", 0.0)),
        ),
        "class_distribution_shift": max(class_shift.values()) if class_shift else 0.0,
        "embedding_distance_shift": _relative_change(current.get("embedding_norm_mean", 0.0), reference.get("embedding_norm_mean", 0.0)),
        "missing_metadata_rate_shift": abs(current.get("missing_metadata_rate", 0.0) - reference.get("missing_metadata_rate", 0.0)),
    }
    return {
        "checks": checks,
        "class_shift_by_label": class_shift,
        "overall_status": "review" if any(value > 0.20 for value in checks.values()) else "within_demo_thresholds",
        "threshold": 0.20,
        "disclaimer": settings.disclaimer,
    }


def drift_report(
    reference_dataset_path: str | Path | None = None,
    current_dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    reference_splits = load_any_dataset(reference_dataset_path or settings.data_dir)
    if not reference_splits.all_records():
        demo_dir = create_synthetic_demo_dataset()
        reference_splits = load_any_dataset(demo_dir)
    current_splits = load_any_dataset(current_dataset_path or reference_dataset_path or settings.data_dir)
    if not current_splits.all_records():
        current_splits = reference_splits
    reference = _image_profile(reference_splits.all_records())
    current = _image_profile(current_splits.all_records())
    comparison = compare_profiles(reference, current)
    return {
        "reference_profile": reference,
        "current_profile": current,
        "drift": comparison,
        "models": model_comparison_table(),
    }


def load_registry() -> list[dict[str, Any]]:
    registry_path = settings.artifacts_dir / "model_registry.json"
    if not registry_path.exists():
        return []
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
