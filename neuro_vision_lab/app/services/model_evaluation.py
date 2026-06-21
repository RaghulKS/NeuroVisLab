from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.ml.metrics import evaluate_predictions, subgroup_performance
from app.services.calibration import calibration_curve_data, confidence_histogram, expected_calibration_error
from app.services.dataset_loader import CaseRecord


def evaluate_model_outputs(
    records: list[CaseRecord],
    predictions: list[str],
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    y_true = [record.label for record in records]
    metrics = evaluate_predictions(y_true, predictions, probabilities=probabilities, labels=labels)
    metrics["calibration"] = {
        "expected_calibration_error": expected_calibration_error(y_true, probabilities, labels),
        "curve": calibration_curve_data(y_true, probabilities, labels),
        "confidence_histogram": confidence_histogram(probabilities),
    }
    if any(record.sex for record in records):
        metrics["subgroup_by_sex"] = subgroup_performance(y_true, predictions, [record.sex for record in records], labels)
    if any(record.age is not None for record in records):
        age_groups = [
            "missing"
            if record.age is None
            else ("under_50" if float(record.age) < 50 else "50_plus")
            for record in records
        ]
        metrics["subgroup_by_age_band"] = subgroup_performance(y_true, predictions, age_groups, labels)
    return metrics


def save_metrics(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return path


def model_comparison_table(artifacts_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(artifacts_dir or settings.artifacts_dir) / "models"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metrics_file in sorted(root.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "model_id": metrics_file.parent.name,
                "model_name": metrics.get("model_name", metrics_file.parent.name),
                "model_type": metrics.get("model_type", "unknown"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
                "pr_auc_ovr_macro": metrics.get("pr_auc_ovr_macro"),
                "expected_calibration_error": (metrics.get("calibration") or {}).get("expected_calibration_error"),
                "n_samples": metrics.get("n_samples"),
                "path": str(metrics_file),
            }
        )
    return rows

