from __future__ import annotations

from pathlib import Path
from typing import Any

from app.mlops.config import GateConfig


def evaluate_gates(run: dict[str, Any], gates: GateConfig) -> dict[str, Any]:
    metrics = run.get("metrics", {})
    checkpoint = Path(run.get("artifacts", {}).get("checkpoint", ""))
    checks = {
        "dataset_lineage": bool(run.get("data_version") and run.get("dataset_sha256") and run.get("dataset_lock")),
        "checkpoint_integrity": checkpoint.exists() and bool(run.get("artifacts", {}).get("checkpoint_sha256")),
        "minimum_accuracy": float(metrics.get("accuracy", 0.0)) >= gates.min_accuracy,
        "minimum_macro_f1": float(metrics.get("macro_f1", 0.0)) >= gates.min_macro_f1,
        "maximum_ece": float(metrics.get("expected_calibration_error", 1.0)) <= gates.max_ece,
    }
    if gates.require_smoke_prediction:
        checks["prediction_artifact"] = Path(run.get("artifacts", {}).get("predictions", "")).exists()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"min_accuracy": gates.min_accuracy, "min_macro_f1": gates.min_macro_f1, "max_ece": gates.max_ece},
        "notice": "These are engineering release gates, not clinical acceptance criteria.",
    }
