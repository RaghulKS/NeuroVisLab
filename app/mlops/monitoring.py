from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.ml.metrics import evaluate_predictions
from app.mlops.registry import _db, get_model, init_registry


def _entropy(probabilities: list[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(-np.sum(values * np.log(np.maximum(values, 1e-12))))


def record_observation(
    request_id: str, model_id: str, prediction: str, probabilities: list[float], brightness: float,
    latency_ms: float, shadow_model_id: str | None = None, shadow_prediction: str | None = None,
) -> None:
    init_registry()
    with _db() as conn:
        conn.execute("""INSERT OR REPLACE INTO observations
        (request_id, model_id, prediction, confidence, entropy, brightness, latency_ms, shadow_model_id, shadow_prediction, created_at, true_label, labeled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT true_label FROM observations WHERE request_id=?), NULL), COALESCE((SELECT labeled_at FROM observations WHERE request_id=?), NULL))""", (
            request_id, model_id, prediction, max(probabilities), _entropy(probabilities), brightness, latency_ms,
            shadow_model_id, shadow_prediction, datetime.now(timezone.utc).isoformat(), request_id, request_id,
        ))


def submit_label(request_id: str, true_label: str) -> dict[str, Any]:
    init_registry()
    with _db() as conn:
        result = conn.execute("UPDATE observations SET true_label = ?, labeled_at = ? WHERE request_id = ?", (true_label, datetime.now(timezone.utc).isoformat(), request_id))
    if not result.rowcount:
        raise KeyError(f"Unknown inference request: {request_id}")
    return {"request_id": request_id, "true_label": true_label, "recorded": True}


def _psi(values: list[float], reference_mean: float = 0.5, reference_std: float = 0.2) -> float:
    if not values:
        return 0.0
    edges = np.linspace(0.0, 1.0, 11)
    observed, _ = np.histogram(np.clip(values, 0, 1), bins=edges)
    samples = np.clip(np.random.default_rng(7).normal(reference_mean, max(reference_std, .01), 10_000), 0, 1)
    expected, _ = np.histogram(samples, bins=edges)
    observed_ratio = np.maximum(observed / max(observed.sum(), 1), 1e-6)
    expected_ratio = np.maximum(expected / expected.sum(), 1e-6)
    return float(np.sum((observed_ratio - expected_ratio) * np.log(observed_ratio / expected_ratio)))


def _js_divergence(counts: Counter[str], baseline: dict[str, float] | None = None) -> float:
    if not counts:
        return 0.0
    # A uniform baseline is used only when a run has no persisted baseline.
    labels = sorted(set(counts) | set(baseline or {}))
    p = np.asarray([counts[label] for label in labels], dtype=float)
    p /= p.sum()
    q = np.asarray([(baseline or {}).get(label, 1.0) for label in labels], dtype=float)
    q /= q.sum()
    midpoint = (p + q) / 2
    return float((np.sum(p * np.log(np.maximum(p / midpoint, 1e-12))) + np.sum(q * np.log(np.maximum(q / midpoint, 1e-12)))) / 2)


def summary(limit: int = 1000) -> dict[str, Any]:
    init_registry()
    with _db() as conn:
        rows = conn.execute("SELECT * FROM observations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    observations = [dict(row) for row in rows]
    predictions = Counter(row["prediction"] for row in observations)
    brightness = [float(row["brightness"]) for row in observations]
    confidences = [float(row["confidence"]) for row in observations]
    latencies = [float(row["latency_ms"]) for row in observations]
    disagreements = [row for row in observations if row["shadow_prediction"] and row["shadow_prediction"] != row["prediction"]]
    labeled = [row for row in observations if row["true_label"]]
    baseline: dict[str, Any] = {}
    if observations:
        try:
            model = get_model(str(observations[0]["model_id"]))
            with open(model["run_manifest_path"], encoding="utf-8") as handle:
                baseline = json.load(handle).get("monitoring_baseline", {})
        except (KeyError, OSError, json.JSONDecodeError):
            baseline = {}
    performance: dict[str, Any] | None = None
    if labeled:
        labels = sorted({str(row["prediction"]) for row in labeled} | {str(row["true_label"]) for row in labeled})
        performance = evaluate_predictions([str(row["true_label"]) for row in labeled], [str(row["prediction"]) for row in labeled], labels=labels)
    input_psi = _psi(brightness, float(baseline.get("brightness_mean", .5)), float(baseline.get("brightness_std", .2)))
    prediction_jsd = _js_divergence(predictions, baseline.get("prediction_distribution"))
    return {
        "window_size": len(observations),
        "data_drift": {"brightness_psi": input_psi, "status": "review" if input_psi > .2 else "within_demo_threshold", "baseline_available": bool(baseline)},
        "prediction_drift": {"jensen_shannon_divergence": prediction_jsd, "status": "review" if prediction_jsd > .1 else "within_demo_threshold", "distribution": dict(predictions), "baseline_available": bool(baseline)},
        "uncertainty": {"mean_confidence": float(np.mean(confidences)) if confidences else None, "review_rate": float(sum(value < .55 for value in confidences) / len(confidences)) if confidences else 0.0},
        "latency": {"p50_ms": float(np.percentile(latencies, 50)) if latencies else None, "p95_ms": float(np.percentile(latencies, 95)) if latencies else None},
        "shadow": {"enabled_observations": sum(bool(row["shadow_model_id"]) for row in observations), "disagreement_rate": len(disagreements) / len(observations) if observations else 0.0},
        "performance_when_labels_available": performance,
        "notice": "Drift thresholds are engineering alerts. They are not clinical safety thresholds.",
    }


def prometheus_metrics() -> str:
    report = summary()
    rows = [
        "# HELP neurovisionlab_inference_observations_total Online inference observations.",
        "# TYPE neurovisionlab_inference_observations_total counter",
        f"neurovisionlab_inference_observations_total {report['window_size']}",
        "# HELP neurovisionlab_data_drift_psi Brightness population stability index.",
        "# TYPE neurovisionlab_data_drift_psi gauge",
        f"neurovisionlab_data_drift_psi {report['data_drift']['brightness_psi']}",
        "# HELP neurovisionlab_prediction_drift_jsd Prediction distribution Jensen-Shannon divergence.",
        "# TYPE neurovisionlab_prediction_drift_jsd gauge",
        f"neurovisionlab_prediction_drift_jsd {report['prediction_drift']['jensen_shannon_divergence']}",
        "# HELP neurovisionlab_shadow_disagreement_ratio Shadow versus served prediction disagreement.",
        "# TYPE neurovisionlab_shadow_disagreement_ratio gauge",
        f"neurovisionlab_shadow_disagreement_ratio {report['shadow']['disagreement_rate']}",
    ]
    if report["latency"]["p95_ms"] is not None:
        rows.append(f"neurovisionlab_inference_latency_p95_ms {report['latency']['p95_ms']}")
    return "\n".join(rows) + "\n"
