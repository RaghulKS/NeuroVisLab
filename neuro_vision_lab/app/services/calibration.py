from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(
    y_true: list[str],
    probabilities: np.ndarray,
    labels: list[str],
    n_bins: int = 10,
) -> float:
    if len(y_true) == 0:
        return 0.0
    probs = np.asarray(probabilities, dtype=np.float32)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    true_indices = np.asarray([labels.index(y) if y in labels else -1 for y in y_true])
    correct = (preds == true_indices).astype(np.float32)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lower = float(edges[i])
        upper = float(edges[i + 1])
        mask = (confidences >= lower) & (confidences < upper if i < n_bins - 1 else confidences <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def calibration_curve_data(
    y_true: list[str],
    probabilities: np.ndarray,
    labels: list[str],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    if len(y_true) == 0:
        return []
    probs = np.asarray(probabilities, dtype=np.float32)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    true_indices = np.asarray([labels.index(y) if y in labels else -1 for y in y_true])
    correct = (preds == true_indices).astype(np.float32)
    rows: list[dict[str, float]] = []
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lower = float(edges[i])
        upper = float(edges[i + 1])
        mask = (confidences >= lower) & (confidences < upper if i < n_bins - 1 else confidences <= upper)
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": float(mask.sum()),
                "accuracy": float(correct[mask].mean()) if np.any(mask) else 0.0,
                "mean_confidence": float(confidences[mask].mean()) if np.any(mask) else 0.0,
            }
        )
    return rows


def confidence_histogram(probabilities: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    if len(probabilities) == 0:
        return []
    confidences = np.max(np.asarray(probabilities, dtype=np.float32), axis=1)
    hist, edges = np.histogram(confidences, bins=n_bins, range=(0.0, 1.0))
    return [
        {"bin_lower": float(edges[i]), "bin_upper": float(edges[i + 1]), "count": float(hist[i])}
        for i in range(n_bins)
    ]


def uncertainty_flags(
    probabilities: dict[str, float] | np.ndarray,
    ood_distance: float | None = None,
    low_confidence_threshold: float = 0.55,
    high_uncertainty_threshold: float = 0.20,
    ood_distance_threshold: float = 0.65,
) -> dict[str, Any]:
    if isinstance(probabilities, dict):
        probs = np.asarray(list(probabilities.values()), dtype=np.float32)
    else:
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    confidence = float(probs.max()) if len(probs) else 0.0
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-8)))) if len(probs) else 0.0
    flags = {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "low_confidence": confidence < low_confidence_threshold,
        "high_uncertainty": margin < high_uncertainty_threshold,
        "ood_warning": bool(ood_distance is not None and ood_distance > ood_distance_threshold),
        "ood_distance": ood_distance,
    }
    flags["requires_review"] = bool(flags["low_confidence"] or flags["high_uncertainty"] or flags["ood_warning"])
    return flags


def temperature_scale_probabilities(probabilities: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    logits = np.log(np.maximum(probs, 1e-8)) / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
