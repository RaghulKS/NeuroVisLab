from __future__ import annotations

from typing import Any

from app.config import settings


def summarize_computed_evidence(payload: dict[str, Any]) -> str:
    prediction = payload.get("prediction", {})
    label = prediction.get("prediction_label", "unknown")
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    similar_cases = payload.get("similar_cases", [])
    heatmap = payload.get("heatmap_path")
    parts = [
        f"The model output label is {label} with confidence {confidence:.2f}.",
        f"{len(similar_cases)} similar reference cases were returned by the embedding index.",
    ]
    if heatmap:
        parts.append("A heatmap artifact was generated for model-behavior review.")
    parts.append(settings.disclaimer)
    return " ".join(parts)

