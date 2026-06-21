from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.clinical_assistant import summarize_computed_evidence


def generate_case_report(
    case_id: str | None,
    metadata: dict[str, Any],
    prediction: dict[str, Any],
    heatmap_path: str | None = None,
    similar_cases: list[dict[str, Any]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    report_dir = Path(output_dir or settings.artifacts_dir / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    case_name = case_id or f"case_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    path = report_dir / f"{case_name}.md"
    similar_cases = similar_cases or []
    probabilities = prediction.get("probabilities", {})
    lines = [
        f"# NeuroVisionLab Case Report: {case_name}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Safety Statement",
        settings.disclaimer,
        "",
        "## Case Summary",
        summarize_computed_evidence(
            {
                "prediction": prediction,
                "similar_cases": similar_cases,
                "heatmap_path": heatmap_path,
            }
        ),
        "",
        "## Input Metadata",
    ]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No structured metadata supplied.")
    lines.extend(
        [
            "",
            "## Model Prediction",
            f"- Predicted label: {prediction.get('prediction_label', 'unknown')}",
            f"- Confidence: {float(prediction.get('confidence', 0.0) or 0.0):.3f}",
            f"- Uncertainty: {prediction.get('uncertainty', prediction.get('uncertainty_flags', 'not supplied'))}",
            "",
            "## Class Probabilities",
        ]
    )
    if probabilities:
        for label, value in probabilities.items():
            lines.append(f"- {label}: {float(value):.3f}")
    else:
        lines.append("- No probability vector supplied.")
    lines.extend(["", "## Explainability"])
    lines.append(f"- Heatmap path: {heatmap_path or 'not generated'}")
    lines.extend(["", "## Similar Cases"])
    if similar_cases:
        for case in similar_cases:
            lines.append(
                f"- {case.get('case_id', 'unknown')}: label={case.get('label')}, "
                f"similarity={float(case.get('similarity_score', 0.0)):.3f}"
            )
    else:
        lines.append("- No similar cases supplied.")
    lines.extend(
        [
            "",
            "## Model Limitations",
            "- Demo data and public research datasets may not represent target deployment populations.",
            "- Similarity retrieval is embedding based and does not imply clinical equivalence.",
            "- Heatmaps are model-behavior visualizations, not causal medical explanations.",
            "- This report must not be used for diagnosis, triage, or treatment decisions.",
            "",
            "## Intended Use",
            "Research, education, portfolio demonstration, and engineering review only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report_path": str(path), "markdown": "\n".join(lines), "disclaimer": settings.disclaimer}

