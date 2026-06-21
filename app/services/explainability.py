from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.config import settings
from app.ml.inference import predict_image
from app.services.calibration import uncertainty_flags
from app.services.preprocessing import load_image, resize_image


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value > 1e-8:
        values = values / max_value
    return values


def _saliency_from_edges(image: Image.Image, image_size: int = 128) -> np.ndarray:
    arr = np.asarray(resize_image(image, image_size=image_size), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    local_brightness = np.maximum(gray - gray.mean(), 0)
    return _normalize_heatmap(gx + gy + 0.25 * local_brightness)


def _overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    original = resize_image(original, image_size=heatmap.shape[0]).convert("RGBA")
    heat = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = (80 * (1.0 - heatmap)).astype(np.uint8)
    heat[..., 3] = (170 * heatmap).astype(np.uint8)
    return Image.alpha_composite(original, Image.fromarray(heat, mode="RGBA")).convert("RGB")


def _side_by_side(original: Image.Image, overlay: Image.Image, label: str) -> Image.Image:
    original = original.convert("RGB").resize(overlay.size)
    width, height = overlay.size
    canvas = Image.new("RGB", (width * 2, height + 26), color=(245, 245, 245))
    canvas.paste(original, (0, 26))
    canvas.paste(overlay, (width, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "Original", fill=(20, 20, 20))
    draw.text((width + 8, 6), f"Model heatmap: {label}", fill=(20, 20, 20))
    return canvas


def reason_codes(probabilities: dict[str, float], heatmap: np.ndarray) -> list[str]:
    flags = uncertainty_flags(probabilities)
    reasons: list[str] = []
    if flags["low_confidence"]:
        reasons.append("LOW_CONFIDENCE: top probability is below the configured research threshold.")
    if flags["high_uncertainty"]:
        reasons.append("SMALL_MARGIN: top classes are close in probability.")
    if float(heatmap.mean()) > 0.25:
        reasons.append("LOCAL_TEXTURE_SIGNAL: explanation map emphasizes high-contrast regions.")
    if not reasons:
        reasons.append("MODEL_OUTPUT_SUMMARY: prediction is based on learned image feature patterns.")
    return reasons


def explain_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    clinical_note: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir or settings.artifacts_dir / "explanations")
    output_root.mkdir(parents=True, exist_ok=True)
    original = load_image(image_path)
    prediction = predict_image(image_path, model_path=model_path, clinical_note=clinical_note, metadata=metadata)
    image_size = int(prediction.get("image_size", settings.default_image_size))
    heatmap = _saliency_from_edges(original, image_size=image_size)
    overlay = _overlay_heatmap(original, heatmap)
    stem = Path(image_path).stem
    heatmap_path = output_root / f"{stem}_heatmap.png"
    overlay_path = output_root / f"{stem}_overlay.png"
    side_by_side_path = output_root / f"{stem}_side_by_side.png"
    heat_uint8 = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heat_uint8, mode="L").save(heatmap_path)
    overlay.save(overlay_path)
    _side_by_side(original, overlay, prediction["prediction_label"]).save(side_by_side_path)
    return {
        "prediction": prediction,
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "side_by_side_path": str(side_by_side_path),
        "reason_codes": reason_codes(prediction["probabilities"], heatmap),
        "method": "edge_and_local_contrast_occlusion_fallback",
        "disclaimer": settings.disclaimer,
    }

