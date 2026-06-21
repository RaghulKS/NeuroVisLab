from __future__ import annotations

from typing import Any

import numpy as np

from app.ml.inference import NumpyFeatureClassifier


def train_numpy_centroid_classifier(
    features: np.ndarray,
    labels: list[str],
    metadata: dict[str, Any] | None = None,
    text_vocabulary: list[str] | None = None,
) -> NumpyFeatureClassifier:
    return NumpyFeatureClassifier.fit(
        features=features,
        labels=labels,
        metadata=metadata,
        text_vocabulary=text_vocabulary,
    )


def train_torch_model_if_available(*_: Any, **__: Any) -> dict[str, Any]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return {
            "status": "skipped",
            "reason": "PyTorch is not installed. The NumPy centroid fallback was used.",
        }
    return {
        "status": "available",
        "reason": "Torch training hooks are available through app.ml.cnn_models.SimpleCNN.",
    }

