from __future__ import annotations

import numpy as np


def contrastive_embedding_fallback(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    image = np.asarray(image_features, dtype=np.float32)
    text = np.asarray(text_features, dtype=np.float32)
    if image.ndim == 1:
        image = image.reshape(1, -1)
    if text.ndim == 1:
        text = text.reshape(1, -1)
    combined = np.concatenate([image, text], axis=1)
    norm = np.linalg.norm(combined, axis=1, keepdims=True)
    return combined / np.maximum(norm, 1e-8)

