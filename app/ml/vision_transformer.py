from __future__ import annotations

from typing import Any


def build_vision_transformer(num_classes: int, pretrained: bool = False) -> Any:
    try:
        import torch.nn as nn
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for the Vision Transformer. "
            "Use the NumPy image model fallback when torchvision is unavailable."
        ) from exc
    model = models.vit_b_16(weights="DEFAULT" if pretrained else None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

