from __future__ import annotations

from typing import Any


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency.
    torch = None
    nn = None


if torch is not None:
    class MultimodalFusionNet(nn.Module):
        def __init__(self, image_dim: int, text_dim: int, metadata_dim: int, num_classes: int):
            super().__init__()
            self.image_projection = nn.Linear(image_dim, 128)
            self.text_projection = nn.Linear(text_dim, 64)
            self.metadata_projection = nn.Linear(metadata_dim, 32)
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128 + 64 + 32, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, image_features, text_features, metadata_features):  # type: ignore[no-untyped-def]
            image_z = self.image_projection(image_features)
            text_z = self.text_projection(text_features)
            metadata_z = self.metadata_projection(metadata_features)
            return self.classifier(torch.cat([image_z, text_z, metadata_z], dim=1))
else:
    class MultimodalFusionNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PyTorch is required for MultimodalFusionNet. Use services.multimodal_fusion fallback.")

