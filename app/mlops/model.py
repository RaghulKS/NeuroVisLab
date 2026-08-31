from __future__ import annotations

import torch
from torch import nn


class MedMNISTCNN(nn.Module):
    """Small CNN intentionally sized for reproducible CPU demo training."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(48, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))
