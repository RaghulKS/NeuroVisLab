from __future__ import annotations

from typing import Any


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency.
    torch = None
    nn = None


if torch is not None:
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x)
            return self.classifier(x.flatten(1))
else:
    class SimpleCNN:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PyTorch is required for SimpleCNN. Use the NumPy fallback training pipeline locally.")


def build_torchvision_model(model_name: str, num_classes: int, pretrained: bool = False):
    if torch is None:
        raise ImportError("PyTorch and torchvision are required for torchvision models.")
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError("torchvision is required for ResNet/EfficientNet models.") from exc
    name = model_name.lower()
    if name in {"resnet18", "resnet"}:
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"efficientnet_b0", "efficientnet"}:
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torchvision model: {model_name}")

