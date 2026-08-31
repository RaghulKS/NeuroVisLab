from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class DataConfig:
    name: str = "pneumoniamnist"
    version: str = "medmnist-v2-28"
    url: str = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
    max_train_samples: int = 512
    max_val_samples: int = 128
    max_test_samples: int = 128


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.001
    num_workers: int = 0
    device: str = "cpu"


@dataclasses.dataclass(frozen=True)
class GateConfig:
    min_accuracy: float = 0.0
    min_macro_f1: float = 0.0
    max_ece: float = 1.0
    require_smoke_prediction: bool = True


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    canary_percent: int = 10
    shadow_enabled: bool = True


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str = "pneumoniamnist-cnn"
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    gates: GateConfig = dataclasses.field(default_factory=GateConfig)
    rollout: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a deliberately small TOML experiment configuration."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        name=str(raw.get("name", "pneumoniamnist-cnn")),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        gates=GateConfig(**raw.get("gates", {})),
        rollout=RolloutConfig(**raw.get("rollout", {})),
    )
