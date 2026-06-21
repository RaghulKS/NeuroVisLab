from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelRegistryEntry:
    model_name: str
    model_version: str
    model_type: str
    training_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_size: int = 0
    classes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    approval_status: str = "research_demo_only"
    intended_use: str = "Educational medical AI engineering demonstration"
    not_for_clinical_use: str = (
        "Outputs are not medical advice, not a diagnosis, and not for clinical use."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "training_date": self.training_date,
            "dataset_size": self.dataset_size,
            "classes": self.classes,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "known_limitations": self.known_limitations,
            "approval_status": self.approval_status,
            "intended_use": self.intended_use,
            "not_for_clinical_use": self.not_for_clinical_use,
        }

