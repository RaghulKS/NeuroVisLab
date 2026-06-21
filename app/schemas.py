from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    patient_id: str | None = None
    age: float | None = None
    sex: str | None = None
    view_position: str | None = None
    clinical_note: str | None = None
    dataset_source: str | None = None


class PrepareDataRequest(BaseModel):
    dataset_path: str | None = None
    csv_path: str | None = None
    image_root: str | None = None
    create_demo_if_missing: bool = True


class TrainRequest(BaseModel):
    dataset_path: str | None = None
    epochs: int = Field(default=2, ge=1, le=100)
    image_size: int = Field(default=128, ge=32, le=512)
    model_name: str = "numpy_centroid_baseline"


class PredictionResponse(BaseModel):
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]
    uncertainty: dict[str, Any]
    embedding: list[float] | None = None
    disclaimer: str


class MetricsResponse(BaseModel):
    models: list[dict[str, Any]]


class SimilarCase(BaseModel):
    case_id: str
    label: str | None = None
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None


class CaseReportRequest(BaseModel):
    case_id: str | None = None
    prediction: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    heatmap_path: str | None = None
    similar_cases: list[dict[str, Any]] = Field(default_factory=list)

