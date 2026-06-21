from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.services.dataset_loader import CaseRecord
from app.services.preprocessing import compute_image_features, metadata_feature_vector


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


@dataclass
class NumpyFeatureClassifier:
    classes: list[str]
    centroids: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    priors: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    text_vocabulary: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: list[str],
        metadata: dict[str, Any] | None = None,
        text_vocabulary: list[str] | None = None,
    ) -> "NumpyFeatureClassifier":
        if len(labels) == 0:
            raise ValueError("Cannot train classifier with zero labels.")
        features = np.asarray(features, dtype=np.float32)
        classes = sorted(set(labels))
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        scaled = (features - feature_mean) / feature_std
        centroids = []
        priors = []
        for label in classes:
            idx = [i for i, y in enumerate(labels) if y == label]
            priors.append(len(idx) / len(labels))
            centroids.append(scaled[idx].mean(axis=0))
        return cls(
            classes=classes,
            centroids=np.asarray(centroids, dtype=np.float32),
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            priors=np.asarray(priors, dtype=np.float32),
            metadata=metadata or {},
            text_vocabulary=text_vocabulary or [],
        )

    def embed(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        scaled = (features - self.feature_mean) / self.feature_std
        return normalize_rows(scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        scaled = (features - self.feature_mean) / self.feature_std
        distances = ((scaled[:, None, :] - self.centroids[None, :, :]) ** 2).mean(axis=2)
        logits = -distances + np.log(np.maximum(self.priors, 1e-6))[None, :]
        return softmax(logits)

    def predict(self, features: np.ndarray) -> list[str]:
        probabilities = self.predict_proba(features)
        indices = np.argmax(probabilities, axis=1)
        return [self.classes[int(i)] for i in indices]

    def save(self, path: str | Path) -> Path:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_path,
            classes=np.asarray(self.classes),
            centroids=self.centroids,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            priors=self.priors,
            metadata_json=json.dumps(self.metadata, default=str),
            text_vocabulary=np.asarray(self.text_vocabulary),
        )
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyFeatureClassifier":
        data = np.load(path, allow_pickle=True)
        metadata_json = str(data.get("metadata_json", "{}"))
        return cls(
            classes=[str(x) for x in data["classes"].tolist()],
            centroids=np.asarray(data["centroids"], dtype=np.float32),
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(data["feature_std"], dtype=np.float32),
            priors=np.asarray(data["priors"], dtype=np.float32),
            metadata=json.loads(metadata_json),
            text_vocabulary=[str(x) for x in data.get("text_vocabulary", np.asarray([])).tolist()],
        )


def latest_model_path(model_type: str = "image") -> Path | None:
    pattern = f"*{model_type}*/model.npz"
    candidates = sorted(settings.artifacts_dir.glob(f"models/{pattern}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _text_features(text: str | None, vocabulary: list[str]) -> np.ndarray:
    if not vocabulary:
        return np.empty((0,), dtype=np.float32)
    tokens = set((text or "").lower().replace("_", " ").split())
    return np.asarray([1.0 if token in tokens else 0.0 for token in vocabulary], dtype=np.float32)


def build_features_for_model(
    model: NumpyFeatureClassifier,
    image_path: str | Path,
    clinical_note: str | None = None,
    metadata: dict[str, Any] | None = None,
    image_size: int | None = None,
) -> np.ndarray:
    model_metadata = model.metadata or {}
    size = int(image_size or model_metadata.get("image_size", settings.default_image_size))
    image_features = compute_image_features(image_path, image_size=size)
    if model_metadata.get("model_type") == "multimodal":
        record = CaseRecord(
            image_path=str(image_path),
            label="unknown",
            clinical_note=clinical_note,
            age=(metadata or {}).get("age"),
            sex=(metadata or {}).get("sex"),
            view_position=(metadata or {}).get("view_position"),
        )
        text_features = _text_features(clinical_note, model.text_vocabulary)
        return np.concatenate([image_features, text_features, metadata_feature_vector(record)]).astype(np.float32)
    return image_features.astype(np.float32)


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    clinical_note: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model = Path(model_path) if model_path else latest_model_path("image")
    if resolved_model is None or not resolved_model.exists():
        raise FileNotFoundError("No trained image model found. Run scripts/train_all.py first.")
    model = NumpyFeatureClassifier.load(resolved_model)
    features = build_features_for_model(model, image_path, clinical_note=clinical_note, metadata=metadata)
    probabilities = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probabilities))
    embedding = model.embed(features)[0]
    return {
        "model_path": str(resolved_model),
        "model_name": model.metadata.get("model_name", "numpy_feature_classifier"),
        "model_version": model.metadata.get("model_version", "v0"),
        "prediction_label": model.classes[pred_idx],
        "confidence": float(probabilities[pred_idx]),
        "probabilities": {label: float(probabilities[i]) for i, label in enumerate(model.classes)},
        "embedding": embedding.astype(float).round(6).tolist(),
        "classes": model.classes,
    }

