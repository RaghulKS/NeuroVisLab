from __future__ import annotations

import io
import time
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.mlops.monitoring import record_observation
from app.mlops.registry import get_model, select_production, shadow_model_id
from app.mlops.training import load_model


def _tensor_from_bytes(content: bytes) -> tuple[torch.Tensor, float]:
    image = Image.open(io.BytesIO(content)).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, None, :, :]), float(array.mean())


class InferenceService:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}

    def _model(self, model: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
        if model["model_id"] not in self._cache:
            self._cache[model["model_id"]] = load_model(model["artifact_path"])
        return self._cache[model["model_id"]]

    def _predict(self, model: dict[str, Any], tensor: torch.Tensor) -> tuple[str, list[float], float]:
        network, payload = self._model(model)
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(network(tensor), dim=1)[0].numpy()
        latency = (time.perf_counter() - started) * 1000
        classes = payload["class_names"]
        calibrated = np.asarray(probabilities)
        temperature = float(payload.get("temperature", 1.0))
        calibrated = np.exp(np.log(np.maximum(calibrated, 1e-8)) / temperature)
        calibrated /= calibrated.sum()
        return str(classes[int(calibrated.argmax())]), calibrated.astype(float).tolist(), latency

    def infer(self, content: bytes, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or uuid.uuid4().hex
        served = select_production(request_id)
        tensor, brightness = _tensor_from_bytes(content)
        prediction, probabilities, latency = self._predict(served, tensor)
        shadow_id = shadow_model_id()
        shadow_prediction: str | None = None
        if shadow_id and shadow_id != served["model_id"]:
            shadow_prediction, _, _ = self._predict(get_model(shadow_id), tensor)
        record_observation(request_id, served["model_id"], prediction, probabilities, brightness, latency, shadow_id, shadow_prediction)
        entropy = float(-sum(value * np.log(max(value, 1e-12)) for value in probabilities))
        return {
            "request_id": request_id, "model_id": served["model_id"], "model_state": served["state"],
            "prediction": prediction, "probabilities": dict(zip(["normal", "pneumonia"], probabilities)),
            "confidence": max(probabilities), "uncertainty_entropy": entropy, "requires_review": max(probabilities) < .55,
            "latency_ms": latency, "shadow_executed": shadow_prediction is not None,
            "disclaimer": "Benchmark research demonstration only. Not medical advice, diagnosis, or clinical use.",
        }

    def batch_infer(self, payloads: list[bytes]) -> list[dict[str, Any]]:
        return [self.infer(payload) for payload in payloads]
