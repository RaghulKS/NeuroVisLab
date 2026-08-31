from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.ml.metrics import evaluate_predictions
from app.mlops.config import ExperimentConfig
from app.mlops.data import load_medmnist_arrays, sha256_file
from app.mlops.model import MedMNISTCNN
from app.services.calibration import expected_calibration_error, temperature_scale_probabilities

CLASS_NAMES = ["normal", "pneumonia"]  # Benchmark labels only; never a patient-facing diagnosis.


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=settings.data_dir.parent, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Older/minimal Torch distributions can omit optional compiler dependencies.
    # Deterministic CPU kernels still receive the explicit seed if this setting
    # cannot be enabled.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, RuntimeError):
        pass
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(images.astype(np.float32)[:, None, :, :] / 255.0)
    y = torch.from_numpy(labels.astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Small deterministic grid search avoids an optimizer in the calibration stage."""
    if not len(labels):
        return 1.0
    candidates = np.linspace(0.5, 3.0, 26)
    losses: list[float] = []
    for temperature in candidates:
        calibrated = temperature_scale_probabilities(probabilities, float(temperature))
        losses.append(float(-np.log(np.maximum(calibrated[np.arange(len(labels)), labels], 1e-8)).mean()))
    return float(candidates[int(np.argmin(losses))])


def train_experiment(config: ExperimentConfig, lock_path: str | Path) -> dict[str, Any]:
    """Train a real PyTorch CNN and persist a self-contained, traceable run."""
    _seed_everything(config.training.seed)
    arrays = load_medmnist_arrays(lock_path, config.data)
    device = torch.device(config.training.device if config.training.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    test_x, test_y = arrays["test"]
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = settings.artifacts_dir / "mlops" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model = MedMNISTCNN(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    # PneumoniaMNIST is class-imbalanced. These deterministic inverse-frequency
    # weights keep the tiny demo from presenting a majority-class-only model as
    # useful; they are not a substitute for clinical performance analysis.
    counts = np.bincount(train_y, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = len(train_y) / np.maximum(len(CLASS_NAMES) * counts, 1.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    train_loader = _loader(train_x, train_y, config.training.batch_size, shuffle=True)
    val_loader = _loader(val_x, val_y, config.training.batch_size, shuffle=False)
    test_loader = _loader(test_x, test_y, config.training.batch_size, shuffle=False)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(config.training.epochs):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_probs, val_labels = _predict(model, val_loader, device)
        val_predictions = val_probs.argmax(axis=1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": float((val_predictions == val_labels).mean()),
        })
    training_seconds = time.perf_counter() - started
    raw_probs, eval_labels = _predict(model, test_loader, device)
    temperature = _fit_temperature(raw_probs, eval_labels)
    probabilities = temperature_scale_probabilities(raw_probs, temperature)
    predictions = probabilities.argmax(axis=1)
    y_true = [CLASS_NAMES[int(value)] for value in eval_labels]
    y_pred = [CLASS_NAMES[int(value)] for value in predictions]
    metrics = evaluate_predictions(y_true, y_pred, probabilities, CLASS_NAMES)
    metrics["expected_calibration_error"] = expected_calibration_error(y_true, probabilities, CLASS_NAMES)
    metrics["temperature"] = temperature
    metrics["training_seconds"] = training_seconds
    metrics["device"] = str(device)
    checkpoint = run_dir / "model.pt"
    torch.save({
        "state_dict": model.state_dict(), "architecture": "MedMNISTCNN", "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES, "temperature": temperature, "run_id": run_id,
    }, checkpoint)
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "truth", "prediction", "confidence", *[f"prob_{name}" for name in CLASS_NAMES]])
        writer.writeheader()
        for index, (truth, pred, row) in enumerate(zip(y_true, y_pred, probabilities)):
            writer.writerow({"index": index, "truth": truth, "prediction": pred, "confidence": float(row.max()), **{f"prob_{name}": float(row[i]) for i, name in enumerate(CLASS_NAMES)}})
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.name,
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint,
        "git_sha": _git_sha(),
        "data_version": lock["data_version"],
        "dataset_lock": str(Path(lock_path).resolve()),
        "dataset_sha256": lock["sha256"],
        "split_sample_counts": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)},
        "training_class_weights": class_weights.astype(float).tolist(),
        "monitoring_baseline": {
            "brightness_mean": float((train_x.astype(np.float32) / 255.0).mean()),
            "brightness_std": float((train_x.astype(np.float32) / 255.0).std()),
            "prediction_distribution": {name: float((predictions == index).mean()) for index, name in enumerate(CLASS_NAMES)},
        },
        "artifacts": {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint), "predictions": str((run_dir / "predictions.csv").resolve())},
        "metrics": metrics,
        "history": history,
        "notice": "Metrics are from a benchmark subset and are not clinical performance evidence.",
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "model_card.md").write_text(
        "# Model card: " + run_id + "\n\n"
        "**Status:** candidate benchmark artifact; not clinically validated or deployable.\n\n"
        "## Lineage\n\n"
        f"- Dataset version: `{lock['data_version']}`\n"
        f"- Dataset SHA-256: `{lock['sha256']}`\n"
        f"- Config fingerprint: `{config.fingerprint}`\n"
        f"- Git revision: `{manifest['git_sha']}`\n"
        f"- Checkpoint SHA-256: `{manifest['artifacts']['checkpoint_sha256']}`\n\n"
        "## Benchmark subset metrics\n\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- Macro F1: {metrics['macro_f1']:.4f}\n"
        f"- Macro AUROC: {metrics.get('roc_auc_ovr_macro')}\n"
        f"- ECE after temperature scaling: {metrics['expected_calibration_error']:.4f}\n\n"
        "These measurements are from the configured MedMNIST benchmark subset. They do not establish clinical performance, safety, fairness, or utility.\n",
        encoding="utf-8",
    )
    experiments = settings.artifacts_dir / "mlops" / "experiments.jsonl"
    experiments.parent.mkdir(parents=True, exist_ok=True)
    with experiments.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True) + "\n")
    return {**manifest, "run_dir": str(run_dir.resolve()), "run_manifest": str((run_dir / "run.json").resolve())}


def load_model(checkpoint_path: str | Path) -> tuple[MedMNISTCNN, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = MedMNISTCNN(num_classes=int(payload["num_classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
