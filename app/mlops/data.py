from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.mlops.config import DataConfig

DATASET_NOTICE = (
    "MedMNIST benchmark data are used for research/education only. The benchmark "
    "is not clinical data validation and this project is not for clinical use."
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_dir(config: DataConfig) -> Path:
    return settings.data_dir / "medmnist" / config.name / config.version


def acquire_dataset(config: DataConfig, force: bool = False) -> dict[str, Any]:
    """Download the official MedMNIST NPZ once and write a content-addressed lock.

    The source URL is config-controlled. A later run verifies the local SHA-256
    rather than silently replacing the data.
    """
    target_dir = dataset_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / f"{config.name}.npz"
    if force or not archive.exists():
        request = urllib.request.Request(config.url, headers={"User-Agent": "NeuroVisionLab/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    with np.load(archive, allow_pickle=False) as data:
        splits = {
            split: {
                "images": int(data[f"{split}_images"].shape[0]),
                "labels": int(data[f"{split}_labels"].shape[0]),
                "image_shape": list(data[f"{split}_images"].shape[1:]),
                "class_counts": {
                    str(int(label)): int(count)
                    for label, count in zip(*np.unique(data[f"{split}_labels"].reshape(-1), return_counts=True))
                },
            }
            for split in ("train", "val", "test")
        }
    lock: dict[str, Any] = {
        "dataset": config.name,
        "dataset_version": config.version,
        "source_url": config.url,
        "archive": str(archive.resolve()),
        "sha256": sha256_file(archive),
        "bytes": archive.stat().st_size,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "splits": splits,
        "notice": DATASET_NOTICE,
    }
    lock["data_version"] = f"{config.name}:{config.version}:{lock['sha256'][:16]}"
    lock_path = target_dir / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return {**lock, "lock_path": str(lock_path.resolve())}


def load_medmnist_arrays(lock_path: str | Path, config: DataConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lock: dict[str, Any] = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    archive = Path(lock["archive"])
    if not archive.exists() or sha256_file(archive) != lock["sha256"]:
        raise ValueError("Dataset bytes do not match dataset.lock.json; reacquire before training.")
    limits = {"train": config.max_train_samples, "val": config.max_val_samples, "test": config.max_test_samples}
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(archive, allow_pickle=False) as data:
        for split, limit in limits.items():
            images = np.asarray(data[f"{split}_images"][:limit], dtype=np.uint8)
            labels = np.asarray(data[f"{split}_labels"][:limit], dtype=np.int64).reshape(-1)
            output[split] = (images, labels)
    return output
