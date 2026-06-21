from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from app.services.dataset_loader import CaseRecord, clean_age, normalize_sex


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_image(image: Image.Image, image_size: int = 128) -> Image.Image:
    return image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)


def image_to_array(image: Image.Image, image_size: int = 128, normalize: bool = True) -> np.ndarray:
    resized = resize_image(image, image_size=image_size)
    arr = np.asarray(resized, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return arr


def load_image_array(path: str | Path, image_size: int = 128, normalize: bool = True) -> np.ndarray:
    return image_to_array(load_image(path), image_size=image_size, normalize=normalize)


def normalize_image(arr: np.ndarray, mean: Iterable[float] | None = None, std: Iterable[float] | None = None) -> np.ndarray:
    image = arr.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    mean_arr = np.asarray(list(mean) if mean is not None else [0.485, 0.456, 0.406], dtype=np.float32)
    std_arr = np.asarray(list(std) if std is not None else [0.229, 0.224, 0.225], dtype=np.float32)
    return (image - mean_arr.reshape(1, 1, 3)) / std_arr.reshape(1, 1, 3)


def clean_metadata(record: CaseRecord) -> CaseRecord:
    record.age = clean_age(record.age)
    record.sex = normalize_sex(record.sex)
    if record.view_position:
        record.view_position = str(record.view_position).strip().upper()
    return record


def compute_image_features(path_or_image: str | Path | Image.Image | np.ndarray, image_size: int = 128) -> np.ndarray:
    if isinstance(path_or_image, np.ndarray):
        arr = path_or_image.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    elif isinstance(path_or_image, Image.Image):
        arr = image_to_array(path_or_image, image_size=image_size, normalize=True)
    else:
        arr = load_image_array(path_or_image, image_size=image_size, normalize=True)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    channel_mean = arr.mean(axis=(0, 1))
    channel_std = arr.std(axis=(0, 1))
    gray = arr.mean(axis=2)
    hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist = hist / max(float(hist.sum()), 1.0)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    features = np.concatenate(
        [
            channel_mean.astype(np.float32),
            channel_std.astype(np.float32),
            hist,
            np.asarray([gray.mean(), gray.std(), gx, gy], dtype=np.float32),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def feature_matrix(records: list[CaseRecord], image_size: int = 128) -> np.ndarray:
    if not records:
        return np.empty((0, 26), dtype=np.float32)
    return np.vstack([compute_image_features(record.image_path, image_size=image_size) for record in records]).astype(np.float32)


def metadata_feature_vector(record: CaseRecord) -> np.ndarray:
    age = clean_age(record.age)
    age_value = 0.0 if age is None else min(max(age, 0.0), 100.0) / 100.0
    sex = normalize_sex(record.sex) or "unknown"
    sex_vec = [float(sex == "male"), float(sex == "female"), float(sex not in {"male", "female"})]
    view = (record.view_position or "unknown").strip().upper()
    view_vec = [float(view == "PA"), float(view == "AP"), float(view in {"LATERAL", "LAT"}), float(view not in {"PA", "AP", "LATERAL", "LAT"})]
    missing = [
        float(record.age is None),
        float(record.sex is None),
        float(record.view_position is None),
    ]
    return np.asarray([age_value, *sex_vec, *view_vec, *missing], dtype=np.float32)


def metadata_matrix(records: list[CaseRecord]) -> np.ndarray:
    if not records:
        return np.empty((0, 11), dtype=np.float32)
    return np.vstack([metadata_feature_vector(record) for record in records]).astype(np.float32)

