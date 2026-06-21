from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.config import settings

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}


@dataclass
class CaseRecord:
    image_path: str
    label: str
    patient_id: str | None = None
    clinical_note: str | None = None
    age: float | None = None
    sex: str | None = None
    view_position: str | None = None
    dataset_source: str = "unknown"
    split: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DatasetSplits:
    train: list[CaseRecord]
    validation: list[CaseRecord]
    test: list[CaseRecord]
    classes: list[str]

    def all_records(self) -> list[CaseRecord]:
        return [*self.train, *self.validation, *self.test]

    def to_dict(self) -> dict[str, object]:
        return {
            "train": [r.to_dict() for r in self.train],
            "validation": [r.to_dict() for r in self.validation],
            "test": [r.to_dict() for r in self.test],
            "classes": self.classes,
        }


def normalize_split_name(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    if key in {"train", "training"}:
        return "train"
    if key in {"val", "valid", "validation"}:
        return "validation"
    if key in {"test", "testing"}:
        return "test"
    return None


def normalize_sex(value: object | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"m", "male"}:
        return "male"
    if raw in {"f", "female"}:
        return "female"
    if raw in {"o", "other", "unknown"}:
        return "other"
    return None


def clean_age(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        age = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(\.\d+)?", str(value))
        if not match:
            return None
        age = float(match.group(0))
    if age < 0 or age > 120:
        return None
    return age


def generate_clinical_text(record: CaseRecord) -> str:
    label_text = record.label.replace("_", " ")
    parts = [f"Research demo case labeled {label_text}."]
    if record.age is not None:
        parts.append(f"Patient age metadata is {record.age:.0f}.")
    if record.sex:
        parts.append(f"Sex metadata is {record.sex}.")
    if record.view_position:
        parts.append(f"Image view position is {record.view_position}.")
    parts.append("Text is synthetic metadata context and must not be interpreted as clinical advice.")
    return " ".join(parts)


def _image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _records_from_class_folder(root: Path, split: str | None = None) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if class_dir.name.startswith("."):
            continue
        label = class_dir.name
        for image_path in sorted(_image_files(class_dir)):
            records.append(
                CaseRecord(
                    image_path=str(image_path.resolve()),
                    label=label,
                    patient_id=image_path.stem,
                    dataset_source=root.name,
                    split=split,
                )
            )
    return records


def load_folder_dataset(root: str | Path) -> DatasetSplits:
    dataset_root = Path(root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    records: list[CaseRecord] = []
    split_dirs = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower() in SPLIT_NAMES]
    if split_dirs:
        for split_dir in sorted(split_dirs):
            split = normalize_split_name(split_dir.name)
            records.extend(_records_from_class_folder(split_dir, split=split))
    else:
        records = _records_from_class_folder(dataset_root)

    for record in records:
        if not record.clinical_note:
            record.clinical_note = generate_clinical_text(record)
    return create_splits(records, seed=settings.random_seed)


def _pick(row: dict[str, str], names: list[str]) -> str | None:
    lower_map = {k.lower().strip(): v for k, v in row.items()}
    for name in names:
        value = lower_map.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def load_csv_dataset(csv_path: str | Path, image_root: str | Path | None = None) -> DatasetSplits:
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_file}")
    base_root = Path(image_root).resolve() if image_root else csv_file.parent
    records: list[CaseRecord] = []
    with csv_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_value = _pick(row, ["image_path", "path", "filename", "file", "image", "image_id"])
            label = _pick(row, ["label", "class", "diagnosis", "target", "finding"])
            if not image_value or not label:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = base_root / image_path
            record = CaseRecord(
                image_path=str(image_path.resolve()),
                label=str(label).strip(),
                patient_id=_pick(row, ["patient_id", "patient", "subject_id", "case_id"]),
                clinical_note=_pick(row, ["clinical_note", "note", "findings", "report", "text"]),
                age=clean_age(_pick(row, ["age", "patient_age"])),
                sex=normalize_sex(_pick(row, ["sex", "gender", "patient_sex"])),
                view_position=_pick(row, ["view_position", "view", "projection"]),
                dataset_source=_pick(row, ["dataset_source", "source"]) or csv_file.stem,
                split=normalize_split_name(_pick(row, ["split", "partition", "set"])),
            )
            if not record.clinical_note:
                record.clinical_note = generate_clinical_text(record)
            records.append(record)
    return create_splits(records, seed=settings.random_seed)


def load_any_dataset(dataset_path: str | Path | None = None, csv_path: str | Path | None = None, image_root: str | Path | None = None) -> DatasetSplits:
    if csv_path:
        return load_csv_dataset(csv_path, image_root=image_root)
    root = Path(dataset_path or settings.data_dir).resolve()
    csv_candidates = sorted(root.glob("*.csv")) if root.exists() else []
    if csv_candidates:
        return load_csv_dataset(csv_candidates[0], image_root=image_root or root)
    demo_dir = root / "demo_medical_images"
    if demo_dir.exists():
        return load_any_dataset(demo_dir)
    nested_csv = sorted(root.glob("*/metadata.csv")) if root.exists() else []
    if nested_csv:
        return load_csv_dataset(nested_csv[0], image_root=image_root or nested_csv[0].parent)
    return load_folder_dataset(root)


def create_splits(
    records: list[CaseRecord],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    if not records:
        return DatasetSplits(train=[], validation=[], test=[], classes=[])
    classes = sorted({record.label for record in records})
    train = [r for r in records if r.split == "train"]
    validation = [r for r in records if r.split == "validation"]
    test = [r for r in records if r.split == "test"]
    unsplit = [r for r in records if r.split is None]
    if unsplit:
        rng = random.Random(seed)
        by_label: dict[str, list[CaseRecord]] = {label: [] for label in classes}
        for record in unsplit:
            by_label[record.label].append(record)
        for label_records in by_label.values():
            rng.shuffle(label_records)
            n_total = len(label_records)
            n_test = max(1, int(round(n_total * test_fraction))) if n_total >= 3 else 0
            n_val = max(1, int(round(n_total * validation_fraction))) if n_total >= 4 else 0
            test.extend(label_records[:n_test])
            validation.extend(label_records[n_test : n_test + n_val])
            train.extend(label_records[n_test + n_val :])
    for split_name, split_records in (("train", train), ("validation", validation), ("test", test)):
        for record in split_records:
            record.split = split_name
            if not record.clinical_note:
                record.clinical_note = generate_clinical_text(record)
    if not train and (validation or test):
        train = [*validation, *test]
    return DatasetSplits(train=train, validation=validation, test=test, classes=classes)


def class_counts(records: list[CaseRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


def compute_class_weights(records: list[CaseRecord]) -> dict[str, float]:
    counts = class_counts(records)
    if not counts:
        return {}
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: total / (n_classes * count) for label, count in counts.items()}


def write_manifest(records: list[CaseRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CaseRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    return path


def _draw_demo_image(label: str, index: int, size: int, rng: random.Random) -> Image.Image:
    base = Image.new("L", (size, size), color=int(rng.uniform(25, 70)))
    draw = ImageDraw.Draw(base)
    if label == "normal":
        for _ in range(4):
            x0 = rng.randint(12, size // 2)
            y0 = rng.randint(10, size - 30)
            x1 = min(size - 8, x0 + rng.randint(20, 46))
            y1 = min(size - 8, y0 + rng.randint(10, 28))
            draw.ellipse((x0, y0, x1, y1), outline=int(rng.uniform(85, 135)), width=2)
    elif label == "pneumonia_like_opacity":
        for _ in range(5):
            cx = rng.randint(size // 4, size - size // 5)
            cy = rng.randint(size // 5, size - size // 5)
            radius = rng.randint(9, 24)
            fill = int(rng.uniform(135, 220))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
    else:
        for _ in range(3):
            x0 = rng.randint(14, size - 34)
            y0 = rng.randint(14, size - 34)
            x1 = x0 + rng.randint(16, 38)
            y1 = y0 + rng.randint(16, 38)
            draw.rectangle((x0, y0, x1, y1), fill=int(rng.uniform(95, 185)))
        draw.line((0, index * 7 % size, size, (index * 7 + size // 3) % size), fill=210, width=2)
    arr = np.asarray(base, dtype=np.int16)
    noise = np.random.default_rng(index).normal(0, 9, size=(size, size))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(radius=0.4)).convert("RGB")


def create_synthetic_demo_dataset(
    output_dir: str | Path | None = None,
    samples_per_class: int = 18,
    image_size: int = 128,
    seed: int = 42,
) -> Path:
    root = Path(output_dir or settings.data_dir / "demo_medical_images").resolve()
    root.mkdir(parents=True, exist_ok=True)
    labels = ["normal", "pneumonia_like_opacity", "artifact_or_other"]
    rng = random.Random(seed)
    metadata_rows: list[CaseRecord] = []
    for label in labels:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            image_path = class_dir / f"{label}_{i:03d}.png"
            image = _draw_demo_image(label, i + len(label), image_size, rng)
            image.save(image_path)
            record = CaseRecord(
                image_path=str(image_path.resolve()),
                label=label,
                patient_id=f"DEMO-{label[:3].upper()}-{i:03d}",
                age=float(rng.randint(22, 88)),
                sex=rng.choice(["male", "female"]),
                view_position=rng.choice(["PA", "AP"]),
                dataset_source="synthetic_demo",
            )
            record.clinical_note = generate_clinical_text(record)
            metadata_rows.append(record)
    write_manifest(metadata_rows, root / "metadata.csv")
    return root
