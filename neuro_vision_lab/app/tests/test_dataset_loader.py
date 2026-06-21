from __future__ import annotations

from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset


def test_demo_dataset_loader_creates_splits(tmp_path):
    dataset_dir = create_synthetic_demo_dataset(tmp_path / "demo", samples_per_class=5, image_size=64)
    splits = load_any_dataset(dataset_dir)
    assert splits.classes == ["artifact_or_other", "normal", "pneumonia_like_opacity"]
    assert len(splits.train) > 0
    assert len(splits.all_records()) == 15
    assert all(record.clinical_note for record in splits.all_records())

