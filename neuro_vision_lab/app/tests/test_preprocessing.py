from __future__ import annotations

import numpy as np

from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset
from app.services.preprocessing import compute_image_features, feature_matrix, load_image_array


def test_preprocessing_outputs_stable_shapes(tmp_path):
    dataset_dir = create_synthetic_demo_dataset(tmp_path / "demo", samples_per_class=2, image_size=64)
    splits = load_any_dataset(dataset_dir)
    record = splits.all_records()[0]
    arr = load_image_array(record.image_path, image_size=64)
    features = compute_image_features(record.image_path, image_size=64)
    matrix = feature_matrix(splits.all_records(), image_size=64)
    assert arr.shape == (64, 64, 3)
    assert features.shape == (26,)
    assert matrix.shape[1] == 26
    assert np.isfinite(matrix).all()

