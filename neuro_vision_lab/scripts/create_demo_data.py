from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.dataset_loader import create_synthetic_demo_dataset, load_any_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic medical imaging demo dataset.")
    parser.add_argument("--output-dir", default=str(settings.data_dir / "demo_medical_images"))
    parser.add_argument("--samples-per-class", type=int, default=18)
    parser.add_argument("--image-size", type=int, default=settings.default_image_size)
    args = parser.parse_args()
    dataset_dir = create_synthetic_demo_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
    )
    splits = load_any_dataset(dataset_dir)
    print(f"Created demo dataset: {dataset_dir}")
    print(f"Classes: {', '.join(splits.classes)}")
    print(f"Split sizes: train={len(splits.train)} validation={len(splits.validation)} test={len(splits.test)}")
    print(settings.disclaimer)


if __name__ == "__main__":
    main()

