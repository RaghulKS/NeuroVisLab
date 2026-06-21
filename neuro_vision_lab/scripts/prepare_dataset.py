from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.dataset_loader import load_any_dataset, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and split a folder or CSV medical image dataset.")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()
    splits = load_any_dataset(args.dataset_path, csv_path=args.csv_path, image_root=args.image_root)
    summary = {
        "classes": splits.classes,
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }
    print(json.dumps(summary, indent=2))
    if args.manifest_out:
        write_manifest(splits.all_records(), args.manifest_out)
        print(f"Wrote manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()

