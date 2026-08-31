from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mlops.config import load_config  # noqa: E402
from app.mlops.data import acquire_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and lock the official MedMNIST benchmark bytes.")
    parser.add_argument("--config", default="configs/demo.toml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire_dataset(load_config(args.config).data, force=args.force), indent=2))


if __name__ == "__main__":
    main()
