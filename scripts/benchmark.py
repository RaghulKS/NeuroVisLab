from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mlops.benchmarks import benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model load, online/batch inference, and monitoring.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.model_id, args.image, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
