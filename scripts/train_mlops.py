from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mlops.config import load_config  # noqa: E402
from app.mlops.data import acquire_dataset  # noqa: E402
from app.mlops.gates import evaluate_gates  # noqa: E402
from app.mlops.training import train_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reproducible PyTorch MedMNIST lifecycle model.")
    parser.add_argument("--config", default="configs/demo.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    lock = acquire_dataset(config.data)
    run = train_experiment(config, lock["lock_path"])
    print(json.dumps({"run": run, "gate_report": evaluate_gates(run, config.gates)}, indent=2, default=str))


if __name__ == "__main__":
    main()
