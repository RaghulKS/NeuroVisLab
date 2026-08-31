"""One command, executable MLOps evidence: data → train → register → serve → monitor → rollout → rollback."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.mlops.benchmarks import benchmark  # noqa: E402
from app.mlops.config import load_config  # noqa: E402
from app.mlops.data import acquire_dataset, load_medmnist_arrays  # noqa: E402
from app.mlops.gates import evaluate_gates  # noqa: E402
from app.mlops.monitoring import submit_label, summary  # noqa: E402
from app.mlops.registry import promote, register_run, rollback, set_shadow  # noqa: E402
from app.mlops.serving import InferenceService  # noqa: E402
from app.mlops.training import CLASS_NAMES, train_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeuroVisionLab MLOps lifecycle demonstration.")
    parser.add_argument("--config", default="configs/demo.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    started = time.perf_counter()
    lock = acquire_dataset(config.data)
    primary_run = train_experiment(config, lock["lock_path"])
    primary = register_run(primary_run, evaluate_gates(primary_run, config.gates), actor="lifecycle-demo")
    staging = promote(primary["model_id"], "staging", actor="lifecycle-demo", reason="gated candidate enters staging")
    production = promote(primary["model_id"], "production", 100, actor="lifecycle-demo", reason="initial controlled production release")
    # A second real training run produces a separately versioned candidate for shadow/canary/rollback.
    candidate_run = train_experiment(config, lock["lock_path"])
    candidate = register_run(candidate_run, evaluate_gates(candidate_run, config.gates), actor="lifecycle-demo")
    candidate_staging = promote(candidate["model_id"], "staging", actor="lifecycle-demo", reason="shadow candidate staging")
    set_shadow(candidate["model_id"], actor="lifecycle-demo")
    arrays = load_medmnist_arrays(lock["lock_path"], config.data)
    image, label = arrays["test"]
    sample = Image.fromarray(image[0])
    sample_path = Path(primary_run["run_dir"]) / "benchmark_sample.png"
    sample.save(sample_path)
    buffer = io.BytesIO()
    sample.save(buffer, format="PNG")
    service = InferenceService()
    online = service.infer(buffer.getvalue(), request_id="lifecycle-online")
    batch = service.batch_infer([buffer.getvalue()] * 4)
    submit_label(online["request_id"], CLASS_NAMES[int(label[0])])
    monitoring = summary()
    canary_started = time.perf_counter()
    canary = promote(candidate["model_id"], "production", config.rollout.canary_percent, actor="lifecycle-demo", reason="10 percent deterministic canary")
    rollback_result = rollback(candidate["model_id"], actor="lifecycle-demo", reason="demonstrated safe rollback to prior version")
    rollout_rollback_ms = (time.perf_counter() - canary_started) * 1000
    benchmark_report = benchmark(rollback_result["model_id"], sample_path, batch_size=8)
    report = {
        "data": {"data_version": lock["data_version"], "sha256": lock["sha256"], "lock_path": lock["lock_path"]},
        "primary": {"run_id": primary_run["run_id"], "metrics": primary_run["metrics"], "model_id": primary["model_id"], "staging": staging["state"], "production": production["state"]},
        "candidate": {"run_id": candidate_run["run_id"], "metrics": candidate_run["metrics"], "model_id": candidate["model_id"], "staging": candidate_staging["state"]},
        "inference": {"online": online, "batch_count": len(batch)},
        "monitoring": monitoring,
        "rollout": {"canary_traffic_percent": canary["traffic_percent"], "rollback_restored_model": rollback_result["model_id"], "promotion_and_rollback_ms": rollout_rollback_ms},
        "benchmarks": benchmark_report,
        "total_seconds": time.perf_counter() - started,
        "disclaimer": "This is an engineering demo using a research benchmark. It is not clinically validated or clinically deployable.",
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
