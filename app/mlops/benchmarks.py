from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import settings
from app.mlops.monitoring import summary
from app.mlops.registry import get_model
from app.mlops.serving import InferenceService
from app.mlops.training import load_model


def benchmark(model_id: str, image_path: str | Path, batch_size: int = 8) -> dict[str, Any]:
    """Measure real local serving operations; results are hardware-specific."""
    model = get_model(model_id)
    started = time.perf_counter()
    load_model(model["artifact_path"])
    load_ms = (time.perf_counter() - started) * 1000
    with Image.open(image_path) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
    service = InferenceService()
    started = time.perf_counter()
    online = service.infer(content, request_id="benchmark-online")
    online_total_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    service.batch_infer([content] * batch_size)
    batch_total_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    monitoring = summary()
    monitoring_ms = (time.perf_counter() - started) * 1000
    report = {
        "model_id": model_id,
        "load_ms": load_ms,
        "online_end_to_end_ms": online_total_ms,
        "model_forward_ms": online["latency_ms"],
        "batch_size": batch_size,
        "batch_total_ms": batch_total_ms,
        "throughput_images_per_second": batch_size / max(batch_total_ms / 1000, 1e-9),
        "monitoring_pipeline_ms": monitoring_ms,
        "monitoring_window_size": monitoring["window_size"],
        "notice": "Local benchmark: values are machine and workload dependent, not production SLOs.",
    }
    out = settings.artifacts_dir / "mlops" / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_id.replace(':', '_')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(path.resolve())
    return report
