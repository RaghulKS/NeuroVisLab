from __future__ import annotations

from pathlib import Path

from app.mlops import registry


def _run(tmp_path: Path, run_id: str) -> dict[str, object]:
    checkpoint = tmp_path / f"{run_id}.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    manifest = tmp_path / f"{run_id}.json"
    manifest.write_text("{}", encoding="utf-8")
    predictions = tmp_path / f"{run_id}.csv"
    predictions.write_text("truth,prediction\n", encoding="utf-8")
    return {
        "run_id": run_id, "data_version": "pneumoniamnist:v2:abc", "dataset_sha256": "abc", "dataset_lock": "lock.json",
        "artifacts": {"checkpoint": str(checkpoint), "checkpoint_sha256": "abc", "predictions": str(predictions)},
        "metrics": {"accuracy": .7, "macro_f1": .7, "expected_calibration_error": .1}, "run_manifest": str(manifest),
    }


def test_registry_canary_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "registry_path", lambda: tmp_path / "registry.db")
    gates = {"passed": True, "checks": {"all": True}}
    first = registry.register_run(_run(tmp_path, "first"), gates)
    registry.promote(first["model_id"], "staging")
    registry.promote(first["model_id"], "production")
    second = registry.register_run(_run(tmp_path, "second"), gates)
    registry.promote(second["model_id"], "staging")
    canary = registry.promote(second["model_id"], "production", traffic_percent=10)
    assert canary["traffic_percent"] == 10
    restored = registry.rollback(second["model_id"])
    assert restored["model_id"] == first["model_id"]
    assert restored["traffic_percent"] == 100
    assert any(event["action"] == "rollback" for event in registry.audit_trail())
