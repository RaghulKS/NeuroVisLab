from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import settings

STATES = {"candidate", "staging", "production", "retired"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path() -> Path:
    path = settings.artifacts_dir / "mlops" / "registry.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(registry_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_registry() -> None:
    with _db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
          model_id TEXT PRIMARY KEY, run_id TEXT UNIQUE NOT NULL, state TEXT NOT NULL,
          traffic_percent INTEGER NOT NULL DEFAULT 0, artifact_path TEXT NOT NULL,
          artifact_sha256 TEXT NOT NULL, data_version TEXT NOT NULL, metrics_json TEXT NOT NULL,
          gates_json TEXT NOT NULL, run_manifest_path TEXT NOT NULL, created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL, previous_production_id TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_events (
          event_id TEXT PRIMARY KEY, model_id TEXT, action TEXT NOT NULL, actor TEXT NOT NULL,
          from_state TEXT, to_state TEXT, reason TEXT NOT NULL, payload_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS registry_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations (
          request_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, prediction TEXT NOT NULL,
          confidence REAL NOT NULL, entropy REAL NOT NULL, brightness REAL NOT NULL,
          latency_ms REAL NOT NULL, shadow_model_id TEXT, shadow_prediction TEXT,
          created_at TEXT NOT NULL, true_label TEXT, labeled_at TEXT
        );
        """)


def _audit(conn: sqlite3.Connection, model_id: str | None, action: str, actor: str, reason: str, from_state: str | None = None, to_state: str | None = None, payload: dict[str, Any] | None = None) -> None:
    conn.execute("INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, model_id, action, actor, from_state, to_state, reason, json.dumps(payload or {}), _now()))


def register_run(run: dict[str, Any], gate_report: dict[str, Any], actor: str = "pipeline") -> dict[str, Any]:
    init_registry()
    if not gate_report.get("passed"):
        raise ValueError("Candidate registration blocked by evaluation gates.")
    model_id = f"model-{run['run_id']}"
    with _db() as conn:
        existing = conn.execute("SELECT * FROM models WHERE run_id = ?", (run["run_id"],)).fetchone()
        if existing:
            return _model_dict(existing)
        conn.execute("INSERT INTO models VALUES (?, ?, 'candidate', 0, ?, ?, ?, ?, ?, ?, ?, ?, NULL)", (
            model_id, run["run_id"], run["artifacts"]["checkpoint"], run["artifacts"]["checkpoint_sha256"], run["data_version"],
            json.dumps(run["metrics"]), json.dumps(gate_report), run["run_manifest"], _now(), _now(),
        ))
        _audit(conn, model_id, "register", actor, "evaluation gates passed", None, "candidate", {"run_id": run["run_id"], "data_version": run["data_version"]})
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    return _model_dict(row)


def _model_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["metrics"] = json.loads(item.pop("metrics_json"))
    item["gates"] = json.loads(item.pop("gates_json"))
    return item


def get_model(model_id: str) -> dict[str, Any]:
    init_registry()
    with _db() as conn:
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    if not row:
        raise KeyError(f"Unknown model: {model_id}")
    return _model_dict(row)


def list_models() -> list[dict[str, Any]]:
    init_registry()
    with _db() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    return [_model_dict(row) for row in rows]


def audit_trail(limit: int = 100) -> list[dict[str, Any]]:
    init_registry()
    with _db() as conn:
        rows = conn.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]


def promote(model_id: str, target: str, traffic_percent: int = 100, actor: str = "operator", reason: str = "controlled promotion") -> dict[str, Any]:
    """Promote through candidate/staging/production; production may be a canary."""
    if target not in STATES - {"retired"}:
        raise ValueError("Target must be staging or production.")
    if not 1 <= traffic_percent <= 100:
        raise ValueError("traffic_percent must be in [1, 100].")
    init_registry()
    with _db() as conn:
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown model: {model_id}")
        source = row["state"]
        gates = json.loads(row["gates_json"])
        if not gates.get("passed"):
            raise ValueError("Promotion blocked: release gates did not pass.")
        if target == "staging" and source != "candidate":
            raise ValueError("Only candidates can enter staging.")
        if target == "production" and source not in {"candidate", "staging"}:
            raise ValueError("Only candidates or staging models can enter production.")
        previous = conn.execute("SELECT model_id FROM models WHERE state = 'production' ORDER BY updated_at DESC LIMIT 1", ()).fetchone()
        previous_id = previous["model_id"] if previous else None
        if target == "production":
            if traffic_percent == 100:
                conn.execute("UPDATE models SET state = 'retired', traffic_percent = 0, updated_at = ? WHERE state = 'production'", (_now(),))
            elif previous_id:
                conn.execute("UPDATE models SET traffic_percent = ?, updated_at = ? WHERE model_id = ?", (100 - traffic_percent, _now(), previous_id))
        conn.execute("UPDATE models SET state = ?, traffic_percent = ?, previous_production_id = ?, updated_at = ? WHERE model_id = ?", (target, traffic_percent if target == "production" else 0, previous_id, _now(), model_id))
        _audit(conn, model_id, "promote", actor, reason, source, target, {"traffic_percent": traffic_percent, "previous_production_id": previous_id})
        updated = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    return _model_dict(updated)


def rollback(model_id: str, actor: str = "operator", reason: str = "rollback requested") -> dict[str, Any]:
    init_registry()
    with _db() as conn:
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if not row or row["state"] != "production":
            raise ValueError("Rollback target must be a currently production model.")
        previous_id = row["previous_production_id"]
        if not previous_id:
            raise ValueError("No recorded prior production model is available for rollback.")
        previous = conn.execute("SELECT * FROM models WHERE model_id = ?", (previous_id,)).fetchone()
        if not previous:
            raise ValueError("Previous production model is no longer available.")
        conn.execute("UPDATE models SET state = 'retired', traffic_percent = 0, updated_at = ? WHERE model_id = ?", (_now(), model_id))
        conn.execute("UPDATE models SET state = 'production', traffic_percent = 100, updated_at = ? WHERE model_id = ?", (_now(), previous_id))
        _audit(conn, model_id, "rollback", actor, reason, "production", "retired", {"restored_model_id": previous_id})
        restored = conn.execute("SELECT * FROM models WHERE model_id = ?", (previous_id,)).fetchone()
    return _model_dict(restored)


def set_shadow(model_id: str | None, actor: str = "operator") -> None:
    init_registry()
    if model_id:
        get_model(model_id)
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO registry_settings VALUES ('shadow_model_id', ?)", (model_id or "",))
        _audit(conn, model_id, "configure_shadow", actor, "shadow configuration changed", payload={"shadow_model_id": model_id})


def shadow_model_id() -> str | None:
    init_registry()
    with _db() as conn:
        row = conn.execute("SELECT value FROM registry_settings WHERE key = 'shadow_model_id'").fetchone()
    return row["value"] if row and row["value"] else None


def select_production(request_id: str) -> dict[str, Any]:
    models = [model for model in list_models() if model["state"] == "production"]
    if not models:
        raise LookupError("No production model is available. Promote a gated candidate first.")
    bucket = int(uuid.uuid5(uuid.NAMESPACE_URL, request_id).hex[:8], 16) % 100
    cursor = 0
    for model in sorted(models, key=lambda item: item["created_at"]):
        cursor += int(model["traffic_percent"])
        if bucket < cursor:
            return model
    return models[-1]
