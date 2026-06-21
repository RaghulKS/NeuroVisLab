from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


def _connect() -> sqlite3.Connection:
    settings.ensure_directories()
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                model_name TEXT,
                model_version TEXT,
                prediction_label TEXT,
                confidence REAL,
                uncertainty_flag INTEGER,
                payload_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def log_case_prediction(payload: dict[str, Any]) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO case_predictions (
                case_id, model_name, model_version, prediction_label,
                confidence, uncertainty_flag, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("case_id"),
                payload.get("model_name"),
                payload.get("model_version"),
                payload.get("prediction_label"),
                float(payload.get("confidence", 0.0)),
                int(bool(payload.get("uncertainty_flag", False))),
                json.dumps(payload, indent=2, default=str),
                now,
            ),
        )
        return int(cur.lastrowid)


def list_case_predictions(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM case_predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def log_model_registry(payload: dict[str, Any]) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO model_registry (model_name, model_version, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                payload.get("model_name", "unknown"),
                payload.get("model_version", "v0"),
                json.dumps(payload, indent=2, default=str),
                now,
            ),
        )
        return int(cur.lastrowid)


def backup_database(target_path: Path) -> Path:
    init_db()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as source, sqlite3.connect(target_path) as target:
        source.backup(target)
    return target_path

