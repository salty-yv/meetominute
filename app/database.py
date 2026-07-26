from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .domain import utc_now


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            slug TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            meeting_date TEXT NOT NULL,
            expected_speakers INTEGER NOT NULL,
            glossary TEXT NOT NULL DEFAULT '',
            processing_mode TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            source_suffix TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            current_step TEXT NOT NULL DEFAULT '',
            error TEXT,
            duration_seconds REAL,
            transcriber_backend TEXT,
            llm_backend TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_meetings_created
            ON meetings(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_meetings_status
            ON meetings(status);
        """
        with self.connect() as connection:
            connection.executescript(schema)

    def create_meeting(self, record: dict[str, Any]) -> None:
        columns = ", ".join(record)
        placeholders = ", ".join(f":{key}" for key in record)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO meetings ({columns}) VALUES ({placeholders})",
                record,
            )

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_meetings(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM meetings ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_meeting(self, meeting_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = :{key}" for key in fields)
        fields["meeting_id"] = meeting_id
        with self.connect() as connection:
            connection.execute(
                f"UPDATE meetings SET {assignments} WHERE id = :meeting_id",
                fields,
            )

    def recover_interrupted(self) -> list[str]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE meetings
                SET status = 'failed',
                    error = '应用上次退出时任务仍在处理中，请点击重新处理。',
                    current_step = '处理被中断',
                    updated_at = ?
                WHERE status IN ('processing', 'generating_minutes')
                """,
                (now,),
            )
            rows = connection.execute(
                "SELECT id FROM meetings WHERE status = 'queued'"
            ).fetchall()
        return [row["id"] for row in rows]

    def serialize_summary(self) -> str:
        return json.dumps(self.list_meetings(), ensure_ascii=False, indent=2)

