from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .domain import utc_now
from .minutes_templates import (
    DEFAULT_TEMPLATE_ID,
    default_minutes_template,
    normalize_minutes_template,
)


LATEST_SCHEMA_VERSION = 4
LIFECYCLE_STATES = {"active", "archived", "trashed"}


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
        with self.connect() as connection:
            current = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current > LATEST_SCHEMA_VERSION:
                raise RuntimeError(
                    "数据库版本高于当前程序支持范围："
                    f"{current} > {LATEST_SCHEMA_VERSION}"
                )
            migrations = {
                1: self._migrate_to_v1,
                2: self._migrate_to_v2,
                3: self._migrate_to_v3,
                4: self._migrate_to_v4,
            }
            for version in range(current + 1, LATEST_SCHEMA_VERSION + 1):
                migrations[version](connection)
                connection.execute(f"PRAGMA user_version = {version}")

    @staticmethod
    def _migrate_to_v1(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
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
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meetings_created
            ON meetings(created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meetings_status
            ON meetings(status)
            """
        )

    @staticmethod
    def _migrate_to_v2(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                checkpoint TEXT NOT NULL DEFAULT 'uploaded',
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                FOREIGN KEY (meeting_id)
                    REFERENCES meetings(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_meeting
            ON jobs(meeting_id, id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status, created_at)
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_active_meeting
            ON jobs(meeting_id)
            WHERE status IN ('queued', 'running')
            """
        )

    @staticmethod
    def _migrate_to_v3(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(meetings)"
            ).fetchall()
        }
        additions = {
            "lifecycle_state": (
                "TEXT NOT NULL DEFAULT 'active'"
            ),
            "archived_at": "TEXT",
            "trashed_at": "TEXT",
            "trashed_from": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE meetings ADD COLUMN {name} {definition}"
                )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meetings_lifecycle_created
            ON meetings(lifecycle_state, created_at DESC)
            """
        )

    @staticmethod
    def _migrate_to_v4(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS minutes_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                instructions TEXT NOT NULL DEFAULT '',
                sections_json TEXT NOT NULL,
                is_builtin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        default = default_minutes_template()
        connection.execute(
            """
            INSERT OR IGNORE INTO minutes_templates (
                id, name, description, instructions, sections_json,
                is_builtin, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                default["id"],
                default["name"],
                default["description"],
                default["instructions"],
                json.dumps(default["sections"], ensure_ascii=False),
                default["created_at"],
                default["updated_at"],
            ),
        )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(meetings)"
            ).fetchall()
        }
        if "minutes_template_id" not in columns:
            connection.execute(
                """
                ALTER TABLE meetings
                ADD COLUMN minutes_template_id
                TEXT NOT NULL DEFAULT 'lab-meeting'
                """
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meetings_minutes_template
            ON meetings(minutes_template_id)
            """
        )

    def schema_version(self) -> int:
        with self.connect() as connection:
            return int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )

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

    def get_meeting_by_slug(self, slug: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM meetings WHERE slug = ?", (slug,)
            ).fetchone()
        return dict(row) if row else None

    def list_meetings(
        self,
        limit: int = 100,
        lifecycle_state: str | None = "active",
    ) -> list[dict[str, Any]]:
        if lifecycle_state is not None and lifecycle_state not in LIFECYCLE_STATES:
            raise ValueError("无效的会议生命周期状态")
        with self.connect() as connection:
            if lifecycle_state is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM meetings
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM meetings
                    WHERE lifecycle_state = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (lifecycle_state, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_meetings_by_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        if start_date > end_date:
            raise ValueError("开始日期不能晚于结束日期")
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM meetings
                WHERE meeting_date BETWEEN ? AND ?
                ORDER BY meeting_date ASC, created_at ASC
                """,
                (start_date, end_date),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_minutes_templates(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM minutes_templates
                ORDER BY is_builtin DESC, updated_at DESC, name
                """
            ).fetchall()
        return [self._minutes_template_from_row(row) for row in rows]

    def get_minutes_template(
        self, template_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM minutes_templates WHERE id = ?",
                (template_id,),
            ).fetchone()
        return self._minutes_template_from_row(row) if row else None

    def create_minutes_template(
        self, template: dict[str, Any]
    ) -> None:
        normalized = normalize_minutes_template(template)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO minutes_templates (
                    id, name, description, instructions, sections_json,
                    is_builtin, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._minutes_template_values(normalized),
            )

    def update_minutes_template(
        self, template: dict[str, Any]
    ) -> bool:
        normalized = normalize_minutes_template(template)
        if normalized["id"] == DEFAULT_TEMPLATE_ID:
            return False
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE minutes_templates
                SET name = ?,
                    description = ?,
                    instructions = ?,
                    sections_json = ?,
                    updated_at = ?
                WHERE id = ? AND is_builtin = 0
                """,
                (
                    normalized["name"],
                    normalized["description"],
                    normalized["instructions"],
                    json.dumps(
                        normalized["sections"], ensure_ascii=False
                    ),
                    normalized["updated_at"],
                    normalized["id"],
                ),
            )
        return cursor.rowcount == 1

    def delete_minutes_template(self, template_id: str) -> str:
        if template_id == DEFAULT_TEMPLATE_ID:
            return "builtin"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            template = connection.execute(
                """
                SELECT is_builtin
                FROM minutes_templates
                WHERE id = ?
                """,
                (template_id,),
            ).fetchone()
            if template is None:
                return "missing"
            if int(template["is_builtin"]):
                return "builtin"
            usage = connection.execute(
                """
                SELECT COUNT(*)
                FROM meetings
                WHERE minutes_template_id = ?
                """,
                (template_id,),
            ).fetchone()[0]
            if int(usage):
                return "in_use"
            connection.execute(
                "DELETE FROM minutes_templates WHERE id = ?",
                (template_id,),
            )
        return "deleted"

    def count_meetings_using_template(self, template_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM meetings
                WHERE minutes_template_id = ?
                """,
                (template_id,),
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _minutes_template_values(
        template: dict[str, Any],
    ) -> tuple[Any, ...]:
        return (
            template["id"],
            template["name"],
            template["description"],
            template["instructions"],
            json.dumps(template["sections"], ensure_ascii=False),
            1 if template["is_builtin"] else 0,
            template["created_at"],
            template["updated_at"],
        )

    @staticmethod
    def _minutes_template_from_row(
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        return normalize_minutes_template(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "instructions": row["instructions"],
                "sections": json.loads(row["sections_json"]),
                "is_builtin": bool(row["is_builtin"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    def lifecycle_counts(self) -> dict[str, int]:
        result = {state: 0 for state in LIFECYCLE_STATES}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT lifecycle_state, COUNT(*) AS count
                FROM meetings
                GROUP BY lifecycle_state
                """
            ).fetchall()
        for row in rows:
            state = str(row["lifecycle_state"])
            if state in result:
                result[state] = int(row["count"])
        return result

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

    def has_active_jobs(self, meeting_id: str | None = None) -> bool:
        query = """
            SELECT 1
            FROM jobs
            WHERE status IN ('queued', 'running')
        """
        parameters: tuple[Any, ...] = ()
        if meeting_id is not None:
            query += " AND meeting_id = ?"
            parameters = (meeting_id,)
        query += " LIMIT 1"
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return row is not None

    def archive_meeting(self, meeting_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_job_exists(connection, meeting_id):
                return False
            cursor = connection.execute(
                """
                UPDATE meetings
                SET lifecycle_state = 'archived',
                    archived_at = ?,
                    trashed_at = NULL,
                    trashed_from = NULL,
                    updated_at = ?
                WHERE id = ? AND lifecycle_state = 'active'
                """,
                (now, now, meeting_id),
            )
        return cursor.rowcount == 1

    def unarchive_meeting(self, meeting_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE meetings
                SET lifecycle_state = 'active',
                    archived_at = NULL,
                    updated_at = ?
                WHERE id = ? AND lifecycle_state = 'archived'
                """,
                (now, meeting_id),
            )
        return cursor.rowcount == 1

    def trash_meeting(self, meeting_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_job_exists(connection, meeting_id):
                return False
            meeting = connection.execute(
                """
                SELECT lifecycle_state
                FROM meetings
                WHERE id = ?
                """,
                (meeting_id,),
            ).fetchone()
            if (
                meeting is None
                or meeting["lifecycle_state"]
                not in {"active", "archived"}
            ):
                return False
            cursor = connection.execute(
                """
                UPDATE meetings
                SET lifecycle_state = 'trashed',
                    trashed_at = ?,
                    trashed_from = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    meeting["lifecycle_state"],
                    now,
                    meeting_id,
                ),
            )
        return cursor.rowcount == 1

    def restore_meeting(self, meeting_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            meeting = connection.execute(
                """
                SELECT trashed_from, archived_at
                FROM meetings
                WHERE id = ? AND lifecycle_state = 'trashed'
                """,
                (meeting_id,),
            ).fetchone()
            if meeting is None:
                return False
            destination = (
                "archived"
                if meeting["trashed_from"] == "archived"
                else "active"
            )
            archived_at = (
                (meeting["archived_at"] or now)
                if destination == "archived"
                else None
            )
            cursor = connection.execute(
                """
                UPDATE meetings
                SET lifecycle_state = ?,
                    archived_at = ?,
                    trashed_at = NULL,
                    trashed_from = NULL,
                    updated_at = ?
                WHERE id = ? AND lifecycle_state = 'trashed'
                """,
                (destination, archived_at, now, meeting_id),
            )
        return cursor.rowcount == 1

    def delete_trashed_meeting(self, meeting_id: str) -> bool:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_job_exists(connection, meeting_id):
                return False
            cursor = connection.execute(
                """
                DELETE FROM meetings
                WHERE id = ? AND lifecycle_state = 'trashed'
                """,
                (meeting_id,),
            )
        return cursor.rowcount == 1

    def rollback_imported_meeting(self, meeting_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM meetings WHERE id = ?", (meeting_id,)
            )

    def import_job_history(
        self,
        meeting_id: str,
        jobs: list[dict[str, Any]],
    ) -> None:
        if not jobs:
            return
        with self.connect() as connection:
            for job in jobs:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        meeting_id, kind, status, checkpoint,
                        cancel_requested, attempts, last_error,
                        created_at, updated_at, started_at, finished_at
                    )
                    VALUES (
                        :meeting_id, :kind, :status, :checkpoint,
                        :cancel_requested, :attempts, :last_error,
                        :created_at, :updated_at, :started_at, :finished_at
                    )
                    """,
                    {"meeting_id": meeting_id, **job},
                )

    def create_snapshot(self, destination: Path) -> None:
        source = self.connect()
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
            source.close()

    @staticmethod
    def _active_job_exists(
        connection: sqlite3.Connection, meeting_id: str
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM jobs
            WHERE meeting_id = ?
              AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (meeting_id,),
        ).fetchone()
        return row is not None

    def create_job(
        self,
        meeting_id: str,
        kind: str,
        checkpoint: str,
    ) -> tuple[int, bool]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT id
                FROM jobs
                WHERE meeting_id = ?
                  AND status IN ('queued', 'running')
                ORDER BY id DESC
                LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"]), False
            cursor = connection.execute(
                """
                INSERT INTO jobs (
                    meeting_id, kind, status, checkpoint,
                    cancel_requested, attempts, created_at, updated_at
                )
                VALUES (?, ?, 'queued', ?, 0, 0, ?, ?)
                """,
                (meeting_id, kind, checkpoint, now, now),
            )
            return int(cursor.lastrowid), True

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_latest_job(self, meeting_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE meeting_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, meeting_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE meeting_id = ?
                ORDER BY id
                """,
                (meeting_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_job(self, job_id: int) -> bool:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    started_at = ?,
                    finished_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                  AND status = 'queued'
                  AND cancel_requested = 0
                """,
                (now, now, job_id),
            )
        return cursor.rowcount == 1

    def update_job_checkpoint(self, job_id: int, checkpoint: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET checkpoint = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (checkpoint, utc_now(), job_id),
            )

    def complete_job(
        self, job_id: int, meeting_id: str | None = None
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    checkpoint = 'completed',
                    cancel_requested = 0,
                    last_error = NULL,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            if meeting_id is not None:
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'completed',
                        progress = 100,
                        current_step = '处理完成',
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, meeting_id),
                )

    def fail_job(
        self,
        job_id: int,
        error: str,
        meeting_id: str | None = None,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    cancel_requested = 0,
                    last_error = ?,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error[:4000], now, now, job_id),
            )
            if meeting_id is not None:
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'failed',
                        current_step = '处理失败',
                        error = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (error[:4000], now, meeting_id),
                )

    def mark_job_canceled(
        self,
        job_id: int,
        meeting_id: str | None = None,
        current_step: str = "任务已取消",
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    cancel_requested = 1,
                    last_error = NULL,
                    finished_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            if meeting_id is not None:
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'canceled',
                        current_step = ?,
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (current_step, now, meeting_id),
                )

    def requeue_job(
        self,
        job_id: int,
        reason: str,
        meeting_id: str | None = None,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    cancel_requested = 0,
                    last_error = ?,
                    started_at = NULL,
                    finished_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (reason[:4000], now, job_id),
            )
            if meeting_id is not None:
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'queued',
                        current_step = ?,
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (reason[:500], now, meeting_id),
                )

    def job_cancel_requested(self, job_id: int) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT status, cancel_requested
                FROM jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        return (
            row is None
            or row["status"] == "canceled"
            or bool(row["cancel_requested"])
        )

    def request_job_cancel(self, meeting_id: str) -> str | None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT id, status
                FROM jobs
                WHERE meeting_id = ?
                  AND status IN ('queued', 'running')
                ORDER BY id DESC
                LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
            if job is None:
                return None
            if job["status"] == "queued":
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled',
                        cancel_requested = 1,
                        finished_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, job["id"]),
                )
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'canceled',
                        current_step = '任务已取消',
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, meeting_id),
                )
                return "canceled"
            connection.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1, updated_at = ?
                WHERE id = ?
                """,
                (now, job["id"]),
            )
            connection.execute(
                """
                UPDATE meetings
                SET status = 'canceling',
                    current_step = '正在安全停止当前处理步骤',
                    error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, meeting_id),
            )
            return "canceling"

    def recover_jobs(self) -> list[int]:
        now = utc_now()
        active_meeting_statuses = (
            "queued",
            "processing",
            "generating_minutes",
            "canceling",
        )
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            canceled_jobs = connection.execute(
                """
                SELECT id, meeting_id
                FROM jobs
                WHERE status IN ('queued', 'running')
                  AND cancel_requested = 1
                """
            ).fetchall()
            for job in canceled_jobs:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled',
                        finished_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, job["id"]),
                )
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'canceled',
                        current_step = '任务已取消',
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job["meeting_id"]),
                )

            interrupted = connection.execute(
                """
                SELECT id, meeting_id
                FROM jobs
                WHERE status = 'running'
                  AND cancel_requested = 0
                """
            ).fetchall()
            for job in interrupted:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        started_at = NULL,
                        finished_at = NULL,
                        last_error = '应用上次退出时任务被中断，已从最近断点重新排队。',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job["id"]),
                )
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'queued',
                        current_step = '应用重启，等待从最近断点继续',
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, job["meeting_id"]),
                )

            placeholders = ", ".join("?" for _ in active_meeting_statuses)
            legacy_meetings = connection.execute(
                f"""
                SELECT m.*
                FROM meetings AS m
                WHERE m.status IN ({placeholders})
                  AND NOT EXISTS (
                    SELECT 1
                    FROM jobs AS j
                    WHERE j.meeting_id = m.id
                      AND j.status IN ('queued', 'running')
                  )
                ORDER BY m.created_at
                """,
                active_meeting_statuses,
            ).fetchall()
            for meeting in legacy_meetings:
                if meeting["status"] == "canceling":
                    connection.execute(
                        """
                        UPDATE meetings
                        SET status = 'canceled',
                            current_step = '任务已取消',
                            error = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (now, meeting["id"]),
                    )
                    continue
                progress = int(meeting["progress"] or 0)
                if meeting["status"] == "generating_minutes" or progress >= 75:
                    kind = "minutes"
                    checkpoint = "transcribed"
                elif progress >= 30 and meeting["duration_seconds"]:
                    kind = "pipeline"
                    checkpoint = "normalized"
                else:
                    kind = "pipeline"
                    checkpoint = "uploaded"
                connection.execute(
                    """
                    INSERT INTO jobs (
                        meeting_id, kind, status, checkpoint,
                        cancel_requested, attempts, last_error,
                        created_at, updated_at
                    )
                    VALUES (?, ?, 'queued', ?, 0, 0, ?, ?, ?)
                    """,
                    (
                        meeting["id"],
                        kind,
                        checkpoint,
                        "从旧版队列迁移，已保留最近处理阶段。",
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE meetings
                    SET status = 'queued',
                        current_step = '等待从最近断点继续',
                        error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, meeting["id"]),
                )

            rows = connection.execute(
                """
                SELECT id
                FROM jobs
                WHERE status = 'queued'
                  AND cancel_requested = 0
                ORDER BY created_at, id
                """
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def recover_interrupted(self) -> list[str]:
        job_ids = self.recover_jobs()
        meeting_ids: list[str] = []
        for job_id in job_ids:
            job = self.get_job(job_id)
            if job is not None:
                meeting_ids.append(str(job["meeting_id"]))
        return meeting_ids

    def serialize_summary(self) -> str:
        return json.dumps(self.list_meetings(), ensure_ascii=False, indent=2)
