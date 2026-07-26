from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backups import BackupError, BackupManager
from app.config import Settings
from app.database import LATEST_SCHEMA_VERSION, Database
from app.domain import utc_now
from app.main import create_app
from app.storage import MeetingStorage


def _meeting_record(
    meeting_id: str = "meeting-one",
    slug: str = "2026-07-27_backup_meeting-one",
) -> dict:
    now = utc_now()
    return {
        "id": meeting_id,
        "slug": slug,
        "title": "资料库测试会议",
        "meeting_date": "2026-07-27",
        "expected_speakers": 2,
        "glossary": "",
        "processing_mode": "local",
        "source_filename": "sample.wav",
        "source_suffix": ".wav",
        "status": "completed",
        "progress": 100,
        "current_step": "处理完成",
        "error": None,
        "duration_seconds": 1.0,
        "transcriber_backend": "mock",
        "llm_backend": "mock",
        "created_at": now,
        "updated_at": now,
    }


def _prepare_database(settings: Settings) -> tuple[Database, MeetingStorage]:
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    storage = MeetingStorage(settings)
    return database, storage


def _seed_meeting(
    database: Database,
    storage: MeetingStorage,
    record: dict | None = None,
) -> dict:
    meeting = record or _meeting_record()
    directory = storage.prepare(meeting)
    (directory / "original.wav").write_bytes(b"RIFF-backup-test")
    storage.write_json(meeting, "meeting.json", meeting)
    storage.write_text(meeting, "transcript.txt", "测试逐字稿")
    database.create_meeting(meeting)
    return database.get_meeting(meeting["id"])


@pytest.mark.parametrize("legacy_version", [0, 1])
def test_migrates_legacy_database_to_latest_schema(
    tmp_path: Path, legacy_version: int
) -> None:
    path = tmp_path / "legacy.sqlite3"
    record = _meeting_record()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE meetings (
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
        columns = ", ".join(record)
        placeholders = ", ".join("?" for _ in record)
        connection.execute(
            f"INSERT INTO meetings ({columns}) VALUES ({placeholders})",
            tuple(record.values()),
        )
        connection.execute(f"PRAGMA user_version = {legacy_version}")

    database = Database(path)
    database.initialize()
    database.initialize()

    assert database.schema_version() == LATEST_SCHEMA_VERSION
    migrated = database.get_meeting(record["id"])
    assert migrated["lifecycle_state"] == "active"
    assert migrated["archived_at"] is None
    with database.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(meetings)"
            ).fetchall()
        }
        jobs_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'jobs'
            """
        ).fetchone()
    assert {
        "lifecycle_state",
        "archived_at",
        "trashed_at",
        "trashed_from",
    } <= columns
    assert jobs_table is not None


def test_archive_trash_restore_and_permanent_delete_routes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    database, storage = _prepare_database(settings)
    meeting = _seed_meeting(database, storage)
    directory = storage.meeting_dir(meeting)
    application = create_app(settings)

    with TestClient(application) as client:
        archive = client.post(
            f"/meetings/{meeting['id']}/archive",
            follow_redirects=False,
        )
        assert archive.status_code == 303
        assert database.list_meetings() == []
        assert (
            database.get_meeting(meeting["id"])["lifecycle_state"]
            == "archived"
        )

        archive_page = client.get("/archive")
        assert archive_page.status_code == 200
        assert "资料库测试会议" in archive_page.text
        detail = client.get(f"/meetings/{meeting['id']}")
        assert "当前为只读模式" in detail.text

        trash = client.post(
            f"/meetings/{meeting['id']}/trash",
            follow_redirects=False,
        )
        assert trash.status_code == 303
        assert client.get(f"/meetings/{meeting['id']}").status_code == 404
        assert "资料库测试会议" in client.get("/trash").text

        restore = client.post(
            f"/meetings/{meeting['id']}/restore",
            follow_redirects=False,
        )
        assert restore.status_code == 303
        assert (
            database.get_meeting(meeting["id"])["lifecycle_state"]
            == "archived"
        )

        client.post(
            f"/meetings/{meeting['id']}/unarchive",
            follow_redirects=False,
        )
        client.post(
            f"/meetings/{meeting['id']}/trash",
            follow_redirects=False,
        )
        deleted = client.post(
            f"/meetings/{meeting['id']}/delete",
            follow_redirects=False,
        )
        assert deleted.status_code == 303
        assert database.get_meeting(meeting["id"]) is None
        assert not directory.exists()


def test_backup_round_trip_merges_without_overwriting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "source-data")
    )
    source_settings = Settings.from_env()
    source_database, source_storage = _prepare_database(
        source_settings
    )
    meeting = _seed_meeting(source_database, source_storage)
    job_id, _ = source_database.create_job(
        meeting["id"], "pipeline", "transcribed"
    )
    assert source_database.claim_job(job_id) is True
    source_database.complete_job(job_id, meeting["id"])
    assert source_database.archive_meeting(meeting["id"]) is True
    source_manager = BackupManager(
        source_settings, source_database, source_storage
    )

    backup = source_manager.create_backup()

    assert backup.path.is_file()
    assert backup.meeting_count == 1
    assert source_manager.get_backup(backup.name) == backup.path
    with zipfile.ZipFile(backup.path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert "database.sqlite3" in names
    assert (
        f"meetings/{meeting['slug']}/original.wav" in names
    )
    assert manifest["contains_external_llm_credentials"] is False

    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "target-data")
    )
    target_settings = Settings.from_env()
    target_database, target_storage = _prepare_database(
        target_settings
    )
    target_manager = BackupManager(
        target_settings, target_database, target_storage
    )

    first = target_manager.restore_backup(backup.path)
    second = target_manager.restore_backup(backup.path)

    assert first.imported == 1
    assert first.skipped == 0
    assert second.imported == 0
    assert second.skipped == 1
    restored = target_database.get_meeting(meeting["id"])
    assert restored["lifecycle_state"] == "archived"
    assert target_database.list_jobs(meeting["id"])[0]["status"] == "completed"
    assert (
        target_storage.path(restored, "transcript.txt").read_text(
            encoding="utf-8"
        )
        == "测试逐字稿"
    )


def test_restore_rejects_zip_path_traversal(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    settings = Settings.from_env()
    database, storage = _prepare_database(settings)
    manager = BackupManager(settings, database, storage)
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "meetominute-backup",
                    "format_version": 1,
                    "database_schema_version": LATEST_SCHEMA_VERSION,
                }
            ),
        )
        archive.writestr("../escaped.txt", "unsafe")

    with pytest.raises(BackupError, match="不安全路径"):
        manager.restore_backup(malicious)

    assert not (tmp_path / "escaped.txt").exists()


def test_backup_page_create_download_and_restore_upload(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    database, storage = _prepare_database(settings)
    _seed_meeting(database, storage)
    application = create_app(settings)

    with TestClient(application) as client:
        page = client.get("/backups")
        assert page.status_code == 200
        assert "创建完整备份" in page.text

        created = client.post("/backups", follow_redirects=False)
        assert created.status_code == 303
        backups = application.state.backup_manager.list_backups()
        assert len(backups) == 1

        download = client.get(f"/backups/{backups[0].name}")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"

        restored = client.post(
            "/backups/restore",
            files={
                "backup_file": (
                    backups[0].name,
                    backups[0].path.read_bytes(),
                    "application/zip",
                )
            },
            follow_redirects=False,
        )
        assert restored.status_code == 303
        assert "imported=0" in restored.headers["location"]
        assert "skipped=1" in restored.headers["location"]


def test_active_job_blocks_lifecycle_moves_and_backup(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    settings = Settings.from_env()
    database, storage = _prepare_database(settings)
    meeting = _seed_meeting(database, storage)
    database.create_job(meeting["id"], "pipeline", "uploaded")
    manager = BackupManager(settings, database, storage)

    assert database.archive_meeting(meeting["id"]) is False
    assert database.trash_meeting(meeting["id"]) is False
    with pytest.raises(BackupError, match="任务正在处理"):
        manager.create_backup()


def test_storage_rejects_meeting_directory_traversal(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    storage = MeetingStorage(Settings.from_env())

    with pytest.raises(ValueError, match="目录"):
        storage.meeting_dir({"slug": "../outside"})
