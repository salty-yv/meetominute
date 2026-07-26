from __future__ import annotations

from datetime import date
from pathlib import Path

from app.database import Database
from app.domain import utc_now
from app.storage import make_slug


def test_slug_removes_windows_invalid_characters() -> None:
    slug = make_slug(date(2026, 7, 26), '课题: "A/B"?', "deadbeef")
    assert slug == "2026-07-26_课题_ _A_B_deadbeef"
    assert slug.endswith("_deadbeef")
    assert all(character not in slug for character in '<>:"/\\|?*')


def test_database_persists_and_recovers_interrupted(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    now = utc_now()
    database.create_meeting(
        {
            "id": "abc",
            "slug": "2026-07-26_test_abc",
            "title": "测试",
            "meeting_date": "2026-07-26",
            "expected_speakers": 2,
            "glossary": "",
            "processing_mode": "local",
            "source_filename": "test.wav",
            "source_suffix": ".wav",
            "status": "processing",
            "progress": 30,
            "current_step": "转写",
            "error": None,
            "duration_seconds": 12.5,
            "transcriber_backend": None,
            "llm_backend": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    assert database.get_meeting("abc")["status"] == "processing"
    job_id, created = database.create_job(
        "abc", "pipeline", "normalized"
    )
    assert created is True
    assert database.claim_job(job_id) is True

    assert database.recover_jobs() == [job_id]
    recovered = database.get_meeting("abc")
    recovered_job = database.get_job(job_id)
    assert recovered["status"] == "queued"
    assert "断点继续" in recovered["current_step"]
    assert recovered["error"] is None
    assert recovered_job["status"] == "queued"
    assert recovered_job["checkpoint"] == "normalized"
    assert recovered_job["attempts"] == 1


def test_database_cancels_queued_job_immediately(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    now = utc_now()
    database.create_meeting(
        {
            "id": "cancel-me",
            "slug": "2026-07-26_cancel_cancel",
            "title": "取消测试",
            "meeting_date": "2026-07-26",
            "expected_speakers": 2,
            "glossary": "",
            "processing_mode": "local",
            "source_filename": "test.wav",
            "source_suffix": ".wav",
            "status": "queued",
            "progress": 0,
            "current_step": "等待处理",
            "error": None,
            "duration_seconds": None,
            "transcriber_backend": None,
            "llm_backend": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    job_id, _ = database.create_job(
        "cancel-me", "pipeline", "uploaded"
    )

    assert database.request_job_cancel("cancel-me") == "canceled"
    assert database.get_job(job_id)["status"] == "canceled"
    assert database.get_meeting("cancel-me")["status"] == "canceled"
