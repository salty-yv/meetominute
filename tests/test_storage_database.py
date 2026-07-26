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
            "duration_seconds": None,
            "transcriber_backend": None,
            "llm_backend": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    assert database.get_meeting("abc")["status"] == "processing"
    assert database.recover_interrupted() == []
    recovered = database.get_meeting("abc")
    assert recovered["status"] == "failed"
    assert "重新处理" in recovered["error"]
