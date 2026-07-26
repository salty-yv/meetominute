from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.database import Database
from app.domain import utc_now
from app.main import create_app


def _meeting_record(
    meeting_id: str,
    title: str,
    meeting_date: str,
) -> dict:
    now = utc_now()
    return {
        "id": meeting_id,
        "slug": f"{meeting_date}_{meeting_id}",
        "title": title,
        "meeting_date": meeting_date,
        "expected_speakers": 2,
        "glossary": "",
        "processing_mode": "local",
        "source_filename": "sample.wav",
        "source_suffix": ".wav",
        "status": "completed",
        "progress": 100,
        "current_step": "处理完成",
        "error": None,
        "duration_seconds": 60.0,
        "transcriber_backend": "mock",
        "llm_backend": "mock",
        "created_at": now,
        "updated_at": now,
    }


def _calendar_app(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Settings, Database]:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    return settings, database


def test_calendar_page_groups_meetings_and_marks_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, database = _calendar_app(tmp_path, monkeypatch)
    active = _meeting_record(
        "active-meeting", "月初项目会", "2026-07-03"
    )
    archived = _meeting_record(
        "archived-meeting", "归档周会", "2026-07-14"
    )
    trashed = _meeting_record(
        "trashed-meeting", "已删除讨论", "2026-07-14"
    )
    adjacent = _meeting_record(
        "adjacent-meeting", "八月计划会", "2026-08-01"
    )
    for meeting in (active, archived, trashed, adjacent):
        database.create_meeting(meeting)
    assert database.archive_meeting(archived["id"]) is True
    assert database.trash_meeting(trashed["id"]) is True

    application = create_app(settings)
    with TestClient(application) as client:
        response = client.get("/calendar?month=2026-07")

    assert response.status_code == 200
    assert "2026年7月" in response.text
    assert "月初项目会" in response.text
    assert "归档周会" in response.text
    assert "已删除讨论" in response.text
    assert "八月计划会" in response.text
    assert "lifecycle-active" in response.text
    assert "lifecycle-archived" in response.text
    assert "lifecycle-trashed" in response.text
    assert "/meetings/active-meeting" in response.text
    assert "/meetings/archived-meeting" in response.text
    assert 'href="http://testserver/trash"' in response.text
    assert "month=2026-06" in response.text
    assert "month=2026-08" in response.text


def test_calendar_date_range_query_and_invalid_month(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, database = _calendar_app(tmp_path, monkeypatch)
    for meeting in (
        _meeting_record("before", "六月会议", "2026-06-30"),
        _meeting_record("inside", "七月会议", "2026-07-15"),
        _meeting_record("after", "八月会议", "2026-08-01"),
    ):
        database.create_meeting(meeting)

    meetings = database.list_meetings_by_date_range(
        "2026-07-01",
        "2026-07-31",
    )
    assert [meeting["id"] for meeting in meetings] == ["inside"]

    application = create_app(settings)
    with TestClient(application) as client:
        response = client.get("/calendar?month=2026-13")

    assert response.status_code == 422
    assert "月份格式应为 YYYY-MM" in response.text
