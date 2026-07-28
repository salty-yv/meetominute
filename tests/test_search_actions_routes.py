from __future__ import annotations

from pathlib import Path

from docx import Document
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.database import Database
from app.domain import utc_now
from app.main import create_app
from app.storage import MeetingStorage


def _meeting_record(
    meeting_id: str,
    *,
    title: str = "客户验收会",
    meeting_date: str = "2026-07-18",
) -> dict:
    now = utc_now()
    return {
        "id": meeting_id,
        "slug": f"{meeting_date}_{meeting_id}",
        "title": title,
        "meeting_date": meeting_date,
        "expected_speakers": 2,
        "glossary": "北极星 交付",
        "processing_mode": "local",
        "source_filename": "acceptance.wav",
        "source_suffix": ".wav",
        "status": "completed",
        "progress": 100,
        "current_step": "处理完成",
        "error": None,
        "duration_seconds": 80.0,
        "transcriber_backend": "mock",
        "llm_backend": "mock",
        "created_at": now,
        "updated_at": now,
    }


def _seed_meeting(
    settings: Settings,
    database: Database,
    meeting_id: str = "meeting-one",
) -> tuple[dict, MeetingStorage]:
    meeting = _meeting_record(meeting_id)
    database.create_meeting(meeting)
    storage = MeetingStorage(settings)
    storage.prepare(meeting)
    storage.write_json(
        meeting,
        "speakers.json",
        {"SPEAKER_01": "李老师"},
    )
    storage.write_json(
        meeting,
        "transcript_edited.json",
        {
            "segments": [
                {
                    "id": "seg_0001",
                    "start": 12,
                    "timestamp": "00:00:12",
                    "speaker": "SPEAKER_01",
                    "text": "客户确认北极星版本可以进入最终验收。",
                }
            ]
        },
    )
    storage.write_json(
        meeting,
        "minutes.json",
        {
            "meeting": {
                "title": meeting["title"],
                "date": meeting["meeting_date"],
            },
            "summary": "客户确认了最终验收范围。",
            "action_items": [
                {
                    "owner": "王同学",
                    "task": "整理验收报告",
                    "due": "2000-01-01",
                    "evidence_time": "00:00:12",
                    "status": "待处理",
                }
            ],
        },
    )
    return meeting, storage


def _seeded_app(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Settings, Database, dict]:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    meeting, _ = _seed_meeting(settings, database)
    return settings, database, meeting


def test_search_and_action_pages_are_connected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, database, meeting = _seeded_app(tmp_path, monkeypatch)
    application = create_app(settings)

    with TestClient(application) as client:
        search = client.get(
            "/search",
            params={
                "q": "客户 验收",
                "scope": "transcript",
                "lifecycle": "all",
            },
        )
        actions = client.get("/actions")
        detail = client.get(f"/meetings/{meeting['id']}")
        database.update_meeting(meeting["id"], status="queued")
        busy_actions = client.get("/actions")

    assert search.status_code == 200
    assert "客户确认北极星版本" in search.text
    assert "00:00:12" in search.text
    assert actions.status_code == 200
    assert "整理验收报告" in actions.text
    assert "已逾期" in actions.text
    assert detail.status_code == 200
    assert "打开待办中心" in detail.text
    assert "待处理" in detail.text
    assert (
        f"/meetings/{meeting['id']}/actions/act_" in detail.text
    )
    assert "会议处理完成后可更新" in busy_actions.text
    assert "标为完成" not in busy_actions.text


def test_action_update_refreshes_all_minutes_exports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, _, meeting = _seeded_app(tmp_path, monkeypatch)
    application = create_app(settings)

    with TestClient(application) as client:
        item = application.state.action_items.list_actions()[0]
        response = client.post(
            (
                f"/meetings/{meeting['id']}/actions/"
                f"{item['action_id']}"
            ),
            data={
                "status": "done",
                "selected_status": "pending",
                "query": "验收",
            },
            follow_redirects=False,
        )
        completed_page = client.get(
            "/actions", params={"status": "done"}
        )

    assert response.status_code == 303
    assert "status=pending" in response.headers["location"]
    assert "q=" in response.headers["location"]
    storage = application.state.storage
    minutes = storage.read_json(meeting, "minutes.json")
    assert minutes["action_items"][0]["status"] == "done"
    assert minutes["updated_at"]
    assert "已完成" in storage.path(
        meeting, "minutes.md"
    ).read_text(encoding="utf-8")
    assert "已完成" in storage.path(
        meeting, "minutes.txt"
    ).read_text(encoding="utf-8")
    document = Document(storage.path(meeting, "minutes.docx"))
    table_text = "\n".join(
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    )
    assert "已完成" in table_text
    assert completed_page.status_code == 200
    assert "整理验收报告" in completed_page.text


def test_action_update_does_not_partially_commit_when_export_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, _, meeting = _seeded_app(tmp_path, monkeypatch)
    application = create_app(settings)
    storage = application.state.storage
    before = storage.read_json(meeting, "minutes.json")

    def fail_docx(*_args, **_kwargs) -> None:
        raise RuntimeError("模拟 Word 导出失败")

    monkeypatch.setattr(
        main_module, "render_minutes_docx", fail_docx
    )
    with TestClient(application) as client:
        item = application.state.action_items.list_actions()[0]
        response = client.post(
            (
                f"/meetings/{meeting['id']}/actions/"
                f"{item['action_id']}"
            ),
            data={"status": "done"},
            follow_redirects=False,
        )

    assert response.status_code == 422
    assert storage.read_json(meeting, "minutes.json") == before
    assert not storage.path(meeting, "minutes.md").exists()
    assert not storage.path(meeting, "minutes.txt").exists()
    assert not storage.path(meeting, "minutes.docx").exists()


def test_calendar_shows_pending_action_count_and_archived_actions_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings, database, meeting = _seeded_app(tmp_path, monkeypatch)
    assert database.archive_meeting(meeting["id"]) is True
    application = create_app(settings)

    with TestClient(application) as client:
        calendar = client.get("/calendar?month=2026-07")
        global_search = client.get("/search?q=客户验收")
        item = application.state.action_items.list_actions()[0]
        update = client.post(
            (
                f"/meetings/{meeting['id']}/actions/"
                f"{item['action_id']}"
            ),
            data={"status": "done"},
            follow_redirects=False,
        )

    assert calendar.status_code == 200
    assert "1 项待办" in calendar.text
    assert "1 项逾期" in calendar.text
    assert global_search.status_code == 200
    assert "客户验收会" in global_search.text
    assert update.status_code == 303
    minutes = application.state.storage.read_json(
        meeting, "minutes.json"
    )
    assert minutes["action_items"][0]["status"] == "done"
