from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

import app.pipeline as pipeline_module
from app.actions import (
    ACTION_STATUS_LABELS,
    ActionItemsError,
    ActionItemsService,
    annotate_action_items,
    is_action_overdue,
    normalize_action_status,
    reconcile_action_items,
)
from app.config import Settings
from app.database import Database
from app.domain import utc_now
from app.external_llm import ExternalLLMConfigStore
from app.pipeline import TaskQueue
from app.storage import MeetingStorage


def _action(
    task: str,
    *,
    owner: str = "李工",
    due: str = "2026-07-30",
    evidence_time: str = "00:03:00",
    status: str = "待处理",
) -> dict[str, str]:
    return {
        "owner": owner,
        "task": task,
        "due": due,
        "evidence_time": evidence_time,
        "status": status,
    }


def _minutes(*actions: dict[str, Any]) -> dict[str, Any]:
    return {
        "meeting": {"title": "测试会议", "date": "2026-07-28"},
        "summary": "测试",
        "action_items": list(actions),
    }


def _environment(tmp_path):
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    storage = MeetingStorage(settings)
    return settings, database, storage


def _create_meeting(
    database: Database,
    storage: MeetingStorage,
    meeting_id: str,
    title: str,
    meeting_date: str,
) -> dict[str, Any]:
    now = utc_now()
    meeting = {
        "id": meeting_id,
        "slug": f"{meeting_date}_{meeting_id}",
        "title": title,
        "meeting_date": meeting_date,
        "expected_speakers": 2,
        "glossary": "",
        "processing_mode": "local",
        "minutes_template_id": "lab-meeting",
        "source_filename": "meeting.wav",
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
    storage.prepare(meeting)
    database.create_meeting(meeting)
    return database.get_meeting(meeting_id)


def test_action_ids_are_stable_and_duplicate_safe() -> None:
    minutes = _minutes(
        _action("整理数据"),
        _action("整理数据"),
        _action("更新报告", status="已完成"),
    )

    first = annotate_action_items("meeting-1", minutes)
    second = annotate_action_items("meeting-1", minutes)

    first_ids = [item["action_id"] for item in first["action_items"]]
    assert first_ids == [
        item["action_id"] for item in second["action_items"]
    ]
    assert len(set(first_ids)) == 3
    assert first["action_items"][2]["status"] == "done"
    assert ACTION_STATUS_LABELS["done"] == "已完成"
    assert normalize_action_status("待处理") == "pending"
    assert normalize_action_status("已忽略") == "dismissed"


def test_regeneration_preserves_matching_statuses_and_drops_deleted_items() -> None:
    previous = _minutes(
        _action("重复任务", status="已完成"),
        _action("重复任务", status="已忽略"),
        _action("已删除任务", status="已完成"),
    )
    generated = _minutes(
        _action("重复任务", status="待处理"),
        _action("重复任务", status="待处理"),
        _action("新增任务", status="已完成"),
    )

    reconciled = reconcile_action_items(
        "meeting-1", generated, previous
    )

    assert [item["status"] for item in reconciled["action_items"]] == [
        "done",
        "dismissed",
        "pending",
    ]
    assert [item["task"] for item in reconciled["action_items"]] == [
        "重复任务",
        "重复任务",
        "新增任务",
    ]
    assert len(
        {item["action_id"] for item in reconciled["action_items"]}
    ) == 3


@pytest.mark.parametrize(
    ("status", "due", "expected"),
    [
        ("pending", "2026-07-27", True),
        ("待处理", "截止到 2026年7月27日", True),
        ("done", "2026-07-27", False),
        ("pending", "2026-07-28", False),
        ("pending", "下周", False),
        ("pending", "2026-99-99", False),
    ],
)
def test_overdue_detection(
    status: str,
    due: str,
    expected: bool,
) -> None:
    assert (
        is_action_overdue(
            {"status": status, "due": due},
            date(2026, 7, 28),
        )
        is expected
    )


def test_service_aggregates_filters_and_atomically_updates(
    tmp_path,
) -> None:
    _, database, storage = _environment(tmp_path)
    active = _create_meeting(
        database, storage, "active-1", "项目甲周会", "2026-07-28"
    )
    archived = _create_meeting(
        database, storage, "archived-1", "项目乙复盘", "2026-07-27"
    )
    trashed = _create_meeting(
        database, storage, "trashed-1", "项目丙评审", "2026-07-26"
    )
    storage.write_json(
        active,
        "minutes.json",
        _minutes(_action("修复接口", due="2026-07-27")),
    )
    storage.write_json(
        archived,
        "minutes.json",
        _minutes(_action("归档文档", status="已完成")),
    )
    storage.write_json(
        trashed,
        "minutes.json",
        _minutes(_action("取消采购", status="已忽略")),
    )
    assert database.archive_meeting(archived["id"])
    assert database.trash_meeting(trashed["id"])
    service = ActionItemsService(database, storage)

    all_actions = service.list_actions(today="2026-07-28")
    assert len(all_actions) == 3
    assert all(item["id"] == item["action_id"] for item in all_actions)
    assert all_actions[0]["meeting_title"] == all_actions[0]["title"]
    assert all_actions[0]["meeting_date"] == all_actions[0]["date"]
    assert (
        all_actions[0]["lifecycle_state"]
        == all_actions[0]["lifecycle"]
    )
    assert "action_id" not in storage.read_json(
        active, "minutes.json"
    )["action_items"][0]
    ensured = service.ensure_meeting_actions(active["id"])
    assert ensured["action_items"][0]["action_id"] == (
        all_actions[0]["action_id"]
    )
    assert storage.read_json(active, "minutes.json")["action_items"][0][
        "action_id"
    ] == all_actions[0]["action_id"]
    assert all_actions[0]["meeting_id"] == active["id"]
    assert all_actions[0]["is_overdue"] is True
    assert service.list_actions(status="已完成")[0]["meeting_id"] == (
        archived["id"]
    )
    assert service.list_actions(query="项目丙")[0]["lifecycle"] == (
        "trashed"
    )
    assert service.list_actions(query="项目甲 李工 接口")[0][
        "meeting_id"
    ] == active["id"]
    assert service.list_actions(lifecycle="archived")[0]["task"] == (
        "归档文档"
    )

    active_action_id = all_actions[0]["action_id"]
    updated = service.update_status(
        active["id"], active_action_id, "已完成"
    )
    assert updated["action_items"][0]["status"] == "done"
    persisted = json.loads(
        storage.path(active, "minutes.json").read_text(encoding="utf-8")
    )
    assert persisted["action_items"][0]["status"] == "done"
    assert (
        service.list_actions(status="done", today="2026-07-28")[0][
            "meeting_id"
        ]
        == active["id"]
    )

    with pytest.raises(ActionItemsError, match="不存在待办"):
        service.update_status(active["id"], "act_missing", "pending")
    with pytest.raises(ActionItemsError, match="无效的待办状态"):
        service.update_status(active["id"], active_action_id, "unknown")


def test_pipeline_regeneration_preserves_action_state(
    tmp_path,
    monkeypatch,
) -> None:
    settings, database, storage = _environment(tmp_path)
    meeting = _create_meeting(
        database, storage, "pipeline-1", "状态保留会议", "2026-07-28"
    )
    previous = annotate_action_items(
        meeting["id"],
        _minutes(_action("更新方案", status="已完成")),
    )
    previous_id = previous["action_items"][0]["action_id"]
    storage.write_json(meeting, "minutes.json", previous)
    storage.write_json(
        meeting,
        "transcript_edited.json",
        {
            "segments": [
                {
                    "id": "seg_0001",
                    "start": 0,
                    "end": 1,
                    "speaker": "SPEAKER_01",
                    "text": "请更新方案。",
                }
            ]
        },
    )
    storage.write_json(
        meeting, "speakers.json", {"SPEAKER_01": "李工"}
    )

    class RegeneratingMinutes:
        name = "mock"

        def generate(self, *args, **kwargs):
            return _minutes(
                _action("更新方案", status="待处理"),
                _action("补充测试", status="已完成"),
            )

    monkeypatch.setattr(
        pipeline_module,
        "create_minutes_generator",
        lambda *args, **kwargs: RegeneratingMinutes(),
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    queue = TaskQueue(
        settings,
        database,
        storage,
        ExternalLLMConfigStore(
            settings.data_dir / "external-llm.json",
            settings,
        ),
    )
    job_id, _ = database.create_job(
        meeting["id"], "minutes", "transcribed"
    )

    queue._generate_minutes(job_id, meeting)

    regenerated = storage.read_json(meeting, "minutes.json")
    assert regenerated["action_items"][0]["status"] == "done"
    assert regenerated["action_items"][0]["action_id"] == previous_id
    assert regenerated["action_items"][1]["status"] == "pending"


def test_service_reports_damaged_minutes_clearly(tmp_path) -> None:
    _, database, storage = _environment(tmp_path)
    meeting = _create_meeting(
        database, storage, "damaged-1", "损坏纪要", "2026-07-28"
    )
    storage.path(meeting, "minutes.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    service = ActionItemsService(database, storage)

    assert service.list_actions() == []
    with pytest.raises(ActionItemsError, match="损坏或无法读取"):
        service.ensure_meeting_actions(meeting["id"])


def test_pipeline_rebuilds_damaged_minutes_and_keeps_a_copy(
    tmp_path,
    monkeypatch,
) -> None:
    settings, database, storage = _environment(tmp_path)
    meeting = _create_meeting(
        database,
        storage,
        "damaged-pipeline",
        "损坏纪要重建",
        "2026-07-28",
    )
    storage.path(meeting, "minutes.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    storage.write_json(
        meeting,
        "transcript_edited.json",
        {
            "segments": [
                {
                    "id": "seg_0001",
                    "start": 0,
                    "end": 1,
                    "speaker": "SPEAKER_01",
                    "text": "请重建纪要。",
                }
            ]
        },
    )
    storage.write_json(
        meeting, "speakers.json", {"SPEAKER_01": "李工"}
    )

    class RebuiltMinutes:
        name = "mock"

        def generate(self, *args, **kwargs):
            return _minutes(_action("核对重建结果"))

    monkeypatch.setattr(
        pipeline_module,
        "create_minutes_generator",
        lambda *args, **kwargs: RebuiltMinutes(),
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    queue = TaskQueue(
        settings,
        database,
        storage,
        ExternalLLMConfigStore(
            settings.data_dir / "external-llm.json",
            settings,
        ),
    )
    job_id, _ = database.create_job(
        meeting["id"], "minutes", "transcribed"
    )

    queue._generate_minutes(job_id, meeting)

    rebuilt = storage.read_json(meeting, "minutes.json")
    assert rebuilt["action_items"][0]["task"] == "核对重建结果"
    assert rebuilt["action_items"][0]["status"] == "pending"
    preserved = storage.path(
        meeting, f"minutes.invalid-{job_id}.json"
    )
    assert preserved.read_text(encoding="utf-8") == "{not-json"
