from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.database import Database
from app.search import MeetingSearchService
from app.storage import MeetingStorage


RESULT_KEYS = {
    "meeting_id",
    "title",
    "meeting_date",
    "lifecycle_state",
    "source",
    "source_label",
    "snippet",
    "anchor",
    "evidence_time",
}


def _search_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MeetingSearchService, Database, MeetingStorage]:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    storage = MeetingStorage(settings)
    return MeetingSearchService(database, storage), database, storage


def _create_meeting(
    database: Database,
    storage: MeetingStorage,
    *,
    meeting_id: str,
    title: str,
    meeting_date: str,
    lifecycle_state: str = "active",
    source_filename: str = "recording.wav",
    glossary: str = "",
    created_at: str = "2026-07-28T08:00:00+00:00",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": meeting_id,
        "slug": f"{meeting_date}_{meeting_id}",
        "title": title,
        "meeting_date": meeting_date,
        "expected_speakers": 2,
        "glossary": glossary,
        "processing_mode": "local",
        "source_filename": source_filename,
        "source_suffix": ".wav",
        "status": "completed",
        "progress": 100,
        "current_step": "完成",
        "error": None,
        "duration_seconds": 120.0,
        "transcriber_backend": "test",
        "llm_backend": "test",
        "lifecycle_state": lifecycle_state,
        "created_at": created_at,
        "updated_at": created_at,
    }
    storage.prepare(record)
    database.create_meeting(record)
    return record


def _write_content(
    storage: MeetingStorage,
    meeting: dict[str, Any],
) -> None:
    storage.write_json(
        meeting,
        "speakers.json",
        {"SPEAKER_01": "Alice Chen", "SPEAKER_02": "李老师"},
    )
    storage.write_json(
        meeting,
        "transcript_edited.json",
        {
            "segments": [
                {
                    "id": "seg-1",
                    "start": 5,
                    "end": 10,
                    "timestamp": "00:00:05",
                    "speaker": "SPEAKER_01",
                    "text": "CUDA latency improved by forty percent.",
                },
                {
                    "id": "seg-2",
                    "start": 20,
                    "end": 24,
                    "speaker": "SPEAKER_02",
                    "text": "下周复现实验并核对数据。",
                },
            ]
        },
    )
    storage.write_json(
        meeting,
        "minutes.json",
        {
            "template": {
                "sections": [
                    {
                        "key": "summary",
                        "title": "会议摘要",
                        "kind": "summary",
                    },
                    {
                        "key": "decisions",
                        "title": "已形成的决定",
                        "kind": "list",
                    },
                    {
                        "key": "action_items",
                        "title": "待办事项",
                        "kind": "actions",
                    },
                    {
                        "key": "custom_01",
                        "title": "发布计划",
                        "kind": "list",
                    },
                ]
            },
            "summary": "团队完成了性能复盘。",
            "decisions": [
                {
                    "content": "预算方案待进一步确认",
                    "evidence_time": "00:00:10",
                },
                {
                    "content": "Launch proposal approved",
                    "evidence_time": "00:01:10",
                }
            ],
            "action_items": [
                {
                    "action_id": "act_internal_value",
                    "owner": "Alice",
                    "task": "prepare benchmark",
                    "due": "Friday",
                    "evidence_time": "00:01:20",
                    "status": "pending",
                }
            ],
            "custom_01": [{"content": "灰度发布到测试环境"}],
            "generator": "test",
        },
    )


def test_metadata_search_is_case_insensitive_and_uses_and_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, storage = _search_service(tmp_path, monkeypatch)
    meeting = _create_meeting(
        database,
        storage,
        meeting_id="aurora",
        title="Project Aurora Review",
        meeting_date="2026-07-28",
        source_filename="Quarterly-Recording.WAV",
        glossary="GPU, CUDA, tensor core",
    )
    _write_content(storage, meeting)

    title_results = service.search(
        "project REVIEW", scope="metadata"
    )
    assert len(title_results) == 1
    assert title_results[0]["source"] == "title"
    assert set(title_results[0]) == RESULT_KEYS

    cross_field = service.search("CUDA Alice", scope="metadata")
    assert len(cross_field) == 1
    assert "CUDA" in cross_field[0]["snippet"]
    assert "Alice" in cross_field[0]["snippet"]

    assert service.search(
        "quarterly-recording.wav", scope="metadata"
    )[0]["source"] == "recording"
    assert service.search(
        "2026-07-28", scope="metadata"
    )[0]["source"] == "date"
    assert service.search(
        "project nonexistent", scope="metadata"
    ) == []
    assert service.search("   ", scope="metadata") == []


def test_scopes_search_transcript_and_custom_minutes_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, storage = _search_service(tmp_path, monkeypatch)
    meeting = _create_meeting(
        database,
        storage,
        meeting_id="content",
        title="内容检索",
        meeting_date="2026-07-27",
    )
    _write_content(storage, meeting)

    transcript = service.search(
        "alice LATENCY", scope="transcript"
    )
    assert len(transcript) == 1
    assert transcript[0]["source"] == "transcript"
    assert transcript[0]["source_label"] == "逐字稿"
    assert transcript[0]["evidence_time"] == "00:00:05"
    assert transcript[0]["anchor"] == (
        "?seek=5&focus=transcript#transcript"
    )
    assert service.search("latency", scope="metadata") == []

    decision = service.search(
        "launch APPROVED", scope="minutes"
    )
    assert len(decision) == 1
    assert decision[0]["source"] == "minutes"
    assert "已形成的决定" in decision[0]["source_label"]
    assert decision[0]["evidence_time"] == "00:01:10"
    assert decision[0]["anchor"] == (
        "?seek=70&focus=minutes#minutes"
    )
    titled_decision = service.search(
        "决定 launch", scope="minutes"
    )
    assert titled_decision[0]["evidence_time"] == "00:01:10"

    custom = service.search("发布计划 灰度", scope="minutes")
    assert len(custom) == 1
    assert "发布计划" in custom[0]["snippet"]
    assert service.search("latency", scope="minutes") == []
    assert service.search("latency", scope="all") == transcript
    action = service.search("benchmark", scope="minutes")[0]
    assert "act_internal_value" not in action["snippet"]
    assert "pending" not in action["snippet"]

    minutes = storage.read_json(meeting, "minutes.json")
    minutes["decisions"] = []
    storage.write_json(meeting, "minutes.json", minutes)
    assert service.search("已形成的决定", scope="minutes") == []


def test_lifecycle_filter_limit_and_sorting_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, storage = _search_service(tmp_path, monkeypatch)
    records = [
        ("active-old", "2026-07-20", "active", "2026-07-20T08:00:00+00:00"),
        ("archived", "2026-07-29", "archived", "2026-07-29T08:00:00+00:00"),
        ("trashed", "2026-07-28", "trashed", "2026-07-28T08:00:00+00:00"),
        ("active-b", "2026-07-27", "active", "2026-07-27T09:00:00+00:00"),
        ("active-a", "2026-07-27", "active", "2026-07-27T09:00:00+00:00"),
    ]
    for meeting_id, meeting_date, state, created_at in records:
        _create_meeting(
            database,
            storage,
            meeting_id=meeting_id,
            title=f"Common topic {meeting_id}",
            meeting_date=meeting_date,
            lifecycle_state=state,
            created_at=created_at,
        )

    assert [
        item["meeting_id"]
        for item in service.search("common", scope="metadata")
    ] == ["active-a", "active-b", "active-old"]
    assert [
        item["meeting_id"]
        for item in service.search(
            "common",
            scope="metadata",
            lifecycle_state="all",
            limit=3,
        )
    ] == ["archived", "trashed", "active-a"]
    assert [
        item["meeting_id"]
        for item in service.search(
            "common", scope="metadata", lifecycle_state="archived"
        )
    ] == ["archived"]

    with pytest.raises(ValueError):
        service.search("common", scope="unknown")
    with pytest.raises(ValueError):
        service.search("common", lifecycle_state="deleted")
    with pytest.raises(ValueError):
        service.search("common", limit=0)


def test_missing_or_damaged_files_do_not_abort_other_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, storage = _search_service(tmp_path, monkeypatch)
    damaged = _create_meeting(
        database,
        storage,
        meeting_id="damaged",
        title="Damaged artifact meeting",
        meeting_date="2026-07-28",
    )
    storage.path(damaged, "speakers.json").write_text(
        "{not-json", encoding="utf-8"
    )
    storage.path(damaged, "transcript_edited.json").write_text(
        "{not-json", encoding="utf-8"
    )
    storage.path(damaged, "minutes.json").write_bytes(b"\xff\xfe")

    valid = _create_meeting(
        database,
        storage,
        meeting_id="valid",
        title="Valid meeting",
        meeting_date="2026-07-27",
    )
    storage.write_json(
        valid,
        "transcript_raw.json",
        {
            "segments": [
                {
                    "start": 3,
                    "speaker": "SPEAKER_01",
                    "text": "fallback searchable evidence",
                }
            ]
        },
    )

    metadata = service.search("damaged artifact", scope="all")
    assert [item["meeting_id"] for item in metadata] == ["damaged"]
    fallback = service.search("fallback searchable", scope="transcript")
    assert [item["meeting_id"] for item in fallback] == ["valid"]
    assert fallback[0]["evidence_time"] == "00:00:03"
    assert service.search("anything", scope="minutes") == []


def test_snippets_are_bounded_safe_plain_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, database, storage = _search_service(tmp_path, monkeypatch)
    meeting = _create_meeting(
        database,
        storage,
        meeting_id="unsafe",
        title="安全片段",
        meeting_date="2026-07-28",
    )
    storage.write_json(
        meeting,
        "transcript_edited.json",
        {
            "segments": [
                {
                    "start": 1,
                    "speaker": "SPEAKER_01",
                    "text": (
                        "prefix " * 80
                        + "<script>alert(1)</script>\x00"
                        + " suffix" * 80
                    ),
                }
            ]
        },
    )

    result = service.search("alert", scope="transcript")[0]
    assert "alert(1)" in result["snippet"]
    assert "<" not in result["snippet"]
    assert ">" not in result["snippet"]
    assert "\x00" not in result["snippet"]
    assert len(result["snippet"]) <= 240
    assert result["snippet"].startswith("…")
    assert result["snippet"].endswith("…")
