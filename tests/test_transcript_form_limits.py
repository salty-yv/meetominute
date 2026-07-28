from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi.testclient import TestClient

from app.config import Settings
from app.domain import utc_now
from app.main import create_app


def _meeting_record(meeting_id: str) -> dict:
    now = utc_now()
    return {
        "id": meeting_id,
        "slug": f"2026-07-28_{meeting_id}",
        "title": "Large transcript",
        "meeting_date": "2026-07-28",
        "expected_speakers": 2,
        "glossary": "",
        "processing_mode": "local",
        "source_filename": "sample.wav",
        "source_suffix": ".wav",
        "status": "transcribed",
        "progress": 75,
        "current_step": "Transcript ready",
        "error": None,
        "duration_seconds": 3_600.0,
        "transcriber_backend": "mock",
        "llm_backend": None,
        "created_at": now,
        "updated_at": now,
    }


def _segments(count: int) -> list[dict]:
    return [
        {
            "id": f"segment-{index}",
            "start": float(index),
            "end": float(index + 1),
            "speaker": "SPEAKER_01",
            "text": f"original-{index}",
        }
        for index in range(count)
    ]


def _transcript_form(segments: list[dict]) -> dict[str, str]:
    form: dict[str, str] = {}
    for index, segment in enumerate(segments):
        segment_id = segment["id"]
        form[f"speaker_{segment_id}"] = (
            "SPEAKER_01" if index % 2 == 0 else "SPEAKER_02"
        )
        form[f"text_{segment_id}"] = f"修改后的中文内容-{index}"
    return form


def test_large_transcript_uses_a_strict_dynamic_form_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    application = create_app(settings)
    segment_count = 600
    meeting_id = "large-transcript"

    with TestClient(application) as client:
        database = application.state.database
        storage = application.state.storage
        database.create_meeting(_meeting_record(meeting_id))
        meeting = database.get_meeting(meeting_id)
        assert meeting is not None
        storage.prepare(meeting)
        transcript = {"segments": _segments(segment_count)}
        storage.write_json(
            meeting, "transcript_edited.json", transcript
        )
        storage.write_json(
            meeting,
            "speakers.json",
            {"SPEAKER_01": "", "SPEAKER_02": ""},
        )

        form = _transcript_form(transcript["segments"])
        response = client.post(
            f"/meetings/{meeting_id}/transcript",
            data=form,
            follow_redirects=False,
        )

        assert response.status_code == 303
        saved = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        assert len(saved["segments"]) == segment_count
        assert saved["segments"][0]["text"] == "修改后的中文内容-0"
        assert saved["segments"][-1]["text"] == "修改后的中文内容-599"
        assert saved["segments"][-1]["speaker"] == "SPEAKER_02"

        rejected_form = _transcript_form(saved["segments"])
        rejected_form["unexpected_field"] = "should-not-be-accepted"
        rejected = client.post(
            f"/meetings/{meeting_id}/transcript",
            data=rejected_form,
            follow_redirects=False,
        )

        assert rejected.status_code == 400
        assert "Too many fields" in rejected.json()["detail"]
        unchanged = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        assert unchanged["segments"] == saved["segments"]


def test_oversized_transcript_form_rejects_forged_content_length(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    application = create_app(settings)
    meeting_id = "oversized-transcript"

    with TestClient(application) as client:
        database = application.state.database
        storage = application.state.storage
        database.create_meeting(_meeting_record(meeting_id))
        meeting = database.get_meeting(meeting_id)
        assert meeting is not None
        storage.prepare(meeting)
        transcript = {"segments": _segments(1)}
        storage.write_json(
            meeting, "transcript_edited.json", transcript
        )
        storage.write_json(
            meeting,
            "speakers.json",
            {"SPEAKER_01": "", "SPEAKER_02": ""},
        )
        original = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        body = urlencode(
            {
                "speaker_segment-0": "SPEAKER_01",
                "text_segment-0": "x" * 1_300_000,
            }
        ).encode()

        response = client.post(
            f"/meetings/{meeting_id}/transcript",
            content=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "1",
            },
            follow_redirects=False,
        )

        assert response.status_code == 413
        assert storage.read_json(
            meeting, "transcript_edited.json", default={}
        ) == original

        original_speakers = storage.read_json(
            meeting, "speakers.json", default={}
        )
        speaker_body = urlencode(
            {
                "name_SPEAKER_01": "x" * 5_000,
                "name_SPEAKER_02": "unchanged",
            }
        ).encode()
        speaker_response = client.post(
            f"/meetings/{meeting_id}/speakers",
            content=speaker_body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "1",
            },
            follow_redirects=False,
        )

        assert speaker_response.status_code == 413
        assert storage.read_json(
            meeting, "speakers.json", default={}
        ) == original_speakers
