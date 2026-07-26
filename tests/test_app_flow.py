from __future__ import annotations

import io
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000)
    return buffer.getvalue()


def _wait_for_status(database, meeting_id: str, wanted: str) -> dict:
    deadline = time.monotonic() + 15
    last = {}
    while time.monotonic() < deadline:
        last = database.get_meeting(meeting_id)
        if last["status"] == wanted:
            return last
        if last["status"] == "failed":
            raise AssertionError(last["error"])
        time.sleep(0.1)
    raise AssertionError(f"等待 {wanted} 超时；最后状态：{last}")


def test_upload_process_edit_and_export(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MEETOMINUTE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    application = create_app(settings)

    with TestClient(application) as client:
        response = client.post(
            "/meetings",
            data={
                "title": "端到端测试",
                "meeting_date": "2026-07-26",
                "expected_speakers": "2",
                "glossary": "MeetOminute",
                "processing_mode": "local",
            },
            files={"recording": ("sample.wav", _wav_bytes(), "audio/wav")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        meeting = application.state.database.list_meetings()[0]
        completed = _wait_for_status(
            application.state.database, meeting["id"], "completed"
        )
        assert completed["progress"] == 100

        directory = (
            settings.meetings_dir / completed["slug"]
        )
        assert (directory / "transcript_raw.json").exists()
        assert (directory / "transcript_edited.json").exists()
        assert (directory / "minutes.docx").exists()

        page = client.get(f"/meetings/{meeting['id']}")
        assert page.status_code == 200
        assert "开发模式占位文本" in page.text

        download = client.get(
            f"/meetings/{meeting['id']}/download/json"
        )
        assert download.status_code == 200
        assert download.json()["generator"] == "mock"

        speaker_response = client.post(
            f"/meetings/{meeting['id']}/speakers",
            data={
                "name_SPEAKER_01": "李老师",
                "name_SPEAKER_02": "王同学",
            },
            follow_redirects=False,
        )
        assert speaker_response.status_code == 303
        changed = application.state.database.get_meeting(meeting["id"])
        assert changed["status"] == "transcribed"
        assert not (directory / "minutes.json").exists()

