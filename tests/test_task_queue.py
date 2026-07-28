from __future__ import annotations

import io
import threading
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

import app.pipeline as pipeline_module
from app.config import Settings
from app.domain import Segment
from app.main import create_app


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000)
    return buffer.getvalue()


def _wait_for_status(
    database, meeting_id: str, wanted: str, timeout: float = 8
) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = database.get_meeting(meeting_id)
        if last["status"] == wanted:
            return last
        if last["status"] == "failed" and wanted != "failed":
            raise AssertionError(last["error"])
        time.sleep(0.03)
    raise AssertionError(f"等待 {wanted} 超时；最后状态：{last}")


def _post_meeting(client: TestClient, title: str) -> None:
    response = client.post(
        "/meetings",
        data={
            "title": title,
            "meeting_date": "2026-07-27",
            "expected_speakers": "2",
            "glossary": "",
            "processing_mode": "local",
        },
        files={"recording": ("sample.wav", _wav_bytes(), "audio/wav")},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _fake_normalize(
    source: Path,
    destination: Path,
    settings: Settings,
    cancel_check=None,
) -> float:
    if cancel_check is not None:
        cancel_check()
    destination.write_bytes(source.read_bytes())
    return 1.0


def _minutes_payload(meeting: dict) -> dict:
    return {
        "meeting": {
            "title": meeting["title"],
            "date": meeting["meeting_date"],
            "expected_speakers": meeting["expected_speakers"],
        },
        "summary": "已完成纪要恢复测试。",
        "member_progress": [],
        "experimental_results": [],
        "suggestions": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "next_followups": [],
        "generator": "test",
    }


def test_cancel_then_resume_from_normalized_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    started = threading.Event()
    calls = {"normalize": 0, "transcribe": 0}

    def counted_normalize(*args, **kwargs):
        calls["normalize"] += 1
        return _fake_normalize(*args, **kwargs)

    class CancelOnceTranscriber:
        name = "cancel-once"

        def transcribe(
            self,
            audio_path,
            expected_speakers,
            glossary,
            duration_seconds,
            *,
            cancel_check=None,
        ):
            calls["transcribe"] += 1
            if calls["transcribe"] == 1:
                started.set()
                while True:
                    time.sleep(0.02)
                    if cancel_check is not None:
                        cancel_check()
            return [
                Segment(
                    id="seg_0001",
                    start=0,
                    end=1,
                    speaker="SPEAKER_01",
                    text="恢复后的转写内容。",
                )
            ]

    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    monkeypatch.setattr(
        pipeline_module, "normalize_audio", counted_normalize
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_transcriber",
        lambda settings, mode: CancelOnceTranscriber(),
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    application = create_app(Settings.from_env())

    with TestClient(application) as client:
        _post_meeting(client, "取消与恢复测试")
        meeting = application.state.database.list_meetings()[0]
        assert started.wait(3)

        response = client.post(
            f"/meetings/{meeting['id']}/cancel",
            follow_redirects=False,
        )
        assert response.status_code == 303
        canceled = _wait_for_status(
            application.state.database, meeting["id"], "canceled"
        )
        canceled_job = application.state.database.get_latest_job(
            meeting["id"]
        )
        assert canceled["progress"] == 30
        assert canceled_job["status"] == "canceled"
        assert canceled_job["checkpoint"] == "normalized"

        page = client.get(f"/meetings/{meeting['id']}")
        assert "从语音转写继续" in page.text
        assert "从断点继续" in page.text

        resumed = client.post(
            f"/meetings/{meeting['id']}/resume",
            follow_redirects=False,
        )
        assert resumed.status_code == 303
        _wait_for_status(
            application.state.database, meeting["id"], "completed"
        )

        assert calls == {"normalize": 1, "transcribe": 2}
        jobs = application.state.database.list_jobs(meeting["id"])
        assert [job["status"] for job in jobs] == [
            "canceled",
            "completed",
        ]


def test_minutes_failure_resumes_without_retranscribing(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {"transcribe": 0, "minutes": 0}

    class CountingTranscriber:
        name = "counting"

        def transcribe(
            self,
            audio_path,
            expected_speakers,
            glossary,
            duration_seconds,
            *,
            cancel_check=None,
        ):
            calls["transcribe"] += 1
            if cancel_check is not None:
                cancel_check()
            return [
                Segment(
                    id="seg_0001",
                    start=0,
                    end=1,
                    speaker="SPEAKER_01",
                    text="只应该转写一次。",
                )
            ]

    class FlakyMinutesGenerator:
        name = "flaky"

        def generate(
            self,
            meeting,
            segments,
            speakers,
            *,
            template=None,
            cancel_check=None,
        ):
            calls["minutes"] += 1
            if cancel_check is not None:
                cancel_check()
            if calls["minutes"] == 1:
                raise RuntimeError("模拟纪要服务暂时失败")
            return _minutes_payload(meeting)

    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    monkeypatch.setattr(
        pipeline_module, "normalize_audio", _fake_normalize
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_transcriber",
        lambda settings, mode: CountingTranscriber(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_minutes_generator",
        lambda settings, mode, external_llm=None: FlakyMinutesGenerator(),
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    application = create_app(Settings.from_env())

    with TestClient(application) as client:
        _post_meeting(client, "纪要断点恢复测试")
        meeting = application.state.database.list_meetings()[0]
        failed = _wait_for_status(
            application.state.database, meeting["id"], "failed"
        )
        failed_job = application.state.database.get_latest_job(
            meeting["id"]
        )
        assert "模拟纪要服务暂时失败" in failed["error"]
        assert failed_job["checkpoint"] == "transcribed"
        assert calls == {"transcribe": 1, "minutes": 1}

        response = client.post(
            f"/meetings/{meeting['id']}/resume",
            follow_redirects=False,
        )
        assert response.status_code == 303
        _wait_for_status(
            application.state.database, meeting["id"], "completed"
        )

        assert calls == {"transcribe": 1, "minutes": 2}
        jobs = application.state.database.list_jobs(meeting["id"])
        assert [job["status"] for job in jobs] == [
            "failed",
            "completed",
        ]


def test_application_restart_automatically_resumes_running_job(
    tmp_path: Path, monkeypatch
) -> None:
    started = threading.Event()

    class BlockingTranscriber:
        name = "blocking"

        def transcribe(
            self,
            audio_path,
            expected_speakers,
            glossary,
            duration_seconds,
            *,
            cancel_check=None,
        ):
            started.set()
            while True:
                time.sleep(0.02)
                if cancel_check is not None:
                    cancel_check()

    class FinishingTranscriber:
        name = "finishing"

        def transcribe(
            self,
            audio_path,
            expected_speakers,
            glossary,
            duration_seconds,
            *,
            cancel_check=None,
        ):
            if cancel_check is not None:
                cancel_check()
            return [
                Segment(
                    id="seg_0001",
                    start=0,
                    end=1,
                    speaker="SPEAKER_01",
                    text="应用重启后自动恢复。",
                )
            ]

    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    monkeypatch.setattr(
        pipeline_module, "normalize_audio", _fake_normalize
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_transcriber",
        lambda settings, mode: BlockingTranscriber(),
    )
    settings = Settings.from_env()
    first_application = create_app(settings)

    with TestClient(first_application) as client:
        _post_meeting(client, "应用重启恢复测试")
        meeting = first_application.state.database.list_meetings()[0]
        assert started.wait(3)

    queued = first_application.state.database.get_meeting(meeting["id"])
    interrupted_job = first_application.state.database.get_latest_job(
        meeting["id"]
    )
    assert queued["status"] == "queued"
    assert interrupted_job["status"] == "queued"
    assert interrupted_job["checkpoint"] == "normalized"
    assert interrupted_job["attempts"] == 1

    monkeypatch.setattr(
        pipeline_module,
        "create_transcriber",
        lambda settings, mode: FinishingTranscriber(),
    )
    second_application = create_app(settings)
    with TestClient(second_application):
        _wait_for_status(
            second_application.state.database,
            meeting["id"],
            "completed",
        )

    resumed_job = second_application.state.database.get_latest_job(
        meeting["id"]
    )
    assert resumed_job["id"] == interrupted_job["id"]
    assert resumed_job["status"] == "completed"
    assert resumed_job["attempts"] == 2


def test_worker_survives_infrastructure_and_recovery_failures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    monkeypatch.setattr(
        pipeline_module, "normalize_audio", _fake_normalize
    )
    monkeypatch.setattr(
        pipeline_module, "release_ollama_model", lambda settings: False
    )
    application = create_app(Settings.from_env())
    database = application.state.database
    task_queue = application.state.task_queue
    original_claim_job = database.claim_job
    original_fail_job = database.fail_job
    original_complete_job = database.complete_job
    fragile_job_id: int | None = None
    claim_failures = 0
    recovery_failures = 0
    completion_order: list[str] = []
    task_queue._infrastructure_retry_limit = 3
    task_queue._infrastructure_retry_delay = 0.15

    def fail_first_claim(job_id: int) -> bool:
        nonlocal fragile_job_id, claim_failures
        if fragile_job_id is None:
            fragile_job_id = job_id
        if job_id == fragile_job_id and claim_failures < 2:
            claim_failures += 1
            raise RuntimeError("simulated claim infrastructure failure")
        return original_claim_job(job_id)

    def fail_first_recovery(
        job_id: int,
        error: str,
        meeting_id: str | None = None,
    ) -> None:
        nonlocal recovery_failures
        if job_id == fragile_job_id and recovery_failures < 2:
            recovery_failures += 1
            raise RuntimeError("simulated recovery write failure")
        original_fail_job(job_id, error, meeting_id)

    def record_completion(
        job_id: int, meeting_id: str | None = None
    ) -> None:
        original_complete_job(job_id, meeting_id)
        if meeting_id is not None:
            completion_order.append(meeting_id)

    monkeypatch.setattr(database, "claim_job", fail_first_claim)
    monkeypatch.setattr(database, "fail_job", fail_first_recovery)
    monkeypatch.setattr(database, "complete_job", record_completion)

    with TestClient(application) as client:
        _post_meeting(client, "Infrastructure failure")
        first_meeting = database.list_meetings()[0]

        _post_meeting(client, "Following healthy meeting")
        meetings = {
            meeting["title"]: meeting
            for meeting in database.list_meetings()
        }
        second_meeting = meetings["Following healthy meeting"]
        completed = _wait_for_status(
            database, second_meeting["id"], "completed"
        )
        first_before_retry = database.get_meeting(first_meeting["id"])
        recovered = _wait_for_status(
            database, first_meeting["id"], "completed"
        )

        worker = task_queue._worker
        assert completed["status"] == "completed"
        assert recovered["status"] == "completed"
        assert first_before_retry["status"] != "completed"
        assert worker is not None
        assert not worker.done()
        assert claim_failures == 2
        assert recovery_failures == 2
        assert completion_order[:2] == [
            second_meeting["id"],
            first_meeting["id"],
        ]

        first_job = database.get_latest_job(first_meeting["id"])
        assert first_job is not None
        assert first_job["status"] == "completed"
