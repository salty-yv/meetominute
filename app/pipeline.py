from __future__ import annotations

import asyncio
import copy
import threading
import traceback
from typing import Any, Literal

from .audio import normalize_audio
from .config import Settings
from .database import Database
from .domain import utc_now
from .external_llm import ExternalLLMConfigStore
from .providers import (
    create_minutes_generator,
    create_transcriber,
    release_ollama_model,
)
from .rendering import (
    render_minutes_docx,
    render_minutes_markdown,
    render_minutes_text,
    render_transcript_markdown,
    render_transcript_text,
)
from .storage import MeetingStorage


TaskKind = Literal["pipeline", "minutes"]
Checkpoint = Literal["uploaded", "normalized", "transcribed"]

CHECKPOINT_PROGRESS: dict[Checkpoint, int] = {
    "uploaded": 0,
    "normalized": 30,
    "transcribed": 75,
}
CHECKPOINT_RESUME_LABELS: dict[Checkpoint, str] = {
    "uploaded": "从音频预处理开始",
    "normalized": "从语音转写继续",
    "transcribed": "从纪要生成继续",
}


class TaskCanceled(RuntimeError):
    pass


class TaskInterrupted(RuntimeError):
    pass


class TaskQueue:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        storage: MeetingStorage,
        external_llm_store: ExternalLLMConfigStore,
    ):
        self.settings = settings
        self.database = database
        self.storage = storage
        self.external_llm_store = external_llm_store
        self._queue: asyncio.Queue[int | None] = asyncio.Queue()
        self._queued: set[int] = set()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = threading.Event()

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._stopping.clear()
        self._worker = asyncio.create_task(
            self._run(), name="meetominute-single-worker"
        )
        for job_id in self.database.recover_jobs():
            await self._queue_job(job_id)

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._stopping.set()
        await self._queue.put(None)
        await self._worker
        self._worker = None
        self._queued.clear()

    async def enqueue(
        self,
        kind: TaskKind,
        meeting_id: str,
        checkpoint: Checkpoint | None = None,
    ) -> bool:
        meeting = self._require_meeting(meeting_id)
        resolved = checkpoint or self.available_checkpoint(meeting)
        job_id, created = self.database.create_job(
            meeting_id, kind, resolved
        )
        await self._queue_job(job_id)
        if created:
            self._log(
                meeting,
                "任务已加入持久化队列，"
                f"{CHECKPOINT_RESUME_LABELS[resolved]}",
            )
            self._sync_metadata(meeting_id)
        return created

    def cancel(self, meeting_id: str) -> str | None:
        meeting = self._require_meeting(meeting_id)
        result = self.database.request_job_cancel(meeting_id)
        if result is not None:
            message = (
                "用户取消了排队任务"
                if result == "canceled"
                else "用户请求取消，正在安全停止当前步骤"
            )
            self._log(meeting, message)
            self._sync_metadata(meeting_id)
        return result

    def available_checkpoint(
        self, meeting: dict[str, Any]
    ) -> Checkpoint:
        try:
            transcript = self.storage.read_json(
                meeting, "transcript_edited.json", default={}
            )
        except (OSError, ValueError):
            transcript = {}
        if isinstance(transcript, dict) and transcript.get("segments"):
            return "transcribed"
        normalized = self.storage.path(meeting, "normalized.wav")
        if normalized.exists() and meeting.get("duration_seconds") is not None:
            return "normalized"
        return "uploaded"

    def resume_info(
        self, meeting: dict[str, Any]
    ) -> tuple[TaskKind, Checkpoint, int, str]:
        checkpoint = self.available_checkpoint(meeting)
        kind: TaskKind = (
            "minutes" if checkpoint == "transcribed" else "pipeline"
        )
        return (
            kind,
            checkpoint,
            CHECKPOINT_PROGRESS[checkpoint],
            CHECKPOINT_RESUME_LABELS[checkpoint],
        )

    async def _queue_job(self, job_id: int) -> None:
        if job_id in self._queued:
            return
        self._queued.add(job_id)
        await self._queue.put(job_id)

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            if job_id is None:
                self._queue.task_done()
                break
            try:
                await asyncio.to_thread(self._execute_job, job_id)
            finally:
                self._queued.discard(job_id)
                self._queue.task_done()

    def _execute_job(self, job_id: int) -> None:
        if not self.database.claim_job(job_id):
            return
        job = self.database.get_job(job_id)
        if job is None:
            return
        meeting = self._require_meeting(str(job["meeting_id"]))
        try:
            checkpoint = self.available_checkpoint(meeting)
            if checkpoint != job["checkpoint"]:
                self.database.update_job_checkpoint(job_id, checkpoint)
                direction = (
                    "检测到更完整的本地成果"
                    if CHECKPOINT_PROGRESS[checkpoint]
                    > CHECKPOINT_PROGRESS.get(job["checkpoint"], -1)
                    else "检测到断点文件缺失"
                )
                self._log(
                    meeting,
                    f"{direction}，已调整为"
                    f"“{CHECKPOINT_RESUME_LABELS[checkpoint]}”",
                )
            self._raise_if_canceled(job_id)
            self._log(
                meeting,
                f"开始执行任务 #{job_id}（第 {int(job['attempts'])} 次），"
                f"{CHECKPOINT_RESUME_LABELS[checkpoint]}",
            )
            if job["kind"] == "minutes":
                if checkpoint != "transcribed":
                    raise RuntimeError(
                        "逐字稿断点不存在，无法单独生成纪要；"
                        "请从头重新处理。"
                    )
                self._generate_minutes(job_id, meeting)
            else:
                self._process_pipeline(job_id, meeting, checkpoint)
            self._raise_if_canceled(job_id)
            self.database.complete_job(job_id, meeting["id"])
        except TaskCanceled:
            current_job = self.database.get_job(job_id) or job
            checkpoint = current_job.get("checkpoint", "uploaded")
            resume_label = CHECKPOINT_RESUME_LABELS.get(
                checkpoint, "从最近断点继续"
            )
            self.database.mark_job_canceled(
                job_id,
                meeting["id"],
                f"任务已取消，可{resume_label}",
            )
            self._log(meeting, f"任务 #{job_id} 已取消，{resume_label}")
            self._sync_metadata(meeting["id"])
            return
        except TaskInterrupted:
            reason = "应用停止时任务被中断，已保留最近断点。"
            self.database.requeue_job(job_id, reason, meeting["id"])
            self._log(
                meeting,
                f"任务 #{job_id} 随应用停止；下次启动会自动继续",
            )
            self._sync_metadata(meeting["id"])
            return
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            self.database.fail_job(job_id, message, meeting["id"])
            self._log(
                meeting,
                f"ERROR: {message}\n{traceback.format_exc(limit=8)}",
            )
            self._sync_metadata(meeting["id"])
            return

        self._log(meeting, f"任务 #{job_id} 处理完成")
        self._sync_metadata(meeting["id"])

    def _process_pipeline(
        self,
        job_id: int,
        meeting: dict[str, Any],
        checkpoint: Checkpoint,
    ) -> None:
        cancel_check = lambda: self._raise_if_canceled(job_id)
        if checkpoint == "uploaded":
            self._raise_if_canceled(job_id)
            if release_ollama_model(self.settings):
                self._log(
                    meeting,
                    "已释放驻留的 Ollama 模型，为语音转写腾出显存",
                )
            self._raise_if_canceled(job_id)
            self._status(
                meeting, "processing", 5, "正在检查并标准化音频"
            )
            source = self.storage.path(
                meeting, f"original{meeting['source_suffix']}"
            )
            normalized = self.storage.path(meeting, "normalized.wav")
            duration = normalize_audio(
                source,
                normalized,
                self.settings,
                cancel_check=cancel_check,
            )
            self.database.update_meeting(
                meeting["id"], duration_seconds=duration
            )
            meeting["duration_seconds"] = duration
            self.database.update_job_checkpoint(job_id, "normalized")
            checkpoint = "normalized"
            self._log(
                meeting, f"音频标准化完成，时长 {duration:.2f} 秒"
            )

        if checkpoint == "normalized":
            self._raise_if_canceled(job_id)
            self._status(
                meeting, "processing", 30, "正在进行语音转写"
            )
            normalized = self.storage.path(meeting, "normalized.wav")
            transcriber = create_transcriber(
                self.settings, meeting["processing_mode"]
            )
            segments = transcriber.transcribe(
                normalized,
                meeting["expected_speakers"],
                meeting["glossary"],
                float(meeting["duration_seconds"] or 0),
                cancel_check=cancel_check,
            )
            self._raise_if_canceled(job_id)
            raw = {
                "version": 1,
                "backend": transcriber.name,
                "created_at": utc_now(),
                "segments": [segment.to_dict() for segment in segments],
            }
            edited = copy.deepcopy(raw)
            edited["source"] = "transcript_raw.json"
            speakers = {
                segment.speaker: ""
                for segment in segments
                if segment.speaker
            }
            self.storage.write_json(meeting, "transcript_raw.json", raw)
            self.storage.write_json(
                meeting, "transcript_edited.json", edited
            )
            self.storage.write_json(meeting, "speakers.json", speakers)
            self.storage.write_text(
                meeting,
                "transcript.md",
                render_transcript_markdown(
                    meeting, edited["segments"], speakers
                ),
            )
            self.storage.write_text(
                meeting,
                "transcript.txt",
                render_transcript_text(
                    meeting, edited["segments"], speakers
                ),
            )
            self.database.update_meeting(
                meeting["id"], transcriber_backend=transcriber.name
            )
            meeting["transcriber_backend"] = transcriber.name
            self.database.update_job_checkpoint(job_id, "transcribed")
            self._log(
                meeting, f"转写完成，共 {len(segments)} 个说话片段"
            )
            self._status(
                meeting, "transcribed", 75, "转写完成，准备生成纪要"
            )

        self._raise_if_canceled(job_id)
        self._generate_minutes(job_id, meeting)

    def _generate_minutes(
        self, job_id: int, meeting: dict[str, Any]
    ) -> None:
        used_ollama = False
        cancel_check = lambda: self._raise_if_canceled(job_id)
        try:
            transcript = self.storage.read_json(
                meeting, "transcript_edited.json"
            )
            if not transcript or not transcript.get("segments"):
                raise RuntimeError("尚无可用于生成纪要的逐字稿。")
            speakers = self.storage.read_json(
                meeting, "speakers.json", default={}
            )
            self._status(
                meeting, "generating_minutes", 85, "正在生成结构化纪要"
            )
            self._raise_if_canceled(job_id)
            external_llm = (
                self.external_llm_store.load()
                if meeting["processing_mode"] != "local"
                else None
            )
            generator = create_minutes_generator(
                self.settings,
                meeting["processing_mode"],
                external_llm=external_llm,
            )
            used_ollama = generator.name == "ollama"
            minutes = generator.generate(
                meeting,
                transcript["segments"],
                speakers,
                cancel_check=cancel_check,
            )
            self._raise_if_canceled(job_id)
            self.storage.write_json(meeting, "minutes.json", minutes)
            self.storage.write_text(
                meeting, "minutes.md", render_minutes_markdown(minutes)
            )
            self.storage.write_text(
                meeting, "minutes.txt", render_minutes_text(minutes)
            )
            render_minutes_docx(
                minutes, self.storage.path(meeting, "minutes.docx")
            )
            self.database.update_meeting(
                meeting["id"], llm_backend=generator.name
            )
            meeting["llm_backend"] = generator.name
            self._raise_if_canceled(job_id)
            self._log(
                meeting, f"纪要生成完成，后端：{generator.name}"
            )
        finally:
            if used_ollama and release_ollama_model(self.settings):
                self._log(
                    meeting,
                    "纪要生成结束，已释放 Ollama 模型和显存",
                )

    def _raise_if_canceled(self, job_id: int) -> None:
        if self.database.job_cancel_requested(job_id):
            raise TaskCanceled("任务已取消")
        if self._stopping.is_set():
            raise TaskInterrupted("应用正在停止")

    def _require_meeting(self, meeting_id: str) -> dict[str, Any]:
        meeting = self.database.get_meeting(meeting_id)
        if meeting is None:
            raise RuntimeError(f"会议 {meeting_id} 不存在")
        return meeting

    def _status(
        self,
        meeting: dict[str, Any],
        status: str,
        progress: int,
        current_step: str,
    ) -> None:
        self.database.update_meeting(
            meeting["id"],
            status=status,
            progress=progress,
            current_step=current_step,
            error=None,
        )
        meeting.update(
            {
                "status": status,
                "progress": progress,
                "current_step": current_step,
                "error": None,
            }
        )
        self._log(meeting, f"[{progress:3d}%] {current_step}")

    def _log(self, meeting: dict[str, Any], message: str) -> None:
        self.storage.append_log(meeting, f"{utc_now()} {message}")

    def _sync_metadata(self, meeting_id: str) -> None:
        current = self.database.get_meeting(meeting_id)
        if current is not None:
            self.storage.write_json(current, "meeting.json", current)
