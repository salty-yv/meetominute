from __future__ import annotations

import asyncio
import copy
import traceback
from pathlib import Path
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
)
from .storage import MeetingStorage


TaskKind = Literal["pipeline", "minutes"]


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
        self._queue: asyncio.Queue[tuple[TaskKind, str] | None] = asyncio.Queue()
        self._queued: set[tuple[TaskKind, str]] = set()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = asyncio.create_task(
            self._run(), name="meetominute-single-worker"
        )
        for meeting_id in self.database.recover_interrupted():
            await self.enqueue("pipeline", meeting_id)

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def enqueue(self, kind: TaskKind, meeting_id: str) -> bool:
        item = (kind, meeting_id)
        if item in self._queued:
            return False
        self._queued.add(item)
        await self._queue.put(item)
        return True

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            kind, meeting_id = item
            try:
                if kind == "pipeline":
                    await asyncio.to_thread(self._process_pipeline, meeting_id)
                else:
                    await asyncio.to_thread(self._generate_minutes, meeting_id)
            finally:
                self._queued.discard(item)
                self._queue.task_done()

    def _process_pipeline(self, meeting_id: str) -> None:
        meeting = self._require_meeting(meeting_id)
        try:
            if release_ollama_model(self.settings):
                self._log(
                    meeting,
                    "已释放驻留的 Ollama 模型，为语音转写腾出显存",
                )
            self._status(
                meeting, "processing", 5, "正在检查并标准化音频"
            )
            source = self.storage.path(
                meeting, f"original{meeting['source_suffix']}"
            )
            normalized = self.storage.path(meeting, "normalized.wav")
            duration = normalize_audio(source, normalized, self.settings)
            self.database.update_meeting(
                meeting_id, duration_seconds=duration
            )
            meeting["duration_seconds"] = duration
            self._log(meeting, f"音频标准化完成，时长 {duration:.2f} 秒")

            self._status(meeting, "processing", 30, "正在进行语音转写")
            transcriber = create_transcriber(
                self.settings, meeting["processing_mode"]
            )
            segments = transcriber.transcribe(
                normalized,
                meeting["expected_speakers"],
                meeting["glossary"],
                duration,
            )
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
            self.storage.write_json(meeting, "transcript_edited.json", edited)
            self.storage.write_json(meeting, "speakers.json", speakers)
            self.storage.write_text(
                meeting,
                "transcript.md",
                render_transcript_markdown(
                    meeting, edited["segments"], speakers
                ),
            )
            self.database.update_meeting(
                meeting_id, transcriber_backend=transcriber.name
            )
            self._log(
                meeting, f"转写完成，共 {len(segments)} 个说话片段"
            )
            self._status(
                meeting, "transcribed", 75, "转写完成，正在生成纪要"
            )
            self._generate_minutes(meeting_id)
        except Exception as exc:
            self._fail(meeting, exc)

    def _generate_minutes(self, meeting_id: str) -> None:
        meeting = self._require_meeting(meeting_id)
        used_ollama = False
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
                meeting, transcript["segments"], speakers
            )
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
                meeting_id, llm_backend=generator.name
            )
            self._status(meeting, "completed", 100, "处理完成")
            self._log(meeting, f"纪要生成完成，后端：{generator.name}")
            self._sync_metadata(meeting_id)
        except Exception as exc:
            self._fail(meeting, exc)
        finally:
            if used_ollama and release_ollama_model(self.settings):
                self._log(
                    meeting,
                    "纪要生成完成，已释放 Ollama 模型和显存",
                )

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
        self._log(meeting, f"[{progress:3d}%] {current_step}")

    def _fail(self, meeting: dict[str, Any], error: Exception) -> None:
        message = str(error).strip() or error.__class__.__name__
        self.database.update_meeting(
            meeting["id"],
            status="failed",
            current_step="处理失败",
            error=message[:4000],
        )
        self._log(
            meeting,
            f"ERROR: {message}\n{traceback.format_exc(limit=8)}",
        )
        self._sync_metadata(meeting["id"])

    def _log(self, meeting: dict[str, Any], message: str) -> None:
        self.storage.append_log(meeting, f"{utc_now()} {message}")

    def _sync_metadata(self, meeting_id: str) -> None:
        current = self.database.get_meeting(meeting_id)
        if current is not None:
            self.storage.write_json(current, "meeting.json", current)
