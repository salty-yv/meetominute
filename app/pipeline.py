from __future__ import annotations

import asyncio
import copy
import logging
import shutil
import threading
import traceback
from pathlib import Path
from typing import Any, Literal

from .actions import (
    ActionItemsError,
    minutes_file_lock,
    reconcile_action_items,
)
from .audio import normalize_audio
from .config import Settings
from .database import Database
from .domain import utc_now
from .external_llm import ExternalLLMConfigStore
from .maintenance import MaintenanceGate
from .minutes_templates import DEFAULT_TEMPLATE_ID
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
InfrastructureRetryState = Literal["enqueue", "retry", "terminal"]

LOGGER = logging.getLogger(__name__)

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
        maintenance_gate: MaintenanceGate | None = None,
        infrastructure_retry_limit: int = 3,
        infrastructure_retry_delay: float = 0.25,
    ):
        self.settings = settings
        self.database = database
        self.storage = storage
        self.external_llm_store = external_llm_store
        self.maintenance_gate = maintenance_gate
        self._queue: asyncio.Queue[int | None] = asyncio.Queue()
        self._queued: set[int] = set()
        self._worker: asyncio.Task[None] | None = None
        self._stopping = threading.Event()
        self._infrastructure_retry_limit = max(
            0, int(infrastructure_retry_limit)
        )
        self._infrastructure_retry_delay = max(
            0.0, float(infrastructure_retry_delay)
        )
        self._infrastructure_retry_attempts: dict[int, int] = {}
        self._infrastructure_retry_tasks: set[
            asyncio.Task[None]
        ] = set()

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
        retry_tasks = list(self._infrastructure_retry_tasks)
        for task in retry_tasks:
            task.cancel()
        if retry_tasks:
            await asyncio.gather(*retry_tasks, return_exceptions=True)
        self._infrastructure_retry_tasks.clear()
        await self._queue.put(None)
        await self._worker
        self._worker = None
        self._queued.clear()
        self._infrastructure_retry_attempts.clear()

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
            retry_required = False
            try:
                try:
                    retry_required = await asyncio.to_thread(
                        self._execute_job, job_id
                    )
                except Exception:
                    retry_required = True
                    LOGGER.exception(
                        "Task %s escaped its execution boundary", job_id
                    )
            finally:
                self._queued.discard(job_id)
                self._queue.task_done()
            if retry_required:
                self._schedule_infrastructure_retry(job_id)
            else:
                self._infrastructure_retry_attempts.pop(job_id, None)

    def _execute_job(self, job_id: int) -> bool:
        if self.maintenance_gate is None:
            return self._execute_job_with_mutation_registered(job_id)
        with self.maintenance_gate.mutation(wait=True):
            return self._execute_job_with_mutation_registered(job_id)

    def _execute_job_with_mutation_registered(self, job_id: int) -> bool:
        job: dict[str, Any] | None = None
        meeting: dict[str, Any] | None = None
        try:
            if not self.database.claim_job(job_id):
                return False
            job = self.database.get_job(job_id)
            if job is None:
                return False
            meeting = self._require_meeting(str(job["meeting_id"]))
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
            assert job is not None and meeting is not None
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
            return False
        except TaskInterrupted:
            assert meeting is not None
            reason = "应用停止时任务被中断，已保留最近断点。"
            self.database.requeue_job(job_id, reason, meeting["id"])
            self._log(
                meeting,
                f"任务 #{job_id} 随应用停止；下次启动会自动继续",
            )
            self._sync_metadata(meeting["id"])
            return False
        except Exception as exc:
            failure_recorded = self._record_job_failure(
                job_id, exc, job, meeting
            )
            return not failure_recorded

        self._log(meeting, f"任务 #{job_id} 处理完成")
        self._sync_metadata(meeting["id"])

        return False

    def _record_job_failure(
        self,
        job_id: int,
        exc: Exception,
        job: dict[str, Any] | None = None,
        meeting: dict[str, Any] | None = None,
    ) -> bool:
        message = str(exc).strip() or exc.__class__.__name__
        trace = "".join(
            traceback.format_exception(
                exc.__class__, exc, exc.__traceback__, limit=8
            )
        )

        if job is None:
            try:
                job = self.database.get_job(job_id)
            except Exception:
                LOGGER.exception(
                    "Could not reload failed task %s", job_id
                )

        meeting_id = (
            str(job["meeting_id"])
            if job is not None and job.get("meeting_id") is not None
            else None
        )
        if meeting is None and meeting_id is not None:
            try:
                meeting = self.database.get_meeting(meeting_id)
            except Exception:
                LOGGER.exception(
                    "Could not reload meeting for failed task %s", job_id
                )

        failure_recorded = True
        try:
            self.database.fail_job(job_id, message, meeting_id)
        except Exception:
            failure_recorded = False
            LOGGER.exception("Could not mark task %s as failed", job_id)

        if meeting is None:
            LOGGER.error(
                "Task %s failed before its meeting could be loaded: %s",
                job_id,
                message,
            )
            return failure_recorded

        try:
            self._log(meeting, f"ERROR: {message}\n{trace}")
        except Exception:
            LOGGER.exception(
                "Could not append the failure log for task %s", job_id
            )
        try:
            self._sync_metadata(meeting["id"])
        except Exception:
            LOGGER.exception(
                "Could not sync failure metadata for task %s", job_id
            )
        return failure_recorded

    def _schedule_infrastructure_retry(self, job_id: int) -> None:
        if self._stopping.is_set():
            return

        previous_attempts = self._infrastructure_retry_attempts.get(
            job_id, 0
        )
        if previous_attempts >= self._infrastructure_retry_limit:
            task = asyncio.create_task(
                self._finish_exhausted_infrastructure_retry(
                    job_id, previous_attempts
                ),
                name=f"meetominute-retry-exhausted-{job_id}",
            )
            self._track_infrastructure_retry_task(task)
            return

        attempt = previous_attempts + 1
        self._infrastructure_retry_attempts[job_id] = attempt
        delay = min(
            self._infrastructure_retry_delay * (2 ** (attempt - 1)),
            5.0,
        )
        LOGGER.warning(
            "Task %s infrastructure retry %s/%s scheduled in %.2fs",
            job_id,
            attempt,
            self._infrastructure_retry_limit,
            delay,
        )
        task = asyncio.create_task(
            self._retry_job_after_delay(job_id, attempt, delay),
            name=f"meetominute-infrastructure-retry-{job_id}-{attempt}",
        )
        self._track_infrastructure_retry_task(task)

    def _track_infrastructure_retry_task(
        self, task: asyncio.Task[None]
    ) -> None:
        self._infrastructure_retry_tasks.add(task)
        task.add_done_callback(
            self._infrastructure_retry_tasks.discard
        )

    async def _retry_job_after_delay(
        self, job_id: int, attempt: int, delay: float
    ) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            if self._stopping.is_set():
                return
            reason = (
                "基础设施异常自动恢复，"
                f"准备第 {attempt}/{self._infrastructure_retry_limit} 次重试"
            )
            state = await asyncio.to_thread(
                self._prepare_job_for_infrastructure_retry,
                job_id,
                reason,
            )
            if self._stopping.is_set():
                return
            if state == "enqueue":
                await self._queue_job(job_id)
            elif state == "retry":
                self._schedule_infrastructure_retry(job_id)
            else:
                self._infrastructure_retry_attempts.pop(job_id, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "Task %s infrastructure retry preparation failed",
                job_id,
            )
            self._schedule_infrastructure_retry(job_id)

    async def _finish_exhausted_infrastructure_retry(
        self, job_id: int, attempts: int
    ) -> None:
        message = (
            f"基础设施自动重试已达到上限（{attempts} 次）；"
            "任务保持排队状态，重启应用后仍可恢复。"
        )
        try:
            state = await asyncio.to_thread(
                self._prepare_job_for_infrastructure_retry,
                job_id,
                message,
            )
            if state != "terminal":
                await asyncio.to_thread(
                    self._record_infrastructure_retry_diagnostic,
                    job_id,
                    message,
                )
            LOGGER.error("Task %s: %s", job_id, message)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "Could not finalize exhausted retries for task %s",
                job_id,
            )
        finally:
            self._infrastructure_retry_attempts.pop(job_id, None)

    def _prepare_job_for_infrastructure_retry(
        self, job_id: int, reason: str
    ) -> InfrastructureRetryState:
        if self.maintenance_gate is None:
            return self._prepare_job_for_infrastructure_retry_registered(
                job_id, reason
            )
        with self.maintenance_gate.mutation(wait=True):
            return self._prepare_job_for_infrastructure_retry_registered(
                job_id, reason
            )

    def _prepare_job_for_infrastructure_retry_registered(
        self, job_id: int, reason: str
    ) -> InfrastructureRetryState:
        try:
            job = self.database.get_job(job_id)
        except Exception:
            LOGGER.exception(
                "Could not inspect task %s before infrastructure retry",
                job_id,
            )
            return "retry"
        if job is None:
            return "terminal"

        status = str(job.get("status") or "")
        if status not in {"queued", "running"}:
            return "terminal"
        meeting_id = (
            str(job["meeting_id"])
            if job.get("meeting_id") is not None
            else None
        )
        if job.get("cancel_requested"):
            try:
                self.database.mark_job_canceled(job_id, meeting_id)
            except Exception:
                LOGGER.exception(
                    "Could not cancel task %s during retry recovery",
                    job_id,
                )
                return "retry"
            return "terminal"

        try:
            self.database.requeue_job(job_id, reason, meeting_id)
        except Exception:
            LOGGER.exception(
                "Could not restore task %s to queued state", job_id
            )
            return "retry"
        return "enqueue"

    def _record_infrastructure_retry_diagnostic(
        self, job_id: int, message: str
    ) -> None:
        try:
            job = self.database.get_job(job_id)
            meeting = (
                self.database.get_meeting(str(job["meeting_id"]))
                if job is not None and job.get("meeting_id") is not None
                else None
            )
            if meeting is None:
                return
            self._log(meeting, f"ERROR: {message}")
            self._sync_metadata(meeting["id"])
        except Exception:
            LOGGER.exception(
                "Could not persist retry diagnostics for task %s",
                job_id,
            )

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
            try:
                previous_minutes = self.storage.read_json(
                    meeting, "minutes.json", default=None
                )
            except (OSError, ValueError):
                backup = self._preserve_invalid_minutes(
                    job_id, meeting
                )
                self._log(
                    meeting,
                    "已有纪要损坏，已保留副本 "
                    f"{backup.name}；本次将重建纪要。",
                )
                previous_minutes = None
            if previous_minutes is not None and not isinstance(
                previous_minutes, dict
            ):
                backup = self._preserve_invalid_minutes(
                    job_id, meeting
                )
                self._log(
                    meeting,
                    "已有纪要格式无效，已保留副本 "
                    f"{backup.name}；本次将重建纪要。",
                )
                previous_minutes = None
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
            template_id = str(
                meeting.get("minutes_template_id")
                or DEFAULT_TEMPLATE_ID
            )
            minutes_template = self.database.get_minutes_template(
                template_id
            )
            if minutes_template is None:
                template_id = DEFAULT_TEMPLATE_ID
                minutes_template = self.database.get_minutes_template(
                    template_id
                )
                if minutes_template is None:
                    raise RuntimeError("默认纪要模板不存在")
                self.database.update_meeting(
                    meeting["id"],
                    minutes_template_id=template_id,
                )
                meeting["minutes_template_id"] = template_id
            used_ollama = generator.name == "ollama"
            minutes = generator.generate(
                meeting,
                transcript["segments"],
                speakers,
                template=minutes_template,
                cancel_check=cancel_check,
            )
            self._raise_if_canceled(job_id)
            with minutes_file_lock(self.storage, meeting):
                try:
                    latest_minutes = self.storage.read_json(
                        meeting,
                        "minutes.json",
                        default=previous_minutes,
                    )
                except (OSError, ValueError):
                    backup = self._preserve_invalid_minutes(
                        job_id, meeting
                    )
                    self._log(
                        meeting,
                        "写入前发现已有纪要损坏，已保留副本 "
                        f"{backup.name}。",
                    )
                    latest_minutes = None
                if latest_minutes is not None and not isinstance(
                    latest_minutes, dict
                ):
                    backup = self._preserve_invalid_minutes(
                        job_id, meeting
                    )
                    self._log(
                        meeting,
                        "写入前发现已有纪要格式无效，已保留副本 "
                        f"{backup.name}。",
                    )
                    latest_minutes = None
                try:
                    minutes = reconcile_action_items(
                        str(meeting["id"]),
                        minutes,
                        latest_minutes,
                    )
                except ActionItemsError as exc:
                    if latest_minutes is None:
                        raise RuntimeError(
                            f"新纪要待办数据无效：{exc}"
                        ) from exc
                    backup = self._preserve_invalid_minutes(
                        job_id, meeting
                    )
                    try:
                        minutes = reconcile_action_items(
                            str(meeting["id"]),
                            minutes,
                            None,
                        )
                    except ActionItemsError as generated_exc:
                        raise RuntimeError(
                            f"新纪要待办数据无效：{generated_exc}"
                        ) from generated_exc
                    self._log(
                        meeting,
                        "已有纪要的待办结构无效，已保留副本 "
                        f"{backup.name}；无法继承旧状态。",
                    )
                self.storage.write_json(
                    meeting, "minutes.json", minutes
                )
                self.storage.write_text(
                    meeting,
                    "minutes.md",
                    render_minutes_markdown(minutes),
                )
                self.storage.write_text(
                    meeting,
                    "minutes.txt",
                    render_minutes_text(minutes),
                )
                render_minutes_docx(
                    minutes,
                    self.storage.path(meeting, "minutes.docx"),
                )
            self.database.update_meeting(
                meeting["id"], llm_backend=generator.name
            )
            meeting["llm_backend"] = generator.name
            self._raise_if_canceled(job_id)
            self._log(
                meeting,
                "纪要生成完成，"
                f"模板：{minutes_template['name']}，后端：{generator.name}",
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

    def _preserve_invalid_minutes(
        self,
        job_id: int,
        meeting: dict[str, Any],
    ) -> Path:
        source = self.storage.path(meeting, "minutes.json")
        destination = self.storage.path(
            meeting, f"minutes.invalid-{job_id}.json"
        )
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
        return destination

    def _sync_metadata(self, meeting_id: str) -> None:
        current = self.database.get_meeting(meeting_id)
        if current is not None:
            self.storage.write_json(current, "meeting.json", current)
