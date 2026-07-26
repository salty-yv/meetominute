from __future__ import annotations

import asyncio
import calendar as calendar_module
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .backups import BackupError, BackupManager
from .config import Settings
from .database import Database
from .diagnostics import run_diagnostics
from .domain import format_timestamp, utc_now
from .external_llm import (
    ExternalLLMConfigStore,
    test_external_llm_connection,
)
from .pipeline import CHECKPOINT_RESUME_LABELS, TaskQueue
from .rendering import render_transcript_markdown, render_transcript_text
from .storage import (
    SAFE_SUFFIXES,
    MeetingStorage,
    UploadTooLargeError,
    make_slug,
)


STATUS_LABELS = {
    "queued": "等待处理",
    "processing": "处理中",
    "canceling": "正在取消",
    "canceled": "已取消",
    "transcribed": "转写已完成",
    "generating_minutes": "正在生成纪要",
    "completed": "已完成",
    "failed": "失败",
}
ACTIVE_STATUSES = {
    "queued",
    "processing",
    "generating_minutes",
    "canceling",
}
DOWNLOAD_FILES = {
    "md": ("minutes.md", "text/markdown; charset=utf-8"),
    "txt": ("minutes.txt", "text/plain; charset=utf-8"),
    "docx": (
        "minutes.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "json": ("minutes.json", "application/json; charset=utf-8"),
}


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    database = Database(config.db_path)
    storage = MeetingStorage(config)
    external_llm_store = ExternalLLMConfigStore(
        config.data_dir / "external-llm.json", config
    )
    task_queue = TaskQueue(
        config, database, storage, external_llm_store
    )
    backup_manager = BackupManager(config, database, storage)
    templates = Jinja2Templates(directory=config.base_dir / "app" / "templates")
    templates.env.filters["timestamp"] = format_timestamp
    templates.env.globals["status_labels"] = STATUS_LABELS
    templates.env.globals["active_statuses"] = ACTIVE_STATUSES

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        config.ensure_directories()
        database.initialize()
        backup_manager.ensure_directory()
        await task_queue.start()
        try:
            yield
        finally:
            await task_queue.stop()

    application = FastAPI(
        title="MeetOminute",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = config
    application.state.database = database
    application.state.storage = storage
    application.state.task_queue = task_queue
    application.state.external_llm_store = external_llm_store
    application.state.backup_manager = backup_manager
    application.mount(
        "/static",
        StaticFiles(directory=config.base_dir / "app" / "static"),
        name="static",
    )

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "meetings": database.list_meetings(),
                "lifecycle_counts": database.lifecycle_counts(),
                "today": date.today().isoformat(),
                "max_upload_mb": config.max_upload_bytes // 1024 // 1024,
                "local_transcriber": config.local_transcriber,
                "local_llm": config.local_llm,
                "ollama_model": config.ollama_model,
                "external_llm": external_llm_store.load().public_dict(),
            },
        )

    @application.get("/archive", response_class=HTMLResponse)
    async def archive_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="library.html",
            context={
                "view": "archived",
                "meetings": database.list_meetings(
                    lifecycle_state="archived"
                ),
                "lifecycle_counts": database.lifecycle_counts(),
            },
        )

    @application.get("/calendar", response_class=HTMLResponse)
    async def calendar_page(
        request: Request,
        month: str | None = None,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="calendar.html",
            context=_calendar_context(
                database,
                month,
                today=date.today(),
            ),
        )

    @application.get("/trash", response_class=HTMLResponse)
    async def trash_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="library.html",
            context={
                "view": "trashed",
                "meetings": database.list_meetings(
                    lifecycle_state="trashed"
                ),
                "lifecycle_counts": database.lifecycle_counts(),
            },
        )

    @application.get("/backups", response_class=HTMLResponse)
    async def backups_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="backups.html",
            context=_backups_context(request, backup_manager),
        )

    @application.post("/backups", response_class=HTMLResponse)
    async def create_backup(request: Request) -> Response:
        try:
            backup = await asyncio.to_thread(
                backup_manager.create_backup
            )
        except (BackupError, OSError) as exc:
            return templates.TemplateResponse(
                request=request,
                name="backups.html",
                context={
                    **_backups_context(request, backup_manager),
                    "backup_error": str(exc),
                },
                status_code=409,
            )
        return RedirectResponse(
            request.url_for("backups_page").include_query_params(
                created=backup.name
            ),
            status_code=303,
        )

    @application.post(
        "/backups/restore", response_class=HTMLResponse
    )
    async def restore_backup(
        request: Request,
        backup_file: UploadFile = File(...),
    ) -> Response:
        temporary: Path | None = None
        try:
            temporary = await backup_manager.save_uploaded_backup(
                backup_file
            )
            result = await asyncio.to_thread(
                backup_manager.restore_backup, temporary
            )
        except (BackupError, UploadTooLargeError, OSError) as exc:
            return templates.TemplateResponse(
                request=request,
                name="backups.html",
                context={
                    **_backups_context(request, backup_manager),
                    "restore_error": str(exc),
                },
                status_code=422,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return RedirectResponse(
            request.url_for("backups_page").include_query_params(
                imported=str(result.imported),
                skipped=str(result.skipped),
            ),
            status_code=303,
        )

    @application.get("/backups/{backup_name}")
    async def download_backup(backup_name: str) -> FileResponse:
        source = backup_manager.get_backup(backup_name)
        if source is None:
            raise HTTPException(404, "备份文件不存在")
        return FileResponse(
            source,
            media_type="application/zip",
            filename=source.name,
        )

    @application.get(
        "/settings/external-llm", response_class=HTMLResponse
    )
    async def external_llm_settings(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="external_llm.html",
            context={
                "external_llm": external_llm_store.load().public_dict(),
                "saved": request.query_params.get("saved") == "1",
                "form_error": "",
            },
        )

    @application.post(
        "/settings/external-llm", response_class=HTMLResponse
    )
    async def save_external_llm_settings(
        request: Request,
        enabled: str | None = Form(None),
        provider_name: str = Form("OpenAI Compatible"),
        base_url: str = Form(""),
        model: str = Form(""),
        api_key: str = Form(""),
        clear_api_key: str | None = Form(None),
        reasoning_effort: str = Form(""),
    ) -> Response:
        try:
            external_llm_store.save(
                enabled=enabled == "on",
                provider_name=provider_name,
                base_url=base_url,
                model=model,
                api_key=api_key,
                clear_api_key=clear_api_key == "on",
                reasoning_effort=reasoning_effort,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            current = external_llm_store.load().public_dict()
            current.update(
                {
                    "enabled": enabled == "on",
                    "provider_name": provider_name,
                    "base_url": base_url,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                }
            )
            return templates.TemplateResponse(
                request=request,
                name="external_llm.html",
                context={
                    "external_llm": current,
                    "saved": False,
                    "form_error": str(exc),
                },
                status_code=422,
            )
        return RedirectResponse(
            request.url_for("external_llm_settings").include_query_params(
                saved="1"
            ),
            status_code=303,
        )

    @application.post("/settings/external-llm/test")
    async def test_external_llm_settings(
        enabled: str | None = Form(None),
        provider_name: str = Form("OpenAI Compatible"),
        base_url: str = Form(""),
        model: str = Form(""),
        api_key: str = Form(""),
        reasoning_effort: str = Form(""),
    ) -> JSONResponse:
        try:
            candidate = external_llm_store.resolve_form_config(
                enabled=enabled == "on",
                provider_name=provider_name,
                base_url=base_url,
                model=model,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
            result = await asyncio.to_thread(
                test_external_llm_connection,
                candidate,
                min(config.request_timeout_seconds, 30),
            )
            return JSONResponse(result.to_dict())
        except (RuntimeError, ValueError, OSError) as exc:
            return JSONResponse(
                {
                    "status": "error",
                    "message": str(exc),
                    "latency_ms": 0,
                    "model_count": None,
                },
                status_code=422,
            )

    @application.get("/api/settings/external-llm")
    async def external_llm_settings_api() -> dict[str, Any]:
        return external_llm_store.load().public_dict()

    @application.get("/diagnostics", response_class=HTMLResponse)
    async def diagnostics_page(request: Request) -> HTMLResponse:
        report = await asyncio.to_thread(run_diagnostics, config, database)
        return templates.TemplateResponse(
            request=request,
            name="diagnostics.html",
            context={"report": report},
        )

    @application.get("/api/diagnostics")
    async def diagnostics_api() -> dict[str, Any]:
        report = await asyncio.to_thread(run_diagnostics, config, database)
        return report.to_dict()

    @application.post("/meetings")
    async def create_meeting(
        request: Request,
        title: str = Form(...),
        meeting_date: date = Form(...),
        expected_speakers: int = Form(...),
        glossary: str = Form(""),
        processing_mode: str = Form("local"),
        recording: UploadFile = File(...),
    ) -> RedirectResponse:
        clean_title = title.strip()
        if not clean_title:
            raise HTTPException(422, "会议名称不能为空")
        if not 1 <= expected_speakers <= 50:
            raise HTTPException(422, "预计发言人数必须在 1 到 50 之间")
        if processing_mode not in {"local", "mixed", "cloud"}:
            raise HTTPException(422, "无效的处理模式")
        original_name = _safe_display_filename(recording.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix not in SAFE_SUFFIXES:
            allowed = "、".join(sorted(SAFE_SUFFIXES))
            raise HTTPException(415, f"文件格式不支持；允许：{allowed}")

        meeting_id = uuid4().hex
        now = utc_now()
        record: dict[str, Any] = {
            "id": meeting_id,
            "slug": make_slug(meeting_date, clean_title, meeting_id[:8]),
            "title": clean_title[:200],
            "meeting_date": meeting_date.isoformat(),
            "expected_speakers": expected_speakers,
            "glossary": glossary.strip()[:20_000],
            "processing_mode": processing_mode,
            "source_filename": original_name,
            "source_suffix": suffix,
            "status": "queued",
            "progress": 0,
            "current_step": "等待后台处理",
            "error": None,
            "duration_seconds": None,
            "transcriber_backend": None,
            "llm_backend": None,
            "created_at": now,
            "updated_at": now,
        }
        directory = storage.prepare(record)
        try:
            await storage.save_upload(
                recording, directory / f"original{suffix}"
            )
        except UploadTooLargeError as exc:
            directory.rmdir()
            raise HTTPException(413, str(exc)) from exc
        except Exception:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
            raise

        try:
            database.create_meeting(record)
            storage.write_json(record, "meeting.json", record)
            storage.write_text(record, "glossary.txt", record["glossary"])
            storage.append_log(record, f"{now} 已接收文件 {original_name}")
        except Exception:
            # 保留已上传的原始录音，避免数据库异常导致用户数据丢失。
            raise
        await task_queue.enqueue(
            "pipeline", meeting_id, checkpoint="uploaded"
        )
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.get(
        "/meetings/{meeting_id}", response_class=HTMLResponse
    )
    async def meeting_detail(
        request: Request, meeting_id: str
    ) -> HTMLResponse:
        meeting = _meeting_or_404(database, meeting_id)
        return templates.TemplateResponse(
            request=request,
            name="meeting.html",
            context=_meeting_context(
                meeting, storage, database, task_queue
            ),
        )

    @application.get(
        "/meetings/{meeting_id}/status", response_class=HTMLResponse
    )
    async def meeting_status(
        request: Request, meeting_id: str
    ) -> HTMLResponse:
        meeting = _meeting_or_404(database, meeting_id)
        return templates.TemplateResponse(
            request=request,
            name="_status.html",
            context=_task_context(meeting, database, task_queue),
        )

    @application.post("/meetings/{meeting_id}/archive")
    async def archive_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        _ensure_lifecycle_change_allowed(meeting)
        if not database.archive_meeting(meeting_id):
            raise HTTPException(409, "会议当前无法归档")
        storage.append_log(meeting, f"{utc_now()} 用户归档了会议")
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/unarchive")
    async def unarchive_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if not database.unarchive_meeting(meeting_id):
            raise HTTPException(409, "会议当前不在归档中")
        storage.append_log(meeting, f"{utc_now()} 用户取消归档")
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/trash")
    async def trash_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        _ensure_lifecycle_change_allowed(meeting)
        if not database.trash_meeting(meeting_id):
            raise HTTPException(409, "会议当前无法移入回收站")
        storage.append_log(meeting, f"{utc_now()} 用户将会议移入回收站")
        return RedirectResponse(
            request.url_for("trash_page"),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/restore")
    async def restore_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(
            database, meeting_id, include_trashed=True
        )
        if not database.restore_meeting(meeting_id):
            raise HTTPException(409, "会议当前不在回收站")
        storage.append_log(meeting, f"{utc_now()} 用户从回收站恢复会议")
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/delete")
    async def permanently_delete_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(
            database, meeting_id, include_trashed=True
        )
        if meeting["lifecycle_state"] != "trashed":
            raise HTTPException(409, "会议必须先移入回收站")
        staged = storage.stage_permanent_delete(meeting)
        try:
            deleted = database.delete_trashed_meeting(meeting_id)
            if not deleted:
                storage.restore_staged_delete(meeting, staged)
                raise HTTPException(409, "会议当前无法永久删除")
        except Exception:
            if database.get_meeting(meeting_id) is not None:
                storage.restore_staged_delete(meeting, staged)
            raise
        await asyncio.to_thread(storage.purge_staged_delete, staged)
        return RedirectResponse(
            request.url_for("trash_page"),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/cancel")
    async def cancel_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        _meeting_or_404(database, meeting_id)
        result = task_queue.cancel(meeting_id)
        if result is None:
            raise HTTPException(409, "当前没有可取消的任务")
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/resume")
    async def resume_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "任务当前正在处理，不能重复加入队列")
        _ensure_meeting_editable(meeting)
        if meeting["status"] not in {"failed", "canceled"}:
            raise HTTPException(409, "当前任务不需要断点续跑")
        kind, checkpoint, progress, label = task_queue.resume_info(meeting)
        database.update_meeting(
            meeting_id,
            status="queued",
            progress=progress,
            current_step=f"已排队，{label}",
            error=None,
        )
        storage.append_log(
            meeting, f"{utc_now()} 用户请求断点续跑：{label}"
        )
        await task_queue.enqueue(
            kind, meeting_id, checkpoint=checkpoint
        )
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/retry")
    async def retry_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "任务当前正在处理，不能重复加入队列")
        _ensure_meeting_editable(meeting)
        storage.remove_generated_files(meeting)
        database.update_meeting(
            meeting_id,
            status="queued",
            progress=0,
            current_step="等待重新处理",
            error=None,
            duration_seconds=None,
            transcriber_backend=None,
            llm_backend=None,
        )
        storage.append_log(meeting, f"{utc_now()} 用户请求重新处理")
        await task_queue.enqueue(
            "pipeline", meeting_id, checkpoint="uploaded"
        )
        return RedirectResponse(
            request.url_for("meeting_detail", meeting_id=meeting_id),
            status_code=303,
        )

    @application.post("/meetings/{meeting_id}/speakers")
    async def update_speakers(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "任务正在处理，请稍后再编辑")
        _ensure_meeting_editable(meeting)
        transcript = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        if not transcript.get("segments"):
            raise HTTPException(409, "逐字稿尚未生成")
        current = storage.read_json(meeting, "speakers.json", default={})
        form = await request.form()
        speakers = {
            key: str(form.get(f"name_{key}", "")).strip()[:100]
            for key in current
        }
        storage.write_json(meeting, "speakers.json", speakers)
        _save_transcript_exports(storage, meeting, transcript, speakers)
        _mark_transcript_changed(database, storage, meeting)
        target = str(
            request.url_for("meeting_detail", meeting_id=meeting_id)
        )
        return RedirectResponse(target + "#transcript", status_code=303)

    @application.post("/meetings/{meeting_id}/transcript")
    async def update_transcript(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "任务正在处理，请稍后再编辑")
        _ensure_meeting_editable(meeting)
        transcript = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        segments = transcript.get("segments") or []
        if not segments:
            raise HTTPException(409, "逐字稿尚未生成")
        speakers = storage.read_json(meeting, "speakers.json", default={})
        form = await request.form()
        known_speakers = set(speakers)
        for segment in segments:
            segment_id = segment["id"]
            text_value = str(form.get(f"text_{segment_id}", "")).strip()
            speaker_value = str(
                form.get(f"speaker_{segment_id}", segment["speaker"])
            )
            if speaker_value not in known_speakers:
                raise HTTPException(422, "说话人标签无效")
            segment["text"] = text_value[:100_000]
            segment["speaker"] = speaker_value
        transcript["edited_at"] = utc_now()
        storage.write_json(meeting, "transcript_edited.json", transcript)
        _save_transcript_exports(storage, meeting, transcript, speakers)
        _mark_transcript_changed(database, storage, meeting)
        target = str(
            request.url_for("meeting_detail", meeting_id=meeting_id)
        )
        return RedirectResponse(target + "#transcript", status_code=303)

    @application.post("/meetings/{meeting_id}/minutes")
    async def generate_minutes(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "当前已有任务正在处理")
        _ensure_meeting_editable(meeting)
        transcript = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        if not transcript.get("segments"):
            raise HTTPException(409, "逐字稿尚未生成")
        database.update_meeting(
            meeting_id,
            status="queued",
            progress=75,
            current_step="纪要已进入队列",
            error=None,
        )
        await task_queue.enqueue(
            "minutes", meeting_id, checkpoint="transcribed"
        )
        target = str(
            request.url_for("meeting_detail", meeting_id=meeting_id)
        )
        return RedirectResponse(target + "#minutes", status_code=303)

    @application.get("/meetings/{meeting_id}/media")
    async def meeting_media(meeting_id: str) -> FileResponse:
        meeting = _meeting_or_404(database, meeting_id)
        source = storage.path(
            meeting, f"original{meeting['source_suffix']}"
        )
        if not source.exists():
            raise HTTPException(404, "原始录音不存在")
        return FileResponse(
            source,
            filename=meeting["source_filename"],
            content_disposition_type="inline",
        )

    @application.get("/meetings/{meeting_id}/transcript/download")
    async def download_transcript(meeting_id: str) -> FileResponse:
        meeting = _meeting_or_404(database, meeting_id)
        transcript = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        if not transcript.get("segments"):
            raise HTTPException(404, "逐字稿尚未生成")
        speakers = storage.read_json(meeting, "speakers.json", default={})
        source = _save_transcript_exports(
            storage, meeting, transcript, speakers
        )
        safe_title = _download_title(meeting["title"])
        return FileResponse(
            source,
            media_type="text/plain; charset=utf-8",
            filename=f"{safe_title}_录音转写.txt",
        )

    @application.get("/meetings/{meeting_id}/download/{kind}")
    async def download_minutes(meeting_id: str, kind: str) -> FileResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if kind not in DOWNLOAD_FILES:
            raise HTTPException(404, "不支持的导出格式")
        filename, media_type = DOWNLOAD_FILES[kind]
        source = storage.path(meeting, filename)
        if not source.exists():
            raise HTTPException(404, "纪要尚未生成")
        safe_title = _download_title(meeting["title"])
        return FileResponse(
            source,
            media_type=media_type,
            filename=f"{safe_title}_会议纪要.{kind}",
        )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": application.version}

    return application


def _meeting_or_404(
    database: Database,
    meeting_id: str,
    *,
    include_trashed: bool = False,
) -> dict[str, Any]:
    meeting = database.get_meeting(meeting_id)
    if meeting is None or (
        meeting.get("lifecycle_state") == "trashed"
        and not include_trashed
    ):
        raise HTTPException(404, "会议不存在")
    return meeting


def _ensure_meeting_editable(meeting: dict[str, Any]) -> None:
    if meeting.get("lifecycle_state") == "archived":
        raise HTTPException(409, "归档会议为只读，请先取消归档")


def _ensure_lifecycle_change_allowed(
    meeting: dict[str, Any],
) -> None:
    if meeting["status"] in ACTIVE_STATUSES:
        raise HTTPException(409, "任务正在处理，不能移动会议")


def _backups_context(
    request: Request, backup_manager: BackupManager
) -> dict[str, Any]:
    return {
        "backups": backup_manager.list_backups(),
        "lifecycle_counts": backup_manager.database.lifecycle_counts(),
        "schema_version": backup_manager.database.schema_version(),
        "created_backup": request.query_params.get("created", ""),
        "restored_count": request.query_params.get("imported", ""),
        "skipped_count": request.query_params.get("skipped", ""),
        "backup_error": "",
        "restore_error": "",
    }


def _calendar_context(
    database: Database,
    requested_month: str | None,
    *,
    today: date,
) -> dict[str, Any]:
    selected = _parse_calendar_month(requested_month, today)
    weeks = calendar_module.Calendar(firstweekday=0).monthdatescalendar(
        selected.year,
        selected.month,
    )
    first_visible = weeks[0][0]
    last_visible = weeks[-1][-1]
    meetings = database.list_meetings_by_date_range(
        first_visible.isoformat(),
        last_visible.isoformat(),
    )
    meetings_by_date: dict[str, list[dict[str, Any]]] = {}
    for meeting in meetings:
        meetings_by_date.setdefault(
            str(meeting["meeting_date"]), []
        ).append(meeting)

    selected_prefix = selected.strftime("%Y-%m")
    month_meetings = [
        meeting
        for meeting in meetings
        if str(meeting["meeting_date"]).startswith(selected_prefix)
    ]
    month_counts = {
        "total": len(month_meetings),
        "active": 0,
        "archived": 0,
        "trashed": 0,
    }
    for meeting in month_meetings:
        lifecycle_state = str(meeting["lifecycle_state"])
        if lifecycle_state in month_counts:
            month_counts[lifecycle_state] += 1

    calendar_weeks = []
    for week in weeks:
        calendar_weeks.append(
            [
                {
                    "date": day,
                    "iso": day.isoformat(),
                    "day": day.day,
                    "in_month": day.month == selected.month,
                    "is_today": day == today,
                    "meetings": meetings_by_date.get(
                        day.isoformat(), []
                    ),
                }
                for day in week
            ]
        )

    previous_month = selected - timedelta(days=1)
    next_month = (
        selected.replace(day=28) + timedelta(days=4)
    ).replace(day=1)
    agenda_days = [
        day
        for week in calendar_weeks
        for day in week
        if day["in_month"] and day["meetings"]
    ]
    return {
        "calendar_weeks": calendar_weeks,
        "agenda_days": agenda_days,
        "weekday_labels": [
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "周六",
            "周日",
        ],
        "selected_month": selected.strftime("%Y-%m"),
        "month_title": f"{selected.year}年{selected.month}月",
        "previous_month": previous_month.strftime("%Y-%m"),
        "next_month": next_month.strftime("%Y-%m"),
        "current_month": today.strftime("%Y-%m"),
        "month_counts": month_counts,
        "days_with_meetings": len(
            {
                str(meeting["meeting_date"])
                for meeting in month_meetings
            }
        ),
        "lifecycle_counts": database.lifecycle_counts(),
    }


def _parse_calendar_month(
    requested_month: str | None,
    today: date,
) -> date:
    value = (requested_month or "").strip()
    if not value:
        return today.replace(day=1)
    if len(value) != 7 or value[4] != "-":
        raise HTTPException(422, "月份格式应为 YYYY-MM")
    try:
        return date.fromisoformat(f"{value}-01")
    except ValueError as exc:
        raise HTTPException(422, "月份格式应为 YYYY-MM") from exc


def _meeting_context(
    meeting: dict[str, Any],
    storage: MeetingStorage,
    database: Database,
    task_queue: TaskQueue,
) -> dict[str, Any]:
    transcript = storage.read_json(
        meeting, "transcript_edited.json", default={}
    )
    speakers = storage.read_json(meeting, "speakers.json", default={})
    minutes = storage.read_json(meeting, "minutes.json")
    log_path = storage.path(meeting, "processing.log")
    log_text = (
        "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-200:])
        if log_path.exists()
        else ""
    )
    return {
        "meeting": meeting,
        "transcript": transcript,
        "segments": transcript.get("segments", []),
        "speakers": speakers,
        "minutes": minutes,
        "log_text": log_text,
        **_task_context(meeting, database, task_queue),
    }


def _task_context(
    meeting: dict[str, Any],
    database: Database,
    task_queue: TaskQueue,
) -> dict[str, Any]:
    job = database.get_latest_job(meeting["id"])
    checkpoint = (
        str(job["checkpoint"])
        if job is not None
        else task_queue.available_checkpoint(meeting)
    )
    resume_label = CHECKPOINT_RESUME_LABELS.get(checkpoint)
    return {
        "meeting": meeting,
        "job": job,
        "resume_label": resume_label,
    }


def _save_transcript_exports(
    storage: MeetingStorage,
    meeting: dict[str, Any],
    transcript: dict[str, Any],
    speakers: dict[str, str],
) -> Path:
    storage.write_text(
        meeting,
        "transcript.md",
        render_transcript_markdown(
            meeting, transcript.get("segments", []), speakers
        ),
    )
    return storage.write_text(
        meeting,
        "transcript.txt",
        render_transcript_text(
            meeting, transcript.get("segments", []), speakers
        ),
    )


def _mark_transcript_changed(
    database: Database,
    storage: MeetingStorage,
    meeting: dict[str, Any],
) -> None:
    storage.remove_minutes_files(meeting)
    database.update_meeting(
        meeting["id"],
        status="transcribed",
        progress=75,
        current_step="逐字稿已修改，请重新生成纪要",
        error=None,
        llm_backend=None,
    )
    storage.append_log(meeting, f"{utc_now()} 用户修改了逐字稿或说话人")


def _safe_display_filename(value: str | None) -> str:
    if not value:
        return "recording"
    normalized = value.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1][:255] or "recording"


def _download_title(value: str) -> str:
    result = "".join(
        "_" if character in '<>:"/\\|?*' else character
        for character in value
    ).strip(" .")
    return result[:80] or "会议"


app = create_app()
