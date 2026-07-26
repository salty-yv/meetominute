from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date
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

from .config import Settings
from .database import Database
from .domain import format_timestamp, utc_now
from .external_llm import (
    ExternalLLMConfigStore,
    test_external_llm_connection,
)
from .pipeline import TaskQueue
from .rendering import render_transcript_markdown
from .storage import (
    SAFE_SUFFIXES,
    MeetingStorage,
    UploadTooLargeError,
    make_slug,
)


STATUS_LABELS = {
    "queued": "等待处理",
    "processing": "处理中",
    "transcribed": "转写已完成",
    "generating_minutes": "正在生成纪要",
    "completed": "已完成",
    "failed": "失败",
}
ACTIVE_STATUSES = {"queued", "processing", "generating_minutes"}
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
    templates = Jinja2Templates(directory=config.base_dir / "app" / "templates")
    templates.env.filters["timestamp"] = format_timestamp
    templates.env.globals["status_labels"] = STATUS_LABELS
    templates.env.globals["active_statuses"] = ACTIVE_STATUSES

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        config.ensure_directories()
        database.initialize()
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
                "today": date.today().isoformat(),
                "max_upload_mb": config.max_upload_bytes // 1024 // 1024,
                "local_transcriber": config.local_transcriber,
                "local_llm": config.local_llm,
                "ollama_model": config.ollama_model,
                "external_llm": external_llm_store.load().public_dict(),
            },
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
        await task_queue.enqueue("pipeline", meeting_id)
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
            context=_meeting_context(meeting, storage),
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
            context={"meeting": meeting},
        )

    @application.post("/meetings/{meeting_id}/retry")
    async def retry_meeting(
        request: Request, meeting_id: str
    ) -> RedirectResponse:
        meeting = _meeting_or_404(database, meeting_id)
        if meeting["status"] in ACTIVE_STATUSES:
            raise HTTPException(409, "任务当前正在处理，不能重复加入队列")
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
        await task_queue.enqueue("pipeline", meeting_id)
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
        _save_transcript_markdown(storage, meeting, transcript, speakers)
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
        _save_transcript_markdown(storage, meeting, transcript, speakers)
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
        transcript = storage.read_json(
            meeting, "transcript_edited.json", default={}
        )
        if not transcript.get("segments"):
            raise HTTPException(409, "逐字稿尚未生成")
        database.update_meeting(
            meeting_id,
            status="generating_minutes",
            progress=80,
            current_step="纪要已进入队列",
            error=None,
        )
        await task_queue.enqueue("minutes", meeting_id)
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
    database: Database, meeting_id: str
) -> dict[str, Any]:
    meeting = database.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(404, "会议不存在")
    return meeting


def _meeting_context(
    meeting: dict[str, Any], storage: MeetingStorage
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
    }


def _save_transcript_markdown(
    storage: MeetingStorage,
    meeting: dict[str, Any],
    transcript: dict[str, Any],
    speakers: dict[str, str],
) -> None:
    storage.write_text(
        meeting,
        "transcript.md",
        render_transcript_markdown(
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
