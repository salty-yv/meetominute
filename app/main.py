from __future__ import annotations

import asyncio
import calendar as calendar_module
import copy
import json
import shutil
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

from .actions import (
    ACTION_STATUS_LABELS,
    ActionItemsError,
    ActionItemsService,
    annotate_action_items,
)
from .backups import BackupError, BackupManager
from .config import Settings
from .database import Database
from .diagnostics import run_diagnostics
from .domain import format_timestamp, utc_now
from .external_llm import (
    ExternalLLMConfigStore,
    test_external_llm_connection,
)
from .minutes_templates import (
    CORE_SECTION_DEFINITIONS,
    DEFAULT_TEMPLATE_ID,
    MinutesTemplateError,
    blank_minutes_template_form,
    build_minutes_template_from_form,
    minutes_template_form_values,
    normalize_minutes_template,
)
from .maintenance import MaintenanceBusyError, MaintenanceGate
from .pipeline import CHECKPOINT_RESUME_LABELS, TaskQueue
from .rendering import (
    minutes_item_text,
    minutes_sections,
    render_minutes_docx,
    render_minutes_markdown,
    render_minutes_text,
    render_transcript_markdown,
    render_transcript_text,
)
from .search import MeetingSearchService
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
MAX_SPEAKER_NAME_CHARS = 100
MAX_TRANSCRIPT_SEGMENT_CHARS = 100_000
_FORM_ENCODED_BYTES_PER_CHAR = 12
_FORM_FIELD_OVERHEAD_BYTES = 1_024
_MAX_SPEAKER_FORM_BODY_BYTES = 1024 * 1024
_MAX_EDIT_FORM_BODY_BYTES = 64 * 1024 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    database = Database(config.db_path)
    storage = MeetingStorage(config)
    external_llm_store = ExternalLLMConfigStore(
        config.data_dir / "external-llm.json", config
    )
    maintenance_gate = MaintenanceGate()
    task_queue = TaskQueue(
        config,
        database,
        storage,
        external_llm_store,
        maintenance_gate=maintenance_gate,
    )
    backup_manager = BackupManager(
        config,
        database,
        storage,
        maintenance_gate=maintenance_gate,
    )
    meeting_search = MeetingSearchService(database, storage)
    action_items = ActionItemsService(database, storage)
    meeting_operation_locks: dict[str, asyncio.Lock] = {}

    def meeting_operation_lock(meeting_id: str) -> asyncio.Lock:
        return meeting_operation_locks.setdefault(
            meeting_id, asyncio.Lock()
        )
    templates = Jinja2Templates(directory=config.base_dir / "app" / "templates")
    templates.env.filters["timestamp"] = format_timestamp
    templates.env.filters["minutes_item_text"] = minutes_item_text
    templates.env.globals["status_labels"] = STATUS_LABELS
    templates.env.globals["active_statuses"] = ACTIVE_STATUSES
    templates.env.globals["action_status_labels"] = ACTION_STATUS_LABELS

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
    application.state.maintenance_gate = maintenance_gate
    application.state.meeting_search = meeting_search
    application.state.action_items = action_items
    application.mount(
        "/static",
        StaticFiles(directory=config.base_dir / "app" / "static"),
        name="static",
    )

    @application.middleware("http")
    async def coordinate_data_mutations(
        request: Request,
        call_next,
    ) -> Response:
        is_mutation = request.method.upper() in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }
        path = request.url.path
        mutates_meeting_data = (
            path == "/meetings"
            or path.startswith("/meetings/")
            or path == "/minutes-templates"
            or path.startswith("/minutes-templates/")
        )
        if not is_mutation or not mutates_meeting_data:
            return await call_next(request)
        try:
            with maintenance_gate.mutation():
                return await call_next(request)
        except MaintenanceBusyError as exc:
            expects_json = (
                request.url.path.startswith("/api/")
                or request.url.path == "/settings/external-llm/test"
                or "application/json"
                in request.headers.get("accept", "").lower()
            )
            if expects_json:
                return JSONResponse(
                    {"detail": str(exc)},
                    status_code=409,
                )
            return templates.TemplateResponse(
                request=request,
                name="maintenance.html",
                context={"maintenance_message": str(exc)},
                status_code=409,
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
                "minutes_templates": database.list_minutes_templates(),
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
                action_items=action_items,
            ),
        )

    @application.get("/search", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = "",
        scope: str = "all",
        lifecycle: str = "all",
    ) -> HTMLResponse:
        query = q.strip()[:300]
        try:
            results = await asyncio.to_thread(
                meeting_search.search,
                query,
                scope=scope,
                lifecycle_state=lifecycle,
                limit=100,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return templates.TemplateResponse(
            request=request,
            name="search.html",
            context={
                "query": query,
                "scope": scope,
                "lifecycle": lifecycle,
                "results": results,
                "result_count": len(results),
            },
        )

    @application.get("/actions", response_class=HTMLResponse)
    async def actions_page(
        request: Request,
        status: str = "pending",
        q: str = "",
    ) -> HTMLResponse:
        selected_status = status.strip().lower() or "pending"
        if selected_status not in {
            "all",
            "pending",
            "overdue",
            "done",
            "dismissed",
        }:
            raise HTTPException(422, "无效的待办状态")
        query = q.strip()[:300]
        all_actions = await asyncio.to_thread(
            action_items.list_actions,
            status=None,
            query=query or None,
            today=date.today(),
        )
        visible_actions = [
            item
            for item in all_actions
            if item.get("lifecycle_state") != "trashed"
        ]
        counts = _action_counts(visible_actions)
        if selected_status == "all":
            filtered_actions = visible_actions
        elif selected_status == "overdue":
            filtered_actions = [
                item
                for item in visible_actions
                if item.get("is_overdue")
            ]
        else:
            filtered_actions = [
                item
                for item in visible_actions
                if item.get("status") == selected_status
            ]
        return templates.TemplateResponse(
            request=request,
            name="actions.html",
            context={
                "actions": filtered_actions,
                "counts": counts,
                "selected_status": selected_status,
                "query": query,
            },
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

    def render_minutes_templates_page(
        request: Request,
        *,
        edit_id: str | None = None,
        form_values: dict[str, Any] | None = None,
        template_error: str = "",
        status_code: int = 200,
    ) -> HTMLResponse:
        selected = (
            database.get_minutes_template(edit_id)
            if edit_id
            else None
        )
        if edit_id and selected is None:
            raise HTTPException(404, "纪要模板不存在")
        values = form_values or (
            minutes_template_form_values(selected)
            if selected
            else blank_minutes_template_form()
        )
        template_records = []
        for template in database.list_minutes_templates():
            template_records.append(
                {
                    **template,
                    "usage_count": database.count_meetings_using_template(
                        template["id"]
                    ),
                }
            )
        return templates.TemplateResponse(
            request=request,
            name="minutes_templates.html",
            context={
                "minutes_templates": template_records,
                "form_values": values,
                "editing_template": selected,
                "core_sections": CORE_SECTION_DEFINITIONS,
                "template_error": template_error,
                "created_template": request.query_params.get(
                    "created", ""
                ),
                "updated_template": request.query_params.get(
                    "updated", ""
                ),
                "duplicated_template": request.query_params.get(
                    "duplicated", ""
                ),
                "deleted_template": request.query_params.get(
                    "deleted", ""
                ),
            },
            status_code=status_code,
        )

    @application.get(
        "/minutes-templates", response_class=HTMLResponse
    )
    async def minutes_templates_page(
        request: Request,
        edit: str | None = None,
    ) -> HTMLResponse:
        return render_minutes_templates_page(
            request,
            edit_id=edit,
        )

    @application.post("/minutes-templates")
    async def create_minutes_template(request: Request) -> Response:
        form = await request.form()
        try:
            template = build_minutes_template_from_form(form)
            database.create_minutes_template(template)
        except (MinutesTemplateError, ValueError) as exc:
            return render_minutes_templates_page(
                request,
                form_values=_minutes_template_raw_form_values(form),
                template_error=str(exc),
                status_code=422,
            )
        target = request.url_for(
            "minutes_templates_page"
        ).include_query_params(
            created=template["name"],
            edit=template["id"],
        )
        return RedirectResponse(target, status_code=303)

    @application.post("/minutes-templates/{template_id}")
    async def update_minutes_template(
        request: Request,
        template_id: str,
    ) -> Response:
        existing = database.get_minutes_template(template_id)
        if existing is None:
            raise HTTPException(404, "纪要模板不存在")
        if existing["is_builtin"]:
            raise HTTPException(409, "内置模板不能直接修改，请先复制")
        form = await request.form()
        try:
            template = build_minutes_template_from_form(
                form,
                template_id=template_id,
                existing=existing,
            )
            if not database.update_minutes_template(template):
                raise MinutesTemplateError("纪要模板未能更新")
        except (MinutesTemplateError, ValueError) as exc:
            return render_minutes_templates_page(
                request,
                edit_id=template_id,
                form_values=_minutes_template_raw_form_values(
                    form,
                    template_id=template_id,
                ),
                template_error=str(exc),
                status_code=422,
            )
        target = request.url_for(
            "minutes_templates_page"
        ).include_query_params(
            updated=template["name"],
            edit=template["id"],
        )
        return RedirectResponse(target, status_code=303)

    @application.post(
        "/minutes-templates/{template_id}/duplicate"
    )
    async def duplicate_minutes_template(
        request: Request,
        template_id: str,
    ) -> RedirectResponse:
        source = database.get_minutes_template(template_id)
        if source is None:
            raise HTTPException(404, "纪要模板不存在")
        now = utc_now()
        duplicate = normalize_minutes_template(
            {
                **copy.deepcopy(source),
                "id": uuid4().hex,
                "name": f"{source['name']} 副本"[:80],
                "is_builtin": False,
                "created_at": now,
                "updated_at": now,
            }
        )
        database.create_minutes_template(duplicate)
        target = request.url_for(
            "minutes_templates_page"
        ).include_query_params(
            duplicated=duplicate["name"],
            edit=duplicate["id"],
        )
        return RedirectResponse(target, status_code=303)

    @application.post("/minutes-templates/{template_id}/delete")
    async def delete_minutes_template(
        request: Request,
        template_id: str,
    ) -> Response:
        template = database.get_minutes_template(template_id)
        if template is None:
            raise HTTPException(404, "纪要模板不存在")
        result = database.delete_minutes_template(template_id)
        if result == "in_use":
            return render_minutes_templates_page(
                request,
                edit_id=template_id,
                template_error=(
                    "这个模板仍被会议使用。请先为这些会议选择其他模板。"
                ),
                status_code=409,
            )
        if result == "builtin":
            raise HTTPException(409, "内置模板不能删除")
        if result != "deleted":
            raise HTTPException(404, "纪要模板不存在")
        target = request.url_for(
            "minutes_templates_page"
        ).include_query_params(deleted=template["name"])
        return RedirectResponse(target, status_code=303)

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
        minutes_template_id: str = Form(DEFAULT_TEMPLATE_ID),
        recording: UploadFile = File(...),
    ) -> RedirectResponse:
        clean_title = title.strip()
        if not clean_title:
            raise HTTPException(422, "会议名称不能为空")
        if not 1 <= expected_speakers <= 50:
            raise HTTPException(422, "预计发言人数必须在 1 到 50 之间")
        if processing_mode not in {"local", "mixed", "cloud"}:
            raise HTTPException(422, "无效的处理模式")
        if database.get_minutes_template(minutes_template_id) is None:
            raise HTTPException(422, "选择的纪要模板不存在")
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
            "minutes_template_id": minutes_template_id,
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
                meeting,
                storage,
                database,
                task_queue,
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

    @application.post(
        "/meetings/{meeting_id}/actions/{action_id}"
    )
    async def update_action_status(
        request: Request,
        meeting_id: str,
        action_id: str,
        status: str = Form(...),
        return_to: str = Form("actions"),
        selected_status: str = Form("pending"),
        query: str = Form(""),
    ) -> RedirectResponse:
        async with meeting_operation_lock(meeting_id):
            meeting = _meeting_or_404(database, meeting_id)
            if meeting["status"] in ACTIVE_STATUSES:
                raise HTTPException(
                    409, "会议正在处理，请完成后再更新待办状态"
                )
            try:
                minutes = await asyncio.to_thread(
                    action_items.update_status,
                    meeting_id,
                    action_id,
                    status,
                    on_updated=lambda previous, updated: (
                        _commit_minutes_bundle(
                            storage,
                            meeting,
                            previous,
                            updated,
                        )
                    ),
                )
            except ActionItemsError as exc:
                raise HTTPException(422, str(exc)) from exc
            storage.append_log(
                meeting,
                (
                    f"{utc_now()} 用户将待办 {action_id} 更新为"
                    f"{ACTION_STATUS_LABELS.get(status, status)}"
                ),
            )
        if return_to == "meeting":
            target = str(
                request.url_for(
                    "meeting_detail", meeting_id=meeting_id
                )
            )
            return RedirectResponse(
                target + "#minutes", status_code=303
            )
        safe_selected_status = (
            selected_status
            if selected_status
            in {
                "all",
                "pending",
                "overdue",
                "done",
                "dismissed",
            }
            else "pending"
        )
        target = request.url_for(
            "actions_page"
        ).include_query_params(status=safe_selected_status)
        clean_query = query.strip()[:300]
        if clean_query:
            target = target.include_query_params(q=clean_query)
        return RedirectResponse(target, status_code=303)

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
        async with meeting_operation_lock(meeting_id):
            meeting = _meeting_or_404(database, meeting_id)
            if meeting["status"] in ACTIVE_STATUSES:
                raise HTTPException(
                    409, "任务当前正在处理，不能重复加入队列"
                )
            _ensure_meeting_editable(meeting)
            if meeting["status"] not in {"failed", "canceled"}:
                raise HTTPException(409, "当前任务不需要断点续跑")
            kind, checkpoint, progress, label = task_queue.resume_info(
                meeting
            )
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
        async with meeting_operation_lock(meeting_id):
            meeting = _meeting_or_404(database, meeting_id)
            if meeting["status"] in ACTIVE_STATUSES:
                raise HTTPException(
                    409, "任务当前正在处理，不能重复加入队列"
                )
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
            storage.append_log(
                meeting, f"{utc_now()} 用户请求重新处理"
            )
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
        form = await _parse_limited_edit_form(
            request,
            max_fields=len(current),
            max_body_bytes=_speaker_form_body_limit(len(current)),
            max_part_size=(
                MAX_SPEAKER_NAME_CHARS * _FORM_ENCODED_BYTES_PER_CHAR
                + _FORM_FIELD_OVERHEAD_BYTES
            ),
        )
        speakers = {
            key: str(form.get(f"name_{key}", "")).strip()[
                :MAX_SPEAKER_NAME_CHARS
            ]
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
        form = await _parse_limited_edit_form(
            request,
            max_fields=len(segments) * 2,
            max_body_bytes=_transcript_form_body_limit(len(segments)),
            max_part_size=(
                MAX_TRANSCRIPT_SEGMENT_CHARS
                * _FORM_ENCODED_BYTES_PER_CHAR
                + _FORM_FIELD_OVERHEAD_BYTES
            ),
        )
        known_speakers = set(speakers)
        for segment in segments:
            segment_id = segment["id"]
            text_value = str(form.get(f"text_{segment_id}", "")).strip()
            speaker_value = str(
                form.get(f"speaker_{segment_id}", segment["speaker"])
            )
            if speaker_value not in known_speakers:
                raise HTTPException(422, "说话人标签无效")
            segment["text"] = text_value[:MAX_TRANSCRIPT_SEGMENT_CHARS]
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
        request: Request,
        meeting_id: str,
        minutes_template_id: str = Form(""),
    ) -> RedirectResponse:
        async with meeting_operation_lock(meeting_id):
            meeting = _meeting_or_404(database, meeting_id)
            if meeting["status"] in ACTIVE_STATUSES:
                raise HTTPException(409, "当前已有任务正在处理")
            _ensure_meeting_editable(meeting)
            transcript = storage.read_json(
                meeting, "transcript_edited.json", default={}
            )
            if not transcript.get("segments"):
                raise HTTPException(409, "逐字稿尚未生成")
            selected_template_id = (
                minutes_template_id.strip()
                or str(
                    meeting.get("minutes_template_id")
                    or DEFAULT_TEMPLATE_ID
                )
            )
            if (
                database.get_minutes_template(selected_template_id)
                is None
            ):
                raise HTTPException(422, "选择的纪要模板不存在")
            database.update_meeting(
                meeting_id,
                minutes_template_id=selected_template_id,
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


def _speaker_form_body_limit(speaker_count: int) -> int:
    per_speaker = (
        MAX_SPEAKER_NAME_CHARS * _FORM_ENCODED_BYTES_PER_CHAR
        + _FORM_FIELD_OVERHEAD_BYTES
    )
    return min(
        _MAX_SPEAKER_FORM_BODY_BYTES,
        max(_FORM_FIELD_OVERHEAD_BYTES, speaker_count * per_speaker),
    )


def _transcript_form_body_limit(segment_count: int) -> int:
    per_segment = (
        MAX_TRANSCRIPT_SEGMENT_CHARS * _FORM_ENCODED_BYTES_PER_CHAR
        + (2 * _FORM_FIELD_OVERHEAD_BYTES)
    )
    return min(
        _MAX_EDIT_FORM_BODY_BYTES,
        max(_FORM_FIELD_OVERHEAD_BYTES, segment_count * per_segment),
    )


async def _parse_limited_edit_form(
    request: Request,
    *,
    max_fields: int,
    max_body_bytes: int,
    max_part_size: int,
) -> Any:
    declared_length = request.headers.get("content-length")
    if declared_length is not None:
        try:
            declared_bytes = int(declared_length)
        except ValueError as exc:
            raise HTTPException(400, "Content-Length 无效") from exc
        if declared_bytes < 0:
            raise HTTPException(400, "Content-Length 无效")
        if declared_bytes > max_body_bytes:
            raise HTTPException(413, "编辑内容过大，请缩短后再保存")

    chunks: list[bytes] = []
    received_bytes = 0
    async for chunk in request.stream():
        received_bytes += len(chunk)
        if received_bytes > max_body_bytes:
            raise HTTPException(413, "编辑内容过大，请缩短后再保存")
        if chunk:
            chunks.append(chunk)

    chunk_index = 0

    async def replay_body() -> dict[str, Any]:
        nonlocal chunk_index
        if chunk_index >= len(chunks):
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        chunk = chunks[chunk_index]
        chunk_index += 1
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": chunk_index < len(chunks),
        }

    replay_request = Request(request.scope, receive=replay_body)
    return await replay_request.form(
        max_files=0,
        max_fields=max_fields,
        max_part_size=max_part_size,
    )


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
    action_items: ActionItemsService | None = None,
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
    action_counts_by_meeting: dict[str, dict[str, int]] = {}
    if action_items is not None:
        try:
            calendar_actions = action_items.list_actions(today=today)
        except ActionItemsError:
            calendar_actions = []
        for item in calendar_actions:
            if item.get("status") != "pending":
                continue
            meeting_counts = action_counts_by_meeting.setdefault(
                str(item["meeting_id"]),
                {"pending": 0, "overdue": 0},
            )
            meeting_counts["pending"] += 1
            if item.get("is_overdue"):
                meeting_counts["overdue"] += 1
    meetings = [
        {
            **meeting,
            "pending_action_count": action_counts_by_meeting.get(
                str(meeting["id"]), {}
            ).get("pending", 0),
            "overdue_action_count": action_counts_by_meeting.get(
                str(meeting["id"]), {}
            ).get("overdue", 0),
        }
        for meeting in meetings
    ]
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
        "pending_actions": sum(
            int(meeting["pending_action_count"])
            for meeting in month_meetings
            if meeting["lifecycle_state"] != "trashed"
        ),
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
    if minutes:
        try:
            minutes = annotate_action_items(
                str(meeting["id"]), minutes
            )
        except ActionItemsError:
            pass
    minute_sections = minutes_sections(minutes) if minutes else []
    selected_template = database.get_minutes_template(
        str(
            meeting.get("minutes_template_id")
            or DEFAULT_TEMPLATE_ID
        )
    )
    if selected_template is None:
        selected_template = database.get_minutes_template(
            DEFAULT_TEMPLATE_ID
        )
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
        "minute_sections": minute_sections,
        "minute_summary_section": next(
            (
                section
                for section in minute_sections
                if section["kind"] == "summary"
            ),
            None,
        ),
        "minute_list_sections": [
            section
            for section in minute_sections
            if section["kind"] == "list"
        ],
        "minute_actions_section": next(
            (
                section
                for section in minute_sections
                if section["kind"] == "actions"
            ),
            None,
        ),
        "selected_minutes_template": selected_template,
        "minutes_templates": database.list_minutes_templates(),
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


def _commit_minutes_bundle(
    storage: MeetingStorage,
    meeting: dict[str, Any],
    _previous_minutes: dict[str, Any],
    updated_minutes: dict[str, Any],
) -> None:
    token = uuid4().hex
    targets = {
        "minutes.json": storage.path(meeting, "minutes.json"),
        "minutes.md": storage.path(meeting, "minutes.md"),
        "minutes.txt": storage.path(meeting, "minutes.txt"),
        "minutes.docx": storage.path(meeting, "minutes.docx"),
    }
    staged = {
        name: target.with_name(f".{target.name}.stage-{token}")
        for name, target in targets.items()
    }
    backups = {
        name: target.with_name(f".{target.name}.rollback-{token}")
        for name, target in targets.items()
    }
    existed = {
        name: target.exists() for name, target in targets.items()
    }
    keep_backups = False
    try:
        staged["minutes.json"].write_text(
            json.dumps(
                updated_minutes,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        staged["minutes.md"].write_text(
            render_minutes_markdown(updated_minutes),
            encoding="utf-8",
        )
        staged["minutes.txt"].write_text(
            render_minutes_text(updated_minutes),
            encoding="utf-8",
        )
        render_minutes_docx(
            updated_minutes, staged["minutes.docx"]
        )

        for name, target in targets.items():
            if existed[name]:
                shutil.copy2(target, backups[name])
        committed: list[str] = []
        try:
            for name, target in targets.items():
                staged[name].replace(target)
                committed.append(name)
        except Exception:
            rollback_errors: list[Exception] = []
            for name in reversed(committed):
                target = targets[name]
                try:
                    if existed[name]:
                        backups[name].replace(target)
                    else:
                        target.unlink(missing_ok=True)
                except Exception as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if rollback_errors:
                keep_backups = True
                raise RuntimeError(
                    "待办状态保存失败，且导出文件回滚不完整；"
                    "恢复副本已保留在会议目录"
                ) from rollback_errors[0]
            raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        if not keep_backups:
            for path in backups.values():
                path.unlink(missing_ok=True)


def _action_counts(
    actions: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "all": len(actions),
        "pending": sum(
            item.get("status") == "pending" for item in actions
        ),
        "overdue": sum(
            bool(item.get("is_overdue")) for item in actions
        ),
        "done": sum(
            item.get("status") == "done" for item in actions
        ),
        "dismissed": sum(
            item.get("status") == "dismissed" for item in actions
        ),
    }


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


def _minutes_template_raw_form_values(
    form: Any,
    *,
    template_id: str = "",
) -> dict[str, Any]:
    return {
        "id": template_id,
        "name": str(form.get("name") or ""),
        "description": str(form.get("description") or ""),
        "instructions": str(form.get("instructions") or ""),
        "included": {
            str(section["key"])
            for section in CORE_SECTION_DEFINITIONS
            if bool(section["required"])
            or str(
                form.get(f"include_{section['key']}") or ""
            ).lower()
            in {"on", "true", "1", "yes"}
        },
        "titles": {
            str(section["key"]): str(
                form.get(f"title_{section['key']}")
                or section["title"]
            )
            for section in CORE_SECTION_DEFINITIONS
        },
        "custom_sections": str(form.get("custom_sections") or ""),
        "is_builtin": False,
    }


app = create_app()
