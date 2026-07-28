from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .config import Settings
from .database import (
    LATEST_SCHEMA_VERSION,
    LIFECYCLE_STATES,
    Database,
)
from .domain import VALID_STATUSES, utc_now
from .minutes_templates import (
    DEFAULT_TEMPLATE_ID,
    MinutesTemplateError,
    normalize_minutes_template,
)
from .storage import SAFE_SUFFIXES, MeetingStorage, UploadTooLargeError


BACKUP_FORMAT = "meetominute-backup"
BACKUP_FORMAT_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_TEMPLATE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BACKUP_NAME = re.compile(
    r"^meetominute-backup-\d{8}-\d{6}-[a-f0-9]{8}\.zip$"
)
_ACTIVE_TASK_STATUSES = {
    "queued",
    "processing",
    "generating_minutes",
    "canceling",
}


class BackupError(RuntimeError):
    pass


@dataclass(slots=True)
class BackupInfo:
    name: str
    path: Path
    created_at: str
    size_bytes: int
    meeting_count: int
    schema_version: int
    error: str | None = None

    @property
    def size_label(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                precision = 0 if unit == "B" else 1
                return f"{size:.{precision}f} {unit}"
            size /= 1024
        return f"{self.size_bytes} B"


@dataclass(slots=True)
class RestoreResult:
    imported: int
    skipped: int
    skipped_items: list[str]


class BackupManager:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        storage: MeetingStorage,
    ):
        self.settings = settings
        self.database = database
        self.storage = storage
        self.backups_dir = settings.data_dir / "backups"

    def ensure_directory(self) -> None:
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self) -> BackupInfo:
        if self.database.has_active_jobs():
            raise BackupError("有任务正在处理，请等待完成或取消后再备份。")
        self.ensure_directory()
        now = datetime.now(timezone.utc)
        token = uuid4().hex[:8]
        name = (
            f"meetominute-backup-{now:%Y%m%d-%H%M%S}-{token}.zip"
        )
        destination = self.backups_dir / name
        temporary_archive = self.backups_dir / f".{name}.tmp"
        snapshot = self.backups_dir / f".snapshot-{uuid4().hex}.sqlite3"
        meetings = self.database.list_meetings(
            limit=1_000_000, lifecycle_state=None
        )
        manifest = {
            "format": BACKUP_FORMAT,
            "format_version": BACKUP_FORMAT_VERSION,
            "created_at": utc_now(),
            "database_schema_version": self.database.schema_version(),
            "meeting_count": len(meetings),
            "contains_external_llm_credentials": False,
        }
        try:
            self.database.create_snapshot(snapshot)
            with zipfile.ZipFile(
                temporary_archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as archive:
                archive.writestr(
                    "manifest.json",
                    json.dumps(
                        manifest, ensure_ascii=False, indent=2
                    ),
                )
                archive.write(snapshot, "database.sqlite3")
                self._add_meeting_files(archive, meetings)
            temporary_archive.replace(destination)
        except Exception:
            temporary_archive.unlink(missing_ok=True)
            raise
        finally:
            snapshot.unlink(missing_ok=True)
        return BackupInfo(
            name=name,
            path=destination,
            created_at=str(manifest["created_at"]),
            size_bytes=destination.stat().st_size,
            meeting_count=len(meetings),
            schema_version=int(
                manifest["database_schema_version"]
            ),
        )

    def list_backups(self) -> list[BackupInfo]:
        self.ensure_directory()
        backups: list[BackupInfo] = []
        for path in sorted(
            self.backups_dir.glob("meetominute-backup-*.zip"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                manifest = self._read_manifest(path)
                backups.append(
                    BackupInfo(
                        name=path.name,
                        path=path,
                        created_at=str(
                            manifest.get("created_at") or ""
                        ),
                        size_bytes=path.stat().st_size,
                        meeting_count=int(
                            manifest.get("meeting_count") or 0
                        ),
                        schema_version=int(
                            manifest.get(
                                "database_schema_version", 0
                            )
                        ),
                    )
                )
            except (BackupError, OSError, ValueError, zipfile.BadZipFile) as exc:
                backups.append(
                    BackupInfo(
                        name=path.name,
                        path=path,
                        created_at="",
                        size_bytes=path.stat().st_size,
                        meeting_count=0,
                        schema_version=0,
                        error=str(exc),
                    )
                )
        return backups

    def get_backup(self, name: str) -> Path | None:
        if not _BACKUP_NAME.fullmatch(name):
            return None
        root = self.backups_dir.resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            return None
        return path

    async def save_uploaded_backup(self, upload: UploadFile) -> Path:
        self.ensure_directory()
        original = (upload.filename or "").lower()
        if not original.endswith(".zip"):
            await upload.close()
            raise BackupError("恢复文件必须是 MeetOminute ZIP 备份包。")
        destination = (
            self.backups_dir / f".upload-{uuid4().hex}.zip"
        )
        written = 0
        limit = self.settings.max_upload_bytes * 8
        try:
            with destination.open("xb") as output:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise UploadTooLargeError(
                            "备份包超过允许的最大大小"
                        )
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return destination

    def restore_backup(self, archive_path: Path) -> RestoreResult:
        if self.database.has_active_jobs():
            raise BackupError("有任务正在处理，请等待完成或取消后再恢复。")
        self.ensure_directory()
        staging = (
            self.backups_dir / f".restore-{uuid4().hex}"
        ).resolve()
        root = self.backups_dir.resolve()
        if staging.parent != root:
            raise BackupError("恢复暂存路径无效。")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            self._extract_archive(archive_path, staging)
            manifest = self._load_manifest_file(
                staging / "manifest.json"
            )
            self._validate_manifest(manifest)
            backup_database = staging / "database.sqlite3"
            if not backup_database.is_file():
                raise BackupError("备份包缺少 database.sqlite3。")
            return self._import_meetings(backup_database, staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _add_meeting_files(
        self,
        archive: zipfile.ZipFile,
        meetings: list[dict[str, Any]],
    ) -> None:
        root = self.settings.meetings_dir.resolve()
        if not root.exists():
            return
        for meeting in meetings:
            directory = self.storage.meeting_dir(meeting)
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file():
                    continue
                if path.is_symlink():
                    raise BackupError(
                        f"会议目录包含符号链接：{path.name}"
                    )
                resolved = path.resolve()
                try:
                    relative = resolved.relative_to(root)
                except ValueError as exc:
                    raise BackupError("会议文件超出数据目录。") from exc
                if (
                    path.name.endswith(".tmp")
                    or path.name.startswith(".funasr-")
                ):
                    continue
                archive.write(
                    resolved,
                    f"meetings/{relative.as_posix()}",
                )

    def _read_manifest(self, archive_path: Path) -> dict[str, Any]:
        with zipfile.ZipFile(archive_path, "r") as archive:
            try:
                info = archive.getinfo("manifest.json")
            except KeyError as exc:
                raise BackupError("备份包缺少 manifest.json。") from exc
            if info.file_size > 1024 * 1024:
                raise BackupError("备份清单异常过大。")
            payload = json.loads(
                archive.read(info).decode("utf-8")
            )
        self._validate_manifest(payload)
        return payload

    @staticmethod
    def _load_manifest_file(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise BackupError("备份清单缺失或异常过大。")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError("无法读取备份清单。") from exc
        if not isinstance(payload, dict):
            raise BackupError("备份清单格式无效。")
        return payload

    @staticmethod
    def _validate_manifest(payload: dict[str, Any]) -> None:
        if payload.get("format") != BACKUP_FORMAT:
            raise BackupError("这不是 MeetOminute 备份包。")
        try:
            version = int(payload.get("format_version") or 0)
            schema = int(
                payload.get("database_schema_version") or 0
            )
        except (TypeError, ValueError) as exc:
            raise BackupError("备份清单版本字段无效。") from exc
        if version < 1 or version > BACKUP_FORMAT_VERSION:
            raise BackupError(f"不支持的备份格式版本：{version}")
        if schema > LATEST_SCHEMA_VERSION:
            raise BackupError(
                "备份来自更高版本的 MeetOminute，当前程序无法恢复。"
            )

    def _extract_archive(
        self, archive_path: Path, staging: Path
    ) -> None:
        limit = self.settings.max_upload_bytes * 8
        total = 0
        try:
            archive = zipfile.ZipFile(archive_path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise BackupError("备份包损坏或不是有效的 ZIP 文件。") from exc
        with archive:
            entries = archive.infolist()
            if len(entries) > 100_000:
                raise BackupError("备份包文件数量异常。")
            for info in entries:
                total += int(info.file_size)
                if total > limit:
                    raise BackupError("备份包解压后超过安全大小限制。")
                self._extract_entry(archive, info, staging)

    @staticmethod
    def _extract_entry(
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        staging: Path,
    ) -> None:
        if info.flag_bits & 0x1:
            raise BackupError("不支持加密的备份包。")
        if "\\" in info.filename:
            raise BackupError("备份包包含无效路径。")
        relative = PurePosixPath(info.filename)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise BackupError("备份包包含不安全路径。")
        if relative.parts[0] not in {
            "manifest.json",
            "database.sqlite3",
            "meetings",
        }:
            raise BackupError("备份包包含未知的顶层文件。")
        if (
            relative.parts[0] in {"manifest.json", "database.sqlite3"}
            and len(relative.parts) != 1
        ):
            raise BackupError("备份包顶层文件路径无效。")
        if relative.parts[0] == "meetings" and len(relative.parts) < 2:
            raise BackupError("备份包会议目录路径无效。")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise BackupError("备份包不能包含符号链接。")
        destination = staging.joinpath(*relative.parts).resolve()
        if not destination.is_relative_to(staging):
            raise BackupError("备份包路径超出恢复目录。")
        if info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with archive.open(info, "r") as source:
                with destination.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        except FileExistsError as exc:
            raise BackupError("备份包包含重复文件路径。") from exc

    def _import_meetings(
        self, backup_database: Path, staging: Path
    ) -> RestoreResult:
        try:
            source = sqlite3.connect(backup_database)
            source.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise BackupError("无法打开备份数据库。") from exc
        try:
            schema = int(
                source.execute("PRAGMA user_version").fetchone()[0]
            )
            if schema > LATEST_SCHEMA_VERSION:
                raise BackupError("备份数据库版本高于当前程序。")
            exists = source.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'meetings'
                """
            ).fetchone()
            if exists is None:
                raise BackupError("备份数据库缺少会议数据表。")
            rows = source.execute(
                "SELECT * FROM meetings ORDER BY created_at"
            ).fetchall()
            jobs_exist = source.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'jobs'
                """
            ).fetchone()
            job_rows = (
                source.execute(
                    "SELECT * FROM jobs ORDER BY id"
                ).fetchall()
                if jobs_exist is not None
                else []
            )
            templates_exist = source.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'minutes_templates'
                """
            ).fetchone()
            template_rows = (
                source.execute(
                    "SELECT * FROM minutes_templates ORDER BY created_at"
                ).fetchall()
                if templates_exist is not None
                else []
            )
        except sqlite3.Error as exc:
            raise BackupError("备份数据库内容无效。") from exc
        finally:
            source.close()

        imported = 0
        skipped_items: list[str] = []
        template_id_map = self._import_minutes_templates(template_rows)
        jobs_by_meeting: dict[str, list[dict[str, Any]]] = {}
        for job_row in job_rows:
            raw_job = dict(job_row)
            meeting_id = str(raw_job.get("meeting_id") or "")
            jobs_by_meeting.setdefault(meeting_id, []).append(
                self._normalize_job(raw_job)
            )
        for row in rows:
            try:
                record = self._normalize_meeting(dict(row))
            except (BackupError, TypeError, ValueError) as exc:
                skipped_items.append(f"无效会议：{exc}")
                continue
            source_template_id = str(
                record.get("minutes_template_id")
                or DEFAULT_TEMPLATE_ID
            )
            record["minutes_template_id"] = template_id_map.get(
                source_template_id,
                (
                    source_template_id
                    if self.database.get_minutes_template(
                        source_template_id
                    )
                    is not None
                    else DEFAULT_TEMPLATE_ID
                ),
            )
            label = f"{record['title']}（{record['meeting_date']}）"
            if (
                self.database.get_meeting(record["id"]) is not None
                or self.database.get_meeting_by_slug(record["slug"])
                is not None
            ):
                skipped_items.append(f"{label}：当前数据中已存在")
                continue
            source_directory = staging / "meetings" / record["slug"]
            destination = self.storage.meeting_dir(record)
            if not source_directory.is_dir():
                skipped_items.append(f"{label}：备份中缺少会议目录")
                continue
            if destination.exists():
                skipped_items.append(f"{label}：目标目录已存在")
                continue
            source_directory.replace(destination)
            inserted = False
            try:
                self.database.create_meeting(record)
                inserted = True
                self.database.import_job_history(
                    record["id"],
                    jobs_by_meeting.get(record["id"], []),
                )
                self.storage.write_json(record, "meeting.json", record)
            except Exception:
                if inserted:
                    self.database.rollback_imported_meeting(record["id"])
                if destination.exists() and not source_directory.exists():
                    destination.replace(source_directory)
                raise
            imported += 1
        return RestoreResult(
            imported=imported,
            skipped=len(skipped_items),
            skipped_items=skipped_items[:50],
        )

    def _import_minutes_templates(
        self,
        rows: list[sqlite3.Row],
    ) -> dict[str, str]:
        template_id_map = {
            DEFAULT_TEMPLATE_ID: DEFAULT_TEMPLATE_ID
        }
        for row in rows:
            try:
                template = self._normalize_template_row(dict(row))
            except (MinutesTemplateError, TypeError, ValueError, json.JSONDecodeError):
                continue
            source_id = template["id"]
            if source_id == DEFAULT_TEMPLATE_ID or template["is_builtin"]:
                template_id_map[source_id] = DEFAULT_TEMPLATE_ID
                continue
            existing = self.database.get_minutes_template(source_id)
            if existing is None:
                self.database.create_minutes_template(template)
                template_id_map[source_id] = source_id
                continue
            comparable_fields = (
                "name",
                "description",
                "instructions",
                "sections",
            )
            if all(
                existing[field] == template[field]
                for field in comparable_fields
            ):
                template_id_map[source_id] = source_id
                continue
            equivalent = next(
                (
                    candidate
                    for candidate in self.database.list_minutes_templates()
                    if candidate["id"] != source_id
                    and candidate["description"]
                    == template["description"]
                    and candidate["instructions"]
                    == template["instructions"]
                    and candidate["sections"] == template["sections"]
                    and candidate["name"]
                    in {
                        template["name"],
                        f"{template['name']}（恢复）"[:80],
                    }
                ),
                None,
            )
            if equivalent is not None:
                template_id_map[source_id] = equivalent["id"]
                continue
            now = utc_now()
            restored = normalize_minutes_template(
                {
                    **template,
                    "id": uuid4().hex,
                    "name": f"{template['name']}（恢复）"[:80],
                    "is_builtin": False,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            self.database.create_minutes_template(restored)
            template_id_map[source_id] = restored["id"]
        return template_id_map

    @staticmethod
    def _normalize_template_row(
        value: dict[str, Any],
    ) -> dict[str, Any]:
        sections = value.get("sections")
        if sections is None:
            sections = json.loads(str(value.get("sections_json") or "[]"))
        return normalize_minutes_template(
            {
                "id": value.get("id"),
                "name": value.get("name"),
                "description": value.get("description"),
                "instructions": value.get("instructions"),
                "sections": sections,
                "is_builtin": bool(value.get("is_builtin")),
                "created_at": value.get("created_at"),
                "updated_at": value.get("updated_at"),
            }
        )

    @staticmethod
    def _normalize_job(value: dict[str, Any]) -> dict[str, Any]:
        status = str(value.get("status") or "failed")
        if status in {"queued", "running"}:
            status = "canceled"
        if status not in {"completed", "failed", "canceled"}:
            status = "failed"
        checkpoint = str(value.get("checkpoint") or "uploaded")
        if checkpoint not in {
            "uploaded",
            "normalized",
            "transcribed",
            "completed",
        }:
            checkpoint = "uploaded"
        kind = str(value.get("kind") or "pipeline")
        if kind not in {"pipeline", "minutes"}:
            kind = "pipeline"
        now = utc_now()
        try:
            attempts = max(0, int(value.get("attempts") or 0))
        except (TypeError, ValueError):
            attempts = 0
        return {
            "kind": kind,
            "status": status,
            "checkpoint": checkpoint,
            "cancel_requested": 1 if status == "canceled" else 0,
            "attempts": attempts,
            "last_error": (
                str(value["last_error"])[:4000]
                if value.get("last_error")
                else None
            ),
            "created_at": str(value.get("created_at") or now),
            "updated_at": str(value.get("updated_at") or now),
            "started_at": value.get("started_at"),
            "finished_at": value.get("finished_at"),
        }

    @staticmethod
    def _normalize_meeting(value: dict[str, Any]) -> dict[str, Any]:
        required = {
            "id",
            "slug",
            "title",
            "meeting_date",
            "expected_speakers",
            "processing_mode",
            "source_filename",
            "source_suffix",
            "status",
            "progress",
            "current_step",
            "created_at",
            "updated_at",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise BackupError("缺少字段 " + "、".join(missing))
        meeting_id = str(value["id"])
        slug = str(value["slug"])
        if not _SAFE_ID.fullmatch(meeting_id):
            raise BackupError("会议 ID 无效")
        if (
            not slug
            or len(slug) > 180
            or Path(slug).name != slug
            or "/" in slug
            or "\\" in slug
        ):
            raise BackupError("会议目录名称无效")
        suffix = str(value["source_suffix"]).lower()
        if suffix not in SAFE_SUFFIXES:
            raise BackupError("录音文件后缀无效")
        mode = str(value["processing_mode"])
        if mode not in {"local", "mixed", "cloud"}:
            raise BackupError("处理模式无效")
        meeting_date = str(value["meeting_date"])[:10]
        try:
            datetime.strptime(meeting_date, "%Y-%m-%d")
        except ValueError as exc:
            raise BackupError("会议日期无效") from exc
        status = str(value["status"])
        if status not in VALID_STATUSES:
            status = "failed"
        progress = max(0, min(100, int(value["progress"])))
        current_step = str(value["current_step"])[:1000]
        error = value.get("error")
        if status in _ACTIVE_TASK_STATUSES:
            status = "canceled"
            current_step = "从备份恢复，原任务未自动重启"
            error = None
        lifecycle = str(
            value.get("lifecycle_state") or "active"
        )
        if lifecycle not in LIFECYCLE_STATES:
            lifecycle = "active"
        trashed_from = value.get("trashed_from")
        if trashed_from not in {"active", "archived"}:
            trashed_from = (
                "active" if lifecycle == "trashed" else None
            )
        raw_duration = value.get("duration_seconds")
        try:
            duration = (
                max(0.0, float(raw_duration))
                if raw_duration is not None
                else None
            )
        except (TypeError, ValueError):
            duration = None
        if duration is not None and not math.isfinite(duration):
            duration = None
        return {
            "id": meeting_id,
            "slug": slug,
            "title": str(value["title"])[:200] or "未命名会议",
            "meeting_date": meeting_date,
            "expected_speakers": max(
                1, min(50, int(value["expected_speakers"]))
            ),
            "glossary": str(value.get("glossary") or "")[:20_000],
            "processing_mode": mode,
            "minutes_template_id": (
                str(
                    value.get("minutes_template_id")
                    or DEFAULT_TEMPLATE_ID
                )
                if _SAFE_TEMPLATE_ID.fullmatch(
                    str(
                        value.get("minutes_template_id")
                        or DEFAULT_TEMPLATE_ID
                    )
                )
                else DEFAULT_TEMPLATE_ID
            ),
            "source_filename": str(value["source_filename"])[:255],
            "source_suffix": suffix,
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "error": str(error)[:4000] if error else None,
            "duration_seconds": duration,
            "transcriber_backend": (
                str(value["transcriber_backend"])[:100]
                if value.get("transcriber_backend")
                else None
            ),
            "llm_backend": (
                str(value["llm_backend"])[:100]
                if value.get("llm_backend")
                else None
            ),
            "created_at": str(value["created_at"]),
            "updated_at": str(value["updated_at"]),
            "lifecycle_state": lifecycle,
            "archived_at": value.get("archived_at"),
            "trashed_at": value.get("trashed_at"),
            "trashed_from": trashed_from,
        }
