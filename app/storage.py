from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .config import Settings


SAFE_SUFFIXES = {".mp3", ".m4a", ".wav", ".mp4", ".aac", ".flac", ".ogg"}
_INVALID_SLUG_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTIPLE_SPACES = re.compile(r"\s+")


class UploadTooLargeError(ValueError):
    pass


def make_slug(meeting_date: date, title: str, short_id: str) -> str:
    clean = _INVALID_SLUG_CHARS.sub("_", title)
    clean = _MULTIPLE_SPACES.sub(" ", clean).strip(" ._")
    clean = clean[:60] or "未命名会议"
    return f"{meeting_date.isoformat()}_{clean}_{short_id}"


class MeetingStorage:
    def __init__(self, settings: Settings):
        self.settings = settings

    def meeting_dir(self, meeting: dict[str, Any]) -> Path:
        slug = str(meeting["slug"])
        if (
            not slug
            or Path(slug).name != slug
            or "/" in slug
            or "\\" in slug
        ):
            raise ValueError("会议目录名称无效")
        root = self.settings.meetings_dir.resolve()
        directory = (root / slug).resolve()
        if directory.parent != root:
            raise ValueError("会议目录超出数据目录")
        return directory

    def path(self, meeting: dict[str, Any], name: str) -> Path:
        return self.meeting_dir(meeting) / name

    def prepare(self, meeting: dict[str, Any]) -> Path:
        directory = self.meeting_dir(meeting)
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    async def save_upload(
        self, upload: UploadFile, destination: Path
    ) -> int:
        written = 0
        try:
            with destination.open("xb") as output:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > self.settings.max_upload_bytes:
                        raise UploadTooLargeError(
                            "上传文件超过允许的最大大小"
                        )
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return written

    def write_json(
        self, meeting: dict[str, Any], filename: str, data: Any
    ) -> Path:
        destination = self.path(meeting, filename)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def read_json(
        self, meeting: dict[str, Any], filename: str, default: Any = None
    ) -> Any:
        source = self.path(meeting, filename)
        if not source.exists():
            return default
        return json.loads(source.read_text(encoding="utf-8"))

    def write_text(
        self, meeting: dict[str, Any], filename: str, content: str
    ) -> Path:
        destination = self.path(meeting, filename)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def append_log(self, meeting: dict[str, Any], message: str) -> None:
        destination = self.path(meeting, "processing.log")
        with destination.open("a", encoding="utf-8") as log:
            log.write(message.rstrip() + "\n")

    def remove_generated_files(self, meeting: dict[str, Any]) -> None:
        for name in (
            "normalized.wav",
            "transcript_raw.json",
            "transcript_edited.json",
            "transcript.md",
            "transcript.txt",
            "speakers.json",
            "minutes.json",
            "minutes.md",
            "minutes.txt",
            "minutes.docx",
        ):
            self.path(meeting, name).unlink(missing_ok=True)

    def remove_minutes_files(self, meeting: dict[str, Any]) -> None:
        for name in (
            "minutes.json",
            "minutes.md",
            "minutes.txt",
            "minutes.docx",
        ):
            self.path(meeting, name).unlink(missing_ok=True)

    def stage_permanent_delete(
        self, meeting: dict[str, Any]
    ) -> Path | None:
        source = self.meeting_dir(meeting)
        if not source.exists():
            return None
        staging_root = (
            self.settings.data_dir / ".pending-deletions"
        ).resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
        destination = (
            staging_root / f"{meeting['slug']}-{uuid4().hex}"
        ).resolve()
        if destination.parent != staging_root:
            raise ValueError("永久删除暂存路径无效")
        source.replace(destination)
        return destination

    def restore_staged_delete(
        self,
        meeting: dict[str, Any],
        staged: Path | None,
    ) -> None:
        if staged is None or not staged.exists():
            return
        self._validate_staged_path(staged)
        destination = self.meeting_dir(meeting)
        if destination.exists():
            raise RuntimeError("无法恢复删除暂存目录：目标已存在")
        staged.replace(destination)

    def purge_staged_delete(self, staged: Path | None) -> None:
        if staged is None or not staged.exists():
            return
        staging_root = self._validate_staged_path(staged)
        shutil.rmtree(staged)
        try:
            staging_root.rmdir()
        except OSError:
            pass

    def _validate_staged_path(self, staged: Path) -> Path:
        staging_root = (
            self.settings.data_dir / ".pending-deletions"
        ).resolve()
        resolved = staged.resolve()
        if resolved.parent != staging_root:
            raise ValueError("永久删除暂存路径超出允许目录")
        return staging_root
