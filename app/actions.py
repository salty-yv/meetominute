from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .database import Database
from .domain import utc_now
from .storage import MeetingStorage


ACTION_STATUSES = ("pending", "done", "dismissed")
ACTION_STATUS_LABELS = {
    "pending": "待处理",
    "done": "已完成",
    "dismissed": "已忽略",
}

_STATUS_ALIASES = {
    "pending": "pending",
    "todo": "pending",
    "open": "pending",
    "待处理": "pending",
    "待办": "pending",
    "未完成": "pending",
    "进行中": "pending",
    "处理中": "pending",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "已完成": "done",
    "完成": "done",
    "dismissed": "dismissed",
    "ignored": "dismissed",
    "cancelled": "dismissed",
    "canceled": "dismissed",
    "已忽略": "dismissed",
    "忽略": "dismissed",
    "已取消": "dismissed",
    "取消": "dismissed",
}
_WHITESPACE = re.compile(r"\s+")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[Path, threading.RLock] = {}
LOGGER = logging.getLogger(__name__)


class ActionItemsError(RuntimeError):
    """A user-facing error while reading or updating meeting actions."""


def normalize_action_status(value: Any) -> str:
    """Return the canonical action status or raise a clear validation error."""

    normalized = str(value or "").strip().casefold()
    status = _STATUS_ALIASES.get(normalized)
    if status is None:
        allowed = "、".join(ACTION_STATUS_LABELS.values())
        raise ActionItemsError(f"无效的待办状态；可选状态：{allowed}")
    return status


def action_fingerprint(item: dict[str, Any]) -> str:
    """Fingerprint the content that identifies an action across regeneration."""

    fields = ("owner", "task", "due", "evidence_time")
    content = "\x1f".join(_normalized_text(item.get(field)) for field in fields)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def stable_action_id(
    meeting_id: str,
    item: dict[str, Any],
    occurrence: int = 0,
) -> str:
    """Build a deterministic, meeting-scoped ID for an action occurrence."""

    identity = (
        f"{_normalized_text(meeting_id)}\x1f"
        f"{action_fingerprint(item)}\x1f{int(occurrence)}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"act_{digest}"


def annotate_action_items(
    meeting_id: str,
    minutes: dict[str, Any],
) -> dict[str, Any]:
    """Normalize legacy actions while preserving their current statuses."""

    result = _copy_minutes(minutes)
    actions = _read_action_list(result)
    occurrences: defaultdict[str, int] = defaultdict(int)
    normalized: list[dict[str, Any]] = []
    for raw_item in actions:
        item = _normalize_action_item(raw_item)
        fingerprint = action_fingerprint(item)
        occurrence = occurrences[fingerprint]
        occurrences[fingerprint] += 1
        item["action_id"] = stable_action_id(
            meeting_id, item, occurrence
        )
        item["status"] = _existing_status(item.get("status"))
        normalized.append(item)
    result["action_items"] = normalized
    return result


def reconcile_action_items(
    meeting_id: str,
    generated_minutes: dict[str, Any],
    previous_minutes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign IDs and preserve matching action statuses after regeneration."""

    result = _copy_minutes(generated_minutes)
    generated_actions = _read_action_list(result)
    previous_statuses: defaultdict[str, deque[str]] = defaultdict(deque)
    if previous_minutes is not None:
        old = annotate_action_items(meeting_id, previous_minutes)
        for item in old["action_items"]:
            previous_statuses[action_fingerprint(item)].append(
                item["status"]
            )

    occurrences: defaultdict[str, int] = defaultdict(int)
    reconciled: list[dict[str, Any]] = []
    for raw_item in generated_actions:
        item = _normalize_action_item(raw_item)
        fingerprint = action_fingerprint(item)
        occurrence = occurrences[fingerprint]
        occurrences[fingerprint] += 1
        statuses = previous_statuses[fingerprint]
        item["action_id"] = stable_action_id(
            meeting_id, item, occurrence
        )
        item["status"] = statuses.popleft() if statuses else "pending"
        reconciled.append(item)
    result["action_items"] = reconciled
    return result


def is_action_overdue(
    item: dict[str, Any],
    today: date | str | None = None,
) -> bool:
    """Return whether a pending action has a concrete past due date."""

    if _existing_status(item.get("status")) != "pending":
        return False
    due_date = _parse_due_date(item.get("due"))
    if due_date is None:
        return False
    reference = _coerce_today(today)
    return due_date < reference


class ActionItemsService:
    def __init__(self, database: Database, storage: MeetingStorage):
        self.database = database
        self.storage = storage

    def list_actions(
        self,
        status: str | None = None,
        query: str | None = None,
        today: date | str | None = None,
        lifecycle: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate and filter actions from every meeting lifecycle."""

        wanted_status = _optional_status_filter(status)
        wanted_lifecycle = _optional_lifecycle_filter(lifecycle)
        reference = _coerce_today(today)
        normalized_query = _normalized_text(query)
        aggregated: list[dict[str, Any]] = []
        meetings = self.database.list_meetings(
            limit=1_000_000,
            lifecycle_state=None,
        )
        for meeting in meetings:
            try:
                minutes = self._read_minutes(meeting, missing_ok=True)
                if minutes is None:
                    continue
                normalized = annotate_action_items(
                    str(meeting["id"]), minutes
                )
            except ActionItemsError as exc:
                LOGGER.warning(
                    "Skipping actions for meeting %s: %s",
                    meeting.get("id"),
                    exc,
                )
                continue
            for item in normalized["action_items"]:
                action = {
                    "id": item["action_id"],
                    "action_id": item["action_id"],
                    "meeting_id": str(meeting["id"]),
                    "title": str(meeting.get("title") or "未命名会议"),
                    "meeting_title": str(
                        meeting.get("title") or "未命名会议"
                    ),
                    "date": str(meeting.get("meeting_date") or ""),
                    "meeting_date": str(
                        meeting.get("meeting_date") or ""
                    ),
                    "lifecycle": str(
                        meeting.get("lifecycle_state") or "active"
                    ),
                    "lifecycle_state": str(
                        meeting.get("lifecycle_state") or "active"
                    ),
                    "meeting_status": str(
                        meeting.get("status") or ""
                    ),
                    "owner": item["owner"],
                    "task": item["task"],
                    "due": item["due"],
                    "evidence_time": item["evidence_time"],
                    "status": item["status"],
                    "status_label": ACTION_STATUS_LABELS[item["status"]],
                    "is_overdue": is_action_overdue(item, reference),
                }
                if wanted_status and action["status"] != wanted_status:
                    continue
                if (
                    wanted_lifecycle
                    and action["lifecycle"] != wanted_lifecycle
                ):
                    continue
                if normalized_query and not _action_matches(
                    action, normalized_query
                ):
                    continue
                aggregated.append(action)

        aggregated.sort(key=_action_sort_key)
        return aggregated

    def ensure_meeting_actions(
        self,
        meeting_id: str,
    ) -> dict[str, Any]:
        """Persist canonical statuses and stable IDs for one legacy meeting."""

        meeting = self.database.get_meeting(meeting_id)
        if meeting is None:
            raise ActionItemsError(f"会议 {meeting_id} 不存在")
        destination = self.storage.path(meeting, "minutes.json")
        lock = _file_lock(destination)
        with lock:
            minutes = self._read_minutes(meeting, missing_ok=False)
            assert minutes is not None
            normalized = annotate_action_items(meeting_id, minutes)
            if normalized != minutes:
                try:
                    self.storage.write_json(
                        meeting, "minutes.json", normalized
                    )
                except OSError as exc:
                    raise ActionItemsError(
                        f"无法规范化会议“{meeting['title']}”的待办：{exc}"
                    ) from exc
            return normalized

    def update_status(
        self,
        meeting_id: str,
        action_id: str,
        status: str,
        *,
        on_updated: Callable[
            [dict[str, Any], dict[str, Any]], None
        ]
        | None = None,
    ) -> dict[str, Any]:
        """Atomically update minutes.json and return it for export rendering."""

        canonical_status = normalize_action_status(status)
        clean_action_id = str(action_id or "").strip()
        if not clean_action_id:
            raise ActionItemsError("待办编号不能为空")
        meeting = self.database.get_meeting(meeting_id)
        if meeting is None:
            raise ActionItemsError(f"会议 {meeting_id} 不存在")

        destination = self.storage.path(meeting, "minutes.json")
        lock = _file_lock(destination)
        with lock:
            minutes = self._read_minutes(meeting, missing_ok=False)
            assert minutes is not None
            normalized = annotate_action_items(meeting_id, minutes)
            for item in normalized["action_items"]:
                if item["action_id"] == clean_action_id:
                    item["status"] = canonical_status
                    break
            else:
                raise ActionItemsError(
                    f"会议“{meeting['title']}”中不存在待办 "
                    f"{clean_action_id}"
                )
            normalized["updated_at"] = utc_now()
            try:
                if on_updated is None:
                    self.storage.write_json(
                        meeting, "minutes.json", normalized
                    )
                else:
                    on_updated(minutes, normalized)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ActionItemsError(
                    f"无法保存会议“{meeting['title']}”的待办状态：{exc}"
                ) from exc
            try:
                self.database.touch_meeting(meeting_id)
            except Exception as exc:
                LOGGER.warning(
                    "Action status was saved, but meeting %s "
                    "could not be touched: %s",
                    meeting_id,
                    exc,
                )
            return normalized

    def _read_minutes(
        self,
        meeting: dict[str, Any],
        *,
        missing_ok: bool,
    ) -> dict[str, Any] | None:
        try:
            source = self.storage.path(meeting, "minutes.json")
            exists = source.exists()
        except (OSError, ValueError) as exc:
            raise ActionItemsError(
                f"无法定位会议“{meeting['title']}”的纪要文件：{exc}"
            ) from exc
        if not exists:
            if missing_ok:
                return None
            raise ActionItemsError(
                f"会议“{meeting['title']}”尚无纪要文件"
            )
        try:
            value = self.storage.read_json(meeting, "minutes.json")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ActionItemsError(
                f"会议“{meeting['title']}”的纪要文件损坏或无法读取：{exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ActionItemsError(
                f"会议“{meeting['title']}”的纪要格式无效："
                "根内容必须是对象"
            )
        return value


def _copy_minutes(minutes: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(minutes, dict):
        raise ActionItemsError("纪要格式无效：根内容必须是对象")
    return copy.deepcopy(minutes)


def _read_action_list(minutes: dict[str, Any]) -> list[Any]:
    actions = minutes.get("action_items", [])
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ActionItemsError("纪要格式无效：action_items 必须是数组")
    return actions


def _normalize_action_item(value: Any) -> dict[str, Any]:
    item = copy.deepcopy(value) if isinstance(value, dict) else {"task": value}
    item["owner"] = _display_text(item.get("owner"), "未明确")
    item["task"] = _display_text(item.get("task"), "待确认")
    item["due"] = _display_text(item.get("due"), "未明确")
    item["evidence_time"] = _display_text(
        item.get("evidence_time") or item.get("time"),
        "未明确",
    )
    item.pop("time", None)
    return item


def _existing_status(value: Any) -> str:
    try:
        return normalize_action_status(value or "pending")
    except ActionItemsError:
        return "pending"


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _display_text(value: Any, fallback: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _WHITESPACE.sub(" ", text).strip() or fallback


def _optional_status_filter(value: str | None) -> str | None:
    clean = str(value or "").strip()
    if not clean or clean.casefold() in {"all", "全部"}:
        return None
    return normalize_action_status(clean)


def _optional_lifecycle_filter(value: str | None) -> str | None:
    clean = str(value or "").strip().casefold()
    if not clean or clean in {"all", "全部"}:
        return None
    if clean not in {"active", "archived", "trashed"}:
        raise ActionItemsError(
            "无效的会议范围；可选：active、archived、trashed"
        )
    return clean


def _coerce_today(value: date | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ActionItemsError("today 必须是 YYYY-MM-DD 日期") from exc


def _parse_due_date(value: Any) -> date | None:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    match = _ISO_DATE.search(text)
    if match is None:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _action_matches(action: dict[str, Any], query: str) -> bool:
    searchable = (
        "title",
        "owner",
        "task",
        "due",
        "evidence_time",
        "status",
        "status_label",
    )
    combined = " ".join(
        _normalized_text(action.get(key)) for key in searchable
    )
    keywords = tuple(
        part for part in query.split() if part
    )
    return all(keyword in combined for keyword in keywords)


def _action_sort_key(action: dict[str, Any]) -> tuple[Any, ...]:
    status_rank = {
        "pending": 0,
        "done": 1,
        "dismissed": 2,
    }
    due = _parse_due_date(action.get("due"))
    meeting_date = _parse_due_date(action.get("date"))
    return (
        status_rank.get(str(action.get("status")), 3),
        0 if action.get("is_overdue") else 1,
        due or date.max,
        -(meeting_date.toordinal() if meeting_date else 0),
        _normalized_text(action.get("title")),
        str(action.get("meeting_id") or ""),
        str(action.get("action_id") or ""),
    )


def _file_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(resolved, threading.RLock())


def minutes_file_lock(
    storage: MeetingStorage,
    meeting: dict[str, Any],
) -> threading.RLock:
    """Return the shared process-local lock for one minutes source file."""

    return _file_lock(storage.path(meeting, "minutes.json"))
