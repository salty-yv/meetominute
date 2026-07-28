from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from .database import Database, LIFECYCLE_STATES
from .domain import format_timestamp
from .minutes_templates import default_minutes_template
from .storage import MeetingStorage


SEARCH_SCOPES = {"all", "metadata", "transcript", "minutes"}
SEARCH_LIFECYCLE_STATES = {*LIFECYCLE_STATES, "all"}
DEFAULT_SEARCH_LIMIT = 50
MAX_SNIPPET_CHARS = 240

_ALL_MEETINGS_LIMIT = 2_147_483_647
_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOURCE_ORDER = {
    "title": 0,
    "date": 1,
    "recording": 2,
    "glossary": 3,
    "speaker": 4,
    "transcript": 5,
    "minutes": 6,
}
_METADATA_FIELDS = (
    ("title", "会议标题", "#recording"),
    ("date", "会议日期", "#recording"),
    ("recording", "录音文件", "#recording"),
    ("glossary", "术语表", "#recording"),
    ("speaker", "发言人", "#transcript"),
)
_MINUTES_RESERVED_KEYS = {
    "meeting",
    "template",
    "generator",
    "created_at",
    "updated_at",
    "version",
}
_MINUTES_INTERNAL_ITEM_KEYS = {
    "action_id",
    "id",
    "status",
}


class MeetingSearchService:
    """Search meeting metadata and generated artifacts without an index.

    This application is local and currently stores its editable source of
    truth in SQLite plus JSON files.  Keeping the search implementation here
    avoids a second index that could get out of sync after edits or restores.
    """

    def __init__(
        self,
        database: Database,
        storage: MeetingStorage,
    ) -> None:
        self.database = database
        self.storage = storage

    def search(
        self,
        query: str,
        *,
        scope: str = "all",
        lifecycle_state: str = "active",
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        if scope not in SEARCH_SCOPES:
            raise ValueError(f"无效的搜索范围：{scope}")
        if lifecycle_state not in SEARCH_LIFECYCLE_STATES:
            raise ValueError(f"无效的会议生命周期状态：{lifecycle_state}")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("搜索结果上限必须是正整数")

        keywords = _query_keywords(query)
        if not keywords:
            return []

        meetings = self.database.list_meetings(
            limit=_ALL_MEETINGS_LIMIT,
            lifecycle_state=(
                None if lifecycle_state == "all" else lifecycle_state
            ),
        )
        ranked_results: list[
            tuple[tuple[int, float, str, int, float, int], dict[str, Any]]
        ] = []
        for meeting in meetings:
            ranked_results.extend(
                self._search_meeting(meeting, keywords, scope)
            )

        ranked_results.sort(key=lambda item: item[0])
        return [result for _, result in ranked_results[:limit]]

    def _search_meeting(
        self,
        meeting: dict[str, Any],
        keywords: tuple[str, ...],
        scope: str,
    ) -> list[
        tuple[tuple[int, float, str, int, float, int], dict[str, Any]]
    ]:
        include_metadata = scope in {"all", "metadata"}
        include_transcript = scope in {"all", "transcript"}
        include_minutes = scope in {"all", "minutes"}
        speakers: dict[str, Any] = {}
        if include_metadata or include_transcript:
            raw_speakers = self._read_json(meeting, "speakers.json")
            if isinstance(raw_speakers, dict):
                speakers = raw_speakers

        candidates: list[dict[str, Any]] = []
        if include_metadata:
            metadata = self._metadata_candidate(meeting, speakers, keywords)
            if metadata is not None:
                candidates.append(metadata)
        if include_transcript:
            candidates.extend(
                self._transcript_candidates(meeting, speakers, keywords)
            )
        if include_minutes:
            candidates.extend(
                self._minutes_candidates(meeting, keywords)
            )

        meeting_rank = (
            -_date_ordinal(meeting.get("meeting_date")),
            -_datetime_timestamp(meeting.get("created_at")),
            str(meeting.get("id") or ""),
        )
        ranked: list[
            tuple[tuple[int, float, str, int, float, int], dict[str, Any]]
        ] = []
        for ordinal, candidate in enumerate(candidates):
            source = str(candidate["source"])
            result = self._result(meeting, candidate)
            rank = (
                *meeting_rank,
                _SOURCE_ORDER.get(source, len(_SOURCE_ORDER)),
                float(candidate.get("_evidence_seconds") or 0.0),
                ordinal,
            )
            ranked.append((rank, result))
        return ranked

    def _metadata_candidate(
        self,
        meeting: dict[str, Any],
        speakers: dict[str, Any],
        keywords: tuple[str, ...],
    ) -> dict[str, Any] | None:
        speaker_names = [
            _plain_text(display_name) or _plain_text(speaker_id)
            for speaker_id, display_name in speakers.items()
            if _plain_text(display_name) or _plain_text(speaker_id)
        ]
        values = {
            "title": _plain_text(meeting.get("title")),
            "date": _plain_text(meeting.get("meeting_date")),
            "recording": _plain_text(meeting.get("source_filename")),
            "glossary": _plain_text(meeting.get("glossary")),
            "speaker": "、".join(speaker_names),
        }
        searchable = " ".join(value for value in values.values() if value)
        if not _matches_all(searchable, keywords):
            return None

        matching_fields = [
            definition
            for definition in _METADATA_FIELDS
            if values[definition[0]]
            and any(
                keyword in _normalize(values[definition[0]])
                for keyword in keywords
            )
        ]
        if not matching_fields:
            return None
        source, source_label, anchor = max(
            matching_fields,
            key=lambda definition: (
                sum(
                    keyword in _normalize(values[definition[0]])
                    for keyword in keywords
                ),
                -_SOURCE_ORDER[definition[0]],
            ),
        )
        snippet_parts = [
            f"{label}：{values[field]}"
            for field, label, _ in matching_fields
        ]
        return {
            "source": source,
            "source_label": source_label,
            "snippet": _make_snippet(
                "；".join(snippet_parts),
                keywords,
            ),
            "anchor": anchor,
            "evidence_time": None,
        }

    def _transcript_candidates(
        self,
        meeting: dict[str, Any],
        speakers: dict[str, Any],
        keywords: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        transcript = self._read_json(
            meeting, "transcript_edited.json"
        )
        if not _has_segments(transcript):
            transcript = self._read_json(meeting, "transcript_raw.json")
        if not _has_segments(transcript):
            return []

        candidates: list[dict[str, Any]] = []
        for segment in transcript["segments"]:
            if not isinstance(segment, dict):
                continue
            text = _plain_text(segment.get("text"))
            speaker_id = _plain_text(segment.get("speaker"))
            speaker = _plain_text(speakers.get(speaker_id)) or speaker_id
            evidence_time, evidence_seconds = _segment_time(segment)
            searchable = " ".join(
                value
                for value in (speaker, speaker_id, evidence_time, text)
                if value
            )
            if not searchable or not _matches_all(searchable, keywords):
                continue
            prefix = " ".join(
                value for value in (evidence_time, speaker) if value
            )
            display = f"{prefix}：{text}" if prefix and text else (
                text or prefix
            )
            candidates.append(
                {
                    "source": "transcript",
                    "source_label": "逐字稿",
                    "snippet": _make_snippet(display, keywords),
                    "anchor": (
                        f"?seek={evidence_seconds:g}"
                        "&focus=transcript#transcript"
                    ),
                    "evidence_time": evidence_time or None,
                    "_evidence_seconds": evidence_seconds,
                }
            )
        return candidates

    def _minutes_candidates(
        self,
        meeting: dict[str, Any],
        keywords: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        minutes = self._read_json(meeting, "minutes.json")
        if not isinstance(minutes, dict):
            return []

        candidates: list[dict[str, Any]] = []
        for key, title in _minutes_sections(minutes):
            value = minutes.get(key)
            content = _flatten_text(value)
            if not content:
                continue
            safe_title = _plain_text(title) or _plain_text(key)
            searchable = " ".join(
                part for part in (safe_title, content) if part
            )
            if not searchable or not _matches_all(searchable, keywords):
                continue
            evidence_time, evidence_seconds = _minutes_evidence_time(
                value,
                keywords,
                section_title=safe_title,
            )
            display = (
                f"{safe_title}：{content}" if content else safe_title
            )
            candidates.append(
                {
                    "source": "minutes",
                    "source_label": f"会议纪要 · {safe_title}",
                    "snippet": _make_snippet(display, keywords),
                    "anchor": (
                        (
                            f"?seek={evidence_seconds:g}"
                            "&focus=minutes#minutes"
                        )
                        if evidence_time
                        else "#minutes"
                    ),
                    "evidence_time": evidence_time,
                    "_evidence_seconds": evidence_seconds,
                }
            )
        return candidates

    def _read_json(
        self,
        meeting: dict[str, Any],
        filename: str,
    ) -> Any:
        try:
            return self.storage.read_json(
                meeting, filename, default=None
            )
        except Exception:
            # A missing, partially-written, or manually damaged artifact must
            # not make search unavailable for every other meeting.
            return None

    @staticmethod
    def _result(
        meeting: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "meeting_id": str(meeting.get("id") or ""),
            "title": _plain_text(meeting.get("title")),
            "meeting_date": _plain_text(meeting.get("meeting_date")),
            "lifecycle_state": str(
                meeting.get("lifecycle_state") or "active"
            ),
            "source": str(candidate["source"]),
            "source_label": str(candidate["source_label"]),
            "snippet": str(candidate["snippet"]),
            "anchor": str(candidate["anchor"]),
            "evidence_time": candidate.get("evidence_time"),
        }


def _query_keywords(query: str) -> tuple[str, ...]:
    normalized = _normalize(query)
    return tuple(dict.fromkeys(part for part in normalized.split() if part))


def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _plain_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _CONTROL_CHARACTERS.sub(" ", text)
    # Full-width brackets preserve what the user typed while preventing the
    # snippet from becoming executable markup even in a careless renderer.
    text = text.replace("<", "＜").replace(">", "＞")
    return _WHITESPACE.sub(" ", text).strip()


def _matches_all(value: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize(value)
    return all(keyword in normalized for keyword in keywords)


def _make_snippet(
    value: Any,
    keywords: tuple[str, ...],
    max_chars: int = MAX_SNIPPET_CHARS,
) -> str:
    text = _plain_text(value)
    if len(text) <= max_chars:
        return text
    normalized = _normalize(text)
    positions = [
        normalized.find(keyword)
        for keyword in keywords
        if normalized.find(keyword) >= 0
    ]
    center = min(positions) if positions else 0
    content_budget = max(1, max_chars - 2)
    start = max(0, center - content_budget // 3)
    end = min(len(text), start + content_budget)
    start = max(0, end - content_budget)
    has_prefix = start > 0
    has_suffix = end < len(text)
    excerpt = text[start:end].strip()
    return (
        ("…" if has_prefix else "")
        + excerpt
        + ("…" if has_suffix else "")
    )


def _has_segments(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("segments"), list)
        and bool(value["segments"])
    )


def _segment_time(segment: dict[str, Any]) -> tuple[str, float]:
    raw_start = segment.get("start")
    try:
        seconds = max(0.0, float(raw_start))
    except (TypeError, ValueError):
        seconds = 0.0
    explicit = _plain_text(segment.get("timestamp"))
    return explicit or format_timestamp(seconds), seconds


def _minutes_sections(minutes: dict[str, Any]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    raw_template = minutes.get("template")
    raw_sections = (
        raw_template.get("sections")
        if isinstance(raw_template, dict)
        else None
    )
    if not isinstance(raw_sections, list):
        raw_sections = default_minutes_template()["sections"]

    seen: set[str] = set()
    for section in raw_sections:
        if not isinstance(section, dict):
            continue
        key = str(section.get("key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        sections.append(
            (key, str(section.get("title") or key).strip())
        )
    for key, value in minutes.items():
        if key in seen or key in _MINUTES_RESERVED_KEYS:
            continue
        if value is None or isinstance(value, (str, int, float, list, dict)):
            sections.append((str(key), str(key)))
            seen.add(str(key))
    return sections


def _flatten_text(value: Any) -> str:
    parts: list[str] = []
    pending = [value]
    while pending:
        current = pending.pop()
        if current is None:
            continue
        if isinstance(current, dict):
            pending.extend(
                reversed(
                    [
                        value
                        for key, value in current.items()
                        if key not in _MINUTES_INTERNAL_ITEM_KEYS
                    ]
                )
            )
        elif isinstance(current, (list, tuple)):
            pending.extend(reversed(current))
        else:
            text = _plain_text(current)
            if text:
                parts.append(text)
    return "；".join(parts)


def _minutes_evidence_time(
    value: Any,
    keywords: tuple[str, ...],
    *,
    section_title: str = "",
) -> tuple[str | None, float]:
    values = value if isinstance(value, list) else [value]
    fallback: tuple[str | None, float] = (None, 0.0)
    normalized_title = _normalize(section_title)
    item_keywords = tuple(
        keyword
        for keyword in keywords
        if keyword not in normalized_title
    )
    for item in values:
        if not isinstance(item, dict):
            continue
        evidence = _plain_text(item.get("evidence_time"))
        if not evidence:
            continue
        parsed = _timestamp_seconds(evidence)
        if fallback[0] is None:
            fallback = (evidence, parsed)
        if _matches_all(_flatten_text(item), item_keywords):
            return evidence, parsed
    return fallback


def _timestamp_seconds(value: str) -> float:
    match = re.search(r"(\d+):([0-5]?\d):([0-5]?\d)", value)
    if not match:
        return 0.0
    hours, minutes, seconds = (int(part) for part in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _date_ordinal(value: Any) -> int:
    try:
        return date.fromisoformat(str(value)).toordinal()
    except (TypeError, ValueError):
        return 0


def _datetime_timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OSError):
        return 0.0
