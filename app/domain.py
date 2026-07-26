from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any


VALID_STATUSES = {
    "queued",
    "processing",
    "canceling",
    "canceled",
    "transcribed",
    "generating_minutes",
    "completed",
    "failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_timestamp(seconds: float | int | None) -> str:
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class Segment:
    id: str
    start: float
    end: float
    speaker: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = format_timestamp(self.start)
        return value


@dataclass(slots=True)
class MeetingCreate:
    title: str
    meeting_date: date
    expected_speakers: int
    glossary: str
    processing_mode: str
