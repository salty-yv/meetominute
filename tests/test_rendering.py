from __future__ import annotations

from pathlib import Path

from docx import Document

from app.minutes_templates import (
    build_minutes_template_from_form,
    minutes_template_snapshot,
)
from app.rendering import (
    render_minutes_docx,
    render_minutes_markdown,
    render_transcript_markdown,
    render_transcript_text,
)


def sample_minutes() -> dict:
    return {
        "meeting": {"title": "测试组会", "date": "2026-07-26"},
        "summary": "讨论了实验进展。",
        "member_progress": [
            {"content": "完成数据清洗", "evidence_time": "00:01:00"}
        ],
        "experimental_results": [],
        "suggestions": [],
        "decisions": [
            {"decision": "采用方案 A", "evidence_time": "00:03:00"}
        ],
        "action_items": [
            {
                "owner": "张三",
                "task": "补充实验",
                "due": "未明确",
                "evidence_time": "00:04:00",
                "status": "待处理",
            }
        ],
        "open_questions": [],
        "next_followups": [],
    }


def test_minutes_markdown_has_traceable_action() -> None:
    rendered = render_minutes_markdown(sample_minutes())
    assert "# 测试组会" in rendered
    assert "张三" in rendered
    assert "00:04:00" in rendered
    assert "未明确" in rendered


def test_docx_is_readable(tmp_path: Path) -> None:
    destination = tmp_path / "minutes.docx"
    render_minutes_docx(sample_minutes(), destination)
    document = Document(destination)
    all_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "测试组会" in all_text
    assert destination.stat().st_size > 1000


def test_custom_template_controls_exported_sections() -> None:
    template = build_minutes_template_from_form(
        {
            "name": "评审模板",
            "title_summary": "评审结论",
            "include_decisions": "on",
            "title_decisions": "确认事项",
            "custom_sections": "客户原话 | 保留客户直接表述",
        }
    )
    minutes = {
        "meeting": {"title": "需求评审", "date": "2026-07-28"},
        "summary": "确认了需求范围。",
        "decisions": [{"content": "采用方案 A"}],
        "custom_01": [{"content": "操作步骤需要更少"}],
        "template": minutes_template_snapshot(template),
    }

    rendered = render_minutes_markdown(minutes)

    assert "## 评审结论" in rendered
    assert "## 确认事项" in rendered
    assert "## 客户原话" in rendered
    assert "## 各成员工作进展" not in rendered


def test_transcript_resolves_speaker_name() -> None:
    meeting = {
        "title": "测试组会",
        "meeting_date": "2026-07-26",
        "expected_speakers": 2,
    }
    segments = [
        {
            "id": "seg_0001",
            "start": 2.5,
            "end": 4,
            "timestamp": "00:00:02",
            "speaker": "SPEAKER_01",
            "text": "你好。",
        }
    ]
    rendered = render_transcript_markdown(
        meeting, segments, {"SPEAKER_01": "李老师"}
    )
    assert "[00:00:02] 李老师" in rendered
    assert "你好。" in rendered

    plain_text = render_transcript_text(
        meeting, segments, {"SPEAKER_01": "李老师"}
    )
    assert plain_text.startswith("测试组会：录音转写\n")
    assert "[00:00:02] 李老师\n你好。" in plain_text
    assert "#" not in plain_text
