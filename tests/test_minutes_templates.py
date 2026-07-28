from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.minutes_templates import (
    MinutesTemplateError,
    build_minutes_template_from_form,
)
from app.providers import (
    OpenAICompatibleMinutesGenerator,
    _normalize_minutes,
)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 16_000)
    return buffer.getvalue()


def _wait_for_completed(database, meeting_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        meeting = database.get_meeting(meeting_id)
        if meeting["status"] == "completed":
            return meeting
        if meeting["status"] == "failed":
            raise AssertionError(meeting["error"])
        time.sleep(0.1)
    raise AssertionError("等待模板会议处理完成超时")


def _custom_form() -> dict[str, str]:
    return {
        "name": "客户需求评审",
        "description": "用于需求确认和方案评审",
        "instructions": "重点区分客户确认项与尚待讨论的建议。",
        "title_summary": "评审摘要",
        "include_decisions": "on",
        "title_decisions": "客户确认事项",
        "include_action_items": "on",
        "title_action_items": "交付任务",
        "custom_sections": (
            "客户原话 | 记录客户对需求和体验的直接表述\n"
            "验收条件 | 提取明确的验收标准和版本范围"
        ),
    }


def test_template_builder_supports_custom_sections_and_constraints() -> None:
    template = build_minutes_template_from_form(_custom_form())

    assert template["name"] == "客户需求评审"
    assert [section["key"] for section in template["sections"]] == [
        "summary",
        "decisions",
        "action_items",
        "custom_01",
        "custom_02",
    ]
    assert template["sections"][0]["title"] == "评审摘要"
    assert template["sections"][3]["title"] == "客户原话"

    invalid = _custom_form()
    invalid["name"] = ""
    try:
        build_minutes_template_from_form(invalid)
    except MinutesTemplateError as exc:
        assert "名称不能为空" in str(exc)
    else:
        raise AssertionError("空模板名称应被拒绝")


def test_template_management_and_generation_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    settings = Settings.from_env()
    application = create_app(settings)

    with TestClient(application) as client:
        page = client.get("/minutes-templates")
        assert page.status_code == 200
        assert "自定义纪要模板" in page.text
        assert "实验室组会" in page.text

        created = client.post(
            "/minutes-templates",
            data=_custom_form(),
            follow_redirects=False,
        )
        assert created.status_code == 303
        custom = next(
            template
            for template in application.state.database.list_minutes_templates()
            if not template["is_builtin"]
        )

        edit_page = client.get(
            f"/minutes-templates?edit={custom['id']}"
        )
        assert edit_page.status_code == 200
        assert "客户原话" in edit_page.text

        updated_form = _custom_form()
        updated_form["title_decisions"] = "最终确认事项"
        updated = client.post(
            f"/minutes-templates/{custom['id']}",
            data=updated_form,
            follow_redirects=False,
        )
        assert updated.status_code == 303
        custom = application.state.database.get_minutes_template(
            custom["id"]
        )
        assert custom is not None
        assert custom["sections"][1]["title"] == "最终确认事项"

        response = client.post(
            "/meetings",
            data={
                "title": "客户评审测试",
                "meeting_date": "2026-07-28",
                "expected_speakers": "2",
                "glossary": "",
                "processing_mode": "local",
                "minutes_template_id": custom["id"],
            },
            files={
                "recording": (
                    "sample.wav",
                    _wav_bytes(),
                    "audio/wav",
                )
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        meeting = application.state.database.list_meetings()[0]
        completed = _wait_for_completed(
            application.state.database, meeting["id"]
        )
        assert completed["minutes_template_id"] == custom["id"]

        minutes_path = (
            settings.meetings_dir
            / completed["slug"]
            / "minutes.json"
        )
        minutes = json.loads(minutes_path.read_text(encoding="utf-8"))
        assert minutes["template"]["id"] == custom["id"]
        assert minutes["template"]["name"] == "客户需求评审"
        assert "custom_01" in minutes

        detail = client.get(f"/meetings/{meeting['id']}")
        assert detail.status_code == 200
        assert "评审摘要" in detail.text
        assert "客户原话" in detail.text
        assert "交付任务" in detail.text

        markdown = client.get(
            f"/meetings/{meeting['id']}/download/md"
        )
        assert markdown.status_code == 200
        assert "## 客户原话" in markdown.text
        assert "## 各成员工作进展" not in markdown.text

        blocked_delete = client.post(
            f"/minutes-templates/{custom['id']}/delete",
            follow_redirects=False,
        )
        assert blocked_delete.status_code == 409
        assert "仍被会议使用" in blocked_delete.text

        duplicate = client.post(
            f"/minutes-templates/{custom['id']}/duplicate",
            follow_redirects=False,
        )
        assert duplicate.status_code == 303
        templates = application.state.database.list_minutes_templates()
        copied = next(
            template
            for template in templates
            if template["name"].endswith("副本")
        )
        deleted = client.post(
            f"/minutes-templates/{copied['id']}/delete",
            follow_redirects=False,
        )
        assert deleted.status_code == 303
        assert (
            application.state.database.get_minutes_template(copied["id"])
            is None
        )

        alternate = build_minutes_template_from_form(
            {
                "name": "精简复盘",
                "title_summary": "复盘摘要",
                "include_action_items": "on",
                "title_action_items": "下一步行动",
            }
        )
        application.state.database.create_minutes_template(alternate)
        regenerated = client.post(
            f"/meetings/{meeting['id']}/minutes",
            data={"minutes_template_id": alternate["id"]},
            follow_redirects=False,
        )
        assert regenerated.status_code == 303
        _wait_for_completed(
            application.state.database, meeting["id"]
        )
        changed = application.state.database.get_meeting(meeting["id"])
        assert changed["minutes_template_id"] == alternate["id"]
        regenerated_minutes = json.loads(
            minutes_path.read_text(encoding="utf-8")
        )
        assert (
            regenerated_minutes["template"]["id"]
            == alternate["id"]
        )
        removed_original = client.post(
            f"/minutes-templates/{custom['id']}/delete",
            follow_redirects=False,
        )
        assert removed_original.status_code == 303


def test_dynamic_minutes_normalization_keeps_template_snapshot() -> None:
    template = build_minutes_template_from_form(_custom_form())
    normalized = _normalize_minutes(
        {
            "summary": "讨论了客户需求。",
            "decisions": [],
            "action_items": [
                {
                    "owner": "李工",
                    "task": "更新方案",
                    "time": "00:03:00",
                }
            ],
            "custom_01": [
                {"content": "希望减少操作步骤", "time": "00:01:20"}
            ],
            "custom_02": [],
        },
        template,
    )

    assert normalized["template"]["id"] == template["id"]
    assert (
        normalized["custom_01"][0]["evidence_time"] == "00:01:20"
    )
    assert normalized["action_items"][0]["owner"] == "李工"
    assert normalized["action_items"][0]["status"] == "待处理"


def test_openai_generator_includes_custom_template_in_both_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    template = build_minutes_template_from_form(_custom_form())
    settings = Settings.from_env(tmp_path)
    generator = OpenAICompatibleMinutesGenerator(
        settings,
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key="test-key",
    )
    calls: list[tuple[str, str]] = []

    def fake_chat_json(system, user, *, cancel_check=None):
        calls.append((system, user))
        return {
            "summary": "讨论了需求。",
            "decisions": [],
            "action_items": [],
            "custom_01": [],
            "custom_02": [],
        }

    monkeypatch.setattr(generator, "_chat_json", fake_chat_json)
    result = generator.generate(
        {
            "title": "需求评审",
            "meeting_date": "2026-07-28",
            "expected_speakers": 2,
        },
        [
            {
                "start": 0,
                "speaker": "SPEAKER_01",
                "text": "需要减少操作步骤。",
            }
        ],
        {"SPEAKER_01": "客户"},
        template=template,
    )

    assert len(calls) == 2
    assert all("客户需求评审" in user for _, user in calls)
    assert all("custom_01" in user for _, user in calls)
    assert result["template"]["id"] == template["id"]
