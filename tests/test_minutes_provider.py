from __future__ import annotations

from app.config import Settings
from app.launcher import _ollama_health_url
from app.minutes_templates import default_minutes_template
from app.providers import (
    OpenAICompatibleMinutesGenerator,
    _compact_json,
    _normalize_minutes,
    _split_minutes_result,
    _transcript_fallback_summary,
    create_minutes_generator,
)


class RecordingMinutesGenerator(OpenAICompatibleMinutesGenerator):
    def __init__(self, settings: Settings):
        super().__init__(
            settings,
            base_url="http://provider.test/v1",
            model="test-model",
            api_key="test-key",
        )
        self.prompts: list[tuple[str, str]] = []
        self.extraction_calls = 0
        self.intermediate_calls = 0
        self.final_calls = 0

    def _chat_json(
        self,
        system: str,
        user: str,
        *,
        cancel_check=None,
    ) -> dict:
        self.prompts.append((system, user))
        if user.startswith("这是第"):
            self.extraction_calls += 1
            return {
                "summary": (
                    f"片段 {self.extraction_calls} 的明确事实。"
                    + "实验数据与讨论边界。" * 70
                ),
                "suggestions": [
                    {
                        "content": "继续验证已有数据。",
                        "evidence_time": "[00:00:01]",
                    }
                ],
                "custom_01": [
                    {
                        "content": "自定义章节事实",
                        "evidence_time": "[00:00:01]",
                    }
                ],
            }
        if user.startswith("把以下一批"):
            self.intermediate_calls += 1
            return {
                "summary": "阶段合并后的已有事实。",
                "suggestions": [],
                "custom_01": [
                    {
                        "content": "阶段性自定义事实",
                        "evidence_time": "[00:00:01]",
                    }
                ],
            }
        self.final_calls += 1
        return {
            "summary": "最终会议摘要",
            "suggestions": [],
            "action_items": [],
            "custom_01": [
                {
                    "content": "最终自定义章节事实",
                    "evidence_time": "[00:00:01]",
                }
            ],
        }


def _meeting() -> dict:
    return {
        "title": "长会议测试",
        "meeting_date": "2026-07-28",
        "expected_speakers": 2,
    }


def _segments(count: int, text_chars: int = 120) -> list[dict]:
    return [
        {
            "speaker": "SPEAKER_01",
            "timestamp": f"00:00:{index:02d}",
            "text": f"第 {index} 段明确讨论：" + "事实内容" * text_chars,
        }
        for index in range(count)
    ]


def test_ollama_generator_uses_separate_local_config(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "ollama")
    monkeypatch.setenv(
        "MEETOMINUTE_OLLAMA_BASE_URL",
        "http://127.0.0.1:11434/v1",
    )
    monkeypatch.setenv(
        "MEETOMINUTE_OLLAMA_MODEL", "meetominute-test"
    )
    settings = Settings.from_env(tmp_path)

    generator = create_minutes_generator(settings, "local")

    assert generator.name == "ollama"
    assert generator.model == "meetominute-test"
    assert generator.base_url == "http://127.0.0.1:11434/v1"


def test_minutes_normalization_repairs_local_model_shapes() -> None:
    normalized = _normalize_minutes(
        {
            "summary": ["第一点", "第二点"],
            "suggestions": [
                {"content": "建议内容", "time": "[00:01:02]"}
            ],
            "action_items": [
                {
                    "task": "复现实验",
                    "time": "[00:02:03]-[00:02:20]",
                }
            ],
        }
    )

    assert normalized["summary"] == "1. 第一点\n2. 第二点"
    assert (
        normalized["suggestions"][0]["evidence_time"]
        == "[00:01:02]"
    )
    assert (
        normalized["action_items"][0]["evidence_time"]
        == "[00:02:03]-[00:02:20]"
    )
    assert normalized["action_items"][0]["owner"] == "未明确"


def test_ollama_health_url_removes_openai_suffix() -> None:
    assert (
        _ollama_health_url("http://127.0.0.1:11434/v1")
        == "http://127.0.0.1:11434/api/version"
    )


def test_transcript_fallback_summary_is_extractive() -> None:
    summary = _transcript_fallback_summary(
        [
            {"text": "讨论开场节目。"},
            {"text": "需要准备老师服装。"},
        ]
    )

    assert "模型未形成进一步概括" in summary
    assert "讨论开场节目" in summary
    assert "准备老师服装" in summary


def test_short_meeting_keeps_single_extract_and_final_merge(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MEETOMINUTE_LLM_INPUT_CHAR_BUDGET", "6000")
    settings = Settings.from_env(tmp_path)
    generator = RecordingMinutesGenerator(settings)

    result = generator.generate(
        _meeting(),
        _segments(1, text_chars=8),
        {"SPEAKER_01": "成员甲"},
    )

    assert generator.extraction_calls == 1
    assert generator.intermediate_calls == 0
    assert generator.final_calls == 1
    assert result["summary"] == "最终会议摘要"
    assert all(
        len(system) + len(user)
        <= settings.llm_input_char_budget
        for system, user in generator.prompts
    )


def test_many_chunks_use_bounded_hierarchical_merge(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MEETOMINUTE_LLM_CHUNK_CHARS", "220")
    monkeypatch.setenv("MEETOMINUTE_LLM_INPUT_CHAR_BUDGET", "4000")
    settings = Settings.from_env(tmp_path)
    generator = RecordingMinutesGenerator(settings)

    result = generator.generate(
        _meeting(),
        _segments(30, text_chars=35),
        {"SPEAKER_01": "成员甲"},
    )

    assert generator.extraction_calls >= 15
    assert generator.intermediate_calls > 0
    assert generator.final_calls == 1
    assert len(generator.prompts) < generator.extraction_calls * 2 + 2
    assert all(
        len(system) + len(user)
        <= settings.llm_input_char_budget
        for system, user in generator.prompts
    )
    expected_keys = {
        section["key"]
        for section in default_minutes_template()["sections"]
    }
    assert expected_keys <= result.keys()
    assert {
        section["key"] for section in result["template"]["sections"]
    } == expected_keys


def test_long_custom_template_is_compacted_but_keeps_structure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MEETOMINUTE_LLM_INPUT_CHAR_BUDGET", "4000")
    settings = Settings.from_env(tmp_path)
    generator = RecordingMinutesGenerator(settings)
    template = default_minutes_template()
    template["id"] = "long-custom"
    template["name"] = "长模板"
    template["is_builtin"] = False
    template["instructions"] = "只整理明确事实。" * 1000
    for section in template["sections"]:
        section["guidance"] = "保留证据，不得推测。" * 100
    template["sections"].extend(
        {
            "key": f"custom_{index:02d}",
            "title": f"自定义章节 {index}",
            "kind": "list",
            "guidance": "提取明确出现的自定义事实。" * 70,
        }
        for index in range(1, 9)
    )

    result = generator.generate(
        _meeting(),
        _segments(1, text_chars=8),
        {"SPEAKER_01": "成员甲"},
        template=template,
    )

    assert all(
        len(system) + len(user)
        <= settings.llm_input_char_budget
        for system, user in generator.prompts
    )
    assert all(
        '"custom_08"' in user for _, user in generator.prompts
    )
    assert "custom_08" in result
    assert result["template"]["sections"][-1]["key"] == "custom_08"
    assert all(
        "不得引入输入中不存在的信息" in system
        for system, user in generator.prompts
        if not user.startswith("这是第")
    )


def test_large_result_sharding_preserves_first_and_last_evidence() -> None:
    template = default_minutes_template()
    fragments = _split_minutes_result(
        {
            "summary": "明确摘要",
            "suggestions": [
                {
                    "content": "FIRST_EVIDENCE-" + "甲" * 500,
                    "evidence_time": "[00:00:01]",
                },
                {
                    "content": "LAST_EVIDENCE-" + "乙" * 500,
                    "evidence_time": "[01:59:59]",
                },
            ],
        },
        template,
        max_chars=260,
    )

    groups: dict[str, list[dict]] = {}
    for fragment in fragments:
        for items in fragment.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                group_id = item.get("__fragment_group_id")
                if group_id:
                    groups.setdefault(str(group_id), []).append(item)

    assert len(groups) == 2
    reconstructed: list[str] = []
    for items in groups.values():
        totals = {int(item["__fragment_total"]) for item in items}
        assert len(totals) == 1
        total = totals.pop()
        assert sorted(
            int(item["__fragment_index"]) for item in items
        ) == list(range(1, total + 1))
        reconstructed.append(
            "".join(
                str(item["__raw_json_fragment"])
                for item in sorted(
                    items,
                    key=lambda value: int(
                        value["__fragment_index"]
                    ),
                )
            )
        )

    serialized = "\n".join(reconstructed)
    assert "FIRST_EVIDENCE" in serialized
    assert "LAST_EVIDENCE" in serialized
    assert "[00:00:01]" in serialized
    assert "[01:59:59]" in serialized
    assert all(len(_compact_json(item)) <= 260 for item in fragments)
