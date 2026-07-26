from __future__ import annotations

from app.config import Settings
from app.launcher import _ollama_health_url
from app.providers import (
    _normalize_minutes,
    _transcript_fallback_summary,
    create_minutes_generator,
)


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
