from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.external_llm import (
    ConnectionTestResult,
    ExternalLLMConfig,
    ExternalLLMConfigStore,
    test_external_llm_connection as check_external_llm_connection,
)
from app.main import create_app
from app.providers import create_minutes_generator


def _settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("MEETOMINUTE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEETOMINUTE_LLM_MODEL", "")
    monkeypatch.setenv("MEETOMINUTE_OPENAI_API_KEY", "")
    return Settings.from_env()


def test_external_llm_secret_is_protected(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    store = ExternalLLMConfigStore(
        settings.data_dir / "external-llm.json", settings
    )
    saved = store.save(
        enabled=True,
        provider_name="测试网关",
        base_url="https://gateway.example/v1/chat/completions",
        model="test-model",
        api_key="sk-secret-value",
        reasoning_effort="low",
    )

    raw = store.path.read_text(encoding="utf-8")
    assert "sk-secret-value" not in raw
    assert json.loads(raw)["api_key_protected"].startswith(
        ("dpapi:", "base64:")
    )
    assert saved.base_url == "https://gateway.example/v1"
    assert store.load().api_key == "sk-secret-value"
    assert "api_key" not in store.load().public_dict()


def test_external_generator_uses_runtime_config(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    external = ExternalLLMConfig(
        enabled=True,
        provider_name="测试网关",
        base_url="https://gateway.example/v1",
        model="external-model",
        api_key="secret",
        reasoning_effort="",
    )
    generator = create_minutes_generator(
        settings, "mixed", external_llm=external
    )

    assert generator.name == "external-openai"
    assert generator.base_url == "https://gateway.example/v1"
    assert generator.model == "external-model"
    assert generator.api_key == "secret"


def test_connection_checks_models(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        request = httpx.Request("GET", args[0])
        return httpx.Response(
            200,
            request=request,
            json={"data": [{"id": "meeting-model"}]},
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    result = check_external_llm_connection(
        ExternalLLMConfig(
            enabled=True,
            provider_name="测试",
            base_url="https://gateway.example/v1",
            model="meeting-model",
            api_key="secret",
            reasoning_effort="",
        )
    )

    assert result.status == "success"
    assert result.model_count == 1


def test_settings_page_saves_without_exposing_key(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    application = create_app(settings)

    with TestClient(application) as client:
        page = client.get("/settings/external-llm")
        assert page.status_code == 200
        assert "连接外部 LLM" in page.text

        response = client.post(
            "/settings/external-llm",
            data={
                "enabled": "on",
                "provider_name": "团队网关",
                "base_url": "https://gateway.example/v1",
                "model": "meeting-model",
                "api_key": "sk-private",
                "reasoning_effort": "",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        public = client.get("/api/settings/external-llm").json()
        assert public["ready"] is True
        assert public["api_key_configured"] is True
        assert "api_key" not in public
        assert "sk-private" not in json.dumps(public)

        def fake_connection(candidate, timeout_seconds):
            assert candidate.api_key == "sk-private"
            return ConnectionTestResult(
                status="success",
                message="测试连接成功",
                latency_ms=12,
            )

        monkeypatch.setattr(
            "app.main.test_external_llm_connection", fake_connection
        )
        connection = client.post(
            "/settings/external-llm/test",
            data={
                "enabled": "on",
                "provider_name": "团队网关",
                "base_url": "https://gateway.example/v1",
                "model": "meeting-model",
                "api_key": "",
                "reasoning_effort": "",
            },
        )
        assert connection.status_code == 200
        assert connection.json()["status"] == "success"
