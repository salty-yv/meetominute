from __future__ import annotations

from types import SimpleNamespace

import app.launcher as launcher


class _ImmediateThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self) -> None:
        return None


def test_launcher_keeps_web_ui_available_when_ollama_fails(
    monkeypatch, capsys
) -> None:
    settings = SimpleNamespace(
        local_llm="ollama",
        cloud_llm="openai",
        host="127.0.0.1",
        port=8000,
    )
    uvicorn_calls: list[dict] = []
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        launcher.Settings, "from_env", lambda: settings
    )
    monkeypatch.setattr(
        launcher,
        "_ensure_ollama_ready",
        lambda _settings: (_ for _ in ()).throw(
            RuntimeError("测试启动失败")
        ),
    )
    monkeypatch.setattr(launcher.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        launcher.uvicorn,
        "run",
        lambda *args, **kwargs: uvicorn_calls.append(
            {"args": args, "kwargs": kwargs}
        ),
    )

    launcher.main()

    assert len(uvicorn_calls) == 1
    assert uvicorn_calls[0]["kwargs"]["port"] == 8000
    error_output = capsys.readouterr().err
    assert "Ollama 未就绪" in error_output
    assert "http://127.0.0.1:8000/diagnostics" in error_output
