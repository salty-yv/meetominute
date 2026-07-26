from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

import app.diagnostics as diagnostics
import app.main as main_module
from app.config import Settings
from app.database import Database
from app.diagnostics import DiagnosticCheck, DiagnosticReport


def _settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("MEETOMINUTE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    return Settings.from_env()


def test_report_rolls_up_warning_and_serializes() -> None:
    report = DiagnosticReport(
        checks=(
            DiagnosticCheck(
                key="python",
                title="Python",
                status="ok",
                summary="正常",
            ),
            DiagnosticCheck(
                key="disk",
                title="磁盘",
                status="warning",
                summary="空间偏少",
                hint="清理磁盘",
            ),
        ),
        generated_at="2026-07-26T00:00:00+00:00",
        duration_ms=42,
    )

    assert report.overall == "warning"
    assert report.counts == {"ok": 1, "warning": 1, "error": 0, "info": 0}
    payload = report.to_dict()
    assert payload["status"] == "warning"
    assert payload["checks"][1]["hint"] == "清理磁盘"


def test_storage_and_database_checks_are_ready(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()

    storage_check = diagnostics._check_storage(settings)
    database_check = diagnostics._check_database(database)

    assert storage_check.status in {"ok", "warning"}
    assert "目录可写" in storage_check.summary
    assert database_check.status == "ok"
    assert "0 场会议" in database_check.summary


def test_missing_ffmpeg_has_actionable_hint(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics, "_resolve_executable", lambda _command: None
    )

    check = diagnostics._check_ffmpeg(settings)

    assert check.status == "error"
    assert "ffmpeg" in check.summary
    assert "PATH" in check.hint


def test_pytorch_cuda_probe_uses_disposable_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    monkeypatch.setattr(
        diagnostics.importlib.metadata,
        "version",
        lambda name: "2.6.0" if name == "torch" else "1.0",
    )
    payload = {
        "torch_version": "2.6.0+cu124",
        "cuda_build": "12.4",
        "cuda_available": True,
        "device_count": 1,
        "devices": ["Test RTX"],
    }
    monkeypatch.setattr(
        diagnostics.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    check = diagnostics._check_pytorch(settings)

    assert check.status == "ok"
    assert "Test RTX" in check.summary
    assert "12.4" in check.detail

    payload.update(
        cuda_available=False,
        device_count=0,
        devices=[],
    )
    inactive_check = diagnostics._check_pytorch(settings)
    assert inactive_check.status == "info"
    assert "无需使用 CUDA" in inactive_check.hint


def test_diagnostics_page_and_api(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    report = DiagnosticReport(
        checks=(
            DiagnosticCheck(
                key="python",
                title="Python 与虚拟环境",
                status="ok",
                summary="Python 3.11 · 项目虚拟环境",
                detail="测试路径",
            ),
        ),
        generated_at="2026-07-26T00:00:00+00:00",
        duration_ms=12,
    )
    monkeypatch.setattr(
        main_module, "run_diagnostics", lambda *_args: report
    )
    application = main_module.create_app(settings)

    with TestClient(application) as client:
        page = client.get("/diagnostics")
        assert page.status_code == 200
        assert "运行诊断" in page.text
        assert "Python 3.11 · 项目虚拟环境" in page.text
        assert "复制诊断信息" in page.text

        response = client.get("/api/diagnostics")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["checks"][0]["key"] == "python"
