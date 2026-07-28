from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.backups import BackupError, BackupManager
from app.config import Settings
from app.database import Database
from app.main import create_app
from app.maintenance import MaintenanceBusyError, MaintenanceGate
from app.storage import MeetingStorage


def test_maintenance_gate_is_exclusive_and_restores_state() -> None:
    gate = MaintenanceGate()

    with gate.maintenance("正在测试维护"):
        assert gate.state.operation == "正在测试维护"
        assert gate.state.active_mutations == 0
        with pytest.raises(MaintenanceBusyError, match="正在测试维护"):
            with gate.mutation():
                pass

    assert gate.state.operation is None
    with gate.mutation():
        assert gate.state.active_mutations == 1
        with pytest.raises(MaintenanceBusyError, match="数据变更"):
            with gate.maintenance("不会进入"):
                pass
    assert gate.state.active_mutations == 0


def test_background_mutation_waits_for_maintenance() -> None:
    gate = MaintenanceGate()
    entered = threading.Event()

    def background_work() -> None:
        with gate.mutation(wait=True):
            entered.set()

    with gate.maintenance("正在测试维护"):
        worker = threading.Thread(target=background_work)
        worker.start()
        assert not entered.wait(timeout=0.05)

    worker.join(timeout=1)
    assert entered.is_set()
    assert not worker.is_alive()


def test_backup_manager_uses_shared_maintenance_gate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "data")
    )
    settings = Settings.from_env()
    settings.ensure_directories()
    database = Database(settings.db_path)
    database.initialize()
    gate = MaintenanceGate()
    manager = BackupManager(
        settings,
        database,
        MeetingStorage(settings),
        maintenance_gate=gate,
    )

    with gate.mutation():
        with pytest.raises(BackupError, match="数据变更"):
            manager.create_backup()
        with pytest.raises(BackupError, match="数据变更"):
            manager.restore_backup(tmp_path / "unused.zip")


def test_application_blocks_mutations_during_maintenance(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "MEETOMINUTE_DATA_DIR", str(tmp_path / "app-data")
    )
    monkeypatch.setenv("MEETOMINUTE_LOCAL_TRANSCRIBER", "mock")
    monkeypatch.setenv("MEETOMINUTE_LOCAL_LLM", "mock")
    application = create_app(Settings.from_env())

    with TestClient(application) as client:
        gate = application.state.maintenance_gate
        with gate.maintenance("正在创建测试备份"):
            assert client.get("/").status_code == 200
            blocked_page = client.post("/minutes-templates")
            assert blocked_page.status_code == 409
            assert "正在创建测试备份" in blocked_page.text
            blocked_api = client.post(
                "/minutes-templates",
                headers={"Accept": "application/json"},
            )
            assert blocked_api.status_code == 409
            assert (
                "正在创建测试备份"
                in blocked_api.json()["detail"]
            )
            unrelated_settings_check = client.post(
                "/settings/external-llm/test"
            )
            assert unrelated_settings_check.status_code != 409
