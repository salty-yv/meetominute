from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class MaintenanceBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaintenanceState:
    operation: str | None
    active_mutations: int


class MaintenanceGate:
    """Coordinates exclusive maintenance with normal data mutations."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._operation: str | None = None
        self._active_mutations = 0

    @property
    def state(self) -> MaintenanceState:
        with self._condition:
            return MaintenanceState(
                operation=self._operation,
                active_mutations=self._active_mutations,
            )

    @contextmanager
    def mutation(self, *, wait: bool = False) -> Iterator[None]:
        with self._condition:
            if self._operation is not None and not wait:
                raise MaintenanceBusyError(
                    f"{self._operation}，暂时不能修改数据，请稍后重试。"
                )
            while self._operation is not None:
                self._condition.wait()
            self._active_mutations += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_mutations -= 1
                self._condition.notify_all()

    @contextmanager
    def maintenance(self, operation: str) -> Iterator[None]:
        clean_operation = operation.strip() or "系统正在维护"
        with self._condition:
            if self._operation is not None:
                raise MaintenanceBusyError(
                    f"{self._operation}，请等待当前操作完成。"
                )
            if self._active_mutations:
                raise MaintenanceBusyError(
                    "当前有数据变更正在进行，请完成后再执行备份或恢复。"
                )
            self._operation = clean_operation
        try:
            yield
        finally:
            with self._condition:
                self._operation = None
                self._condition.notify_all()
