from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .config import Settings
from .database import Database
from .domain import utc_now


DiagnosticStatus = Literal["ok", "warning", "error", "info"]
_GIB = 1024**3
_MIB = 1024**2


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    key: str
    title: str
    status: DiagnosticStatus
    summary: str
    detail: str = ""
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    checks: tuple[DiagnosticCheck, ...]
    generated_at: str
    duration_ms: int

    @property
    def overall(self) -> DiagnosticStatus:
        statuses = {check.status for check in self.checks}
        if "error" in statuses:
            return "error"
        if "warning" in statuses:
            return "warning"
        return "ok"

    @property
    def counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in ("ok", "warning", "error", "info")
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall,
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
            "summary": self.counts,
            "checks": [check.to_dict() for check in self.checks],
        }


CheckFactory = Callable[[], DiagnosticCheck]


def run_diagnostics(
    settings: Settings,
    database: Database,
) -> DiagnosticReport:
    started = time.perf_counter()
    checkers: tuple[tuple[str, str, CheckFactory], ...] = (
        (
            "python",
            "Python 与虚拟环境",
            lambda: _check_python(settings),
        ),
        (
            "storage",
            "数据目录与磁盘",
            lambda: _check_storage(settings),
        ),
        (
            "database",
            "SQLite 数据库",
            lambda: _check_database(database),
        ),
        (
            "ffmpeg",
            "FFmpeg 音频工具",
            lambda: _check_ffmpeg(settings),
        ),
        (
            "pytorch",
            "PyTorch 与 CUDA",
            lambda: _check_pytorch(settings),
        ),
        (
            "funasr",
            "FunASR 转写环境",
            lambda: _check_funasr(settings),
        ),
        (
            "ollama",
            "Ollama 本地模型",
            lambda: _check_ollama(settings),
        ),
    )
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="meetominute-diagnostic"
    ) as executor:
        futures = [
            executor.submit(_guard_check, key, title, checker)
            for key, title, checker in checkers
        ]
        checks = tuple(future.result() for future in futures)
    return DiagnosticReport(
        checks=checks,
        generated_at=utc_now(),
        duration_ms=round((time.perf_counter() - started) * 1000),
    )


def _guard_check(
    key: str,
    title: str,
    checker: CheckFactory,
) -> DiagnosticCheck:
    try:
        return checker()
    except Exception as exc:
        return DiagnosticCheck(
            key=key,
            title=title,
            status="error",
            summary="诊断检查执行失败",
            detail=f"{type(exc).__name__}: {exc}",
            hint="请复制诊断信息和 processing.log 后再排查。",
        )


def _check_python(settings: Settings) -> DiagnosticCheck:
    version = ".".join(str(part) for part in sys.version_info[:3])
    executable = Path(sys.executable).resolve()
    expected_venv = (settings.base_dir / ".venv").resolve()
    project_venv = _is_relative_to(executable, expected_venv)
    any_venv = sys.prefix != sys.base_prefix
    detail = f"可执行文件：{executable}"

    if sys.version_info < (3, 11):
        return DiagnosticCheck(
            key="python",
            title="Python 与虚拟环境",
            status="error",
            summary=f"Python {version} 版本过低",
            detail=detail,
            hint="请在项目目录创建 Python 3.11 虚拟环境后重新安装依赖。",
        )
    if project_venv:
        return DiagnosticCheck(
            key="python",
            title="Python 与虚拟环境",
            status="ok",
            summary=f"Python {version} · 项目虚拟环境",
            detail=detail,
        )
    return DiagnosticCheck(
        key="python",
        title="Python 与虚拟环境",
        status="warning",
        summary=(
            f"Python {version} · 使用其他虚拟环境"
            if any_venv
            else f"Python {version} · 未使用虚拟环境"
        ),
        detail=detail,
        hint=f"建议使用 {expected_venv / 'Scripts' / 'python.exe'} 启动。",
    )


def _check_storage(settings: Settings) -> DiagnosticCheck:
    try:
        settings.ensure_directories()
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".diagnostic-",
            suffix=".tmp",
            dir=settings.data_dir,
            delete=True,
        ) as probe:
            probe.write(b"ok")
            probe.flush()
        usage = shutil.disk_usage(settings.data_dir)
    except OSError as exc:
        return DiagnosticCheck(
            key="storage",
            title="数据目录与磁盘",
            status="error",
            summary="数据目录不可写",
            detail=f"{settings.data_dir} · {exc}",
            hint="检查目录权限，或通过 MEETOMINUTE_DATA_DIR 更换数据目录。",
        )

    free = usage.free
    critical = max(settings.max_upload_bytes, _GIB)
    recommended = max(settings.max_upload_bytes * 2, 5 * _GIB)
    detail = (
        f"数据目录：{settings.data_dir} · "
        f"最大上传：{_format_bytes(settings.max_upload_bytes)}"
    )
    if free < critical:
        return DiagnosticCheck(
            key="storage",
            title="数据目录与磁盘",
            status="error",
            summary=f"空间不足 · 仅剩 {_format_bytes(free)}",
            detail=detail,
            hint="清理磁盘或迁移数据目录后再上传长录音。",
        )
    if free < recommended:
        return DiagnosticCheck(
            key="storage",
            title="数据目录与磁盘",
            status="warning",
            summary=f"目录可写 · 剩余 {_format_bytes(free)}",
            detail=detail,
            hint=(
                f"建议至少保留 {_format_bytes(recommended)}，"
                "用于原始录音、标准化音频和模型缓存。"
            ),
        )
    return DiagnosticCheck(
        key="storage",
        title="数据目录与磁盘",
        status="ok",
        summary=f"目录可写 · 剩余 {_format_bytes(free)}",
        detail=detail,
    )


def _check_database(database: Database) -> DiagnosticCheck:
    try:
        with database.connect() as connection:
            quick_check = connection.execute(
                "PRAGMA quick_check(1)"
            ).fetchone()[0]
            meeting_count = connection.execute(
                "SELECT COUNT(*) FROM meetings"
            ).fetchone()[0]
            journal_mode = connection.execute(
                "PRAGMA journal_mode"
            ).fetchone()[0]
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
    except Exception as exc:
        return DiagnosticCheck(
            key="database",
            title="SQLite 数据库",
            status="error",
            summary="数据库无法读取",
            detail=f"{database.path} · {type(exc).__name__}: {exc}",
            hint="先备份 data 目录，再检查数据库文件权限或完整性。",
        )
    if str(quick_check).lower() != "ok":
        return DiagnosticCheck(
            key="database",
            title="SQLite 数据库",
            status="error",
            summary="数据库完整性检查未通过",
            detail=f"{database.path} · {quick_check}",
            hint="停止应用并备份 data 目录，再从最近备份恢复数据库。",
        )
    return DiagnosticCheck(
        key="database",
        title="SQLite 数据库",
        status="ok",
        summary=f"数据库正常 · {meeting_count} 场会议",
        detail=(
            f"{database.path} · schema=v{schema_version} · "
            f"journal_mode={journal_mode}"
        ),
    )


def _check_ffmpeg(settings: Settings) -> DiagnosticCheck:
    ffmpeg = _resolve_executable(settings.ffmpeg_bin)
    ffprobe = _resolve_executable(settings.ffprobe_bin)
    missing = [
        name
        for name, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        if not path
    ]
    if missing:
        return DiagnosticCheck(
            key="ffmpeg",
            title="FFmpeg 音频工具",
            status="error",
            summary=f"缺少 {'、'.join(missing)}",
            detail=(
                f"FFmpeg 配置：{settings.ffmpeg_bin} · "
                f"ffprobe 配置：{settings.ffprobe_bin}"
            ),
            hint=(
                "安装 FFmpeg 并加入 PATH，或设置 "
                "MEETOMINUTE_FFMPEG 与 MEETOMINUTE_FFPROBE。"
            ),
        )
    try:
        ffmpeg_version = _command_version(ffmpeg)
        ffprobe_version = _command_version(ffprobe)
    except (OSError, subprocess.SubprocessError) as exc:
        return DiagnosticCheck(
            key="ffmpeg",
            title="FFmpeg 音频工具",
            status="error",
            summary="FFmpeg 命令无法执行",
            detail=str(exc),
            hint="检查可执行文件是否损坏，以及安全软件是否阻止运行。",
        )
    return DiagnosticCheck(
        key="ffmpeg",
        title="FFmpeg 音频工具",
        status="ok",
        summary="ffmpeg 与 ffprobe 均可执行",
        detail=f"{ffmpeg_version} · {ffprobe_version}",
    )


def _check_pytorch(settings: Settings) -> DiagnosticCheck:
    active = settings.local_transcriber == "funasr"
    try:
        installed_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return DiagnosticCheck(
            key="pytorch",
            title="PyTorch 与 CUDA",
            status="error" if active else "info",
            summary="PyTorch 未安装",
            hint=(
                "当前启用了 FunASR，请安装与显卡匹配的 CUDA 版 PyTorch。"
                if active
                else "当前转写后端不依赖 PyTorch。"
            ),
        )

    script = "\n".join(
        (
            "import json",
            "import torch",
            "payload = {",
            "  'torch_version': str(torch.__version__),",
            "  'cuda_build': str(torch.version.cuda or ''),",
            "  'cuda_available': bool(torch.cuda.is_available()),",
            "  'device_count': int(torch.cuda.device_count()),",
            "  'devices': [],",
            "}",
            "if payload['cuda_available']:",
            "  payload['devices'] = [",
            "    torch.cuda.get_device_name(index)",
            "    for index in range(payload['device_count'])",
            "  ]",
            "print(json.dumps(payload, ensure_ascii=False))",
        )
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            env=environment,
            creationflags=_no_window_flag(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DiagnosticCheck(
            key="pytorch",
            title="PyTorch 与 CUDA",
            status="error" if active else "warning",
            summary=f"PyTorch {installed_version} 探测失败",
            detail=str(exc),
            hint="运行 python -c \"import torch; print(torch.cuda.is_available())\" 排查。",
        )
    if completed.returncode != 0:
        error = completed.stderr.strip().splitlines()
        return DiagnosticCheck(
            key="pytorch",
            title="PyTorch 与 CUDA",
            status="error" if active else "warning",
            summary=f"PyTorch {installed_version} 无法加载",
            detail=error[-1] if error else "子进程未返回错误详情",
            hint="重新安装与当前 Python 和 CUDA 驱动匹配的 PyTorch。",
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return DiagnosticCheck(
            key="pytorch",
            title="PyTorch 与 CUDA",
            status="error" if active else "warning",
            summary="无法解析 PyTorch 探测结果",
            detail=str(exc),
            hint="检查 PyTorch 安装是否被其他启动脚本注入了额外输出。",
        )

    torch_version = str(payload.get("torch_version") or installed_version)
    cuda_build = str(payload.get("cuda_build") or "无")
    cuda_available = bool(payload.get("cuda_available"))
    devices = [str(value) for value in payload.get("devices") or []]
    detail = (
        f"PyTorch {torch_version} · CUDA 构建 {cuda_build} · "
        f"设备数 {payload.get('device_count', 0)}"
    )
    if cuda_available:
        return DiagnosticCheck(
            key="pytorch",
            title="PyTorch 与 CUDA",
            status="ok",
            summary=f"CUDA 可用 · {devices[0] if devices else 'NVIDIA GPU'}",
            detail=detail,
        )

    explicit_cuda = settings.funasr_device.startswith("cuda")
    status: DiagnosticStatus
    hint: str
    if not active:
        status = "info"
        hint = f"当前本地转写后端为 {settings.local_transcriber}，无需使用 CUDA。"
    elif explicit_cuda:
        status = "error"
        hint = "检查 NVIDIA 驱动和 CUDA 版 PyTorch 是否匹配。"
    else:
        status = "warning"
        hint = "FunASR 将回退到 CPU，处理速度会明显降低。"
    return DiagnosticCheck(
        key="pytorch",
        title="PyTorch 与 CUDA",
        status=status,
        summary=f"PyTorch {torch_version} 可用 · CUDA 不可用",
        detail=detail,
        hint=hint,
    )


def _check_funasr(settings: Settings) -> DiagnosticCheck:
    active = settings.local_transcriber == "funasr"
    try:
        funasr_version = importlib.metadata.version("funasr")
    except importlib.metadata.PackageNotFoundError:
        funasr_version = ""
    try:
        modelscope_version = importlib.metadata.version("modelscope")
    except importlib.metadata.PackageNotFoundError:
        modelscope_version = ""

    if not funasr_version:
        return DiagnosticCheck(
            key="funasr",
            title="FunASR 转写环境",
            status="error" if active else "info",
            summary="FunASR 未安装",
            detail=f"当前本地转写后端：{settings.local_transcriber}",
            hint=(
                "运行 pip install -e \".[dev,local-asr]\" 安装本地转写依赖。"
                if active
                else "当前配置不使用 FunASR，无需处理。"
            ),
        )

    cache_ready = False
    try:
        cache_ready = settings.models_dir.exists() and any(
            settings.models_dir.iterdir()
        )
    except OSError:
        cache_ready = False
    versions = f"FunASR {funasr_version}"
    if modelscope_version:
        versions += f" · ModelScope {modelscope_version}"
    detail = (
        f"{versions} · 设备 {settings.funasr_device} · "
        f"模型 {settings.funasr_model} / {settings.funasr_vad_model} / "
        f"{settings.funasr_punc_model} / {settings.funasr_spk_model}"
    )

    if not active:
        return DiagnosticCheck(
            key="funasr",
            title="FunASR 转写环境",
            status="info",
            summary=f"已安装但未启用 · 当前为 {settings.local_transcriber}",
            detail=detail,
        )
    if not cache_ready:
        return DiagnosticCheck(
            key="funasr",
            title="FunASR 转写环境",
            status="warning",
            summary="依赖已安装 · 模型缓存尚未就绪",
            detail=f"{detail} · 缓存目录 {settings.models_dir}",
            hint="首次转写会下载模型，请确认网络和磁盘空间充足。",
        )
    return DiagnosticCheck(
        key="funasr",
        title="FunASR 转写环境",
        status="ok",
        summary=f"FunASR {funasr_version} · 模型缓存已就绪",
        detail=f"{detail} · 缓存目录 {settings.models_dir}",
    )


def _check_ollama(settings: Settings) -> DiagnosticCheck:
    active = "ollama" in {settings.local_llm, settings.cloud_llm}
    binary = _resolve_executable(settings.ollama_bin)
    if not active:
        return DiagnosticCheck(
            key="ollama",
            title="Ollama 本地模型",
            status="info",
            summary=f"当前未启用 · 本地纪要后端为 {settings.local_llm}",
            detail=f"可执行文件：{binary or '未找到'}",
        )
    if not settings.ollama_model:
        return DiagnosticCheck(
            key="ollama",
            title="Ollama 本地模型",
            status="error",
            summary="未配置 Ollama 模型名称",
            detail=f"服务地址：{settings.ollama_base_url}",
            hint="设置 MEETOMINUTE_OLLAMA_MODEL 后重新启动应用。",
        )

    root = settings.ollama_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        version_payload = _read_json_url(f"{root}/api/version", timeout=3)
        tags_payload = _read_json_url(f"{root}/api/tags", timeout=5)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return DiagnosticCheck(
            key="ollama",
            title="Ollama 本地模型",
            status="error",
            summary="Ollama 服务不可达",
            detail=f"{root} · {exc} · 可执行文件：{binary or '未找到'}",
            hint=(
                "确认 Ollama 已安装，然后重新双击 start.bat；"
                "若使用自定义端口，请检查 MEETOMINUTE_OLLAMA_BASE_URL。"
            ),
        )

    model_names = {
        str(model.get("name") or model.get("model") or "")
        for model in tags_payload.get("models", [])
        if isinstance(model, dict)
    }
    configured = settings.ollama_model.removesuffix(":latest")
    model_ready = any(
        name.removesuffix(":latest") == configured for name in model_names
    )
    version = str(version_payload.get("version") or "未知版本")
    detail = (
        f"Ollama {version} · {root} · 模型 {settings.ollama_model} · "
        f"可执行文件：{binary or '未找到'}"
    )
    if not model_ready:
        return DiagnosticCheck(
            key="ollama",
            title="Ollama 本地模型",
            status="error",
            summary=f"服务在线 · 未找到模型 {settings.ollama_model}",
            detail=detail,
            hint=(
                f"运行 ollama create {settings.ollama_model} "
                "-f ollama\\Modelfile.meetominute。"
            ),
        )
    if not binary:
        return DiagnosticCheck(
            key="ollama",
            title="Ollama 本地模型",
            status="warning",
            summary=f"Ollama {version} 在线 · 启动命令未找到",
            detail=detail,
            hint="将 ollama 加入 PATH，确保下次能够由 start.bat 自动启动。",
        )
    return DiagnosticCheck(
        key="ollama",
        title="Ollama 本地模型",
        status="ok",
        summary=f"Ollama {version} · {settings.ollama_model} 已就绪",
        detail=detail,
    )


def _resolve_executable(command: str) -> str | None:
    candidate = Path(command)
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(command)


def _command_version(executable: str | None) -> str:
    if not executable:
        raise FileNotFoundError("未找到可执行文件")
    completed = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=8,
        check=False,
        creationflags=_no_window_flag(),
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return (lines[0] if lines else Path(executable).name)[:220]


def _read_json_url(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "MeetOminute/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("服务返回的不是 JSON 对象")
    return payload


def _format_bytes(value: int) -> str:
    if value >= _GIB:
        return f"{value / _GIB:.1f} GB"
    if value >= _MIB:
        return f"{value / _MIB:.0f} MB"
    return f"{value / 1024:.0f} KB"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _no_window_flag() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
