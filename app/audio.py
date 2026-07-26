from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from .config import Settings


class AudioProcessingError(RuntimeError):
    pass


CancelCheck = Callable[[], None]


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def _run_process(
    command: list[str],
    *,
    timeout: float,
    cancel_check: CancelCheck | None,
) -> subprocess.CompletedProcess[str]:
    if cancel_check is not None:
        cancel_check()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
    )
    started = time.monotonic()
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel_check is not None:
                    cancel_check()
                if time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
    except BaseException:
        _terminate_process(process)
        raise
    result = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return result


def probe_duration(
    source: Path,
    settings: Settings,
    cancel_check: CancelCheck | None = None,
) -> float:
    command = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = _run_process(
            command,
            timeout=120,
            cancel_check=cancel_check,
        )
        payload = json.loads(result.stdout)
        return float(payload["format"]["duration"])
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise AudioProcessingError(
            "无法读取音频信息，请确认 FFmpeg 已安装且文件未损坏。"
        ) from exc


def normalize_audio(
    source: Path,
    destination: Path,
    settings: Settings,
    cancel_check: CancelCheck | None = None,
) -> float:
    duration = probe_duration(source, settings, cancel_check)
    temporary = destination.with_suffix(".tmp.wav")
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        _run_process(
            command,
            timeout=max(600, int(duration * 2)),
            cancel_check=cancel_check,
        )
        if cancel_check is not None:
            cancel_check()
        temporary.replace(destination)
    except FileNotFoundError as exc:
        temporary.unlink(missing_ok=True)
        raise AudioProcessingError(
            "找不到 FFmpeg。请安装 FFmpeg 或设置 MEETOMINUTE_FFMPEG。"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        temporary.unlink(missing_ok=True)
        raise AudioProcessingError("音频预处理超时。") from exc
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = (exc.stderr or "").strip()[-1000:]
        raise AudioProcessingError(
            f"音频预处理失败：{detail or '未知 FFmpeg 错误'}"
        ) from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return duration
