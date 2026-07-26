from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .config import Settings


class AudioProcessingError(RuntimeError):
    pass


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def probe_duration(source: Path, settings: Settings) -> float:
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
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=True,
            creationflags=_creation_flags(),
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
) -> float:
    duration = probe_duration(source, settings)
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
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(600, int(duration * 2)),
            check=True,
            creationflags=_creation_flags(),
        )
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
    return duration

