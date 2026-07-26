from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

import uvicorn
from dotenv import load_dotenv

from .config import Settings


def _url_is_ready(url: str, timeout: float = 1) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _open_when_ready(url: str) -> None:
    health_url = f"{url}/health"
    for _ in range(60):
        if _url_is_ready(health_url):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def _ollama_health_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/api/version"


def _ensure_ollama_ready(settings: Settings) -> None:
    health_url = _ollama_health_url(settings.ollama_base_url)
    if _url_is_ready(health_url):
        return
    executable = (
        shutil.which(settings.ollama_bin) or settings.ollama_bin
    )
    environment = os.environ.copy()
    environment.setdefault("GGML_VK_VISIBLE_DEVICES", "-1")
    environment.setdefault("OLLAMA_VULKAN", "0")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    environment.setdefault("OLLAMA_NUM_PARALLEL", "1")
    environment.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    environment.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
    environment.setdefault("OLLAMA_KEEP_ALIVE", "10m")
    parsed = urllib.parse.urlsplit(settings.ollama_base_url)
    if parsed.hostname:
        port = parsed.port or (
            443 if parsed.scheme == "https" else 80
        )
        environment["OLLAMA_HOST"] = f"{parsed.hostname}:{port}"
    creationflags = (
        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )
    try:
        subprocess.Popen(
            [executable, "serve"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise RuntimeError(
            f"无法启动 Ollama（{executable}）：{exc}"
        ) from exc
    for _ in range(120):
        if _url_is_ready(health_url):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"Ollama 启动超时，请检查 {health_url} 和 Ollama 日志。"
    )


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    if "ollama" in {settings.local_llm, settings.cloud_llm}:
        _ensure_ollama_ready(settings)
    url = f"http://{settings.host}:{settings.port}"
    threading.Thread(
        target=_open_when_ready, args=(url,), daemon=True
    ).start()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
