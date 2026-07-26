from __future__ import annotations

import base64
import ctypes
import json
import os
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .domain import utc_now


ALLOWED_REASONING_EFFORTS = {"", "none", "low", "medium", "high"}


@dataclass(frozen=True, slots=True)
class ExternalLLMConfig:
    enabled: bool
    provider_name: str
    base_url: str
    model: str
    api_key: str
    reasoning_effort: str
    updated_at: str = ""
    source: str = "file"

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.base_url and self.model)

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider_name": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "api_key_configured": self.api_key_configured,
            "ready": self.ready,
            "updated_at": self.updated_at,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ConnectionTestResult:
    status: str
    message: str
    latency_ms: int
    model_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "model_count": self.model_count,
        }


class ExternalLLMConfigStore:
    def __init__(self, path: Path, settings: Settings):
        self.path = path
        self.settings = settings
        self._lock = threading.Lock()

    @property
    def has_saved_config(self) -> bool:
        return self.path.exists()

    def load(self) -> ExternalLLMConfig:
        with self._lock:
            if not self.path.exists():
                return self._from_environment()
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                encrypted = str(payload.get("api_key_protected", ""))
                api_key = _unprotect_secret(encrypted) if encrypted else ""
                return ExternalLLMConfig(
                    enabled=bool(payload.get("enabled", False)),
                    provider_name=str(
                        payload.get("provider_name", "OpenAI Compatible")
                    ),
                    base_url=str(payload.get("base_url", "")).rstrip("/"),
                    model=str(payload.get("model", "")),
                    api_key=api_key,
                    reasoning_effort=str(
                        payload.get("reasoning_effort", "")
                    ).lower(),
                    updated_at=str(payload.get("updated_at", "")),
                    source="file",
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "外部 LLM 配置无法读取，请在设置页重新保存。"
                ) from exc

    def save(
        self,
        *,
        enabled: bool,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        clear_api_key: bool = False,
        reasoning_effort: str = "",
    ) -> ExternalLLMConfig:
        current = self.load()
        next_key = (
            ""
            if clear_api_key
            else (api_key.strip() if api_key.strip() else current.api_key)
        )
        config = validate_external_llm_config(
            ExternalLLMConfig(
                enabled=enabled,
                provider_name=provider_name.strip() or "OpenAI Compatible",
                base_url=_normalize_base_url(base_url),
                model=model.strip(),
                api_key=next_key,
                reasoning_effort=reasoning_effort.strip().lower(),
                updated_at=utc_now(),
                source="file",
            ),
            require_ready=enabled,
        )
        payload = {
            "version": 1,
            "enabled": config.enabled,
            "provider_name": config.provider_name,
            "base_url": config.base_url,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "api_key_protected": (
                _protect_secret(config.api_key) if config.api_key else ""
            ),
            "updated_at": config.updated_at,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return config

    def resolve_form_config(
        self,
        *,
        enabled: bool,
        provider_name: str,
        base_url: str,
        model: str,
        api_key: str,
        reasoning_effort: str,
    ) -> ExternalLLMConfig:
        current = self.load()
        return validate_external_llm_config(
            ExternalLLMConfig(
                enabled=enabled,
                provider_name=provider_name.strip() or "OpenAI Compatible",
                base_url=_normalize_base_url(base_url),
                model=model.strip(),
                api_key=api_key.strip() or current.api_key,
                reasoning_effort=reasoning_effort.strip().lower(),
                source="form",
            ),
            require_ready=True,
        )

    def _from_environment(self) -> ExternalLLMConfig:
        configured = bool(self.settings.llm_model)
        return ExternalLLMConfig(
            enabled=configured,
            provider_name="环境变量配置",
            base_url=self.settings.openai_base_url,
            model=self.settings.llm_model,
            api_key=self.settings.openai_api_key,
            reasoning_effort=self.settings.llm_reasoning_effort,
            source="environment",
        )


def validate_external_llm_config(
    config: ExternalLLMConfig, *, require_ready: bool
) -> ExternalLLMConfig:
    if config.reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("推理强度必须为空、none、low、medium 或 high。")
    if len(config.provider_name) > 100:
        raise ValueError("服务名称不能超过 100 个字符。")
    if len(config.model) > 200:
        raise ValueError("模型名称不能超过 200 个字符。")
    if not config.base_url and not require_ready:
        return config
    parsed = urllib.parse.urlsplit(config.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API 地址必须是有效的 http:// 或 https:// 地址。")
    if parsed.username or parsed.password:
        raise ValueError("API 地址中不能包含用户名或密码。")
    if require_ready and not config.model:
        raise ValueError("请填写外部 LLM 的模型名称。")
    if (
        require_ready
        and parsed.hostname == "api.openai.com"
        and not config.api_key
    ):
        raise ValueError("连接 OpenAI 官方接口时必须填写 API Key。")
    return config


def test_external_llm_connection(
    config: ExternalLLMConfig, timeout_seconds: int = 20
) -> ConnectionTestResult:
    validate_external_llm_config(config, require_ready=True)
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    started = time.perf_counter()
    try:
        response = httpx.get(
            f"{config.base_url}/models",
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        )
    except httpx.RequestError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return ConnectionTestResult(
            status="error",
            message=f"无法连接服务：{exc}",
            latency_ms=elapsed,
        )

    elapsed = int((time.perf_counter() - started) * 1000)
    if response.status_code in {404, 405}:
        return _test_chat_endpoint(config, headers, timeout_seconds, started)
    if not response.is_success:
        return ConnectionTestResult(
            status="error",
            message=_provider_error_message(response),
            latency_ms=elapsed,
        )

    model_ids: list[str] = []
    try:
        payload = response.json()
        model_ids = [
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
    except (ValueError, AttributeError):
        pass
    if model_ids and config.model not in model_ids:
        return ConnectionTestResult(
            status="warning",
            message=(
                f"服务连接成功，但模型列表中没有找到 {config.model}。"
                "请确认模型标识是否正确。"
            ),
            latency_ms=elapsed,
            model_count=len(model_ids),
        )
    return ConnectionTestResult(
        status="success",
        message=f"连接成功，模型 {config.model} 可用于外部纪要。",
        latency_ms=elapsed,
        model_count=len(model_ids) if model_ids else None,
    )


def _test_chat_endpoint(
    config: ExternalLLMConfig,
    headers: dict[str, str],
    timeout_seconds: int,
    started: float,
) -> ConnectionTestResult:
    try:
        response = httpx.post(
            f"{config.base_url}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": "Reply OK."}],
                "temperature": 0,
                "max_tokens": 2,
            },
            timeout=timeout_seconds,
            follow_redirects=True,
        )
    except httpx.RequestError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return ConnectionTestResult(
            status="error",
            message=f"无法连接聊天接口：{exc}",
            latency_ms=elapsed,
        )
    elapsed = int((time.perf_counter() - started) * 1000)
    if not response.is_success:
        return ConnectionTestResult(
            status="error",
            message=_provider_error_message(response),
            latency_ms=elapsed,
        )
    return ConnectionTestResult(
        status="success",
        message=f"连接成功，模型 {config.model} 已返回响应。",
        latency_ms=elapsed,
    )


def _provider_error_message(response: httpx.Response) -> str:
    detail = response.text.strip().replace("\n", " ")[:500]
    if response.status_code in {401, 403}:
        return f"鉴权失败（HTTP {response.status_code}），请检查 API Key。"
    if response.status_code == 404:
        return "接口不存在（HTTP 404），请确认 API 地址包含正确的 /v1 路径。"
    return f"服务返回 HTTP {response.status_code}：{detail or '无错误详情'}"


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized.rstrip("/")


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cb_data", ctypes.c_uint32),
        ("pb_data", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _protect_secret(value: str) -> str:
    if not value:
        return ""
    raw = value.encode("utf-8")
    if os.name != "nt":
        return "base64:" + base64.b64encode(raw).decode("ascii")
    buffer = ctypes.create_string_buffer(raw)
    input_blob = _DataBlob(
        len(raw),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "MeetOminute external LLM key",
        None,
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        protected = ctypes.string_at(output_blob.pb_data, output_blob.cb_data)
        return "dpapi:" + base64.b64encode(protected).decode("ascii")
    finally:
        kernel32.LocalFree(output_blob.pb_data)


def _unprotect_secret(value: str) -> str:
    if not value:
        return ""
    if value.startswith("base64:"):
        return base64.b64decode(value[7:]).decode("utf-8")
    if not value.startswith("dpapi:") or os.name != "nt":
        raise ValueError("不支持的密钥保护格式。")
    protected = base64.b64decode(value[6:])
    buffer = ctypes.create_string_buffer(protected)
    input_blob = _DataBlob(
        len(protected),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    ):
        raise ctypes.WinError()
    try:
        raw = ctypes.string_at(output_blob.pb_data, output_blob.cb_data)
        return raw.decode("utf-8")
    finally:
        kernel32.LocalFree(output_blob.pb_data)
