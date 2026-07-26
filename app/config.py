from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true 或 false")


@dataclass(frozen=True, slots=True)
class Settings:
    base_dir: Path
    data_dir: Path
    meetings_dir: Path
    models_dir: Path
    db_path: Path
    host: str
    port: int
    max_upload_bytes: int
    ffmpeg_bin: str
    ffprobe_bin: str
    local_transcriber: str
    cloud_transcriber: str
    local_llm: str
    cloud_llm: str
    openai_base_url: str
    openai_api_key: str
    transcribe_model: str
    llm_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_bin: str
    request_timeout_seconds: int
    llm_chunk_chars: int
    llm_max_tokens: int
    llm_reasoning_effort: str
    ollama_reasoning_effort: str
    funasr_model: str
    funasr_vad_model: str
    funasr_punc_model: str
    funasr_spk_model: str
    funasr_device: str
    funasr_batch_size_s: int
    funasr_isolate_process: bool

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root = (base_dir or Path(__file__).resolve().parents[1]).resolve()
        configured_data = Path(os.getenv("MEETOMINUTE_DATA_DIR", "data"))
        data_dir = (
            configured_data
            if configured_data.is_absolute()
            else root / configured_data
        ).resolve()
        max_upload_mb = _env_int("MEETOMINUTE_MAX_UPLOAD_MB", 4096)
        return cls(
            base_dir=root,
            data_dir=data_dir,
            meetings_dir=data_dir / "meetings",
            models_dir=data_dir / "models",
            db_path=data_dir / "meetominute.sqlite3",
            host=os.getenv("MEETOMINUTE_HOST", "127.0.0.1"),
            port=_env_int("MEETOMINUTE_PORT", 8000),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            ffmpeg_bin=os.getenv("MEETOMINUTE_FFMPEG", "ffmpeg"),
            ffprobe_bin=os.getenv("MEETOMINUTE_FFPROBE", "ffprobe"),
            local_transcriber=os.getenv(
                "MEETOMINUTE_LOCAL_TRANSCRIBER", "mock"
            ).lower(),
            cloud_transcriber=os.getenv(
                "MEETOMINUTE_CLOUD_TRANSCRIBER", "openai"
            ).lower(),
            local_llm=os.getenv("MEETOMINUTE_LOCAL_LLM", "mock").lower(),
            cloud_llm=os.getenv("MEETOMINUTE_CLOUD_LLM", "openai").lower(),
            openai_base_url=os.getenv(
                "MEETOMINUTE_OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            openai_api_key=os.getenv("MEETOMINUTE_OPENAI_API_KEY", ""),
            transcribe_model=os.getenv(
                "MEETOMINUTE_TRANSCRIBE_MODEL", "whisper-1"
            ),
            llm_model=os.getenv("MEETOMINUTE_LLM_MODEL", ""),
            ollama_base_url=os.getenv(
                "MEETOMINUTE_OLLAMA_BASE_URL",
                "http://127.0.0.1:11435/v1",
            ).rstrip("/"),
            ollama_model=os.getenv(
                "MEETOMINUTE_OLLAMA_MODEL", ""
            ),
            ollama_bin=os.getenv(
                "MEETOMINUTE_OLLAMA_BIN", "ollama"
            ),
            request_timeout_seconds=_env_int(
                "MEETOMINUTE_REQUEST_TIMEOUT_SECONDS", 600
            ),
            llm_chunk_chars=_env_int(
                "MEETOMINUTE_LLM_CHUNK_CHARS", 6000
            ),
            llm_max_tokens=_env_int(
                "MEETOMINUTE_LLM_MAX_TOKENS", 4096
            ),
            llm_reasoning_effort=os.getenv(
                "MEETOMINUTE_LLM_REASONING_EFFORT", ""
            ).lower(),
            ollama_reasoning_effort=os.getenv(
                "MEETOMINUTE_OLLAMA_REASONING_EFFORT", "none"
            ).lower(),
            funasr_model=os.getenv(
                "MEETOMINUTE_FUNASR_MODEL", "paraformer-zh"
            ),
            funasr_vad_model=os.getenv(
                "MEETOMINUTE_FUNASR_VAD_MODEL", "fsmn-vad"
            ),
            funasr_punc_model=os.getenv(
                "MEETOMINUTE_FUNASR_PUNC_MODEL", "ct-punc"
            ),
            funasr_spk_model=os.getenv(
                "MEETOMINUTE_FUNASR_SPK_MODEL", "cam++"
            ),
            funasr_device=os.getenv(
                "MEETOMINUTE_FUNASR_DEVICE", "auto"
            ).lower(),
            funasr_batch_size_s=_env_int(
                "MEETOMINUTE_FUNASR_BATCH_SIZE_S", 120
            ),
            funasr_isolate_process=_env_bool(
                "MEETOMINUTE_FUNASR_ISOLATE_PROCESS", True
            ),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meetings_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
