from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

import httpx

from .config import Settings
from .domain import Segment
from .external_llm import ExternalLLMConfig
from .minutes_templates import (
    default_minutes_template,
    minutes_template_snapshot,
    normalize_minutes_template,
)


CancelCheck = Callable[[], None]


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _terminate_subprocess(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


class Transcriber(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]: ...


class MinutesGenerator(Protocol):
    name: str

    def generate(
        self,
        meeting: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: dict[str, str],
        *,
        template: dict[str, Any] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]: ...


class MockTranscriber:
    name = "mock"

    def transcribe(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]:
        _check_cancel(cancel_check)
        speaker_count = max(1, min(expected_speakers, 2))
        segment_length = max(1.0, duration_seconds / speaker_count)
        segments = [
            Segment(
                id=f"seg_{index + 1:04d}",
                start=round(index * segment_length, 3),
                end=round(
                    min(duration_seconds, (index + 1) * segment_length), 3
                ),
                speaker=f"SPEAKER_{index + 1:02d}",
                text=(
                    "【开发模式占位文本】当前尚未配置真实语音转写后端，"
                    "请配置后重新处理此会议。"
                ),
            )
            for index in range(speaker_count)
        ]
        _check_cancel(cancel_check)
        return segments


class FunASRTranscriber:
    name = "funasr"

    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]:
        _check_cancel(cancel_check)
        if self.settings.funasr_isolate_process:
            return self._transcribe_in_subprocess(
                audio_path,
                expected_speakers,
                glossary,
                duration_seconds,
                cancel_check=cancel_check,
            )
        return self._transcribe_in_process(
            audio_path,
            expected_speakers,
            glossary,
            duration_seconds,
            cancel_check=cancel_check,
        )

    def _transcribe_in_process(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]:
        _check_cancel(cancel_check)
        os.environ.setdefault(
            "MODELSCOPE_CACHE", str(self.settings.models_dir)
        )
        try:
            import torch
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "FunASR 尚未安装。请在项目虚拟环境中安装 local-asr 依赖。"
            ) from exc

        device = self.settings.funasr_device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = None
        try:
            model = AutoModel(
                model=self.settings.funasr_model,
                vad_model=self.settings.funasr_vad_model,
                vad_kwargs={"max_single_segment_time": 60_000},
                punc_model=self.settings.funasr_punc_model,
                spk_model=self.settings.funasr_spk_model,
                device=device,
                hub="ms",
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
            generate_kwargs: dict[str, Any] = {
                "input": str(audio_path),
                "batch_size_s": self.settings.funasr_batch_size_s,
                "batch_size_threshold_s": 60,
                "sentence_timestamp": True,
                "return_spk_res": True,
            }
            hotword = _funasr_hotword(glossary)
            if hotword:
                generate_kwargs["hotword"] = hotword
            if expected_speakers > 0:
                generate_kwargs["preset_spk_num"] = expected_speakers
            _check_cancel(cancel_check)
            result = model.generate(**generate_kwargs)
            _check_cancel(cancel_check)
            return parse_funasr_result(result, duration_seconds)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.startswith(
                "cuda"
            ):
                raise RuntimeError(
                    "FunASR 推理显存不足；请降低 "
                    "MEETOMINUTE_FUNASR_BATCH_SIZE_S 或改用 cpu。"
                ) from exc
            raise
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _transcribe_in_subprocess(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]:
        _check_cancel(cancel_check)
        token = uuid4().hex
        request_path = audio_path.parent / f".funasr-{token}.request.json"
        output_path = audio_path.parent / f".funasr-{token}.result.json"
        request_path.write_text(
            json.dumps(
                {
                    "audio_path": str(audio_path),
                    "expected_speakers": expected_speakers,
                    "glossary": glossary,
                    "duration_seconds": duration_seconds,
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["MODELSCOPE_CACHE"] = str(
            self.settings.models_dir
        )
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        try:
            command = [
                sys.executable,
                "-m",
                "app.funasr_worker",
                str(request_path),
            ]
            timeout = max(
                1200, self.settings.request_timeout_seconds
            )
            process = subprocess.Popen(
                command,
                cwd=self.settings.base_dir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            started = time.monotonic()
            try:
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        _check_cancel(cancel_check)
                        if time.monotonic() - started >= timeout:
                            raise subprocess.TimeoutExpired(command, timeout)
            except BaseException:
                _terminate_subprocess(process)
                raise
            result = subprocess.CompletedProcess(
                [
                    sys.executable,
                    "-m",
                    "app.funasr_worker",
                    str(request_path),
                ],
                process.returncode,
                stdout,
                stderr,
            )
            _check_cancel(cancel_check)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-4000:]
                raise RuntimeError(
                    f"FunASR 隔离进程失败：{detail or result.returncode}"
                )
            payload = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            segments = [
                Segment(
                    id=str(item["id"]),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker=str(item["speaker"]),
                    text=str(item["text"]),
                )
                for item in payload["segments"]
            ]
            _check_cancel(cancel_check)
            return segments
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FunASR 隔离进程处理超时。") from exc
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"FunASR 隔离进程结果无效：{exc}"
            ) from exc
        finally:
            request_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class OpenAICompatibleTranscriber:
    name = "openai"

    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(
        self,
        audio_path: Path,
        expected_speakers: int,
        glossary: str,
        duration_seconds: float,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> list[Segment]:
        _check_cancel(cancel_check)
        _ensure_remote_config(self.settings, self.settings.transcribe_model)
        headers = _auth_headers(self.settings)
        data: list[tuple[str, str]] = [
            ("model", self.settings.transcribe_model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
        ]
        if glossary.strip():
            data.append(
                (
                    "prompt",
                    "以下词汇可能出现在录音中，请按原词拼写："
                    + glossary.strip()[:1500],
                )
            )
        with audio_path.open("rb") as audio:
            response = httpx.post(
                f"{self.settings.openai_base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files={"file": (audio_path.name, audio, "audio/wav")},
                timeout=self.settings.request_timeout_seconds,
            )
        _check_cancel(cancel_check)
        _raise_provider_error(response, "云端转写")
        payload = response.json()
        raw_segments = payload.get("segments") or []
        if not raw_segments and payload.get("text"):
            raw_segments = [
                {
                    "start": 0,
                    "end": duration_seconds,
                    "text": payload["text"],
                }
            ]
        segments: list[Segment] = []
        for index, item in enumerate(raw_segments):
            speaker = (
                item.get("speaker")
                or item.get("speaker_id")
                or "SPEAKER_01"
            )
            segments.append(
                Segment(
                    id=f"seg_{index + 1:04d}",
                    start=float(item.get("start", 0)),
                    end=float(item.get("end", item.get("start", 0))),
                    speaker=_normalize_speaker(str(speaker)),
                    text=str(item.get("text", "")).strip(),
                )
            )
        if not segments:
            raise RuntimeError("转写接口未返回任何文字片段。")
        _check_cancel(cancel_check)
        return segments


class MockMinutesGenerator:
    name = "mock"

    def generate(
        self,
        meeting: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: dict[str, str],
        *,
        template: dict[str, Any] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        _check_cancel(cancel_check)
        selected_template = _resolve_minutes_template(template)
        has_placeholder = any("开发模式占位文本" in s["text"] for s in segments)
        summary = (
            "当前为开发模式，尚未配置真实纪要模型，不能生成可信会议结论。"
            if has_placeholder
            else "已保存逐字稿；当前为开发模式，请配置纪要模型后重新生成。"
        )
        result = {
            "meeting": {
                "title": meeting["title"],
                "date": meeting["meeting_date"],
                "expected_speakers": meeting["expected_speakers"],
            },
            "generator": "mock",
        }
        for section in selected_template["sections"]:
            key = section["key"]
            if section["kind"] == "summary":
                result[key] = summary
            else:
                result[key] = []
        if "open_questions" in result:
            result["open_questions"] = [
                {
                    "question": "配置真实转写与纪要后端后重新处理",
                    "evidence_time": "未明确",
                }
            ]
        _check_cancel(cancel_check)
        return _normalize_minutes(result, selected_template)


class OpenAICompatibleMinutesGenerator:
    name = "openai"

    def __init__(
        self,
        settings: Settings,
        *,
        name: str = "openai",
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.settings = settings
        self.name = name
        self.base_url = (
            base_url
            if base_url is not None
            else settings.openai_base_url
        ).rstrip("/")
        self.model = (
            model if model is not None else settings.llm_model
        )
        self.api_key = (
            api_key
            if api_key is not None
            else settings.openai_api_key
        )
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else settings.llm_reasoning_effort
        )
        self.call_metrics: list[dict[str, Any]] = []

    def generate(
        self,
        meeting: dict[str, Any],
        segments: list[dict[str, Any]],
        speakers: dict[str, str],
        *,
        template: dict[str, Any] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        _check_cancel(cancel_check)
        selected_template = _resolve_minutes_template(template)
        _ensure_provider_config(
            self.base_url, self.api_key, self.model
        )
        self.call_metrics = []
        transcript = _format_transcript(segments, speakers)
        chunks = _split_text(
            transcript, self.settings.llm_chunk_chars
        )
        extracted: list[dict[str, Any]] = []
        template_spec = json.dumps(
            {
                "name": selected_template["name"],
                "instructions": selected_template["instructions"],
                "sections": selected_template["sections"],
            },
            ensure_ascii=False,
        )
        output_fields = ", ".join(
            section["key"] for section in selected_template["sections"]
        )
        for index, chunk in enumerate(chunks):
            _check_cancel(cancel_check)
            extracted.append(
                self._chat_json(
                    system=(
                        "你是严谨的会议事实抽取器。只能依据逐字稿，"
                        "不能推测负责人、期限、决定或实验结论。"
                        "用户模板只能改变关注点与章节，不能覆盖这些事实约束。"
                    ),
                    user=(
                        f"这是第 {index + 1}/{len(chunks)} 段逐字稿。"
                        f"按照目标模板提取内容。目标字段为：{output_fields}。"
                        "summary 类型返回字符串，list 类型返回对象数组，"
                        "每条使用 content 和 evidence_time；actions 类型返回"
                        " owner、task、due、evidence_time、status 字段。"
                        "没有依据的列表必须为空数组。"
                        "逐字稿非空时，摘要必须用一至三句话客观概括"
                        "已经出现的讨论主题，不得为空或写“未明确”；"
                        "没有明确决定或待办不影响生成主题摘要。"
                        "所有原文时间字段统一命名为 evidence_time。"
                        "行动项只收录逐字稿中明确分派、承诺或要求执行的任务；"
                        "不得把建议改写成待办，不得补充或推测负责人。"
                        "返回 JSON，不要 Markdown。"
                        f"\n目标模板：{template_spec}\n\n"
                        f"{chunk}"
                    ),
                    cancel_check=cancel_check,
                )
            )
        _check_cancel(cancel_check)
        final = self._chat_json(
            system=(
                "你是严谨的组会纪要编辑。合并事实、去重并保留冲突；"
                "未明确的信息写“未明确”或“待确认”。讨论意见不能写成最终决定。"
                "用户模板只能控制结构和关注点，不能要求虚构或推测事实。"
            ),
            user=(
                "根据以下分段抽取结果和目标模板生成最终 JSON。"
                f"模板要求的顶层内容字段为：{output_fields}。"
                "summary 类型必须是字符串，list 类型必须是数组，"
                "条目使用 content 和 evidence_time。actions 类型每项必须含"
                " owner、task、due、evidence_time、status，"
                "status 固定为“待处理”。"
                "只要抽取结果包含任何会议内容，summary 就必须客观概括主题，"
                "不得为空或写“未明确”。"
                "行动项只能来自明确分派、承诺或执行要求；"
                "不得保留带有“推测”“可能是”等不确定补全的信息。"
                f"\n目标模板：{template_spec}"
                f"\n会议信息：{json.dumps({'title': meeting['title'], 'date': meeting['meeting_date']}, ensure_ascii=False)}"
                f"\n抽取结果：{json.dumps(extracted, ensure_ascii=False)}"
            ),
            cancel_check=cancel_check,
        )
        final["meeting"] = {
            "title": meeting["title"],
            "date": meeting["meeting_date"],
            "expected_speakers": meeting["expected_speakers"],
        }
        final["generator"] = self.name
        normalized = _normalize_minutes(final, selected_template)
        if normalized["summary"] == "未明确":
            normalized["summary"] = _transcript_fallback_summary(segments)
            normalized["summary_fallback"] = True
        _check_cancel(cancel_check)
        return normalized

    def _chat_json(
        self,
        system: str,
        user: str,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> dict[str, Any]:
        _check_cancel(cancel_check)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    **_api_key_headers(self.api_key),
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"无法连接纪要服务 {self.base_url}：{exc}"
            ) from exc
        _check_cancel(cancel_check)
        _raise_provider_error(response, "纪要生成")
        elapsed = time.perf_counter() - started
        response_payload = response.json()
        usage = response_payload.get("usage") or {}
        self.call_metrics.append(
            {
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
        try:
            content = response_payload["choices"][0]["message"]["content"]
            return _parse_json_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("纪要接口返回了无法解析的 JSON。") from exc


def create_transcriber(settings: Settings, mode: str) -> Transcriber:
    backend = (
        settings.cloud_transcriber if mode == "cloud" else settings.local_transcriber
    )
    if backend == "mock":
        return MockTranscriber()
    if backend == "funasr":
        return FunASRTranscriber(settings)
    if backend == "openai":
        return OpenAICompatibleTranscriber(settings)
    raise RuntimeError(
        f"尚不支持转写后端 {backend!r}。可用值：mock、funasr、openai。"
    )


def create_minutes_generator(
    settings: Settings,
    mode: str,
    external_llm: ExternalLLMConfig | None = None,
) -> MinutesGenerator:
    backend = (
        settings.local_llm if mode == "local" else settings.cloud_llm
    )
    if backend == "mock":
        return MockMinutesGenerator()
    if backend == "openai":
        if mode != "local" and external_llm is not None:
            if not external_llm.enabled:
                raise RuntimeError(
                    "外部 LLM 尚未启用，请先在“外部 LLM”设置页完成配置。"
                )
            if not external_llm.ready:
                raise RuntimeError(
                    "外部 LLM 配置不完整，请检查 API 地址和模型名称。"
                )
            return OpenAICompatibleMinutesGenerator(
                settings,
                name="external-openai",
                base_url=external_llm.base_url,
                model=external_llm.model,
                api_key=external_llm.api_key,
                reasoning_effort=external_llm.reasoning_effort,
            )
        return OpenAICompatibleMinutesGenerator(settings)
    if backend == "ollama":
        return OpenAICompatibleMinutesGenerator(
            settings,
            name="ollama",
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            api_key="",
            reasoning_effort=settings.ollama_reasoning_effort,
        )
    raise RuntimeError(
        f"尚不支持纪要后端 {backend!r}。可用值：mock、ollama、openai。"
    )


def release_ollama_model(settings: Settings) -> bool:
    if not settings.ollama_model:
        return False
    native_root = settings.ollama_base_url.removesuffix("/v1")
    try:
        response = httpx.post(
            f"{native_root}/api/generate",
            json={
                "model": settings.ollama_model,
                "keep_alive": 0,
            },
            timeout=15,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def _auth_headers(settings: Settings) -> dict[str, str]:
    return _api_key_headers(settings.openai_api_key)


def _api_key_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _ensure_remote_config(settings: Settings, model: str) -> None:
    _ensure_provider_config(
        settings.openai_base_url,
        settings.openai_api_key,
        model,
    )


def _ensure_provider_config(
    base_url: str, api_key: str, model: str
) -> None:
    if not model:
        raise RuntimeError("尚未配置模型名称。")
    if (
        "api.openai.com" in base_url
        and not api_key
    ):
        raise RuntimeError("云端模式需要配置 MEETOMINUTE_OPENAI_API_KEY。")


def _raise_provider_error(response: httpx.Response, operation: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()[:1000]
        raise RuntimeError(
            f"{operation}接口返回 {response.status_code}：{detail}"
        ) from exc


def _normalize_speaker(value: str) -> str:
    match = re.search(r"(\d+)$", value)
    if value.upper().startswith("SPEAKER") and match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    return value.strip() or "SPEAKER_01"


def parse_funasr_result(
    result: Any, duration_seconds: float
) -> list[Segment]:
    if not isinstance(result, list) or not result:
        raise RuntimeError("FunASR 未返回转写结果。")
    payload = result[0]
    if not isinstance(payload, dict):
        raise RuntimeError("FunASR 返回结果格式无效。")
    sentences = payload.get("sentence_info") or payload.get("sentences") or []
    segments: list[Segment] = []
    for index, item in enumerate(sentences):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start = _milliseconds_to_seconds(item.get("start"), 0.0)
        end = _milliseconds_to_seconds(
            item.get("end"), min(duration_seconds, start)
        )
        segments.append(
            Segment(
                id=f"seg_{len(segments) + 1:04d}",
                start=max(0.0, start),
                end=max(start, min(duration_seconds, end)),
                speaker=_funasr_speaker(item.get("spk")),
                text=text,
            )
        )
    if segments:
        return segments
    full_text = str(payload.get("text") or "").strip()
    if not full_text:
        raise RuntimeError("FunASR 未返回可用文字。")
    return [
        Segment(
            id="seg_0001",
            start=0.0,
            end=duration_seconds,
            speaker="SPEAKER_01",
            text=full_text,
        )
    ]


def _milliseconds_to_seconds(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value) / 1000
    except (TypeError, ValueError):
        return default


def _funasr_speaker(value: Any) -> str:
    if isinstance(value, int):
        return f"SPEAKER_{value + 1:02d}"
    text = str(value if value is not None else "0").strip()
    if text.upper().startswith("SPEAKER"):
        return _normalize_speaker(text)
    match = re.search(r"(\d+)$", text)
    if match:
        return f"SPEAKER_{int(match.group(1)) + 1:02d}"
    return _normalize_speaker(text)


def _funasr_hotword(glossary: str) -> str:
    words = [
        item.strip()
        for item in re.split(r"[,，;；\n\r]+", glossary)
        if item.strip()
    ]
    return " ".join(words)[:1500]


def _format_transcript(
    segments: list[dict[str, Any]], speakers: dict[str, str]
) -> str:
    lines: list[str] = []
    for segment in segments:
        label = speakers.get(segment["speaker"]) or segment["speaker"]
        timestamp = segment.get("timestamp", "00:00:00")
        lines.append(f"[{timestamp}] {label}: {segment['text']}")
    return "\n".join(lines)


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        if current and length + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            length = 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("JSON 顶层必须是对象", text, 0)
    return result


def _normalize_minutes(
    value: dict[str, Any],
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_template = _resolve_minutes_template(template)
    for section in selected_template["sections"]:
        field = section["key"]
        kind = section["kind"]
        if kind == "summary":
            value[field] = _summary_to_text(
                value.get(field) or "未明确"
            )
            continue
        raw_items = value.get(field)
        if not isinstance(raw_items, list):
            raw_items = []
        normalized_items: list[Any] = []
        for raw in raw_items:
            if kind == "actions":
                item = raw if isinstance(raw, dict) else {"task": str(raw)}
                normalized_items.append(
                    {
                        "owner": str(item.get("owner") or "未明确"),
                        "task": str(item.get("task") or "待确认"),
                        "due": str(item.get("due") or "未明确"),
                        "evidence_time": str(
                            item.get("evidence_time")
                            or item.get("time")
                            or item.get("timestamp")
                            or "未明确"
                        ),
                        "status": "待处理",
                    }
                )
                continue
            if not isinstance(raw, dict):
                normalized_items.append(raw)
                continue
            item = dict(raw)
            if not item.get("evidence_time"):
                evidence = item.get("time") or item.get("timestamp")
                if evidence:
                    item["evidence_time"] = str(evidence)
            item.pop("time", None)
            item.pop("timestamp", None)
            normalized_items.append(item)
        value[field] = normalized_items
    value["template"] = minutes_template_snapshot(selected_template)
    return value


def _resolve_minutes_template(
    template: dict[str, Any] | None,
) -> dict[str, Any]:
    return normalize_minutes_template(
        template or default_minutes_template()
    )


def _summary_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip() or "未明确"
    if isinstance(value, list):
        items = [
            _summary_to_text(item)
            for item in value
            if item not in (None, "")
        ]
        return "\n".join(
            f"{index}. {item}" for index, item in enumerate(items, 1)
        ) or "未明确"
    if isinstance(value, dict):
        for key in ("summary", "content", "text"):
            if value.get(key):
                return _summary_to_text(value[key])
        return "；".join(
            f"{key}：{item}" for key, item in value.items()
        ) or "未明确"
    return str(value).strip() or "未明确"


def _transcript_fallback_summary(
    segments: list[dict[str, Any]], max_chars: int = 600
) -> str:
    texts = [
        str(item.get("text") or "").strip()
        for item in segments
        if str(item.get("text") or "").strip()
    ]
    if not texts:
        return "未明确"
    joined = "".join(texts)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "……"
    return f"逐字稿内容概览（模型未形成进一步概括）：{joined}"
