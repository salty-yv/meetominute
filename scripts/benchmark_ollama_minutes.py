from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.config import Settings
from app.launcher import _ensure_ollama_ready
from app.providers import create_minutes_generator


LIST_FIELDS = (
    "member_progress",
    "experimental_results",
    "suggestions",
    "decisions",
    "action_items",
    "open_questions",
    "next_followups",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用已转写的会议逐字稿测试 Ollama 纪要生成。"
    )
    parser.add_argument("transcript", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark-results/ollama-qwen35-9b-minutes.json"
        ),
    )
    parser.add_argument("--title", default="AISHELL-4 中文会议测试")
    parser.add_argument("--date", default="2026-07-26")
    parser.add_argument("--expected-speakers", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    transcript_path = args.transcript.resolve()
    payload = json.loads(
        transcript_path.read_text(encoding="utf-8")
    )
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SystemExit("输入 JSON 不含可用的 segments。")

    settings = Settings.from_env()
    _ensure_ollama_ready(settings)
    generator = create_minutes_generator(settings, "local")
    if generator.name != "ollama":
        raise SystemExit(
            f"本地纪要后端不是 ollama，而是 {generator.name!r}。"
        )
    meeting = {
        "title": args.title,
        "meeting_date": args.date,
        "expected_speakers": args.expected_speakers,
    }
    speakers = {
        item["speaker"]: "" for item in segments if item.get("speaker")
    }

    print(
        f"开始生成纪要：{len(segments)} 段，"
        f"{sum(len(str(item.get('text', ''))) for item in segments)} 字符，"
        f"模型 {settings.ollama_model}",
        flush=True,
    )
    started = time.perf_counter()
    minutes = generator.generate(meeting, segments, speakers)
    elapsed = time.perf_counter() - started
    call_metrics = getattr(generator, "call_metrics", [])
    completion_tokens = sum(
        int(item.get("completion_tokens") or 0)
        for item in call_metrics
    )
    evidence = list(_evidence_values(minutes))
    valid_evidence = [
        value
        for value in evidence
        if re.search(r"\d{2}:\d{2}:\d{2}", value)
    ]
    runtime = _ollama_runtime(settings)
    metrics: dict[str, Any] = {
        "backend": generator.name,
        "model": settings.ollama_model,
        "transcript": str(transcript_path),
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "transcript_character_count": sum(
            len(str(item.get("text", ""))) for item in segments
        ),
        "processing_seconds": round(elapsed, 3),
        "llm_call_count": len(call_metrics),
        "llm_calls": call_metrics,
        "completion_tokens": completion_tokens,
        "effective_completion_tokens_per_second": (
            round(completion_tokens / elapsed, 2)
            if completion_tokens
            else None
        ),
        "summary_character_count": len(minutes["summary"]),
        "section_counts": {
            key: len(minutes[key]) for key in LIST_FIELDS
        },
        "evidence_field_count": len(evidence),
        "valid_timestamp_evidence_count": len(valid_evidence),
        "all_required_outputs_present": (
            isinstance(minutes.get("summary"), str)
            and all(isinstance(minutes.get(key), list) for key in LIST_FIELDS)
        ),
        "ollama_runtime": runtime,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"metrics": metrics, "minutes": minutes}
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"结果已写入：{output}", flush=True)


def _evidence_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_time" and isinstance(item, str):
                yield item
            else:
                yield from _evidence_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _evidence_values(item)


def _ollama_runtime(settings: Settings) -> dict[str, Any] | None:
    native_root = settings.ollama_base_url.removesuffix("/v1")
    try:
        response = httpx.get(
            f"{native_root}/api/ps", timeout=5
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    for item in response.json().get("models", []):
        if item.get("name", "").split(":", 1)[0] == (
            settings.ollama_model.split(":", 1)[0]
        ):
            return {
                "size_bytes": item.get("size"),
                "size_vram_bytes": item.get("size_vram"),
                "context_length": item.get("context_length"),
            }
    return None


if __name__ == "__main__":
    main()
