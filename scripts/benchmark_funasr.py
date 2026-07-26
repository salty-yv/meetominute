from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.providers import FunASRTranscriber


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 FunASR 对中文会议样本执行本地基准测试"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--textgrid", type=Path)
    parser.add_argument("--clip-offset", type=float, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/funasr.json"),
    )
    parser.add_argument("--expected-speakers", type=int, default=0)
    parser.add_argument("--glossary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"音频不存在：{audio_path}")
    settings = replace(
        Settings.from_env(), funasr_device=args.device.lower()
    )
    settings.ensure_directories()

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    transcriber = FunASRTranscriber(settings)
    segments = transcriber.transcribe(
        audio_path,
        expected_speakers=args.expected_speakers,
        glossary=args.glossary,
        duration_seconds=_duration(audio_path),
    )
    elapsed = time.perf_counter() - started
    duration = _duration(audio_path)
    predicted_text = "".join(segment.text for segment in segments)
    predicted_speakers = sorted({segment.speaker for segment in segments})

    metrics: dict[str, Any] = {
        "backend": "funasr",
        "model": settings.funasr_model,
        "vad_model": settings.funasr_vad_model,
        "punc_model": settings.funasr_punc_model,
        "speaker_model": settings.funasr_spk_model,
        "requested_device": settings.funasr_device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "audio": str(audio_path),
        "audio_duration_seconds": round(duration, 3),
        "processing_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / duration, 4),
        "speed_x_realtime": round(duration / elapsed, 2),
        "segment_count": len(segments),
        "predicted_speaker_count": len(predicted_speakers),
        "predicted_speakers": predicted_speakers,
        "recognized_character_count": len(
            normalize_for_cer(predicted_text)
        ),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
    }
    if args.textgrid:
        reference = extract_textgrid_reference(
            args.textgrid.resolve(),
            start=args.clip_offset,
            end=args.clip_offset + duration,
        )
        reference_text = "".join(item["text"] for item in reference)
        reference_normalized = normalize_for_cer(reference_text)
        predicted_normalized = normalize_for_cer(predicted_text)
        distance = levenshtein_distance(
            reference_normalized, predicted_normalized
        )
        metrics.update(
            {
                "reference_segment_count": len(reference),
                "reference_speaker_count": len(
                    {item["speaker"] for item in reference}
                ),
                "reference_character_count": len(reference_normalized),
                "character_edit_distance": distance,
                "character_error_rate": (
                    round(distance / len(reference_normalized), 4)
                    if reference_normalized
                    else None
                ),
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "segments": [segment.to_dict() for segment in segments],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果已写入：{output}")


def extract_textgrid_reference(
    path: Path, start: float, end: float
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tier_pattern = re.compile(
        r'item \[\d+\]:\s+class = "IntervalTier"\s+'
        r'name = "([^"]+)"(?P<body>.*?)(?=\n\s*item \[\d+\]:|\Z)',
        re.DOTALL,
    )
    interval_pattern = re.compile(
        r"intervals \[\d+\]:\s+"
        r"xmin = ([0-9.]+)\s+"
        r"xmax = ([0-9.]+)\s+"
        r'text = "(.*?)"(?=\s+intervals \[\d+\]:|\s*\Z)',
        re.DOTALL,
    )
    result: list[dict[str, Any]] = []
    for tier_match in tier_pattern.finditer(text):
        speaker = tier_match.group(1)
        for match in interval_pattern.finditer(tier_match.group("body")):
            interval_start = float(match.group(1))
            interval_end = float(match.group(2))
            content = match.group(3).replace('""', '"').strip()
            if not content or interval_end <= start or interval_start >= end:
                continue
            result.append(
                {
                    "speaker": speaker,
                    "start": max(interval_start, start) - start,
                    "end": min(interval_end, end) - start,
                    "text": content,
                }
            )
    result.sort(key=lambda item: (item["start"], item["end"]))
    return result


def normalize_for_cer(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    return "".join(
        character
        for character in text
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


if __name__ == "__main__":
    main()

