from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from dotenv import load_dotenv

from app.config import Settings
from app.launcher import _ensure_ollama_ready
from app.main import create_app


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过网页上传接口验证 FunASR 到导出文件的完整流程。"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument(
        "--minutes-backend",
        choices=("mock", "ollama", "openai"),
        default="mock",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/funasr-app-smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"音频不存在：{audio_path}")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_data = root / "benchmark-results" / "app-smoke-data" / run_name
    base_settings = Settings.from_env(root)
    settings = replace(
        base_settings,
        data_dir=run_data,
        meetings_dir=run_data / "meetings",
        db_path=run_data / "meetominute.sqlite3",
        models_dir=root / "data" / "models",
        local_transcriber="funasr",
        local_llm=args.minutes_backend,
        funasr_device="auto",
    )
    if settings.local_llm == "ollama":
        _ensure_ollama_ready(settings)
    application = create_app(settings)

    started = time.perf_counter()
    with TestClient(application) as client, audio_path.open("rb") as audio:
        response = client.post(
            "/meetings",
            data={
                "title": "FunASR 应用级冒烟测试",
                "meeting_date": datetime.now().date().isoformat(),
                "expected_speakers": "7",
                "glossary": "策划,节目,主持人,赞助",
                "processing_mode": "local",
            },
            files={
                "recording": (
                    audio_path.name,
                    audio,
                    "audio/wav",
                )
            },
            follow_redirects=False,
        )
        if response.status_code != 303:
            raise RuntimeError(
                f"上传失败：HTTP {response.status_code} {response.text[:500]}"
            )
        meetings = application.state.database.list_meetings()
        meeting_id = meetings[0]["id"]
        deadline = time.monotonic() + args.timeout
        meeting = meetings[0]
        while time.monotonic() < deadline:
            meeting = application.state.database.get_meeting(meeting_id)
            if meeting["status"] == "completed":
                break
            if meeting["status"] == "failed":
                raise RuntimeError(meeting["error"])
            time.sleep(0.25)
        else:
            raise TimeoutError(
                f"等待处理完成超时；最后状态：{meeting['status']}"
            )

        detail = client.get(f"/meetings/{meeting_id}")
        export = client.get(f"/meetings/{meeting_id}/download/json")
        if detail.status_code != 200 or export.status_code != 200:
            raise RuntimeError(
                "处理完成，但详情页面或 JSON 导出接口验证失败。"
            )

    directory = settings.meetings_dir / meeting["slug"]
    transcript = json.loads(
        (directory / "transcript_raw.json").read_text(encoding="utf-8")
    )
    required_outputs = (
        "normalized.wav",
        "transcript_raw.json",
        "transcript_edited.json",
        "transcript.md",
        "minutes.json",
        "minutes.md",
        "minutes.txt",
        "minutes.docx",
    )
    payload = {
        "status": meeting["status"],
        "processing_seconds": round(time.perf_counter() - started, 3),
        "meeting_id": meeting_id,
        "meeting_directory": str(directory),
        "transcriber_backend": meeting["transcriber_backend"],
        "minutes_backend": meeting["llm_backend"],
        "duration_seconds": meeting["duration_seconds"],
        "segment_count": len(transcript["segments"]),
        "speaker_count": len(
            {item["speaker"] for item in transcript["segments"]}
        ),
        "sample_segments": transcript["segments"][:5],
        "outputs": {
            name: (directory / name).exists() for name in required_outputs
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
