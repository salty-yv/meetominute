from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import Settings
from .providers import FunASRTranscriber


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法：python -m app.funasr_worker request.json")
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    transcriber = FunASRTranscriber(settings)
    segments = transcriber._transcribe_in_process(
        Path(request["audio_path"]),
        int(request["expected_speakers"]),
        str(request.get("glossary") or ""),
        float(request["duration_seconds"]),
    )
    output_path = Path(request["output_path"])
    output_path.write_text(
        json.dumps(
            {
                "segments": [
                    segment.to_dict() for segment in segments
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
