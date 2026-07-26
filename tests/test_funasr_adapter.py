from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from app.config import Settings
from app.providers import FunASRTranscriber, parse_funasr_result


def test_parse_funasr_sentence_info() -> None:
    result = [
        {
            "text": "大家好今天开始开会",
            "sentence_info": [
                {
                    "text": "大家好。",
                    "start": 1250,
                    "end": 2800,
                    "spk": 0,
                },
                {
                    "text": "今天开始开会。",
                    "start": 3000,
                    "end": 6200,
                    "spk": 2,
                },
            ],
        }
    ]
    segments = parse_funasr_result(result, 10)
    assert len(segments) == 2
    assert segments[0].start == 1.25
    assert segments[0].speaker == "SPEAKER_01"
    assert segments[1].speaker == "SPEAKER_03"
    assert segments[1].text == "今天开始开会。"


def test_parse_funasr_falls_back_to_full_text() -> None:
    segments = parse_funasr_result([{"text": "测试文本"}], 5)
    assert len(segments) == 1
    assert segments[0].end == 5
    assert segments[0].text == "测试文本"


def test_transcriber_passes_expected_speaker_count(
    tmp_path, monkeypatch
) -> None:
    calls: dict = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [
                {
                    "sentence_info": [
                        {
                            "text": "测试。",
                            "start": 0,
                            "end": 1000,
                            "spk": 0,
                        }
                    ]
                }
            ]

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
    )
    fake_funasr = ModuleType("funasr")
    fake_funasr.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    monkeypatch.setenv(
        "MEETOMINUTE_FUNASR_ISOLATE_PROCESS", "false"
    )

    settings = Settings.from_env(tmp_path)
    segments = FunASRTranscriber(settings).transcribe(
        tmp_path / "sample.wav",
        expected_speakers=7,
        glossary="策划,主持人",
        duration_seconds=1,
    )

    assert calls["init"]["device"] == "cuda:0"
    assert calls["init"]["disable_update"] is True
    assert calls["generate"]["preset_spk_num"] == 7
    assert calls["generate"]["hotword"] == "策划 主持人"
    assert segments[0].text == "测试。"
