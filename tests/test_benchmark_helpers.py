from __future__ import annotations

from pathlib import Path

from scripts.benchmark_funasr import (
    extract_textgrid_reference,
    levenshtein_distance,
    normalize_for_cer,
)


def test_levenshtein_and_normalization() -> None:
    assert normalize_for_cer("你好，<sil> World!") == "你好world"
    assert levenshtein_distance("你好世界", "你好试界") == 1


def test_extract_textgrid_reference(tmp_path: Path) -> None:
    textgrid = tmp_path / "sample.TextGrid"
    textgrid.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
item []:
  item [1]:
    class = "IntervalTier"
    name = "001-M"
    intervals [1]:
      xmin = 0
      xmax = 2
      text = "片段一"
    intervals [2]:
      xmin = 2
      xmax = 4
      text = ""
  item [2]:
    class = "IntervalTier"
    name = "002-F"
    intervals [1]:
      xmin = 1
      xmax = 3
      text = "片段二"
""",
        encoding="utf-8",
    )
    result = extract_textgrid_reference(textgrid, start=1, end=3)
    assert [item["text"] for item in result] == ["片段一", "片段二"]
    assert {item["speaker"] for item in result} == {"001-M", "002-F"}

