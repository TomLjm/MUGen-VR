import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_data" / "index_msrvtt_archive.py"
SPEC = importlib.util.spec_from_file_location("index_msrvtt_archive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_index_by_stem_accepts_separate_audio(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "video1.wav").write_bytes(b"audio")
    (tmp_path / "video2.txt").write_text("ignored", encoding="utf-8")

    indexed = MODULE.index_by_stem(tmp_path, {".wav"})

    assert indexed == {"video1": tmp_path / "nested" / "video1.wav"}


def test_index_by_stem_handles_missing_optional_root():
    assert MODULE.index_by_stem(None, {".wav"}) == {}
