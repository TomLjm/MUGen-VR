import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "build_condition_scale_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_condition_scale_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_scale_manifest_uses_only_deterministic_validation_rows():
    rows = [
        {
            "sample_id": f"{split}:{index}",
            "video_id": f"video-{split}-{index}",
            "caption": "caption",
            "split": split,
            "status": "ok",
            "keyframe_path": "frame.jpg",
            "audio_path": "audio.wav",
            "video_path": "video.mp4",
        }
        for split in ("train", "val", "test")
        for index in range(12)
    ]
    selected = MODULE.select_validation_rows(rows, sample_count=8, generation_seed=42)

    assert len(selected) == 8
    assert all(row["sample_id"].startswith("val:") for row in selected)
    assert {row["generation_seed"] for row in selected} == {42}
