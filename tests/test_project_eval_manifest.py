import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "build_project_eval_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_project_eval_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_project_eval_selection_is_fixed_and_uses_one_seed():
    rows = [
        {
            "sample_id": f"test:video{i}",
            "video_id": f"video{i}",
            "caption": f"caption {i}",
            "split": "test",
            "status": "ok",
            "keyframe_path": f"{i}.jpg",
            "audio_path": f"{i}.wav",
            "video_path": f"{i}.mp4",
        }
        for i in range(60)
    ]
    first = MODULE.select_rows(rows, 40, 42)
    second = MODULE.select_rows(list(reversed(rows)), 40, 42)
    assert first == second
    assert {row["generation_seed"] for row in first} == {42}
    assert len(MODULE.select_cases(first, 8)) == 8
