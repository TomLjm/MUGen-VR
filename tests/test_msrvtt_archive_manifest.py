import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_data" / "build_msrvtt_archive_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_msrvtt_archive_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_manifest_is_deterministic_and_isolated():
    train = [
        {"video_id": f"video{i}", "caption": [f"caption {i}", "alternate"]}
        for i in range(5)
    ]
    test = [{"video_id": "video9", "caption": ["test caption"]}]

    first = MODULE.build_manifest(train, test, val_size=2, revision="abc")
    second = MODULE.build_manifest(list(reversed(train)), test, val_size=2, revision="abc")

    assert first == second
    assert sum(row["split"] == "train" for row in first) == 3
    assert sum(row["split"] == "val" for row in first) == 2
    assert sum(row["split"] == "test" for row in first) == 1
    assert len({row["video_id"] for row in first}) == len(first)
    assert all(row["source_revision"] == "abc" for row in first)
