import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "select_consistency_cases.py"
SPEC = importlib.util.spec_from_file_location("select_consistency_cases", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selects_largest_paired_improvements(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"video_id": "a", "pair_id": "test:a", "caption": "A", "generation_seed": 42},
        {"video_id": "b", "pair_id": "test:b", "caption": "B", "generation_seed": 42},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def write(name, a, b):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "subject_consistency": [
                        0.0,
                        [
                            {"video_path": "x/a.mp4", "video_results": a},
                            {"video_path": "x/b.mp4", "video_results": b},
                        ],
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    baseline = write("b0.json", 0.5, 0.8)
    mugen = write("b3.json", 0.7, 0.81)

    cases = MODULE.select_cases(manifest, baseline, mugen, count=1)

    assert cases[0]["pair_id"] == "test:a"
    assert cases[0]["subject_consistency_change"] == pytest.approx(0.2)
