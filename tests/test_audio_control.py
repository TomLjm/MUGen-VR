import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mugen.evaluation.audio_control import resample_series


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "evaluate_audio_control.py"
SPEC = importlib.util.spec_from_file_location("evaluate_audio_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resample_series_preserves_endpoints():
    result = resample_series([0.0, 1.0], 5)
    assert result.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_resample_series_rejects_empty_input():
    with pytest.raises(ValueError):
        resample_series(np.asarray([]), 3)


def test_audio_evaluation_loads_all_generation_partitions(tmp_path):
    manifests = []
    for index in range(2):
        path = tmp_path / f"part-{index}.jsonl"
        path.write_text(json.dumps({"sample_id": str(index)}) + "\n", encoding="utf-8")
        manifests.append(path)

    assert [row["sample_id"] for row in MODULE.load_rows(manifests)] == ["0", "1"]
