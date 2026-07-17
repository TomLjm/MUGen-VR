import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "build_targeted_report.py"
SPEC = importlib.util.spec_from_file_location("build_targeted_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_contains_only_targeted_consistency_metrics(tmp_path):
    def write(name, values):
        path = tmp_path / name
        path.write_text(json.dumps({"vbench": {"metrics": values}}), encoding="utf-8")
        return path

    baseline = write(
        "b0.json",
        {
            "subject_consistency": 0.8,
            "motion_smoothness": 0.9,
            "temporal_flickering": 0.95,
            "vbench_total": 0.7,
        },
    )
    mugen = write(
        "b3.json",
        {
            "subject_consistency": 0.82,
            "motion_smoothness": 0.91,
            "temporal_flickering": 0.96,
            "vbench_total": 0.69,
        },
    )

    report = MODULE.build_report(baseline, mugen)

    assert [row["name"] for row in report["metrics"]] == list(MODULE.METRICS)
    assert report["metrics"][0]["relative_error_reduction"] == pytest.approx(0.1)
    assert "vbench_total" not in json.dumps(report)
