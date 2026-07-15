import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "merge_project_metrics.py"
SPEC = importlib.util.spec_from_file_location("merge_project_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixtures():
    generation = []
    audio = []
    retrieval = []
    for pair_id in ("a", "b"):
        for variant in MODULE.VARIANTS:
            generation.append(
                {
                    "pair_id": pair_id,
                    "sample_id": pair_id,
                    "variant": variant,
                    "metrics": {"latency_seconds": 1.0, "peak_vram_gib": 2.0},
                }
            )
            audio.append(
                {
                    "sample_id": pair_id,
                    "variant": variant,
                    "imagebind_audio_video_alignment": 0.4,
                    "onset_flow_correlation": 0.5,
                }
            )
            if variant in {"B2", "B3"}:
                retrieval.append(
                    {
                        "pair_id": pair_id,
                        "variant": variant,
                        "retrieval_rank": 2,
                        "retrieval_reciprocal_rank": 0.5,
                    }
                )
    vbench = {
        variant: {
            "vbench": {
                "details": {"status": "ok"},
                "metrics": {"subject_consistency": 0.7, "vbench_total": 0.7},
            }
        }
        for variant in MODULE.VARIANTS
    }
    return generation, {"per_sample": retrieval}, {"per_sample": audio}, vbench


def test_merge_requires_and_combines_all_real_metric_sources():
    merged = MODULE.merge_metrics(*fixtures())
    assert len(merged) == 8
    b3 = next(row for row in merged if row["pair_id"] == "a" and row["variant"] == "B3")
    assert b3["metrics"]["vbench_total"] == 0.7
    assert b3["metrics"]["retrieval_mrr"] == 0.5
    assert b3["metrics"]["onset_flow_correlation"] == 0.5


def test_merge_rejects_missing_audio_rows():
    generation, retrieval, audio, vbench = fixtures()
    audio["per_sample"].pop()
    with pytest.raises(ValueError, match="audio-control"):
        MODULE.merge_metrics(generation, retrieval, audio, vbench)
