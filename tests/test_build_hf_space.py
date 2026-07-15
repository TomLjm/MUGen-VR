import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "release" / "build_hf_space.py"
SPEC = importlib.util.spec_from_file_location("build_hf_space", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_space_builder_copies_fixed_pair_and_summarizes_gates(tmp_path):
    source_b0 = tmp_path / "source-b0.mp4"
    source_b3 = tmp_path / "source-b3.mp4"
    source_b0.write_bytes(b"b0")
    source_b3.write_bytes(b"b3")
    rows = [
        {
            "pair_id": "test:1",
            "variant": "B0",
            "generated_video_path": str(source_b0),
            "metrics": {"latency_seconds": 1.0},
        },
        {
            "pair_id": "test:1",
            "variant": "B3",
            "generated_video_path": str(source_b3),
            "metrics": {"latency_seconds": 1.2},
            "references": [{"sample_id": "train:7"}],
            "gates": [[[0.2, 0.8]], [[0.6, 0.4]]],
        },
    ]
    report = {
        "summary": {
            variant: {"means": {"vbench_total": 0.5}} for variant in ("B0", "B1", "B2", "B3")
        }
    }
    cases = [{"pair_id": "test:1", "caption": "case", "generation_seed": 42}]

    payload = MODULE.build_space(rows, report, cases, tmp_path / "space", {})

    assert payload["status"] == "held-out-evaluation-complete"
    assert payload["cases"][0]["gates"] == [0.4, 0.6]
    assert (tmp_path / "space" / payload["cases"][0]["b0_video"]).read_bytes() == b"b0"
