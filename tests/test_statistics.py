import pytest

from mugen.evaluation.statistics import compare_full_to_best_baselines, paired_bootstrap_delta


def test_paired_bootstrap_detects_consistent_improvement():
    result = paired_bootstrap_delta([2, 3, 4], [1, 2, 3], samples=1000, seed=1)
    assert result["delta"] == 1.0
    assert result["ci_lower"] == 1.0


def test_comparison_selects_best_baseline_and_pairs_by_id():
    rows = []
    for pair_id in ["a", "b", "c"]:
        rows.extend(
            [
                {"pair_id": pair_id, "variant": "B0", "metrics": {"score": 0.5}},
                {"pair_id": pair_id, "variant": "B1", "metrics": {"score": 0.7}},
                {"pair_id": pair_id, "variant": "B5", "metrics": {"score": 0.8}},
            ]
        )
    result = compare_full_to_best_baselines(rows, baseline_variants=("B0", "B1"), samples=1000)
    assert result["score"]["best_baseline"] == "B1"
    assert result["score"]["delta"] == pytest.approx(0.1)


def test_paired_bootstrap_rejects_unpaired_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1, 2], [1])
