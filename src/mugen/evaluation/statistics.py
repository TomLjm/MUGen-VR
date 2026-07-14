"""Paired statistical comparisons for release evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def paired_bootstrap_delta(
    full,
    baseline,
    *,
    higher_is_better=True,
    samples=10_000,
    confidence=0.95,
    seed=42,
):
    full = np.asarray(full, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if full.shape != baseline.shape or full.ndim != 1 or len(full) < 2:
        raise ValueError("paired bootstrap requires equal one-dimensional arrays with at least two pairs")
    raw_delta = full - baseline if higher_is_better else baseline - full
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(raw_delta), size=(samples, len(raw_delta)))
    bootstrapped = raw_delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "pairs": int(len(raw_delta)),
        "delta": float(raw_delta.mean()),
        "ci_lower": float(np.quantile(bootstrapped, alpha)),
        "ci_upper": float(np.quantile(bootstrapped, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": samples,
    }


def organize_ablation_rows(rows):
    organized = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        pair_id = str(row.get("pair_id", row.get("sample_id", "")))
        variant = str(row.get("variant", ""))
        metrics = row.get("metrics", {})
        if not pair_id or not variant or not isinstance(metrics, dict):
            raise ValueError("every row requires pair_id/sample_id, variant, and a metrics object")
        if variant in organized[pair_id]:
            raise ValueError(f"duplicate pair/variant row: {pair_id}/{variant}")
        organized[pair_id][variant] = metrics
    return organized


def compare_full_to_best_baselines(
    rows,
    *,
    full_variant="B5",
    baseline_variants=("B0", "B1", "B2", "B3", "B4"),
    higher_is_better=None,
    samples=10_000,
    seed=42,
):
    organized = organize_ablation_rows(rows)
    higher_is_better = higher_is_better or {}
    metrics = sorted(
        set.intersection(
            *[
                set(values[full_variant])
                for values in organized.values()
                if full_variant in values
            ]
        )
    )
    if not metrics:
        raise ValueError(f"no common metrics found for {full_variant}")
    comparisons = {}
    for metric in metrics:
        direction = bool(higher_is_better.get(metric, True))
        candidates = {}
        for baseline in baseline_variants:
            pairs = [
                pair_id
                for pair_id, variants in organized.items()
                if full_variant in variants
                and baseline in variants
                and metric in variants[full_variant]
                and metric in variants[baseline]
            ]
            if len(pairs) < 2:
                continue
            baseline_values = np.asarray([organized[pair][baseline][metric] for pair in pairs])
            score = baseline_values.mean()
            candidates[baseline] = (score, pairs)
        if not candidates:
            continue
        best = max(candidates, key=lambda name: candidates[name][0]) if direction else min(
            candidates, key=lambda name: candidates[name][0]
        )
        _, pairs = candidates[best]
        result = paired_bootstrap_delta(
            [organized[pair][full_variant][metric] for pair in pairs],
            [organized[pair][best][metric] for pair in pairs],
            higher_is_better=direction,
            samples=samples,
            seed=seed,
        )
        comparisons[metric] = {"best_baseline": best, **result}
    return comparisons
