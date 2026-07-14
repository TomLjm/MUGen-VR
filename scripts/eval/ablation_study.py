#!/usr/bin/env python3
"""Summarize the four resume-project ablations with optional bootstrap diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mugen.evaluation.statistics import compare_full_to_best_baselines


VARIANTS = {
    "B0": "AnyFlow image + original prompt",
    "B1": "prompt rewrite prototype",
    "B2": "fusion tokens without audio or reference",
    "B3": "full fusion + audio + reference",
}
LOWER_IS_BETTER = {"latency_seconds", "peak_vram_gib"}
REQUIRED_METRICS = {
    "vbench_total",
    "retrieval_mrr",
    "onset_flow_correlation",
    "latency_seconds",
    "peak_vram_gib",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Strict MUGen B0-B5 ablation analysis")
    parser.add_argument("--input", required=True, help="JSONL with pair_id, variant, and metrics")
    parser.add_argument("--output", default="reports/ablation/report.json")
    parser.add_argument("--case-manifest", help="Optional JSONL describing 6-10 side-by-side cases")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows):
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["variant"] not in VARIANTS:
            raise ValueError(f"unknown ablation variant: {row['variant']}")
        for metric, value in row["metrics"].items():
            if value is not None:
                values[row["variant"]][metric].append(float(value))
    return {
        variant: {
            "description": VARIANTS[variant],
            "count": max((len(items) for items in metrics.values()), default=0),
            "means": {metric: sum(items) / len(items) for metric, items in metrics.items()},
        }
        for variant, metrics in values.items()
    }


def completion_check(rows, case_count=None):
    pair_variants = defaultdict(set)
    seeds = set()
    metrics = defaultdict(set)
    for row in rows:
        pair_id = str(row.get("pair_id", row.get("sample_id", "")))
        pair_variants[pair_id].add(row["variant"])
        seeds.add(row.get("generation_seed"))
        metrics[row["variant"]].update(
            key for key, value in row.get("metrics", {}).items() if value is not None
        )
    checks = {
        "held_out_pairs_30_to_50": 30 <= len(pair_variants) <= 50,
        "all_pairs_have_b0_to_b3": all(values == set(VARIANTS) for values in pair_variants.values()),
        "uniform_generation_seed": len(seeds) == 1 and None not in seeds,
        "required_metrics_present": all(
            REQUIRED_METRICS <= metrics[variant] for variant in VARIANTS
        ),
        "side_by_side_cases_6_to_10": case_count is not None and 6 <= case_count <= 10,
    }
    return {"checks": checks, "passed": all(checks.values())}


def main():
    args = parse_args()
    rows = load_rows(args.input)
    present = {row.get("variant") for row in rows}
    missing = set(VARIANTS) - present
    if missing:
        raise ValueError(f"ablation input is missing variants: {sorted(missing)}")
    summary = summarize(rows)
    comparisons = compare_full_to_best_baselines(
        rows,
        full_variant="B3",
        baseline_variants=("B0", "B1", "B2"),
        higher_is_better={metric: False for metric in LOWER_IS_BETTER},
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    case_count = None
    if args.case_manifest:
        case_count = len(load_rows(args.case_manifest))
    report = {
        "schema_version": 1,
        "summary": summary,
        "paired_bootstrap_advisory": comparisons,
        "completion_check": completion_check(rows, case_count),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "completion_check": report["completion_check"]}))


if __name__ == "__main__":
    main()
