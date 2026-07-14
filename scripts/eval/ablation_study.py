#!/usr/bin/env python3
"""Summarize B0-B5 ablations and enforce paired-bootstrap release gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mugen.evaluation.statistics import compare_full_to_best_baselines


VARIANTS = {
    "B0": "AnyFlow image + original prompt",
    "B1": "prompt rewrite prototype",
    "B2": "real retrieval + reference video prefix",
    "B3": "fusion tokens without reference",
    "B4": "fusion + reference adapter without audio",
    "B5": "full text + image + audio + reference",
}
PRIMARY = {"retrieval_mrr", "vbench_total", "audio_control"}
NON_REGRESSION = {"subject_consistency", "temporal_consistency"}
LOWER_IS_BETTER = {"latency_seconds", "peak_vram_gib"}


def parse_args():
    parser = argparse.ArgumentParser(description="Strict MUGen B0-B5 ablation analysis")
    parser.add_argument("--input", required=True, help="JSONL with pair_id, variant, and metrics")
    parser.add_argument("--output", default="reports/ablation/report.json")
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


def release_gates(comparisons, summary):
    gates = {}
    for metric in PRIMARY:
        result = comparisons.get(metric)
        gates[metric] = bool(result and result["ci_lower"] > 0)
    for metric in NON_REGRESSION:
        result = comparisons.get(metric)
        gates[metric] = bool(result and result["ci_upper"] >= 0)
    b0 = summary.get("B0", {}).get("means", {})
    b5 = summary.get("B5", {}).get("means", {})
    baseline_latency = b0.get("latency_seconds")
    full_latency = b5.get("latency_seconds")
    gates["latency_overhead"] = bool(
        baseline_latency
        and full_latency is not None
        and (full_latency - baseline_latency) / baseline_latency <= 0.10
    )
    gates["single_3090_vram"] = bool(b5.get("peak_vram_gib", float("inf")) <= 24.0)
    return {"checks": gates, "passed": bool(gates) and all(gates.values())}


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
        higher_is_better={metric: False for metric in LOWER_IS_BETTER},
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        "schema_version": 1,
        "summary": summary,
        "paired_bootstrap": comparisons,
        "release_gate": release_gates(comparisons, summary),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "release_gate": report["release_gate"]}))
    if not report["release_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
