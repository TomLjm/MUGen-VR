#!/usr/bin/env python3
"""Build the public consistency report from completed VBench artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("subject_consistency", "motion_smoothness", "temporal_flickering")


def load_vbench_metrics(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload["vbench"]["metrics"]
    return {name: float(metrics[name]) for name in METRICS}


def build_report(baseline_path, mugen_path, samples=40, generation_seed=42):
    baseline = load_vbench_metrics(baseline_path)
    mugen = load_vbench_metrics(mugen_path)
    metrics = []
    for name in METRICS:
        before = baseline[name]
        after = mugen[name]
        row = {
            "name": name,
            "baseline": before,
            "mugen": after,
            "absolute_change": after - before,
        }
        if name == "subject_consistency":
            row["relative_error_reduction"] = (after - before) / (1.0 - before)
        metrics.append(row)
    return {
        "schema_version": 1,
        "title": "Targeted Consistency Evaluation",
        "protocol": {
            "samples": int(samples),
            "generation_seed": int(generation_seed),
            "backbone": "frozen AnyFlow-FAR 1.3B",
            "condition_scale": 0.05,
        },
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mugen", required=True)
    parser.add_argument("--output", default="reports/public/targeted-consistency.json")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--generation-seed", type=int, default=42)
    args = parser.parse_args()
    report = build_report(
        args.baseline, args.mugen, args.samples, args.generation_seed
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": len(report["metrics"])}))


if __name__ == "__main__":
    main()
