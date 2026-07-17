#!/usr/bin/env python3
"""Measure whether source clips contain learnable audio-onset/motion alignment."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mugen.data.feature_store import load_feature_store
from mugen.evaluation.audio_control import onset_flow_correlation


def evaluate_row(row):
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "correlation": onset_flow_correlation(row["audio_path"], row["video_path"]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    _, records, _ = load_feature_store(args.feature_store)
    rows = [row for row in records if row["split"] == args.split]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_row, rows))
    values = np.asarray([row["correlation"] for row in results], dtype=np.float64)
    summary = {
        "count": len(results),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_fraction": float((values > 0).mean()),
        "above_0_2_fraction": float((values > 0.2).mean()),
        "below_minus_0_2_fraction": float((values < -0.2).mean()),
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "per_sample": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
