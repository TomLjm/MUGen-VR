#!/usr/bin/env python3
"""Select held-out showcase pairs by subject-consistency improvement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def subject_scores(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["subject_consistency"][1]
    return {
        Path(row["video_path"]).stem: float(row["video_results"])
        for row in rows
    }


def select_cases(manifest, baseline_path, mugen_path, count=8):
    baseline = subject_scores(baseline_path)
    mugen = subject_scores(mugen_path)
    by_video = {str(row["video_id"]): row for row in read_jsonl(manifest)}
    ranked = sorted(
        set(baseline) & set(mugen),
        key=lambda video_id: mugen[video_id] - baseline[video_id],
        reverse=True,
    )
    cases = []
    for video_id in ranked[: int(count)]:
        row = by_video[video_id]
        cases.append(
            {
                "pair_id": row["pair_id"],
                "caption": row["caption"],
                "generation_seed": row["generation_seed"],
                "subject_consistency_change": mugen[video_id] - baseline[video_id],
            }
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mugen", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    cases = select_cases(
        args.manifest, args.baseline, args.mugen, args.count
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
