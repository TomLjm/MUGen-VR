#!/usr/bin/env python3
"""Select a fixed held-out evaluation set and representative comparison cases."""

from __future__ import annotations

import argparse
import hashlib

from mugen.data.manifest import read_jsonl, write_jsonl


def select_rows(rows, sample_count, generation_seed):
    candidates = [
        row for row in rows if row.get("split") == "test" and row.get("status") == "ok"
    ]
    if not 30 <= sample_count <= 50:
        raise ValueError("project evaluation requires 30-50 held-out samples")
    if len(candidates) < sample_count:
        raise ValueError(f"only {len(candidates)} valid held-out rows")
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(str(row["video_id"]).encode()).hexdigest(),
    )[:sample_count]
    return [
        {
            "pair_id": row["sample_id"],
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "caption": row["caption"],
            "image_path": row["keyframe_path"],
            "input_audio_path": row["audio_path"],
            "reference_video_path": row["video_path"],
            "generation_seed": generation_seed,
        }
        for row in ranked
    ]


def select_cases(rows, case_count):
    if not 6 <= case_count <= 10:
        raise ValueError("side-by-side case count must be 6-10")
    positions = [round(index * (len(rows) - 1) / (case_count - 1)) for index in range(case_count)]
    return [
        {
            "pair_id": rows[position]["pair_id"],
            "caption": rows[position]["caption"],
            "generation_seed": rows[position]["generation_seed"],
            "selection": "deterministic coverage sample",
        }
        for position in positions
    ]


def main():
    parser = argparse.ArgumentParser(description="Build practical MUGen project evaluation manifests")
    parser.add_argument("--media-manifest", required=True)
    parser.add_argument("--output", default="data/msrvtt/project_eval_40.jsonl")
    parser.add_argument("--case-output", default="data/msrvtt/project_cases_8.jsonl")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=42)
    args = parser.parse_args()
    selected = select_rows(read_jsonl(args.media_manifest), args.samples, args.generation_seed)
    cases = select_cases(selected, args.cases)
    write_jsonl(selected, args.output)
    write_jsonl(cases, args.case_output)
    print(
        {
            "evaluation_manifest": args.output,
            "samples": len(selected),
            "case_manifest": args.case_output,
            "cases": len(cases),
            "generation_seed": args.generation_seed,
        }
    )


if __name__ == "__main__":
    main()
