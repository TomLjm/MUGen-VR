#!/usr/bin/env python3
"""Build a deterministic validation-only manifest for condition-scale selection."""

from __future__ import annotations

import argparse
import hashlib

from mugen.data.manifest import read_jsonl, write_jsonl


def select_validation_rows(rows, sample_count=8, generation_seed=42):
    candidates = [
        row for row in rows if row.get("split") == "val" and row.get("status") == "ok"
    ]
    if not 6 <= sample_count <= 12:
        raise ValueError("condition-scale calibration requires 6-12 validation samples")
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(str(row["video_id"]).encode()).hexdigest(),
    )[:sample_count]
    if len(ranked) != sample_count:
        raise ValueError(f"only {len(ranked)} valid validation rows")
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


def main():
    parser = argparse.ArgumentParser(description="Build condition-scale validation manifest")
    parser.add_argument("--media-manifest", required=True)
    parser.add_argument("--output", default="data/msrvtt/condition_scale_val_8.jsonl")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=42)
    args = parser.parse_args()
    rows = select_validation_rows(
        read_jsonl(args.media_manifest), args.samples, args.generation_seed
    )
    write_jsonl(rows, args.output)
    print({"output": args.output, "samples": len(rows), "generation_seed": args.generation_seed})


if __name__ == "__main__":
    main()
