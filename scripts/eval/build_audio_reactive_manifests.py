#!/usr/bin/env python3
"""Build locked validation/test manifests from source audio-motion synchronization only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from mugen.data.feature_store import load_feature_store


def load_scores(paths):
    scores = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scores.update({row["sample_id"]: float(row["correlation"]) for row in payload["per_sample"]})
    return scores


def build_rows(records, scores, split, threshold, count, seed):
    candidates = [
        row
        for row in records
        if row["split"] == split and scores.get(row["sample_id"], -1.0) >= threshold
    ]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} {split} clips pass threshold {threshold}")
    return [
        {
            "pair_id": row["sample_id"],
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "caption": row["caption"],
            "image_path": row["keyframe_path"],
            "input_audio_path": row["audio_path"],
            "reference_video_path": row["video_path"],
            "source_onset_flow_correlation": scores[row["sample_id"]],
            "generation_seed": 42,
        }
        for row in candidates[:count]
    ]


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--audits", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=40)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, records, _ = load_feature_store(args.feature_store)
    scores = load_scores(args.audits)
    val_rows = build_rows(records, scores, "val", args.threshold, args.val_count, args.seed)
    test_rows = build_rows(records, scores, "test", args.threshold, args.test_count, args.seed)
    print(
        {
            "val": len(val_rows),
            "val_sha256": write_manifest(args.val_output, val_rows),
            "test": len(test_rows),
            "test_sha256": write_manifest(args.test_output, test_rows),
        }
    )


if __name__ == "__main__":
    main()
