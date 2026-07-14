#!/usr/bin/env python3
"""Build deterministic train/val/test metadata from MSR-VTT 9K/1K JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json

from mugen.data.manifest import validate_split_isolation, write_jsonl


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list: {path}")
    return rows


def validation_ids(train_rows, val_size):
    if val_size < 1 or val_size >= len(train_rows):
        raise ValueError("val_size must be positive and smaller than the 9K training split")
    ranked = sorted(
        (hashlib.sha256(str(row["video_id"]).encode()).hexdigest(), str(row["video_id"]))
        for row in train_rows
    )
    return {video_id for _, video_id in ranked[:val_size]}


def normalize_row(row, split, revision):
    captions = row.get("caption", row.get("captions"))
    if isinstance(captions, str):
        captions = [captions]
    if not captions:
        raise ValueError(f"video {row.get('video_id')} has no captions")
    video_id = str(row["video_id"])
    return {
        "sample_id": f"{split}:{video_id}",
        "video_id": video_id,
        "caption": str(captions[0]),
        "captions": [str(value) for value in captions],
        "split": split,
        "source_revision": revision,
    }


def build_manifest(train_rows, test_rows, val_size, revision):
    held_out = validation_ids(train_rows, val_size)
    records = [
        normalize_row(row, "val" if str(row["video_id"]) in held_out else "train", revision)
        for row in train_rows
    ]
    records.extend(normalize_row(row, "test", revision) for row in test_rows)
    validate_split_isolation(records)
    return sorted(records, key=lambda row: row["sample_id"])


def main():
    parser = argparse.ArgumentParser(description="Build MSR-VTT 9K/1K canonical manifest")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output", default="data/msrvtt/archive_source_manifest.jsonl")
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    records = build_manifest(
        load_rows(args.train_json),
        load_rows(args.test_json),
        args.val_size,
        args.revision,
    )
    write_jsonl(records, args.output)
    counts = {split: sum(row["split"] == split for row in records) for split in ["train", "val", "test"]}
    print({"output": args.output, "rows": len(records), "splits": counts})


if __name__ == "__main__":
    main()
