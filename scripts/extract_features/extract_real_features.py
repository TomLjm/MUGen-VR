#!/usr/bin/env python3
"""Extract release-grade ImageBind and InternVideo features into safe shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mugen.data.feature_store import FeatureShardWriter
from mugen.data.manifest import read_jsonl
from mugen.encoders import ImageBindEncoder, InternVideoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="cache/features/msrvtt-real-v1")
    parser.add_argument("--internvideo-checkpoint", required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-valid", type=int, default=5000)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.manifest) if row.get("status") == "ok"]
    if args.limit:
        rows = rows[: args.limit]
    if args.num_partitions < 1 or not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must be in [0, num-partitions)")
    rows = rows[args.partition_index :: args.num_partitions]
    imagebind = ImageBindEncoder()
    internvideo = InternVideoEncoder(args.internvideo_checkpoint)
    writer = FeatureShardWriter(
        args.output,
        {"imagebind": imagebind.version, "internvideo": internvideo.version},
        shard_size=args.shard_size,
        resume=args.resume,
    )
    rows = [row for row in rows if row["sample_id"] not in writer.existing_sample_ids]
    failures = []
    for index, row in enumerate(rows, start=1):
        try:
            text = imagebind.encode_text(row["caption"]).embedding[0]
            image = imagebind.encode_image(row["keyframe_path"]).embedding[0]
            audio = imagebind.encode_audio(row["audio_path"]).embedding[0]
            video = internvideo.encode_video(row["video_path"]).embedding[0]
            writer.add(
                {**row, "feature_source": "real"},
                {"text": text, "image": image, "audio": audio, "video": video, "reference": video},
            )
        except Exception as exc:
            failures.append({"sample_id": row["sample_id"], "reason": str(exc)[:4000]})
        if index % 25 == 0:
            print({"processed": index, "failures": len(failures)})
    manifest = writer.close()
    failure_path = Path(args.output) / "failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    new_valid = len(rows) - len(failures)
    valid = len(writer.existing_sample_ids)
    print(
        {
            "manifest": str(manifest),
            "valid": valid,
            "new_valid": new_valid,
            "failed": len(failures),
        }
    )
    if args.num_partitions == 1 and not args.limit and valid < args.min_valid:
        raise RuntimeError(f"only {valid} real feature rows; release requires {args.min_valid}")


if __name__ == "__main__":
    main()
