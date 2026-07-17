#!/usr/bin/env python3
"""Add time-resolved audio features to an existing real feature store."""

from __future__ import annotations

import argparse

from mugen.data.feature_store import FeatureShardWriter, load_feature_store
from mugen.encoders.temporal_audio import (
    TEMPORAL_AUDIO_VERSION,
    extract_temporal_audio_features,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_partitions < 1 or not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must be in [0, num-partitions)")
    tensors, records, manifest = load_feature_store(args.input)
    indices = list(range(len(records)))[args.partition_index :: args.num_partitions]
    if args.limit:
        indices = indices[: args.limit]
    versions = {**manifest["encoder_versions"], "temporal_audio": TEMPORAL_AUDIO_VERSION}
    writer = FeatureShardWriter(args.output, versions, shard_size=args.shard_size)
    for position, index in enumerate(indices, start=1):
        record = records[index]
        temporal = extract_temporal_audio_features(record["audio_path"])
        row_tensors = {name: values[index] for name, values in tensors.items()}
        row_tensors["audio_temporal"] = temporal
        writer.add(record, row_tensors)
        if position % 100 == 0:
            print({"processed": position, "partition": args.partition_index})
    output = writer.close()
    print({"manifest": str(output), "rows": len(indices)})


if __name__ == "__main__":
    main()
