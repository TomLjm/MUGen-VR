#!/usr/bin/env python3
"""Merge deterministic extraction partitions into one release feature store."""

from __future__ import annotations

import argparse

from mugen.data.feature_store import FeatureShardWriter, load_feature_store


def main():
    parser = argparse.ArgumentParser(description="Merge real MUGen feature stores")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--min-valid", type=int, default=5000)
    args = parser.parse_args()
    writer = None
    total = 0
    for path in args.inputs:
        tensors, records, manifest = load_feature_store(path)
        if writer is None:
            writer = FeatureShardWriter(
                args.output, manifest["encoder_versions"], shard_size=args.shard_size
            )
        elif writer.encoder_versions != manifest["encoder_versions"]:
            raise ValueError(f"encoder version mismatch: {path}")
        for index, record in enumerate(records):
            writer.add(record, {name: values[index] for name, values in tensors.items()})
            total += 1
    if writer is None:
        raise ValueError("at least one input feature store is required")
    manifest_path = writer.close()
    print({"manifest": str(manifest_path), "rows": total})
    if total < args.min_valid:
        raise RuntimeError(f"only {total} merged rows; release requires {args.min_valid}")


if __name__ == "__main__":
    main()
