#!/usr/bin/env python3
"""Convert MSR-VTT parquet metadata into the canonical JSONL manifest."""

import argparse
from collections import Counter

from mugen.data.manifest import load_msrvtt_metadata, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", nargs="+")
    parser.add_argument("--output", default="data/msrvtt/source_manifest.jsonl")
    args = parser.parse_args()
    rows = load_msrvtt_metadata(args.metadata)
    count = write_jsonl(rows, args.output)
    print({"output": args.output, "rows": count, "splits": dict(Counter(row["split"] for row in rows))})


if __name__ == "__main__":
    main()
