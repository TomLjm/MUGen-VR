#!/usr/bin/env python3
"""Download and prepare a dataset for MUGen-VR."""
import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for MUGen-VR")
    parser.add_argument("--dataset", type=str, default="webvid", choices=["webvid", "msrvtt", "youcook2"])
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument("--clip_duration", type=int, default=8)
    return parser.parse_args()


def prepare_webvid(output_dir, max_videos):
    print("Preparing WebVid dataset...")
    print(f"Output: {output_dir}, max: {max_videos}")
    print("WebVid preparation: placeholder")


def prepare_msrvtt(output_dir, max_videos):
    print("Preparing MSR-VTT...")
    os.makedirs(output_dir, exist_ok=True)
    metadata = {"name": "MSR-VTT", "version": "1.0", "total_clips": 0, "clips": []}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("MSR-VTT metadata template created.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset == "webvid":
        prepare_webvid(args.output_dir, args.max_videos)
    elif args.dataset == "msrvtt":
        prepare_msrvtt(args.output_dir, args.max_videos)
    else:
        print(f"Dataset {args.dataset} not yet supported")
    print(f"Dataset preparation complete: {args.output_dir}")


if __name__ == "__main__":
    main()
