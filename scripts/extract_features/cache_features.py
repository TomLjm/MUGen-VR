#!/usr/bin/env python3
"""Cache video features for faster retrieval."""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Cache video features")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, default="./cache/features")
    parser.add_argument("--encoder", type=str, default="internvideo")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Caching features using {args.encoder} encoder")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
