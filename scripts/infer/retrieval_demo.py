#!/usr/bin/env python3
"""Demo script for cross-modal retrieval."""
import argparse
from mugen.common.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval demo")
    parser.add_argument("--query", type=str, default="a dog running in the park")
    parser.add_argument("--modality", type=str, default="text", choices=["text", "image", "audio"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_path", type=str, default="./cache/features/index.pkl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Query ({args.modality}): {args.query}")
    print(f"Retrieving top-{args.top_k}...")
    print("Retrieval demo - to be implemented")


if __name__ == "__main__":
    main()
