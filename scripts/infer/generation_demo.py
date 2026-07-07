#!/usr/bin/env python3
"""Demo script for video generation."""
import argparse
from mugen.common.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generation demo")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--output", type=str, default="./outputs/generated.mp4")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Generating {args.num_frames} frames...")
    print("Generation demo - to be implemented")


if __name__ == "__main__":
    main()
