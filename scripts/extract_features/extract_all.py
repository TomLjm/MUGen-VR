#!/usr/bin/env python3
"""Extract features for all videos in the dataset."""
import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract multi-modal features")
    parser.add_argument("--config", type=str, default="configs/data/default.yaml")
    parser.add_argument("--metadata", type=str, default="./data/metadata.json")
    parser.add_argument("--output_dir", type=str, default="./cache/features")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Feature extraction pipeline - to be implemented")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
