#!/usr/bin/env python3
"""Train the HierarchicalConditionFusion module."""
import argparse
from mugen.common.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion module")
    parser.add_argument("--config", type=str, default="configs/training/fusion.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config)
    print("Fusion module training - to be implemented")
    print(f"Config: {args.config}")


if __name__ == "__main__":
    main()
