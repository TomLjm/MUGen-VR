#!/usr/bin/env python3
"""Run full evaluation pipeline."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="./reports")
    parser.add_argument("--vbench_config", type=str, default="configs/evaluation/vbench.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Evaluation pipeline - to be implemented")
    print(f"Generated videos: {args.generated_dir}")


if __name__ == "__main__":
    main()
