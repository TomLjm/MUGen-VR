#!/usr/bin/env python3
"""Run modality ablation study."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--experiment", type=str, default="modality_ablation")
    parser.add_argument("--output", type=str, default="./experiments/ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Ablation study: {args.experiment}")
    print("Ablation study - to be implemented")


if __name__ == "__main__":
    main()
