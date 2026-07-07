#!/usr/bin/env python3
"""Test model robustness under modality perturbations."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Robustness test")
    parser.add_argument("--output", type=str, default="./experiments/robustness")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Robustness test - to be implemented")


if __name__ == "__main__":
    main()
