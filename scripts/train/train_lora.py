#!/usr/bin/env python3
"""LoRA fine-tuning for the generation model."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/training/lora.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    print("LoRA fine-tuning - to be implemented")
    print(f"Config: {args.config}")


if __name__ == "__main__":
    main()
