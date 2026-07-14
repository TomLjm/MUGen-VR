#!/usr/bin/env python3
"""Run MUGen evaluation and write a unified report."""
import argparse
import os

import torch

from mugen.evaluation.custom_metrics import CustomMetrics
from mugen.evaluation.report_generator import ReportGenerator
from mugen.evaluation.vbench_adapter import VBenchAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/demo_report")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    smoke_video = torch.zeros(8, 3, 64, 64)
    custom = CustomMetrics().evaluate(smoke_video)
    vbench = VBenchAdapter().evaluate(args.generated_dir, args.reference_dir)
    payload = {
        "inputs": {"generated_dir": args.generated_dir, "reference_dir": args.reference_dir},
        "custom_metrics": {"metrics": custom.metrics, "details": custom.details},
        "vbench": {"metrics": vbench.metrics, "details": vbench.details},
    }
    report = ReportGenerator(args.output).write(payload)
    print(f"report={report}")


if __name__ == "__main__":
    main()
