#!/usr/bin/env python3
"""Run MUGen evaluation and write a unified report."""
import argparse
import os
from pathlib import Path

from mugen.evaluation.custom_metrics import CustomMetrics
from mugen.evaluation.report_generator import ReportGenerator
from mugen.evaluation.vbench_adapter import VBenchAdapter
from mugen.evaluation.video_io import find_videos


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/demo_report")
    parser.add_argument(
        "--allow-missing-vbench",
        action="store_true",
        help="Write custom metrics when VBench is unavailable. Never use this for release results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    videos = find_videos(args.generated_dir)
    custom = CustomMetrics().evaluate_video_files(videos)
    vbench = VBenchAdapter(output_path=Path(args.output) / "vbench").evaluate(
        args.generated_dir,
        args.reference_dir,
        required=not args.allow_missing_vbench,
    )
    payload = {
        "inputs": {
            "generated_dir": args.generated_dir,
            "reference_dir": args.reference_dir,
            "videos": [str(path) for path in videos],
        },
        "custom_metrics": {"metrics": custom.metrics, "details": custom.details},
        "vbench": {"metrics": vbench.metrics, "details": vbench.details},
    }
    report = ReportGenerator(args.output).write(payload)
    print(f"report={report}")


if __name__ == "__main__":
    main()
