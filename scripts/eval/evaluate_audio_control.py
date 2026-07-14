#!/usr/bin/env python3
"""Evaluate real audio conditioning against generated videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mugen.encoders import ImageBindEncoder
from mugen.evaluation.audio_control import (
    imagebind_audio_video_alignment,
    onset_flow_correlation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audio-video control evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/audio_control/report.json")
    parser.add_argument("--frame-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("audio-control manifest is empty")
    encoder = ImageBindEncoder()
    results = []
    for row in rows:
        audio_path = row["input_audio_path"]
        video_path = row["generated_video_path"]
        results.append(
            {
                "sample_id": row["sample_id"],
                "variant": row.get("variant", "B5"),
                "generation_seed": row.get("generation_seed"),
                "imagebind_audio_video_alignment": imagebind_audio_video_alignment(
                    encoder, audio_path, video_path, args.frame_count
                ),
                "onset_flow_correlation": onset_flow_correlation(audio_path, video_path),
            }
        )
    metrics = ["imagebind_audio_video_alignment", "onset_flow_correlation"]
    aggregate = {
        metric: float(sum(row[metric] for row in results) / len(results)) for metric in metrics
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"aggregate": aggregate, "per_sample": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "aggregate": aggregate}))


if __name__ == "__main__":
    main()
