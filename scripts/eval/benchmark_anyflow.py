#!/usr/bin/env python3
"""Benchmark AnyFlow latency and memory across inference step counts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from mugen.common.interfaces import ConditionBundle
from mugen.generation.generators.anyflow_generator import AnyFlowVideoGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AnyFlow inference")
    parser.add_argument("--model", default="models/anyflow-local")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", default="1,2,4,8")
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--output", default="reports/generated/anyflow_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AnyFlow benchmark requires CUDA")
    image = Image.open(args.image).convert("RGB")
    bundle = ConditionBundle(prompt=args.prompt, image=image, modality_mask={"image": True})
    generator = AnyFlowVideoGenerator(
        model_id=args.model,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
    )
    results = []
    for steps in [int(value) for value in args.steps.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = generator.generate(bundle, num_inference_steps=steps, seed=42)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results.append(
            {
                "steps": steps,
                "seconds": elapsed,
                "frames": int(result.video_frames.shape[0]),
                "generated_fps": float(result.video_frames.shape[0] / elapsed),
                "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "results": results}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
