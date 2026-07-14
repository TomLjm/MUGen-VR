#!/usr/bin/env python3
"""Video generation demo with smoke and AnyFlow modes."""
import argparse
import os

import torch
from PIL import Image

from mugen.common.interfaces import GenerationResult
from mugen.common.utils import set_seed
from mugen.generation.generators.anyflow_generator import AnyFlowVideoGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="MUGen generation demo")
    parser.add_argument("--mode", choices=["smoke", "real"], default="real")
    parser.add_argument("--generator", choices=["dummy", "anyflow"], default="dummy")
    parser.add_argument("--model_id", type=str, default="models/anyflow-local")
    parser.add_argument("--prompt", "--text", dest="prompt", type=str, default="a dynamic scene")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_inference_steps", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--chunk_partition", type=str, default="1,2,2,2")
    parser.add_argument("--output", type=str, default="outputs/generated.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-dummy", action="store_true")
    return parser.parse_args()


def make_smoke_video(num_frames, height=128, width=128):
    frames = torch.zeros(num_frames, 3, height, width)
    for i in range(num_frames):
        frames[i, 0] = i / max(1, num_frames - 1)
        frames[i, 1, :, i % width] = 1.0
        frames[i, 2, i % height, :] = 1.0
    return GenerationResult(video_frames=frames, metadata={"mode": "smoke", "generator": "dummy"})


def save_video_tensor(result, output):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    frames = (result.video_frames.detach().cpu().clamp(0, 1) * 255).byte()
    if not output.lower().endswith(".gif"):
        output = output.rsplit(".", 1)[0] + ".gif"
    pil_frames = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]
    pil_frames[0].save(output, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
    return output


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "smoke" or args.generator == "dummy":
        result = make_smoke_video(args.num_frames, height=min(args.height, 128), width=min(args.width, 128))
    else:
        if not args.image:
            raise ValueError("AnyFlow mode requires --image")
        image = Image.open(args.image).convert("RGB")
        gen = AnyFlowVideoGenerator(
            model_id=args.model_id,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
        )
        if gen.pipeline is None and not args.allow_dummy:
            raise RuntimeError("AnyFlow failed to load. Use --mode smoke for CPU smoke tests.")
        chunk_partition = [int(x) for x in args.chunk_partition.split(",") if x.strip()]
        result = gen.generate(
            {"image": image, "prompt": args.prompt},
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            chunk_partition=chunk_partition,
        )
    output = save_video_tensor(result, args.output)
    print(f"generated={output}")
    print(f"metadata={result.metadata}")


if __name__ == "__main__":
    main()
