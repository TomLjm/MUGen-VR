#!/usr/bin/env python3
"""Generate the fixed B0-B3 project ablation set with real MUGen checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from PIL import Image

from mugen.common.interfaces import ConditionBundle
from mugen.data.feature_store import load_feature_store
from mugen.data.manifest import read_jsonl
from mugen.generation.conditioner import MultimodalConditioner
from mugen.generation.generators.anyflow_generator import AnyFlowVideoGenerator


VARIANTS = ("B0", "B1", "B2", "B3")


def rewrite_prompt(caption, references):
    reference_caption = references[0]["caption"] if references else "retrieved visual reference"
    return (
        f"{caption}, reference-guided composition: {reference_caption}, "
        "audio-reactive rhythmic motion, high temporal consistency, realistic motion, "
        "cinematic natural lighting"
    )


def save_video(frames, path, fps=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = frames.permute(0, 2, 3, 1).mul(255).byte().numpy()
    iio.imwrite(path, array, fps=fps, codec="libx264")


def gather_reference_embeddings(gallery, indices):
    """Keep the reference-adapter contract as [batch, top_k, feature_dim]."""
    references = gallery[indices]
    if references.ndim != 3:
        raise ValueError(f"expected 3D reference embeddings, got {tuple(references.shape)}")
    return references


class AblationGenerator:
    def __init__(self, args):
        self.args = args
        self.tensors, self.records, self.feature_manifest = load_feature_store(args.feature_store)
        self.by_sample_id = {row["sample_id"]: index for index, row in enumerate(self.records)}
        self.generator = AnyFlowVideoGenerator(
            model_id=args.model,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        )
        dims = {name: int(self.tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
        dims["reference"] = int(self.tensors["video"].shape[-1])
        self.conditioner = MultimodalConditioner(
            dims=dims,
            hidden_dim=dims["reference"],
            generator_dim=int(self.generator.pipeline.text_encoder.config.d_model),
            num_reference_tokens=4,
        ).to(self.generator.device)
        checkpoint = Path(args.lora_checkpoint)
        self.conditioner.load_state_dict(
            torch.load(checkpoint / "conditioner.pt", map_location=self.generator.device, weights_only=True)
        )
        self.conditioner.eval()
        self.generator.pipeline.load_lora_weights(
            checkpoint,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="mugen",
        )

    def conditions(self, row):
        index = self.by_sample_id.get(row["sample_id"])
        if index is None:
            raise KeyError(f"evaluation sample is absent from feature store: {row['sample_id']}")
        device = self.generator.device
        embeddings = {
            name: self.tensors[name][index].unsqueeze(0).to(device)
            for name in ["text", "image", "audio"]
        }
        with torch.no_grad():
            initial = self.conditioner.fusion(
                {name: (value, None) for name, value in embeddings.items()}
            )
        gallery_indices = [
            gallery_index for gallery_index, record in enumerate(self.records) if record["split"] == "train"
        ]
        gallery = self.tensors["video"][gallery_indices].to(device)
        scores = F.normalize(initial.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
        values, indices = scores[0].topk(min(self.args.top_k, len(gallery_indices)))
        references = gather_reference_embeddings(gallery, indices)
        reference_metadata = [
            {
                "sample_id": self.records[gallery_indices[int(reference_index)]]["sample_id"],
                "caption": self.records[gallery_indices[int(reference_index)]]["caption"],
                "score": float(values[position]),
            }
            for position, reference_index in enumerate(indices)
        ]
        image = Image.open(row["image_path"]).convert("RGB")
        with torch.no_grad():
            full = self.conditioner(
                prompt=row["caption"],
                image_condition=image,
                modality_embeddings=embeddings,
                reference_embeddings=references,
                reference_scores=values.unsqueeze(0),
                metadata={"sample_id": row["sample_id"]},
            )
            no_audio_reference = self.conditioner(
                prompt=row["caption"],
                image_condition=image,
                modality_embeddings={"text": embeddings["text"], "image": embeddings["image"]},
                reference_embeddings=None,
                reference_scores=None,
                metadata={"sample_id": row["sample_id"]},
            )
        no_audio_reference.condition_tokens = no_audio_reference.condition_tokens[:, :3]
        baseline = ConditionBundle(prompt=row["caption"], image=image)
        rewritten = ConditionBundle(prompt=rewrite_prompt(row["caption"], reference_metadata), image=image)
        return {
            "B0": baseline,
            "B1": rewritten,
            "B2": no_audio_reference,
            "B3": full,
        }, reference_metadata

    def run_variant(self, row, variant, bundle, references, output_dir):
        if variant in {"B0", "B1"}:
            self.generator.pipeline.disable_lora()
        else:
            self.generator.pipeline.enable_lora()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self.generator.generate(bundle, seed=int(row["generation_seed"]))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        video_path = output_dir / variant / f"{row['video_id']}.mp4"
        save_video(result.video_frames, video_path)
        gates = bundle.metadata.get("gate_weights")
        return {
            "pair_id": row["pair_id"],
            "sample_id": row["sample_id"],
            "variant": variant,
            "generation_seed": int(row["generation_seed"]),
            "caption": row["caption"],
            "input_audio_path": row["input_audio_path"],
            "generated_video_path": str(video_path),
            "references": references if variant == "B3" else [],
            "gates": gates.tolist() if isinstance(gates, torch.Tensor) else None,
            "metrics": {
                "latency_seconds": seconds,
                "generated_fps": float(result.video_frames.shape[0] / seconds),
                "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed MUGen B0-B3 ablations")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/project-ablation")
    parser.add_argument("--model", default="nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_partitions < 1 or not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must be in [0, num-partitions)")
    rows = read_jsonl(args.eval_manifest)[args.partition_index :: args.num_partitions]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results-part-{args.partition_index}.jsonl"
    complete = set()
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            complete = {
                (row["pair_id"], row["variant"])
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            }
    backend = AblationGenerator(args)
    with result_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            bundles, references = backend.conditions(row)
            for variant in VARIANTS:
                if (row["pair_id"], variant) in complete:
                    continue
                result = backend.run_variant(row, variant, bundles[variant], references, output_dir)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print({"pair_id": row["pair_id"], "variant": variant, **result["metrics"]})


if __name__ == "__main__":
    main()
