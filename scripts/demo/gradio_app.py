#!/usr/bin/env python3
"""Local GPU demo for side-by-side AnyFlow B0 and full MUGen B3 generation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn.functional as F
from PIL import Image

from mugen.common.interfaces import ConditionBundle
from mugen.data.feature_store import load_feature_store
from mugen.encoders import ImageBindEncoder
from mugen.generation.conditioner import MultimodalConditioner
from mugen.generation.generators.anyflow_generator import AnyFlowVideoGenerator


class MUGenDemo:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("the full MUGen demo requires a CUDA GPU")
        self.args = args
        self.tensors, self.records, self.feature_manifest = load_feature_store(args.feature_store)
        self.imagebind = ImageBindEncoder()
        self.generator = AnyFlowVideoGenerator(
            model_id=args.model,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
        )
        dims = {name: int(self.tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
        dims["reference"] = int(self.tensors["video"].shape[-1])
        generator_dim = int(self.generator.pipeline.text_encoder.config.d_model)
        self.conditioner = MultimodalConditioner(
            dims=dims,
            hidden_dim=dims["reference"],
            generator_dim=generator_dim,
            num_reference_tokens=4,
        ).to(self.generator.device)
        conditioner_path = Path(args.lora_checkpoint) / "conditioner.pt"
        if not conditioner_path.is_file():
            raise FileNotFoundError(f"conditioner checkpoint missing: {conditioner_path}")
        self.conditioner.load_state_dict(
            torch.load(conditioner_path, map_location=self.generator.device, weights_only=True)
        )
        self.conditioner.eval()
        self.generator.pipeline.load_lora_weights(
            args.lora_checkpoint,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="mugen",
        )

    def _features(self, text, image_path, audio_path):
        return {
            "text": self.imagebind.encode_text(text).embedding,
            "image": self.imagebind.encode_image(image_path).embedding,
            "audio": self.imagebind.encode_audio(audio_path).embedding,
        }

    def _retrieve(self, embeddings):
        device = self.generator.device
        inputs = {name: (value.to(device), None) for name, value in embeddings.items()}
        with torch.no_grad():
            query = self.conditioner.fusion(inputs)
        gallery = self.tensors["video"].to(device)
        scores = F.normalize(query.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
        values, indices = scores[0].topk(min(self.args.top_k, len(self.records)))
        references = gallery[indices].unsqueeze(0)
        metadata = [
            {
                "rank": rank + 1,
                "score": float(values[rank]),
                "sample_id": self.records[int(index)]["sample_id"],
                "caption": self.records[int(index)]["caption"],
                "video_path": self.records[int(index)]["video_path"],
            }
            for rank, index in enumerate(indices)
        ]
        return references, values.unsqueeze(0), metadata

    @staticmethod
    def _save_video(frames, prefix):
        output = Path(tempfile.mkdtemp(prefix="mugen-demo-")) / f"{prefix}.mp4"
        array = frames.permute(0, 2, 3, 1).mul(255).byte().numpy()
        iio.imwrite(output, array, fps=8, codec="libx264")
        return str(output)

    def generate(self, text, image_path, audio_path, seed):
        if not text or not image_path or not audio_path:
            raise ValueError("text, image, and audio are required")
        image = Image.open(image_path).convert("RGB")
        embeddings = self._features(text, image_path, audio_path)
        references, scores, reference_metadata = self._retrieve(embeddings)
        with torch.no_grad():
            full_bundle = self.conditioner(
                prompt=text,
                image_condition=image,
                modality_embeddings={
                    name: value.to(self.generator.device) for name, value in embeddings.items()
                },
                reference_embeddings=references,
                reference_scores=scores,
            )
        baseline_bundle = ConditionBundle(
            prompt=text,
            image=image,
            modality_mask={"text": True, "image": True, "audio": False, "reference": False},
        )
        torch.cuda.reset_peak_memory_stats()
        self.generator.pipeline.disable_lora()
        start = time.perf_counter()
        baseline = self.generator.generate(baseline_bundle, seed=int(seed))
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter() - start
        self.generator.pipeline.enable_lora()
        start = time.perf_counter()
        full = self.generator.generate(full_bundle, seed=int(seed))
        torch.cuda.synchronize()
        full_seconds = time.perf_counter() - start
        peak_vram = torch.cuda.max_memory_allocated() / 2**30
        gate_weights = full_bundle.metadata["gate_weights"].tolist()
        runtime = {
            "generation_seed": int(seed),
            "baseline_seconds": baseline_seconds,
            "full_seconds": full_seconds,
            "adapter_overhead_percent": 100 * (full_seconds - baseline_seconds) / baseline_seconds,
            "peak_vram_gib": peak_vram,
            "condition_tokens": full_bundle.metadata["condition_token_count"],
        }
        return (
            self._save_video(baseline.video_frames, "b0-baseline"),
            self._save_video(full.video_frames, "b3-full"),
            reference_metadata,
            {"modalities": list(embeddings), "weights": gate_weights},
            runtime,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the local MUGen B0/B3 GPU demo")
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--model", default="nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("install the demo dependencies with: pip install -e '.[serving]'") from exc
    backend = MUGenDemo(args)
    with gr.Blocks(title="MUGen-VR") as demo:
        gr.Markdown("# MUGen-VR: B0 vs B3")
        with gr.Row():
            text = gr.Textbox(label="Text prompt")
            image = gr.Image(label="Image condition", type="filepath")
            audio = gr.Audio(label="Audio condition", type="filepath")
        seed = gr.Number(label="Generation seed", value=42, precision=0)
        run = gr.Button("Generate", variant="primary")
        with gr.Row():
            baseline_video = gr.Video(label="B0: AnyFlow image + prompt")
            full_video = gr.Video(label="B3: Fusion + audio + reference")
        with gr.Row():
            references = gr.JSON(label="Top-k references")
            gates = gr.JSON(label="Fusion gates")
            runtime = gr.JSON(label="Runtime")
        run.click(
            backend.generate,
            inputs=[text, image, audio, seed],
            outputs=[baseline_video, full_video, references, gates, runtime],
        )
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
