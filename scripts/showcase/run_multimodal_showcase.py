#!/usr/bin/env python3
"""MUGen-VR showcase entrypoint.

Default mode is `generate`: parse prompt audio style, fuse multimodal conditions,
retrieve references, build an enhanced prompt, and generate an MP4 with AnyFlow.
`report` is a CPU-friendly debug path. `smoke` is a minimal visual fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat

from mugen.conditioning import AudioPromptParser
from mugen.common.utils import set_seed
from mugen.fusion.fusion_module import HierarchicalConditionFusion
from mugen.generation.generators.anyflow_generator import AnyFlowVideoGenerator
from mugen.retrieval.retriever import CrossModalRetriever


DEMO_REFERENCES = [
    {
        "clip_id": "dog_grass_running_ref",
        "caption": "dog running across grass with natural handheld motion",
        "visual_tags": ["dog", "grass", "outdoor", "running", "low-angle"],
        "motion_tags": ["forward motion", "leg cadence", "camera follow"],
        "audio_tags": ["light rhythmic beat", "outdoor ambience"],
    },
    {
        "clip_id": "dog_closeup_ref",
        "caption": "close-up dog portrait with shallow depth of field",
        "visual_tags": ["dog", "close-up", "fur texture", "bokeh"],
        "motion_tags": ["subtle head movement", "slow camera push"],
        "audio_tags": ["quiet ambience"],
    },
    {
        "clip_id": "park_running_ref",
        "caption": "subject running in a park with stable temporal motion",
        "visual_tags": ["park", "green background", "running", "daylight"],
        "motion_tags": ["temporal consistency", "smooth tracking", "wide shot"],
        "audio_tags": ["steady rhythm", "footstep-like beat"],
    },
    {
        "clip_id": "cinematic_pet_ref",
        "caption": "cinematic pet video with warm color and dynamic background blur",
        "visual_tags": ["cinematic", "warm light", "pet", "background blur"],
        "motion_tags": ["subject centric", "controlled motion"],
        "audio_tags": ["soft soundtrack"],
    },
]


class DemoEmbeddingFactory:
    """Deterministic local embeddings for the showcase and report path."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def text(self, prompt: str) -> torch.Tensor:
        return self._hash_embedding(f"text:{prompt}")

    def image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.asarray(image).astype(np.float32) / 255.0
        stats = np.concatenate([
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            np.percentile(arr.reshape(-1, 3), [10, 50, 90], axis=0).reshape(-1),
        ])
        return self._numeric_embedding(stats, namespace=f"image:{Path(image_path).name}")

    def audio_from_file(self, audio_path: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats = self._audio_stats(audio_path)
        values = np.array(
            [
                stats["duration_sec"],
                stats["rms"],
                stats["zero_crossing_rate"],
                stats["peak"],
                stats["estimated_bpm"],
            ],
            dtype=np.float32,
        )
        return self._numeric_embedding(values, namespace=f"audio:{Path(audio_path).name}"), stats

    def audio_from_prompt(self, audio_prompt: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        prompt = audio_prompt.strip() or "neutral ambient background sound"
        descriptors = self._audio_descriptors(prompt)
        values = np.array(
            [
                0.0,
                0.03 if "ambient" in descriptors else 0.06,
                0.18 if "rhythmic" in descriptors else 0.08,
                0.25 if "soft" in descriptors else 0.4,
                140.0 if "energetic" in descriptors else 90.0 if "rhythmic" in descriptors else 70.0,
            ],
            dtype=np.float32,
        )
        embedding = self._numeric_embedding(values, namespace=f"audio_prompt:{prompt}")
        return embedding, {
            "source": "prompt",
            "duration_sec": 0.0,
            "rms": round(float(values[1]), 5),
            "peak": round(float(values[3]), 5),
            "zero_crossing_rate": round(float(values[2]), 5),
            "estimated_bpm": round(float(values[4]), 2),
            "audio_prompt": prompt,
            "descriptors": descriptors,
        }

    def reference(self, meta: Dict[str, object]) -> torch.Tensor:
        text = " ".join(
            [
                str(meta.get("caption", "")),
                " ".join(meta.get("visual_tags", [])),
                " ".join(meta.get("motion_tags", [])),
                " ".join(meta.get("audio_tags", [])),
            ]
        )
        return self._hash_embedding(f"reference:{text}")

    def _hash_embedding(self, text: str) -> torch.Tensor:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.dim, generator=gen)
        return torch.nn.functional.normalize(emb, dim=-1)

    def _numeric_embedding(self, values: np.ndarray, namespace: str) -> torch.Tensor:
        base = self._hash_embedding(namespace)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return base
        values = (values - values.mean()) / (values.std() + 1e-6)
        signal = torch.zeros(1, self.dim)
        repeat = math.ceil(self.dim / values.size)
        signal[0] = torch.from_numpy(np.tile(values, repeat)[: self.dim]).float()
        return torch.nn.functional.normalize(base + 0.08 * signal, dim=-1)

    def _audio_stats(self, audio_path: str) -> Dict[str, float]:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            duration = wf.getnframes() / float(sample_rate)
        if sample_width != 2:
            raise ValueError("Demo audio reader expects 16-bit PCM wav")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        audio = audio / 32768.0
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio))))) if audio.size > 1 else 0.0
        envelope = np.abs(audio)
        threshold = max(0.05, envelope.mean() + envelope.std())
        peaks = np.where(
            (envelope[1:-1] > envelope[:-2])
            & (envelope[1:-1] > envelope[2:])
            & (envelope[1:-1] > threshold)
        )[0]
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / float(sample_rate)
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            bpm = float(60.0 / np.median(intervals)) if len(intervals) else 0.0
        else:
            bpm = 0.0
        return {
            "duration_sec": round(float(duration), 3),
            "rms": round(rms, 5),
            "peak": round(peak, 5),
            "zero_crossing_rate": round(zcr, 5),
            "estimated_bpm": round(bpm, 2),
        }

    def _audio_descriptors(self, prompt: str) -> List[str]:
        lower = prompt.lower()
        descriptors = []
        if any(k in lower for k in ["rhythm", "rhythmic", "beat", "drum", "??", "??"]):
            descriptors.append("rhythmic")
        if any(k in lower for k in ["cinematic", "soundtrack", "??", "???"]):
            descriptors.append("cinematic")
        if any(k in lower for k in ["ambient", "???", "ambiently"]):
            descriptors.append("ambient")
        if any(k in lower for k in ["fast", "upbeat", "energetic", "??", "??"]):
            descriptors.append("energetic")
        if any(k in lower for k in ["soft", "quiet", "slow", "??", "??"]):
            descriptors.append("soft")
        return descriptors or ["ambient"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MUGen-VR showcase")
    parser.add_argument("--prompt", default="a dog running on grass with upbeat rhythmic background music")
    parser.add_argument("--image", default="third_party/ImageBind/.assets/dog_image.jpg")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output_dir", default="outputs/showcase/demo")
    parser.add_argument("--mode", choices=["generate", "report", "smoke"], default="generate")
    parser.add_argument("--model_id", default="models/anyflow-local")
    parser.add_argument("--fusion_checkpoint", default=None)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--chunk_partition", default="1,2,2,2,2,2,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def ensure_demo_audio(path: Path, seconds: float = 3.0, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    carrier = 0.18 * np.sin(2 * np.pi * 220 * t)
    beat = (np.sin(2 * np.pi * 2.0 * t) > 0.92).astype(np.float32)
    envelope = 0.35 + 0.65 * beat
    audio = carrier * envelope
    audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    return path


def summarize_image(image_path: str) -> Dict[str, object]:
    img = Image.open(image_path).convert("RGB")
    stat = ImageStat.Stat(img)
    return {
        "path": image_path,
        "size": list(img.size),
        "mean_rgb": [round(x, 2) for x in stat.mean],
        "std_rgb": [round(x, 2) for x in stat.stddev],
    }


def load_fusion_checkpoint(module: HierarchicalConditionFusion, checkpoint_path: Optional[str]) -> Dict[str, object]:
    if not checkpoint_path:
        return {"loaded": False, "path": None}
    path = Path(checkpoint_path)
    if not path.exists():
        return {"loaded": False, "path": str(path), "reason": "missing"}
    payload = torch.load(path, map_location="cpu")
    state = payload.get("fusion") or payload.get("state_dict") or payload
    module.load_state_dict(state, strict=False)
    return {"loaded": True, "path": str(path)}


def build_reference_retriever(factory: DemoEmbeddingFactory) -> CrossModalRetriever:
    retriever = CrossModalRetriever(dim=factory.dim, top_k=3)
    embeddings = torch.cat([factory.reference(meta) for meta in DEMO_REFERENCES], dim=0)
    retriever.build_index(embeddings, DEMO_REFERENCES)
    return retriever


def layer_weight_summary(layer_weights: torch.Tensor, modalities: List[str]) -> Dict[str, Dict[str, float]]:
    weights = layer_weights.detach().cpu()
    if weights.dim() == 3:
        weights = weights[:, 0, :]
    names = ["low_level_visual", "mid_level_motion_audio", "high_level_semantic"]
    summary = {}
    for idx in range(weights.shape[0]):
        name = names[idx] if idx < len(names) else f"layer_{idx}"
        summary[name] = {mod: round(float(weights[idx, j]), 4) for j, mod in enumerate(modalities)}
    return summary


def make_prompt_plan(content_prompt: str, audio_prompt: str, retrieval_meta: List[Dict[str, object]], audio_info: Dict[str, float]) -> Dict[str, object]:
    visual_tags, motion_tags, audio_tags = [], [], []
    for meta in retrieval_meta:
        visual_tags.extend(meta.get("visual_tags", []))
        motion_tags.extend(meta.get("motion_tags", []))
        audio_tags.extend(meta.get("audio_tags", []))

    def uniq(items):
        out = []
        for item in items:
            if item not in out:
                out.append(item)
        return out

    visual_tags = uniq(visual_tags)[:5]
    motion_tags = uniq(motion_tags)[:4]
    audio_tags = uniq(audio_tags)[:3]
    motion_phrase = audio_info.get("motion_phrase") or _audio_motion_phrase(audio_info.get("descriptors", []))
    mood_phrase = audio_info.get("mood_phrase") or _audio_mood_phrase(audio_info.get("descriptors", []))
    enhanced_prompt = ", ".join(
        [
            content_prompt,
            f"audio direction: {audio_prompt}",
            f"reference-guided composition: {', '.join(visual_tags)}",
            f"motion style: {', '.join(motion_tags + [motion_phrase])}",
            f"mood/style: {mood_phrase}",
            "high temporal consistency, realistic motion, cinematic natural lighting",
        ]
    )
    return {
        "content_prompt": content_prompt,
        "audio_prompt": audio_prompt,
        "enhanced_prompt": enhanced_prompt,
        "retrieval_visual_tags": visual_tags,
        "retrieval_motion_tags": motion_tags,
        "audio_style_tags": audio_tags + [motion_phrase, mood_phrase],
        "note": "Audio affects the MUGen condition layer and prompt planning. AnyFlow still consumes image + enhanced prompt, not direct audio latent conditioning.",
    }


def _audio_motion_phrase(descriptors: List[str]) -> str:
    descriptors = descriptors or []
    if "energetic" in descriptors and "rhythmic" in descriptors:
        return "fast rhythmic motion"
    if "rhythmic" in descriptors:
        return "steady rhythmic motion"
    if "soft" in descriptors:
        return "soft slow pacing"
    if "ambient" in descriptors:
        return "smooth ambient pacing"
    return "neutral natural motion"


def _audio_mood_phrase(descriptors: List[str]) -> str:
    descriptors = descriptors or []
    if "cinematic" in descriptors and "ambient" in descriptors:
        return "cinematic ambient soundtrack mood"
    if "cinematic" in descriptors:
        return "cinematic soundtrack mood"
    if "energetic" in descriptors:
        return "energetic music mood"
    if "soft" in descriptors:
        return "soft quiet sound mood"
    return "neutral ambient sound"


def save_json(path: Path, data: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, payload: Dict[str, object]):
    lines = [
        "# MUGen-VR Showcase Report",
        "",
        "## Inputs",
        f"- Prompt: `{payload['inputs']['prompt']}`",
        f"- Content prompt: `{payload['inputs']['content_prompt']}`",
        f"- Audio prompt: `{payload['inputs']['audio_prompt']}`",
        f"- Image: `{payload['inputs']['image']['path']}` size={payload['inputs']['image']['size']}",
        f"- Audio source: `{payload['inputs']['audio_source']}` stats={payload['inputs']['audio_stats']}",
        f"- Reference source: `{payload['inputs']['reference_source']}`",
        "",
        "## Retrieved References",
    ]
    for item in payload["retrieval"]["top_results"]:
        lines.append(f"- {item['rank']}. `{item['clip_id']}` score={item['score']} caption={item['caption']}")
    lines.extend([
        "",
        "## Fusion Gating Weights",
        "",
        "```json",
        json.dumps(payload["fusion"]["gating_weights"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Generation Plan",
        f"- Enhanced prompt: `{payload['generation_plan']['enhanced_prompt']}`",
        f"- Output video: `{payload['generation'].get('video_path')}`",
        f"- Backend: `{payload['generation']['backend']}`",
        "",
        "## Takeaway",
        "MUGen-VR keeps the backbone frozen and focuses on the condition layer: audio-aware prompt parsing, hierarchical multimodal fusion, reference retrieval, and a reproducible video generation plan.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_smoke_mp4(path: Path, num_frames: int, height: int, width: int):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = np.linspace(30, 180, width, dtype=np.uint8)[None, :]
        x = int((i / max(1, num_frames - 1)) * (width - 80))
        cv2.circle(frame, (x + 40, height // 2), 28, (40, 130, 230), -1)
        cv2.putText(frame, "MUGen showcase smoke", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def save_generation_mp4(frames: torch.Tensor, path: Path, fps: int = 8):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (frames.detach().cpu().clamp(0, 1) * 255).byte().numpy()
    h, w = frames.shape[2], frames.shape[3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        rgb = np.transpose(frame, (1, 2, 0))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = AudioPromptParser()
    parsed = parser.parse(args.prompt)
    content_prompt = parsed.content_prompt
    audio_prompt = parsed.audio_prompt

    audio_source = args.audio
    factory = DemoEmbeddingFactory(dim=768)
    if audio_source:
        audio_emb, audio_stats = factory.audio_from_file(audio_source)
    else:
        audio_emb, audio_stats = factory.audio_from_prompt(audio_prompt)
        audio_source = None

    text_emb = factory.text(content_prompt)
    image_emb = factory.image(args.image)

    fusion_dims = {"text": 768, "image": 768, "audio": 512, "reference": 768}
    fusion = HierarchicalConditionFusion(dims=fusion_dims, hidden_dim=768, num_heads=8, modality_dropout=0.0)
    fusion_info = load_fusion_checkpoint(fusion, args.fusion_checkpoint)
    fusion.eval()
    audio_emb = audio_emb[:, :512]
    retrieval_audio = F.pad(audio_emb, (0, 256))
    with torch.no_grad():
        retriever = build_reference_retriever(factory)
        retrieval_seed = retriever.multi_modal_retrieve({"text": text_emb, "image": image_emb, "audio": retrieval_audio}, top_k=args.top_k)
        ref_indices = retrieval_seed.retrieved_indices[0]
        ref_scores = retrieval_seed.retrieved_scores[0]
        ref_embs = retriever.get_embeddings(ref_indices)
        if ref_embs is None:
            reference_emb = fused if "fused" in locals() else torch.zeros_like(text_emb)
        else:
            scores_t = torch.tensor(ref_scores, dtype=ref_embs.dtype, device=ref_embs.device)
            weights_ref = torch.softmax(scores_t, dim=-1).unsqueeze(-1)
            reference_emb = F.normalize((ref_embs * weights_ref).sum(dim=0, keepdim=True), dim=-1)
        fused, weights = fusion({"text": (text_emb, None), "image": (image_emb, None), "audio": (audio_emb, None), "reference": (reference_emb, None)}, return_weights=True)

    result = retriever.multi_modal_retrieve({"text": text_emb, "image": image_emb, "audio": retrieval_audio, "reference": fused}, top_k=args.top_k)
    top_meta = result.retrieved_metadata[0]
    top_scores = result.retrieved_scores[0]
    top_results = []
    for rank, (score, meta) in enumerate(zip(top_scores, top_meta), start=1):
        clean = {k: v for k, v in meta.items() if k != "embedding"}
        clean.update({"rank": rank, "score": round(float(score), 4)})
        top_results.append(clean)

    prompt_plan = make_prompt_plan(content_prompt, audio_prompt, top_meta, {**audio_stats, "motion_phrase": parsed.motion_phrase, "mood_phrase": parsed.mood_phrase, "descriptors": parsed.descriptors})

    video_path = None
    backend = args.mode
    if args.mode == "report":
        backend = "report"
    elif args.mode == "smoke":
        video_path = out_dir / "result.mp4"
        save_smoke_mp4(video_path, args.num_frames, min(args.height, 256), min(args.width, 448))
    else:
        image = Image.open(args.image).convert("RGB")
        generator = AnyFlowVideoGenerator(
            model_id=args.model_id,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        )
        chunk_partition = [int(x) for x in args.chunk_partition.split(",") if x.strip()]
        gen_result = generator.generate(
            {"image": image, "prompt": prompt_plan["enhanced_prompt"]},
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            chunk_partition=chunk_partition,
        )
        video_path = out_dir / "result.mp4"
        save_generation_mp4(gen_result.video_frames, video_path)

    payload = {
        "inputs": {
            "prompt": args.prompt,
            "content_prompt": content_prompt,
            "audio_prompt": audio_prompt,
            "image": summarize_image(args.image),
            "audio_source": "file" if args.audio else "prompt-derived",
            "audio_stats": audio_stats,
            "reference_source": "retrieved reference clips",
            "fusion_checkpoint": fusion_info,
        },
        "fusion": {
            "module": "HierarchicalConditionFusion",
            "modalities": ["text", "image", "audio", "reference"],
            "fused_embedding_shape": list(fused.shape),
            "gating_weights": layer_weight_summary(weights, ["text", "image", "audio"]),
        },
        "retrieval": {
            "module": "CrossModalRetriever",
            "top_k": args.top_k,
            "top_results": top_results,
        },
        "generation_plan": prompt_plan,
        "generation": {
            "mode": args.mode,
            "backend": backend,
            "video_path": str(video_path) if video_path else None,
            "frames": args.num_frames if video_path else 0,
            "resolution": [args.height, args.width] if video_path else None,
        },
    }

    save_json(out_dir / "report.json", payload)
    save_json(out_dir / "gating_weights.json", payload["fusion"]["gating_weights"])
    save_json(out_dir / "retrieval_results.json", payload["retrieval"])
    write_report(out_dir / "report.md", payload)
    print(f"report={out_dir / 'report.md'}")
    print(f"json={out_dir / 'report.json'}")
    if video_path:
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
