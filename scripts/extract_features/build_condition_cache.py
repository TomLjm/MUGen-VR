#!/usr/bin/env python3
"""Build a lightweight condition-fusion feature cache from MSR-VTT metadata.

This is a metadata-first training cache. It uses real video-caption metadata from
MSR-VTT and deterministic feature projections so the fusion/reference modules can
be trained before full backbone feature extraction is available. The cache schema
matches scripts/train/train_fusion.py and can later be replaced by ImageBind /
InternVideo / CLAP features without changing the trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

AUDIO_WORDS = {
    "music", "sing", "song", "dance", "talk", "speaks", "speaking", "sound", "laugh", "playing",
    "instrument", "guitar", "piano", "drum", "crowd", "conversation", "voice",
}
MOTION_WORDS = {
    "run", "running", "walk", "walking", "ride", "riding", "jump", "jumping", "dance", "dancing",
    "move", "moving", "play", "playing", "drive", "driving", "cut", "stir", "cook", "surfing",
}
VISUAL_WORDS = {
    "man", "woman", "girl", "boy", "dog", "cat", "car", "bike", "kitchen", "water", "grass", "door",
    "mountain", "sky", "food", "person", "people", "animal", "room", "outside",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build MUGen condition-fusion feature cache")
    parser.add_argument("--metadata", default="cache/hf_probe/data/train-00000-of-00001-60e50ff5fbbd1bb5.parquet")
    parser.add_argument("--output", default="cache/features/msrvtt_condition_cache.pt")
    parser.add_argument("--manifest", default="cache/features/msrvtt_condition_cache.json")
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--audio_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_embedding(text: str, dim: int) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    gen = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=gen), dim=0)


def numeric_embedding(values: Iterable[float], dim: int, namespace: str) -> torch.Tensor:
    values = np.asarray(list(values), dtype=np.float32)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)
    values = (values - values.mean()) / (values.std() + 1e-6)
    repeat = math.ceil(dim / values.size)
    signal = torch.from_numpy(np.tile(values, repeat)[:dim]).float()
    return F.normalize(stable_embedding(namespace, dim) + 0.10 * signal, dim=0)


def lexical_counts(caption: str) -> Dict[str, int]:
    words = {w.strip(".,!?;:'\"()[]").lower() for w in caption.split()}
    return {
        "visual": sum(w in VISUAL_WORDS for w in words),
        "motion": sum(w in MOTION_WORDS for w in words),
        "audio": sum(w in AUDIO_WORDS for w in words),
        "length": len(words),
    }


def build_row(row, dim: int, audio_dim: int):
    caption = str(row["caption"])
    video_id = str(row.get("video_id", row.get("id", "unknown")))
    category = int(row.get("category", 0) or 0)
    start = float(row.get("start time", 0.0) or 0.0)
    end = float(row.get("end time", start + 8.0) or start + 8.0)
    duration = max(0.1, end - start)
    counts = lexical_counts(caption)

    semantic = stable_embedding(f"caption:{caption}", dim)
    video = stable_embedding(f"video:{video_id}:cat:{category}", dim)
    category_emb = stable_embedding(f"category:{category}", dim)
    motion = numeric_embedding([duration, counts["motion"], counts["length"], category], dim, f"motion:{video_id}")
    visual = numeric_embedding([counts["visual"], category, start, duration], dim, f"visual:{video_id}:{caption}")
    audio = numeric_embedding([counts["audio"], counts["motion"], duration, category], audio_dim, f"audio:{caption}")

    target = F.normalize(0.58 * semantic + 0.22 * video + 0.12 * category_emb + 0.08 * motion, dim=0)
    text = F.normalize(semantic + 0.04 * stable_embedding(f"text-noise:{video_id}", dim), dim=0)
    image = F.normalize(0.68 * visual + 0.22 * category_emb + 0.10 * semantic, dim=0)
    reference = F.normalize(0.55 * video + 0.30 * target + 0.15 * motion, dim=0)

    return text, image, audio, reference, target, {
        "video_id": video_id,
        "caption": caption,
        "category": category,
        "start": start,
        "end": end,
        "duration": duration,
        "lexical_counts": counts,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata parquet not found: {metadata_path}")

    df = pd.read_parquet(metadata_path).head(args.max_samples)
    tensors = {"text": [], "image": [], "audio": [], "reference": [], "target": []}
    manifest: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        text, image, audio, reference, target, meta = build_row(row, args.dim, args.audio_dim)
        tensors["text"].append(text)
        tensors["image"].append(image)
        tensors["audio"].append(audio)
        tensors["reference"].append(reference)
        tensors["target"].append(target)
        manifest.append(meta)

    cache = {k: torch.stack(v, dim=0) for k, v in tensors.items()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"source": str(metadata_path), "num_samples": len(manifest), "items": manifest[:50]}, indent=2), encoding="utf-8")
    print(f"saved={out}")
    print({k: tuple(v.shape) for k, v in cache.items()})
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
