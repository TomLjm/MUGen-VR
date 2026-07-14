#!/usr/bin/env python3
"""Train the project-owned fusion/reference adapter modules on cached features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from mugen.fusion.fusion_module import HierarchicalConditionFusion
from mugen.generation.retrieve_then_generate import ReferenceAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train MUGen fusion modules")
    parser.add_argument("--config", type=str, default="configs/training/fusion.yaml")
    parser.add_argument("--feature_path", type=str, default="cache/features/msrvtt_condition_cache.pt")
    parser.add_argument("--output", type=str, default="checkpoints/mugen_condition_fusion.pt")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--audio_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_cache(feature_path: str):
    if not feature_path:
        return None
    path = Path(feature_path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def batch_from_cache(cache, args, device, step):
    n = cache["target"].shape[0]
    if n == 0:
        raise ValueError("empty feature cache")
    idx = torch.randint(0, n, (args.batch_size,))
    batch = {k: v[idx].to(device).float() for k, v in cache.items()}
    return batch


def synthetic_batch(args, device):
    base = F.normalize(torch.randn(args.batch_size, args.dim, device=device), dim=-1)
    audio_base = base[:, : args.audio_dim]
    return {
        "text": F.normalize(base + 0.02 * torch.randn_like(base), dim=-1),
        "image": F.normalize(base + 0.02 * torch.randn_like(base), dim=-1),
        "audio": F.normalize(audio_base + 0.02 * torch.randn(args.batch_size, args.audio_dim, device=device), dim=-1),
        "reference": F.normalize(base + 0.02 * torch.randn_like(base), dim=-1),
        "target": F.normalize(base, dim=-1),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cache = load_cache(args.feature_path)
    use_cache = cache is not None

    fusion = HierarchicalConditionFusion(
        dims={"text": args.dim, "image": args.dim, "audio": args.audio_dim, "reference": args.dim},
        hidden_dim=args.dim,
    ).to(device)
    ref_adapter = ReferenceAdapter(args.dim).to(device)
    opt = torch.optim.AdamW(list(fusion.parameters()) + list(ref_adapter.parameters()), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.max_steps))

    last = {}
    fusion.train()
    ref_adapter.train()
    for step in range(args.max_steps):
        if use_cache:
            batch = batch_from_cache(cache, args, device, step)
        else:
            batch = synthetic_batch(args, device)

        modality_inputs = {
            "text": (batch["text"], None),
            "image": (batch["image"], None),
            "audio": (batch["audio"], None),
            "reference": (ref_adapter(batch["reference"].unsqueeze(1)), None),
        }
        out = F.normalize(fusion(modality_inputs), dim=-1)
        target = F.normalize(batch["target"], dim=-1)
        logits = out @ target.T / 0.07
        labels = torch.arange(logits.shape[0], device=device)
        contrastive = F.cross_entropy(logits, labels)
        alignment = 1 - F.cosine_similarity(out, target, dim=-1).mean()
        loss = contrastive + 0.2 * alignment
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(ref_adapter.parameters()), 1.0)
        opt.step()
        scheduler.step()
        last = {
            "loss": float(loss.item()),
            "contrastive": float(contrastive.item()),
            "alignment": float(alignment.item()),
            "mode": "cache" if use_cache else "synthetic",
            "device": str(device),
        }
        if step % 50 == 0 or step == args.max_steps - 1:
            print(
                f"step={step} mode={last['mode']} loss={last['loss']:.4f} "
                f"contrastive={last['contrastive']:.4f} alignment={last['alignment']:.4f}"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"fusion": fusion.state_dict(), "reference_adapter": ref_adapter.state_dict(), "metrics": last}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
