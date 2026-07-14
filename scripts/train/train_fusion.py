#!/usr/bin/env python3
"""Train MUGen fusion and reference tokens exclusively on real cached features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from mugen.data.feature_store import load_feature_store
from mugen.fusion.fusion_module import HierarchicalConditionFusion
from mugen.generation.retrieve_then_generate import ReferenceAdapter
from mugen.training.losses import gate_balance_loss, symmetric_info_nce


def parse_args():
    parser = argparse.ArgumentParser(description="Train MUGen real-feature fusion")
    parser.add_argument("--config", default="configs/training/fusion.yaml")
    parser.add_argument("--feature-store")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def split_indices(records, split):
    return torch.tensor([index for index, row in enumerate(records) if row["split"] == split], dtype=torch.long)


def batches(indices, batch_size, generator):
    order = indices[torch.randperm(len(indices), generator=generator)]
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        if len(batch) > 1:
            yield batch


def retrieve_references(query, gallery, top_k, excluded=None):
    scores = F.normalize(query.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    if excluded is not None:
        scores[torch.arange(scores.shape[0], device=scores.device), excluded] = -torch.inf
    values, indices = scores.topk(min(top_k, gallery.shape[0]), dim=-1)
    return gallery[indices], values, indices


def fuse_twice(fusion, adapter, inputs, gallery, top_k, excluded=None):
    initial = fusion(inputs)
    references, scores, reference_indices = retrieve_references(initial, gallery, top_k, excluded)
    reference_tokens = adapter(references, scores)
    final_inputs = {**inputs, "reference": (reference_tokens.mean(dim=1), None)}
    fused, weights = fusion(final_inputs, return_weights=True)
    return fused, weights, reference_tokens, reference_indices


@torch.no_grad()
def validate(fusion, adapter, tensors, indices, gallery, batch_size, top_k, device):
    fusion.eval()
    adapter.eval()
    queries = []
    targets = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        inputs = {
            name: (tensors[name][batch].to(device), None)
            for name in ["text", "image", "audio"]
        }
        fused, _, _, _ = fuse_twice(fusion, adapter, inputs, gallery, top_k)
        queries.append(F.normalize(fused, dim=-1).cpu())
        targets.append(F.normalize(tensors["video"][batch], dim=-1).cpu())
    query = torch.cat(queries)
    target = torch.cat(targets)
    scores = query @ target.T
    ranks = scores.argsort(dim=-1, descending=True)
    labels = torch.arange(len(query)).unsqueeze(1)
    positions = (ranks == labels).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "mrr": float((1.0 / positions.float()).mean()),
        "recall@1": float((positions <= 1).float().mean()),
        "recall@5": float((positions <= 5).float().mean()),
        "recall@10": float((positions <= 10).float().mean()),
    }


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    feature_store = args.feature_store or config.data.feature_store
    output_dir = Path(args.output_dir or config.training.output_dir)
    seed = args.seed if args.seed is not None else int(config.training.seed)
    max_steps = args.max_steps if args.max_steps is not None else int(config.training.max_steps)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tensors, records, manifest = load_feature_store(feature_store)
    required = {"text", "image", "audio", "video"}
    if missing := required - set(tensors):
        raise ValueError(f"feature store is missing tensors: {sorted(missing)}")
    train_indices = split_indices(records, "train")
    val_indices = split_indices(records, "val")
    if len(train_indices) < 2 or len(val_indices) < 2:
        raise ValueError("real feature store requires train and val splits")
    dims = {name: int(tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
    dims["reference"] = int(tensors["video"].shape[-1])
    hidden_dim = int(config.model.hidden_dim)
    if dims["reference"] != hidden_dim:
        raise ValueError("InternVideo dimension must match fusion hidden_dim")
    fusion = HierarchicalConditionFusion(
        dims=dims,
        hidden_dim=hidden_dim,
        num_heads=int(config.model.num_heads),
        modality_dropout=float(config.model.modality_dropout),
    ).to(device)
    adapter = ReferenceAdapter(hidden_dim, int(config.model.reference_tokens)).to(device)
    parameters = list(fusion.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    gallery = tensors["video"][train_indices].to(device)
    train_position = {int(global_index): position for position, global_index in enumerate(train_indices.tolist())}
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train-seed-{seed}.jsonl"
    generator = torch.Generator().manual_seed(seed)
    best_mrr = -1.0
    step = 0
    with log_path.open("w", encoding="utf-8") as log:
        while step < max_steps:
            for batch in batches(train_indices, int(config.training.batch_size), generator):
                batch = batch.to(device)
                fusion.train()
                adapter.train()
                inputs = {name: (tensors[name][batch.cpu()].to(device), None) for name in ["text", "image", "audio"]}
                excluded = torch.tensor([train_position[int(index)] for index in batch.cpu()], device=device)
                fused_a, weights, _, _ = fuse_twice(
                    fusion, adapter, inputs, gallery, int(config.retrieval.top_k), excluded
                )
                fused_b, _, _, _ = fuse_twice(
                    fusion, adapter, inputs, gallery, int(config.retrieval.top_k), excluded
                )
                target = tensors["video"][batch.cpu()].to(device)
                contrastive, _ = symmetric_info_nce(
                    fused_a, target, float(config.loss.temperature)
                )
                consistency = 1 - F.cosine_similarity(fused_a, fused_b, dim=-1).mean()
                gate = gate_balance_loss(weights)
                loss = (
                    contrastive
                    + float(config.loss.consistency_weight) * consistency
                    + float(config.loss.gate_weight) * gate
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, float(config.training.grad_clip))
                optimizer.step()
                scheduler.step()
                step += 1
                event = {
                    "step": step,
                    "loss": float(loss),
                    "contrastive": float(contrastive),
                    "consistency": float(consistency),
                    "gate_balance": float(gate),
                }
                if step % int(config.training.eval_interval) == 0 or step == max_steps:
                    metrics = validate(
                        fusion, adapter, tensors, val_indices, gallery,
                        int(config.training.eval_batch_size), int(config.retrieval.top_k), device,
                    )
                    event["validation"] = metrics
                    checkpoint = {
                        "fusion": fusion.state_dict(),
                        "reference_adapter": adapter.state_dict(),
                        "config": OmegaConf.to_container(config, resolve=True),
                        "feature_manifest": manifest,
                        "seed": seed,
                        "step": step,
                        "metrics": metrics,
                    }
                    torch.save(checkpoint, output_dir / f"latest-seed-{seed}.pt")
                    if metrics["mrr"] > best_mrr:
                        best_mrr = metrics["mrr"]
                        torch.save(checkpoint, output_dir / f"best-seed-{seed}.pt")
                log.write(json.dumps(event) + "\n")
                log.flush()
                if step >= max_steps:
                    break
    print({"best_mrr": best_mrr, "output_dir": str(output_dir), "seed": seed})


if __name__ == "__main__":
    main()
