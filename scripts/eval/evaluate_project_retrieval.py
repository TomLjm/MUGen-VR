#!/usr/bin/env python3
"""Evaluate B2/B3 retrieval on the fixed held-out project set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mugen.data.feature_store import load_feature_store
from mugen.data.manifest import read_jsonl
from mugen.evaluation.custom_metrics import CustomMetrics
from mugen.generation.conditioner import MultimodalConditioner


def retrieve_train_references(query, train_gallery, top_k):
    scores = F.normalize(query.float(), dim=-1) @ F.normalize(train_gallery.float(), dim=-1).T
    values, indices = scores.topk(min(top_k, train_gallery.shape[0]), dim=-1)
    return train_gallery[indices], values


def reciprocal_ranks(queries, targets):
    scores = F.normalize(queries.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T
    ranks = scores.argsort(dim=-1, descending=True)
    labels = torch.arange(len(queries), device=scores.device)
    positions = (ranks == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    return positions


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed B2/B3 retrieval")
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--conditioner-checkpoint", required=True)
    parser.add_argument("--output", default="reports/project/retrieval.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors, records, feature_manifest = load_feature_store(args.feature_store)
    by_id = {row["sample_id"]: index for index, row in enumerate(records)}
    eval_rows = read_jsonl(args.eval_manifest)
    eval_indices = [by_id[row["sample_id"]] for row in eval_rows]
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    train_gallery = tensors["video"][train_indices].to(device)
    target_gallery = tensors["video"][eval_indices].to(device)
    dims = {name: int(tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
    dims["reference"] = int(tensors["video"].shape[-1])
    conditioner = MultimodalConditioner(dims=dims, hidden_dim=dims["reference"])
    conditioner.load_state_dict(
        torch.load(args.conditioner_checkpoint, map_location="cpu", weights_only=True)
    )
    conditioner = conditioner.to(device).eval()
    b2_queries, b3_queries = [], []
    with torch.no_grad():
        for index in eval_indices:
            text = tensors["text"][index].unsqueeze(0).to(device)
            image = tensors["image"][index].unsqueeze(0).to(device)
            audio = tensors["audio"][index].unsqueeze(0).to(device)
            b2 = conditioner.fusion({"text": (text, None), "image": (image, None)})
            initial = conditioner.fusion(
                {"text": (text, None), "image": (image, None), "audio": (audio, None)}
            )
            references, scores = retrieve_train_references(initial, train_gallery, args.top_k)
            reference_tokens = conditioner.reference_adapter(references, scores)
            b3 = conditioner.fusion(
                {
                    "text": (text, None),
                    "image": (image, None),
                    "audio": (audio, None),
                    "reference": (reference_tokens.mean(dim=1), None),
                }
            )
            b2_queries.append(b2)
            b3_queries.append(b3)
    metrics = {}
    per_sample = []
    labels = torch.arange(len(eval_rows), device=device)
    for variant, queries in {"B2": torch.cat(b2_queries), "B3": torch.cat(b3_queries)}.items():
        variant_metrics = CustomMetrics().compute_retrieval_metrics(
            queries, target_gallery, labels
        )
        metrics[variant] = variant_metrics
        ranks = reciprocal_ranks(queries, target_gallery)
        per_sample.extend(
            {
                "pair_id": row["pair_id"],
                "variant": variant,
                "retrieval_rank": int(rank),
                "retrieval_reciprocal_rank": float(1.0 / rank),
            }
            for row, rank in zip(eval_rows, ranks)
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"metrics": metrics, "per_sample": per_sample, "feature_manifest": feature_manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "metrics": metrics}))


if __name__ == "__main__":
    main()
