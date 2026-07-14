#!/usr/bin/env python3
"""Retrieval demo using a saved index or a deterministic demo index."""
import argparse
import os

import torch

from mugen.retrieval.retriever import CrossModalRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="MUGen retrieval demo")
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--query", type=str, default="a dog running on grass")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def text_to_demo_embedding(text, dim):
    generator = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
    emb = torch.randn(1, dim, generator=generator)
    return torch.nn.functional.normalize(emb, dim=-1)


def build_demo_index(dim):
    retriever = CrossModalRetriever(dim=dim)
    features = torch.nn.functional.normalize(torch.randn(8, dim, generator=torch.Generator().manual_seed(7)), dim=-1)
    metadata = [{"clip_id": f"demo_{i}", "caption": f"demo reference clip {i}"} for i in range(8)]
    retriever.build_index(features, metadata)
    return retriever


def main():
    args = parse_args()
    retriever = CrossModalRetriever(dim=args.dim)
    if args.index and os.path.exists(args.index):
        retriever.load_index(args.index)
    else:
        retriever = build_demo_index(args.dim)
    query = text_to_demo_embedding(args.query, args.dim)
    result = retriever.retrieve(query, top_k=args.top_k)
    for idx, score, meta in zip(result.retrieved_indices[0], result.retrieved_scores[0], result.retrieved_metadata[0]):
        clean_meta = {k: v for k, v in meta.items() if k != "embedding"}
        print(f"idx={idx} score={score:.4f} meta={clean_meta}")


if __name__ == "__main__":
    main()
