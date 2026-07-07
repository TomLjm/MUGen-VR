"""Retrieval reranking module."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reranker(nn.Module):
    """Lightweight reranker for refining retrieval results."""

    def __init__(self, dim=768, hidden_dim=256):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_emb, candidate_embs):
        """
        Args:
            query_emb: [B, D]
            candidate_embs: [B, K, D]
        Returns:
            scores: [B, K]
        """
        B, K, D = candidate_embs.shape
        query_expanded = query_emb.unsqueeze(1).expand(-1, K, -1)
        pairs = torch.cat([query_expanded, candidate_embs], dim=-1)
        scores = self.score_net(pairs).squeeze(-1)
        return scores

    def rerank(self, query_emb, candidate_embs, candidate_metadata, top_k=5):
        scores = self.forward(query_emb.unsqueeze(0), candidate_embs.unsqueeze(0))
        sorted_idx = scores[0].argsort(descending=True)
        reranked_idx = sorted_idx[:top_k].tolist()
        
        return {
            "indices": reranked_idx,
            "scores": scores[0][reranked_idx].tolist(),
            "metadata": [candidate_metadata[i] for i in reranked_idx],
        }
