"""Training objectives for real multimodal retrieval features."""

import torch
import torch.nn.functional as F


def symmetric_info_nce(query: torch.Tensor, target: torch.Tensor, temperature: float = 0.07):
    query = F.normalize(query.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = query @ target.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, logits


def gate_balance_loss(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Maximize gate entropy so no modality collapses before evidence supports it."""
    probabilities = weights.float().clamp_min(eps)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return -entropy.mean()
