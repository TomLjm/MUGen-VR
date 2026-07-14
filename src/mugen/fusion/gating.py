"""Learnable modality gating mechanism."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGating(nn.Module):
    """Learnable gating mechanism for dynamic modality contribution."""

    def __init__(self, hidden_dim=768, num_modalities=3):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * num_modalities, num_modalities), nn.Softmax(dim=-1))
        self.num_modalities = num_modalities

    def forward(self, modality_embeddings):
        concat = torch.cat(modality_embeddings, dim=-1)
        weights = self.gate(concat)
        stacked = torch.stack(modality_embeddings, dim=1)
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return weighted, weights


class HierarchicalGating(nn.Module):
    """Multi-layer gating with modality-priority biases."""

    def __init__(self, hidden_dim=768, num_modalities=3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.num_modalities = num_modalities
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim * num_modalities, num_modalities)
            for _ in range(num_layers)
        ])
        biases = torch.full((num_layers, num_modalities), 0.5)
        if num_modalities >= 1:
            biases[0, 0] = 2.0  # low-level priority: first configured modality
        if num_modalities >= 2:
            biases[2, 0] = 2.0  # high-level semantic default for text-first configs
            biases[0, 1] = 1.5  # low-level support for image when present
        if num_modalities >= 3:
            biases[1, 2] = 2.0  # mid-level audio/rhythm when present
        self.layer_biases = nn.Parameter(biases)

    def forward(self, modality_embeddings):
        concat = torch.cat(modality_embeddings, dim=-1)
        stacked = torch.stack(modality_embeddings, dim=1)
        layer_outputs = []
        layer_weights = []
        for i in range(self.num_layers):
            logits = self.gates[i](concat) + self.layer_biases[i].unsqueeze(0)
            weights = F.softmax(logits, dim=-1)
            weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            layer_outputs.append(weighted)
            layer_weights.append(weights)
        return torch.stack(layer_outputs), torch.stack(layer_weights)
