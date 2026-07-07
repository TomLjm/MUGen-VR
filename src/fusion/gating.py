"""Learnable modality gating mechanism."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGating(nn.Module):
    """Learnable gating mechanism for dynamic modality contribution.
    
    Innovation A: hierarchical gating where different layers
    use different modality priorities (low-level: image,
    mid-level: audio, high-level: text).
    """

    def __init__(self, hidden_dim=768, num_modalities=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1),
        )
        self.num_modalities = num_modalities

    def forward(self, modality_embeddings):
        """
        Args:
            modality_embeddings: [B, N, D] list of modality embeddings
        Returns:
            weighted_sum: [B, D]
            weights: [B, N] gating weights
        """
        concat = torch.cat(modality_embeddings, dim=-1)  # [B, N*D]
        weights = self.gate(concat)  # [B, N]
        stacked = torch.stack(modality_embeddings, dim=1)  # [B, N, D]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return weighted, weights


class HierarchicalGating(nn.Module):
    """Multi-layer gating with different modality priorities per layer.
    
    - Low-level layer: favors image appearance
    - Mid-level layer: favors audio rhythm
    - High-level layer: favors text semantics
    """

    def __init__(self, hidden_dim=768, num_modalities=3, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim * num_modalities, num_modalities)
            for _ in range(num_layers)
        ])
        # Layer-specific bias: [image_bias, text_bias, audio_bias]
        self.layer_biases = nn.Parameter(torch.tensor([
            [2.0, 0.5, 0.5],   # low: favors image
            [0.5, 1.0, 2.0],   # mid: favors audio
            [0.5, 2.0, 0.5],   # high: favors text
        ]))

    def forward(self, modality_embeddings):
        """
        Returns:
            layer_outputs: [num_layers, B, D]
            layer_weights: [num_layers, B, N]
        """
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
