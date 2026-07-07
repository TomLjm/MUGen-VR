"""Modality projector: maps diverse encoder outputs to a unified space."""
import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Project modality-specific embeddings to a shared latent space."""

    def __init__(self, input_dim=768, output_dim=768, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
