"""Cross-attention based multimodal fusion."""
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse multiple modalities via cross-attention."""

    def __init__(self, hidden_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_tokens):
        """
        Args:
            modality_tokens: [B, N_modalities, D] stacked modality embeddings
        Returns:
            fused: [B, D] fused representation
        """
        fused = self.transformer(modality_tokens)
        fused = fused.mean(dim=1)  # pool over modalities
        return self.norm(fused)
