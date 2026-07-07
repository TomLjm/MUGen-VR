"""HierarchicalConditionFusion: the core innovation A module."""
import torch
import torch.nn as nn

from .projector import ModalityProjector
from .cross_attention import CrossAttentionFusion
from .gating import HierarchicalGating
from .modality_dropout import ModalityDropout


class HierarchicalConditionFusion(nn.Module):
    """
    Innovation A: Hierarchical multi-modal condition fusion.
    
    Architecture:
        1. Project each modality to a unified space
        2. Hierarchical gating at multiple levels
        3. Cross-attention fusion for final representation
    
    The hierarchy:
        - Low level: emphasizes image/appearance features
        - Mid level: emphasizes audio/rhythm features
        - High level: emphasizes text/semantic features
    """

    def __init__(
        self,
        dims={"text": 768, "image": 1024, "audio": 512},
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        modality_dropout=0.2,
    ):
        super().__init__()
        self.projectors = nn.ModuleDict({
            mod: ModalityProjector(dim, hidden_dim)
            for mod, dim in dims.items()
        })
        self.hierarchical_gating = HierarchicalGating(hidden_dim, len(dims))
        self.cross_attention = CrossAttentionFusion(hidden_dim, num_heads)
        self.modality_dropout = ModalityDropout(modality_dropout, len(dims))
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)  # combine 3 layers

    def forward(self, modality_inputs, return_weights=False):
        """
        Args:
            modality_inputs: dict of {modality: (tensor, mask)} pairs
        Returns:
            fused: [B, D] fused embedding
        """
        # Project to unified space
        projected = {}
        for mod, (emb, mask) in modality_inputs.items():
            if emb is not None:
                projected[mod] = self.projectors[mod](emb)

        # Apply modality dropout in training
        projected = self.modality_dropout(projected)

        # Filter None
        projected = {k: v for k, v in projected.items() if v is not None}

        if len(projected) == 0:
            return torch.zeros(1, self.output_proj.in_features)

        emb_list = list(projected.values())

        # Hierarchical gating
        layer_outputs, layer_weights = self.hierarchical_gating(emb_list)
        
        # Combine hierarchical outputs
        B = emb_list[0].size(0)
        combined = layer_outputs.permute(1, 0, 2).reshape(B, -1)  # [B, L*D]
        combined = self.output_proj(combined)

        # Stack for cross-attention
        stacked = torch.stack(emb_list, dim=1)
        fused = self.cross_attention(stacked)

        # Final combination
        output = (combined + fused) / 2

        if return_weights:
            return output, layer_weights
        return output
