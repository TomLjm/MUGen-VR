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
        dims=None,
        hidden_dim=768,
        num_heads=8,
        dropout=0.1,
        modality_dropout=0.2,
    ):
        super().__init__()
        if dims is None:
            dims = {"text": 768, "image": 1024, "audio": 512}
        self.modality_order = list(dims.keys())
        self.num_modalities = len(dims)

        self.projectors = nn.ModuleDict({
            mod: ModalityProjector(dim, hidden_dim)
            for mod, dim in dims.items()
        })
        self.hierarchical_gating = HierarchicalGating(hidden_dim, self.num_modalities)
        self.cross_attention = CrossAttentionFusion(hidden_dim, num_heads)
        self.modality_dropout = ModalityDropout(modality_dropout, self.num_modalities)
        self.output_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, modality_inputs, return_weights=False, return_tokens=False):
        """
        Args:
            modality_inputs: dict of {modality: (tensor, mask)} pairs
        Returns:
            fused: [B, D] fused embedding
        """
        B = None
        for mod, (emb, mask) in modality_inputs.items():
            if emb is not None:
                B = emb.size(0)
                break

        # Project to unified space
        projected = {}
        for mod in self.modality_order:
            item = modality_inputs.get(mod)
            if item is not None:
                emb, mask = item
                if emb is not None:
                    projected[mod] = self.projectors[mod](emb)

        # Apply modality dropout
        projected = self.modality_dropout(projected)
        projected = {k: v for k, v in projected.items() if v is not None}

        if len(projected) == 0:
            raise ValueError("at least one modality embedding is required")

        # Pad missing modalities with zeros for gating
        D = next(iter(projected.values())).size(-1)
        device = next(iter(projected.values())).device
        emb_list = []
        for mod in self.modality_order:
            if mod in projected:
                emb_list.append(projected[mod])
            else:
                emb_list.append(torch.zeros(B, D, device=device, dtype=next(iter(projected.values())).dtype))

        layer_outputs, layer_weights = self.hierarchical_gating(emb_list)

        # Combine hierarchical outputs
        combined = layer_outputs.permute(1, 0, 2).reshape(B, -1)
        combined = self.output_proj(combined)

        # Stack for cross-attention
        stacked = torch.stack(emb_list, dim=1)
        fused = self.cross_attention(stacked)

        output = (combined + fused) / 2

        if return_tokens and return_weights:
            return output, layer_weights, layer_outputs.permute(1, 0, 2)
        if return_tokens:
            return output, layer_outputs.permute(1, 0, 2)
        if return_weights:
            return output, layer_weights
        return output
