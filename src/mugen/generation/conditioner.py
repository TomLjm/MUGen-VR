"""Project MUGen-owned multimodal representations into AnyFlow text space."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..common.interfaces import ConditionBundle
from ..fusion.fusion_module import HierarchicalConditionFusion
from .retrieve_then_generate import ReferenceAdapter


class MultimodalConditioner(nn.Module):
    """Build three fusion tokens and four reference tokens for AnyFlow."""

    def __init__(
        self,
        dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 768,
        generator_dim: int = 4096,
        num_reference_tokens: int = 4,
    ):
        super().__init__()
        dims = dims or {"text": 1024, "image": 1024, "audio": 1024, "reference": 768}
        self.modalities = list(dims)
        self.fusion = HierarchicalConditionFusion(dims=dims, hidden_dim=hidden_dim)
        self.reference_adapter = ReferenceAdapter(dims["reference"], num_reference_tokens)
        self.reference_to_hidden = nn.Linear(dims["reference"], hidden_dim)
        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, generator_dim),
        )
        self.type_embeddings = nn.Parameter(
            torch.randn(3 + num_reference_tokens, hidden_dim) * (hidden_dim ** -0.5)
        )

    def forward(
        self,
        prompt: str,
        image_condition,
        modality_embeddings: Dict[str, torch.Tensor],
        reference_embeddings: Optional[torch.Tensor] = None,
        reference_scores: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> ConditionBundle:
        fusion_inputs = {
            name: (modality_embeddings.get(name), None)
            for name in self.modalities
            if name != "reference" and modality_embeddings.get(name) is not None
        }
        reference_tokens = None
        if reference_embeddings is not None:
            reference_tokens = self.reference_adapter(reference_embeddings, reference_scores)
            fusion_inputs["reference"] = (reference_tokens.mean(dim=1), None)
        _, gate_weights, fusion_tokens = self.fusion(
            fusion_inputs, return_weights=True, return_tokens=True
        )
        batch_size = fusion_tokens.shape[0]
        if reference_tokens is None:
            token_count = self.reference_adapter.num_tokens
            reference_tokens = torch.zeros(
                batch_size,
                token_count,
                self.reference_to_hidden.in_features,
                device=fusion_tokens.device,
                dtype=fusion_tokens.dtype,
            )
        reference_tokens = self.reference_to_hidden(reference_tokens)
        tokens = torch.cat([fusion_tokens, reference_tokens], dim=1)
        tokens = tokens + self.type_embeddings.unsqueeze(0).to(tokens)
        condition_tokens = self.token_projector(tokens)
        mask = {name: name in fusion_inputs for name in self.modalities}
        return ConditionBundle(
            prompt=prompt,
            image=image_condition,
            condition_tokens=condition_tokens,
            modality_mask=mask,
            metadata={
                **(metadata or {}),
                "gate_weights": gate_weights.detach().cpu(),
                "condition_token_count": condition_tokens.shape[1],
            },
        )
