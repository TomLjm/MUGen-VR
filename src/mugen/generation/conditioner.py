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
        temporal_audio_dim: Optional[int] = None,
        num_temporal_audio_tokens: int = 8,
    ):
        super().__init__()
        dims = dims or {"text": 1024, "image": 1024, "audio": 1024, "reference": 768}
        self.modalities = list(dims)
        self.fusion = HierarchicalConditionFusion(dims=dims, hidden_dim=hidden_dim)
        self.reference_adapter = ReferenceAdapter(dims["reference"], num_reference_tokens)
        self.reference_to_hidden = nn.Linear(dims["reference"], hidden_dim)
        self.num_temporal_audio_tokens = (
            int(num_temporal_audio_tokens) if temporal_audio_dim is not None else 0
        )
        self.temporal_audio_projector = None
        self.temporal_position_embeddings = None
        if temporal_audio_dim is not None:
            if int(temporal_audio_dim) % self.num_temporal_audio_tokens:
                raise ValueError("temporal_audio_dim must be divisible by token count")
            temporal_frame_dim = int(temporal_audio_dim) // self.num_temporal_audio_tokens
            self.temporal_audio_projector = nn.Sequential(
                nn.LayerNorm(temporal_frame_dim),
                nn.Linear(temporal_frame_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.temporal_position_embeddings = nn.Parameter(
                torch.randn(self.num_temporal_audio_tokens, hidden_dim) * (hidden_dim ** -0.5)
            )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, generator_dim),
        )
        self.type_embeddings = nn.Parameter(
            torch.randn(
                3 + num_reference_tokens + self.num_temporal_audio_tokens, hidden_dim
            )
            * (hidden_dim ** -0.5)
        )

    def forward(
        self,
        prompt: str,
        image_condition,
        modality_embeddings: Dict[str, torch.Tensor],
        reference_embeddings: Optional[torch.Tensor] = None,
        reference_scores: Optional[torch.Tensor] = None,
        temporal_audio_embeddings: Optional[torch.Tensor] = None,
        reference_gallery: Optional[torch.Tensor] = None,
        retrieval_modality_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        reference_top_k: int = 3,
        reference_exclude_indices: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> ConditionBundle:
        fusion_inputs = {
            name: (modality_embeddings.get(name), None)
            for name in self.modalities
            if name != "reference" and modality_embeddings.get(name) is not None
        }
        retrieved_indices = None
        if reference_embeddings is None and reference_gallery is not None:
            retrieval_embeddings = retrieval_modality_embeddings or modality_embeddings
            retrieval_inputs = {
                name: (retrieval_embeddings.get(name), None)
                for name in self.modalities
                if name != "reference" and retrieval_embeddings.get(name) is not None
            }
            retrieval_query = self.fusion(retrieval_inputs)
            scores = torch.nn.functional.normalize(retrieval_query.float(), dim=-1) @ (
                torch.nn.functional.normalize(reference_gallery.float(), dim=-1).T
            )
            if reference_exclude_indices is not None:
                for batch_index, gallery_index in enumerate(reference_exclude_indices.tolist()):
                    if gallery_index >= 0:
                        scores[batch_index, gallery_index] = -torch.inf
            reference_scores, retrieved_indices = scores.topk(
                min(int(reference_top_k), reference_gallery.shape[0]), dim=-1
            )
            reference_embeddings = reference_gallery[retrieved_indices]

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
        tokens = [fusion_tokens, reference_tokens]
        if self.temporal_audio_projector is not None:
            if temporal_audio_embeddings is None:
                temporal_tokens = torch.zeros(
                    batch_size,
                    self.num_temporal_audio_tokens,
                    self.type_embeddings.shape[-1],
                    device=fusion_tokens.device,
                    dtype=fusion_tokens.dtype,
                )
            else:
                temporal_tokens = temporal_audio_embeddings.reshape(
                    batch_size, self.num_temporal_audio_tokens, -1
                )
                temporal_tokens = self.temporal_audio_projector(temporal_tokens)
                temporal_tokens = temporal_tokens + self.temporal_position_embeddings.unsqueeze(0).to(
                    temporal_tokens
                )
            tokens.append(temporal_tokens)
        elif temporal_audio_embeddings is not None:
            raise ValueError("conditioner was not configured for temporal audio features")
        tokens = torch.cat(tokens, dim=1)
        tokens = tokens + self.type_embeddings.unsqueeze(0).to(tokens)
        condition_tokens = self.token_projector(tokens)
        mask = {name: name in fusion_inputs for name in self.modalities}
        if self.temporal_audio_projector is not None:
            mask["audio_temporal"] = temporal_audio_embeddings is not None
        return ConditionBundle(
            prompt=prompt,
            image=image_condition,
            condition_tokens=condition_tokens,
            modality_mask=mask,
            metadata={
                **(metadata or {}),
                "gate_weights": gate_weights.detach().cpu(),
                "condition_token_count": condition_tokens.shape[1],
                "reference_indices": None
                if retrieved_indices is None
                else retrieved_indices.detach().cpu(),
            },
        )
