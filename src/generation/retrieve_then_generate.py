import torch
import torch.nn as nn
from .base import BaseVideoGenerator
from ..common.interfaces import GenerationResult
from ..retrieval.retriever import CrossModalRetriever


class RetrieveThenGenerate(BaseVideoGenerator):
    def __init__(self, retriever=None, generator=None, dim=768, top_k=3):
        super().__init__()
        self.retriever = retriever or CrossModalRetriever(dim=dim)
        self.generator = generator
        self.top_k = top_k
        self.aggregator = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def set_generator(self, generator):
        self.generator = generator

    def generate(self, conditions, **kwargs):
        query_emb = self._encode_query(conditions)
        retrieval_result = self.retriever.retrieve(query_emb, top_k=self.top_k)
        ref_features = self._get_reference_features(retrieval_result)
        if ref_features is not None:
            enhanced = self.aggregator(torch.cat([query_emb, ref_features], dim=-1))
            conditions["reference_embedding"] = enhanced
        if self.generator:
            return self.generator.generate(conditions, **kwargs)
        else:
            return self._dummy_generate(conditions)

    def _encode_query(self, conditions):
        embeddings = []
        for mod in ["text", "image", "audio"]:
            if mod in conditions and conditions[mod] is not None:
                emb = conditions[mod]
                if hasattr(emb, "embedding"):
                    emb = emb.embedding
                embeddings.append(emb)
        if not embeddings:
            return torch.zeros(1, 768)
        return torch.cat(embeddings, dim=-1) if len(embeddings) > 1 else embeddings[0]

    def _get_reference_features(self, retrieval_result):
        if not retrieval_result.retrieved_indices[0]:
            return None
        return None

    def _dummy_generate(self, conditions):
        frames = torch.randn(16, 3, 256, 256).clamp(0, 1)
        return GenerationResult(video_frames=frames, metadata={"pipeline": "retrieve-then-generate"})
