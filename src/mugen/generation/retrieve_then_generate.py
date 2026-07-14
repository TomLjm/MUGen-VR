import torch
import torch.nn as nn

from .base import BaseVideoGenerator
from ..common.interfaces import GenerationResult
from ..retrieval.retriever import CrossModalRetriever


class ReferenceAdapter(nn.Module):
    """Converts scored reference embeddings into a fixed token sequence."""

    def __init__(self, dim=768, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.keys = nn.Linear(dim, dim)
        self.values = nn.Linear(dim, dim)
        self.token_queries = nn.Parameter(torch.randn(num_tokens, dim) * (dim ** -0.5))
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, reference_embeddings, scores=None):
        if reference_embeddings is None or reference_embeddings.numel() == 0:
            return None
        if reference_embeddings.dim() == 2:
            reference_embeddings = reference_embeddings.unsqueeze(0)
        batch_size = reference_embeddings.shape[0]
        score_bias = 0.0
        if scores is not None:
            scores = torch.as_tensor(scores, dtype=reference_embeddings.dtype, device=reference_embeddings.device)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            score_bias = torch.log_softmax(scores, dim=-1).unsqueeze(1)
        queries = self.token_queries.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.keys(reference_embeddings)
        values = self.values(reference_embeddings)
        logits = torch.matmul(queries, keys.transpose(-1, -2)) * (keys.shape[-1] ** -0.5)
        weights = torch.softmax(logits + score_bias, dim=-1)
        return self.output(torch.matmul(weights, values))


class RetrieveThenGenerate(BaseVideoGenerator):
    def __init__(self, retriever=None, generator=None, dim=768, top_k=3, allow_dummy=False):
        super().__init__()
        self.retriever = retriever or CrossModalRetriever(dim=dim)
        self.generator = generator
        self.top_k = top_k
        self.allow_dummy = allow_dummy
        self.reference_adapter = ReferenceAdapter(dim)
        self.aggregator = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))

    def set_generator(self, generator):
        self.generator = generator

    def generate(self, conditions, **kwargs):
        query_emb = self._encode_query(conditions)
        retrieval_result = self.retriever.retrieve(query_emb, top_k=self.top_k)
        ref_features = self._get_reference_features(retrieval_result)
        if ref_features is not None:
            scores = retrieval_result.retrieved_scores[0] if retrieval_result.retrieved_scores else None
            reference_tokens = self.reference_adapter(ref_features.unsqueeze(0), scores=scores)
            reference_embedding = reference_tokens.mean(dim=1)
            enhanced = self.aggregator(torch.cat([query_emb, reference_embedding], dim=-1))
            conditions["reference_tokens"] = reference_tokens
            conditions["reference_embedding"] = reference_embedding
            conditions["enhanced_condition"] = enhanced
            conditions["retrieval_result"] = retrieval_result
        if self.generator:
            return self.generator.generate(conditions, **kwargs)
        if kwargs.get("allow_dummy", self.allow_dummy):
            return self._dummy_generate(conditions)
        raise RuntimeError("RetrieveThenGenerate has no generator. Pass allow_dummy=True only for smoke tests.")

    def _encode_query(self, conditions):
        embeddings = []
        for mod in ["text", "image", "audio", "reference"]:
            emb = conditions.get(mod)
            if emb is None:
                continue
            if hasattr(emb, "embedding"):
                emb = emb.embedding
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
        if not embeddings:
            return torch.zeros(1, self.retriever.dim)
        query = torch.stack(embeddings, dim=0).mean(dim=0)
        return torch.nn.functional.normalize(query, dim=-1)

    def _get_reference_features(self, retrieval_result):
        indices = retrieval_result.retrieved_indices[0] if retrieval_result.retrieved_indices else []
        if not indices:
            return None
        refs = []
        metadata = retrieval_result.retrieved_metadata[0] if retrieval_result.retrieved_metadata else []
        for i, meta in zip(indices, metadata):
            emb = meta.get("embedding") if isinstance(meta, dict) else None
            if emb is None:
                emb = self.retriever.index.get_embedding(i)
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb, dtype=torch.float32)
            refs.append(emb.float())
        return torch.stack(refs, dim=0) if refs else None

    def _dummy_generate(self, conditions):
        frames = torch.linspace(0, 1, steps=16).view(16, 1, 1, 1).repeat(1, 3, 256, 256)
        return GenerationResult(video_frames=frames, metadata={"pipeline": "retrieve-then-generate", "mode": "dummy"})
