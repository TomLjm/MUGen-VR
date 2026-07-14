import torch

from ..common.interfaces import BaseRetriever, RetrievalResult
from .feature_index import FeatureIndex


class CrossModalRetriever(BaseRetriever):
    """Unified cross-modal retriever."""

    def __init__(self, dim=768, top_k=5):
        self.dim = dim
        self.top_k = top_k
        self.index = FeatureIndex(dim)
        self.similarity_weights = {"text": 0.4, "image": 0.3, "audio": 0.2, "reference": 0.1}

    def build_index(self, features, metadata_list=None):
        self.index.add(features, metadata_list)

    def retrieve(self, query_emb, top_k=None):
        k = top_k or self.top_k
        indices, scores, meta = self.index.search(query_emb, k)
        return RetrievalResult(query_embedding=query_emb, retrieved_indices=[indices], retrieved_scores=[scores], retrieved_metadata=[meta])

    def multi_modal_retrieve(self, query_dict, top_k=None):
        k = top_k or self.top_k
        weighted_queries = []
        weights = []
        for modality, emb in query_dict.items():
            if emb is None:
                continue
            if hasattr(emb, "embedding"):
                emb = emb.embedding
            weighted_queries.append(emb)
            weights.append(self.similarity_weights.get(modality, 1.0))
        if not weighted_queries:
            return RetrievalResult(torch.zeros(1, self.dim), [[]], [[]], [[]])
        weights_t = torch.tensor(weights, dtype=weighted_queries[0].dtype, device=weighted_queries[0].device)
        weights_t = weights_t / weights_t.sum().clamp_min(1e-8)
        query = sum(w * e for w, e in zip(weights_t, weighted_queries))
        return self.retrieve(query, top_k=k)

    def get_embeddings(self, indices):
        if not indices:
            return None
        return torch.stack([self.index.get_embedding(i) for i in indices], dim=0)

    def save_index(self, path):
        self.index.save(path)

    def load_index(self, path):
        self.index.load(path)
