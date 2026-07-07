"""Cross-modal retrieval pipeline."""
import torch
import numpy as np

from ..common.interfaces import BaseRetriever, RetrievalResult
from .feature_index import FeatureIndex


class CrossModalRetriever(BaseRetriever):
    """Unified cross-modal retriever.
    
    Supports:
        - text-to-video retrieval
        - image-to-video retrieval
        - audio-to-video retrieval
        - multi-modal fusion retrieval
    """

    def __init__(self, dim=768, top_k=5):
        self.dim = dim
        self.top_k = top_k
        self.index = FeatureIndex(dim)
        self.similarity_weights = {"text": 0.4, "image": 0.3, "audio": 0.3}

    def build_index(self, features, metadata_list):
        self.index.add(features, metadata_list)

    def retrieve(self, query_emb, top_k=None):
        k = top_k or self.top_k
        indices, scores, meta = self.index.search(query_emb, k)
        return RetrievalResult(
            query_embedding=query_emb,
            retrieved_indices=[indices],
            retrieved_scores=[scores],
            retrieved_metadata=[meta],
        )

    def multi_modal_retrieve(self, query_dict, top_k=None):
        """
        Args:
            query_dict: dict of {modality: embedding}
        Returns:
            fused retrieval result
        """
        k = top_k or self.top_k
        all_scores = []

        for modality, emb in query_dict.items():
            if emb is None:
                continue
            weight = self.similarity_weights.get(modality, 1.0 / len(query_dict))
            indices, scores, _ = self.index.search(emb, k)
            all_scores.append(weight * np.array(scores))

        if not all_scores:
            return RetrievalResult(
                query_embedding=torch.zeros(1, self.dim),
                retrieved_indices=[[]],
                retrieved_scores=[[]],
            )

        fused_scores = np.mean(all_scores, axis=0)
        sorted_idx = np.argsort(fused_scores)[-k:][::-1]
        
        return RetrievalResult(
            query_embedding=torch.cat([v for v in query_dict.values() if v is not None], dim=-1),
            retrieved_indices=[sorted_idx.tolist()],
            retrieved_scores=[fused_scores[sorted_idx].tolist()],
            retrieved_metadata=[[]],
        )

    def save_index(self, path):
        self.index.save(path)

    def load_index(self, path):
        self.index.load(path)
