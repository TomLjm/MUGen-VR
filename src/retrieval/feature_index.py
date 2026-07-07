"""Feature indexing and storage."""
import os
import pickle
import numpy as np
import torch

from .base import BaseRetrievalIndex
from ..common.utils import ensure_dir


class FeatureIndex(BaseRetrievalIndex):
    """FAISS-like feature index for similarity search."""

    def __init__(self, dim=768, index_type="flat"):
        self.dim = dim
        self.index_type = index_type
        self.embeddings = []
        self.metadata = []
        self._index = None

    def add(self, embeddings, metadata_list=None):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        self.embeddings.append(embeddings)
        if metadata_list:
            self.metadata.extend(metadata_list)
        self._build_index()

    def _build_index(self):
        all_embs = np.concatenate(self.embeddings, axis=0) if len(self.embeddings) > 1 else self.embeddings[0]
        self._index = all_embs

    def search(self, query, top_k=5):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        if self._index is None or len(self._index) == 0:
            return [], []
        
        similarities = np.dot(self._index, query.T).squeeze()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        result_metadata = [self.metadata[i] for i in top_indices] if self.metadata else []
        return top_indices.tolist(), top_scores.tolist(), result_metadata

    def save(self, path):
        ensure_dir(os.path.dirname(path))
        data = {
            "dim": self.dim,
            "embeddings": self.embeddings,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.dim = data["dim"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self._build_index()

    @property
    def size(self):
        if self._index is not None:
            return len(self._index)
        return sum(len(e) for e in self.embeddings)
