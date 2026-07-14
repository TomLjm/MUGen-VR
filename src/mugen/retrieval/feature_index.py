import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .base import BaseRetrievalIndex
from ..common.utils import ensure_dir


class FeatureIndex(BaseRetrievalIndex):
    """Small flat feature index with metadata-aware lookup."""

    def __init__(self, dim: int = 768, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index: np.ndarray | None = None

    def add(self, embeddings, metadata_list=None):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().float().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        if embeddings.shape[-1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {embeddings.shape[-1]}")
        start = self.size
        self.embeddings.append(embeddings)
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(embeddings))]
        if len(metadata_list) != len(embeddings):
            raise ValueError("metadata_list length must match embeddings")
        for i, meta in enumerate(metadata_list):
            item = dict(meta or {})
            item.setdefault("index", start + i)
            item.setdefault("embedding", embeddings[i])
            self.metadata.append(item)
        self._build_index()

    def _build_index(self):
        if not self.embeddings:
            self._index = None
            return
        self._index = np.concatenate(self.embeddings, axis=0)
        norms = np.linalg.norm(self._index, axis=1, keepdims=True).clip(min=1e-8)
        self._index = self._index / norms

    def search(self, query, top_k=5) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        if self._index is None or len(self._index) == 0:
            return [], [], []
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().float().numpy()
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        if query.shape[-1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {query.shape[-1]}")
        query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
        similarities = np.dot(self._index, query[0].T).squeeze()
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        top_scores = similarities[top_indices]
        result_metadata = [self.metadata[i] for i in top_indices]
        return top_indices.tolist(), top_scores.astype(float).tolist(), result_metadata

    def get_embedding(self, index: int) -> torch.Tensor:
        if self._index is None:
            raise IndexError("index is empty")
        return torch.from_numpy(self._index[index]).float()

    def save(self, path):
        parent = os.path.dirname(path)
        if parent:
            ensure_dir(parent)
        data = {"dim": self.dim, "embeddings": self.embeddings, "metadata": self.metadata}
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
