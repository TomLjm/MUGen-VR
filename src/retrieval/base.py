"""Base retrieval index interface."""
import torch


class BaseRetrievalIndex:
    """Base class for retrieval index."""
    
    def add(self, embeddings, metadata):
        raise NotImplementedError

    def search(self, query, top_k=5):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
