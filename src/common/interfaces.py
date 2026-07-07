from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import torch


@dataclass
class EncoderOutput:
    """Unified encoder output format."""
    embedding: torch.Tensor
    mask: Optional[torch.Tensor] = None
    seq_length: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Retrieval result."""
    query_embedding: torch.Tensor
    retrieved_indices: List[List[int]]
    retrieved_scores: List[List[float]]
    retrieved_metadata: List[List[Dict[str, Any]]] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Generation result."""
    video_frames: torch.Tensor
    video_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result."""
    metrics: Dict[str, float]
    report_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class BaseEncoder(torch.nn.Module):
    """Base encoder."""
    def encode_text(self, text, **kwargs):
        raise NotImplementedError
    def encode_image(self, image, **kwargs):
        raise NotImplementedError
    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError
    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError


class BaseRetriever:
    """Base retriever."""
    def retrieve(self, query_emb, top_k=5):
        raise NotImplementedError
    def build_index(self, features, metadata):
        raise NotImplementedError


class BaseGenerator(torch.nn.Module):
    """Base generator."""
    def generate(self, conditions, **kwargs):
        raise NotImplementedError


class BaseEvaluator:
    """Base evaluator."""
    def evaluate(self, generated_videos, references=None):
        raise NotImplementedError
