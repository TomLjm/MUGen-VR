from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import torch


@dataclass
class ConditionBundle:
    """Normalized multimodal input consumed by video generators."""

    prompt: str
    image: Any
    prompt_embeds: Optional[torch.Tensor] = None
    condition_tokens: Optional[torch.Tensor] = None
    reference_video: Optional[torch.Tensor] = None
    modality_mask: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt and self.prompt_embeds is None:
            raise ValueError("ConditionBundle requires prompt text or prompt_embeds")
        if self.image is None and self.reference_video is None:
            raise ValueError("ConditionBundle requires an image or reference video")
        if self.condition_tokens is not None and self.condition_tokens.ndim != 3:
            raise ValueError("condition_tokens must have shape [batch, tokens, dim]")
        if self.prompt_embeds is not None and self.prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must have shape [batch, sequence, dim]")

    def merged_prompt_embeds(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        if self.condition_tokens is None:
            return prompt_embeds
        tokens = self.condition_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        if tokens.shape[0] != prompt_embeds.shape[0] or tokens.shape[-1] != prompt_embeds.shape[-1]:
            raise ValueError(
                "condition tokens must match prompt embedding batch and hidden dimensions"
            )
        return torch.cat([prompt_embeds, tokens], dim=1)


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
