"""AnyFlow-FAR-Wan2.1 video generator wrapper."""
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseVideoGenerator
from ...common.interfaces import ConditionBundle, GenerationResult


def build_chunk_partition(num_pixel_frames, temporal_scale, default_partition):
    latent_frames = (num_pixel_frames - 1) // temporal_scale + 1
    default_partition = list(default_partition)
    if sum(default_partition) == latent_frames:
        return default_partition
    chunk_size = default_partition[1] if len(default_partition) > 1 else 3
    partition = [1]
    remaining = latent_frames - 1
    while remaining:
        size = min(chunk_size, remaining)
        partition.append(size)
        remaining -= size
    return partition


class AnyFlowVideoGenerator(BaseVideoGenerator):
    """AnyFlow-FAR generator using Diffusers' `video=` conditioning API."""

    def __init__(
        self,
        model_id: str = "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 4,
    ):
        super().__init__()
        if device is not None:
            self.device = device
        self.model_id = model_id
        self.dtype = dtype
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.pipeline = None
        self.load_error = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from diffusers import AnyFlowFARPipeline

            self.pipeline = AnyFlowFARPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
            self.pipeline.to(self.device)
            print(f"[AnyFlow] Loaded from {self.model_id}")
        except Exception as exc:
            self.load_error = exc
            self.pipeline = None

    def _preprocess_image(self, image, height=None, width=None):
        height = height or self.height
        width = width or self.width
        transform = transforms.Compose([transforms.Resize([height, width]), transforms.ToTensor()])
        frame = transform(image.convert("RGB"))
        return frame.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, conditions: ConditionBundle, **kwargs):
        if not isinstance(conditions, ConditionBundle):
            raise TypeError("AnyFlowVideoGenerator.generate requires a ConditionBundle")
        conditions.validate()
        if self.pipeline is None:
            raise RuntimeError(f"AnyFlow pipeline is unavailable: {self.load_error}")

        prompt = conditions.prompt
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        n_frames = kwargs.get("num_frames", self.num_frames)
        n_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        seed = kwargs.get("seed")

        video = conditions.reference_video
        if video is None:
            video = self._preprocess_image(conditions.image, height=height, width=width)
        prompt_embeds = conditions.prompt_embeds
        if conditions.condition_tokens is not None:
            if prompt_embeds is None:
                prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    max_sequence_length=512,
                    device=self.device,
                )
            prompt_embeds = conditions.merged_prompt_embeds(prompt_embeds)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        chunk_partition = kwargs.get("chunk_partition")
        if chunk_partition is None:
            chunk_partition = build_chunk_partition(
                n_frames,
                self.pipeline.vae_scale_factor_temporal,
                self.pipeline.transformer.config.chunk_partition,
            )

        output = self.pipeline(
            prompt=None if prompt_embeds is not None else prompt,
            prompt_embeds=prompt_embeds,
            video=video,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            generator=generator,
            output_type="np",
            chunk_partition=chunk_partition,
            use_kv_cache=kwargs.get("use_kv_cache", True),
        )

        frames = output.frames[0]
        video_tensor = self._frames_to_tensor(frames)
        return GenerationResult(
            video_frames=video_tensor,
            metadata={
                "model": self.model_id,
                "num_frames": n_frames,
                "prompt": prompt,
                "seed": seed,
                "condition_tokens": 0 if conditions.condition_tokens is None else conditions.condition_tokens.shape[1],
                "chunk_partition": chunk_partition,
            },
        )

    def _frames_to_tensor(self, frames):
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 4 and frames.shape[-1] in (1, 3):
                return frames.permute(0, 3, 1, 2).float().clamp(0, 1).cpu()
            return frames.float().clamp(0, 1).cpu()
        tensors = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                tensors.append(transforms.ToTensor()(frame))
            else:
                arr = np.asarray(frame)
                if arr.dtype != np.float32 and arr.dtype != np.float64:
                    arr = arr.astype("float32") / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
        return torch.stack(tensors).clamp(0, 1)

    def generate_from_text(self, text, **kwargs):
        conditions = kwargs.pop("conditions", {})
        conditions["prompt"] = text
        return self.generate(conditions, **kwargs)

    def generate_from_image(self, image, **kwargs):
        conditions = kwargs.pop("conditions", {})
        conditions["image"] = image
        return self.generate(conditions, **kwargs)
