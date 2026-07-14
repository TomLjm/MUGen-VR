"""AnyFlow-FAR-Wan2.1 video generator wrapper."""
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..base import BaseVideoGenerator
from ...common.interfaces import GenerationResult


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
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from diffusers import AnyFlowFARPipeline

            self.pipeline = AnyFlowFARPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
            self.pipeline.to(self.device)
            print(f"[AnyFlow] Loaded from {self.model_id}")
        except Exception as exc:
            print(f"[AnyFlow] Load failed: {exc}")
            self.pipeline = None

    def _preprocess_image(self, image, height=None, width=None):
        height = height or self.height
        width = width or self.width
        transform = transforms.Compose([transforms.Resize([height, width]), transforms.ToTensor()])
        frame = transform(image.convert("RGB"))
        return frame.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _prompt_from_conditions(self, conditions):
        prompt = conditions.get("prompt")
        if prompt and isinstance(prompt, str):
            return prompt
        return "a dynamic scene with natural motion"

    @torch.no_grad()
    def generate(self, conditions, **kwargs):
        if self.pipeline is None:
            return self._dummy_generate(conditions)

        image = conditions.get("image")
        if image is None:
            raise ValueError("AnyFlow-FAR requires conditions['image'] for image-conditioned generation")

        prompt = self._prompt_from_conditions(conditions)
        height = kwargs.get("height", self.height)
        width = kwargs.get("width", self.width)
        n_frames = kwargs.get("num_frames", self.num_frames)
        n_steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        seed = kwargs.get("seed")

        video = self._preprocess_image(image, height=height, width=width)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipeline(
            prompt=prompt,
            video=video,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=n_steps,
            generator=generator,
            output_type="np",
            chunk_partition=kwargs.get("chunk_partition"),
            use_kv_cache=kwargs.get("use_kv_cache", True),
        )

        frames = output.frames[0]
        video_tensor = self._frames_to_tensor(frames)
        return GenerationResult(
            video_frames=video_tensor,
            metadata={"model": self.model_id, "num_frames": n_frames, "prompt": prompt, "seed": seed},
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

    def _dummy_generate(self, conditions):
        frames = torch.randn(25, 3, self.height, self.width).clamp(0, 1)
        return GenerationResult(video_frames=frames, metadata={"model": "dummy-anyflow"})
