import torch
from .base import BaseVideoGenerator
from ..common.interfaces import GenerationResult

class VideoGenerator(BaseVideoGenerator):
    def __init__(self, model_name="stabilityai/stable-video-diffusion-img2vid", dtype=torch.float16):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from diffusers import StableVideoDiffusionPipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_name, torch_dtype=self.dtype, variant="fp16"
            ).to(self.device)
            self.pipeline.enable_model_cpu_offload()
            print(f"Loaded {self.model_name}")
        except Exception as e:
            print(f"Could not load {self.model_name}: {e}")
            print("Using dummy generator for development")

    @torch.no_grad()
    def generate(self, conditions, **kwargs):
        if self.pipeline is None:
            return self._dummy_generate(conditions)
        image = conditions.get("image")
        if image is None:
            raise ValueError("SVD requires an image input")
        frames = self.pipeline(
            image, decode_chunk_size=8,
            motion_bucket_id=kwargs.get("motion_bucket_id", 127),
            noise_aug_strength=kwargs.get("noise_aug_strength", 0.02),
            num_frames=kwargs.get("num_frames", 16),
        ).frames[0]
        video_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames])
        return GenerationResult(video_frames=video_tensor, metadata={"model": self.model_name})

    def generate_from_text(self, text, **kwargs):
        return self.generate({"text": text}, **kwargs)

    def generate_from_image(self, image, **kwargs):
        return self.generate({"image": image}, **kwargs)

    def _dummy_generate(self, conditions):
        frames = torch.randn(16, 3, 256, 256).clamp(0, 1)
        return GenerationResult(video_frames=frames, metadata={"model": "dummy"})
