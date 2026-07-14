"""Image encoder using CLIP / ViT."""
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput


class ImageEncoder(BaseEncoderWrapper):
    """Image encoder backed by CLIP vision encoder."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image, **kwargs):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return EncoderOutput(
            embedding=outputs,
            metadata={"model": "clip", "input_type": "image"},
        )

    def encode_text(self, text, **kwargs):
        raise NotImplementedError

    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError

    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError
