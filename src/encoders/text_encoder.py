"""Text encoder using CLIP / BERT."""
import torch
from transformers import AutoTokenizer, AutoModel
from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput


class TextEncoder(BaseEncoderWrapper):
    """Text encoder backed by pretrained transformer."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", max_length=77):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        embedding = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
        return EncoderOutput(
            embedding=embedding,
            mask=inputs["attention_mask"],
            metadata={"model": "clip", "texts": text},
        )

    def encode_image(self, image, **kwargs):
        raise NotImplementedError("TextEncoder does not support image input")

    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError("TextEncoder does not support audio input")

    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError("TextEncoder does not support video input")
