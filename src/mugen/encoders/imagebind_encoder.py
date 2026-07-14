"""Strict ImageBind wrapper for real text, image, and audio features."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput


class ImageBindEncoder(BaseEncoderWrapper):
    def __init__(
        self,
        repository: str | Path = "third_party/ImageBind",
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__()
        repository = Path(repository).resolve()
        if not (repository / "imagebind").is_dir():
            raise FileNotFoundError(f"ImageBind repository not found: {repository}")
        sys.path.insert(0, str(repository))
        try:
            from imagebind import data
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
        except Exception as exc:
            raise RuntimeError(f"failed to import ImageBind from {repository}: {exc}") from exc
        self.data = data
        self.modality_type = ModalityType
        self.model = imagebind_model.imagebind_huge(pretrained=False)
        if pretrained:
            checkpoint = Path(checkpoint_path) if checkpoint_path else repository / ".checkpoints" / "imagebind_huge.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"ImageBind checkpoint not found: {checkpoint}. "
                    "Run scripts/setup_third_party.sh before real feature extraction."
                )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self.version = "ImageBind@53680b0"

    def _encode(self, modality, inputs):
        with torch.no_grad():
            output = self.model({modality: inputs})[modality]
        return F.normalize(output.float(), dim=-1)

    def encode_text(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        inputs = self.data.load_and_transform_text(texts, self.device)
        return EncoderOutput(
            embedding=self._encode(self.modality_type.TEXT, inputs),
            metadata={"model": self.version, "count": len(texts)},
        )

    def encode_image(self, image, **kwargs):
        paths: Sequence[str] = [str(image)] if isinstance(image, (str, Path)) else [str(value) for value in image]
        inputs = self.data.load_and_transform_vision_data(paths, self.device)
        return EncoderOutput(
            embedding=self._encode(self.modality_type.VISION, inputs),
            metadata={"model": self.version, "count": len(paths)},
        )

    def encode_audio(self, audio_path, **kwargs):
        paths: Sequence[str] = (
            [str(audio_path)]
            if isinstance(audio_path, (str, Path))
            else [str(value) for value in audio_path]
        )
        inputs = self.data.load_and_transform_audio_data(paths, self.device)
        return EncoderOutput(
            embedding=self._encode(self.modality_type.AUDIO, inputs),
            metadata={"model": self.version, "count": len(paths)},
        )

    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError("Use InternVideoEncoder for video features")
