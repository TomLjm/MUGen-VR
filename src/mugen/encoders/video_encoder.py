"""Video encoder using InternVideo or VideoMAE."""
import torch
from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput
from ..common.utils import load_video


class VideoEncoder(BaseEncoderWrapper):
    """Video encoder wrapper (backed by InternVideo or similar)."""

    def __init__(self, model_name="internvideo", checkpoint_path=None):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.transform = None
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path):
        """Load pretrained video encoder weights."""
        print(f"Loading video encoder from {path}")
        # TODO: Implement actual InternVideo model loading
        # self.model = InternVideo2(...)
        self.model = DummyVideoEncoder().to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_video(self, video_path, **kwargs):
        if self.model is None:
            raise RuntimeError("InternVideo checkpoint is required; dummy video features are not allowed")
        video = load_video(video_path, num_frames=8).unsqueeze(0).to(self.device)
        embedding = self.model(video)
        return EncoderOutput(
            embedding=embedding,
            metadata={"model": self.model_name, "video_path": video_path},
        )

    def encode_text(self, text, **kwargs):
        raise NotImplementedError

    def encode_image(self, image, **kwargs):
        raise NotImplementedError

    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError


class DummyVideoEncoder(torch.nn.Module):
    """Placeholder video encoder for initial development."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, embed_dim),
        )

    def forward(self, video):
        # video: [B, T, C, H, W] -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        return self.backbone(video)
