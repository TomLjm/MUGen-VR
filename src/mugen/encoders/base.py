"""Base encoder wrapper."""
import torch
from ..common.interfaces import BaseEncoder, EncoderOutput


class BaseEncoderWrapper(BaseEncoder):
    """Wrapper that adapts third-party encoders to BaseEncoder interface."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, tensor):
        return tensor.to(self.device)

    def encode_text(self, text, **kwargs):
        raise NotImplementedError

    def encode_image(self, image, **kwargs):
        raise NotImplementedError

    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError

    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError
