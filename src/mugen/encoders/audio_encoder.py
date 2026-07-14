"""Audio encoder using CLAP or Wav2Vec2."""
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput


class AudioEncoder(BaseEncoderWrapper):
    """Audio encoder for sound understanding."""

    def __init__(self, model_name="laion/clap-htsat-fused", sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        except Exception as exc:
            raise RuntimeError(f"failed to load required audio encoder {model_name}: {exc}") from exc

    @torch.no_grad()
    def encode_audio(self, audio_path, **kwargs):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), return_tensors="pt",
            sampling_rate=self.sample_rate,
        ).to(self.device)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, "last_hidden_state") else outputs.embedding
        return EncoderOutput(
            embedding=embedding,
            metadata={"model": "clap", "audio_path": audio_path},
        )

    def encode_text(self, text, **kwargs):
        raise NotImplementedError

    def encode_image(self, image, **kwargs):
        raise NotImplementedError

    def encode_video(self, video_path, **kwargs):
        raise NotImplementedError
