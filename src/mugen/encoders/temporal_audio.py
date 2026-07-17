"""Time-resolved audio features for motion-aware video conditioning."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


TEMPORAL_AUDIO_TOKENS = 8
TEMPORAL_AUDIO_MELS = 64
TEMPORAL_AUDIO_DIM = TEMPORAL_AUDIO_MELS + 2
TEMPORAL_AUDIO_VERSION = "logmel-onset-v1:t8:m64:sr16000"


def extract_temporal_audio_features(
    audio_path: str | Path,
    num_tokens: int = TEMPORAL_AUDIO_TOKENS,
    n_mels: int = TEMPORAL_AUDIO_MELS,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Return [time_tokens, mel_bins + energy + onset] real audio features."""
    waveform, source_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError(f"empty audio waveform: {audio_path}")
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    if waveform.shape[-1] < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        f_min=20,
        f_max=7_600,
        n_mels=n_mels,
        power=2.0,
    )(waveform).squeeze(0)
    log_mel = torch.log1p(mel)
    log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-5)
    energy = torch.log1p(mel.sum(dim=0))
    energy = (energy - energy.mean()) / energy.std().clamp_min(1e-5)
    onset = torch.nn.functional.pad(torch.relu(energy[1:] - energy[:-1]), (1, 0))
    onset = onset / onset.std().clamp_min(1e-5)

    boundaries = torch.linspace(0, log_mel.shape[-1], num_tokens + 1).round().long()
    features = []
    for index in range(num_tokens):
        start = min(int(boundaries[index]), log_mel.shape[-1] - 1)
        end = max(start + 1, int(boundaries[index + 1]))
        end = min(end, log_mel.shape[-1])
        features.append(
            torch.cat(
                [
                    log_mel[:, start:end].mean(dim=-1),
                    energy[start:end].mean().view(1),
                    onset[start:end].mean().view(1),
                ]
            )
        )
    result = torch.stack(features).float()
    if not torch.isfinite(result).all():
        raise ValueError(f"non-finite temporal audio features: {audio_path}")
    return result
