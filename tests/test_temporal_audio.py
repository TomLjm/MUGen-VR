from pathlib import Path

import torch
import torchaudio

from mugen.encoders.temporal_audio import (
    TEMPORAL_AUDIO_DIM,
    TEMPORAL_AUDIO_TOKENS,
    extract_temporal_audio_features,
)


def test_temporal_audio_features_preserve_time_and_onset(tmp_path: Path):
    sample_rate = 16_000
    waveform = torch.zeros(1, sample_rate * 2)
    waveform[:, sample_rate // 2 : sample_rate // 2 + 800] = 1.0
    waveform[:, sample_rate + sample_rate // 2 : sample_rate + sample_rate // 2 + 800] = 1.0
    path = tmp_path / "pulses.wav"
    torchaudio.save(path, waveform, sample_rate)

    features = extract_temporal_audio_features(path)

    assert features.shape == (TEMPORAL_AUDIO_TOKENS, TEMPORAL_AUDIO_DIM)
    assert torch.isfinite(features).all()
    assert features[:, -1].std() > 0.1
