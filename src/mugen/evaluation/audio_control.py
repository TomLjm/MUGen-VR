"""Audio-conditioned video alignment and rhythm metrics."""

from __future__ import annotations

import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile


def resample_series(values, length):
    values = np.asarray(values, dtype=np.float64)
    if length < 1 or values.size < 1:
        raise ValueError("values and target length must be non-empty")
    if values.size == length:
        return values
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values)


def audio_onset_envelope(path, frame_size=1024, hop_size=512):
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    rms = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        window = audio[start : start + frame_size]
        rms.append(np.sqrt(np.mean(window**2)))
    onset = np.maximum(np.diff(np.asarray(rms), prepend=rms[0]), 0.0)
    return onset, int(sample_rate)


def optical_flow_envelope(path, max_side=224):
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            gray = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2GRAY)
            scale = min(1.0, max_side / max(gray.shape))
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(gray)
    if len(frames) < 2:
        raise ValueError(f"at least two video frames are required: {path}")
    flow_energy = []
    for previous, current in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_energy.append(float(np.linalg.norm(flow, axis=-1).mean()))
    return np.asarray(flow_energy)


def onset_flow_correlation(audio_path, video_path):
    onset, _ = audio_onset_envelope(audio_path)
    flow = optical_flow_envelope(video_path)
    onset = resample_series(onset, len(flow))
    if np.std(onset) == 0 or np.std(flow) == 0:
        return 0.0
    return float(np.corrcoef(onset, flow)[0, 1])


def imagebind_audio_video_alignment(encoder, audio_path, video_path, frame_count=8):
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
    if not frames:
        raise ValueError(f"no frames decoded: {video_path}")
    indices = np.linspace(0, len(frames) - 1, min(frame_count, len(frames))).astype(int)
    with tempfile.TemporaryDirectory(prefix="mugen-imagebind-") as directory:
        paths = []
        for position, index in enumerate(indices):
            path = Path(directory) / f"frame-{position:03d}.jpg"
            Image.fromarray(frames[index]).save(path, quality=95)
            paths.append(path)
        vision = encoder.encode_image(paths).embedding.mean(dim=0, keepdim=True)
    audio = encoder.encode_audio(audio_path).embedding
    return float(torch.nn.functional.cosine_similarity(vision.float(), audio.float()).mean())
