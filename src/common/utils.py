"""Utility functions."""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_video(frames, path, fps=8):
    """Save tensor or numpy frames as a video file."""
    import cv2
    if isinstance(frames, torch.Tensor):
        frames = (frames.cpu().numpy() * 255).astype(np.uint8)
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim == 4:
        _, h, w, c = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()

def load_video(path, num_frames=None):
    """Load video file and return numpy array."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if num_frames and len(frames) > num_frames:
        step = len(frames) // num_frames
        frames = [frames[i] for i in range(0, len(frames), step)][:num_frames]
    return np.stack(frames)
