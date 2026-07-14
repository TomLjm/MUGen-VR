"""Video decoding helpers used by evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import av
import torch


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def find_videos(path: str | Path) -> List[Path]:
    root = Path(path)
    if root.is_file():
        candidates: Iterable[Path] = [root]
    elif root.is_dir():
        candidates = root.rglob("*")
    else:
        raise FileNotFoundError(f"generated video path does not exist: {root}")
    videos = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"no supported video files found under {root}")
    return videos


def decode_video(path: str | Path, max_frames: int | None = None) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            array = frame.to_rgb().to_ndarray()
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    return torch.stack(frames).float().div_(255.0)
