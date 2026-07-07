"""Dataset loading."""
import os
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .schema import DatasetMetadata, VideoClipSchema
from ..common.utils import load_video, load_json


class VideoDataset(Dataset):
    def __init__(self, metadata_path, transform=None, split="train"):
        self.metadata = self._load_metadata(metadata_path)
        self.clips = self.metadata.get_split(split)
        self.transform = transform

    def _load_metadata(self, path):
        data = load_json(path)
        metadata = DatasetMetadata(name=data.get("name", "unknown"))
        for clip_data in data.get("clips", []):
            clip = VideoClipSchema(**clip_data)
            metadata.add_clip(clip)
        return metadata

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        video = load_video(clip.video_path)
        if self.transform:
            video = self.transform(video)
        return {"video": video, "caption": clip.caption, "clip_id": clip.clip_id}


class MultiModalDataset(Dataset):
    def __init__(self, metadata_path, video_transform=None, audio_transform=None, split="train"):
        self.metadata = load_json(metadata_path)
        self.clips = [c for c in self.metadata.get("clips", []) if c.get("split") == split]
        self.video_transform = video_transform
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        video = load_video(clip["video_path"])
        if self.video_transform:
            video = self.video_transform(video)
        sample = {"video": video, "caption": clip.get("caption", ""), "clip_id": clip["clip_id"]}
        if clip.get("audio_path") and os.path.exists(clip["audio_path"]):
            sample["audio_path"] = clip["audio_path"]
        return sample
