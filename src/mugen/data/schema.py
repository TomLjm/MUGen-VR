"""Data schema."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class VideoClipSchema:
    clip_id: str
    video_path: str
    audio_path: Optional[str] = None
    keyframes_path: Optional[str] = None
    caption: str = ""
    duration: float = 0.0
    fps: float = 8.0
    split: str = "train"
    cached_features: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    name: str
    version: str = "1.0"
    total_clips: int = 0
    splits: Dict[str, int] = field(default_factory=dict)
    clips: Dict[str, VideoClipSchema] = field(default_factory=dict)

    def add_clip(self, clip: VideoClipSchema):
        self.clips[clip.clip_id] = clip
        self.total_clips = len(self.clips)

    def get_split(self, split: str):
        return [c for c in self.clips.values() if c.split == split]
