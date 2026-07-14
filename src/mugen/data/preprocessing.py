"""Video preprocessing."""
import os
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import av

from ..common.utils import ensure_dir


class VideoPreprocessor:
    def __init__(self, target_fps=8, target_resolution=256):
        self.target_fps = target_fps
        self.target_resolution = target_resolution

    def clip_video(self, video_path, output_dir, clip_duration=8, stride=4):
        ensure_dir(output_dir)
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        fps = float(container.streams.video[0].average_rate)
        container.close()
        total_duration = total_frames / fps
        clip_paths = []
        base_name = Path(video_path).stem
        start = 0
        clip_idx = 0
        while start < total_duration:
            end = min(start + clip_duration, total_duration)
            if end - start < 2:
                break
            output_path = os.path.join(output_dir, f"{base_name}_clip{clip_idx:04d}.mp4")
            cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-to", str(end),
                   "-vf", f"fps={self.target_fps},scale={self.target_resolution}:-1",
                   "-c:v", "libx264", "-an", output_path]
            subprocess.run(cmd, capture_output=True)
            clip_paths.append(output_path)
            start += stride
            clip_idx += 1
        return clip_paths

    def resize_video(self, video):
        T, C, H, W = video.shape
        if max(H, W) <= self.target_resolution:
            return video
        scale = self.target_resolution / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        return torch.nn.functional.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)


class AudioExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, video_path, output_path):
        ensure_dir(os.path.dirname(output_path))
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", str(self.sample_rate), "-ac", "1", output_path]
        subprocess.run(cmd, capture_output=True)
        return output_path


class KeyFrameExtractor:
    def __init__(self, method="uniform", num_frames=8):
        self.method = method
        self.num_frames = num_frames

    def extract(self, video_path, output_dir):
        ensure_dir(output_dir)
        base_name = Path(video_path).stem
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        container.close()
        if self.method == "uniform":
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = self._scene_detect(video_path)
        frame_paths = []
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                path = os.path.join(output_dir, f"{base_name}_frame{i:04d}.jpg")
                img = frame.to_rgb().to_ndarray()
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                frame_paths.append(path)
        container.close()
        return frame_paths

    def _scene_detect(self, video_path):
        container = av.open(video_path)
        hist_diff = []
        prev_hist = None
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                hist_diff.append(diff)
            prev_hist = hist
        container.close()
        if not hist_diff:
            return np.linspace(0, len(hist_diff), self.num_frames, dtype=int)
        threshold = np.mean(hist_diff) + np.std(hist_diff)
        scene_changes = [i for i, d in enumerate(hist_diff) if d > threshold]
        if len(scene_changes) < self.num_frames:
            scene_changes = np.linspace(0, len(hist_diff), self.num_frames, dtype=int).tolist()
        return sorted(scene_changes[:self.num_frames])
