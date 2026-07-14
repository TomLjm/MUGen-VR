"""InternVideo2 Stage-2 video encoder without modifying upstream source files."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoderWrapper
from ..common.interfaces import EncoderOutput


class _AttrDict(dict):
    def __init__(self, values=None):
        super().__init__()
        for key, value in (values or {}).items():
            converted = _AttrDict(value) if isinstance(value, dict) else value
            self[key] = converted
            setattr(self, key, converted)


class _OptionalFlashFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    prefixes = ("flash_attn", "flash_attention_class")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self.prefixes):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        class MissingOptional:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                raise RuntimeError("flash-attn path was invoked although use_flash_attn=False")

            def __getattr__(self, name):
                return self

        for name in [
            "FusedMLP", "FlashAttention", "RotaryEmbedding", "DropoutAddRMSNorm", "Block",
            "pad_input", "unpad_input", "flash_attn_varlen_qkvpacked_func",
        ]:
            setattr(module, name, MissingOptional)
        return module

    def exec_module(self, module):
        return None


class InternVideoEncoder(BaseEncoderWrapper):
    def __init__(
        self,
        checkpoint_path: str | Path,
        repository: str | Path = "third_party/InternVideo",
        num_frames: int = 4,
        image_size: int = 224,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"InternVideo checkpoint not found: {self.checkpoint_path}")
        root = Path(repository).resolve() / "InternVideo2" / "multi_modality"
        if not root.is_dir():
            raise FileNotFoundError(f"InternVideo repository not found: {root}")
        self.num_frames = num_frames
        self.image_size = image_size
        self.version = "InternVideo@3965eef:Stage2-1B-224p-f4"
        self.model = self._load_model(root)

    def _load_model(self, root: Path):
        import transformers.modeling_utils as modeling_utils
        from transformers.pytorch_utils import apply_chunking_to_forward

        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
        if not any(isinstance(value, _OptionalFlashFinder) for value in sys.meta_path):
            sys.meta_path.insert(0, _OptionalFlashFinder())
        package = types.ModuleType("models")
        package.__path__ = [str(root / "models")]
        sys.modules.setdefault("models", package)
        sys.path.insert(0, str(root))
        from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
        from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new

        config = _AttrDict(
            {
                "vision_encoder": {
                    "name": "pretrain_internvideo2_1b_patch14_224",
                    "img_size": self.image_size,
                    "num_frames": self.num_frames,
                    "tubelet_size": 1,
                    "patch_size": 14,
                    "d_model": 1408,
                    "clip_embed_dim": 768,
                    "clip_teacher_embed_dim": 3200,
                    "clip_teacher_final_dim": 768,
                    "clip_norm_type": "l2",
                    "clip_return_layer": 6,
                    "clip_student_return_interval": 1,
                    "use_checkpoint": False,
                    "checkpoint_num": 0,
                    "use_flash_attn": False,
                    "use_fused_rmsnorm": False,
                    "use_fused_mlp": False,
                    "clip_teacher": None,
                    "clip_input_resolution": self.image_size,
                    "clip_teacher_return_interval": 1,
                    "video_mask_type": "random",
                    "video_mask_ratio": 0.8,
                    "image_mask_type": "random",
                    "image_mask_ratio": 0.5,
                    "sep_image_video_pos_embed": True,
                    "keep_temporal": False,
                    "only_mask": True,
                    "pretrained": None,
                }
            }
        )
        vision_encoder = pretrain_internvideo2_1b_patch14_224(config)
        holder = nn.Module()
        holder.vision_encoder = vision_encoder
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        state = payload.get("model", payload.get("module", payload))
        interpolate_pos_embed_internvideo2_new(state, vision_encoder, orig_t_size=self.num_frames)
        result = holder.load_state_dict(state, strict=False)
        matched = len(state) - len(result.unexpected_keys)
        if matched <= 0:
            raise RuntimeError("InternVideo checkpoint did not match the vision encoder")
        return vision_encoder.to(self.device).eval()

    def _frames(self, path: str | Path) -> torch.Tensor:
        decoded = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                decoded.append(frame.to_rgb().to_ndarray())
        if not decoded:
            raise ValueError(f"no video frames decoded: {path}")
        indices = np.linspace(0, len(decoded) - 1, self.num_frames).astype(int)
        frames = []
        for index in indices:
            frame = torch.from_numpy(decoded[index]).permute(2, 0, 1).float().div(255.0)
            frame = F.interpolate(
                frame.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).squeeze(0)
            frames.append(frame)
        video = torch.stack(frames, dim=1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return ((video - mean) / std).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_video(self, video_path, **kwargs):
        _, feature, _, _ = self.model(self._frames(video_path), None, False)
        feature = F.normalize(feature.float(), dim=-1)
        return EncoderOutput(
            embedding=feature,
            metadata={"model": self.version, "video_path": str(video_path)},
        )

    def encode_text(self, text, **kwargs):
        raise NotImplementedError

    def encode_image(self, image, **kwargs):
        raise NotImplementedError

    def encode_audio(self, audio_path, **kwargs):
        raise NotImplementedError
