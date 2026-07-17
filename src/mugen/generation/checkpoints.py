"""Load project-owned conditioner weights from training or HF release formats."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


INDEX_FILE = "conditioner.safetensors.index.json"


def load_conditioner_state(path, device="cpu"):
    path = Path(path)
    checkpoint = path / "conditioner.pt" if path.is_dir() else path
    if checkpoint.is_file():
        return torch.load(checkpoint, map_location=device, weights_only=True)
    root = path if path.is_dir() else path.parent
    index_path = root / INDEX_FILE
    if not index_path.is_file():
        raise FileNotFoundError(f"conditioner weights not found under: {path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    state = {}
    for filename in sorted(set(index["weight_map"].values())):
        state.update(load_file(str(root / filename), device=str(device)))
    missing = set(index["weight_map"]) - set(state)
    if missing:
        raise ValueError(f"conditioner shards are missing tensors: {sorted(missing)}")
    return state
