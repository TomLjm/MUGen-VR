"""Versioned sharded storage for real multimodal features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file


class FeatureShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        encoder_versions: Dict[str, str],
        shard_size: int = 256,
        resume: bool = False,
    ):
        if not encoder_versions or any(not value for value in encoder_versions.values()):
            raise ValueError("all encoder versions must be explicit")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_versions = dict(encoder_versions)
        self.shard_size = shard_size
        self._tensors: Dict[str, List[torch.Tensor]] = {}
        self._records: List[Dict] = []
        self._shards: List[Dict] = []
        self.existing_sample_ids = set()
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(f"feature store already exists: {self.output_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("feature_source") != "real":
                raise ValueError("cannot resume a non-real feature store")
            if manifest.get("encoder_versions") != self.encoder_versions:
                raise ValueError("encoder versions changed; start a new feature store")
            self._shards = list(manifest["shards"])
            for shard in self._shards:
                with (self.output_dir / shard["metadata"]).open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            self.existing_sample_ids.add(json.loads(line)["sample_id"])

    def add(self, record: Dict, tensors: Dict[str, torch.Tensor]) -> None:
        if record.get("feature_source") != "real":
            raise ValueError("release feature stores only accept feature_source='real'")
        if not record.get("media_sha256"):
            raise ValueError("media_sha256 is required")
        if record.get("sample_id") in self.existing_sample_ids:
            raise ValueError(f"duplicate sample_id: {record['sample_id']}")
        normalized = {name: tensor.detach().cpu().float().reshape(-1) for name, tensor in tensors.items()}
        if not normalized:
            raise ValueError("at least one feature tensor is required")
        if self._records and set(normalized) != set(self._tensors):
            raise ValueError("all records in a shard must contain identical feature names")
        for name, tensor in normalized.items():
            self._tensors.setdefault(name, []).append(tensor)
        self._records.append(dict(record))
        self.existing_sample_ids.add(record["sample_id"])
        if len(self._records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        shard_id = len(self._shards)
        stem = f"shard-{shard_id:05d}"
        tensor_path = self.output_dir / f"{stem}.safetensors"
        metadata_path = self.output_dir / f"{stem}.jsonl"
        stacked = {name: torch.stack(values) for name, values in self._tensors.items()}
        save_file(stacked, str(tensor_path), metadata={"feature_source": "real"})
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in self._records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._shards.append(
            {"tensors": tensor_path.name, "metadata": metadata_path.name, "rows": len(self._records)}
        )
        self._tensors = {}
        self._records = []

    def close(self) -> Path:
        self.flush()
        manifest = {
            "schema_version": 1,
            "feature_source": "real",
            "encoder_versions": self.encoder_versions,
            "rows": sum(shard["rows"] for shard in self._shards),
            "shards": self._shards,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def load_feature_store(path: str | Path) -> tuple[Dict[str, torch.Tensor], List[Dict], Dict]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_source") != "real":
        raise ValueError("synthetic feature stores are not valid release inputs")
    tensors: Dict[str, List[torch.Tensor]] = {}
    records = []
    for shard in manifest["shards"]:
        loaded = load_file(str(root / shard["tensors"]))
        for name, values in loaded.items():
            tensors.setdefault(name, []).append(values)
        with (root / shard["metadata"]).open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    merged = {name: torch.cat(parts) for name, parts in tensors.items()}
    if len(records) != manifest["rows"] or any(value.shape[0] != len(records) for value in merged.values()):
        raise ValueError("feature shard row counts do not match the manifest")
    return merged, records, manifest
