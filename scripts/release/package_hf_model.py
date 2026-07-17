#!/usr/bin/env python3
"""Stage a whitelist-only Hugging Face model upload without upstream or optimizer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file


CONDITIONER_INDEX = "conditioner.safetensors.index.json"
LORA_FILE = "pytorch_lora_weights.safetensors"


def save_conditioner_shards(source, output, max_shard_size="8MB"):
    state = torch.load(source, map_location="cpu", weights_only=True)
    split = split_torch_state_dict_into_shards(
        state,
        filename_pattern="conditioner{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    for stale in output.glob("conditioner*.safetensors"):
        stale.unlink()
    for filename, tensor_names in split.filename_to_tensors.items():
        save_file(
            {name: state[name].contiguous() for name in tensor_names},
            output / filename,
        )
    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (output / CONDITIONER_INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    stale_pt = output / "conditioner.pt"
    if stale_pt.exists():
        stale_pt.unlink()
    return sorted(split.filename_to_tensors)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_model(
    checkpoint,
    model_card,
    evaluation,
    output,
    condition_scale=0.05,
    include_lora=False,
    condition_token_count=15,
    temporal_audio_tokens=8,
    license_path=None,
    max_shard_size="8MB",
):
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    source = checkpoint / "conditioner.pt"
    if not source.is_file():
        raise FileNotFoundError(f"required MUGen weight is missing: {source}")
    save_conditioner_shards(source, output, max_shard_size=max_shard_size)
    if include_lora:
        shutil.copy2(checkpoint / LORA_FILE, output / LORA_FILE)
    stale_lora = output / LORA_FILE
    if not include_lora and stale_lora.exists():
        stale_lora.unlink()
    shutil.copy2(model_card, output / "README.md")
    shutil.copy2(evaluation, output / "evaluation.json")
    if license_path is not None:
        shutil.copy2(license_path, output / "LICENSE")
    config = {
        "schema_version": 2,
        "base_model": state["config"]["model"]["base"],
        "checkpoint_step": state["step"],
        "condition_scale": float(condition_scale),
        "condition_token_count": int(condition_token_count),
        "temporal_audio_tokens": int(temporal_audio_tokens),
        "use_lora": bool(include_lora),
        "reference_tokens": state["config"]["model"]["reference_tokens"],
        "reference_top_k": state["config"]["data"]["reference_top_k"],
        "lora": state["config"]["model"]["lora"] if include_lora else None,
        "generation": {
            key: state["config"]["data"][key]
            for key in ("num_frames", "height", "width", "latent_chunk_size")
        },
        "encoder_versions": state["feature_manifest"]["encoder_versions"],
        "feature_rows": state["feature_manifest"]["rows"],
        "training_metrics": state["metrics"],
        "upstream_weights_included": False,
        "optimizer_state_included": False,
    }
    (output / "mugen_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 2,
        "files": files,
        "forbidden_files_absent": ["optimizer.pt", "AnyFlow base weights", "dataset media"],
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Stage whitelist-only MUGen HF model files")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    parser.add_argument("--evaluation", default="reports/project-final/final-report.json")
    parser.add_argument("--output", default="release/hf_model")
    parser.add_argument("--condition-scale", type=float, default=0.05)
    parser.add_argument("--include-lora", action="store_true")
    parser.add_argument("--condition-token-count", type=int, default=15)
    parser.add_argument("--temporal-audio-tokens", type=int, default=8)
    parser.add_argument("--license", default="LICENSE")
    parser.add_argument("--max-shard-size", default="8MB")
    args = parser.parse_args()
    manifest = package_model(
        args.checkpoint,
        args.model_card,
        args.evaluation,
        args.output,
        args.condition_scale,
        args.include_lora,
        args.condition_token_count,
        args.temporal_audio_tokens,
        args.license,
        args.max_shard_size,
    )
    print(json.dumps({"output": args.output, "files": manifest["files"]}))


if __name__ == "__main__":
    main()
