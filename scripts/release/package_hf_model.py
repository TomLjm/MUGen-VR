#!/usr/bin/env python3
"""Stage a whitelist-only Hugging Face model upload without upstream or optimizer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


CONDITIONER_FILE = "conditioner.pt"
LORA_FILE = "pytorch_lora_weights.safetensors"


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
):
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    weight_files = [CONDITIONER_FILE]
    if include_lora:
        weight_files.append(LORA_FILE)
    for filename in weight_files:
        source = checkpoint / filename
        if not source.is_file():
            raise FileNotFoundError(f"required MUGen weight is missing: {source}")
        shutil.copy2(source, output / filename)
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
    )
    print(json.dumps({"output": args.output, "files": manifest["files"]}))


if __name__ == "__main__":
    main()
