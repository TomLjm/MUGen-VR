#!/usr/bin/env python3
"""Stage a whitelist-only Hugging Face model upload without upstream or optimizer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


WEIGHT_FILES = ("conditioner.pt", "pytorch_lora_weights.safetensors")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_model(checkpoint, model_card, evaluation, output, condition_scale=0.1):
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    for filename in WEIGHT_FILES:
        source = checkpoint / filename
        if not source.is_file():
            raise FileNotFoundError(f"required MUGen weight is missing: {source}")
        shutil.copy2(source, output / filename)
    shutil.copy2(model_card, output / "README.md")
    shutil.copy2(evaluation, output / "evaluation.json")
    config = {
        "schema_version": 1,
        "base_model": state["config"]["model"]["base"],
        "checkpoint_step": state["step"],
        "condition_scale": float(condition_scale),
        "reference_tokens": state["config"]["model"]["reference_tokens"],
        "reference_top_k": state["config"]["data"]["reference_top_k"],
        "lora": state["config"]["model"]["lora"],
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
        "schema_version": 1,
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
    parser.add_argument("--condition-scale", type=float, default=0.1)
    args = parser.parse_args()
    manifest = package_model(
        args.checkpoint,
        args.model_card,
        args.evaluation,
        args.output,
        args.condition_scale,
    )
    print(json.dumps({"output": args.output, "files": manifest["files"]}))


if __name__ == "__main__":
    main()
