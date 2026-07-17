import importlib.util
import json
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "release" / "package_hf_model.py"
SPEC = importlib.util.spec_from_file_location("package_hf_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hf_package_is_whitelist_only(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save({"conditioner.weight": torch.ones(4)}, checkpoint / "conditioner.pt")
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"lora")
    (checkpoint / "optimizer.pt").write_bytes(b"must-not-copy")
    state = {
        "step": 300,
        "config": {
            "model": {"base": "upstream/base", "reference_tokens": 4, "lora": {"rank": 16}},
            "data": {"reference_top_k": 3, "num_frames": 25, "height": 256, "width": 448, "latent_chunk_size": 2},
        },
        "feature_manifest": {"encoder_versions": {"imagebind": "v1"}, "rows": 6000},
        "metrics": {"validation_loss": 0.4},
    }
    (checkpoint / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    model_card = tmp_path / "MODEL_CARD.md"
    evaluation = tmp_path / "evaluation.json"
    model_card.write_text("model card", encoding="utf-8")
    evaluation.write_text("{}", encoding="utf-8")
    license_path = tmp_path / "LICENSE"
    license_path.write_text("MIT", encoding="utf-8")

    output = tmp_path / "release"
    manifest = MODULE.package_model(
        checkpoint, model_card, evaluation, output, license_path=license_path
    )

    assert not (output / "optimizer.pt").exists()
    assert {item["name"] for item in manifest["files"]} == {
        "README.md",
        "LICENSE",
        "conditioner.safetensors",
        "conditioner.safetensors.index.json",
        "evaluation.json",
        "mugen_config.json",
    }
    config = json.loads((output / "mugen_config.json").read_text())
    assert config["condition_scale"] == 0.05
    assert config["condition_token_count"] == 15
    assert config["temporal_audio_tokens"] == 8
    assert config["use_lora"] is False


def test_hf_package_can_include_legacy_lora(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save({"conditioner.weight": torch.ones(4)}, checkpoint / "conditioner.pt")
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"lora")
    state = {
        "step": 1,
        "config": {
            "model": {"base": "upstream/base", "reference_tokens": 4, "lora": {"rank": 16}},
            "data": {"reference_top_k": 3, "num_frames": 25, "height": 256, "width": 448, "latent_chunk_size": 2},
        },
        "feature_manifest": {"encoder_versions": {}, "rows": 1},
        "metrics": {},
    }
    (checkpoint / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    model_card = tmp_path / "MODEL_CARD.md"
    evaluation = tmp_path / "evaluation.json"
    model_card.write_text("model card", encoding="utf-8")
    evaluation.write_text("{}", encoding="utf-8")

    output = tmp_path / "release"
    MODULE.package_model(
        checkpoint, model_card, evaluation, output, include_lora=True
    )

    assert (output / "pytorch_lora_weights.safetensors").is_file()
    assert json.loads((output / "mugen_config.json").read_text())["use_lora"] is True
