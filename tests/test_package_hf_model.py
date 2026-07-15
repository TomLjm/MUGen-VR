import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "release" / "package_hf_model.py"
SPEC = importlib.util.spec_from_file_location("package_hf_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hf_package_is_whitelist_only(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "conditioner.pt").write_bytes(b"conditioner")
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

    output = tmp_path / "release"
    manifest = MODULE.package_model(checkpoint, model_card, evaluation, output)

    assert not (output / "optimizer.pt").exists()
    assert {item["name"] for item in manifest["files"]} == {
        "README.md",
        "conditioner.pt",
        "evaluation.json",
        "mugen_config.json",
        "pytorch_lora_weights.safetensors",
    }
    assert json.loads((output / "mugen_config.json").read_text())["condition_scale"] == 0.1
