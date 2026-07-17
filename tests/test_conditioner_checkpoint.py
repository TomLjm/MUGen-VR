import json

import torch
from safetensors.torch import save_file

from mugen.generation.checkpoints import INDEX_FILE, load_conditioner_state


def test_loads_training_checkpoint(tmp_path):
    expected = {"weight": torch.arange(4)}
    torch.save(expected, tmp_path / "conditioner.pt")

    loaded = load_conditioner_state(tmp_path)

    assert torch.equal(loaded["weight"], expected["weight"])


def test_loads_sharded_safetensors_release(tmp_path):
    save_file({"first": torch.ones(2)}, tmp_path / "conditioner-00001-of-00002.safetensors")
    save_file({"second": torch.zeros(2)}, tmp_path / "conditioner-00002-of-00002.safetensors")
    (tmp_path / INDEX_FILE).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 16},
                "weight_map": {
                    "first": "conditioner-00001-of-00002.safetensors",
                    "second": "conditioner-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_conditioner_state(tmp_path)

    assert torch.equal(loaded["first"], torch.ones(2))
    assert torch.equal(loaded["second"], torch.zeros(2))
