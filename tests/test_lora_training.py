import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "train" / "train_lora.py"
SPEC = importlib.util.spec_from_file_location("train_lora", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_latent_chunk_partition_preserves_first_frame_prefix():
    assert MODULE.latent_chunk_partition(7, 2) == [1, 2, 2, 2]
    assert sum(MODULE.latent_chunk_partition(13, 4)) == 13


def test_flow_matching_endpoints_and_target():
    clean = torch.ones(2, 3, 1, 1, 1)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([0, 1000])

    noisy, target = MODULE.flow_matching_sample(clean, timesteps, 1000, noise=noise)

    assert torch.equal(noisy[0], clean[0])
    assert torch.equal(noisy[1], noise[1])
    assert torch.equal(target, noise - clean)
