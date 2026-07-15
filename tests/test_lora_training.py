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


def test_decode_video_contract_is_channel_first_before_batching(monkeypatch):
    decoded = torch.zeros(3, 25, 16, 24)
    monkeypatch.setattr(MODULE, "decode_video", lambda *args, **kwargs: decoded)

    class Pipeline:
        class Transformer:
            dtype = torch.float32

        transformer = Transformer()

        def encode_video(self, videos, height, width):
            assert videos.shape == (1, 25, 3, 16, 24)
            return torch.zeros(1, 7, 16, 2, 3)

    result = MODULE.encode_latents(
        Pipeline(), [{"video_path": "unused"}], [0], 25, 16, 24, torch.device("cpu")
    )
    assert result.shape == (1, 7, 16, 2, 3)


def test_encode_latents_accepts_ddp_style_transformer_wrapper(monkeypatch):
    monkeypatch.setattr(MODULE, "decode_video", lambda *args, **kwargs: torch.zeros(3, 5, 8, 8))

    class Inner:
        dtype = torch.bfloat16

    class Wrapped:
        module = Inner()

    class Pipeline:
        transformer = Wrapped()

        def encode_video(self, videos, height, width):
            return torch.zeros(1, 2, 16, 1, 1)

    result = MODULE.encode_latents(Pipeline(), [{"video_path": "unused"}], [0], 5, 8, 8, "cpu")
    assert result.dtype == torch.bfloat16


def test_training_script_does_not_save_frozen_accelerator_model_state():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "accelerator.save_state" not in source
    assert 'checkpoint_dir / "optimizer.pt"' in source
    assert 'weight_name="pytorch_lora_weights.safetensors"' in source
    assert 'adapter_name="default"' in source
    assert "DistributedDataParallelKwargs(find_unused_parameters=True)" in source
