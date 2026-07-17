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


def test_reference_retrieval_is_restricted_to_explicit_train_gallery():
    features = torch.eye(4)
    references, _ = MODULE.retrieve_reference(
        features, index=3, top_k=3, gallery_indices=torch.tensor([0, 1, 2])
    )

    assert references.shape == (3, 4)
    assert not torch.any(torch.all(references == features[3], dim=-1))


def test_counterfactual_sources_never_match_targets():
    generator = torch.Generator().manual_seed(7)
    targets = [0, 1, 2, 3]

    sources = MODULE.sample_mismatched_indices(targets, torch.arange(4), generator)

    assert len(sources) == len(targets)
    assert all(source != target for source, target in zip(sources, targets))


def test_condition_scale_sampling_stays_in_training_range():
    generator = torch.Generator().manual_seed(3)
    scales = [MODULE.sample_condition_scale(0.03, 0.15, generator) for _ in range(20)]

    assert all(0.03 <= scale <= 0.15 for scale in scales)


def test_condition_budget_normalization_preserves_seven_token_reference_scale():
    seven = MODULE.normalize_condition_token_budget(torch.ones(1, 7, 2), 0.05)
    fifteen = MODULE.normalize_condition_token_budget(torch.ones(1, 15, 2), 0.05)

    assert torch.allclose(seven, torch.full_like(seven, 0.05))
    assert torch.allclose(fifteen, torch.full_like(fifteen, 0.05 * 7 / 15))


def test_cuda_rng_restore_truncates_multi_gpu_checkpoint_for_single_gpu(monkeypatch):
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))

    MODULE.restore_cuda_rng_states(["gpu0", "gpu1", "gpu2", "gpu3"])

    assert restored == ["gpu0"]


def test_conditioner_checkpoint_migration_preserves_old_type_embeddings(tmp_path):
    old = MODULE.MultimodalConditioner(
        dims={"text": 8, "image": 8, "audio": 4, "reference": 8},
        hidden_dim=8,
        generator_dim=16,
    )
    upgraded = MODULE.MultimodalConditioner(
        dims={"text": 8, "image": 8, "audio": 4, "reference": 8},
        hidden_dim=8,
        generator_dim=16,
        temporal_audio_dim=8 * 6,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(old.state_dict(), checkpoint / "conditioner.pt")

    missing = MODULE.load_migrated_conditioner_state(upgraded, checkpoint)

    assert torch.equal(upgraded.type_embeddings[:7], old.type_embeddings)
    assert any(name.startswith("temporal_audio_projector.") for name in missing)


def test_sync_selection_is_split_safe_and_thresholded():
    records = [
        {"sample_id": "train-low", "split": "train"},
        {"sample_id": "train-high", "split": "train"},
        {"sample_id": "val-high", "split": "val"},
    ]
    scores = {"train-low": 0.09, "train-high": 0.11, "val-high": 0.5}

    selected = MODULE.select_sync_indices(records, "train", scores, 0.1)

    assert selected.tolist() == [1]


def test_rhythm_alignment_prefers_matching_motion_profile():
    temporal = torch.zeros(1, 8, 6)
    temporal[0, :, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    profile = torch.nn.functional.interpolate(
        temporal[..., -1].unsqueeze(1), size=7, mode="linear", align_corners=False
    ).squeeze(1)
    matching = torch.cat([torch.zeros(1, 1), profile.cumsum(dim=1)], dim=1).view(1, 8, 1, 1, 1)
    reversed_profile = profile.flip(1)
    mismatched = torch.cat(
        [torch.zeros(1, 1), reversed_profile.cumsum(dim=1)], dim=1
    ).view(1, 8, 1, 1, 1)

    good = MODULE.rhythm_alignment_loss(matching, temporal)
    bad = MODULE.rhythm_alignment_loss(mismatched, temporal)

    assert good < bad


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
