import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "generate_project_ablation.py"
SPEC = importlib.util.spec_from_file_location("generate_project_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_project_generation_uses_four_requested_variants():
    assert MODULE.VARIANTS == ("B0", "B1", "B2", "B3")


def test_frozen_anyflow_is_default_and_lora_is_explicit():
    assert not MODULE.variant_uses_lora("B0")
    assert not MODULE.variant_uses_lora("B1")
    assert not MODULE.variant_uses_lora("B2")
    assert not MODULE.variant_uses_lora("B3")
    assert MODULE.variant_uses_lora("B3", enable_token_lora=True)


def test_condition_token_group_selection_preserves_expected_order():
    class Bundle:
        def __init__(self, tokens):
            self.condition_tokens = tokens
            self.metadata = {}

    tokens = torch.arange(15).reshape(1, 15, 1)
    bundles = {"B3": Bundle(tokens.clone())}
    MODULE.select_condition_token_groups(bundles, "no-reference")
    assert bundles["B3"].condition_tokens.flatten().tolist() == [0, 1, 2, *range(7, 15)]

    bundles = {"B3": Bundle(tokens.clone())}
    MODULE.select_condition_token_groups(bundles, "no-temporal")
    assert bundles["B3"].condition_tokens.flatten().tolist() == list(range(7))


def test_prompt_rewrite_remains_a_text_only_historical_baseline():
    prompt = MODULE.rewrite_prompt("a person dancing", [{"caption": "stage lights"}])
    assert prompt.startswith("a person dancing")
    assert "stage lights" in prompt
    assert "audio-reactive" in prompt


def test_reference_gather_preserves_batch_top_k_feature_contract():
    gallery = torch.randn(10, 8)
    indices = torch.tensor([1, 4, 7])

    references = MODULE.gather_reference_embeddings(gallery, indices)

    assert references.shape == (1, 3, 8)


def test_condition_scale_only_changes_token_variants():
    class Bundle:
        def __init__(self, tokens):
            self.condition_tokens = tokens
            self.metadata = {}

    bundles = {
        "B0": Bundle(None),
        "B1": Bundle(None),
        "B2": Bundle(torch.ones(1, 3, 2)),
        "B3": Bundle(torch.ones(1, 7, 2)),
    }
    MODULE.apply_condition_scale(bundles, 0.25)

    assert torch.all(bundles["B2"].condition_tokens == 0.25)
    assert torch.all(bundles["B3"].condition_tokens == 0.25)
    assert bundles["B0"].condition_tokens is None


def test_shifted_condition_row_rotates_without_changing_target_order():
    rows = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]

    assert MODULE.shifted_condition_row(rows, 0, 1)["sample_id"] == "b"
    assert MODULE.shifted_condition_row(rows, 2, 1)["sample_id"] == "a"
    assert MODULE.shifted_condition_row(rows, 1, 0)["sample_id"] == "b"
