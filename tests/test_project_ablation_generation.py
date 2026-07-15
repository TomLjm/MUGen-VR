import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval" / "generate_project_ablation.py"
SPEC = importlib.util.spec_from_file_location("generate_project_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_project_generation_uses_four_requested_variants():
    assert MODULE.VARIANTS == ("B0", "B1", "B2", "B3")


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
