import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "demo" / "gradio_app.py"
SPEC = importlib.util.spec_from_file_location("gradio_app", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_demo_applies_validation_selected_condition_scale():
    class Bundle:
        condition_tokens = torch.ones(1, 7, 4)
        metadata = {}

    bundle = MODULE.apply_condition_scale(Bundle(), 0.1)

    assert torch.allclose(bundle.condition_tokens, torch.full((1, 7, 4), 0.1))
    assert bundle.metadata["condition_scale"] == 0.1
