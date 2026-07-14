import importlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_PATH = ROOT / "scripts" / "showcase" / "run_multimodal_showcase.py"
spec = importlib.util.spec_from_file_location("run_multimodal_showcase", SHOWCASE_PATH)
showcase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(showcase)

AudioPromptParser = importlib.import_module("mugen.conditioning.audio_prompt_parser").AudioPromptParser
DEMO_REFERENCES = showcase.DEMO_REFERENCES
DemoEmbeddingFactory = showcase.DemoEmbeddingFactory
ensure_demo_audio = showcase.ensure_demo_audio
layer_weight_summary = showcase.layer_weight_summary
make_prompt_plan = showcase.make_prompt_plan


def test_audio_prompt_parser_extracts_style_from_prompt():
    parser = AudioPromptParser()
    parsed = parser.parse("a dog running on grass with upbeat rhythmic background music")
    assert "dog running on grass" in parsed.content_prompt
    assert "background music" in parsed.audio_prompt.lower()
    assert "rhythmic" in parsed.descriptors
    assert parsed.motion_phrase == "fast rhythmic motion" or parsed.motion_phrase == "steady rhythmic motion"


def test_audio_prompt_parser_defaults_to_neutral_when_missing_audio():
    parser = AudioPromptParser()
    parsed = parser.parse("a dog running on grass")
    assert parsed.audio_prompt == "neutral ambient background sound"
    assert parsed.source == "default"


def test_showcase_audio_embedding_and_prompt_plan():
    with tempfile.TemporaryDirectory() as td:
        audio_path = ensure_demo_audio(Path(td) / "demo.wav")
        factory = DemoEmbeddingFactory(dim=32)
        audio_emb, stats = factory.audio_from_file(str(audio_path))

    assert audio_emb.shape == (1, 32)
    assert stats["duration_sec"] > 0
    assert stats["rms"] > 0

    plan = make_prompt_plan(
        "a dog running on grass",
        "upbeat rhythmic background music",
        DEMO_REFERENCES[:2],
        {"motion_phrase": "steady rhythmic motion", "mood_phrase": "cinematic soundtrack mood", "descriptors": ["rhythmic", "cinematic"]},
    )
    assert "enhanced_prompt" in plan
    assert "audio direction" in plan["enhanced_prompt"]
    assert any("rhythmic" in tag for tag in plan["audio_style_tags"])
    assert "audio latent conditioning" in plan["note"]


def test_showcase_report_mode_writes_expected_files():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "demo"
        cmd = [
            sys.executable,
            str(SHOWCASE_PATH),
            "--mode",
            "report",
            "--output_dir",
            str(out_dir),
            "--prompt",
            "a dog running on grass with upbeat rhythmic background music",
            "--image",
            str(ROOT / "third_party" / "ImageBind" / ".assets" / "dog_image.jpg"),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)
        assert (out_dir / "report.md").exists()
        assert (out_dir / "report.json").exists()
        assert (out_dir / "gating_weights.json").exists()
        assert (out_dir / "retrieval_results.json").exists()


def test_showcase_gating_summary_has_three_modalities():
    weights = torch.tensor([
        [[0.2, 0.5, 0.3]],
        [[0.1, 0.2, 0.7]],
        [[0.6, 0.2, 0.2]],
    ])
    summary = layer_weight_summary(weights, ["text", "image", "audio"])
    assert summary["low_level_visual"]["image"] == 0.5
    assert summary["mid_level_motion_audio"]["audio"] == 0.7
    assert summary["high_level_semantic"]["text"] == 0.6
