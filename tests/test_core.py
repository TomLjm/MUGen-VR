import os
import tempfile
import json
from pathlib import Path

import torch

from mugen.evaluation.vbench_adapter import VBenchAdapter
from mugen.common.interfaces import ConditionBundle
from mugen.fusion.fusion_module import HierarchicalConditionFusion
from mugen.fusion.modality_dropout import ModalityDropout
from mugen.generation.retrieve_then_generate import RetrieveThenGenerate
from mugen.generation.conditioner import MultimodalConditioner
from mugen.retrieval.feature_index import FeatureIndex
from mugen.retrieval.retriever import CrossModalRetriever
from mugen.data.feature_store import FeatureShardWriter, load_feature_store
from mugen.data.manifest import validate_split_isolation
from mugen.training.losses import gate_balance_loss, symmetric_info_nce


def test_feature_index_save_load_search_consistency():
    features = torch.eye(4, 4)
    index = FeatureIndex(dim=4)
    index.add(features, [{"clip_id": str(i)} for i in range(4)])
    before = index.search(features[2:3], top_k=2)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.pkl")
        index.save(path)
        loaded = FeatureIndex(dim=4)
        loaded.load(path)
        after = loaded.search(features[2:3], top_k=2)
    assert before[0] == after[0]
    assert before[1] == after[1]


def test_fusion_single_double_triple_modalities():
    model = HierarchicalConditionFusion(dims={"text": 8, "image": 8, "audio": 4}, hidden_dim=8, num_heads=2, modality_dropout=0.0)
    model.eval()
    text = torch.randn(2, 8)
    image = torch.randn(2, 8)
    audio = torch.randn(2, 4)
    assert model({"text": (text, None)}).shape == (2, 8)
    assert model({"text": (text, None), "image": (image, None)}).shape == (2, 8)
    assert model({"text": (text, None), "image": (image, None), "audio": (audio, None)}).shape == (2, 8)


def test_modality_dropout_train_eval_behavior():
    dropout = ModalityDropout(dropout_prob=1.0, num_modalities=2)
    inputs = {"text": torch.ones(1, 4), "image": torch.ones(1, 4)}
    dropout.train()
    out_train = dropout(inputs)
    assert any(v is None for v in out_train.values())
    dropout.eval()
    out_eval = dropout(inputs)
    assert all(v is not None for v in out_eval.values())


def test_retrieve_then_generate_writes_reference_embedding():
    retriever = CrossModalRetriever(dim=4)
    retriever.build_index(torch.eye(4, 4), [{"clip_id": str(i)} for i in range(4)])
    rtg = RetrieveThenGenerate(retriever=retriever, dim=4, allow_dummy=True)
    conditions = {"text": torch.eye(4, 4)[1:2]}
    result = rtg.generate(conditions, allow_dummy=True)
    assert "reference_embedding" in conditions
    assert conditions["reference_embedding"].shape == (1, 4)
    assert conditions["reference_tokens"].shape == (1, 4, 4)
    assert result.video_frames.shape[0] == 16


def test_vbench_unavailable_is_skipped_not_zero():
    adapter = VBenchAdapter()
    result = adapter.evaluate("missing_dir")
    assert result.details["status"] in {"skipped", "ok"}
    if result.details["status"] == "skipped":
        assert all(v is None for v in result.metrics.values())


def test_vbench_adapter_uses_official_custom_input_api(monkeypatch, tmp_path):
    calls = {}

    class FakeVBench:
        def __init__(self, device, full_info_dir, output_path):
            calls["init"] = (str(device), full_info_dir, output_path)
            self.output_path = Path(output_path)

        def evaluate(self, **kwargs):
            calls["evaluate"] = kwargs
            payload = {dimension: [0.8, {}] for dimension in kwargs["dimension_list"]}
            (self.output_path / f"{kwargs['name']}_eval_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    adapter = VBenchAdapter(dims=["subject_consistency"], output_path=tmp_path / "vbench")
    monkeypatch.setattr(adapter, "_load_vbench_class", lambda: (FakeVBench, tmp_path))

    result = adapter.evaluate(tmp_path, required=True)

    assert result.details["status"] == "ok"
    assert result.metrics == {"subject_consistency": 0.8, "vbench_total": 0.8}
    assert calls["evaluate"]["mode"] == "custom_input"
    assert calls["evaluate"]["local"] is True


def test_condition_bundle_appends_tokens():
    bundle = ConditionBundle(
        prompt="test",
        image=object(),
        condition_tokens=torch.ones(2, 7, 16),
    )
    merged = bundle.merged_prompt_embeds(torch.zeros(2, 5, 16))
    assert merged.shape == (2, 12, 16)
    assert torch.all(merged[:, -7:] == 1)


def test_multimodal_conditioner_outputs_anyflow_tokens():
    model = MultimodalConditioner(
        dims={"text": 8, "image": 8, "audio": 4, "reference": 8},
        hidden_dim=8,
        generator_dim=16,
    )
    model.eval()
    bundle = model(
        prompt="a running dog",
        image_condition=object(),
        modality_embeddings={
            "text": torch.randn(2, 8),
            "image": torch.randn(2, 8),
            "audio": torch.randn(2, 4),
        },
        reference_embeddings=torch.randn(2, 3, 8),
        reference_scores=torch.randn(2, 3),
    )
    assert bundle.condition_tokens.shape == (2, 7, 16)
    assert all(bundle.modality_mask.values())


def test_split_isolation_rejects_video_leakage():
    rows = [
        {"video_id": "v1", "split": "train"},
        {"video_id": "v1", "split": "val"},
    ]
    try:
        validate_split_isolation(rows)
    except ValueError as exc:
        assert "both train and val" in str(exc)
    else:
        raise AssertionError("split leakage must be rejected")


def test_real_feature_store_round_trip():
    with tempfile.TemporaryDirectory() as td:
        writer = FeatureShardWriter(td, {"imagebind": "commit-a", "internvideo": "commit-b"}, shard_size=1)
        for index in range(2):
            writer.add(
                {"sample_id": str(index), "feature_source": "real", "media_sha256": f"hash-{index}"},
                {"text": torch.ones(4) * index, "video": torch.ones(3) * index},
            )
        manifest_path = writer.close()
        tensors, records, manifest = load_feature_store(Path(manifest_path).parent)
    assert tensors["text"].shape == (2, 4)
    assert len(records) == 2
    assert manifest["rows"] == 2


def test_real_feature_store_resume_appends_without_overwrite():
    with tempfile.TemporaryDirectory() as td:
        versions = {"imagebind": "commit-a", "internvideo": "commit-b"}
        first = FeatureShardWriter(td, versions, shard_size=1)
        first.add(
            {"sample_id": "a", "feature_source": "real", "media_sha256": "hash-a"},
            {"text": torch.ones(4), "video": torch.ones(3)},
        )
        first.close()
        resumed = FeatureShardWriter(td, versions, shard_size=1, resume=True)
        assert resumed.existing_sample_ids == {"a"}
        resumed.add(
            {"sample_id": "b", "feature_source": "real", "media_sha256": "hash-b"},
            {"text": torch.zeros(4), "video": torch.zeros(3)},
        )
        resumed.close()
        _, records, manifest = load_feature_store(td)
    assert [row["sample_id"] for row in records] == ["a", "b"]
    assert manifest["rows"] == 2


def test_symmetric_info_nce_prefers_matching_pairs():
    target = torch.eye(4)
    good, _ = symmetric_info_nce(target, target)
    bad, _ = symmetric_info_nce(target.flip(0), target)
    assert good < bad


def test_gate_balance_penalizes_collapsed_weights():
    balanced = torch.full((3, 2, 4), 0.25)
    collapsed = torch.zeros(3, 2, 4)
    collapsed[..., 0] = 1.0
    assert gate_balance_loss(balanced) < gate_balance_loss(collapsed)
