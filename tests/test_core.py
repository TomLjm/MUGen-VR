import os
import tempfile

import torch

from mugen.evaluation.vbench_adapter import VBenchAdapter
from mugen.fusion.fusion_module import HierarchicalConditionFusion
from mugen.fusion.modality_dropout import ModalityDropout
from mugen.generation.retrieve_then_generate import RetrieveThenGenerate
from mugen.retrieval.feature_index import FeatureIndex
from mugen.retrieval.retriever import CrossModalRetriever


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
    assert result.video_frames.shape[0] == 16


def test_vbench_unavailable_is_skipped_not_zero():
    adapter = VBenchAdapter()
    result = adapter.evaluate("missing_dir")
    assert result.details["status"] in {"skipped", "ok"}
    if result.details["status"] == "skipped":
        assert all(v is None for v in result.metrics.values())
