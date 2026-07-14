# Experiments

The project should report results for four variants:

| Variant | Fusion | Retrieval | Modality Dropout |
| --- | --- | --- | --- |
| no fusion | average or concat | optional | no |
| no retrieval | HierarchicalConditionFusion | no | yes |
| no dropout | HierarchicalConditionFusion | yes | no |
| full MUGen-VR | HierarchicalConditionFusion | yes | yes |

## Required Metrics

- Retrieval: Recall@1/5/10 and MRR.
- Alignment: cosine similarity between generated/reference condition features.
- Generation: VBench dimensions when installed, otherwise explicit `skipped` fields.
- Robustness: score drop under text-only, image-only, text+image, and text+image+reference inputs.

## Required Qualitative Cases

- Same prompt with and without retrieved reference.
- Missing modality comparison.
- Gating weights visualization for text/image/audio/reference.
- SVD baseline vs AnyFlow baseline vs MUGen-VR enhanced condition.
