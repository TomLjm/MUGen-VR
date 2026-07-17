# Targeted Consistency Evaluation

## Protocol

- Dataset: 40 fixed held-out MSR-VTT clips, isolated by `video_id`.
- Generation: one shared seed (`42`) for the frozen AnyFlow and MUGen-VR outputs;
  condition scale `0.05` was selected on a separate validation split.
- Retrieval: train-only reference gallery with self and near-duplicates excluded.
- Metrics: VBench subject consistency, motion smoothness, and temporal flickering.
- Qualitative review: eight fixed B0/B3 pairs in the Hugging Face Space.

| Metric | Frozen AnyFlow | MUGen-VR | Change |
|---|---:|---:|---:|
| Subject consistency | 0.88328 | **0.88596** | **+0.00268** |
| Motion smoothness | 0.98201 | **0.98248** | **+0.00046** |
| Temporal flickering | 0.96963 | **0.96978** | **+0.00014** |

Subject-consistency error is reduced by `2.29%` relative to the frozen backbone.

## Reproduction

Build the held-out manifest and run retrieval:

```bash
python scripts/eval/build_project_eval_manifest.py --help
python scripts/eval/evaluate_project_retrieval.py --help
```

Generate B0-B3 videos and evaluate decoded MP4 files:

```bash
python scripts/eval/generate_project_ablation.py --help
python scripts/eval/run_evaluation.py --help
python scripts/eval/build_targeted_report.py --help
```

The public machine-readable report is generated directly from the two VBench result
files. Raw evaluation artifacts remain local and are not included in the repository.
