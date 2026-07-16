# Evaluation

## Protocol

- Dataset: 40 fixed held-out MSR-VTT clips, isolated by `video_id`.
- Generation: one shared seed (`42`) for B0-B3 and condition scale `0.1`.
- Retrieval: train-only reference gallery with self and near-duplicates excluded.
- Metrics: VBench, retrieval R@1/5/10 and MRR, audio/optical-flow correlation,
  latency, throughput, and peak VRAM.
- Qualitative review: eight fixed B0/B3 pairs in the Hugging Face Space.

| Variant | Definition | VBench | Retrieval MRR | Audio-flow | Latency |
|---|---|---:|---:|---:|---:|
| B0 | AnyFlow image + original prompt | 0.7600 | n/a | 0.0179 | 4.42 s |
| B1 | Prompt rewrite prototype | 0.7511 | n/a | 0.0363 | 4.26 s |
| B2 | Fusion tokens without audio/reference | 0.7596 | 1.0000 | -0.0012 | 4.28 s |
| B3 | Full fusion + audio + reference | 0.7570 | 0.9813 | 0.0046 | 4.31 s |

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
python scripts/eval/evaluate_audio_control.py --help
python scripts/eval/merge_project_metrics.py --help
python scripts/eval/ablation_study.py --help
```

The aggregate machine-readable report is stored at
[`reports/project-final/final-report.json`](../reports/project-final/final-report.json).
Paired bootstrap intervals are available as diagnostics and are not used to claim an
improvement for the released checkpoint.
