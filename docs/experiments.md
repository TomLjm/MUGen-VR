# Experiment Status

Release metrics must come from real held-out media. Smoke artifacts, dummy videos,
missing VBench results, and `skipped` metrics are not accepted as evidence.

## 2026-07-14 encoder smoke

| Encoder | Input | Output | Result |
|---|---|---:|---|
| ImageBind | real caption + decoded JPEG keyframe | `(1, 1024)` each | finite, L2 norm 1.0 |
| InternVideo2 Stage2 1B | decoded MSR-VTT MP4 | `(1, 768)` | finite, L2 norm 1.0 |

This verifies the backbone wrappers only. It is not a retrieval or generation result.

## Required variants

| Variant | Definition |
|---|---|
| B0 | AnyFlow image + original prompt |
| B1 | historical prompt rewrite prototype |
| B2 | real retrieval + reference video prefix |
| B3 | fusion tokens without reference |
| B4 | fusion + reference adapter without audio |
| B5 | full text + image + audio + reference with modality dropout |

Each variant uses the same held-out samples, three training seeds, and three generation
seeds per sample. Reports include retrieval R@1/5/10 and MRR; VBench total, subject
consistency, motion smoothness, and temporal consistency; audio-video alignment and
onset/flow correlation; latency, generated FPS, and peak VRAM.

## Release gate

`scripts/eval/ablation_study.py` consumes per-sample JSONL metrics. B5 must beat the
best B0-B4 baseline on retrieval MRR, VBench total, and the primary audio-control
metric with the 95% paired-bootstrap confidence interval strictly above zero.
Subject and temporal consistency may not significantly regress. Four-step condition
overhead must remain within 10% of B0 and peak VRAM must fit a 24 GB GPU.
