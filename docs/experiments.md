# Experiment Status

This is a resume and interview project, not a paper submission. Release metrics must
still come from real held-out media, but the evaluation is intentionally compact.
Smoke artifacts, dummy videos, and `skipped` metrics are not accepted as evidence.

## Verified engineering smoke

| Component | Result |
|---|---|
| ImageBind text/image | `(1, 1024)`, finite, L2 norm 1.0 |
| InternVideo2 video | `(1, 768)`, finite, L2 norm 1.0 |
| AnyFlow 25f 256x448 1-step | 4.12 s, 6.07 generated FPS, 15.47 GiB peak VRAM |
| Real Fusion training | 8 real samples, 10 optimizer steps, best/latest checkpoint written |
| AnyFlow LoRA training | real VAE latents and condition tokens, 1 optimizer step, lightweight checkpoint |

These numbers verify the pipeline only and are not claims of model quality.

## Four required variants

| Variant | Definition |
|---|---|
| B0 | AnyFlow image + original prompt |
| B1 | historical prompt rewrite prototype |
| B2 | Fusion tokens without audio or reference |
| B3 | full Fusion + audio + reference |

Use 30-50 fixed held-out samples and one shared generation seed. Report key VBench
dimensions, retrieval R@1/5/10 and MRR, onset/optical-flow correlation, latency, FPS,
and peak VRAM. Publish 6-10 side-by-side qualitative cases including failures.

`scripts/eval/ablation_study.py` checks sample coverage, seed consistency, metric
presence, and case count. It also emits paired-bootstrap intervals as advisory context;
confidence intervals are not a completion gate.
