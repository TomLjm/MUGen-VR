---
library_name: diffusers
license: other
license_name: mit-code-with-noncommercial-upstream-weight-restrictions
base_model: nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers
pipeline_tag: image-to-video
tags:
  - multimodal
  - video-generation
  - lora
  - retrieval
---

# MUGen-VR AnyFlow Conditioner

MUGen-VR adds a trainable multimodal condition path to AnyFlow-FAR. ImageBind encodes
text, image, and audio inputs; InternVideo2 encodes videos and retrieved references.
Three fusion tokens and four reference tokens are projected into the 4096-dimensional
UMT5 space and appended to AnyFlow `prompt_embeds`.

## Files

The release contains only project-owned lightweight artifacts:

- Fusion, Reference Adapter, condition projector, and modality/type embeddings.
- AnyFlow cross-attention LoRA weights.
- Sanitized training configuration, evaluation metadata, and checksums.

It excludes AnyFlow, ImageBind, and InternVideo2 weights, MSR-VTT media, cached features,
and optimizer state. These dependencies must be obtained under their upstream terms.

## Evaluation

The fixed evaluation uses 40 held-out samples with generation seed 42. Condition scale
`0.1` was selected on eight validation samples.

| Variant | VBench | Retrieval MRR | Audio-flow | Latency |
|---|---:|---:|---:|---:|
| B0 | 0.7600 | n/a | 0.0179 | 4.42 s |
| B1 | 0.7511 | n/a | 0.0363 | 4.26 s |
| B2 | 0.7596 | 1.0000 | -0.0012 | 4.28 s |
| B3 | 0.7570 | 0.9813 | 0.0046 | 4.31 s |

B0 remains the strongest aggregate VBench baseline. B2 preserves quality within
`0.0004`; adding audio and reference tokens in B3 does not improve the reported metrics.

## Limitations

- AnyFlow is restricted to non-commercial use under NVIDIA NSCLv1.
- ImageBind is governed by non-commercial research terms.
- Audio conditions influence video tokens but do not create an output soundtrack.
- Retrieval requires a separately licensed reference gallery.
- Results are specific to the released checkpoint and evaluation subset.

## References

Please cite AnyFlow, ImageBind, InternVideo2, MSR-VTT, and VBench as applicable. Source
versions and license notes are recorded in `THIRD_PARTY_NOTICES.md` and
`docs/third_party_commits.md` in the GitHub repository.
