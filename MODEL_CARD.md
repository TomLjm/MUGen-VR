---
library_name: diffusers
license: other
license_name: mit-code-with-noncommercial-upstream-weight-restrictions
base_model: nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers
pipeline_tag: image-to-video
tags:
  - multimodal
  - video-generation
  - retrieval
---

# MUGen-VR AnyFlow Conditioner

MUGen-VR adds a trainable multimodal condition path to a frozen AnyFlow-FAR backbone.
ImageBind encodes text, image, and audio inputs; InternVideo2 encodes videos and retrieved
references. Three fusion tokens, four reference tokens, and eight temporal audio tokens
are projected into the 4096-dimensional UMT5 space and appended to `prompt_embeds`.

## Files

The release contains only project-owned lightweight artifacts:

- Fusion, Reference Adapter, condition projector, and modality/type embeddings.
- Sanitized training configuration, evaluation metadata, and checksums.

It excludes AnyFlow, ImageBind, and InternVideo2 weights, MSR-VTT media, cached features,
and optimizer state. These dependencies must be obtained under their upstream terms.

## Targeted Consistency Evaluation

The fixed evaluation uses 40 held-out samples with generation seed 42. Condition scale
`0.05` was selected on a separate validation split.

| Metric | Frozen AnyFlow | MUGen-VR | Change |
|---|---:|---:|---:|
| Subject consistency | 0.88328 | **0.88596** | **+0.00268** |
| Motion smoothness | 0.98201 | **0.98248** | **+0.00046** |
| Temporal flickering | 0.96963 | **0.96978** | **+0.00014** |

MUGen-VR reduces subject-consistency error by `2.29%` relative to the frozen backbone
and improves both reported temporal-consistency dimensions.

## Scope

- AnyFlow is restricted to non-commercial use under NVIDIA NSCLv1.
- ImageBind is governed by non-commercial research terms.
- Audio conditions influence video tokens but do not create an output soundtrack.
- Retrieval requires a separately licensed reference gallery.
- Results are specific to the released checkpoint and evaluation subset.

## References

Please cite AnyFlow, ImageBind, InternVideo2, MSR-VTT, and VBench as applicable. Source
versions and license notes are recorded in `THIRD_PARTY_NOTICES.md` and
`docs/third_party_commits.md` in the GitHub repository.
