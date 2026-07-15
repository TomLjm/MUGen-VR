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

# MUGen-VR

MUGen-VR adds project-owned multimodal condition modules to AnyFlow-FAR. ImageBind
encodes text, image, and audio; InternVideo2 encodes videos and retrieved references.
Three hierarchical fusion tokens and four score-aware reference tokens are projected
to the 4096-dimensional UMT5 prompt space and appended to AnyFlow `prompt_embeds`.

## Published files

The Hugging Face release contains only MUGen-owned lightweight files:

- Fusion and Reference Adapter weights.
- The condition projector and modality/type embeddings.
- AnyFlow cross-attention LoRA weights.
- Training configuration, encoder/data versions, and evaluation metadata.

It does not contain AnyFlow, ImageBind, InternVideo2, MSR-VTT media, or cached third-party
features. Users must obtain those assets under their upstream terms.

## Evaluation

The practical project evaluation compares four variants on 40 fixed held-out samples
with generation seed 42:

| Variant | Definition | Status |
|---|---|---|
| B0 | AnyFlow image + original prompt | pending final run |
| B1 | prompt rewrite prototype | pending final run |
| B2 | Fusion tokens without audio/reference | pending final run |
| B3 | full Fusion + audio + reference | pending final run |

Reported metrics include key VBench dimensions, retrieval R@1/5/10 and MRR,
audio-onset/optical-flow correlation, latency, generated FPS, and peak VRAM. No quality
improvement is claimed until the real held-out report is attached.

## Limitations

- The base AnyFlow model is restricted to non-commercial use under NVIDIA NSCLv1.
- ImageBind is also governed by non-commercial research terms.
- Audio control is indirect through condition tokens; it does not synthesize an output soundtrack.
- Retrieval quality depends on the licensed local reference gallery.
- The project is evaluated as an internship portfolio system, not a paper-scale benchmark.

## Citation

Please cite the upstream AnyFlow, ImageBind, InternVideo2, MSR-VTT, and VBench projects.
See `THIRD_PARTY_NOTICES.md` and `docs/third_party_commits.md` in the source repository.
