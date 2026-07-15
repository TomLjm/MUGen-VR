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

Condition scale `0.1` was selected on eight validation samples before the fixed test run.

| Variant | Definition | VBench | Retrieval MRR | Audio-flow | Latency |
|---|---|---:|---:|---:|---:|
| B0 | AnyFlow image + original prompt | 0.7600 | n/a | 0.0179 | 4.42 s |
| B1 | prompt rewrite prototype | 0.7511 | n/a | 0.0363 | 4.26 s |
| B2 | Fusion tokens without audio/reference | 0.7596 | 1.0000 | -0.0012 | 4.28 s |
| B3 | full Fusion + audio + reference | 0.7570 | 0.9813 | 0.0046 | 4.31 s |

The completion check passed for all 40 pairs, four variants, one seed, required metrics,
and eight showcase cases. B0 remained the strongest VBench baseline. B2 preserved quality
within 0.0004, while audio/reference conditioning in B3 did not improve retrieval, audio
control, or aggregate video quality. The release therefore claims a reproducible real
condition path and an honest negative result, not a generation-quality gain.

## Limitations

- The base AnyFlow model is restricted to non-commercial use under NVIDIA NSCLv1.
- ImageBind is also governed by non-commercial research terms.
- Audio control is indirect through condition tokens; it does not synthesize an output soundtrack.
- Retrieval quality depends on the licensed local reference gallery.
- On this checkpoint, adding audio and reference tokens slightly reduced aggregate VBench and retrieval MRR.
- The project is evaluated as an internship portfolio system, not a paper-scale benchmark.

## Citation

Please cite the upstream AnyFlow, ImageBind, InternVideo2, MSR-VTT, and VBench projects.
See `THIRD_PARTY_NOTICES.md` and `docs/third_party_commits.md` in the source repository.
