# MUGen-VR Job Materials

## Resume version

- Built MUGen-VR, a multimodal controllable-video pipeline on AnyFlow 1.3B, using ImageBind text/image/audio and InternVideo2 video features with seven learned condition tokens injected directly into 4096-d UMT5 prompt embeddings.
- Designed HierarchicalConditionFusion and a score-aware Reference Adapter; restricted rank-16 LoRA updates to 240 cross-attention q/k/v/out tensors while freezing UMT5, VAE, and the base video transformer.
- Built a leakage-aware MSR-VTT data pipeline with deterministic train/val/test splits, media/audio SHA-256, decodability checks, resumable four-GPU `safetensors + JSONL` feature extraction, and explicit bad-sample tracking.
- Implemented real MP4/VBench evaluation, retrieval R@K/MRR, audio-onset/optical-flow correlation, latency/VRAM profiling, and a fixed 40-sample B0-B3 comparison with eight side-by-side cases.
- Verified a 25-frame 256x448 AnyFlow engineering baseline at 4.12 s, 6.07 generated FPS, and 15.47 GiB peak VRAM on one RTX 3090; final quality metrics are reported only from the held-out run.

## 90-second project explanation

The original project only rewrote prompts, so the trained Fusion and Reference Adapter
did not actually affect AnyFlow attention. I replaced that path with seven learned
condition tokens: three represent visual, motion/audio, and semantic fusion levels;
four summarize top-k retrieved videos with similarity-aware attention. These tokens are
projected from 768 to AnyFlow's 4096-dimensional UMT5 space and appended to
`prompt_embeds`. The base model stays frozen; only the MUGen modules and cross-attention
LoRA train. The second focus was credibility: real audio/video features, strict split
isolation, no dummy fallback, fixed held-out comparisons, and lightweight checkpoints
that do not redistribute upstream weights.

## Core code reading route

1. `src/mugen/common/interfaces.py`: `ConditionBundle` contract and token merge.
2. `src/mugen/fusion/fusion_module.py`: hierarchical multimodal fusion and gates.
3. `src/mugen/generation/retrieve_then_generate.py`: score-aware reference tokens.
4. `src/mugen/generation/conditioner.py`: seven-token projection into AnyFlow space.
5. `src/mugen/generation/generators/anyflow_generator.py`: strict generation and frame chunking.
6. `scripts/train/train_fusion.py`: real-feature retrieval supervision.
7. `scripts/train/train_lora.py`: VAE latents, flow matching, cross-attention LoRA, resume.
8. `scripts/eval/ablation_study.py`: practical B0-B3 completion check.

## Failure review

- Prompt rewrite made Fusion look active while AnyFlow only consumed text. Fixed by direct token injection.
- Silent MSR-VTT mirrors could not support audio conditioning. Replaced with an audio-inclusive 9K/1K archive and tracked 1,189 missing-audio failures.
- ImageBind used a working-directory checkpoint path and started a duplicate 4.47 GB download. Fixed with explicit repository checkpoint loading.
- AnyFlow cache missed one UMT5 shard. Completed the cache and added strict offline loading checks.
- 25/49-frame inference inherited the 81-frame chunk schedule. Added automatic latent chunk partitioning.
- LoRA training initially passed `[B,C,T,H,W]` to an API expecting `[B,T,C,H,W]`. Added a regression test.
- Generic accelerator state duplicated 2.86 GB of frozen base weights. Replaced it with lightweight LoRA, conditioner, optimizer, and RNG checkpoints.

## Interview questions and answers

1. **Why not concatenate raw ImageBind features directly?**  The modalities have different reliability and roles. Hierarchical tokens preserve visual detail, motion/audio cues, and semantics while gates make modality use inspectable.
2. **Why seven tokens?**  Three fusion levels plus four reference queries are a small, fixed attention budget. It is expressive enough to separate roles without materially extending the prompt sequence.
3. **Why project into UMT5 space?**  AnyFlow cross-attention already consumes 4096-d text states. Matching that contract reuses the trained attention path without changing the base architecture.
4. **How do you prevent text dominance?**  Modality dropout, gate entropy regularization, missing-modality tests, and B2/B3 comparison expose collapse rather than hiding it.
5. **How is retrieval trained?**  The fused query is supervised against the real InternVideo2 embedding of its target video with symmetric InfoNCE; self references are excluded.
6. **Why score-aware reference aggregation?**  Top-k references are not equally relevant. Similarity scores enter attention logits so weak references contribute less.
7. **How do you avoid leakage?**  Split by `video_id`, hash media, reject overlap, exclude self retrieval, and build the held-out manifest only from test rows.
8. **Why LoRA only on cross-attention?**  The new information enters through condition tokens. Updating self-attention or feed-forward blocks increases cost and makes attribution less clear.
9. **What remains frozen?**  UMT5, Wan VAE, and all base AnyFlow parameters; only MUGen modules and cross-attention LoRA update.
10. **What is the training target?**  Flow-matching velocity on real VAE video latents, with the first latent frame kept clean as the image prefix.
11. **Why 25 frames first?**  It validates gradients and memory on a 24 GB GPU before increasing temporal length to 49 frames.
12. **How do you handle missing audio?**  Invalid media is tracked during preprocessing; runtime modality masks and training dropout handle optional inputs without fabricated features.
13. **Why is audio-flow correlation useful?**  It measures whether visual motion energy follows input audio onsets, which is closer to rhythm control than generic semantic similarity.
14. **Why still keep ImageBind alignment?**  It provides a complementary semantic audio-video score; rhythm correlation and semantic alignment fail in different ways.
15. **Why not require statistical significance?**  This is a portfolio project with 30-50 held-out cases. Fixed inputs, consistent seeds, transparent metrics, and side-by-side failures are more proportionate; bootstrap remains advisory.
16. **What did the 15.47 GiB number measure?**  A real 25-frame, 256x448, one-step AnyFlow generation smoke, including the loaded pipeline and generation allocations.
17. **Can the weights be commercialized?**  No. The base AnyFlow and ImageBind terms restrict usage; the repository and model card state non-commercial research use.
18. **What is published on Hugging Face?**  Only MUGen-owned Fusion, Reference Adapter, projector, LoRA, configs, and metrics; no upstream weights or dataset media.
19. **What would you optimize next?**  Compare 4/8 condition tokens, top-k 1/3/5, and rank 16/32 only if the compact B0-B3 run shows a specific bottleneck.
20. **What is the strongest engineering contribution?**  Turning an apparently multimodal prompt prototype into a real, testable condition path with reproducible data, constrained parameter updates, and honest evaluation.

## Presentation outline

1. Target role and the original prototype gap.
2. End-to-end architecture and seven condition tokens.
3. Real data, encoders, retrieval, and leakage prevention.
4. AnyFlow injection and cross-attention LoRA parameter scope.
5. Training and checkpoint design for four RTX 3090s.
6. B0-B3 evaluation, efficiency, and side-by-side cases.
7. Failure cases and what was changed.
8. GitHub/Hugging Face release boundaries and licensing.
