# MUGen-VR

[简体中文](README.zh-CN.md)
Multimodal condition-token injection and retrieval-augmented video generation on AnyFlow.

[Hugging Face Model](https://huggingface.co/TomLjm/MUGen-VR-AnyFlow-Conditioner) | [Evaluation Space](https://huggingface.co/spaces/TomLjm/MUGen-VR-Evaluation)

MUGen-VR combines ImageBind text, image, and audio features with InternVideo2 video
features. A trainable conditioner produces three fusion tokens, four retrieved-reference
tokens, and eight temporal audio tokens, projects them into the UMT5 embedding space,
and appends them directly to AnyFlow `prompt_embeds`. The AnyFlow backbone remains frozen.

## Architecture

<p align="center">
  <a href="docs/assets/mugen-vr-architecture.jpeg">
    <img
      src="docs/assets/mugen-vr-architecture.jpeg"
      alt="MUGen-VR architecture: frozen multimodal representations, trainable condition fusion and retrieval, and AnyFlow video generation"
      width="100%"
    >
  </a>
</p>

Core components:

- `HierarchicalConditionFusion` maps text, image, and audio features to three condition tokens.
- `ReferenceAdapter` aggregates top-k retrieved videos into four score-aware tokens.
- `MultimodalConditioner` adds modality/type embeddings and projects all 15 tokens to 4096 dimensions.
- `ConditionBundle` carries normalized prompt, image, audio, reference, mask, and trace metadata.
- `AnyFlowVideoGenerator` consumes the bundle through direct `prompt_embeds` injection.

## Targeted Consistency Results

Evaluation uses 40 fixed held-out MSR-VTT clips, generation seed 42, and a condition
scale selected on a separate validation split. Metrics are computed from decoded MP4
files with the official VBench dimension evaluators.

| Metric | Frozen AnyFlow | MUGen-VR | Change |
|---|---:|---:|---:|
| Subject consistency | 0.88328 | **0.88596** | **+0.00268** |
| Motion smoothness | 0.98201 | **0.98248** | **+0.00046** |
| Temporal flickering | 0.96963 | **0.96978** | **+0.00014** |

The condition path reduces subject-consistency error by `2.29%` relative to the frozen
backbone while also improving both reported temporal-consistency dimensions. The default
inference path adds only the project-owned conditioner and keeps AnyFlow frozen. Peak
inference memory is `15.61 GiB` on one RTX 3090.

See [docs/experiments.md](docs/experiments.md) for the evaluation protocol and commands.

## Installation

```bash
conda create -n mugen python=3.10 -y
conda activate mugen
pip install -e .
python -m pytest -q
```

The CPU test suite does not download model weights. Full training and generation require
local copies of the upstream repositories and checkpoints listed in
[`third_party/manifest.yaml`](third_party/manifest.yaml).

## Data And Features

MUGen-VR does not distribute MSR-VTT media or cached third-party features. Build a local
manifest, then extract versioned `safetensors + JSONL` feature shards:

```bash
python scripts/prepare_data/build_msrvtt_manifest.py --help
python scripts/extract_features/extract_real_features.py \
  --manifest data/msrvtt/media_manifest.jsonl \
  --output cache/features/msrvtt-real-v1 \
  --resume
```

Splits are isolated by `video_id`. Reference retrieval excludes the query video and
near-duplicate media hashes.

## Training

Train the fusion and reference modules:

```bash
python scripts/train/train_fusion.py \
  --config configs/training/fusion.yaml \
  --feature-store cache/features/msrvtt-real-v1
```

Train the condition projector (the entrypoint also supports optional cross-attention LoRA):

```bash
accelerate launch --num_processes 4 scripts/train/train_lora.py \
  --config configs/training/lora_project.yaml \
  --feature-store cache/features/msrvtt-real-v1 \
  --fusion-checkpoint outputs/fusion-real-v1/best-seed-42.pt \
  --output-dir outputs/mugen-conditioner
```

Checkpoints include optimizer state, RNG state, feature versions, and configuration for
deterministic resume. The released inference profile loads only project-owned conditioner
weights and keeps AnyFlow frozen.

## Demo

The local demo runs B0 and B3 side by side and displays retrieved references, gating
weights, and latency:

```bash
python scripts/demo/gradio_app.py --help
```

The hosted Space is a static evaluation viewer and does not run persistent GPU inference.

## Repository Layout

```text
src/mugen/       project-owned Python package
scripts/         data, training, inference, evaluation, and release entrypoints
configs/         reproducible experiment configuration
tests/           unit, integration, and resume tests
hf_space/        static B0/B3 evaluation viewer
```

## Scope

- Audio conditions control video tokens and do not synthesize an output soundtrack.
- Retrieval uses a separately licensed local reference gallery.
- Reproduction requires upstream models that are not redistributed by this repository.

## License

Project-owned code is released under MIT. AnyFlow and ImageBind retain non-commercial
upstream restrictions. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before using
the model or generated assets.
