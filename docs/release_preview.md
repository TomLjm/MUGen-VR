# MUGen-VR Release Preview

Generated for explicit publication review. Nothing in this preview has been pushed or uploaded.

## GitHub

- Branch: `release/v1.0`
- Preview commit: `783da71` plus this documentation commit
- Tracked working tree: approximately 2.3 MB
- Largest blob in all Git history: 181,042 bytes
- Git remote: not configured
- Included: source, configs, tests, documentation, final aggregate report, and 16 showcase MP4 files
- Excluded: MSR-VTT media, cached features, upstream repositories, model checkpoints, optimizer state, and local reports

## Hugging Face Model

- Staging directory: `release/hf_model`
- Total size: approximately 134 MB
- `conditioner.pt`: 128,615,217 bytes, SHA-256 `77c38f544576c26c80e0c1767bfb02802572266c9de95baf99265d18eafda909`
- `pytorch_lora_weights.safetensors`: 11,824,784 bytes, SHA-256 `9d164a72cca4195070da8d1ccc2b42258673f4c64e88c3304eba32ea78e0fcfd`
- Also included: Model Card, sanitized MUGen config, aggregate evaluation, and checksum manifest
- Explicitly absent: optimizer state, AnyFlow/ImageBind/InternVideo weights, data, and feature caches
- License metadata: `other`, because project code is MIT while upstream model terms impose non-commercial restrictions

## Hugging Face Space

- Source directory: `hf_space`
- Static SDK; no persistent GPU cost
- Eight fixed held-out B0/B3 pairs, 16 MP4 files, approximately 1.6 MB total
- Desktop and mobile browser QA passed with no document overflow or video load errors

## Final Evidence

- Tests: 49 passed
- Data: 6,000 real-feature clips, including 5,000 train / 423 val / 577 test
- Fusion: 2,000 steps; best validation MRR 0.9560
- LoRA: 300 four-GPU steps with train-only reference gallery
- Evaluation: 40 fixed test samples, seed 42, eight showcase cases
- Completion check: passed

| Variant | VBench | Retrieval MRR | Audio-flow | Latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| B0 | 0.7600 | n/a | 0.0179 | 4.42 s | 15.61 GiB |
| B1 | 0.7511 | n/a | 0.0363 | 4.26 s | 15.61 GiB |
| B2 | 0.7596 | 1.0000 | -0.0012 | 4.28 s | 15.61 GiB |
| B3 | 0.7570 | 0.9813 | 0.0046 | 4.31 s | 15.61 GiB |

Condition scale 0.1 was selected only on eight validation clips. It substantially reduced the
B3 quality loss, but B0 remained the best VBench baseline. Public materials must not claim a
generation-quality, retrieval, or audio-control improvement for B3.

## Publication Gate

Before publication, the owner must explicitly confirm the target GitHub repository and Hugging
Face model/Space namespaces. Push/upload is intentionally blocked until that confirmation.
