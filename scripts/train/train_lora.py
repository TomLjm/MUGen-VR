#!/usr/bin/env python3
"""Train MUGen condition tokens and AnyFlow cross-attention LoRA adapters."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from mugen.data.feature_store import load_feature_store
from mugen.generation.conditioner import MultimodalConditioner


def parse_args():
    parser = argparse.ArgumentParser(description="AnyFlow multimodal condition-token LoRA training")
    parser.add_argument("--config", default="configs/training/lora.yaml")
    parser.add_argument("--feature-store")
    parser.add_argument("--fusion-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load LoRA/conditioner weights but start a fresh optimizer and step counter.",
    )
    return parser.parse_args()


def latent_chunk_partition(num_frames: int, chunk_size: int = 2) -> list[int]:
    if num_frames < 1 or chunk_size < 1:
        raise ValueError("num_frames and chunk_size must be positive")
    partition = [1]
    remaining = num_frames - 1
    while remaining:
        size = min(chunk_size, remaining)
        partition.append(size)
        remaining -= size
    return partition


def flow_matching_sample(clean, timesteps, num_train_timesteps, noise=None):
    noise = torch.randn_like(clean) if noise is None else noise
    sigma = timesteps.float().div(num_train_timesteps).view(-1, 1, 1, 1, 1).to(clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    return noisy, noise - clean


def cross_attention_lora_config(rank: int, alpha: int):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=r".*\.attn2\.(to_q|to_k|to_v|to_out\.0)$",
    )


def decode_video(path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    decoded = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            decoded.append(frame.to_rgb().to_ndarray())
    if not decoded:
        raise ValueError(f"no video frames decoded: {path}")
    indices = np.linspace(0, len(decoded) - 1, num_frames).astype(int)
    frames = []
    for index in indices:
        frame = torch.from_numpy(decoded[index]).permute(2, 0, 1).float().div(255.0)
        frame = F.interpolate(
            frame.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
        frames.append(frame)
    return torch.stack(frames, dim=1)


def retrieve_reference(features, index: int, top_k: int, gallery_indices):
    gallery_indices = torch.as_tensor(gallery_indices, dtype=torch.long)
    gallery_indices = gallery_indices[gallery_indices != index]
    if not len(gallery_indices):
        raise ValueError("reference gallery is empty after excluding the query")
    query = F.normalize(features[index].float(), dim=-1)
    scores = F.normalize(features[gallery_indices].float(), dim=-1) @ query
    values, positions = scores.topk(min(top_k, len(scores)))
    return features[gallery_indices[positions]], values


def sample_indices(indices, batch_size, generator):
    positions = torch.randint(len(indices), (batch_size,), generator=generator)
    return indices[positions].tolist()


def sample_mismatched_indices(target_indices, pool_indices, generator):
    pool_indices = torch.as_tensor(pool_indices, dtype=torch.long)
    mismatched = []
    for target in target_indices:
        candidates = pool_indices[pool_indices != int(target)]
        if not len(candidates):
            raise ValueError("counterfactual conditioning requires a different source sample")
        position = torch.randint(len(candidates), (1,), generator=generator).item()
        mismatched.append(int(candidates[position]))
    return mismatched


def sample_condition_scale(minimum, maximum, generator):
    minimum, maximum = float(minimum), float(maximum)
    if minimum <= 0 or maximum < minimum:
        raise ValueError("condition scale range must satisfy 0 < minimum <= maximum")
    return minimum + (maximum - minimum) * float(torch.rand((), generator=generator))


def normalize_condition_token_budget(tokens, scale, baseline_tokens=7):
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError("condition tokens must have shape [batch, tokens, dim]")
    budget = min(1.0, float(baseline_tokens) / tokens.shape[1])
    return tokens * float(scale) * budget


def restore_cuda_rng_states(states):
    if not torch.cuda.is_available() or not states:
        return
    device_count = torch.cuda.device_count()
    if len(states) < device_count:
        raise ValueError(
            f"checkpoint has {len(states)} CUDA RNG states for {device_count} visible devices"
        )
    torch.cuda.set_rng_state_all(states[:device_count])


def load_sync_scores(paths):
    scores = {}
    for path in paths or []:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload["per_sample"]:
            scores[row["sample_id"]] = float(row["correlation"])
    return scores


def select_sync_indices(records, split, scores, threshold):
    return torch.tensor(
        [
            index
            for index, row in enumerate(records)
            if row["split"] == split and scores.get(row["sample_id"], -float("inf")) >= threshold
        ],
        dtype=torch.long,
    )


def rhythm_alignment_loss(clean_prediction, temporal_audio):
    if clean_prediction.shape[1] < 2:
        return clean_prediction.new_zeros(())
    motion = (clean_prediction[:, 1:] - clean_prediction[:, :-1]).abs().mean(dim=(2, 3, 4))
    onset = temporal_audio.reshape(temporal_audio.shape[0], 8, -1)[..., -1]
    onset = F.interpolate(onset.unsqueeze(1), size=motion.shape[1], mode="linear", align_corners=False)
    onset = onset.squeeze(1)
    motion = (motion - motion.mean(dim=-1, keepdim=True)) / motion.std(
        dim=-1, keepdim=True
    ).clamp_min(1e-5)
    onset = (onset - onset.mean(dim=-1, keepdim=True)) / onset.std(
        dim=-1, keepdim=True
    ).clamp_min(1e-5)
    correlation = (motion * onset).mean(dim=-1)
    return (1.0 - correlation).mean()


def load_conditioner(tensors, checkpoint_path, generator_dim, reference_tokens, device):
    dims = {name: int(tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
    dims["reference"] = int(tensors["video"].shape[-1])
    conditioner = MultimodalConditioner(
        dims=dims,
        hidden_dim=dims["reference"],
        generator_dim=generator_dim,
        num_reference_tokens=reference_tokens,
        temporal_audio_dim=(
            int(tensors["audio_temporal"].shape[-1]) if "audio_temporal" in tensors else None
        ),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conditioner.fusion.load_state_dict(checkpoint["fusion"])
    conditioner.reference_adapter.load_state_dict(checkpoint["reference_adapter"])
    return conditioner.to(device)


def load_migrated_conditioner_state(conditioner, checkpoint_path):
    state = torch.load(Path(checkpoint_path) / "conditioner.pt", map_location="cpu", weights_only=True)
    current = conditioner.state_dict()
    if "type_embeddings" in state and state["type_embeddings"].shape != current["type_embeddings"].shape:
        migrated = current["type_embeddings"].clone()
        rows = min(migrated.shape[0], state["type_embeddings"].shape[0])
        migrated[:rows] = state["type_embeddings"][:rows]
        state["type_embeddings"] = migrated
    incompatible = conditioner.load_state_dict(state, strict=False)
    allowed_missing = (
        "temporal_audio_projector.",
        "temporal_position_embeddings",
    )
    unexpected_missing = [
        name for name in incompatible.missing_keys if not name.startswith(allowed_missing)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"conditioner migration failed; missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys


def build_condition_tokens(
    conditioner,
    tensors,
    records,
    indices,
    top_k,
    reference_gallery_indices,
    device,
    condition_indices=None,
):
    condition_indices = indices if condition_indices is None else condition_indices
    if len(condition_indices) != len(indices):
        raise ValueError("condition source count must match target count")
    gallery_indices = torch.as_tensor(reference_gallery_indices, dtype=torch.long)
    gallery_positions = {int(value): position for position, value in enumerate(gallery_indices)}
    exclude_indices = torch.tensor(
        [gallery_positions.get(int(index), -1) for index in condition_indices],
        device=device,
    )
    embeddings = {
        "text": tensors["text"][indices].to(device),
        "image": tensors["image"][indices].to(device),
        "audio": tensors["audio"][condition_indices].to(device),
    }
    retrieval_embeddings = {
        name: tensors[name][condition_indices].to(device) for name in ["text", "image", "audio"]
    }
    bundle = conditioner(
        prompt=records[indices[0]]["caption"],
        image_condition=None,
        modality_embeddings=embeddings,
        temporal_audio_embeddings=tensors["audio_temporal"][condition_indices].to(device),
        reference_gallery=tensors["video"][gallery_indices].to(device),
        retrieval_modality_embeddings=retrieval_embeddings,
        reference_top_k=top_k,
        reference_exclude_indices=exclude_indices,
        metadata={"sample_ids": [records[index]["sample_id"] for index in indices]},
    )
    return bundle.condition_tokens


def encode_prompts(pipeline, prompts, condition_tokens, device, dtype):
    with torch.no_grad():
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=device,
        )
    return torch.cat([prompt_embeds.to(dtype), condition_tokens.to(dtype)], dim=1)


def encode_latents(pipeline, records, indices, num_frames, height, width, device):
    videos = torch.stack(
        [decode_video(records[index]["video_path"], num_frames, height, width) for index in indices]
    ).permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        transformer = pipeline.transformer
        transformer_dtype = getattr(transformer, "dtype", None)
        if transformer_dtype is None and hasattr(transformer, "module"):
            transformer_dtype = transformer.module.dtype
        if transformer_dtype is None:
            raise AttributeError("prepared AnyFlow transformer does not expose dtype")
        return pipeline.encode_video(videos, height=height, width=width).to(transformer_dtype)


def forward_loss(
    pipeline,
    conditioner,
    tensors,
    records,
    batch,
    config,
    device,
    dtype,
    chunks,
    reference_gallery_indices,
    condition_scale=1.0,
    counterfactual=False,
    generator=None,
):
    clean = encode_latents(
        pipeline,
        records,
        batch,
        int(config.data.num_frames),
        int(config.data.height),
        int(config.data.width),
        device,
    )
    num_train_timesteps = int(config.training.num_train_timesteps)
    timesteps = torch.randint(1, num_train_timesteps, (len(batch),), device=device)
    noisy, target = flow_matching_sample(clean, timesteps, num_train_timesteps)
    noisy[:, 0] = clean[:, 0]
    timestep_map = timesteps[:, None].repeat(1, clean.shape[1])
    timestep_map[:, 0] = 0
    target_batch = list(batch)
    condition_indices = list(batch)
    if counterfactual:
        if generator is None:
            raise ValueError("counterfactual training requires a random generator")
        mismatched = sample_mismatched_indices(batch, reference_gallery_indices, generator)
        target_batch += list(batch)
        condition_indices += mismatched
    condition_tokens = build_condition_tokens(
        conditioner,
        tensors,
        records,
        target_batch,
        int(config.data.reference_top_k),
        reference_gallery_indices,
        device,
        condition_indices=condition_indices,
    )
    condition_tokens = normalize_condition_token_budget(condition_tokens, condition_scale)
    prompts = [records[index]["caption"] for index in target_batch]
    prompt_embeds = encode_prompts(pipeline, prompts, condition_tokens, device, dtype)
    transformer_noisy = noisy if not counterfactual else torch.cat([noisy, noisy], dim=0)
    transformer_timestep = (
        timestep_map if not counterfactual else torch.cat([timestep_map, timestep_map], dim=0)
    )
    prediction = pipeline.transformer(
        hidden_states=transformer_noisy,
        timestep=transformer_timestep,
        r_timestep=transformer_timestep,
        encoder_hidden_states=prompt_embeds,
        chunk_partition=chunks,
    ).sample
    if prediction.shape[1] != target.shape[1] - 1:
        raise RuntimeError(
            f"AnyFlow output frames {prediction.shape[1]} do not match non-prefix target frames "
            f"{target.shape[1] - 1}; check chunk_partition and full_chunk_limit"
        )
    correct_prediction = prediction[: len(batch)]
    correct_loss = F.mse_loss(correct_prediction.float(), target[:, 1:].float())
    sigma = timesteps.float().div(num_train_timesteps).view(-1, 1, 1, 1, 1).to(noisy)
    predicted_clean = torch.cat(
        [clean[:, :1], noisy[:, 1:] - sigma * correct_prediction], dim=1
    )
    rhythm_weight = float(config.training.get("rhythm_weight", 0.0))
    rhythm_loss = rhythm_alignment_loss(
        predicted_clean.float(), tensors["audio_temporal"][batch].to(device)
    )
    objective = correct_loss + rhythm_weight * rhythm_loss
    if not counterfactual:
        return objective
    shuffled_prediction = prediction[len(batch) :]
    shuffled_loss = F.mse_loss(shuffled_prediction.float(), target[:, 1:].float())
    margin = float(config.training.counterfactual_margin)
    weight = float(config.training.counterfactual_weight)
    return objective + weight * F.relu(margin + correct_loss - shuffled_loss)


def save_checkpoint(
    accelerator,
    pipeline,
    conditioner,
    output_dir,
    step,
    config,
    feature_manifest,
    metrics,
    optimizer,
):
    from peft import get_peft_model_state_dict

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transformer = accelerator.unwrap_model(pipeline.transformer)
    pipeline.save_lora_weights(
        checkpoint_dir,
        transformer_lora_layers=get_peft_model_state_dict(transformer),
        safe_serialization=True,
    )
    torch.save(accelerator.get_state_dict(conditioner), checkpoint_dir / "conditioner.pt")
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(
            {
                "step": step,
                "config": OmegaConf.to_container(config, resolve=True),
                "feature_manifest": feature_manifest,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": step,
        },
        checkpoint_dir / "optimizer.pt",
    )
    return checkpoint_dir


def main():
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers import AnyFlowFARPipeline

    args = parse_args()
    config = OmegaConf.load(args.config)
    feature_store = args.feature_store or config.data.feature_store
    fusion_checkpoint = args.fusion_checkpoint or config.model.fusion_checkpoint
    output_dir = Path(args.output_dir or config.training.output_dir)
    max_steps = args.max_steps or int(config.training.max_steps)
    accelerator = Accelerator(
        mixed_precision=str(config.training.mixed_precision),
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    seed = int(config.training.seed) + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    tensors, records, feature_manifest = load_feature_store(feature_store)
    required = {"text", "image", "audio", "audio_temporal", "video"}
    if missing := required - set(tensors):
        raise ValueError(f"feature store is missing tensors: {sorted(missing)}")
    sync_scores = load_sync_scores(config.data.get("sync_audits", []))
    sync_threshold = float(config.data.get("sync_threshold", -float("inf")))
    if sync_scores:
        train_indices = select_sync_indices(records, "train", sync_scores, sync_threshold)
        val_indices = select_sync_indices(records, "val", sync_scores, sync_threshold)
    else:
        train_indices = torch.tensor([i for i, row in enumerate(records) if row["split"] == "train"])
        val_indices = torch.tensor([i for i, row in enumerate(records) if row["split"] == "val"])
    if len(train_indices) < 2 or len(val_indices) < 1:
        raise ValueError("LoRA training requires at least two train rows and one validation row")
    if accelerator.is_main_process:
        print(
            {
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
                "sync_threshold": sync_threshold if sync_scores else None,
            }
        )

    dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
    pipeline = AnyFlowFARPipeline.from_pretrained(config.model.base, torch_dtype=dtype)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    if args.resume_from:
        pipeline.load_lora_weights(
            args.resume_from,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="default",
        )
    else:
        pipeline.transformer.add_adapter(
            cross_attention_lora_config(int(config.model.lora.rank), int(config.model.lora.alpha))
        )
    pipeline.transformer.enable_gradient_checkpointing()
    trainable_lora = [
        name for name, parameter in pipeline.transformer.named_parameters() if parameter.requires_grad
    ]
    if not trainable_lora or any(".attn2." not in name for name in trainable_lora):
        raise RuntimeError(f"LoRA parameter audit failed: {trainable_lora[:20]}")

    generator_dim = int(pipeline.text_encoder.config.d_model)
    conditioner = load_conditioner(
        tensors,
        fusion_checkpoint,
        generator_dim,
        int(config.model.reference_tokens),
        accelerator.device,
    )
    if args.resume_from:
        load_migrated_conditioner_state(conditioner, args.resume_from)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter in pipeline.transformer.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(
                    config.training.get("lora_learning_rate", config.training.learning_rate)
                ),
            },
            {
                "params": list(conditioner.parameters()),
                "lr": float(
                    config.training.get(
                        "conditioner_learning_rate", config.training.learning_rate
                    )
                ),
            },
        ],
        weight_decay=float(config.training.weight_decay),
    )
    pipeline.transformer, conditioner, optimizer = accelerator.prepare(
        pipeline.transformer, conditioner, optimizer
    )
    pipeline.transformer.train()
    conditioner.train()
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train-rank-{accelerator.process_index}.jsonl"
    step = 0
    if args.resume_from and not args.reset_optimizer:
        resume_state = torch.load(
            Path(args.resume_from) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        torch.set_rng_state(resume_state["cpu_rng_state"])
        restore_cuda_rng_states(resume_state["cuda_rng_states"])
        step = int(resume_state["step"])

    latent_frames = (int(config.data.num_frames) - 1) // pipeline.vae_scale_factor_temporal + 1
    chunks = latent_chunk_partition(latent_frames, int(config.data.latent_chunk_size))
    with log_path.open("a", encoding="utf-8") as log:
        while step < max_steps:
            batch = sample_indices(train_indices, int(config.training.batch_size), generator)
            with accelerator.accumulate(pipeline.transformer, conditioner):
                loss = forward_loss(
                    pipeline,
                    conditioner,
                    tensors,
                    records,
                    batch,
                    config,
                    accelerator.device,
                    dtype,
                    chunks,
                    train_indices,
                    condition_scale=sample_condition_scale(
                        config.training.condition_scale_min,
                        config.training.condition_scale_max,
                        generator,
                    ),
                    counterfactual=True,
                    generator=generator,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for group in optimizer.param_groups for parameter in group["params"]],
                        float(config.training.grad_clip),
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                event = {"step": step, "loss": float(accelerator.gather(loss.detach()).mean())}
                if step % int(config.training.checkpoint_interval) == 0 or step == max_steps:
                    pipeline.transformer.eval()
                    conditioner.eval()
                    with torch.no_grad():
                        val_losses = []
                        validation_samples = min(
                            int(config.training.validation_samples), len(val_indices)
                        )
                        for val_index in val_indices[:validation_samples].tolist():
                            val_losses.append(
                                forward_loss(
                                    pipeline,
                                    conditioner,
                                    tensors,
                                    records,
                                    [val_index],
                                    config,
                                    accelerator.device,
                                    dtype,
                                    chunks,
                                    train_indices,
                                    condition_scale=float(
                                        config.training.validation_condition_scale
                                    ),
                                )
                            )
                        val_loss = torch.stack(val_losses).mean()
                    val_loss = float(accelerator.gather(val_loss.detach()).mean())
                    event["validation_loss"] = val_loss
                    pipeline.transformer.train()
                    conditioner.train()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator,
                            pipeline,
                            conditioner,
                            output_dir,
                            step,
                            config,
                            feature_manifest,
                            {"train_loss": event["loss"], "validation_loss": val_loss},
                            optimizer,
                        )
                log.write(json.dumps(event) + "\n")
                log.flush()
    accelerator.end_training()


if __name__ == "__main__":
    main()
