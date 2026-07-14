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


def retrieve_reference(features, index: int, top_k: int):
    query = F.normalize(features[index].float(), dim=-1)
    scores = F.normalize(features.float(), dim=-1) @ query
    scores[index] = -torch.inf
    values, indices = scores.topk(min(top_k, len(scores) - 1))
    return features[indices], values


def sample_indices(indices, batch_size, generator):
    positions = torch.randint(len(indices), (batch_size,), generator=generator)
    return indices[positions].tolist()


def load_conditioner(tensors, checkpoint_path, generator_dim, reference_tokens, device):
    dims = {name: int(tensors[name].shape[-1]) for name in ["text", "image", "audio"]}
    dims["reference"] = int(tensors["video"].shape[-1])
    conditioner = MultimodalConditioner(
        dims=dims,
        hidden_dim=dims["reference"],
        generator_dim=generator_dim,
        num_reference_tokens=reference_tokens,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conditioner.fusion.load_state_dict(checkpoint["fusion"])
    conditioner.reference_adapter.load_state_dict(checkpoint["reference_adapter"])
    return conditioner.to(device)


def build_condition_tokens(conditioner, tensors, records, indices, top_k, device):
    bundles = []
    for index in indices:
        references, scores = retrieve_reference(tensors["video"], index, top_k)
        embeddings = {
            name: tensors[name][index].unsqueeze(0).to(device)
            for name in ["text", "image", "audio"]
        }
        bundles.append(
            conditioner(
                prompt=records[index]["caption"],
                image_condition=None,
                modality_embeddings=embeddings,
                reference_embeddings=references.unsqueeze(0).to(device),
                reference_scores=scores.unsqueeze(0).to(device),
                metadata={"sample_id": records[index]["sample_id"]},
            )
        )
    return torch.cat([bundle.condition_tokens for bundle in bundles], dim=0)


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
    condition_tokens = build_condition_tokens(
        conditioner,
        tensors,
        records,
        batch,
        int(config.data.reference_top_k),
        device,
    )
    prompt_embeds = encode_prompts(
        pipeline,
        [records[index]["caption"] for index in batch],
        condition_tokens,
        device,
        dtype,
    )
    prediction = pipeline.transformer(
        hidden_states=noisy,
        timestep=timestep_map,
        r_timestep=timestep_map,
        encoder_hidden_states=prompt_embeds,
        chunk_partition=chunks,
    ).sample
    if prediction.shape[1] != target.shape[1] - 1:
        raise RuntimeError(
            f"AnyFlow output frames {prediction.shape[1]} do not match non-prefix target frames "
            f"{target.shape[1] - 1}; check chunk_partition and full_chunk_limit"
        )
    return F.mse_loss(prediction.float(), target[:, 1:].float())


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
    )
    seed = int(config.training.seed) + accelerator.process_index
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    tensors, records, feature_manifest = load_feature_store(feature_store)
    required = {"text", "image", "audio", "video"}
    if missing := required - set(tensors):
        raise ValueError(f"feature store is missing tensors: {sorted(missing)}")
    train_indices = torch.tensor([i for i, row in enumerate(records) if row["split"] == "train"])
    val_indices = torch.tensor([i for i, row in enumerate(records) if row["split"] == "val"])
    if len(train_indices) < 2 or len(val_indices) < 1:
        raise ValueError("LoRA training requires at least two train rows and one validation row")

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
        conditioner.load_state_dict(
            torch.load(Path(args.resume_from) / "conditioner.pt", map_location="cpu", weights_only=True)
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in pipeline.transformer.parameters() if parameter.requires_grad]
        + list(conditioner.parameters()),
        lr=float(config.training.learning_rate),
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
    if args.resume_from:
        resume_state = torch.load(
            Path(args.resume_from) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        torch.set_rng_state(resume_state["cpu_rng_state"])
        if torch.cuda.is_available() and resume_state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
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
                        val_batch = val_indices[: int(config.training.batch_size)].tolist()
                        val_loss = forward_loss(
                            pipeline,
                            conditioner,
                            tensors,
                            records,
                            val_batch,
                            config,
                            accelerator.device,
                            dtype,
                            chunks,
                        )
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
