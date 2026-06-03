"""CLI entrypoint for source-conditioned tabular DDPM training."""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

from diffusion import ConditionalGaussianMultinomialDiffusion
from models import MLPDenoiser, make_eddi_source_encoder, make_mlp_source_encoder
from train import MLPDDPMTrainer, create_grouped_cvae_dataloader, load_grouped_cvae_schema


def main() -> None:
    args = parse_args()
    distributed = init_distributed_if_requested()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0

    config_path = Path(args.config)
    config = load_config(config_path)
    apply_overrides(config, args)

    seed = int(config.get("seed", 42))
    set_seed(seed + rank)
    device = resolve_device(str(config.get("device", "cuda")), local_rank=local_rank)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        shutil.copy2(config_path, output_dir / config_path.name)

    schema = load_grouped_cvae_schema(config)
    source_encoder, diffusion = build_model_stack(config, schema)
    diffusion = diffusion.to(device)
    if source_encoder is not None:
        source_encoder = source_encoder.to(device)

    if distributed:
        diffusion.denoise_fn = DistributedDataParallel(
            diffusion.denoise_fn,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
        if source_encoder is not None and has_trainable_parameters(source_encoder):
            source_encoder = DistributedDataParallel(
                source_encoder,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )

    optimizer_parameters = list(diffusion.denoise_fn.parameters())
    if source_encoder is not None and has_trainable_parameters(source_encoder):
        optimizer_parameters.extend(source_encoder.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    lr_scheduler = build_lr_scheduler(optimizer, config)

    train_loader = create_grouped_cvae_dataloader(
        config,
        split="train",
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 0)),
        seed=seed,
        distributed=distributed,
    )
    valid_loader = create_grouped_cvae_dataloader(
        config,
        split="valid",
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        seed=seed,
        distributed=distributed,
    )

    trainer = MLPDDPMTrainer(
        diffusion=diffusion,
        source_encoder=source_encoder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
        config=config,
        schema=schema,
        steps=int(config["train"]["steps"]),
        gradient_clip_norm=float(config["train"].get("gradient_clip_norm", 0.0)),
        mixed_precision=bool(config["train"].get("mixed_precision", False)),
        log_every=int(config["logging"].get("log_every", 100)),
        validate_every=int(config["logging"].get("validate_every", 1000)),
        checkpoint_every=int(config["logging"].get("checkpoint_every", 5000)),
        save_best=bool(config["logging"].get("save_best", True)),
        source_feature_dropout=float(config["train"].get("source_feature_dropout", 0.0)),
        gradient_accumulation_steps=int(config["train"].get("gradient_accumulation_steps", 1)),
        distributed=distributed,
        rank=rank,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )

    if is_main_process:
        condition_dim = 0 if source_encoder is None else unwrap_ddp(source_encoder).output_dim
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch_size = (
            int(config["train"]["batch_size"])
            * int(config["train"].get("gradient_accumulation_steps", 1))
            * world_size
        )
        print(
            "DDPM setup: "
            f"device={device}, distributed={distributed}, world_size={world_size}, "
            f"dataset={config['data']['dataset_name']}, "
            f"source_groups={config['data']['source_groups']}, "
            f"target_group={config['data']['target_group']}, "
            f"source_num={schema['source']['n_num_features']}, "
            f"source_cat={schema['source']['n_cat_features']}, "
            f"target_num={schema['target']['n_num_features']}, "
            f"target_cat={schema['target']['n_cat_features']}, "
            f"target_dim={diffusion.target_dim}, condition_dim={condition_dim}, "
            f"timesteps={diffusion.num_timesteps}, "
            f"batch_size_per_rank={config['train']['batch_size']}, "
            f"effective_batch_size={effective_batch_size}",
            flush=True,
        )

    try:
        trainer.fit()
    finally:
        if distributed:
            dist.destroy_process_group()


def build_model_stack(
    config: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[nn.Module | None, ConditionalGaussianMultinomialDiffusion]:
    model_config = config["model"]
    source_schema = schema["source"]
    target_schema = schema["target"]
    source_encoder = build_source_encoder(model_config, source_schema)
    condition_dim = 0 if source_encoder is None else int(source_encoder.output_dim)
    target_n_num = int(target_schema["n_num_features"])
    target_category_sizes = [int(size) for size in target_schema["category_sizes"]]
    target_dim = target_n_num + sum(target_category_sizes)
    denoise_fn = MLPDenoiser(
        target_dim=target_dim,
        condition_dim=condition_dim,
        time_embedding_dim=int(model_config["time_embedding_dim"]),
        hidden_layers=model_config["hidden_layers"],
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        batch_norm=bool(model_config.get("batch_norm", False)),
    )
    diffusion_config = config["diffusion"]
    diffusion = ConditionalGaussianMultinomialDiffusion(
        target_n_num_features=target_n_num,
        target_category_sizes=target_category_sizes,
        denoise_fn=denoise_fn,
        num_timesteps=int(diffusion_config["num_timesteps"]),
        scheduler=str(diffusion_config["scheduler"]),
        gaussian_loss_type=str(diffusion_config["gaussian_loss_type"]),
        gaussian_parametrization=str(diffusion_config["gaussian_parametrization"]),
        multinomial_loss_type=str(diffusion_config["multinomial_loss_type"]),
        categorical_parametrization=str(diffusion_config["categorical_parametrization"]),
    )
    return source_encoder, diffusion


def build_source_encoder(
    model_config: dict[str, Any],
    source_schema: dict[str, Any],
) -> nn.Module | None:
    encoder_config = model_config.get("source_encoder")
    if encoder_config is None:
        return None
    encoder_type = str(encoder_config.get("type", "eddi")).lower()
    n_num_features = int(source_schema["n_num_features"])
    category_sizes = [int(size) for size in source_schema["category_sizes"]]
    if encoder_type == "eddi":
        return make_eddi_source_encoder(
            role="source",
            n_num_features=n_num_features,
            category_sizes=category_sizes,
            embedding_dim=int(encoder_config.get("embedding_dim", 128)),
            hidden_layers=encoder_config.get("hidden_layers", []),
            output_dim=int(encoder_config.get("output_dim", 256)),
            aggregation=str(encoder_config.get("aggregation", "sum")),
        )
    if encoder_type == "mlp":
        return make_mlp_source_encoder(
            n_num_features=n_num_features,
            category_sizes=category_sizes,
            hidden_layers=encoder_config.get("hidden_layers", []),
            output_dim=int(encoder_config.get("output_dim", 256)),
            dropout=float(encoder_config.get("dropout", 0.0)),
            activation=str(encoder_config.get("activation", model_config.get("activation", "relu"))),
            normalization=str(encoder_config.get("normalization", "none")),
        )
    raise ValueError(f"Unsupported source_encoder.type: {encoder_type!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_ddpm_mlp.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--source-groups", nargs="+")
    parser.add_argument("--target-group")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-timesteps", type=int)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-encoder-type", choices=["eddi", "mlp"])
    parser.add_argument("--gradient-accumulation-steps", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping.")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.dataset_root is not None:
        config.setdefault("data", {})["dataset_root"] = args.dataset_root
    if args.dataset_name is not None:
        config.setdefault("data", {})["dataset_name"] = args.dataset_name
    if args.source_groups is not None:
        config.setdefault("data", {})["source_groups"] = args.source_groups
    if args.target_group is not None:
        config.setdefault("data", {})["target_group"] = args.target_group
    if args.max_rows_per_split is not None:
        config.setdefault("data", {})["max_rows_per_split"] = args.max_rows_per_split
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.num_timesteps is not None:
        config["diffusion"]["num_timesteps"] = args.num_timesteps
    if args.log_every is not None:
        config["logging"]["log_every"] = args.log_every
    if args.validate_every is not None:
        config["logging"]["validate_every"] = args.validate_every
    if args.checkpoint_every is not None:
        config["logging"]["checkpoint_every"] = args.checkpoint_every
    if args.seed is not None:
        config["seed"] = args.seed
    if args.source_encoder_type is not None:
        config.setdefault("model", {}).setdefault("source_encoder", {})["type"] = args.source_encoder_type


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_config = config["train"].get("lr_scheduler")
    if not scheduler_config:
        return None
    scheduler_type = str(scheduler_config.get("type", "constant")).lower()
    if scheduler_type == "constant":
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported lr_scheduler.type: {scheduler_type}")
    total_steps = int(config["train"]["steps"])
    warmup_steps = int(scheduler_config.get("warmup_steps", 0))
    min_lr = float(scheduler_config.get("min_learning_rate", 0.0))
    base_lr = float(config["train"]["learning_rate"])

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def init_distributed_if_requested() -> bool:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return True


def resolve_device(device_name: str, *, local_rank: int) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda", local_rank)
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_trainable_parameters(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


if __name__ == "__main__":
    main()
