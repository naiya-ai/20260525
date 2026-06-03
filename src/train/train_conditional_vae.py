"""CLI entrypoint for group-wise conditional VAE training."""

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
from torch.nn.parallel import DistributedDataParallel
import yaml

from models import build_conditional_vae_model
from train import (
    ConditionalVAETrainer,
    create_grouped_cvae_dataloader,
    load_grouped_cvae_schema,
)


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
    model_config = config["model"]
    model = build_conditional_vae_model(
        source_n_num_features=schema["source"]["n_num_features"],
        source_category_sizes=schema["source"]["category_sizes"],
        target_n_num_features=schema["target"]["n_num_features"],
        target_category_sizes=schema["target"]["category_sizes"],
        model_config=model_config,
    )
    load_pretrained_source_encoder_if_requested(model, config)
    model = model.to(device)
    if distributed:
        model = ConditionalVAELossModule(model)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    optimizer_kwargs: dict[str, Any] = {}
    optimizer_foreach = config["train"].get("optimizer_foreach")
    if optimizer_foreach is not None:
        optimizer_kwargs["foreach"] = bool(optimizer_foreach)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        **optimizer_kwargs,
    )
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
    trainer = ConditionalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
        config=config,
        schema=schema,
        steps=int(config["train"]["steps"]),
        beta=float(config["train"].get("beta", 1.0)),
        gradient_clip_norm=float(config["train"].get("gradient_clip_norm", 0.0)),
        mixed_precision=bool(config["train"].get("mixed_precision", False)),
        log_every=int(config["logging"].get("log_every", 100)),
        validate_every=int(config["logging"].get("validate_every", 1000)),
        checkpoint_every=int(config["logging"].get("checkpoint_every", 5000)),
        save_best=bool(config["logging"].get("save_best", True)),
        masked_target_dropout=float(config["train"].get("masked_target_dropout", 0.0)),
        source_mask_dropout=float(config["train"].get("source_mask_dropout", 0.0)),
        lr_warmup_steps=int(
            config["train"].get(
                "lr_warmup_steps",
                config["train"].get("lr_warmup", 0),
            )
        ),
        beta_warmup_steps=int(config["train"].get("beta_warmup_steps", 0)),
        beta_warmup_start_step=int(config["train"].get("beta_warmup_start_step", 0)),
        lr_schedule=str(config["train"].get("lr_schedule", "constant")),
        min_learning_rate=float(config["train"].get("min_learning_rate", 0.0)),
        gradient_accumulation_steps=int(
            config["train"].get("gradient_accumulation_steps", 1)
        ),
        grad_scaler_init_scale=resolve_grad_scaler_init_scale(config),
        early_stopping_patience_steps=resolve_early_stopping_patience_steps(config),
        early_stopping_min_delta=resolve_early_stopping_min_delta(config),
        is_main_process=is_main_process,
        distributed=distributed,
    )
    if args.resume_checkpoint is not None:
        resume_training_state(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            checkpoint_path=Path(args.resume_checkpoint),
            device=device,
            is_main_process=is_main_process,
        )

    if is_main_process:
        unwrapped_model = unwrap_model_for_metadata(model)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch_size = (
            int(config["train"]["batch_size"])
            * int(config["train"].get("gradient_accumulation_steps", 1))
            * world_size
        )
        print(
            "conditional VAE setup: "
            f"device={device}, distributed={distributed}, world_size={world_size}, "
            f"dataset={config['data']['dataset_name']}, "
            f"source_groups={config['data']['source_groups']}, "
            f"target_group={config['data']['target_group']}, "
            f"source_num={schema['source']['n_num_features']}, "
            f"source_cat={schema['source']['n_cat_features']}, "
            f"target_num={schema['target']['n_num_features']}, "
            f"target_cat={schema['target']['n_cat_features']}, "
            f"condition_dim={unwrapped_model.condition_dim}, "
            f"latent_dim={unwrapped_model.latent_dim}, "
            f"batch_size_per_rank={config['train']['batch_size']}, "
            f"gradient_accumulation_steps={config['train'].get('gradient_accumulation_steps', 1)}, "
            f"effective_batch_size={effective_batch_size}, "
            f"lr_warmup_steps={config['train'].get('lr_warmup_steps', config['train'].get('lr_warmup', 0))}, "
            f"lr_schedule={config['train'].get('lr_schedule', 'constant')}, "
            f"min_learning_rate={config['train'].get('min_learning_rate', 0.0)}, "
            f"grad_scaler_init_scale={config['train'].get('grad_scaler_init_scale', '')}, "
            f"optimizer_foreach={config['train'].get('optimizer_foreach', '')}, "
            f"source_mask_dropout={config['train'].get('source_mask_dropout', 0.0)}, "
            f"beta={config['train'].get('beta', 1.0)}, "
            f"beta_warmup_steps={config['train'].get('beta_warmup_steps', 0)}, "
            f"beta_warmup_start_step={config['train'].get('beta_warmup_start_step', 0)}, "
            f"early_stopping_patience_steps={resolve_early_stopping_patience_steps(config)}, "
            f"early_stopping_min_delta={resolve_early_stopping_min_delta(config)}",
            flush=True,
        )
    try:
        trainer.fit()
    finally:
        if distributed:
            dist.destroy_process_group()


class ConditionalVAELossModule(nn.Module):
    """DDP wrapper whose forward path computes the configured CVAE loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.model.loss(**kwargs)


def unwrap_model_for_metadata(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "model"):
        model = model.model
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train/train_conditional_vae_eddi.yaml",
    )
    parser.add_argument("--dataset-root", help="Override data.dataset_root.")
    parser.add_argument("--dataset-name", help="Override data.dataset_name.")
    parser.add_argument("--source-groups", nargs="+", help="Override source groups.")
    parser.add_argument("--target-group", help="Override target group.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--device", help="Override device.")
    parser.add_argument("--steps", type=int, help="Override train.steps.")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size.")
    parser.add_argument("--learning-rate", type=float, help="Override train.learning_rate.")
    parser.add_argument("--lr-warmup-steps", type=int, help="Override train.lr_warmup_steps.")
    parser.add_argument("--lr-schedule", help="Override train.lr_schedule.")
    parser.add_argument("--min-learning-rate", type=float, help="Override train.min_learning_rate.")
    parser.add_argument("--source-mask-dropout", type=float, help="Override train.source_mask_dropout.")
    parser.add_argument("--beta-warmup-steps", type=int, help="Override train.beta_warmup_steps.")
    parser.add_argument(
        "--beta-warmup-start-step",
        type=int,
        help="Override train.beta_warmup_start_step. Beta stays 0 until this step, then warms up to train.beta by beta_warmup_steps.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override train.gradient_accumulation_steps.",
    )
    parser.add_argument("--beta", type=float, help="Override train.beta.")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--source-encoder-pretrained-path")
    parser.add_argument(
        "--freeze-source-encoder",
        choices=["true", "false"],
        help="Freeze pretrained source encoder during CVAE training.",
    )
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
    if args.lr_warmup_steps is not None:
        config["train"]["lr_warmup_steps"] = args.lr_warmup_steps
    if args.lr_schedule is not None:
        config["train"]["lr_schedule"] = args.lr_schedule
    if args.min_learning_rate is not None:
        config["train"]["min_learning_rate"] = args.min_learning_rate
    if args.source_mask_dropout is not None:
        config["train"]["source_mask_dropout"] = args.source_mask_dropout
    if args.beta_warmup_steps is not None:
        config["train"]["beta_warmup_steps"] = args.beta_warmup_steps
    if args.beta_warmup_start_step is not None:
        config["train"]["beta_warmup_start_step"] = args.beta_warmup_start_step
    if args.gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.beta is not None:
        config["train"]["beta"] = args.beta
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.log_every is not None:
        config["logging"]["log_every"] = args.log_every
    if args.validate_every is not None:
        config["logging"]["validate_every"] = args.validate_every
    if args.checkpoint_every is not None:
        config["logging"]["checkpoint_every"] = args.checkpoint_every
    if args.seed is not None:
        config["seed"] = args.seed
    if args.disable_early_stopping:
        config.setdefault("train", {}).setdefault("early_stopping", {})["enabled"] = False
    if args.source_encoder_pretrained_path is not None:
        config.setdefault("model", {}).setdefault("source_encoder", {})[
            "pretrained_path"
        ] = args.source_encoder_pretrained_path
    if args.freeze_source_encoder is not None:
        config.setdefault("model", {}).setdefault("source_encoder", {})[
            "freeze_pretrained"
        ] = args.freeze_source_encoder == "true"


def resolve_early_stopping_patience_steps(config: dict[str, Any]) -> int | None:
    train_config = config.get("train", {})
    early_stopping = train_config.get("early_stopping", {})
    if early_stopping is None:
        early_stopping = {}
    if not isinstance(early_stopping, dict):
        raise ValueError("train.early_stopping must be a mapping when set.")
    if not bool(early_stopping.get("enabled", True)):
        return None
    patience_steps = early_stopping.get("patience_steps")
    if patience_steps is not None:
        return max(1, int(patience_steps))
    fraction = float(early_stopping.get("patience_fraction", 0.1))
    if fraction <= 0:
        raise ValueError("train.early_stopping.patience_fraction must be positive.")
    return max(1, int(math.ceil(int(train_config["steps"]) * fraction)))


def resolve_early_stopping_min_delta(config: dict[str, Any]) -> float:
    train_config = config.get("train", {})
    early_stopping = train_config.get("early_stopping", {})
    if not isinstance(early_stopping, dict):
        return 0.0
    return max(0.0, float(early_stopping.get("min_delta", 0.0)))


def resume_training_state(
    *,
    trainer: ConditionalVAETrainer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
    is_main_process: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    target_model = unwrap_model_for_metadata(model)
    target_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler_state:
        trainer.scaler.load_state_dict(scaler_state)
    trainer.best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
    trainer.best_valid_step = int(checkpoint.get("best_valid_step", 0))
    if is_main_process:
        print(
            "resumed conditional VAE training: "
            f"checkpoint={checkpoint_path}, "
            f"checkpoint_step={checkpoint.get('step')}, "
            f"best_valid_step={trainer.best_valid_step}, "
            f"best_valid_loss={trainer.best_valid_loss:.6f}",
            flush=True,
        )


def resolve_grad_scaler_init_scale(config: dict[str, Any]) -> float | None:
    value = config.get("train", {}).get("grad_scaler_init_scale")
    if value is None:
        return None
    init_scale = float(value)
    if init_scale <= 0:
        raise ValueError("train.grad_scaler_init_scale must be positive when set.")
    return init_scale


def load_pretrained_source_encoder_if_requested(
    model: ConditionalMixedTypeVAE,
    config: dict[str, Any],
) -> None:
    source_encoder_config = config.get("model", {}).get("source_encoder", {})
    if not isinstance(source_encoder_config, dict):
        return
    checkpoint_path = source_encoder_config.get("pretrained_path")
    if not checkpoint_path:
        return
    if not hasattr(model, "source_encoder"):
        raise ValueError("Pretrained source_encoder is only supported by the EDDI backbone.")
    checkpoint = torch.load(Path(str(checkpoint_path)), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("source_encoder_state_dict")
    if state_dict is None:
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is None:
            raise ValueError(
                f"Pretrained checkpoint does not contain source encoder weights: {checkpoint_path}"
            )
        prefix = "source_encoder."
        state_dict = {
            key.removeprefix(prefix): value
            for key, value in model_state_dict.items()
            if key.startswith(prefix)
        }
    if not state_dict:
        raise ValueError(f"No source_encoder weights found in {checkpoint_path}")
    missing, unexpected = model.source_encoder.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected source encoder load result from {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    freeze = bool(source_encoder_config.get("freeze_pretrained", False))
    if freeze:
        for parameter in model.source_encoder.parameters():
            parameter.requires_grad = False
    print(
        "loaded pretrained source encoder: "
        f"path={checkpoint_path}, freeze={freeze}",
        flush=True,
    )


def init_distributed_if_requested() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=backend)
    return True


def resolve_device(name: str, *, local_rank: int = 0) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
