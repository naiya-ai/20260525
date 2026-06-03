"""CLI entrypoint for masked source autoencoder pretraining."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models.masked_autoencoder import MaskedSourceAutoencoder
from train.masked_autoencoder import (
    MaskedAutoencoderTrainer,
    create_masked_ae_dataloader,
    load_source_schema,
)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    apply_overrides(config, args)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "cpu")))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = output_dir / config_path.name
    if config_path.resolve() != config_copy_path.resolve():
        shutil.copy2(config_path, config_copy_path)

    schema = load_source_schema(config)
    model_config = config["model"]
    model = MaskedSourceAutoencoder(
        source_n_num_features=schema["source"]["n_num_features"],
        source_category_sizes=schema["source"]["category_sizes"],
        source_encoder_config=model_config["source_encoder"],
        decoder_hidden_layers=model_config.get("decoder_hidden_layers", []),
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        batch_norm=bool(model_config.get("batch_norm", False)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    checkpoint = None
    start_step = 0
    best_valid_loss = None
    if args.resume_checkpoint is not None:
        checkpoint_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        start_step = int(checkpoint.get("step", 0))
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        if args.additional_steps is not None:
            config["train"]["steps"] = int(args.additional_steps)
    train_loader = create_masked_ae_dataloader(
        config,
        split="train",
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"].get("num_workers", 0)),
        seed=seed,
    )
    valid_loader = create_masked_ae_dataloader(
        config,
        split="valid",
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        seed=seed,
    )
    trainer = MaskedAutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=output_dir,
        config=config,
        schema=schema,
        steps=int(config["train"]["steps"]),
        mask_probability=float(config["train"].get("mask_probability", 0.3)),
        gradient_clip_norm=float(config["train"].get("gradient_clip_norm", 0.0)),
        mixed_precision=bool(config["train"].get("mixed_precision", False)),
        log_every=int(config["logging"].get("log_every", 100)),
        validate_every=int(config["logging"].get("validate_every", 1000)),
        checkpoint_every=int(config["logging"].get("checkpoint_every", 5000)),
        save_best=bool(config["logging"].get("save_best", True)),
        start_step=start_step,
        best_valid_loss=best_valid_loss,
    )
    if checkpoint is not None and "scaler_state_dict" in checkpoint:
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(
        "masked AE setup: "
        f"device={device}, dataset={config['data']['dataset_name']}, "
        f"source_groups={config['data']['source_groups']}, "
        f"source_num={schema['source']['n_num_features']}, "
        f"source_cat={schema['source']['n_cat_features']}, "
        f"condition_dim={model.condition_dim}, "
        f"mask_probability={config['train'].get('mask_probability', 0.3)}, "
        f"resume_step={start_step}, run_steps={config['train']['steps']}",
        flush=True,
    )
    trainer.fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_masked_autoencoder_source.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--source-groups", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--mask-probability", type=float)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--additional-steps", type=int)
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
    if args.mask_probability is not None:
        config["train"]["mask_probability"] = args.mask_probability
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.log_every is not None:
        config["logging"]["log_every"] = args.log_every
    if args.validate_every is not None:
        config["logging"]["validate_every"] = args.validate_every
    if args.checkpoint_every is not None:
        config["logging"]["checkpoint_every"] = args.checkpoint_every


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


if __name__ == "__main__":
    main()
