"""DDP smoke test for a transformer CVAE config without writing checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
import yaml

from models import build_conditional_vae_model
from train.dataset import load_grouped_cvae_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-step", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"rank={rank} start device={device}", flush=True)

    config = yaml.safe_load(Path(args.config).read_text())
    schema = load_grouped_cvae_schema(config)
    model = build_conditional_vae_model(
        source_n_num_features=schema["source"]["n_num_features"],
        source_category_sizes=schema["source"]["category_sizes"],
        target_n_num_features=schema["target"]["n_num_features"],
        target_category_sizes=schema["target"]["category_sizes"],
        model_config=config["model"],
    )
    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"rank={rank} built params={param_count:,}", flush=True)

    model = model.to(device)
    print(f"rank={rank} moved_to_cuda", flush=True)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    print(f"rank={rank} wrapped_ddp", flush=True)
    if args.skip_step:
        dist.barrier()
        report_memory(rank)
        dist.destroy_process_group()
        return

    optimizer_kwargs = {}
    optimizer_foreach = config["train"].get("optimizer_foreach")
    if optimizer_foreach is not None:
        optimizer_kwargs["foreach"] = bool(optimizer_foreach)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
        **optimizer_kwargs,
    )
    batch = synthetic_batch(schema, batch_size=args.batch_size, device=device)
    scaler = GradScaler(
        "cuda",
        enabled=bool(config["train"].get("mixed_precision", False)),
        init_scale=float(config["train"].get("grad_scaler_init_scale", 1.0)),
    )
    optimizer.zero_grad(set_to_none=True)
    with autocast("cuda", enabled=bool(config["train"].get("mixed_precision", False))):
        losses = model.module.loss(
            **batch,
            beta=float(config["train"].get("beta", 1.0)),
            sample_latent=True,
            sample_loss_weights=torch.ones(args.batch_size, device=device),
        )
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["train"].get("gradient_clip_norm", 1.0)),
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print(f"rank={rank} step_done loss={float(losses['loss'].detach().cpu()):.6f}", flush=True)
    dist.barrier()
    report_memory(rank)
    dist.destroy_process_group()


def synthetic_batch(schema: dict, *, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    source_num = torch.randn(batch_size, schema["source"]["n_num_features"], device=device)
    source_cat = random_categories(
        batch_size,
        schema["source"]["category_sizes"],
        device=device,
    )
    target_num = torch.randn(batch_size, schema["target"]["n_num_features"], device=device)
    target_cat = random_categories(
        batch_size,
        schema["target"]["category_sizes"],
        device=device,
    )
    return {
        "source_num": source_num,
        "source_cat": source_cat,
        "source_num_mask": torch.ones_like(source_num),
        "source_cat_mask": torch.ones_like(source_cat, dtype=torch.float32),
        "target_num": target_num,
        "target_cat": target_cat,
        "target_num_mask": torch.ones_like(target_num),
        "target_cat_mask": torch.ones_like(target_cat, dtype=torch.float32),
    }


def random_categories(
    batch_size: int,
    category_sizes: list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not category_sizes:
        return torch.empty(batch_size, 0, dtype=torch.long, device=device)
    values = [
        torch.randint(0, int(size), (batch_size,), device=device)
        for size in category_sizes
    ]
    return torch.stack(values, dim=1).to(dtype=torch.long)


def report_memory(rank: int) -> None:
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"rank={rank} max_allocated_gb={allocated:.2f} max_reserved_gb={reserved:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
