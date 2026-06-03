"""Training loop for MLP-based conditional DDPM."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from diffusion import ConditionalGaussianMultinomialDiffusion


class MLPDDPMTrainer:
    """Train a conditional Gaussian/multinomial diffusion model."""

    def __init__(
        self,
        *,
        diffusion: ConditionalGaussianMultinomialDiffusion,
        source_encoder: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        device: torch.device,
        output_dir: Path,
        config: dict[str, Any],
        schema: dict[str, Any],
        steps: int,
        gradient_clip_norm: float | None,
        mixed_precision: bool,
        log_every: int,
        validate_every: int,
        checkpoint_every: int,
        save_best: bool,
        source_feature_dropout: float = 0.0,
        gradient_accumulation_steps: int = 1,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not 0 <= source_feature_dropout < 1:
            raise ValueError("source_feature_dropout must be in [0, 1).")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")

        self.diffusion = diffusion.to(device)
        self.denoise_model = _unwrap_ddp(self.diffusion.denoise_fn)
        self.source_encoder = source_encoder.to(device) if source_encoder is not None else None
        self.source_encoder_model = (
            _unwrap_ddp(self.source_encoder) if self.source_encoder is not None else None
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.output_dir = output_dir
        self.config = config
        self.schema = schema
        self.steps = int(steps)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.log_every = max(1, int(log_every))
        self.validate_every = max(1, int(validate_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.save_best = bool(save_best)
        self.source_feature_dropout = float(source_feature_dropout)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.distributed = bool(distributed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.is_main_process = self.rank == 0
        self.scaler = GradScaler("cuda", enabled=self.use_mixed_precision)
        self.best_valid_loss = float("inf")
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"
        self._ensure_metrics_header()

    def fit(self) -> None:
        """Run the full training loop."""
        train_iterator = _cycle(self.train_loader)
        running: dict[str, float] = {}
        running_count = 0
        for step in range(1, self.steps + 1):
            metrics = self.train_step(train_iterator)
            running_count += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if step == 1 or step % self.log_every == 0:
                averaged = {
                    key: value / running_count
                    for key, value in running.items()
                }
                running.clear()
                running_count = 0
                if self.is_main_process:
                    self._log_metrics(step, "train", averaged)
                    print(_format_metrics(step, "train", averaged))

            if self.valid_loader is not None and step % self.validate_every == 0:
                valid_metrics = self.validate()
                if self.is_main_process:
                    self._log_metrics(step, "valid", valid_metrics)
                    print(_format_metrics(step, "valid", valid_metrics))
                    valid_loss = valid_metrics["loss"]
                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        if self.save_best:
                            self.save_checkpoint(step, self.output_dir / "checkpoint_best.pt")

            if self.is_main_process and step % self.checkpoint_every == 0:
                self.save_checkpoint(step, self.output_dir / "checkpoint_latest.pt")

        if self.is_main_process:
            self.save_checkpoint(self.steps, self.output_dir / "checkpoint_latest.pt")

    def train_step(self, train_iterator: Iterator[dict[str, Tensor]]) -> dict[str, float]:
        self.diffusion.train()
        if self.source_encoder is not None:
            self.source_encoder.train(_has_trainable_parameters(self.source_encoder))
        self.optimizer.zero_grad(set_to_none=True)
        metric_totals: dict[str, Tensor] = {}
        for _ in range(self.gradient_accumulation_steps):
            batch = _move_batch_to_device(next(train_iterator), self.device)
            if self.source_encoder is not None and _has_trainable_parameters(self.source_encoder):
                condition = self._encode_condition(
                    batch,
                    apply_source_dropout=self.source_feature_dropout > 0,
                )
            else:
                with torch.no_grad():
                    condition = self._encode_condition(
                        batch,
                        apply_source_dropout=self.source_feature_dropout > 0,
                    )
            model_kwargs = self._build_model_kwargs(batch)

            with autocast("cuda", enabled=self.use_mixed_precision):
                losses = self.diffusion.mixed_loss(
                    target_num=batch["target_num"],
                    target_cat=batch["target_cat"],
                    target_num_mask=batch["target_num_mask"],
                    target_cat_mask=batch["target_cat_mask"],
                    condition=condition,
                    model_kwargs=model_kwargs,
                )
                loss = losses["loss"] / float(self.gradient_accumulation_steps)

            self.scaler.scale(loss).backward()
            for key, value in losses.items():
                metric_totals[key] = metric_totals.get(key, value.detach().new_zeros(())) + value.detach().float()

        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            parameters = list(self.denoise_model.parameters())
            if self.source_encoder_model is not None:
                parameters.extend(self.source_encoder_model.parameters())
            nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
        learning_rate = self._current_learning_rate()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        metrics = {
            key: value / float(self.gradient_accumulation_steps)
            for key, value in metric_totals.items()
        }
        metrics["lr"] = torch.tensor(
            learning_rate,
            device=self.device,
            dtype=torch.float32,
        )
        metrics = _all_reduce_mean(metrics, self.distributed)
        return {key: float(value.cpu()) for key, value in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.valid_loader is None:
            raise ValueError("valid_loader is not configured.")
        self.diffusion.eval()
        if self.source_encoder is not None:
            self.source_encoder.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.valid_loader:
            batch = _move_batch_to_device(batch, self.device)
            condition = self._encode_condition(batch)
            model_kwargs = self._build_model_kwargs(batch)
            losses = self.diffusion.mixed_loss(
                target_num=batch["target_num"],
                target_cat=batch["target_cat"],
                target_num_mask=batch["target_num_mask"],
                target_cat_mask=batch["target_cat_mask"],
                condition=condition,
                model_kwargs=model_kwargs,
            )
            batch_size = int(batch["target_num"].shape[0])
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu()) * batch_size

        if self.distributed:
            total_tensor = torch.tensor(
                [
                    totals.get("loss", 0.0),
                    totals.get("loss_gaussian", 0.0),
                    totals.get("loss_multinomial", 0.0),
                    float(count),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            totals = {
                "loss": float(total_tensor[0].cpu()),
                "loss_gaussian": float(total_tensor[1].cpu()),
                "loss_multinomial": float(total_tensor[2].cpu()),
            }
            count = int(total_tensor[3].item())

        if count == 0:
            raise ValueError("Validation loader produced no batches.")
        return {key: value / count for key, value in totals.items()}

    def save_checkpoint(self, step: int, path: Path) -> None:
        payload = {
            "step": int(step),
            "config": self.config,
            "schema": self.schema,
            "model_state_dict": self.denoise_model.state_dict(),
            "source_encoder_state_dict": (
                None
                if self.source_encoder_model is None
                else self.source_encoder_model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        torch.save(payload, path)

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _encode_condition(
        self,
        batch: dict[str, Tensor],
        *,
        apply_source_dropout: bool = False,
    ) -> Tensor | None:
        if self.source_encoder is None:
            return None
        source_num_mask = batch["source_num_mask"]
        source_cat_mask = batch["source_cat_mask"]
        if apply_source_dropout:
            source_num_mask, source_cat_mask = self._drop_source_features(
                source_num_mask,
                source_cat_mask,
            )
        return self.source_encoder(
            batch["source_num"],
            batch["source_cat"],
            source_num_mask,
            source_cat_mask,
        )

    def _drop_source_features(
        self,
        source_num_mask: Tensor,
        source_cat_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        keep_prob = 1.0 - self.source_feature_dropout
        if keep_prob >= 1.0:
            return source_num_mask, source_cat_mask
        num_keep = torch.rand_like(source_num_mask) < keep_prob
        cat_keep = torch.rand_like(source_cat_mask) < keep_prob
        return source_num_mask * num_keep.to(source_num_mask.dtype), source_cat_mask * cat_keep.to(source_cat_mask.dtype)

    def _build_model_kwargs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.source_encoder is not None:
            return {}
        keys = ("source_num", "source_cat", "source_num_mask", "source_cat_mask")
        return {key: batch[key] for key in keys if key in batch}

    def _ensure_metrics_header(self) -> None:
        if not self.is_main_process:
            return
        if self.metrics_path.exists():
            return
        self._write_schema()
        with self.metrics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "split", "loss", "loss_gaussian", "loss_multinomial", "lr"])

    def _write_schema(self) -> None:
        path = self.output_dir / "schema.json"
        if path.exists():
            return
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.schema, file, ensure_ascii=False, indent=2)

    def _log_metrics(self, step: int, split: str, metrics: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step,
                    split,
                    metrics.get("loss", 0.0),
                    metrics.get("loss_gaussian", 0.0),
                    metrics.get("loss_multinomial", 0.0),
                    metrics.get("lr", ""),
                ]
            )


def _cycle(loader: DataLoader) -> Iterator[dict[str, Tensor]]:
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _move_batch_to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _format_metrics(step: int, split: str, metrics: dict[str, float]) -> str:
    return (
        f"step={step} split={split} "
        f"loss={metrics.get('loss', 0.0):.6f} "
        f"gaussian={metrics.get('loss_gaussian', 0.0):.6f} "
        f"multinomial={metrics.get('loss_multinomial', 0.0):.6f}"
        + (
            f" lr={metrics['lr']:.8f}"
            if "lr" in metrics
            else ""
        )
    )


def _unwrap_ddp(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _has_trainable_parameters(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def _all_reduce_mean(
    metrics: dict[str, Tensor],
    distributed: bool,
) -> dict[str, Tensor]:
    if not distributed:
        return metrics
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        reduced[key] = tensor
    return reduced
