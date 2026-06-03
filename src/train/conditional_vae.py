"""Training loop for conditional mixed-type VAE."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.conditional_vae import ConditionalMixedTypeVAE


METRIC_COLUMNS = [
    "step",
    "split",
    "loss",
    "loss_reconstruction",
    "loss_num",
    "loss_cat",
    "loss_kl",
    "loss_per_instance",
    "loss_reconstruction_per_instance",
    "loss_num_per_instance",
    "loss_cat_per_instance",
    "loss_kl_per_instance",
    "lr",
    "beta",
]
PER_INSTANCE_KEYS = (
    "loss",
    "loss_reconstruction",
    "loss_num",
    "loss_cat",
    "loss_kl",
)


class ConditionalVAETrainer:
    """Train a source-conditioned mixed-type VAE."""

    def __init__(
        self,
        *,
        model: ConditionalMixedTypeVAE,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        device: torch.device,
        output_dir: Path,
        config: dict[str, Any],
        schema: dict[str, Any],
        steps: int,
        beta: float,
        gradient_clip_norm: float | None,
        mixed_precision: bool,
        log_every: int,
        validate_every: int,
        checkpoint_every: int,
        save_best: bool,
        masked_target_dropout: float = 0.0,
        source_mask_dropout: float = 0.0,
        lr_warmup_steps: int = 0,
        beta_warmup_steps: int = 0,
        beta_warmup_start_step: int = 0,
        lr_schedule: str = "constant",
        min_learning_rate: float = 0.0,
        gradient_accumulation_steps: int = 1,
        grad_scaler_init_scale: float | None = None,
        early_stopping_patience_steps: int | None = None,
        early_stopping_min_delta: float = 0.0,
        is_main_process: bool = True,
        distributed: bool = False,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if beta < 0:
            raise ValueError("beta must not be negative.")
        if not 0 <= masked_target_dropout <= 1:
            raise ValueError("masked_target_dropout must be in [0, 1].")
        if not 0 <= source_mask_dropout <= 1:
            raise ValueError("source_mask_dropout must be in [0, 1].")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.output_dir = output_dir
        self.config = config
        self.schema = schema
        self.steps = int(steps)
        self.beta = float(beta)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.log_every = max(1, int(log_every))
        self.validate_every = max(1, int(validate_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.save_best = bool(save_best)
        self.masked_target_dropout = float(masked_target_dropout)
        self.source_mask_dropout = float(source_mask_dropout)
        self.lr_warmup_steps = max(0, int(lr_warmup_steps))
        self.beta_warmup_steps = max(0, int(beta_warmup_steps))
        self.beta_warmup_start_step = max(0, int(beta_warmup_start_step))
        self.lr_schedule = _resolve_lr_schedule(lr_schedule)
        self.min_learning_rate = float(min_learning_rate)
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must not be negative.")
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.early_stopping_patience_steps = (
            None
            if early_stopping_patience_steps is None
            else max(1, int(early_stopping_patience_steps))
        )
        self.early_stopping_min_delta = max(0.0, float(early_stopping_min_delta))
        self.is_main_process = bool(is_main_process)
        self.distributed = bool(distributed)
        scaler_kwargs = {}
        if grad_scaler_init_scale is not None:
            scaler_kwargs["init_scale"] = float(grad_scaler_init_scale)
        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_mixed_precision,
            **scaler_kwargs,
        )
        self.best_valid_loss = float("inf")
        self.best_valid_step = 0
        self.stopped_early = False
        self.metrics_path = self.output_dir / "metrics.csv"
        self._step_index = 0
        self.base_lrs = [
            float(parameter_group["lr"])
            for parameter_group in self.optimizer.param_groups
        ]

    def fit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main_process:
            self._write_schema()
            self._ensure_metrics_header()
        if self.distributed:
            dist.barrier()
        train_iterator = _cycle(self.train_loader)
        running: dict[str, float] = {}
        final_step = 0
        for step in range(1, self.steps + 1):
            final_step = step
            metrics = self.train_step(train_iterator)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if step == 1 or step % self.log_every == 0:
                averaged = _aggregate_metric_totals(running)
                running.clear()
                if self.is_main_process:
                    self._log_metrics(step, "train", averaged)
                    print(_format_metrics(step, "train", averaged), flush=True)

            if self.valid_loader is not None and step % self.validate_every == 0:
                valid_metrics = self.validate()
                if self.is_main_process:
                    self._log_metrics(step, "valid", valid_metrics)
                    print(_format_metrics(step, "valid", valid_metrics), flush=True)
                improved = (
                    valid_metrics["loss"]
                    < self.best_valid_loss - self.early_stopping_min_delta
                )
                if improved:
                    self.best_valid_loss = valid_metrics["loss"]
                    self.best_valid_step = step
                    if self.save_best and self.is_main_process:
                        self.save_checkpoint(step, self.output_dir / "checkpoint_best.pt")
                elif (
                    self.early_stopping_patience_steps is not None
                    and self.best_valid_step > 0
                    and step - self.best_valid_step >= self.early_stopping_patience_steps
                ):
                    self.stopped_early = True
                    if self.is_main_process:
                        print(
                            "early stopping: "
                            f"step={step} best_valid_step={self.best_valid_step} "
                            f"best_valid_loss={self.best_valid_loss:.6f} "
                            f"patience_steps={self.early_stopping_patience_steps}",
                            flush=True,
                        )
                    break

            if self.is_main_process and step % self.checkpoint_every == 0:
                self.save_checkpoint(step, self.output_dir / "checkpoint_latest.pt")

        if self.is_main_process:
            self.save_checkpoint(final_step, self.output_dir / "checkpoint_latest.pt")

    def train_step(self, train_iterator: Iterator[dict[str, Tensor]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._apply_lr_warmup(self._step_index + 1)
        lr = self._current_learning_rate()
        beta = self._scheduled_beta(self._step_index + 1)
        batches = [
            _move_batch_to_device(next(train_iterator), self.device)
            for _ in range(self.gradient_accumulation_steps)
        ]
        total_instances = sum(int(batch["target_num"].shape[0]) for batch in batches)
        metric_totals: dict[str, float] = {}
        for batch in batches:
            source = _apply_source_mask_dropout(
                batch,
                probability=self.source_mask_dropout,
            )
            context = _apply_masked_target_dropout(
                batch,
                probability=self.masked_target_dropout,
            )
            batch_size = int(batch["target_num"].shape[0])
            with autocast("cuda", enabled=self.use_mixed_precision):
                losses = self._model_loss(
                    source_num=source["source_num"],
                    source_cat=source["source_cat"],
                    source_num_mask=source["source_num_mask"],
                    source_cat_mask=source["source_cat_mask"],
                    target_num=batch["target_num"],
                    target_cat=batch["target_cat"],
                    target_num_mask=batch["target_num_mask"],
                    target_cat_mask=batch["target_cat_mask"],
                    context_target_num=context["target_num"],
                    context_target_cat=context["target_cat"],
                    context_target_num_mask=context["target_num_mask"],
                    context_target_cat_mask=context["target_cat_mask"],
                    beta=beta,
                    sample_latent=True,
                    sample_loss_weights=batch["sample_loss_weights"],
                )
                loss = losses["loss"] * (batch_size / total_instances)
            self.scaler.scale(loss).backward()
            metrics = _logging_metrics_from_losses(losses)
            for key, value in metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        metric_totals["lr"] = lr * metric_totals.get("step_count", 1.0)
        metric_totals["beta"] = beta * metric_totals.get("step_count", 1.0)
        self._step_index += 1
        metric_totals = _reduce_metric_totals(metric_totals) if self.distributed else metric_totals
        return metric_totals

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.valid_loader is None:
            raise ValueError("valid_loader is not configured.")
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.valid_loader:
            batch = _move_batch_to_device(batch, self.device)
            losses = self._model_loss(
                source_num=batch["source_num"],
                source_cat=batch["source_cat"],
                source_num_mask=batch["source_num_mask"],
                source_cat_mask=batch["source_cat_mask"],
                target_num=batch["target_num"],
                target_cat=batch["target_cat"],
                target_num_mask=batch["target_num_mask"],
                target_cat_mask=batch["target_cat_mask"],
                beta=self.beta,
                sample_latent=False,
                sample_loss_weights=batch["sample_loss_weights"],
            )
            batch_size = int(batch["target_num"].shape[0])
            count += batch_size
            for key in PER_INSTANCE_KEYS:
                totals[key] = totals.get(key, 0.0) + float(losses[key].cpu()) * batch_size
            for key in (
                "loss_observed_sum",
                "loss_reconstruction_observed_sum",
                "loss_num_observed_sum",
                "loss_cat_observed_sum",
                "loss_kl_observed_sum",
                "observed_feature_count",
                "observed_num_count",
                "observed_cat_count",
            ):
                totals[key] = totals.get(key, 0.0) + float(losses[key].cpu())
        totals["instance_count"] = float(count)
        totals = _reduce_metric_totals(totals) if self.distributed else totals
        count = int(totals.pop("instance_count"))
        if count == 0:
            raise ValueError("Validation loader produced no batches.")
        metrics = {f"{key}_per_instance": totals[key] / count for key in PER_INSTANCE_KEYS}
        metrics["loss"] = _safe_float_divide(
            totals["loss_observed_sum"],
            totals["observed_feature_count"],
        )
        metrics["loss_reconstruction"] = _safe_float_divide(
            totals["loss_reconstruction_observed_sum"],
            totals["observed_feature_count"],
        )
        metrics["loss_num"] = _safe_float_divide(
            totals["loss_num_observed_sum"],
            totals["observed_num_count"],
        )
        metrics["loss_cat"] = _safe_float_divide(
            totals["loss_cat_observed_sum"],
            totals["observed_cat_count"],
        )
        metrics["loss_kl"] = _safe_float_divide(
            totals["loss_kl_observed_sum"],
            totals["observed_feature_count"],
        )
        metrics["beta"] = self.beta
        return metrics

    def save_checkpoint(self, step: int, path: Path) -> None:
        model = _unwrap_checkpoint_model(self.model)
        torch.save(
            {
                "step": int(step),
                "config": self.config,
                "schema": self.schema,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_valid_loss": self.best_valid_loss,
                "best_valid_step": self.best_valid_step,
                "stopped_early": self.stopped_early,
            },
            path,
        )

    def _model_loss(self, **kwargs: Any) -> dict[str, Tensor]:
        if hasattr(self.model, "loss"):
            return self.model.loss(**kwargs)
        return self.model(**kwargs)

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply_lr_warmup(self, step: int) -> None:
        for base_lr, parameter_group in zip(self.base_lrs, self.optimizer.param_groups):
            parameter_group["lr"] = self._scheduled_learning_rate(base_lr, step)

    def _scheduled_learning_rate(self, base_lr: float, step: int) -> float:
        if self.lr_warmup_steps > 0 and step <= self.lr_warmup_steps:
            return base_lr * float(step) / float(self.lr_warmup_steps)
        if self.lr_schedule == "constant":
            return base_lr
        if self.lr_schedule == "cosine":
            decay_steps = max(1, self.steps - self.lr_warmup_steps)
            progress = min(
                1.0,
                max(0.0, float(step - self.lr_warmup_steps) / float(decay_steps)),
            )
            min_lr = min(self.min_learning_rate, base_lr)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (base_lr - min_lr) * cosine
        raise RuntimeError(f"Unsupported lr_schedule: {self.lr_schedule!r}")

    def _scheduled_beta(self, step: int) -> float:
        if self.beta_warmup_steps <= 0:
            return self.beta
        if step <= self.beta_warmup_start_step:
            return 0.0
        if step <= self.beta_warmup_steps:
            warmup_span = max(1, self.beta_warmup_steps - self.beta_warmup_start_step)
            return self.beta * float(step - self.beta_warmup_start_step) / float(warmup_span)
        return self.beta

    def _write_schema(self) -> None:
        path = self.output_dir / "schema.json"
        if path.exists():
            return
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.schema, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _ensure_metrics_header(self) -> None:
        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8", newline="") as file:
                current_header = next(csv.reader(file), None)
            if current_header == METRIC_COLUMNS:
                return
            raise ValueError(f"{self.metrics_path} has an old metrics schema.")
        with self.metrics_path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(METRIC_COLUMNS)

    def _log_metrics(self, step: int, split: str, metrics: dict[str, float]) -> None:
        with self.metrics_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step if column == "step" else
                    split if column == "split" else
                    metrics.get("lr", "") if column == "lr" else
                    metrics.get(column, 0.0)
                    for column in METRIC_COLUMNS
                ]
            )


def _cycle(loader: DataLoader) -> Iterator[dict[str, Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _unwrap_checkpoint_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "model"):
        model = model.model
    return model


def _apply_masked_target_dropout(
    batch: dict[str, Tensor],
    *,
    probability: float,
) -> dict[str, Tensor]:
    if probability <= 0:
        return {
            "target_num": batch["target_num"],
            "target_cat": batch["target_cat"],
            "target_num_mask": batch["target_num_mask"],
            "target_cat_mask": batch["target_cat_mask"],
        }

    target_num_mask = batch["target_num_mask"].to(dtype=torch.float32).clamp(0.0, 1.0)
    target_cat_mask = batch["target_cat_mask"].to(dtype=torch.float32).clamp(0.0, 1.0)
    keep_num = (torch.rand_like(target_num_mask) >= probability).to(target_num_mask.dtype)
    keep_cat = (torch.rand_like(target_cat_mask) >= probability).to(target_cat_mask.dtype)
    context_num_mask = target_num_mask * keep_num
    context_cat_mask = target_cat_mask * keep_cat
    return {
        "target_num": batch["target_num"] * context_num_mask.to(batch["target_num"].dtype),
        "target_cat": torch.where(
            context_cat_mask.to(dtype=torch.bool),
            batch["target_cat"],
            torch.zeros_like(batch["target_cat"]),
        ),
        "target_num_mask": context_num_mask,
        "target_cat_mask": context_cat_mask,
    }


def _apply_source_mask_dropout(
    batch: dict[str, Tensor],
    *,
    probability: float,
) -> dict[str, Tensor]:
    if probability <= 0:
        return {
            "source_num": batch["source_num"],
            "source_cat": batch["source_cat"],
            "source_num_mask": batch["source_num_mask"],
            "source_cat_mask": batch["source_cat_mask"],
        }

    source_num_mask = batch["source_num_mask"].to(dtype=torch.float32).clamp(0.0, 1.0)
    source_cat_mask = batch["source_cat_mask"].to(dtype=torch.float32).clamp(0.0, 1.0)
    keep_num = (torch.rand_like(source_num_mask) >= probability).to(source_num_mask.dtype)
    keep_cat = (torch.rand_like(source_cat_mask) >= probability).to(source_cat_mask.dtype)
    masked_num_mask = source_num_mask * keep_num
    masked_cat_mask = source_cat_mask * keep_cat
    return {
        "source_num": batch["source_num"] * masked_num_mask.to(batch["source_num"].dtype),
        "source_cat": torch.where(
            masked_cat_mask.to(dtype=torch.bool),
            batch["source_cat"],
            torch.zeros_like(batch["source_cat"]),
        ),
        "source_num_mask": masked_num_mask,
        "source_cat_mask": masked_cat_mask,
    }


def _logging_metrics_from_losses(losses: dict[str, Tensor]) -> dict[str, float]:
    batch_size = float(losses["mu"].shape[0])
    metrics = {
        "loss": losses["loss_per_observed_feature"],
        "loss_reconstruction": losses["loss_reconstruction_per_observed_feature"],
        "loss_num": losses["loss_num_per_observed_feature"],
        "loss_cat": losses["loss_cat_per_observed_feature"],
        "loss_kl": losses["loss_kl_per_observed_feature"],
        "loss_per_instance": losses["loss"],
        "loss_reconstruction_per_instance": losses["loss_reconstruction"],
        "loss_num_per_instance": losses["loss_num"],
        "loss_cat_per_instance": losses["loss_cat"],
        "loss_kl_per_instance": losses["loss_kl"],
        "loss_per_instance_sum": losses["loss"] * batch_size,
        "loss_reconstruction_per_instance_sum": losses["loss_reconstruction"] * batch_size,
        "loss_num_per_instance_sum": losses["loss_num"] * batch_size,
        "loss_cat_per_instance_sum": losses["loss_cat"] * batch_size,
        "loss_kl_per_instance_sum": losses["loss_kl"] * batch_size,
        "instance_count": losses["mu"].new_tensor(batch_size),
        "loss_observed_sum": losses["loss_observed_sum"],
        "loss_reconstruction_observed_sum": losses["loss_reconstruction_observed_sum"],
        "loss_num_observed_sum": losses["loss_num_observed_sum"],
        "loss_cat_observed_sum": losses["loss_cat_observed_sum"],
        "loss_kl_observed_sum": losses["loss_kl_observed_sum"],
        "observed_feature_count": losses["observed_feature_count"],
        "observed_num_count": losses["observed_num_count"],
        "observed_cat_count": losses["observed_cat_count"],
        "step_count": losses["mu"].new_tensor(1.0),
    }
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _aggregate_metric_totals(totals: dict[str, float]) -> dict[str, float]:
    observed = totals.get("observed_feature_count", 0.0)
    observed_num = totals.get("observed_num_count", 0.0)
    observed_cat = totals.get("observed_cat_count", 0.0)
    instances = totals.get("instance_count", 0.0)
    return {
        "loss": _safe_float_divide(totals.get("loss_observed_sum", 0.0), observed),
        "loss_reconstruction": _safe_float_divide(
            totals.get("loss_reconstruction_observed_sum", 0.0),
            observed,
        ),
        "loss_num": _safe_float_divide(totals.get("loss_num_observed_sum", 0.0), observed_num),
        "loss_cat": _safe_float_divide(totals.get("loss_cat_observed_sum", 0.0), observed_cat),
        "loss_kl": _safe_float_divide(totals.get("loss_kl_observed_sum", 0.0), observed),
        "loss_per_instance": _safe_float_divide(
            totals.get("loss_per_instance_sum", 0.0),
            instances,
        ),
        "loss_reconstruction_per_instance": _safe_float_divide(
            totals.get("loss_reconstruction_per_instance_sum", 0.0),
            instances,
        ),
        "loss_num_per_instance": _safe_float_divide(
            totals.get("loss_num_per_instance_sum", 0.0),
            instances,
        ),
        "loss_cat_per_instance": _safe_float_divide(
            totals.get("loss_cat_per_instance_sum", 0.0),
            instances,
        ),
        "loss_kl_per_instance": _safe_float_divide(
            totals.get("loss_kl_per_instance_sum", 0.0),
            instances,
        ),
        "lr": totals.get("lr", 0.0) / max(totals.get("step_count", 1.0), 1.0),
        "beta": totals.get("beta", 0.0) / max(totals.get("step_count", 1.0), 1.0),
    }


def _resolve_lr_schedule(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "constant"
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "constant",
        "off": "constant",
        "flat": "constant",
        "cos": "cosine",
        "cosine_decay": "cosine",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be one of: constant, cosine.")
    return normalized


def _reduce_metric_totals(totals: dict[str, float]) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return totals
    keys = sorted(totals)
    values = torch.tensor(
        [float(totals[key]) for key in keys],
        dtype=torch.float64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, values.cpu())}


def _format_metrics(step: int, split: str, metrics: dict[str, float]) -> str:
    return (
        f"[{split}] step={step} loss={metrics.get('loss', 0.0):.6f} "
        f"recon={metrics.get('loss_reconstruction', 0.0):.6f} "
        f"num={metrics.get('loss_num', 0.0):.6f} "
        f"cat={metrics.get('loss_cat', 0.0):.6f} "
        f"kl={metrics.get('loss_kl', 0.0):.6f} "
        f"beta={metrics.get('beta', 0.0):.6f}"
    )


def _safe_float_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
