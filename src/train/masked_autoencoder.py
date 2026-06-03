"""Training loop for masked source autoencoder pretraining."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.masked_autoencoder import MaskedSourceAutoencoder, apply_source_feature_dropout
from train.dataset import (
    _as_tensor,
    _dataset_dir,
    _load_group_arrays,
    _read_metadata,
    _schema_from_metadata,
    _split_indices,
    _string_list,
)


METRIC_COLUMNS = [
    "step",
    "split",
    "loss",
    "loss_num",
    "loss_cat",
    "observed_feature_count",
    "observed_num_count",
    "observed_cat_count",
    "lr",
]


class GroupedSourceDataset(Dataset):
    def __init__(
        self,
        *,
        source_num: Tensor,
        source_cat: Tensor,
        source_num_mask: Tensor,
        source_cat_mask: Tensor,
    ) -> None:
        self.source_num = source_num
        self.source_cat = source_cat
        self.source_num_mask = source_num_mask
        self.source_cat_mask = source_cat_mask
        lengths = {
            len(source_num),
            len(source_cat),
            len(source_num_mask),
            len(source_cat_mask),
        }
        if len(lengths) != 1:
            raise ValueError("Source dataset arrays have inconsistent row counts.")

    def __len__(self) -> int:
        return int(self.source_num.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "source_num": self.source_num[index],
            "source_cat": self.source_cat[index],
            "source_num_mask": self.source_num_mask[index],
            "source_cat_mask": self.source_cat_mask[index],
        }


def create_masked_ae_dataloader(
    config: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = load_source_dataset(config, split=split)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def load_source_dataset(config: dict[str, Any], *, split: str) -> GroupedSourceDataset:
    data_config = config["data"]
    dataset_dir = _dataset_dir(data_config)
    source_groups = _string_list(data_config["source_groups"])
    split_group = str(data_config.get("split_group", source_groups[0]))
    indices = _split_indices(dataset_dir / split_group / "split.csv", split=split)
    source = _load_group_arrays(dataset_dir, source_groups, indices=indices)
    max_rows = data_config.get("max_rows_per_split")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("data.max_rows_per_split must be positive when set.")
        source = {key: value[:max_rows] for key, value in source.items()}
    return GroupedSourceDataset(
        source_num=_as_tensor(source["num"], torch.float32),
        source_cat=_as_tensor(source["cat"], torch.long),
        source_num_mask=_as_tensor(source["num_mask"], torch.float32),
        source_cat_mask=_as_tensor(source["cat_mask"], torch.float32),
    )


def load_source_schema(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    dataset_dir = _dataset_dir(data_config)
    source_groups = _string_list(data_config["source_groups"])
    metadata = [_read_metadata(dataset_dir / group) for group in source_groups]
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_name": data_config["dataset_name"],
        "source_groups": source_groups,
        "source": _schema_from_metadata(metadata),
    }


class MaskedAutoencoderTrainer:
    def __init__(
        self,
        *,
        model: MaskedSourceAutoencoder,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        device: torch.device,
        output_dir: Path,
        config: dict[str, Any],
        schema: dict[str, Any],
        steps: int,
        mask_probability: float,
        gradient_clip_norm: float | None,
        mixed_precision: bool,
        log_every: int,
        validate_every: int,
        checkpoint_every: int,
        save_best: bool,
        start_step: int = 0,
        best_valid_loss: float | None = None,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not 0 <= mask_probability <= 1:
            raise ValueError("mask_probability must be in [0, 1].")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.output_dir = output_dir
        self.config = config
        self.schema = schema
        self.steps = int(steps)
        self.mask_probability = float(mask_probability)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.log_every = max(1, int(log_every))
        self.validate_every = max(1, int(validate_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.save_best = bool(save_best)
        self.scaler = GradScaler("cuda", enabled=self.use_mixed_precision)
        self.start_step = int(start_step)
        self.best_valid_loss = float("inf") if best_valid_loss is None else float(best_valid_loss)
        self.metrics_path = self.output_dir / "metrics.csv"

    def fit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_schema()
        self._ensure_metrics_header()
        train_iterator = _cycle(self.train_loader)
        running: dict[str, float] = {}
        running_count = 0
        end_step = self.start_step + self.steps
        for step in range(self.start_step + 1, end_step + 1):
            metrics = self.train_step(train_iterator)
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            running_count += 1

            if step == 1 or step % self.log_every == 0:
                averaged = _average_metrics(running, running_count)
                running.clear()
                running_count = 0
                self._log_metrics(step, "train", averaged)
                print(_format_metrics(step, "train", averaged), flush=True)

            if self.valid_loader is not None and step % self.validate_every == 0:
                valid_metrics = self.validate()
                self._log_metrics(step, "valid", valid_metrics)
                print(_format_metrics(step, "valid", valid_metrics), flush=True)
                if valid_metrics["loss"] < self.best_valid_loss:
                    self.best_valid_loss = valid_metrics["loss"]
                    if self.save_best:
                        self.save_checkpoint(step, self.output_dir / "checkpoint_best.pt")

            if step % self.checkpoint_every == 0:
                self.save_checkpoint(step, self.output_dir / "checkpoint_latest.pt")
        self.save_checkpoint(end_step, self.output_dir / "checkpoint_latest.pt")

    def train_step(self, train_iterator: Iterator[dict[str, Tensor]]) -> dict[str, float]:
        self.model.train()
        batch = _move_batch_to_device(next(train_iterator), self.device)
        dropout = apply_source_feature_dropout(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            probability=self.mask_probability,
        )
        self.optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=self.use_mixed_precision):
            losses = self.model.loss(
                input_num=dropout["input_num"],
                input_cat=dropout["input_cat"],
                input_num_mask=dropout["input_num_mask"],
                input_cat_mask=dropout["input_cat_mask"],
                target_num=batch["source_num"],
                target_cat=batch["source_cat"],
                loss_num_mask=dropout["loss_num_mask"],
                loss_cat_mask=dropout["loss_cat_mask"],
            )
            loss = losses["loss"]
        self.scaler.scale(loss).backward()
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        lr = self._current_learning_rate()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        metrics = _metrics_from_losses(losses)
        metrics["lr"] = lr
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.valid_loader is None:
            raise ValueError("valid_loader is not configured.")
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in self.valid_loader:
            batch = _move_batch_to_device(batch, self.device)
            dropout = apply_source_feature_dropout(
                source_num=batch["source_num"],
                source_cat=batch["source_cat"],
                source_num_mask=batch["source_num_mask"],
                source_cat_mask=batch["source_cat_mask"],
                probability=self.mask_probability,
            )
            losses = self.model.loss(
                input_num=dropout["input_num"],
                input_cat=dropout["input_cat"],
                input_num_mask=dropout["input_num_mask"],
                input_cat_mask=dropout["input_cat_mask"],
                target_num=batch["source_num"],
                target_cat=batch["source_cat"],
                loss_num_mask=dropout["loss_num_mask"],
                loss_cat_mask=dropout["loss_cat_mask"],
            )
            count += 1
            for key in ("loss", "loss_num", "loss_cat"):
                totals[key] = totals.get(key, 0.0) + float(losses[key].detach().cpu())
            for key in ("observed_feature_count", "observed_num_count", "observed_cat_count"):
                totals[key] = totals.get(key, 0.0) + float(losses[key].detach().cpu())
        if count == 0:
            raise ValueError("Validation loader produced no batches.")
        metrics = {key: totals[key] / count for key in ("loss", "loss_num", "loss_cat")}
        for key in ("observed_feature_count", "observed_num_count", "observed_cat_count"):
            metrics[key] = totals[key] / count
        metrics["lr"] = self._current_learning_rate()
        return metrics

    def save_checkpoint(self, step: int, path: Path) -> None:
        torch.save(
            {
                "step": int(step),
                "config": self.config,
                "schema": self.schema,
                "model_state_dict": self.model.state_dict(),
                "source_encoder_state_dict": self.model.source_encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_valid_loss": self.best_valid_loss,
            },
            path,
        )

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _write_schema(self) -> None:
        path = self.output_dir / "schema.json"
        if not path.exists():
            path.write_text(json.dumps(self.schema, indent=2) + "\n")

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
            writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
            writer.writerow({"step": step, "split": split, **metrics})


def _move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _cycle(loader: DataLoader) -> Iterator[dict[str, Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _metrics_from_losses(losses: dict[str, Tensor]) -> dict[str, float]:
    return {
        "loss": float(losses["loss"].detach().cpu()),
        "loss_num": float(losses["loss_num"].detach().cpu()),
        "loss_cat": float(losses["loss_cat"].detach().cpu()),
        "observed_feature_count": float(losses["observed_feature_count"].detach().cpu()),
        "observed_num_count": float(losses["observed_num_count"].detach().cpu()),
        "observed_cat_count": float(losses["observed_cat_count"].detach().cpu()),
    }


def _average_metrics(totals: dict[str, float], count: int) -> dict[str, float]:
    if not totals:
        return {}
    return {key: value / count for key, value in totals.items()}


def _format_metrics(step: int, split: str, metrics: dict[str, float]) -> str:
    return (
        f"[{split}] step={step} "
        f"loss={metrics['loss']:.4f} "
        f"num={metrics['loss_num']:.4f} "
        f"cat={metrics['loss_cat']:.4f} "
        f"observed={metrics['observed_feature_count']:.1f}"
    )
