#!/usr/bin/env python
"""Train a raw-tabular MLP binary disease classifier."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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
    "auroc",
    "average_precision",
    "accuracy",
    "positive_rate",
    "lr",
]


class SourceDiseaseDataset(Dataset):
    def __init__(
        self,
        *,
        row_indices: np.ndarray,
        source_num: torch.Tensor,
        source_cat: torch.Tensor,
        source_num_mask: torch.Tensor,
        source_cat_mask: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self.row_indices = row_indices.astype(np.int64, copy=False)
        self.source_num = source_num
        self.source_cat = source_cat
        self.source_num_mask = source_num_mask
        self.source_cat_mask = source_cat_mask
        self.target = target
        lengths = {
            len(self.row_indices),
            len(source_num),
            len(source_cat),
            len(source_num_mask),
            len(source_cat_mask),
            len(target),
        }
        if len(lengths) != 1:
            raise ValueError("Dataset arrays have inconsistent row counts.")

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "row_index": torch.tensor(int(self.row_indices[index]), dtype=torch.long),
            "source_num": self.source_num[index],
            "source_cat": self.source_cat[index],
            "source_num_mask": self.source_num_mask[index],
            "source_cat_mask": self.source_cat_mask[index],
            "target": self.target[index],
        }


class RawMLPDiseaseClassifier(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        category_sizes: list[int],
        hidden_layers: list[int],
        dropout: float,
        activation: str,
        batch_norm: bool,
        normalization: str | None,
        pos_weight: float | None,
    ) -> None:
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.category_sizes = [int(size) for size in category_sizes]
        self.input_dim = 2 * self.n_num_features + sum(self.category_sizes) + len(self.category_sizes)
        normalization = resolve_normalization(normalization, batch_norm)
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def encode_raw(self, source_num: torch.Tensor, source_cat: torch.Tensor, source_num_mask: torch.Tensor, source_cat_mask: torch.Tensor) -> torch.Tensor:
        pieces = []
        if self.n_num_features > 0:
            pieces.extend([source_num * source_num_mask, source_num_mask])
        for index, size in enumerate(self.category_sizes):
            values = source_cat[:, index].clamp(min=0, max=size - 1)
            one_hot = F.one_hot(values, num_classes=size).to(dtype=source_num.dtype)
            pieces.append(one_hot * source_cat_mask[:, index : index + 1])
        if self.category_sizes:
            pieces.append(source_cat_mask)
        if not pieces:
            raise ValueError("Raw classifier requires at least one source feature.")
        return torch.cat(pieces, dim=1)

    def forward(self, source_num: torch.Tensor, source_cat: torch.Tensor, source_num_mask: torch.Tensor, source_cat_mask: torch.Tensor) -> torch.Tensor:
        x = self.encode_raw(source_num, source_cat, source_num_mask, source_cat_mask)
        return self.net(x).squeeze(-1)

    def loss(self, *, source_num: torch.Tensor, source_cat: torch.Tensor, source_num_mask: torch.Tensor, source_cat_mask: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(source_num, source_cat, source_num_mask, source_cat_mask)
        loss = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=self.pos_weight)
        return {"loss": loss, "logits": logits}


def main() -> None:
    args = parse_args()
    distributed = is_distributed_run()
    rank, world_size, local_rank = setup_distributed()
    config_path = Path(args.config)
    config = load_config(config_path)
    apply_overrides(config, args)
    seed = int(config.get("seed", 42)) + rank
    set_seed(seed)

    output_dir = Path(str(config["output_dir"]))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        copied_config = output_dir / config_path.name
        if config_path.resolve() != copied_config.resolve():
            shutil.copy2(config_path, copied_config)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    distributed_barrier(distributed)

    device = resolve_device(str(config.get("device", "cpu")), local_rank=local_rank)
    train_dataset = load_classifier_dataset(config, split="train")
    valid_dataset = load_classifier_dataset(config, split="valid")
    test_dataset = load_classifier_dataset(config, split="test")
    train_loader = make_loader(config, train_dataset, split="train", seed=seed, distributed=distributed)
    valid_loader = make_loader(config, valid_dataset, split="valid", seed=seed)
    test_loader = make_loader(config, test_dataset, split="test", seed=seed)

    schema = load_source_schema(config)
    pos_weight = resolve_pos_weight(config, train_dataset)
    model_config = config["model"]
    model = RawMLPDiseaseClassifier(
        n_num_features=int(schema["source"]["n_num_features"]),
        category_sizes=list(schema["source"]["category_sizes"]),
        hidden_layers=[int(value) for value in model_config.get("hidden_layers", [])],
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        batch_norm=bool(model_config.get("batch_norm", False)),
        normalization=model_config.get("normalization"),
        pos_weight=pos_weight,
    ).to(device)
    base_model = model
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler("cuda", enabled=bool(config["train"].get("mixed_precision", False)) and device.type == "cuda")

    parameter_count = count_parameters(base_model)
    forward_flops = estimate_mlp_forward_flops(base_model)
    metadata = {
        "input_dim": base_model.input_dim,
        "n_num_features": base_model.n_num_features,
        "n_cat_features": len(base_model.category_sizes),
        "category_size_sum": sum(base_model.category_sizes),
        "parameter_count": parameter_count,
        "estimated_forward_flops_per_sample": forward_flops,
        "hidden_layers": [int(value) for value in model_config.get("hidden_layers", [])],
        "normalization": resolve_normalization(model_config.get("normalization"), bool(model_config.get("batch_norm", False))),
        "target_disease": config["data"]["target_disease"],
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "test_rows": len(test_dataset),
        "pos_weight": pos_weight,
    }
    if rank == 0:
        (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(
            "raw MLP disease classifier setup: "
            f"device={device}, world_size={world_size}, disease={metadata['target_disease']}, "
            f"hidden={metadata['hidden_layers']}, params={parameter_count}, flops/sample={forward_flops}, "
            f"train/valid/test={len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}, pos_weight={pos_weight}",
            flush=True,
        )

    fit(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        config=config,
        output_dir=output_dir,
        metadata=metadata,
        summary_csv=args.summary_csv,
        figures_dir=args.figures_dir,
        loss_dir=args.loss_dir,
        distributed=distributed,
        is_main_process=rank == 0,
    )
    cleanup_distributed(distributed)


def fit(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
    metadata: dict[str, Any],
    summary_csv: Path | None,
    figures_dir: Path | None,
    loss_dir: Path | None,
    distributed: bool = False,
    is_main_process: bool = True,
) -> None:
    steps = int(config["train"]["steps"])
    log_every = int(config["logging"].get("log_every", 100))
    validate_every = int(config["logging"].get("validate_every", 100))
    checkpoint_every = int(config["logging"].get("checkpoint_every", steps))
    gradient_clip_norm = float(config["train"].get("gradient_clip_norm", 0.0))
    mixed_precision = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"
    early_config = config["train"].get("early_stopping", {})
    early_enabled = bool(early_config.get("enabled", False))
    patience = int(early_config.get("patience", 0) or round(steps * float(early_config.get("patience_fraction", 0.1))))
    min_delta = float(early_config.get("min_delta", 0.0))

    metrics_path = output_dir / "metrics.csv"
    if is_main_process:
        with metrics_path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=METRIC_COLUMNS).writeheader()

    best_valid_auroc = -float("inf")
    best_valid_loss = float("inf")
    best_step = 0
    steps_since_improvement = 0
    train_iter = iter(train_loader)
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    running_loss = 0.0
    running_count = 0

    for step in range(1, steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=mixed_precision):
            loss_dict = model_loss(
                model,
                source_num=batch["source_num"],
                source_cat=batch["source_cat"],
                source_num_mask=batch["source_num_mask"],
                source_cat_mask=batch["source_cat_mask"],
                target=batch["target"],
            )
            loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step={step}: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(batch["target"].numel())
        running_loss += float(loss.detach().cpu()) * batch_size
        running_count += batch_size
        if step % log_every == 0 or step == 1:
            reduced_running = torch.tensor([running_loss, float(running_count)], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(reduced_running, op=dist.ReduceOp.SUM)
            if is_main_process:
                train_metrics = {
                    "step": step,
                    "split": "train",
                    "loss": float(reduced_running[0].item() / max(1.0, reduced_running[1].item())),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                append_metrics(metrics_path, train_metrics)
                print(f"step={step} train_loss={train_metrics['loss']:.6f} lr={train_metrics['lr']:.3g}", flush=True)
            running_loss = 0.0
            running_count = 0

        should_stop = False
        if step % validate_every == 0 or step == steps:
            if is_main_process:
                valid_metrics = evaluate(model, valid_loader, device=device)
                valid_metrics.update({"step": step, "split": "valid", "lr": optimizer.param_groups[0]["lr"]})
                append_metrics(metrics_path, valid_metrics)
                valid_auroc = float(valid_metrics["auroc"])
                valid_loss = float(valid_metrics["loss"])
                print(
                    f"step={step} valid_loss={valid_loss:.6f} "
                    f"valid_auroc={valid_auroc:.6f} valid_ap={float(valid_metrics['average_precision']):.6f}",
                    flush=True,
                )
                improved = valid_auroc > best_valid_auroc + min_delta
                if improved:
                    best_valid_auroc = valid_auroc
                    best_valid_loss = valid_loss
                    best_step = step
                    steps_since_improvement = 0
                    save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scaler, config, step, best_valid_auroc)
                else:
                    steps_since_improvement += validate_every
                if early_enabled and patience > 0 and steps_since_improvement >= patience:
                    print(f"early stopping at step={step} patience={patience}", flush=True)
                    should_stop = True
            should_stop = distributed_broadcast_bool(should_stop, distributed, device)
            if should_stop:
                break

        if is_main_process and (step % checkpoint_every == 0 or step == steps):
            save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, scaler, config, step, best_valid_auroc)

    if not is_main_process:
        distributed_barrier(distributed)
        return

    best_path = output_dir / "checkpoint_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    else:
        best_step = step
        best_valid_loss = float("nan")
        save_checkpoint(best_path, model, optimizer, scaler, config, best_step, best_valid_auroc)

    test_metrics, predictions = evaluate(model, test_loader, device=device, return_predictions=True)
    test_metrics.update({"step": best_step, "split": "test", "lr": optimizer.param_groups[0]["lr"]})
    append_metrics(metrics_path, test_metrics)
    pd.DataFrame(predictions).to_csv(output_dir / "test_predictions.csv", index=False)
    last_train_metrics = last_metrics_for_split(metrics_path, "train")
    last_valid_metrics = last_metrics_for_split(metrics_path, "valid")

    summary = {
        **metadata,
        "run_name": str(output_dir.name),
        "output_dir": str(output_dir),
        "seed": int(config.get("seed", 42)),
        "learning_rate": float(config["train"]["learning_rate"]),
        "weight_decay": float(config["train"].get("weight_decay", 0.0)),
        "batch_size": int(config["train"]["batch_size"]),
        "steps_configured": steps,
        "best_step": int(best_step),
        "best_valid_auroc": float(best_valid_auroc),
        "best_valid_loss": float(best_valid_loss),
        "last_train_step": metric_value(last_train_metrics, "step"),
        "last_train_loss": metric_value(last_train_metrics, "loss"),
        "last_valid_step": metric_value(last_valid_metrics, "step"),
        "last_valid_loss": metric_value(last_valid_metrics, "loss"),
        "last_valid_auroc": metric_value(last_valid_metrics, "auroc"),
        "last_valid_average_precision": metric_value(last_valid_metrics, "average_precision"),
        "last_valid_accuracy": metric_value(last_valid_metrics, "accuracy"),
        "test_auroc": float(test_metrics["auroc"]),
        "test_average_precision": float(test_metrics["average_precision"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "test_positive_rate": float(test_metrics["positive_rate"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if loss_dir is not None:
        plot_loss_curve(metrics_path, loss_dir / f"{output_dir.parent.name}_{output_dir.name}_loss.png")
    if summary_csv is not None:
        with summary_file_lock(summary_csv):
            upsert_summary(summary_csv, summary)
            if figures_dir is not None:
                update_scatter_plots(summary_csv, figures_dir)
    print(json.dumps(summary, indent=2), flush=True)
    distributed_barrier(distributed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    losses = []
    weights = []
    logits = []
    targets = []
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss_dict = model_loss(
            model,
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            target=batch["target"],
        )
        batch_size = int(batch["target"].numel())
        losses.append(float(loss_dict["loss"].detach().cpu()) * batch_size)
        weights.append(batch_size)
        logits.append(loss_dict["logits"].detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
        rows.append(batch["row_index"].detach().cpu().numpy())
    y_logit = np.concatenate(logits) if logits else np.asarray([], dtype=np.float32)
    y_true = np.concatenate(targets).astype(np.int64) if targets else np.asarray([], dtype=np.int64)
    y_prob = sigmoid_numpy(y_logit)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    metrics = {
        "loss": float(sum(losses) / max(1, sum(weights))),
        "auroc": safe_auroc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "accuracy": float((y_pred == y_true).mean()) if y_true.size else 0.0,
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
    }
    if not return_predictions:
        return metrics
    row_values = np.concatenate(rows).astype(np.int64) if rows else np.asarray([], dtype=np.int64)
    return metrics, {
        "row_index": row_values,
        "y_true": y_true,
        "y_logit": y_logit,
        "y_probability": y_prob,
        "y_score": y_prob,
        "y_pred": y_pred,
    }


def load_classifier_dataset(config: dict[str, Any], *, split: str) -> SourceDiseaseDataset:
    data_config = config["data"]
    dataset_dir = _dataset_dir(data_config)
    source_groups = _string_list(data_config["source_groups"])
    split_group = str(data_config.get("split_group", source_groups[0]))
    split_indices = _split_indices(dataset_dir / split_group / "split.csv", split=split)
    labels = read_disease_labels(
        dataset_name=str(data_config["dataset_name"]),
        disease=str(data_config["target_disease"]),
        dataset_dir=dataset_dir,
        source_groups=source_groups,
    )
    kept_indices = []
    targets = []
    for row_index in split_indices:
        label = labels.get(int(row_index))
        if label is None:
            continue
        kept_indices.append(int(row_index))
        targets.append(float(label))
    max_rows = data_config.get("max_rows_per_split")
    if max_rows is not None:
        max_rows = int(max_rows)
        kept_indices = kept_indices[:max_rows]
        targets = targets[:max_rows]
    kept = np.asarray(kept_indices, dtype=np.int64)
    source = _load_group_arrays(dataset_dir, source_groups, indices=kept)
    return SourceDiseaseDataset(
        row_indices=kept,
        source_num=_as_tensor(source["num"], torch.float32),
        source_cat=_as_tensor(source["cat"], torch.long),
        source_num_mask=_as_tensor(source["num_mask"], torch.float32),
        source_cat_mask=_as_tensor(source["cat_mask"], torch.float32),
        target=torch.as_tensor(targets, dtype=torch.float32).contiguous(),
    )


def read_disease_labels(*, dataset_name: str, disease: str, dataset_dir: Path, source_groups: list[str]) -> dict[int, bool | None]:
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    column = f"disease_{disease}"
    row_metadata = read_row_metadata(dataset_dir, source_groups)
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if column not in fieldnames:
            raise ValueError(f"{path} does not contain {column}.")
        if row_metadata is not None and {"ID", "year"}.issubset(fieldnames):
            labels_by_key = {
                (row.get("ID", ""), row.get("year", "")): parse_label(row[column])
                for row in reader
            }
            labels: dict[int, bool | None] = {}
            missing = []
            for meta in row_metadata:
                key = (meta.get("ID", ""), meta.get("year", ""))
                if key not in labels_by_key:
                    missing.append(key)
                    continue
                labels[int(meta["row_index"])] = labels_by_key[key]
            if missing:
                raise ValueError(f"{path} is missing {len(missing)} row IDs; examples={missing[:5]}")
            return labels
        return {int(row["row_index"]): parse_label(row[column]) for row in reader}


def read_row_metadata(dataset_dir: Path, source_groups: list[str]) -> list[dict[str, str]] | None:
    for group in source_groups:
        path = dataset_dir / group / "row_metadata.csv"
        if path.exists():
            with path.open(newline="", errors="replace") as file:
                return list(csv.DictReader(file))
    return None


def parse_label(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "na", "null"}:
        return None
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    raise ValueError(f"Unsupported disease label value: {value!r}")


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


def resolve_pos_weight(config: dict[str, Any], train_dataset: SourceDiseaseDataset) -> float | None:
    model_config = config["model"]
    if model_config.get("pos_weight") is not None:
        return float(model_config["pos_weight"])
    class_weight = str(model_config.get("class_weight", "balanced")).lower()
    if class_weight in {"none", "off", "false"}:
        return None
    if class_weight != "balanced":
        raise ValueError(f"Unsupported class_weight: {class_weight}")
    positive = float(train_dataset.target.sum().item())
    negative = float(len(train_dataset) - positive)
    if positive <= 0 or negative <= 0:
        return 1.0
    return negative / positive


def make_loader(config: dict[str, Any], dataset: Dataset, *, split: str, seed: int, distributed: bool = False) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=split == "train", drop_last=False) if distributed and split == "train" else None
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, config: dict[str, Any], step: int, best_valid_auroc: float) -> None:
    raw_model = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "config": config,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_valid_auroc": best_valid_auroc,
            "input_dim": raw_model.input_dim,
            "category_sizes": raw_model.category_sizes,
            "n_num_features": raw_model.n_num_features,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def unwrap_model(model: nn.Module) -> RawMLPDiseaseClassifier:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def model_loss(model: nn.Module, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
    raw_model = unwrap_model(model)
    target = kwargs["target"]
    forward_kwargs = {
        "source_num": kwargs["source_num"],
        "source_cat": kwargs["source_cat"],
        "source_num_mask": kwargs["source_num_mask"],
        "source_cat_mask": kwargs["source_cat_mask"],
    }
    # During DDP training, calling model.module.loss(...) bypasses DDP forward hooks,
    # so gradients are not all-reduced. Evaluation runs on rank 0 only, so keep it on
    # the unwrapped module to avoid rank-mismatched DDP communication.
    if isinstance(model, DistributedDataParallel) and torch.is_grad_enabled():
        logits = model(**forward_kwargs)
    else:
        logits = raw_model(**forward_kwargs)
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=raw_model.pos_weight)
    return {"loss": loss, "logits": logits}


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(y_score)
    sorted_true = y_true[order]
    n_pos = float(sorted_true.sum())
    n_neg = float(sorted_true.size - sorted_true.sum())
    if n_pos <= 0 or n_neg <= 0:
        return 0.5
    ranks = np.arange(1, sorted_true.size + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[sorted_true == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_true.sum() <= 0:
        return 0.0
    order = np.argsort(-y_score)
    sorted_true = y_true[order].astype(np.float64)
    tp = np.cumsum(sorted_true)
    precision = tp / np.arange(1, sorted_true.size + 1, dtype=np.float64)
    return float((precision * sorted_true).sum() / max(1.0, sorted_true.sum()))


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32)


def make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def resolve_normalization(normalization: str | None, batch_norm: bool) -> str:
    if normalization is None:
        return "batch_norm" if batch_norm else "none"
    normalized = str(normalization).strip().lower()
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "none": "none",
        "batch": "batch_norm",
        "batchnorm": "batch_norm",
        "batch_norm": "batch_norm",
        "bn": "batch_norm",
        "layer": "layer_norm",
        "layernorm": "layer_norm",
        "layer_norm": "layer_norm",
        "ln": "layer_norm",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return aliases[normalized]


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def estimate_mlp_forward_flops(model: RawMLPDiseaseClassifier) -> int:
    flops = 0
    for module in model.net:
        if isinstance(module, nn.Linear):
            flops += 2 * int(module.in_features) * int(module.out_features)
    return int(flops)


def plot_loss_curve(metrics_path: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for split, group in df[df["loss"].notna()].groupby("split"):
        plt.plot(group["step"], group["loss"], marker="o", markersize=2, linewidth=1, label=split)
    plt.xlabel("step")
    plt.ylabel("BCE loss")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def last_metrics_for_split(metrics_path: Path, split: str) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty or "split" not in df.columns:
        return {}
    df = df[df["split"] == split].copy()
    if df.empty:
        return {}
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"].notna()]
    if df.empty:
        return {}
    return df.sort_values("step").iloc[-1].to_dict()


def metric_value(metrics: dict[str, Any], key: str) -> float:
    if key not in metrics:
        return float("nan")
    value = pd.to_numeric(pd.Series([metrics[key]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def upsert_summary(summary_csv: Path, summary: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_summary(summary)
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = df[df["output_dir"] != row["output_dir"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["parameter_count", "seed", "output_dir"]).reset_index(drop=True)
    df.to_csv(summary_csv, index=False)


@contextlib.contextmanager
def summary_file_lock(summary_csv: Path):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_csv.with_suffix(summary_csv.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    row["hidden_width"] = int(row["hidden_layers"][0]) if row.get("hidden_layers") else 0
    row["hidden_layers"] = json.dumps(row.get("hidden_layers", []))
    return row


def update_scatter_plots(summary_csv: Path, figures_dir: Path) -> None:
    df = pd.read_csv(summary_csv)
    df = enrich_summary_with_last_metrics(df)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(df, "parameter_count", "last_valid_auroc", figures_dir / "auroc_vs_parameters.png", "parameters", "last valid AUROC")
    plot_scatter(df, "estimated_forward_flops_per_sample", "last_valid_auroc", figures_dir / "auroc_vs_compute.png", "forward FLOPs / sample", "last valid AUROC")
    plot_scatter(df, "parameter_count", "last_valid_loss", figures_dir / "loss_vs_parameters.png", "parameters", "last valid BCE loss")
    plot_scatter(df, "estimated_forward_flops_per_sample", "last_valid_loss", figures_dir / "loss_vs_compute.png", "forward FLOPs / sample", "last valid BCE loss")


def enrich_summary_with_last_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["last_train_loss", "last_valid_loss", "last_valid_auroc", "last_valid_average_precision"]:
        if column not in df.columns:
            df[column] = float("nan")
    if "output_dir" not in df.columns:
        return df
    for index, row in df.iterrows():
        metrics_path = Path(str(row["output_dir"])) / "metrics.csv"
        if not metrics_path.exists():
            continue
        train_metrics = last_metrics_for_split(metrics_path, "train")
        valid_metrics = last_metrics_for_split(metrics_path, "valid")
        if pd.isna(df.at[index, "last_train_loss"]):
            df.at[index, "last_train_loss"] = metric_value(train_metrics, "loss")
        if pd.isna(df.at[index, "last_valid_loss"]):
            df.at[index, "last_valid_loss"] = metric_value(valid_metrics, "loss")
        if pd.isna(df.at[index, "last_valid_auroc"]):
            df.at[index, "last_valid_auroc"] = metric_value(valid_metrics, "auroc")
        if pd.isna(df.at[index, "last_valid_average_precision"]):
            df.at[index, "last_valid_average_precision"] = metric_value(valid_metrics, "average_precision")
    return df


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(7, 5))
    for width, group in df.groupby("hidden_width"):
        plt.scatter(group[x_col], group[y_col], label=str(width), alpha=0.8)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(title="width", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_raw_mlp_disease_classifier_diabetes.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--source-groups", nargs="+")
    parser.add_argument("--split-group")
    parser.add_argument("--target-disease")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--class-weight", choices=["balanced", "none"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--normalization", choices=["none", "batch_norm", "batchnorm", "bn", "layer_norm", "layernorm", "ln"])
    parser.add_argument("--hidden-layers", nargs="*", type=int)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--loss-dir", type=Path)
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
    if args.split_group is not None:
        config.setdefault("data", {})["split_group"] = args.split_group
    if args.target_disease is not None:
        config.setdefault("data", {})["target_disease"] = args.target_disease
    if args.max_rows_per_split is not None:
        config.setdefault("data", {})["max_rows_per_split"] = args.max_rows_per_split
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.class_weight is not None:
        config.setdefault("model", {})["class_weight"] = args.class_weight
    if args.dropout is not None:
        config.setdefault("model", {})["dropout"] = args.dropout
    if args.normalization is not None:
        config.setdefault("model", {})["normalization"] = args.normalization
        config.setdefault("model", {})["batch_norm"] = args.normalization in {"batch_norm", "batchnorm", "bn"}
    if args.hidden_layers is not None:
        config.setdefault("model", {})["hidden_layers"] = args.hidden_layers
    if args.steps is not None:
        config.setdefault("train", {})["steps"] = args.steps
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config.setdefault("train", {})["weight_decay"] = args.weight_decay
    if args.gradient_clip_norm is not None:
        config.setdefault("train", {})["gradient_clip_norm"] = args.gradient_clip_norm
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    if args.log_every is not None:
        config.setdefault("logging", {})["log_every"] = args.log_every
    if args.validate_every is not None:
        config.setdefault("logging", {})["validate_every"] = args.validate_every
    if args.checkpoint_every is not None:
        config.setdefault("logging", {})["checkpoint_every"] = args.checkpoint_every


def resolve_device(name: str, *, local_rank: int = 0) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, int]:
    if not is_distributed_run():
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def distributed_broadcast_bool(value: bool, distributed: bool, device: torch.device) -> bool:
    if not distributed:
        return value
    tensor = torch.tensor([1 if value else 0], dtype=torch.int64, device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


if __name__ == "__main__":
    main()
