#!/usr/bin/env python
"""Train a binary disease classifier on top of a masked-AE source encoder."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models.masked_ae_classifier import MaskedAEClassifier
from models.masked_autoencoder import MaskedSourceAutoencoder
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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    apply_overrides(config, args)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / config_path.name)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    device = resolve_device(str(config.get("device", "cpu")))
    train_dataset = load_classifier_dataset(config, split="train")
    valid_dataset = load_classifier_dataset(config, split="valid")
    test_dataset = load_classifier_dataset(config, split="test")
    train_loader = make_loader(config, train_dataset, split="train", seed=seed)
    valid_loader = make_loader(config, valid_dataset, split="valid", seed=seed)
    test_loader = make_loader(config, test_dataset, split="test", seed=seed)

    autoencoder, ae_checkpoint = load_autoencoder(config)
    pos_weight = resolve_pos_weight(config, train_dataset)
    model_config = config["model"]
    model = MaskedAEClassifier(
        autoencoder=autoencoder,
        head_hidden_layers=model_config.get("head_hidden_layers", []),
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        batch_norm=bool(model_config.get("batch_norm", False)),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        pos_weight=pos_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    scaler = GradScaler("cuda", enabled=bool(config["train"].get("mixed_precision", False)) and device.type == "cuda")

    print(
        "masked AE downstream setup: "
        f"device={device}, disease={config['data']['target_disease']}, "
        f"pretrained={config['model']['pretrained_autoencoder_path']}, "
        f"ae_best_valid_loss={ae_checkpoint.get('best_valid_loss')}, "
        f"freeze_backbone={model.freeze_backbone}, pos_weight={pos_weight}, "
        f"train/valid/test={len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}",
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
    )


def fit(
    *,
    model: MaskedAEClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    steps = int(config["train"]["steps"])
    log_every = int(config["logging"].get("log_every", 100))
    validate_every = int(config["logging"].get("validate_every", 100))
    checkpoint_every = int(config["logging"].get("checkpoint_every", steps))
    gradient_clip_norm = float(config["train"].get("gradient_clip_norm", 0.0))
    mixed_precision = bool(config["train"].get("mixed_precision", False)) and device.type == "cuda"

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=METRIC_COLUMNS).writeheader()

    best_valid_auroc = -float("inf")
    best_step = 0
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    for step in range(1, steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=mixed_precision):
            loss_dict = model.loss(
                source_num=batch["source_num"],
                source_cat=batch["source_cat"],
                source_num_mask=batch["source_num_mask"],
                source_cat_mask=batch["source_cat_mask"],
                target=batch["target"],
            )
            loss = loss_dict["loss"]
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
            train_metrics = {
                "step": step,
                "split": "train",
                "loss": running_loss / max(1, running_count),
                "auroc": "",
                "average_precision": "",
                "positive_rate": "",
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_metrics(metrics_path, train_metrics)
            print(
                f"step={step} train_loss={train_metrics['loss']:.6f} "
                f"lr={train_metrics['lr']:.3g}",
                flush=True,
            )
            running_loss = 0.0
            running_count = 0

        if step % validate_every == 0 or step == steps:
            valid_metrics = evaluate(model, valid_loader, device=device)
            valid_metrics.update({"step": step, "split": "valid", "lr": optimizer.param_groups[0]["lr"]})
            append_metrics(metrics_path, valid_metrics)
            print(
                f"step={step} valid_loss={valid_metrics['loss']:.6f} "
                f"valid_auroc={valid_metrics['auroc']:.6f}",
                flush=True,
            )
            if float(valid_metrics["auroc"]) > best_valid_auroc:
                best_valid_auroc = float(valid_metrics["auroc"])
                best_step = step
                save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scaler, config, step, best_valid_auroc)
        if step % checkpoint_every == 0 or step == steps:
            save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, scaler, config, step, best_valid_auroc)

    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, predictions = evaluate(model, test_loader, device=device, return_predictions=True)
    test_metrics.update({"step": best_step, "split": "test", "lr": optimizer.param_groups[0]["lr"]})
    append_metrics(metrics_path, test_metrics)
    pd.DataFrame(predictions).to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "best_step": best_step,
        "best_valid_auroc": best_valid_auroc,
        "test_auroc": float(test_metrics["auroc"]),
        "test_average_precision": float(test_metrics["average_precision"]),
        "test_loss": float(test_metrics["loss"]),
        "test_positive_rate": float(test_metrics["positive_rate"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


@torch.no_grad()
def evaluate(
    model: MaskedAEClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    row_indices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss_dict = model.loss(
            source_num=batch["source_num"],
            source_cat=batch["source_cat"],
            source_num_mask=batch["source_num_mask"],
            source_cat_mask=batch["source_cat_mask"],
            target=batch["target"],
        )
        n = int(batch["target"].numel())
        losses.append(float(loss_dict["loss"].detach().cpu()) * n)
        weights.append(n)
        row_indices.append(batch["row_index"].detach().cpu().numpy())
        targets.append(batch["target"].detach().cpu().numpy())
        logits.append(loss_dict["logit"].detach().cpu().numpy())
        probabilities.append(loss_dict["probability"].detach().cpu().numpy())
    y_true = np.concatenate(targets) if targets else np.empty((0,), dtype=np.float32)
    y_prob = np.concatenate(probabilities) if probabilities else np.empty((0,), dtype=np.float32)
    y_logit = np.concatenate(logits) if logits else np.empty((0,), dtype=np.float32)
    rows = np.concatenate(row_indices) if row_indices else np.empty((0,), dtype=np.int64)
    auroc = safe_auroc(y_true, y_prob)
    metrics = {
        "loss": float(sum(losses) / max(1, sum(weights))),
        "auroc": auroc,
        "average_precision": safe_average_precision(y_true, y_prob),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
    }
    if not return_predictions:
        return metrics
    return metrics, {
        "row_index": rows,
        "y_true": y_true.astype(np.int64),
        "y_logit": y_logit,
        "y_probability": y_prob,
        "y_score": y_prob,
        "y_pred": (y_prob >= 0.5).astype(np.int64),
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


def read_disease_labels(
    *,
    dataset_name: str,
    disease: str,
    dataset_dir: Path,
    source_groups: list[str],
) -> dict[int, bool | None]:
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


def load_autoencoder(config: dict[str, Any]) -> tuple[MaskedSourceAutoencoder, dict[str, Any]]:
    checkpoint_path = Path(str(config["model"]["pretrained_autoencoder_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ae_config = checkpoint["config"]
    ae_schema = checkpoint["schema"]
    model_config = ae_config["model"]
    autoencoder = MaskedSourceAutoencoder(
        source_n_num_features=int(ae_schema["source"]["n_num_features"]),
        source_category_sizes=ae_schema["source"]["category_sizes"],
        source_encoder_config=model_config["source_encoder"],
        decoder_hidden_layers=model_config.get("decoder_hidden_layers", []),
        dropout=float(model_config.get("dropout", 0.0)),
        activation=str(model_config.get("activation", "relu")),
        batch_norm=bool(model_config.get("batch_norm", False)),
    )
    missing, unexpected = autoencoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected AE state load result: missing={missing}, unexpected={unexpected}")
    return autoencoder, checkpoint


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


def make_loader(config: dict[str, Any], dataset: Dataset, *, split: str, seed: int) -> DataLoader:
    batch_size = int(config["train"]["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def save_checkpoint(
    path: Path,
    model: MaskedAEClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict[str, Any],
    step: int,
    best_valid_auroc: float,
) -> None:
    torch.save(
        {
            "step": step,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_valid_auroc": best_valid_auroc,
            "condition_dim": model.condition_dim,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/train_masked_ae_downstream_diabetes.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--source-groups", nargs="+")
    parser.add_argument("--target-disease")
    parser.add_argument("--pretrained-autoencoder-path")
    parser.add_argument("--variant-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--freeze-backbone", choices=["true", "false"])
    parser.add_argument("--class-weight", choices=["balanced", "none"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--head-hidden-layers", nargs="*", type=int)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
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
    if args.target_disease is not None:
        config.setdefault("data", {})["target_disease"] = args.target_disease
    if args.max_rows_per_split is not None:
        config.setdefault("data", {})["max_rows_per_split"] = args.max_rows_per_split
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device
    model_config = config.setdefault("model", {})
    if args.pretrained_autoencoder_path is not None:
        model_config["pretrained_autoencoder_path"] = args.pretrained_autoencoder_path
    if args.variant_name is not None:
        model_config["variant_name"] = args.variant_name
    if args.freeze_backbone is not None:
        model_config["freeze_backbone"] = args.freeze_backbone == "true"
    if args.class_weight is not None:
        model_config["class_weight"] = args.class_weight
    if args.dropout is not None:
        model_config["dropout"] = args.dropout
    if args.head_hidden_layers is not None:
        model_config["head_hidden_layers"] = args.head_hidden_layers
    train_config = config.setdefault("train", {})
    if args.steps is not None:
        train_config["steps"] = args.steps
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        train_config["weight_decay"] = args.weight_decay
    if args.num_workers is not None:
        train_config["num_workers"] = args.num_workers
    logging_config = config.setdefault("logging", {})
    if args.log_every is not None:
        logging_config["log_every"] = args.log_every
    if args.validate_every is not None:
        logging_config["validate_every"] = args.validate_every
    if args.checkpoint_every is not None:
        logging_config["checkpoint_every"] = args.checkpoint_every


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
