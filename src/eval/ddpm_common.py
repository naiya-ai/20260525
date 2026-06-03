"""Shared DDPM evaluation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.cvae_common import inverse_gaussian_quantile
from train.dataset import load_grouped_cvae_dataset


def load_ddpm_checkpoint_config_schema(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    schema = checkpoint.get("schema")
    if config is None or schema is None:
        raise ValueError(f"DDPM checkpoint must contain config and schema: {checkpoint_path}")
    return checkpoint, config, schema


def load_target_metadata(config: dict[str, Any]) -> dict[str, Any]:
    target_dir = (
        Path(config["data"]["dataset_root"])
        / str(config["data"]["dataset_name"])
        / str(config["data"]["target_group"])
    )
    return json.loads((target_dir / "metadata.json").read_text())


def split_row_ids_for_loaded_dataset(config: dict[str, Any], *, split: str) -> np.ndarray:
    data_config = config["data"]
    dataset_dir = Path(data_config["dataset_root"]) / str(data_config["dataset_name"])
    target_group = str(data_config["target_group"])
    split_path = dataset_dir / target_group / "split.csv"
    row_ids: list[int] = []
    with split_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                row_ids.append(int(row["row_index"]))
    indices = np.asarray(row_ids, dtype=np.int64)
    if bool(data_config.get("require_complete_target", False)):
        dataset = load_grouped_cvae_dataset(config, split=split)
        num_mask = dataset.target_num_mask.numpy()
        cat_mask = dataset.target_cat_mask.numpy()
        keep_parts = []
        if num_mask.shape[1] > 0:
            keep_parts.append(num_mask.astype(bool).all(axis=1))
        if cat_mask.shape[1] > 0:
            keep_parts.append(cat_mask.astype(bool).all(axis=1))
        if keep_parts:
            keep = keep_parts[0]
            for part in keep_parts[1:]:
                keep &= part
            indices = indices[keep]
    max_rows = data_config.get("max_rows_per_split")
    if max_rows is not None:
        indices = indices[: int(max_rows)]
    return indices


def restore_target_values(
    *,
    target_metadata: dict[str, Any],
    num: np.ndarray,
    cat: np.ndarray,
) -> dict[str, np.ndarray]:
    restored: dict[str, np.ndarray] = {}
    for idx, feature in enumerate(target_metadata["features"]["num"]):
        state = target_metadata["gaussian_quantile"]["features"][idx]
        restored[feature] = inverse_gaussian_quantile(num[:, idx], state)
    for idx, feature in enumerate(target_metadata["features"]["cat"]):
        restored[feature] = cat[:, idx]
    return restored
