"""Dataset loading for group-wise preprocessed CVAE data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


SplitName = Literal["train", "valid", "test"]


class GroupedConditionalVAEDataset(Dataset):
    """Torch dataset assembled from preprocessed variable-group folders."""

    def __init__(
        self,
        *,
        source_num: Tensor,
        source_cat: Tensor,
        source_num_mask: Tensor,
        source_cat_mask: Tensor,
        target_num: Tensor,
        target_cat: Tensor,
        target_num_mask: Tensor,
        target_cat_mask: Tensor,
        sample_loss_weights: Tensor,
    ) -> None:
        self.source_num = source_num
        self.source_cat = source_cat
        self.source_num_mask = source_num_mask
        self.source_cat_mask = source_cat_mask
        self.target_num = target_num
        self.target_cat = target_cat
        self.target_num_mask = target_num_mask
        self.target_cat_mask = target_cat_mask
        self.sample_loss_weights = sample_loss_weights
        self._validate_lengths()

    def __len__(self) -> int:
        return int(self.target_num.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "source_num": self.source_num[index],
            "source_cat": self.source_cat[index],
            "source_num_mask": self.source_num_mask[index],
            "source_cat_mask": self.source_cat_mask[index],
            "target_num": self.target_num[index],
            "target_cat": self.target_cat[index],
            "target_num_mask": self.target_num_mask[index],
            "target_cat_mask": self.target_cat_mask[index],
            "sample_loss_weights": self.sample_loss_weights[index],
        }

    def _validate_lengths(self) -> None:
        lengths = {
            "source_num": len(self.source_num),
            "source_cat": len(self.source_cat),
            "source_num_mask": len(self.source_num_mask),
            "source_cat_mask": len(self.source_cat_mask),
            "target_num": len(self.target_num),
            "target_cat": len(self.target_cat),
            "target_num_mask": len(self.target_num_mask),
            "target_cat_mask": len(self.target_cat_mask),
            "sample_loss_weights": len(self.sample_loss_weights),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Dataset arrays have inconsistent row counts: {lengths}")


def create_grouped_cvae_dataloader(
    config: dict[str, Any],
    *,
    split: SplitName,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    distributed: bool = False,
) -> DataLoader:
    dataset = load_grouped_cvae_dataset(config, split=split)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = (
        DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def load_grouped_cvae_dataset(
    config: dict[str, Any],
    *,
    split: SplitName,
) -> GroupedConditionalVAEDataset:
    data_config = config["data"]
    dataset_dir = _dataset_dir(data_config)
    source_groups = _string_list(data_config["source_groups"])
    target_group = str(data_config["target_group"])
    require_complete_target = bool(data_config.get("require_complete_target", False))
    max_rows = data_config.get("max_rows_per_split")

    indices = _split_indices(dataset_dir / target_group / "split.csv", split=split)
    source = _load_group_arrays(dataset_dir, source_groups, indices=indices)
    target = _load_group_arrays(dataset_dir, [target_group], indices=indices)
    sample_weights = _load_sample_weights(
        dataset_dir / target_group,
        indices=indices,
        enabled=bool(data_config.get("sample_weight", {}).get("enabled", False)),
        default=float(data_config.get("sample_weight", {}).get("default", 1.0)),
    )

    keep = None
    if require_complete_target:
        keep = _complete_target_rows(target["num_mask"], target["cat_mask"])
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            raise ValueError("data.max_rows_per_split must be positive when set.")
        row_limit = np.zeros((len(indices),), dtype=bool)
        row_limit[: min(max_rows, len(indices))] = True
        keep = row_limit if keep is None else keep & row_limit

    if keep is not None:
        source = {key: value[keep] for key, value in source.items()}
        target = {key: value[keep] for key, value in target.items()}
        sample_weights = sample_weights[keep]

    return GroupedConditionalVAEDataset(
        source_num=_as_tensor(source["num"], torch.float32),
        source_cat=_as_tensor(source["cat"], torch.long),
        source_num_mask=_as_tensor(source["num_mask"], torch.float32),
        source_cat_mask=_as_tensor(source["cat_mask"], torch.float32),
        target_num=_as_tensor(target["num"], torch.float32),
        target_cat=_as_tensor(target["cat"], torch.long),
        target_num_mask=_as_tensor(target["num_mask"], torch.float32),
        target_cat_mask=_as_tensor(target["cat_mask"], torch.float32),
        sample_loss_weights=_as_tensor(sample_weights, torch.float32),
    )


def load_grouped_cvae_schema(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    dataset_dir = _dataset_dir(data_config)
    source_groups = _string_list(data_config["source_groups"])
    target_group = str(data_config["target_group"])
    source_metadata = [_read_metadata(dataset_dir / group) for group in source_groups]
    target_metadata = [_read_metadata(dataset_dir / target_group)]
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_name": data_config["dataset_name"],
        "source_groups": source_groups,
        "target_group": target_group,
        "source": _schema_from_metadata(source_metadata),
        "target": _schema_from_metadata(target_metadata),
    }


def _dataset_dir(data_config: dict[str, Any]) -> Path:
    return Path(str(data_config["dataset_root"])) / str(data_config["dataset_name"])


def _load_group_arrays(
    dataset_dir: Path,
    group_names: list[str],
    *,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    if not group_names:
        raise ValueError("At least one variable group is required.")
    num_parts = []
    cat_parts = []
    num_mask_parts = []
    cat_mask_parts = []
    row_count: int | None = None
    for group_name in group_names:
        group_dir = dataset_dir / group_name
        num = _load_rows(group_dir / "num.npy", indices)
        cat = _load_rows(group_dir / "cat.npy", indices)
        num_mask = _load_rows(group_dir / "num_valid_mask.npy", indices)
        cat_mask = (cat != 0).astype(np.float32, copy=False)
        if row_count is None:
            row_count = int(num.shape[0])
        elif row_count != int(num.shape[0]):
            raise ValueError(f"Group row counts differ under {dataset_dir}.")
        num_parts.append(num.astype(np.float32, copy=False))
        cat_parts.append(cat.astype(np.int64, copy=False))
        num_mask_parts.append(num_mask.astype(np.float32, copy=False))
        cat_mask_parts.append(cat_mask)
    return {
        "num": _concat_columns(num_parts, np.float32),
        "cat": _concat_columns(cat_parts, np.int64),
        "num_mask": _concat_columns(num_mask_parts, np.float32),
        "cat_mask": _concat_columns(cat_mask_parts, np.float32),
    }


def _load_sample_weights(
    target_dir: Path,
    *,
    indices: np.ndarray,
    enabled: bool,
    default: float,
) -> np.ndarray:
    if not enabled:
        return np.full((len(indices),), default, dtype=np.float32)
    path = target_dir / "sample_weight.npy"
    if not path.exists():
        return np.full((len(indices),), default, dtype=np.float32)
    weights = np.asarray(np.load(path, mmap_mode="r")[indices], dtype=np.float32)
    return weights.reshape(-1)


def _load_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path, mmap_mode="r")[indices])


def _split_indices(path: Path, *, split: SplitName) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    indices: list[int] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["split"] == split:
                indices.append(int(row["row_index"]))
    if not indices:
        raise ValueError(f"No rows found for split={split!r} in {path}.")
    return np.asarray(indices, dtype=np.int64)


def _read_metadata(group_dir: Path) -> dict[str, Any]:
    path = group_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _schema_from_metadata(metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    categorical_features: list[str] = []
    numerical_features: list[str] = []
    category_sizes: list[int] = []
    groups: list[str] = []
    for metadata in metadata_list:
        groups.append(str(metadata["group_name"]))
        categorical_features.extend(metadata["features"]["cat"])
        numerical_features.extend(metadata["features"]["num"])
        category_sizes.extend(int(size) for size in metadata["category_sizes"])
    return {
        "groups": groups,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "category_sizes": category_sizes,
        "n_cat_features": len(categorical_features),
        "n_num_features": len(numerical_features),
    }


def _concat_columns(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    row_count = parts[0].shape[0]
    non_empty = [part for part in parts if part.shape[1] > 0]
    if not non_empty:
        return np.empty((row_count, 0), dtype=dtype)
    return np.concatenate(non_empty, axis=1).astype(dtype, copy=False)


def _complete_target_rows(num_mask: np.ndarray, cat_mask: np.ndarray) -> np.ndarray:
    masks = []
    if num_mask.shape[1] > 0:
        masks.append(num_mask.astype(bool).all(axis=1))
    if cat_mask.shape[1] > 0:
        masks.append(cat_mask.astype(bool).all(axis=1))
    if not masks:
        raise ValueError("Target must contain at least one feature.")
    keep = masks[0]
    for mask in masks[1:]:
        keep = keep & mask
    return keep


def _as_tensor(array: np.ndarray, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(array, dtype=dtype).contiguous()


def _string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]
