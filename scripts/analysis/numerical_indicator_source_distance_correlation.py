"""Find source variables with high distance correlation to numerical indicators.

The analysis uses the already preprocessed gaussian-quantile ``num.npy`` arrays.
Distance correlation is computed on sampled complete pairs because the exact
pairwise-distance estimator is quadratic in the number of rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INDICATOR_GROUPS = (
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "kidney_disease",
    "anemia",
)
DEFAULT_SOURCE_GROUPS = ("questionnaire_without_disease", "dietary")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_root) / args.dataset_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_joined_num_groups(dataset_dir, args.source_groups)
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for indicator_group in args.indicator_groups:
        indicator = load_num_group(dataset_dir / indicator_group)
        split_indices = load_split_indices(
            dataset_dir / indicator_group / "split.csv",
            split=args.split,
        )
        for indicator_index, indicator_name in enumerate(indicator["features"]):
            scores = []
            y_all = indicator["num"][:, indicator_index]
            y_mask_all = indicator["mask"][:, indicator_index].astype(bool)
            for source_index, source_name in enumerate(source["features"]):
                x_all = source["num"][:, source_index]
                x_mask_all = source["mask"][:, source_index].astype(bool)
                valid = split_indices[y_mask_all[split_indices] & x_mask_all[split_indices]]
                if len(valid) < args.min_pairs:
                    score = np.nan
                    n_pairs = int(len(valid))
                    sampled = n_pairs
                else:
                    if len(valid) > args.max_rows:
                        valid = np.sort(rng.choice(valid, size=args.max_rows, replace=False))
                    x = np.asarray(x_all[valid], dtype=np.float64)
                    y = np.asarray(y_all[valid], dtype=np.float64)
                    score = distance_correlation_1d(x, y)
                    n_pairs = int((y_mask_all[split_indices] & x_mask_all[split_indices]).sum())
                    sampled = int(len(valid))
                row = {
                    "indicator_group": indicator_group,
                    "indicator": indicator_name,
                    "source": source_name,
                    "distance_correlation": score,
                    "n_complete_pairs": n_pairs,
                    "n_sampled": sampled,
                    "split": args.split,
                    "seed": args.seed,
                }
                rows.append(row)
                scores.append(row)

            finite_scores = [
                row
                for row in scores
                if np.isfinite(float(row["distance_correlation"]))
            ]
            if finite_scores:
                best = max(
                    finite_scores,
                    key=lambda row: float(row["distance_correlation"]),
                )
                rank = sorted(
                    finite_scores,
                    key=lambda row: float(row["distance_correlation"]),
                    reverse=True,
                )[: args.top_k]
                for position, row in enumerate(rank, start=1):
                    best_rows.append({"rank": position, **row})
                print(
                    f"{indicator_group}/{indicator_name}: "
                    f"best={best['source']} dcor={float(best['distance_correlation']):.4f} "
                    f"sampled={best['n_sampled']}",
                    flush=True,
                )

    write_rows(output_dir / "indicator_source_distance_correlation.csv", rows)
    write_rows(output_dir / "indicator_source_distance_correlation_top.csv", best_rows)
    print("wrote", output_dir / "indicator_source_distance_correlation.csv")
    print("wrote", output_dir / "indicator_source_distance_correlation_top.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datasets/preprocessed/gaussian_quantile")
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--source-groups", nargs="+", default=list(DEFAULT_SOURCE_GROUPS))
    parser.add_argument("--indicator-groups", nargs="+", default=list(DEFAULT_INDICATOR_GROUPS))
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="train")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="outputs/distance_correlation/indicator_source_gaussian_quantile",
    )
    return parser.parse_args()


def load_joined_num_groups(dataset_dir: Path, group_names: list[str]) -> dict[str, Any]:
    groups = [load_num_group(dataset_dir / group_name) for group_name in group_names]
    row_counts = {group["num"].shape[0] for group in groups}
    if len(row_counts) != 1:
        raise ValueError(f"Source group row counts differ: {sorted(row_counts)}")
    return {
        "features": [
            f"{group_name}:{feature}"
            for group_name, group in zip(group_names, groups)
            for feature in group["features"]
        ],
        "num": np.concatenate([group["num"] for group in groups], axis=1),
        "mask": np.concatenate([group["mask"] for group in groups], axis=1),
    }


def load_num_group(group_dir: Path) -> dict[str, Any]:
    metadata = json.loads((group_dir / "metadata.json").read_text())
    features = list(metadata["features"]["num"])
    return {
        "features": features,
        "num": np.load(group_dir / "num.npy", mmap_mode="r"),
        "mask": np.load(group_dir / "num_valid_mask.npy", mmap_mode="r"),
    }


def load_split_indices(path: Path, *, split: str) -> np.ndarray:
    indices: list[int] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if split == "all" or row["split"] == split:
                indices.append(int(row["row_index"]))
    if not indices:
        raise ValueError(f"No rows found for split={split!r} in {path}")
    return np.asarray(indices, dtype=np.int64)


def distance_correlation_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays with equal length.")
    if len(x) < 2:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0

    ax = centered_distance_matrix_1d(x)
    ay = centered_distance_matrix_1d(y)
    dcov2 = np.mean(ax * ay)
    dvar_x = np.mean(ax * ax)
    dvar_y = np.mean(ay * ay)
    if dcov2 <= 0 or dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y)))


def centered_distance_matrix_1d(values: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - values[None, :])
    return distances - distances.mean(axis=0, keepdims=True) - distances.mean(axis=1, keepdims=True) + distances.mean()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
