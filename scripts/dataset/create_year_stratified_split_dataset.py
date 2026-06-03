#!/usr/bin/env python3
"""Create a dataset variant with train/valid/test split stratified by survey year."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    args = parse_args()
    if args.valid_fraction < 0 or args.test_fraction < 0:
        raise ValueError("split fractions must be non-negative.")
    if args.valid_fraction + args.test_fraction >= 1:
        raise ValueError("valid_fraction + test_fraction must be smaller than 1.")

    source_root = args.source_root
    output_root = args.output_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = find_reference_metadata(source_root, args.reference_group)
    rows = read_row_metadata(metadata_path)
    split_by_row, summary_rows = make_year_stratified_split(
        rows=rows,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    group_dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir()]
    if not group_dirs:
        raise ValueError(f"{source_root} does not contain group directories.")

    for group_dir in group_dirs:
        output_group = output_root / group_dir.name
        output_group.mkdir(parents=True, exist_ok=True)
        link_group_files(group_dir, output_group, mode=args.link_mode)
        write_split_csv(output_group / "split.csv", rows, split_by_row)

    write_summary_csv(output_root / "split_summary.csv", summary_rows)
    write_config_json(
        output_root / "split_config.json",
        source_root=source_root,
        output_root=output_root,
        reference_metadata=metadata_path,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        link_mode=args.link_mode,
    )
    maybe_link_disease_labels(args.source_dataset_name, args.output_dataset_name)

    totals = Counter(split_by_row.values())
    print(f"created {output_root}")
    print(f"rows: train={totals['train']} valid={totals['valid']} test={totals['test']}")
    print(f"years: {len(summary_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/preprocessed/gaussian_quantile/harmonized_knhanes_1998_2024"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/preprocessed/gaussian_quantile/harmonized_knhanes_1998_2024_year_stratified_split"),
    )
    parser.add_argument("--source-dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument("--output-dataset-name", default="harmonized_knhanes_1998_2024_year_stratified_split")
    parser.add_argument("--reference-group", default="questionnaire_without_disease")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def find_reference_metadata(source_root: Path, reference_group: str) -> Path:
    preferred = source_root / reference_group / "row_metadata.csv"
    if preferred.exists():
        return preferred
    for path in sorted(source_root.glob("*/row_metadata.csv")):
        return path
    raise FileNotFoundError(f"No row_metadata.csv found under {source_root}")


def read_row_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} is empty.")
    required = {"row_index", "year"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def make_year_stratified_split(
    *,
    rows: list[dict[str, str]],
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[dict[int, str], list[dict[str, int | str]]]:
    by_year: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_year[normalize_year(row["year"])].append(int(row["row_index"]))

    rng = random.Random(seed)
    split_by_row: dict[int, str] = {}
    summary_rows: list[dict[str, int | str]] = []
    for year in sorted(by_year, key=year_sort_key):
        indices = list(by_year[year])
        rng.shuffle(indices)
        n_rows = len(indices)
        n_valid = round_count(n_rows, valid_fraction)
        n_test = round_count(n_rows, test_fraction)
        if n_valid + n_test > n_rows:
            overflow = n_valid + n_test - n_rows
            n_valid = max(0, n_valid - overflow)
        test_indices = indices[:n_test]
        valid_indices = indices[n_test : n_test + n_valid]
        train_indices = indices[n_test + n_valid :]

        for row_index in train_indices:
            split_by_row[row_index] = "train"
        for row_index in valid_indices:
            split_by_row[row_index] = "valid"
        for row_index in test_indices:
            split_by_row[row_index] = "test"

        summary_rows.append(
            {
                "year": year,
                "rows": n_rows,
                "train": len(train_indices),
                "valid": len(valid_indices),
                "test": len(test_indices),
            }
        )
    if len(split_by_row) != len(rows):
        raise ValueError("Split assignment did not cover every row.")
    return split_by_row, summary_rows


def normalize_year(value: str) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def year_sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(float(value)):04d}")
    except ValueError:
        return (1, value)


def round_count(n_rows: int, fraction: float) -> int:
    return int(round(n_rows * fraction))


def link_group_files(source_group: Path, output_group: Path, *, mode: str) -> None:
    for source_path in sorted(source_group.iterdir()):
        if source_path.name == "split.csv":
            continue
        target_path = output_group / source_path.name
        if target_path.exists() or target_path.is_symlink():
            continue
        if mode == "copy":
            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            continue
        relative_source = os.path.relpath(source_path, start=output_group)
        target_path.symlink_to(relative_source)


def write_split_csv(path: Path, rows: list[dict[str, str]], split_by_row: dict[int, str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["row_index", "split", "year"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["row_index"])):
            row_index = int(row["row_index"])
            writer.writerow(
                {
                    "row_index": row_index,
                    "split": split_by_row[row_index],
                    "year": normalize_year(row["year"]),
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["year", "rows", "train", "valid", "test"])
        writer.writeheader()
        writer.writerows(rows)


def write_config_json(path: Path, **payload: object) -> None:
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def maybe_link_disease_labels(source_dataset_name: str, output_dataset_name: str) -> None:
    source_suffix = source_dataset_name.removeprefix("harmonized_")
    output_suffix = output_dataset_name.removeprefix("harmonized_")
    source_path = Path("datasets/harmonized") / f"disease_labels_harmonized_{source_suffix}.csv"
    output_path = Path("datasets/harmonized") / f"disease_labels_harmonized_{output_suffix}.csv"
    if not source_path.exists() or output_path.exists() or output_path.is_symlink():
        return
    relative_source = os.path.relpath(source_path, start=output_path.parent)
    output_path.symlink_to(relative_source)


if __name__ == "__main__":
    main()
