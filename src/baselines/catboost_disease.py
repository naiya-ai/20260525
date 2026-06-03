"""Disease-label CatBoost baseline for group-wise preprocessed data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eval.cvae_common import binary_auroc


DISEASES = (
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_dir = args.dataset_root / args.dataset_name
    split_by_name = read_split(dataset_dir / args.target_group / "split.csv")
    source = load_source_frame(dataset_dir, args.source_groups)
    row_metadata = read_row_metadata(dataset_dir, args.source_groups, args.target_group)
    if row_metadata is not None and len(row_metadata) != len(source):
        raise ValueError(
            f"row_metadata has {len(row_metadata)} rows but source has {len(source)} rows."
        )
    labels = read_disease_labels(args.dataset_name, args.target_group, row_metadata=row_metadata)

    train_indices, train_y = observed_indices_and_labels(split_by_name["train"], labels)
    valid_indices, valid_y = observed_indices_and_labels(split_by_name["valid"], labels)
    test_indices = split_by_name["test"]
    test_labels = [labels.get(idx) for idx in test_indices]

    if len(set(train_y.tolist())) < 2:
        raise ValueError(f"{args.target_group} train labels do not contain both classes.")

    class_weight = positive_class_weight(train_y)
    sample_weight = make_train_weights(train_y, class_weight, args.class_weight)
    model = build_model(args, class_weight)
    cat_features = [
        column
        for column in source.columns
        if column.startswith("cat:")
    ]

    train_pool = Pool(
        source.iloc[train_indices],
        label=train_y,
        cat_features=cat_features,
        weight=sample_weight,
    )
    eval_set = None
    if len(valid_y) and len(set(valid_y.tolist())) == 2:
        eval_set = Pool(
            source.iloc[valid_indices],
            label=valid_y,
            cat_features=cat_features,
        )

    use_early_stopping = eval_set is not None and args.early_stopping_rounds > 0
    model.fit(
        train_pool,
        eval_set=eval_set,
        verbose=args.verbose_eval,
        use_best_model=use_early_stopping,
        early_stopping_rounds=args.early_stopping_rounds if use_early_stopping else None,
    )

    test_pool = Pool(source.iloc[test_indices], cat_features=cat_features)
    probabilities = model.predict_proba(test_pool)[:, 1]
    predictions = [bool(value >= args.threshold) for value in probabilities]
    metrics = binary_metrics(predictions, test_labels)
    auroc = binary_auroc(probabilities.astype(float).tolist(), test_labels)
    result = {
        "variant": args.variant,
        "target_group": args.target_group,
        "dataset_name": args.dataset_name,
        "source_groups": args.source_groups,
        "threshold": args.threshold,
        "class_weight": args.class_weight,
        "positive_class_weight": class_weight,
        "train_label_counts": label_counts(train_y),
        "valid_label_counts": label_counts(valid_y),
        "test_observed_label_counts": label_counts(
            np.asarray([int(label) for label in test_labels if label is not None], dtype=np.int64)
        ),
        "auroc": auroc,
        "catboost_params": model.get_params(),
        **metrics,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.variant}_{args.target_group}.cbm"
    metrics_path = output_dir / f"{args.variant}_{args.target_group}_metrics.json"
    predictions_path = output_dir / f"{args.variant}_{args.target_group}_predictions.csv"
    model.save_model(model_path)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_predictions:
        write_predictions(predictions_path, test_indices, probabilities, predictions, test_labels)
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/preprocessed/gaussian_quantile"))
    parser.add_argument("--dataset-name", default="harmonized_knhanes_1998_2024")
    parser.add_argument(
        "--source-groups",
        nargs="+",
        default=["questionnaire_without_disease", "dietary"],
    )
    parser.add_argument("--target-group", required=True, choices=DISEASES)
    parser.add_argument("--variant", default="catboost")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Set to 0 to disable early stopping and best-model shrink.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--class-weight", choices=["none", "inverse_prevalence"], default="inverse_prevalence")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_source_frame(dataset_dir: Path, source_groups: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for group in source_groups:
        group_dir = dataset_dir / group
        metadata = json.loads((group_dir / "metadata.json").read_text())
        num = np.load(group_dir / "num.npy", mmap_mode="r")
        num_mask = np.load(group_dir / "num_valid_mask.npy", mmap_mode="r")
        cat = np.load(group_dir / "cat.npy", mmap_mode="r")
        row_count = int(num.shape[0])

        columns: dict[str, Any] = {}
        for idx, feature in enumerate(metadata["features"]["num"]):
            values = np.asarray(num[:, idx], dtype=np.float32)
            mask = np.asarray(num_mask[:, idx] > 0)
            clean = values.astype(np.float32, copy=True)
            clean[~mask] = np.nan
            columns[f"num:{group}:{feature}"] = clean
            columns[f"num_mask:{group}:{feature}"] = mask.astype(np.float32)
        for idx, feature in enumerate(metadata["features"]["cat"]):
            values = np.asarray(cat[:, idx], dtype=np.int64)
            text_values = values.astype(str)
            text_values[values == 0] = "__MISSING__"
            columns[f"cat:{group}:{feature}"] = text_values
        frame = pd.DataFrame(columns)
        if len(frame) != row_count:
            raise ValueError(f"Unexpected row count while loading {group_dir}.")
        parts.append(frame)
    if not parts:
        raise ValueError("At least one source group is required.")
    return pd.concat(parts, axis=1)


def read_split(path: Path) -> dict[str, list[int]]:
    result = {"train": [], "valid": [], "test": []}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            split = row["split"]
            if split in result:
                result[split].append(int(row["row_index"]))
    return result


def read_row_metadata(
    dataset_dir: Path,
    source_groups: list[str],
    target_group: str,
) -> list[dict[str, str]] | None:
    candidate_dirs = [*(dataset_dir / group for group in source_groups), dataset_dir / target_group]
    for directory in candidate_dirs:
        path = directory / "row_metadata.csv"
        if path.exists():
            with path.open(newline="", errors="replace") as file:
                return list(csv.DictReader(file))
    return None


def read_disease_labels(
    dataset_name: str,
    disease: str,
    *,
    row_metadata: list[dict[str, str]] | None = None,
) -> dict[int, bool | None]:
    suffix = dataset_name.removeprefix("harmonized_")
    path = Path("datasets/harmonized") / f"disease_labels_harmonized_{suffix}.csv"
    column = f"disease_{disease}"
    labels: dict[int, bool | None] = {}
    with path.open(newline="", errors="replace") as file:
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain {column}.")
        if row_metadata is not None:
            if not {"ID", "year"}.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{path} must contain ID and year for ID-based matching.")
            labels_by_key = {
                (row["ID"], row["year"]): parse_label(row[column])
                for row in reader
            }
            missing = []
            for meta in row_metadata:
                key = (meta.get("ID", ""), meta.get("year", ""))
                if key not in labels_by_key:
                    missing.append(key)
                    continue
                labels[int(meta["row_index"])] = labels_by_key[key]
            if missing:
                raise ValueError(
                    f"{path} is missing {len(missing)} row_metadata IDs; "
                    f"examples={missing[:5]}"
                )
            return labels
        for row in reader:
            labels[int(row["row_index"])] = parse_label(row[column])
    return labels


def observed_indices_and_labels(indices: list[int], labels: dict[int, bool | None]) -> tuple[list[int], np.ndarray]:
    kept_indices = []
    y = []
    for idx in indices:
        label = labels.get(idx)
        if label is None:
            continue
        kept_indices.append(idx)
        y.append(1 if label else 0)
    return kept_indices, np.asarray(y, dtype=np.int64)


def parse_label(value: str) -> bool | None:
    value = value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def positive_class_weight(y: np.ndarray) -> float:
    n_true = int(np.sum(y == 1))
    n_false = int(np.sum(y == 0))
    if n_true == 0:
        return 1.0
    return n_false / n_true


def make_train_weights(y: np.ndarray, class_weight: float, strategy: str) -> np.ndarray | None:
    if strategy == "none":
        return None
    weights = np.ones((len(y),), dtype=np.float32)
    weights[y == 1] = float(class_weight)
    return weights


def build_model(args: argparse.Namespace, class_weight: float) -> CatBoostClassifier:
    params: dict[str, Any] = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": args.seed,
        "thread_count": args.thread_count,
        "task_type": args.task_type,
        "allow_writing_files": False,
    }
    if args.devices:
        params["devices"] = args.devices
    return CatBoostClassifier(**params)


def binary_metrics(predictions: list[bool], labels: list[bool | None]) -> dict[str, Any]:
    tp = tn = fp = fn = missing_label = 0
    for pred, label in zip(predictions, labels):
        if label is None:
            missing_label += 1
        elif pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "missing_label": missing_label,
        "missing_prediction": 0,
        "n_evaluable": tp + tn + fp + fn,
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
    }


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def label_counts(y: np.ndarray) -> dict[str, int]:
    if y.size == 0:
        return {"true": 0, "false": 0}
    return {
        "true": int(np.sum(y == 1)),
        "false": int(np.sum(y == 0)),
    }


def write_predictions(
    path: Path,
    test_indices: list[int],
    probabilities: np.ndarray,
    predictions: list[bool],
    labels: list[bool | None],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["row_index", "probability", "prediction", "label"],
        )
        writer.writeheader()
        for idx, probability, prediction, label in zip(test_indices, probabilities, predictions, labels):
            writer.writerow(
                {
                    "row_index": idx,
                    "probability": float(probability),
                    "prediction": prediction,
                    "label": label,
                }
            )


def set_seed(seed: int) -> None:
    np.random.seed(seed)


if __name__ == "__main__":
    main()
